"""Persona C — evidence-based exception resolvers (POLICIES.md §3): SHORT_BANK_CHARGES and DUPLICATE_PAYMENT.

Pipeline resolvers f(ctx, pending) -> rows (see deterministic.mk_rows), exposed through two entry modules:

  EARLY   module `c_bank_charges_early` (before ar_text): `bank_charges`
          cross-border receipt (the statement says so + payer country != entity country) short of exactly one open invoice of
          the single best-evidenced payer by 0 < short <= min(60 EUR eq., 2 %), not its discount, and no document combination of
          any evidenced payer explains the cash exactly -> SHORT_BANK_CHARGES, row = full open amount.
  LATE    module `c_duplicates` (after every matcher): `duplicates`
          a line nobody explained, equal to the total of an item that was open at the window start and is already settled
          (shared ledger) or whose earlier same-amount, same-payer line an earlier resolver accepted -> DUPLICATE_PAYMENT.

  BANKREC_ORDER=remittance,pdf_clauses,c_bank_charges_early,ar_text,ar_exact,ar_multi,c_discount,c_fx,c_duplicates

Every rule is a policy formula with an abstention; a line is only claimed when the formula has exactly one solution.
(An exception post-pass over every module's rows was tried too: with BANKREC_LEDGER=1 it lowers the score, because the ledger
already settles repeat payments; it is not part of this module.)
"""
from __future__ import annotations

import math
import os

import polars as pl

from .data import DATA, Ctx, norm_expr
from .deterministic import _acc_entity, empty_rows, mk_rows

# =====================================================================================================================
# CONFIG — every constant with its source. Nothing tuned on labels.
# =====================================================================================================================
AMT_TOL = 0.011                   # cent rounding of amounts already rounded to the cent
# POLICIES §3 SHORT_BANK_CHARGES: cross-border transfer, intermediary bank deducted its fee: shortfall ≤ 60 EUR eq. and ≤ 2 %.
CHARGES_MAX_EUR = 60.0
CHARGES_MAX_PCT = 0.02
# POLICIES §3 SHORT_DISCOUNT: booking ≤ invoice date + discount days + 3 days grace, cash = total − pct × total.
#   A shortfall equal to the early-payment discount of the invoice is never read as bank charges.
DISCOUNT_GRACE_DAYS = 3
# "Cross-border transfer" as the bank statement itself states it (standard statement semantics, not customer vocabulary):
#   ISO 20022 bank transaction code PMNT/RCDT/XBCT (camt.053: received cross-border credit transfer),
#   AEB Norma 43 common concept 13 ("operaciones con el extranjero"),
#   SWIFT charge fields: 71A charge option SHA/BEN or 71F /CHGS/ (only foreign transfers carry them).
CROSS_BORDER_CODES = {("camt053", "PMNT/RCDT/XBCT"), ("n43", "13")}
CHARGE_FIELD_RE = r"/CHGS/|\b(?:GASTOS|CHARGES?|CHGS?|FRAIS|SPESE|SPESEN|ENTGELT)\s*:?\s*/?(?:SHA|BEN)\b|\b(?:SHA|BEN)\s+CHARGES\b"
# Party evidence (same rule as ar_exact, shared team convention): ≥ MIN_KEY_LEN chars of the first NAME_KEY_LEN normalised
# chars of a party name in the bank text, the master IBAN as counterparty account, or our document number (≥ 6 chars / its
# last 6 digits) in the text.
NAME_KEY_LEN = 10
MIN_KEY_LEN = 5
DOC_MIN_LEN = 6
# "Explained without a difference": cash = exact sum of 1..SUBSET_MAX_K open documents of the payer (complexity bound only;
# triples are only searched in pools of ≤ SUBSET_MAX_POOL documents).
SUBSET_MAX_K = 3
SUBSET_MAX_POOL = 400
# Log-bucket width for the amount-keyed joins (performance only: avoids an entity×currency cartesian join; any factor works).
BUCKET = 1.05


# =====================================================================================================================
# helpers
# =====================================================================================================================
def _cache(ctx: Ctx) -> dict:
    return ctx.__dict__.setdefault("_cache", {})


def _entity_country(ctx: Ctx) -> pl.DataFrame:
    c = _cache(ctx)
    if "c_entity_country" not in c:
        p = os.path.join(DATA, ctx.phase, "masters_final", "entities.csv") if ctx.phase == "final" else None
        if not (p and os.path.exists(p)):
            p = os.path.join(DATA, "masters", "entities.csv")
        c["c_entity_country"] = pl.read_csv(p, infer_schema_length=0).select("entity", pl.col("country").alias("entity_country"))
    return c["c_entity_country"]


def _party_keys(ctx: Ctx, side: str) -> pl.DataFrame:
    """party_id -> name keys k0.., master IBAN, country."""
    c = _cache(ctx)
    key = f"c_keys_{side}"
    if key not in c:
        if side == "AR":
            df, names = ctx.customers.rename({"customer_id": "party_id"}), ("name", "trade_name")
        else:
            df, names = ctx.suppliers.rename({"supplier_id": "party_id"}), ("name",)
        cols = [pl.col("party_id")] + [norm_expr(n).str.slice(0, NAME_KEY_LEN).alias(f"k{i}") for i, n in enumerate(names) if n in df.columns]
        cols += [norm_expr(n).alias(f"f{i}") for i, n in enumerate(names) if n in df.columns]
        cols.append(pl.col("iban") if "iban" in df.columns else pl.lit(None, pl.Utf8).alias("iban"))
        cols.append(pl.col("country").alias("party_country") if "country" in df.columns else pl.lit(None, pl.Utf8).alias("party_country"))
        c[key] = df.select(cols).unique("party_id")
    return c[key]


def _with_evidence(c: pl.DataFrame, ctx: Ctx, side: str) -> pl.DataFrame:
    """Candidate pairs (need tnorm, cp_iban, docnorm, party_id, party_name) + flags doc / name / iban_ok and ev score."""
    keys = _party_keys(ctx, side)
    c = c.join(keys, on="party_id", how="left")
    kcols = [k for k in keys.columns if k.startswith("k")]
    fcols = [k for k in keys.columns if k.startswith("f")]

    def has(e: pl.Expr) -> pl.Expr:
        return ((e.str.len_chars() >= MIN_KEY_LEN) & pl.col("tnorm").str.contains(e, literal=True)).fill_null(False)

    # doc cited: the full normalised document number in the text, or a standalone number of the text (a digit run of
    # ≥ DOC_MIN_LEN digits) that is the tail of the document's digits ("FRAS 208961" for FV25-0208961). A 6-digit window
    # inside a longer reference (payment ids, bank refs) is not a citation.
    c = c.with_row_index("_ev_r")
    runs = c.select("_ev_r", pl.col("docnorm").str.replace_all(r"\D", "").alias("_dd"), pl.col("druns")).explode("druns").filter(
        pl.col("druns").is_not_null() & (pl.col("_dd").str.len_chars() >= DOC_MIN_LEN) & pl.col("_dd").str.ends_with(pl.col("druns")))
    c = c.with_columns(pl.col("_ev_r").is_in(runs["_ev_r"].unique().implode()).alias("_drun"))
    name = pl.any_horizontal([has(pl.col(k)) for k in kcols] + [has(norm_expr("party_name").str.slice(0, NAME_KEY_LEN))])
    c = c.with_columns(
        (((pl.col("docnorm").str.len_chars() >= DOC_MIN_LEN) & pl.col("tnorm").str.contains(pl.col("docnorm"), literal=True))
         | pl.col("_drun")).fill_null(False).alias("doc"),
        name.fill_null(False).alias("name"),
        pl.any_horizontal([has(pl.col(k)) for k in fcols]).fill_null(False).alias("fullname"),
        (pl.col("cp_iban").is_not_null() & (pl.col("cp_iban") == pl.col("iban"))).fill_null(False).alias("iban_ok"))
    return c.with_columns((pl.col("doc").cast(pl.Int32) * 4 + pl.col("iban_ok").cast(pl.Int32) * 2 + pl.col("name").cast(pl.Int32)).alias("ev")
                          ).drop(kcols + fcols + ["_ev_r", "_drun"])


def _lines(ctx: Ctx, pending: pl.DataFrame, credit: bool) -> pl.DataFrame:
    """Same-currency lines of one direction (lines with an original foreign amount are FX cases, not ours here)."""
    side = pl.col("is_credit") if credit else ~pl.col("is_credit")
    return (pending.filter(side & pl.col("orig_amt").is_null())
            .join(_acc_entity(ctx), on="account_id")
            .select("bank_txn_id", "booking_date", "currency", pl.col("acc_entity").alias("entity"),
                    pl.col("abs_amt").round(2).alias("cash"), norm_expr("alltext").alias("tnorm"),
                    pl.col("alltext").str.extract_all(r"\d{%d,}" % DOC_MIN_LEN).alias("druns"),
                    pl.col("counterparty_account").alias("cp_iban")))


def _best_party(c: pl.DataFrame) -> pl.DataFrame:
    """Keep, per line, the candidate pairs of the single best-evidenced party (party evidence = IBAN 4, full name 2, name key 1,
    plus 8 when one of its candidate documents is cited); lines where two parties tie at the top are dropped (abstain)."""
    c = c.with_columns((pl.col("iban_ok").cast(pl.Int32) * 4 + pl.col("fullname").cast(pl.Int32) * 2 + pl.col("name").cast(pl.Int32)
                        ).alias("_pev"))
    ps = c.group_by("bank_txn_id", "party_id").agg((pl.col("_pev").max() + pl.col("doc").any().cast(pl.Int32) * 8).alias("_ps"))
    top = ps.group_by("bank_txn_id").agg(pl.col("_ps").max().alias("_top"))
    ps = ps.join(top, on="bank_txn_id").filter(pl.col("_ps") == pl.col("_top"))
    uniq = ps.group_by("bank_txn_id").agg(pl.len().alias("_n")).filter(pl.col("_n") == 1)
    keep = ps.join(uniq, on="bank_txn_id", how="semi").select("bank_txn_id", "party_id")
    return c.join(keep, on=["bank_txn_id", "party_id"], how="semi").drop("_pev")


def _line_parties(ctx: Ctx, lines: pl.DataFrame, side: str, items: pl.DataFrame | None = None) -> pl.DataFrame:
    """bank_txn_id -> party_id candidates by evidence, without a lines×parties cartesian join: name keys and our document
    numbers are found in the text with one multi-pattern (Aho-Corasick) scan; IBANs by equality."""
    keys = _party_keys(ctx, side)
    kcols = [k for k in keys.columns if k.startswith("k")]
    kk = pl.concat([keys.select("party_id", pl.col(k).alias("key")) for k in kcols]).filter(
        pl.col("key").str.len_chars() >= MIN_KEY_LEN).unique()
    parts = [lines.join(keys.select("party_id", "iban").filter(pl.col("iban").is_not_null()), left_on="cp_iban", right_on="iban")
             .select("bank_txn_id", "party_id")]
    pats = [(kk, "key")]
    if items is not None:
        pats.append((items.select("party_id", pl.col("docnorm").alias("key")).filter(pl.col("key").str.len_chars() >= DOC_MIN_LEN).unique(), "key"))
    for df, col in pats:
        vocab = df[col].unique().to_list()
        if not vocab:
            continue
        m = lines.select("bank_txn_id", pl.col("tnorm").str.extract_many(vocab, overlapping=True).alias(col)).explode(col).drop_nulls()
        parts.append(m.join(df, on=col).select("bank_txn_id", "party_id"))
    return pl.concat(parts).unique()


def _bucket(col: str) -> pl.Expr:
    return (pl.col(col).clip(lower_bound=0.01).log() / math.log(BUCKET)).floor().cast(pl.Int64)


def _keyed(lines: pl.DataFrame, items: pl.DataFrame, line_amt: str, item_amt: str, lo: float, hi: float) -> pl.DataFrame:
    """Pairs (line, item) of the same entity+currency with item_amt / line_amt in [lo, hi] (amount-bucketed join)."""
    it = items.with_columns(_bucket(item_amt).alias("_k"))
    ln = lines.with_columns(_bucket(line_amt).alias("_k0"))
    d0, d1 = math.floor(math.log(lo) / math.log(BUCKET)) - 1, math.ceil(math.log(hi) / math.log(BUCKET)) + 1
    parts = [ln.with_columns((pl.col("_k0") + d).alias("_k")) for d in range(d0, d1 + 1)]
    c = pl.concat(parts).join(it, on=["entity", "currency", "_k"]).drop("_k", "_k0")
    return c.filter((pl.col(item_amt) >= pl.col(line_amt) * lo - AMT_TOL) & (pl.col(item_amt) <= pl.col(line_amt) * hi + AMT_TOL))


def _per_eur(ctx: Ctx, c: pl.DataFrame) -> pl.DataFrame:
    """Add per_eur = units of the line currency per EUR at the booking date (last reference rate on/before it)."""
    fx = _cache(ctx).get("c_fx")
    if fx is None:
        fx = ctx.fx.select(pl.col("date").cast(pl.Utf8).str.to_date().alias("fx_date"), "currency",
                           pl.col("units_per_eur").cast(pl.Float64).alias("per_eur")).sort("fx_date")
        _cache(ctx)["c_fx"] = fx
    c = c.with_columns(pl.col("booking_date").str.to_date().alias("_bd")).sort("_bd")
    c = c.join_asof(fx, left_on="_bd", right_on="fx_date", by="currency", strategy="backward", check_sortedness=False)
    return c.with_columns(pl.when(pl.col("currency") == "EUR").then(1.0).otherwise(pl.col("per_eur")).alias("per_eur")).drop("_bd", "fx_date")


def _discount_cash(ctx: Ctx, items: pl.DataFrame) -> pl.DataFrame:
    """item_id -> cash that a within-terms early-payment discount would leave (POLICIES §3 SHORT_DISCOUNT) + deadline."""
    t = ctx.terms.select(pl.col("code").alias("payment_terms"),
                         pl.col("early_payment_discount_pct").cast(pl.Float64, strict=False).alias("dpct"),
                         pl.col("discount_days").cast(pl.Int32, strict=False).alias("ddays"))
    return items.join(t, on="payment_terms", how="inner").filter(pl.col("dpct") > 0).select(
        "item_id", (pl.col("total") - (pl.col("total") * pl.col("dpct") / 100).round(2)).round(2).alias("disc_cash"))


def _unique_solution(c: pl.DataFrame) -> pl.DataFrame:
    """Keep a line only when its formula has one solution: the single candidate item, or the single doc-cited one."""
    n = c.group_by("bank_txn_id").agg(pl.col("item_id").n_unique().alias("_n"), pl.col("doc").sum().alias("_nd"))
    c = c.join(n, on="bank_txn_id")
    c = c.filter((pl.col("_n") == 1) | ((pl.col("_nd") == 1) & pl.col("doc")))
    # one item per line and one line per item (an item explained twice is ambiguous → keep the strongest, earliest line)
    c = c.sort(["ev", "booking_date"], descending=[True, False])
    return c.unique("item_id", keep="first", maintain_order=True).drop("_n", "_nd")


def _exactly_explained(ctx: Ctx, c: pl.DataFrame, max_k: int = SUBSET_MAX_K, probe: pl.DataFrame | None = None) -> pl.DataFrame:
    """(bank_txn_id, party_id) of candidate pairs whose cash is explained WITHOUT a difference by the party's open documents:
    one, two or three open items (invoices +, credit notes −) summing exactly to the cash, or a cited document of the party
    other than the candidate (the payer told us which document it pays). Such a line has no shortfall/excess to explain.
    `probe` (bank_txn_id, party_id, entity, currency, cash, tnorm) widens the test to every evidenced party of the line."""
    keys = (probe if probe is not None else c).select("bank_txn_id", "party_id", "entity", "currency", "cash", "tnorm").unique(
        ["bank_txn_id", "party_id"])
    docs = ctx.items.filter(pl.col("item_type").is_in(["ARI", "ARC"]) & (pl.col("open_amount") > 0.005)).select(
        "item_id", "party_id", "entity", "currency", "docnorm",
        (pl.when(pl.col("item_type") == "ARC").then(-1).otherwise(1) * (pl.col("open_amount") * 100).round(0)).cast(pl.Int64).alias("v"))
    d = keys.join(docs, on=["party_id", "entity", "currency"])
    if not d.height:
        return keys.select("bank_txn_id", "party_id").clear()
    # cited other document (full normalised doc number in the text)
    cited = d.filter((pl.col("docnorm").str.len_chars() >= DOC_MIN_LEN) & pl.col("tnorm").str.contains(pl.col("docnorm"), literal=True))
    cand_items = c.select("bank_txn_id", "party_id", pl.col("item_id").alias("_cand"))
    cited_other = cited.join(cand_items, on=["bank_txn_id", "party_id"]).filter(pl.col("item_id") != pl.col("_cand")).join(
        cited.join(cand_items, left_on=["bank_txn_id", "party_id", "item_id"], right_on=["bank_txn_id", "party_id", "_cand"], how="semi")
        .select("bank_txn_id", "party_id"), on=["bank_txn_id", "party_id"], how="anti")
    hit = [cited_other.select("bank_txn_id", "party_id")]
    # exact subset sums of size 1..max_k (small per-party pools; arithmetic only)
    exact = []
    for (tid, pid), grp in d.group_by(["bank_txn_id", "party_id"]):
        target = int(round(float(grp["cash"][0]) * 100))
        vals = grp["v"].to_list()
        if _has_subset(vals, target, max_k):
            exact.append((tid, pid))
    if exact:
        hit.append(pl.DataFrame(exact, schema=["bank_txn_id", "party_id"], orient="row"))
    return pl.concat(hit).unique()


def _has_subset(vals: list[int], target: int, max_k: int) -> bool:
    if target in vals:
        return True
    if max_k < 2:
        return False
    seen = set()
    for v in vals:
        if target - v in seen:
            return True
        seen.add(v)
    if max_k < 3 or len(vals) > SUBSET_MAX_POOL:
        return False
    s = sorted(vals)
    n = len(s)
    for i in range(n - 2):
        lo, hi = i + 1, n - 1
        while lo < hi:
            t = s[i] + s[lo] + s[hi]
            if t == target:
                return True
            if t < target:
                lo += 1
            else:
                hi -= 1
    return False


def _open_invoices(ctx: Ctx, kind: str = "ARI") -> pl.DataFrame:
    return ctx.items.filter((pl.col("item_type") == kind) & (pl.col("open_amount") > 0.005)).select(
        "item_id", "entity", "party_id", "party_name", "currency", "docnorm", "issue_date", "payment_terms", "total",
        pl.col("open_amount").round(2).alias("open"))


# =====================================================================================================================
# SHORT_BANK_CHARGES (early)
# =====================================================================================================================
def bank_charges(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    """POLICIES §3: cross-border receipt short of one open invoice by 0 < short ≤ min(60 EUR eq., 2 % of the item).
    Cross-border = the payer's master country differs from the receiving entity's country. The shortfall must not be the
    invoice's own early-payment discount (that is SHORT_DISCOUNT / PARTIAL, handled by the discount resolvers)."""
    codes = pl.DataFrame(list(CROSS_BORDER_CODES), schema=["statement_format", "bank_code"], orient="row")
    xb = pending.filter(pl.col("is_credit")).select("bank_txn_id", "statement_format", "bank_code", "alltext")
    xb = pl.concat([xb.join(codes, on=["statement_format", "bank_code"], how="semi").select("bank_txn_id"),
                    xb.filter(pl.col("alltext").str.contains(CHARGE_FIELD_RE)).select("bank_txn_id")]).unique()
    lines = _lines(ctx, pending.join(xb, on="bank_txn_id", how="semi"), credit=True)
    items = _open_invoices(ctx)
    if not lines.height or not items.height:
        return empty_rows()
    slim = lines.select("bank_txn_id", "entity", "currency", "cash", "booking_date")
    c = _keyed(slim, items, "cash", "open", 1.0, 1.0 / (1.0 - CHARGES_MAX_PCT)).with_columns(
        (pl.col("open") - pl.col("cash")).round(2).alias("short"))
    c = c.filter((pl.col("short") > AMT_TOL) & (pl.col("short") <= pl.col("open") * CHARGES_MAX_PCT + AMT_TOL)
                 & (pl.col("issue_date") <= pl.col("booking_date")))
    if not c.height:
        return empty_rows()
    c = _per_eur(ctx, c).filter(pl.col("short") <= CHARGES_MAX_EUR * pl.col("per_eur") + AMT_TOL)
    # cross-border: payer country ≠ entity country
    c = c.join(_party_keys(ctx, "AR").select("party_id", "party_country"), on="party_id", how="left").join(
        _entity_country(ctx), on="entity", how="left").filter(
        pl.col("party_country").is_not_null() & (pl.col("party_country") != pl.col("entity_country")))
    # the shortfall is not an early-payment discount of that invoice
    c = c.join(_discount_cash(ctx, items), on="item_id", how="left").filter(
        pl.col("disc_cash").is_null() | ((pl.col("disc_cash") - pl.col("cash")).abs() > AMT_TOL))
    if not c.height:
        return empty_rows()
    c = c.join(lines.select("bank_txn_id", "tnorm", "druns", "cp_iban"), on="bank_txn_id").drop("party_country")
    c = _with_evidence(c, ctx, "AR").filter(pl.col("ev") > 0)
    if not c.height:
        return empty_rows()
    # the payer is the single best-evidenced party of the line (several companies can share a 10-char name prefix) …
    cl = lines.join(c.select("bank_txn_id").unique(), on="bank_txn_id", how="semi")
    probe = cl.join(_line_parties(ctx, cl, "AR", items), on="bank_txn_id")
    pc = _with_evidence(probe.join(items.select("party_id", "entity", "currency", "item_id", "party_name", "docnorm"),
                                   on=["party_id", "entity", "currency"]), ctx, "AR").filter(pl.col("ev") > 0)
    best = _best_party(pc).select("bank_txn_id", "party_id").unique()
    c = c.join(best, on=["bank_txn_id", "party_id"], how="semi")
    # … and no evidenced party of the line has documents that explain the cash exactly (then there is no shortfall)
    c = c.join(_exactly_explained(ctx, c, probe=probe).select("bank_txn_id").unique(), on="bank_txn_id", how="anti")
    if not c.height:
        return empty_rows()
    out = _unique_solution(c)
    return mk_rows(out.with_columns(pl.col("open").alias("amount")), "c_bank_charges", "SHORT_BANK_CHARGES")


# =====================================================================================================================
# DUPLICATE_PAYMENT (late)
# =====================================================================================================================
def duplicates(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    """POLICIES §3: an item already settled is paid again (customer pays twice / we pay a supplier twice). A pending line equal
    to the full total of a settled item of the evidenced party → that item, amount = what was paid again (= the total).
    Settled = known item no longer open (ledger), or an exact-amount item that an earlier line took (see module doc)."""
    out = []
    claimed = ctx.bank.join(pending.select("bank_txn_id"), on="bank_txn_id", how="anti")   # lines earlier resolvers explained
    for credit, kind, side in ((True, "ARI", "AR"), (False, "API", "AP")):
        lines = _lines(ctx, pending, credit)
        if not lines.height:
            continue
        # only items that were open when the window started: a repeat follows the first payment closely, and an item
        # closed long before is far more often an unrelated look-alike amount (POLICIES §3 UNIDENTIFIED warns about these)
        allx = ctx.all_items.filter((pl.col("item_type") == kind) & (pl.col("open_amount") > AMT_TOL)).select(
            "item_id", "entity", "party_id", "party_name", "currency", "docnorm", "issue_date", pl.col("total").round(2).alias("cash"))
        c = lines.join(allx, on=["entity", "currency", "cash"]).filter(pl.col("issue_date") <= pl.col("booking_date"))
        if not c.height:
            continue
        c = _with_evidence(c, ctx, side).filter(pl.col("ev") > 0)
        if not c.height:
            continue
        # (a) the item is no longer open: settled before the window, or by an earlier resolver (shared ledger)
        openx = ctx.items.filter(pl.col("item_type") == kind).select("item_id", pl.col("open_amount").round(2).alias("open"))
        c = c.join(openx, on="item_id", how="left")
        # (b) an earlier line of the same amount, with the same party evidence, was already accepted by an earlier resolver
        first = _lines(ctx, claimed, credit).select(
            pl.col("bank_txn_id").alias("first_txn"), pl.col("booking_date").alias("first_date"), "entity", "currency", "cash",
            "tnorm", "druns", "cp_iban")
        f = c.select("bank_txn_id", "booking_date", "item_id", "entity", "party_id", "party_name", "currency", "docnorm", "cash").join(
            first, on=["entity", "currency", "cash"]).filter(pl.col("first_date") <= pl.col("booking_date"))
        f = _with_evidence(f, ctx, side).filter(pl.col("ev") > 0).group_by("bank_txn_id", "item_id").agg(
            pl.col("first_date").max().alias("first_date"))
        c = c.join(f, on=["bank_txn_id", "item_id"], how="left").filter(pl.col("open").is_null() | pl.col("first_date").is_not_null())
        # same-amount FIFO (POLICIES §5.6): if the payer had several items of this amount open at the window start, an
        # earlier same-amount line may have paid another one and this line the next — not a repeat; abstain
        same = c.select("bank_txn_id", "party_id", "entity", "currency", "cash", "booking_date").unique().join(
            allx.select("party_id", "entity", "currency", "cash", pl.col("issue_date").alias("_iss")),
            on=["party_id", "entity", "currency", "cash"]).filter(pl.col("_iss") <= pl.col("booking_date")).group_by(
            "bank_txn_id", "party_id").agg(pl.len().alias("_nsame"))
        c = c.join(same, on=["bank_txn_id", "party_id"], how="left").filter(pl.col("_nsame").fill_null(0) <= 1)
        if not c.height:
            continue
        # one item per line: cited document, then the most recent first payment / most recent invoice (a repeat follows soon)
        c = c.sort(["bank_txn_id", "ev", "first_date", "issue_date"], descending=[False, True, True, True], nulls_last=True)
        n = c.group_by("bank_txn_id").agg(pl.col("party_id").n_unique().alias("_np"))
        c = c.join(n.filter(pl.col("_np") == 1), on="bank_txn_id", how="semi")
        c = c.unique("bank_txn_id", keep="first", maintain_order=True).unique("item_id", keep="first", maintain_order=True)
        out.append(c.select("bank_txn_id", "item_id", pl.col("cash").alias("amount")))
    if not out:
        return empty_rows()
    return mk_rows(pl.concat(out), "c_duplicate", "DUPLICATE_PAYMENT")
