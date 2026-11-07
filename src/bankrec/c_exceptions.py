"""Rare-exception resolvers (Persona C): SHORT_BANK_CHARGES, DUPLICATE_PAYMENT, OVERPAYMENT.

Each resolver claims only pending lines it can explain with a POLICIES.md rule and customer/supplier evidence in the text
(name / IBAN / document number) — precision before recall. Pipeline contract: f(ctx, pending) -> rows (see deterministic.mk_rows).

Needs the shared ledger (BANKREC_LEDGER=1): duplicates are "an item already settled is paid again", which is only visible when
earlier resolvers' settlements have left ctx.items. Without the ledger only items settled before the window are seen.

=====================================================================================================================
CONFIG — every constant with its policy clause. Nothing tuned on labels.
=====================================================================================================================
"""
from __future__ import annotations

import os

import polars as pl

from .data import DATA, Ctx, norm_expr
from .deterministic import _acc_entity, empty_rows, mk_rows

AMT_TOL = 0.011                   # cent rounding (scorer tolerance max(0.01, 0.5 %) — POLICIES §2)
# POLICIES §3 SHORT_BANK_CHARGES: cross-border receipt, intermediary bank deducted its fee: shortfall ≤ 60 EUR equivalent and ≤ 2 %.
CHARGES_MAX_EUR = 60.0
CHARGES_MAX_PCT = 0.02
CHARGES_MIN = 0.50                # below this a shortfall is rounding, not a fee (keeps 1-cent cases out)
# "cross-border" evidence (both required): payer's country differs from the entity's country AND the bank marks the line as a
# cross-border / foreign-currency transfer or shows a charges field. ISO 20022 sub-family XBCT, SWIFT :86: /CHGS/, bank wording.
# Measured on TRAIN: payers in SEPA countries outside the euro area (CH, PL, GB…) do carry intermediary charges, so the
# effective rule is "payer outside the euro area", not the literal SEPA list. Euro-area membership is public knowledge.
EURO_AREA = {"AT", "BE", "CY", "DE", "EE", "ES", "FI", "FR", "GR", "HR", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PT", "SI", "SK",
             "AD", "MC", "SM", "VA", "ME", "XK"}
CROSS_BORDER_CODES = ("XBCT",)
CROSS_BORDER_TEXT = ("/CHGS/", "EN DIV", "INTERNACIONAL", "INTERNATIONAL", "ESTERO", "ETRANGER", "ÉTRANGER", "AUSLAND", "WIRE", "SWIFT")
# Arithmetic first (POLICIES §3): a shortfall that equals the invoice's early-payment discount or a withholding rate is that,
# not a bank fee — those are left to the discount / withholding rules.
WHT_RATES = (0.0125, 0.015, 0.10)
# POLICIES §5.10: a returned direct debit is re-presented or paid by transfer afterwards (no exception). A receipt equal to a
# direct-debit invoice that is no longer open is therefore NOT treated as a duplicate (abstain).
DUP_SKIP_PAYMENT_METHODS = ("dd",)
# POLICIES §3 OVERPAYMENT: paid more than the open amount of a single invoice; invoice cleared at open amount, excess stays.
OVERPAY_MAX_RATIO = 2.5           # above this the receipt is more likely several invoices or an unrelated payment
# POLICIES §3 DUPLICATE_PAYMENT: the item had already been settled and is paid again (customer twice / we pay a supplier twice).
#   amount = what was paid again (= the item total when paid in full).
# POLICIES §5.6 / §5.3: a receipt that equals an OPEN invoice of the customer (or the sum of two) is a normal payment of those
# documents — the matching modules own it. Both resolvers abstain on such lines instead of calling them fee / duplicate.
PAIR_CAP = 30                     # pair sums are checked for customers with up to this many open invoices (n² pairs)
NAME_KEY_LEN = 10                 # same evidence rule as ar_exact: first 10 normalised chars of a party name in the bank text
MIN_KEY_LEN = 5


# ---------------------------------------------------------------------------------------------------------------- helpers
def _entity_country(ctx: Ctx) -> dict:
    cache = ctx.__dict__.setdefault("_cache", {})
    if "entity_country" not in cache:
        p = os.path.join(DATA, ctx.phase, "masters_final", "entities.csv") if ctx.phase == "final" else None
        if not (p and os.path.exists(p)):
            p = os.path.join(DATA, "masters", "entities.csv")
        e = pl.read_csv(p, infer_schema_length=0)
        cache["entity_country"] = dict(e.select("entity", "country").iter_rows())
    return cache["entity_country"]


def _eur_rate(ctx: Ctx) -> pl.DataFrame:
    """currency, date -> units_per_eur (last available rate on/before the date is applied with an asof join)."""
    return ctx.fx.select(pl.col("date").str.to_date().alias("fx_date"), "currency",
                         pl.col("units_per_eur").cast(pl.Float64).alias("per_eur")).sort("fx_date")


def _party_keys(df: pl.DataFrame, id_col: str, name_cols: tuple[str, ...], iban_col: str | None) -> pl.DataFrame:
    cols = [pl.col(id_col).alias("party_id")]
    for i, c in enumerate(name_cols):
        cols.append(norm_expr(c).str.slice(0, NAME_KEY_LEN).alias(f"k{i}"))
    cols.append(pl.col(iban_col).alias("iban") if iban_col and iban_col in df.columns else pl.lit(None, pl.Utf8).alias("iban"))
    return df.select(cols)


def _has(k: str) -> pl.Expr:
    return ((pl.col(k).str.len_chars() >= MIN_KEY_LEN) & pl.col("tnorm").str.contains(pl.col(k), literal=True)).fill_null(False)


def _evidence(c: pl.DataFrame, keys: pl.DataFrame) -> pl.DataFrame:
    """Add doc / name / iban evidence flags to candidate pairs (needs tnorm, docnorm, party_id, cp_iban)."""
    c = c.join(keys, on="party_id", how="left")
    kcols = [k for k in keys.columns if k.startswith("k")]
    digits = pl.col("docnorm").str.replace_all(r"\D", "")
    name = pl.any_horizontal([_has(k) for k in kcols]) if kcols else pl.lit(False)
    return c.with_columns(
        (((pl.col("docnorm").str.len_chars() >= 6) & pl.col("tnorm").str.contains(pl.col("docnorm"), literal=True))
         | ((digits.str.len_chars() >= 6) & pl.col("tnorm").str.contains(digits.str.slice(-6), literal=True))).fill_null(False).alias("doc"),
        name.alias("name"),
        (pl.col("cp_iban").is_not_null() & (pl.col("cp_iban") == pl.col("iban"))).fill_null(False).alias("iban_ok"),
    )


def _lines(ctx: Ctx, pending: pl.DataFrame, credit: bool) -> pl.DataFrame:
    side = pl.col("is_credit") if credit else ~pl.col("is_credit")
    return (pending.filter(side & pl.col("orig_amt").is_null())        # same-currency lines only (FX lines are FX_DIFFERENCE)
            .join(_acc_entity(ctx), on="account_id")
            .select("bank_txn_id", "booking_date", "currency", pl.col("acc_entity").alias("entity"),
                    pl.col("abs_amt").round(2).alias("cash"), norm_expr("alltext").alias("tnorm"), "alltext", "bank_code",
                    pl.col("counterparty_account").alias("cp_iban")))


def _customer_keys(ctx: Ctx) -> pl.DataFrame:
    return _party_keys(ctx.customers, "customer_id", ("name", "trade_name"), "iban")


def _supplier_keys(ctx: Ctx) -> pl.DataFrame:
    return _party_keys(ctx.suppliers, "supplier_id", ("name",), "iban")


def _lbucket(col: str, factor: float) -> pl.Expr:
    """Logarithmic amount bucket: amounts whose ratio is < factor fall in the same or the next bucket. Lets us join lines to
    items by (entity, currency, bucket) instead of a cartesian (entity, currency) join that blows up on a half-year."""
    import math
    return (pl.col(col).clip(lower_bound=0.01).log() / math.log(factor)).floor().cast(pl.Int64)


def _keyed(lines: pl.DataFrame, items: pl.DataFrame, line_amt: str, item_amt: str, factor: float, offsets: tuple[int, ...]) -> pl.DataFrame:
    """Join lines×items on (entity, currency, bucket) where the item bucket may be offset from the line bucket."""
    it = items.with_columns(_lbucket(item_amt, factor).alias("_k"))
    ln = lines.with_columns(_lbucket(line_amt, factor).alias("_k0"))
    parts = [ln.with_columns((pl.col("_k0") + d).alias("_k")) for d in offsets]
    return pl.concat(parts).join(it, on=["entity", "currency", "_k"]).drop("_k", "_k0")


def _cents(col: str) -> pl.Expr:
    return (pl.col(col) * 100).round(0).cast(pl.Int64)


def _explained_by_open(ctx: Ctx, c: pl.DataFrame, item_type: str = "ARI") -> pl.DataFrame:
    """Candidate pairs (bank_txn_id, party_id, entity, currency, cash) whose cash equals one open item of that party, or the
    sum of two. Returns the bank_txn_ids to drop."""
    keys = c.select("bank_txn_id", "party_id", "entity", "currency", _cents("cash").alias("ck")).unique()
    inv = (ctx.items.filter((pl.col("item_type") == item_type) & (pl.col("open_amount") > 0))
           .join(keys.select("party_id").unique(), on="party_id", how="semi")
           .select("item_id", "party_id", "entity", "currency", _cents("open_amount").alias("ck")))
    one = keys.join(inv, on=["party_id", "entity", "currency", "ck"], how="semi")
    small = inv.join(inv.group_by("party_id").len().filter(pl.col("len") <= PAIR_CAP).select("party_id"), on="party_id", how="semi")
    pairs = (small.join(small, on=["party_id", "entity", "currency"], suffix="_b").filter(pl.col("item_id") < pl.col("item_id_b"))
             .select("party_id", "entity", "currency", (pl.col("ck") + pl.col("ck_b")).alias("ck")))
    two = keys.join(pairs, on=["party_id", "entity", "currency", "ck"], how="semi")
    return pl.concat([one, two]).select("bank_txn_id").unique()


def _pick_one(c: pl.DataFrame, score: pl.Expr) -> pl.DataFrame:
    """Greedy one line ↔ one item, best evidence first, then oldest invoice (FIFO, POLICIES §5.6)."""
    c = c.with_columns(score.alias("score")).sort(["score", "issue_date", "booking_date"], descending=[True, False, False])
    return c.unique("bank_txn_id", keep="first", maintain_order=True).unique("item_id", keep="first", maintain_order=True)


# ------------------------------------------------------------------------------------------------- SHORT_BANK_CHARGES
def bank_charges(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    """Cross-border receipt a little short of exactly one open invoice of an evidenced customer: fee deducted by an intermediary
    bank. Row = full open amount, SHORT_BANK_CHARGES."""
    lines = _lines(ctx, pending, credit=True)
    if not lines.height:
        return empty_rows()
    terms = ctx.terms.select(pl.col("code").alias("payment_terms"),
                             (pl.col("early_payment_discount_pct").cast(pl.Float64, strict=False) / 100).alias("dpct"))
    items = ctx.items.filter((pl.col("item_type") == "ARI") & (pl.col("open_amount") > 0)).select(
        "item_id", "entity", "party_id", "currency", "docnorm", "issue_date", "total", "payment_terms",
        pl.col("open_amount").round(2).alias("open")).join(terms, on="payment_terms", how="left")
    # keyed join: open ∈ (cash, cash·1.0204] ⇒ same or next 3 %-log bucket; numeric filter before any text column is attached
    slim = lines.select("bank_txn_id", "entity", "currency", "cash", "booking_date")
    c = _keyed(slim, items, "cash", "open", 1.03, (0, 1)).with_columns((pl.col("open") - pl.col("cash")).alias("short"))
    c = c.filter((pl.col("short") >= CHARGES_MIN) & (pl.col("short") <= pl.col("open") * CHARGES_MAX_PCT + AMT_TOL)
                 & (pl.col("issue_date") <= pl.col("booking_date")))
    # arithmetic first: shortfall equal to the discount or a withholding rate is not a fee
    formula = pl.any_horizontal([((pl.col("short") - (pl.col("total") * r).round(2)).abs() <= AMT_TOL) for r in WHT_RATES]
                                + [((pl.col("short") - (pl.col("total") * pl.col("dpct")).round(2)).abs() <= AMT_TOL).fill_null(False)])
    c = c.filter(~formula)
    if not c.height:
        return empty_rows()
    c = c.join(lines.select("bank_txn_id", "tnorm", "alltext", "bank_code", "cp_iban"), on="bank_txn_id")
    # ≤ 60 EUR equivalent at the booking date
    c = c.with_columns(pl.col("booking_date").str.to_date().alias("bd")).sort("bd")
    c = c.join_asof(_eur_rate(ctx), left_on="bd", right_on="fx_date", by="currency", strategy="backward", check_sortedness=False)
    c = c.with_columns(pl.when(pl.col("currency") == "EUR").then(1.0).otherwise(pl.col("per_eur")).alias("per_eur"))
    c = c.filter((pl.col("short") / pl.col("per_eur")) <= CHARGES_MAX_EUR + AMT_TOL)
    # evidence + cross-border
    c = _evidence(c, _customer_keys(ctx)).join(
        ctx.customers.select(pl.col("customer_id").alias("party_id"), pl.col("country").alias("payer_country")), on="party_id", how="left")
    ec = _entity_country(ctx)
    c = c.with_columns(pl.col("entity").replace_strict(ec, default=None).alias("entity_country"))
    marker = pl.any_horizontal([pl.col("alltext").str.to_uppercase().str.contains(m, literal=True) for m in CROSS_BORDER_TEXT]
                               + [pl.col("bank_code").fill_null("").str.contains(m, literal=True) for m in CROSS_BORDER_CODES])
    c = c.filter((pl.col("doc") | pl.col("name") | pl.col("iban_ok"))
                 & (pl.col("payer_country") != pl.col("entity_country")) & ~pl.col("payer_country").is_in(list(EURO_AREA)) & marker)
    if not c.height:
        return empty_rows()
    c = c.join(_explained_by_open(ctx, c), on="bank_txn_id", how="anti")      # an exact open match exists -> not a fee
    if not c.height:
        return empty_rows()
    # a line must have exactly one explanation among this customer's invoices (several near-miss invoices = ambiguous)
    n = c.group_by("bank_txn_id").agg(pl.col("item_id").n_unique().alias("n"), pl.col("doc").any().alias("anydoc"))
    c = c.join(n, on="bank_txn_id").filter((pl.col("n") == 1) | pl.col("doc"))
    out = _pick_one(c, pl.col("doc").cast(pl.Int32) * 4 + pl.col("iban_ok").cast(pl.Int32) * 2 + pl.col("name").cast(pl.Int32))
    return mk_rows(out.with_columns(pl.col("open").alias("amount")), "c_bank_charges", "SHORT_BANK_CHARGES")


# ------------------------------------------------------------------------------------------------- DUPLICATE_PAYMENT
def _settled(ctx: Ctx, item_type: str) -> pl.DataFrame:
    """Items of a type that exist but are no longer open (settled before the window, or by an earlier resolver via the ledger)."""
    allx = ctx.all_items.filter((pl.col("item_type") == item_type)
                                & ~pl.col("payment_method").fill_null("").is_in(list(DUP_SKIP_PAYMENT_METHODS)))
    return allx.join(ctx.items.select("item_id"), on="item_id", how="anti").select(
        "item_id", "entity", "party_id", "currency", "docnorm", "issue_date", pl.col("total").round(2).alias("total"))


def duplicates(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    """A pending line equal to the full total of an already-settled item of an evidenced party (customer receipt or supplier
    debit) is the item paid a second time. Row = the amount paid again, DUPLICATE_PAYMENT."""
    out = []
    for credit, item_type, keys in ((True, "ARI", _customer_keys(ctx)), (False, "API", _supplier_keys(ctx))):
        lines = _lines(ctx, pending, credit)
        if not lines.height:
            continue
        items = _settled(ctx, item_type)
        if not items.height:
            continue
        c = lines.join(items, left_on=["entity", "currency", "cash"], right_on=["entity", "currency", "total"]).filter(
            pl.col("issue_date") <= pl.col("booking_date"))
        if not c.height:
            continue
        c = _evidence(c, keys).filter(pl.col("doc") | pl.col("name") | pl.col("iban_ok"))
        if not c.height:
            continue
        c = c.join(_explained_by_open(ctx, c, item_type), on="bank_txn_id", how="anti")   # an open item explains it -> not a duplicate
        if not c.height:
            continue
        out.append(_pick_one(c, pl.col("doc").cast(pl.Int32) * 4 + pl.col("iban_ok").cast(pl.Int32) * 2 + pl.col("name").cast(pl.Int32))
                   .select("bank_txn_id", "item_id", pl.col("cash").alias("amount")))   # common schema for AR and AP parts
    if not out:
        return empty_rows()
    return mk_rows(pl.concat(out), "c_duplicate", "DUPLICATE_PAYMENT")


# ------------------------------------------------------------------------------------------------------ OVERPAYMENT
def overpayments(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    """Receipt from an evidenced customer that exceeds its only open invoice (same entity/currency): the invoice is cleared at
    its open amount, the excess stays on account. Row = open amount, OVERPAYMENT."""
    lines = _lines(ctx, pending, credit=True)
    if not lines.height:
        return empty_rows()
    items = ctx.items.filter((pl.col("item_type") == "ARI") & (pl.col("open_amount") > 0)).select(
        "item_id", "entity", "party_id", "currency", "docnorm", "issue_date", pl.col("open_amount").round(2).alias("open"))
    # customers with exactly one open invoice in that entity+currency
    one = items.group_by("entity", "party_id", "currency").agg(pl.len().alias("n_open")).filter(pl.col("n_open") == 1)
    items = items.join(one.select("entity", "party_id", "currency"), on=["entity", "party_id", "currency"], how="semi")
    # keyed join: open ∈ [cash/2.5, cash) ⇒ item bucket within 3 below the line's 1.5-log bucket (1.5^3 > 2.5)
    slim = lines.select("bank_txn_id", "entity", "currency", "cash", "booking_date")
    c = _keyed(slim, items, "cash", "open", 1.5, (0, -1, -2, -3)).filter(
        (pl.col("cash") > pl.col("open") + AMT_TOL) & (pl.col("cash") <= pl.col("open") * OVERPAY_MAX_RATIO)
        & (pl.col("issue_date") <= pl.col("booking_date")))
    if not c.height:
        return empty_rows()
    c = c.join(lines.select("bank_txn_id", "tnorm", "cp_iban"), on="bank_txn_id")
    c = _evidence(c, _customer_keys(ctx)).filter(pl.col("doc") | pl.col("name") | pl.col("iban_ok"))
    if not c.height:
        return empty_rows()
    # one customer per line: if the text evidences two different customers, abstain
    n = c.group_by("bank_txn_id").agg(pl.col("party_id").n_unique().alias("np"))
    c = c.join(n, on="bank_txn_id").filter(pl.col("np") == 1)
    out = _pick_one(c, pl.col("doc").cast(pl.Int32) * 4 + pl.col("iban_ok").cast(pl.Int32) * 2 + pl.col("name").cast(pl.Int32))
    return mk_rows(out.with_columns(pl.col("open").alias("amount")), "c_overpayment", "OVERPAYMENT")


# ------------------------------------------------------------------------------------------------------ FX_DIFFERENCE
# POLICIES §3 FX_DIFFERENCE: the bank line is in a different currency than the invoice; the payer pays the invoice amount in
# the invoice currency (the bank shows the instructed amount) and converts at its own rate. Every AR matcher skips lines
# with orig_amount, so these receipts were never matched. We present them to the existing matchers *in the instructed
# currency* (abs_amt := orig_amount, currency := orig_currency) and relabel: fully-cleared invoice rows -> FX_DIFFERENCE
# (replacing empty / SHORT_DISCOUNT / SHORT_WITHHOLDING / SHORT_BANK_CHARGES); PARTIAL / SHORT_DISPUTE keep; credit notes empty.
FX_REPLACES = ("", "SHORT_DISCOUNT", "SHORT_WITHHOLDING", "SHORT_BANK_CHARGES", "OVERPAYMENT")


def fx_receipts(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    from .ar_exact import adjusted_amount, exact_amount
    from .ar_multi import multi_invoice
    fx = pending.filter(pl.col("is_credit") & pl.col("orig_amt").is_not_null() & (pl.col("orig_amt") > 0)
                        & pl.col("orig_currency").is_not_null() & (pl.col("orig_currency") != pl.col("currency")))
    if not fx.height:
        return empty_rows()
    p2 = fx.with_columns(pl.col("orig_amt").alias("abs_amt"), pl.col("orig_amt").alias("amt"), pl.col("orig_currency").alias("currency"),
                         pl.col("orig_amt").map_elements(lambda v: f"{v:.2f}", return_dtype=pl.Utf8).alias("amount"),   # raw string column used by ar_text
                         pl.lit(None, pl.Float64).alias("orig_amt"), pl.lit(None, pl.Utf8).alias("orig_currency"),
                         pl.lit("", pl.Utf8).alias("orig_amount"))
    # (ar_text was tried here too: on the transformed frame it adds ~16 lines/window at ~25 % precision -> not used)
    rows, pend = [], p2
    for f in (exact_amount, adjusted_amount, multi_invoice):
        r = f(ctx, pend)
        if r.height:
            rows.append(r)
            pend = pend.join(r.select("bank_txn_id").unique(), on="bank_txn_id", how="anti")
        if not pend.height:
            break
    if not rows:
        return empty_rows()
    out = pl.concat(rows).join(ctx.items.select("item_id", "item_type"), on="item_id", how="left")
    exc = pl.col("exception").fill_null("")
    out = out.with_columns(
        pl.when(pl.col("item_type") == "ARC").then(pl.lit(""))
          .when(exc.is_in(list(FX_REPLACES))).then(pl.lit("FX_DIFFERENCE")).otherwise(exc).alias("exception"),
        pl.concat_str([pl.lit("c_fx_"), pl.col("module")]).alias("module"))
    return out.select("bank_txn_id", "item_id", "amount", "category", "exception", "module")


# ---------------------------------------------------------------------------------------------- SHORT_DISCOUNT (1:N)
# POLICIES §3 SHORT_DISCOUNT: terms carry a discount %; cash = total − % × total (to the cent) when booking ≤ invoice date +
# discount days + 3 days of grace (invoice fully cleared); after the deadline the discount is not accepted → PARTIAL at the
# cash. ar_adjusted handles one discounted invoice per line; receipts that pay SEVERAL discounted invoices are not matched
# by anyone. Same pattern as the FX pass: re-run the existing matchers on a view of the open items where every untouched
# invoice with discount terms is valued at its discounted amount, then relabel per the policy.
DISCOUNT_GRACE_DAYS = 3


def discount_receipts(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    import copy
    from .ar_exact import exact_amount
    from .ar_multi import multi_invoice
    lines = pending.filter(pl.col("is_credit") & pl.col("orig_amt").is_null())
    if not lines.height:
        return empty_rows()
    terms = ctx.terms.select(pl.col("code").alias("payment_terms"),
                             (pl.col("early_payment_discount_pct").cast(pl.Float64, strict=False) / 100).alias("dpct"),
                             pl.col("discount_days").cast(pl.Int32, strict=False).alias("ddays")).filter(pl.col("dpct") > 0)
    items = ctx.items.join(terms, on="payment_terms", how="left")
    disc = (pl.col("item_type") == "ARI") & pl.col("dpct").is_not_null() & ((pl.col("open_amount") - pl.col("total")).abs() <= AMT_TOL)
    if not items.filter(disc).height:
        return empty_rows()
    view = items.with_columns(
        pl.when(disc).then((pl.col("total") * (1 - pl.col("dpct"))).round(2)).otherwise(pl.col("open_amount")).alias("open_amount"),
        pl.col("open_amount").alias("open_full")).drop("dpct", "ddays")
    ctx2 = copy.copy(ctx)
    ctx2.items = view
    # exact_amount only: on the discounted view the multi-invoice matcher is 38-45 % right (its FIFO/pair rules find
    # coincidental sums once amounts are shifted); the single-invoice exact match stays at 87-93 %.
    rows, pend = [], lines
    for f in (exact_amount,):
        r = f(ctx2, pend)
        if r.height:
            rows.append(r)
            pend = pend.join(r.select("bank_txn_id").unique(), on="bank_txn_id", how="anti")
        if not pend.height:
            break
    if not rows:
        return empty_rows()
    out = (pl.concat(rows)
           .join(view.select("item_id", "item_type", "open_full", "issue_date", "payment_terms", pl.col("open_amount").alias("open_disc")), on="item_id", how="left")
           .join(terms, on="payment_terms", how="left")
           .join(lines.select("bank_txn_id", "booking_date"), on="bank_txn_id", how="left"))
    discounted = (pl.col("item_type") == "ARI") & pl.col("dpct").is_not_null() & ((pl.col("amount") - pl.col("open_disc")).abs() <= AMT_TOL) \
        & ((pl.col("open_full") - pl.col("amount")).abs() > AMT_TOL)
    in_terms = pl.col("booking_date").str.to_date() <= pl.col("issue_date").str.to_date().dt.offset_by(
        pl.concat_str([(pl.col("ddays").fill_null(0) + DISCOUNT_GRACE_DAYS).cast(pl.Utf8), pl.lit("d")]))
    out = out.with_columns(
        pl.when(discounted & in_terms).then(pl.lit("SHORT_DISCOUNT")).when(discounted).then(pl.lit("PARTIAL"))
          .otherwise(pl.col("exception").fill_null("")).alias("exception"),
        pl.when(discounted & in_terms).then(pl.col("open_full")).otherwise(pl.col("amount")).alias("amount"),
        pl.concat_str([pl.lit("c_disc_"), pl.col("module")]).alias("module"))
    return out.select("bank_txn_id", "item_id", "amount", "category", "exception", "module")


# Exposure (measured on feat/b-ar-hard, both TRAIN windows, see PR):
#   bank_charges  -> c_bank_charges.py, placed AFTER ar_text/ar_exact/ar_multi: it only takes what ar_text's own
#                    bank-charges rule leaves (SHORT_BANK_CHARGES F1 0.745 -> 0.823 / 0.743 -> 0.785). Placed before
#                    ar_text it steals lines and is a wash.
#   fx_receipts   -> c_fx.py, last.
#   duplicates    -> NOT exposed: on top of ar_text's duplicate rule (100 % precision) it adds 200 lines at 45 %.
#   overpayments  -> NOT exposed: 6/43 precision on the sample (a parent paying for subsidiaries defeats the
#                    single-open-invoice test). Both kept as reference implementations.
# Recommended BANKREC_ORDER: remittance,pdf_clauses,ar_text,ar_exact,ar_multi,c_bank_charges,c_fx
RESOLVERS: list = []
