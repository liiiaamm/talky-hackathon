"""Remittance-advice resolver (Persona C).

A customer's advice (csv/xlsx/txt/pdf, parsed by remittance_parse) lists the documents a receipt pays, with deductions and
reasons. We link each advice to the receipt (same entity, same amount, booking within a few days of the advice), map its
documents to our items, and set amounts/exceptions from the advice (POLICIES §2-§3):

    recognised reason, or a shortfall that satisfies a policy formula -> SHORT_DISCOUNT / PARTIAL (late) / SHORT_WITHHOLDING
    unrecognised stated reason                                        -> SHORT_DISPUTE (amount actually paid)
    unexplained shortfall                                             -> PARTIAL
    credit note rows                                                  -> positive amount, no exception
    receipt in another currency                                       -> FX_DIFFERENCE on fully-cleared invoice rows
See the CONFIG block for every constant and the policy clause it comes from.
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from .data import Ctx, norm_expr
from .deterministic import _acc_entity, empty_rows
from .remittance_parse import load_lines

# =====================================================================================================================
# CONFIG — every constant the resolver uses, with the policy clause it comes from. Nothing below is tuned on labels.
# To run this on another company's data, change this block only.
# =====================================================================================================================
# Linking an advice to its receipt.  POLICIES §5.4: "the advice can arrive a few days after the cash".
LAG_MIN, LAG_MAX = -5, 20          # advice received_date − bank booking_date, days (widened around the policy's "few days")
AMT_TOL = 0.011                    # cent rounding (scorer tolerance is max(0.01, 0.5 %) — POLICIES §2)
MIN_REF_DIGITS = 5                 # a quoted number must keep ≥ 5 digits to be matched by suffix (POLICIES §5.5: numbers
                                   # arrive without prefix / leading zeros; shorter suffixes are not unique)

# Early-payment discount.  POLICIES §3 SHORT_DISCOUNT: terms carry the %, booking ≤ invoice date + discount days + grace.
DISCOUNT_GRACE_DAYS = 3

# Withholding at source.  POLICIES §3 SHORT_WITHHOLDING: "some customers in Mexico, Brazil and Morocco; 1.25 %, 1.5 % or 10 %".
WHT_RATES = (0.0125, 0.015, 0.10)
WHT_COUNTRIES = {"MX", "BR", "MA"}   # empty set = no country gate (then a rate match alone decides)

# Vocabulary.  Used ONLY to read a *stated* reason. A reason we do not recognise falls through to the arithmetic below,
# so missing words cost nothing but precision on coincidences; add words freely. Whole words / word prefixes, upper-case.
DISCOUNT_PREFIXES = ("DISCOUNT", "DTO", "DESCUENTO", "DESCONTO", "ESCOMPTE", "SKONTO", "SCONTO", "RABATT", "EARLY")
WHT_WORDS = {"WHT", "ISR", "IRRF", "TDS", "RAS", "IRPF"}
WHT_PREFIXES = ("WITHHOLD", "RETENC", "RETENÇ", "RETENU", "RITENUT", "QUELLENST")
# ("RET" — the Brazilian advices' abbreviation for retenção — is deliberately NOT listed: the arithmetic fallback must
#  recover it on its own. That is the regression test for the vocabulary-free path.)

# Decision order for a shortfall on an invoice (POLICIES §3):
#   1. stated reason we recognise         -> that class (data check: 100 % of such rows also satisfy the formula)
#   2. shortfall satisfies a policy formula -> that class (discount % of total; withholding rate × total, customer in a
#                                             withholding country). Used for unstated or unrecognised reasons.
#   3. stated reason we do not recognise  -> SHORT_DISPUTE ("deducted an amount with a stated reason")
#   4. nothing explains it                 -> PARTIAL
# Measured cost of step 2 over-riding an unrecognised reason: 197 of 17,431 stated disputes in TRAIN (1.1 %) equal a
# withholding rate for a withholding-country customer; 0 equal a discount formula.
# =====================================================================================================================


def _words(reason: str) -> list[str]:
    import re
    return [w for w in re.split(r"[^A-ZÀ-Ü0-9]+", reason.upper()) if w]


def is_wht(reason: str) -> bool:
    return any(w in WHT_WORDS or w.startswith(WHT_PREFIXES) for w in _words(reason))


def is_discount(reason: str) -> bool:
    return any(w.startswith(DISCOUNT_PREFIXES) for w in _words(reason))


def fits(short: float, total: float, rate: float) -> bool:
    return abs(short - round(total * rate, 2)) <= AMT_TOL


def explain_shortfall(short: float, total: float, reason: str, discount_pct: float | None, in_wht_country: bool) -> str:
    """Return 'DISCOUNT' | 'WHT' | 'DISPUTE' | 'PARTIAL' for an invoice paid `short` less than its open amount."""
    if reason and is_discount(reason):
        return "DISCOUNT"
    if reason and is_wht(reason):
        return "WHT"
    if discount_pct and fits(short, total, discount_pct):
        return "DISCOUNT"
    if (in_wht_country or not WHT_COUNTRIES) and any(fits(short, total, r) for r in WHT_RATES):
        return "WHT"
    return "DISPUTE" if reason else "PARTIAL"


PHASES_ALL = ("train", "live", "final")


def _advices(ctx: Ctx) -> tuple[pl.DataFrame, pl.DataFrame] | None:
    """Parsed advices of EVERY phase folder available: an advice can arrive days after the cash, i.e. in the next phase's
    folder (POLICIES §5.4), so a receipt at the end of TRAIN/LIVE may have its advice in live/ / final/."""
    parts = [x for x in (load_lines(ph) for ph in PHASES_ALL) if x is not None and x.height]
    if not parts:
        return None
    lines = pl.concat(parts, how="vertical_relaxed").unique(["file", "doc", "gross", "paid"], keep="first")
    lines = lines.filter(pl.col("error").is_null() & pl.col("doc").is_not_null())
    adv = lines.group_by("file").agg(
        pl.first("entity"), pl.first("payer_name"), pl.first("received_date"), pl.first("ref"), pl.first("total"), pl.first("currency"))
    return adv, lines


def _link(ctx: Ctx, adv: pl.DataFrame, pending: pl.DataFrame) -> pl.DataFrame:
    """One bank line per advice and vice versa: exact amount, same entity, closest date; payment ref in the text breaks ties."""
    credits = (pending.filter(pl.col("is_credit"))
               .join(_acc_entity(ctx).select("account_id", "acc_entity"), on="account_id", how="left")
               .select("bank_txn_id", "acc_entity", "booking_date", "abs_amt", "orig_amt", pl.col("currency").alias("bank_ccy"), "alltext"))
    if not credits.height:
        return pl.DataFrame(schema={"file": pl.Utf8, "bank_txn_id": pl.Utf8, "bank_ccy": pl.Utf8})
    lo = (date.fromisoformat(credits["booking_date"].min()) + timedelta(days=LAG_MIN)).isoformat()
    hi = (date.fromisoformat(credits["booking_date"].max()) + timedelta(days=LAG_MAX)).isoformat()
    adv = adv.filter(pl.col("received_date").is_between(pl.lit(lo), pl.lit(hi)) & pl.col("total").is_not_null())
    cents = lambda c: (pl.col(c) * 100).round(0).cast(pl.Int64)  # noqa: E731  exact-amount join key
    adv = adv.with_columns(cents("total").alias("k"))
    cand = pl.concat([
        adv.join(credits.with_columns(cents("abs_amt").alias("k")), left_on=["entity", "k"], right_on=["acc_entity", "k"], how="inner"),
        adv.join(credits.filter(pl.col("orig_amt").is_not_null()).with_columns(cents("orig_amt").alias("k")),
                 left_on=["entity", "k"], right_on=["acc_entity", "k"], how="inner"),
    ], how="vertical_relaxed").unique(["file", "bank_txn_id"])
    cand = cand.with_columns(
        (pl.col("received_date").str.to_date() - pl.col("booking_date").str.to_date()).dt.total_days().alias("lag"),
        pl.when(pl.col("ref").is_not_null() & (pl.col("ref").str.len_chars() >= 6))
          .then(pl.col("alltext").str.contains(norm_expr("ref"), literal=False)).otherwise(False).alias("ref_hit"),
    ).filter(pl.col("lag").is_between(LAG_MIN, LAG_MAX))
    cand = cand.sort(["ref_hit", pl.col("lag").abs(), "booking_date"], descending=[True, False, False])
    used_f, used_b, keep = set(), set(), []
    for f, b, ccy in cand.select("file", "bank_txn_id", "bank_ccy").iter_rows():
        if f in used_f or b in used_b:
            continue
        used_f.add(f), used_b.add(b)
        keep.append((f, b, ccy))
    return pl.DataFrame(keep, schema={"file": pl.Utf8, "bank_txn_id": pl.Utf8, "bank_ccy": pl.Utf8}, orient="row")


def _map_docs(ctx: Ctx, docs: pl.DataFrame) -> pl.DataFrame:
    """Advice document -> item. Exact normalised number; else our number ending with the quoted digits for the same customer;
    else an open item of the same customer with exactly the gross amount (customer's own voucher numbers)."""
    items = ctx.items.filter(pl.col("item_type").is_in(["ARI", "ARC"])).select(
        "item_id", "item_type", "entity", "party_id", "docnorm", pl.col("currency").alias("item_ccy"), "total", "open_amount",
        "issue_date", "payment_terms", pl.col("po").alias("ipo"))
    # advice-level columns get their own names so item columns (total, currency…) never get suffixed in the joins
    docs = (docs.rename({"total": "adv_total", "currency": "adv_ccy"}, strict=False)
            .with_row_index("_r").with_columns(norm_expr("doc").alias("dn")))
    exact = docs.join(items, left_on=["entity", "dn"], right_on=["entity", "docnorm"], how="inner").unique("_r", keep="first")
    exact = pl.concat([exact, _map_entity_level(items, docs.join(exact.select("_r"), on="_r", how="anti")).select(exact.columns)],
                      how="vertical_relaxed")
    party = exact.group_by("file").agg(pl.col("party_id").mode().first().alias("party_id"))
    rest = docs.join(exact.select("_r"), on="_r", how="anti").join(party, on="file", how="inner")
    if rest.height:
        # keyed joins only (party + 5-digit suffix, party + amount in cents): big customers have thousands of items
        pitems = (items.join(rest.select("party_id").unique(), on="party_id", how="semi")
                  .with_columns(pl.col("docnorm").str.slice(-5).alias("s5"), (pl.col("total") * 100).round(0).cast(pl.Int64).alias("k")))
        suf = (rest.with_columns(pl.col("dn").str.replace(r"^[A-Z]+", "").str.strip_chars_start("0").alias("dig"))
               .filter(pl.col("dig").str.len_chars() >= MIN_REF_DIGITS)
               .with_columns(pl.col("dig").str.slice(-5).alias("s5"))
               .join(pitems, on=["party_id", "s5"], how="inner", suffix="_i")
               .filter(pl.col("docnorm").str.ends_with(pl.col("dig")) & (pl.col("entity") == pl.col("entity_i")))
               .unique("_r", keep="first"))
        rest2 = rest.join(suf.select("_r"), on="_r", how="anti").with_columns(
            (pl.col("gross").abs() * 100).round(0).cast(pl.Int64).alias("k"))
        amt = (rest2.join(pitems, on=["party_id", "k"], how="inner", suffix="_i")
               .filter(pl.col("entity") == pl.col("entity_i"))
               .sort("issue_date").unique("_r", keep="first"))
        exact = pl.concat([exact, suf.select(exact.columns), amt.select(exact.columns)], how="vertical_relaxed")
    return exact.unique(["file", "item_id"], keep="first")


def _map_entity_level(items: pl.DataFrame, rest: pl.DataFrame) -> pl.DataFrame:
    """Documents quoted without our prefix/year (e.g. the Sept-2025 voucher layout: Invoice '320871' = our INV24-0320871),
    keyed within the entity: our PO field == advice PO, or our number ends with the quoted digits; amount must agree."""
    if not rest.height:
        return rest.join(items, left_on=["entity", "dn"], right_on=["entity", "docnorm"], how="inner")
    ari = items.filter(pl.col("item_type") == "ARI")
    rest = rest.with_columns(pl.col("dn").str.replace(r"^[A-Z]+", "").str.strip_chars_start("0").alias("dig"),
                             norm_expr("po").alias("pon"))
    amt_ok = (pl.col("total") - pl.col("gross").abs()).abs() <= AMT_TOL
    by_po = (rest.filter(pl.col("pon").str.len_chars() >= MIN_REF_DIGITS)
             .join(ari.with_columns(norm_expr("ipo").alias("pon")).filter(pl.col("pon").str.len_chars() >= MIN_REF_DIGITS),
                   on=["entity", "pon"], how="inner", suffix="_i")
             .filter(amt_ok | pl.col("docnorm").str.ends_with(pl.col("dig")))
             .unique("_r", keep="first"))
    left = rest.join(by_po.select("_r"), on="_r", how="anti").filter(pl.col("dig").str.len_chars() >= MIN_REF_DIGITS)
    by_suffix = (left.with_columns(pl.col("dig").str.slice(-5).alias("s5"))
                 .join(ari.with_columns(pl.col("docnorm").str.slice(-5).alias("s5")), on=["entity", "s5"], how="inner", suffix="_i")
                 .filter(pl.col("docnorm").str.ends_with(pl.col("dig")) & amt_ok)
                 .unique("_r", keep="first"))
    return pl.concat([by_po, by_suffix], how="diagonal_relaxed")


def _terms(ctx: Ctx) -> dict:
    out = {}
    for code, pct, days in ctx.terms.select("code", "early_payment_discount_pct", "discount_days").iter_rows():
        if pct not in (None, "") and days not in (None, ""):
            out[code] = (float(pct) / 100, int(float(days)))
    return out


def remittances(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    a = _advices(ctx)
    if a is None:
        return empty_rows()
    adv, lines = a
    link = _link(ctx, adv, pending)
    if not link.height:
        return empty_rows()
    docs = lines.join(link, on="file", how="inner")
    m = _map_docs(ctx, docs).join(pending.select("bank_txn_id", "booking_date"), on="bank_txn_id", how="left")
    terms = _terms(ctx)
    country = dict(ctx.customers.select("customer_id", "country").iter_rows())

    rows = []
    for r in m.iter_rows(named=True):
        open_amt = r["open_amount"] if r["open_amount"] is not None else r["total"]
        paid = abs(r["paid"]) if r["paid"] is not None else abs(r["gross"])
        reason = (r["reason"] or "").upper()
        exc, amount = "", open_amt
        if r["item_type"] == "ARC":                                   # credit note netted: full amount, positive, no exception
            amount = abs(r["gross"]) if r["gross"] is not None else open_amt
        elif paid is not None and paid < open_amt - max(AMT_TOL, 0.005 * open_amt):
            short, total = open_amt - paid, r["total"]
            t = terms.get(r["payment_terms"])
            kind = explain_shortfall(short, total, reason, t[0] if t else None, country.get(r["party_id"]) in WHT_COUNTRIES)
            if kind == "DISCOUNT":
                in_terms = bool(t) and date.fromisoformat(r["booking_date"]) <= (
                    date.fromisoformat(r["issue_date"]) + timedelta(days=t[1] + DISCOUNT_GRACE_DAYS))
                exc, amount = ("SHORT_DISCOUNT", open_amt) if in_terms else ("PARTIAL", paid)   # late discount: not accepted
            elif kind == "WHT":
                exc = "SHORT_WITHHOLDING"                             # fully cleared, tax certificate covers it
            elif kind == "DISPUTE":
                exc, amount = "SHORT_DISPUTE", paid                   # disputed part stays open
            else:
                exc, amount = "PARTIAL", paid
        if r["item_type"] == "ARI" and r["bank_ccy"] != r["item_ccy"] and exc in ("", "SHORT_DISCOUNT", "SHORT_WITHHOLDING"):
            exc = "FX_DIFFERENCE"
        rows.append((r["bank_txn_id"], r["item_id"], round(float(amount), 2), None, exc, "remittance"))
    return pl.DataFrame(rows, schema={"bank_txn_id": pl.Utf8, "item_id": pl.Utf8, "amount": pl.Float64, "category": pl.Utf8,
                                      "exception": pl.Utf8, "module": pl.Utf8}, orient="row")


RESOLVERS = [remittances]
