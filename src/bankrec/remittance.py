"""Remittance-advice resolver (Persona C).

A customer's advice (csv/xlsx/txt/pdf, parsed by remittance_parse) lists the documents a receipt pays, with deductions and
reasons. We link each advice to the receipt (same entity, same amount, booking within a few days of the advice), map its
documents to our items, and set amounts/exceptions from the advice (POLICIES §2-§3):

    reason DISCOUNT       -> SHORT_DISCOUNT (within terms, full open amount) or PARTIAL (late, amount paid)
    reason withholding    -> SHORT_WITHHOLDING (full open amount)
    any other reason      -> SHORT_DISPUTE (amount actually paid)
    credit note rows      -> positive amount, no exception
    receipt in another ccy-> FX_DIFFERENCE on fully-cleared invoice rows
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from .data import Ctx, norm_expr
from .deterministic import _acc_entity, empty_rows
from .remittance_parse import load_lines

LAG_MIN, LAG_MAX = -5, 20          # advice received_date - bank booking_date, in days
AMT_TOL = 0.011
# withholding-tax reasons, matched as whole words / word prefixes ("RAS" must not hit "PENAL. RETRASO")
WHT_WORDS = {"RET", "WHT", "ISR", "IRRF", "TDS", "RAS"}
WHT_PREFIXES = ("WITHHOLD", "RETENC", "RETENÇ", "RETENU")
DISCOUNT_WORDS = ("DISCOUNT", "DTO", "DESCUENTO", "ESCOMPTE", "SKONTO", "SCONTO")
# POLICIES §3: withholding at source by some customers in MX/BR/MA, rates on the invoice total
WHT_RATES = (0.0125, 0.015, 0.10)
WHT_COUNTRIES = {"MX", "BR", "MA"}


def _words(reason: str) -> list[str]:
    import re
    return [w for w in re.split(r"[^A-ZÀ-Ü0-9]+", reason.upper()) if w]


def is_wht(reason: str) -> bool:
    ws = _words(reason)
    return any(w in WHT_WORDS or w.startswith(WHT_PREFIXES) for w in ws)


def is_discount(reason: str) -> bool:
    return any(w.startswith(DISCOUNT_WORDS) for w in _words(reason))


def _advices(ctx: Ctx) -> tuple[pl.DataFrame, pl.DataFrame] | None:
    lines = load_lines(ctx.phase)
    if lines is None or lines.height == 0:
        return None
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
               .filter(pl.col("dig").str.len_chars() >= 5)
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
    by_po = (rest.filter(pl.col("pon").str.len_chars() >= 5)
             .join(ari.with_columns(norm_expr("ipo").alias("pon")).filter(pl.col("pon").str.len_chars() >= 5),
                   on=["entity", "pon"], how="inner", suffix="_i")
             .filter(amt_ok | pl.col("docnorm").str.ends_with(pl.col("dig")))
             .unique("_r", keep="first"))
    left = rest.join(by_po.select("_r"), on="_r", how="anti").filter(pl.col("dig").str.len_chars() >= 5)
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

    def discount(r, open_amt, paid):
        t = terms.get(r["payment_terms"])
        ok = bool(t) and date.fromisoformat(r["booking_date"]) <= date.fromisoformat(r["issue_date"]) + timedelta(days=t[1] + 3)
        return ("SHORT_DISCOUNT", open_amt) if ok else ("PARTIAL", abs(paid))

    rows = []
    for r in m.iter_rows(named=True):
        open_amt = r["open_amount"] if r["open_amount"] is not None else r["total"]
        paid = r["paid"] if r["paid"] is not None else r["gross"]
        reason = (r["reason"] or "").upper()
        exc, amount = "", open_amt
        if r["item_type"] == "ARC":
            amount = abs(r["gross"]) if r["gross"] is not None else open_amt
        elif reason:
            if is_discount(reason):
                exc, amount = discount(r, open_amt, paid)
            elif is_wht(reason):                       # e.g. BR01 advices: "RET" = 10 % retenção na fonte
                exc = "SHORT_WITHHOLDING"
            else:
                exc, amount = "SHORT_DISPUTE", abs(paid)
        elif paid is not None and abs(paid) < open_amt - max(AMT_TOL, 0.005 * open_amt):
            # short without a stated reason: explain it with the policy formulas before calling it PARTIAL
            short, total = open_amt - abs(paid), r["total"]
            t = terms.get(r["payment_terms"])
            if t and abs(short - round(total * t[0], 2)) <= AMT_TOL:
                exc, amount = discount(r, open_amt, paid)
            elif country.get(r["party_id"]) in WHT_COUNTRIES and any(abs(short - round(total * x, 2)) <= AMT_TOL for x in WHT_RATES):
                exc = "SHORT_WITHHOLDING"
            else:
                exc, amount = "PARTIAL", abs(paid)
        if r["item_type"] == "ARI" and r["bank_ccy"] != r["item_ccy"] and exc in ("", "SHORT_DISCOUNT", "SHORT_WITHHOLDING"):
            exc = "FX_DIFFERENCE"
        rows.append((r["bank_txn_id"], r["item_id"], round(float(amount), 2), None, exc, "remittance"))
    return pl.DataFrame(rows, schema={"bank_txn_id": pl.Utf8, "item_id": pl.Utf8, "amount": pl.Float64, "category": pl.Utf8,
                                      "exception": pl.Utf8, "module": pl.Utf8}, orient="row")


RESOLVERS = [remittances]
