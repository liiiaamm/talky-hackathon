"""AR receipts that pay exactly one open invoice of the payer (ar_1to1, single-invoice ar_ref_noise, same_amount_fifo/month…).

Candidate = open ARI of an allowed entity with open_amount == bank amount (same currency), issued on/before the booking date.
Evidence (at least one required, so unidentified look-alikes are not forced):
  iban   camt counterparty_account == customer IBAN in the master
  name   first 10 normalised chars of customer name / trade name / invoice name appear in the bank text
  doc    our printed doc number appears (normalised) / its last 6 digits appear
Ranking among candidates of a line: doc > doc6 > service month matches > oldest open invoice (FIFO).
One invoice can be used once; lines that lose their best invoice get their next candidate (few rounds).
"""
from __future__ import annotations

import polars as pl

from .data import Ctx, norm_expr
from .deterministic import _acc_entity, empty_rows, mk_rows

MONTHS = {
    1: "ENERO|JANUARY|JANVIER|JANUAR|GENNAIO|JANEIRO|STYCZEN",
    2: "FEBRERO|FEBRUARY|FEVRIER|FÉVRIER|FEBRUAR|FEBBRAIO|FEVEREIRO",
    3: "MARZO|MARCH|MARS|MÄRZ|MAERZ|MARCO|MARÇO",
    4: "ABRIL|APRIL|AVRIL|APRILE",
    5: "MAYO|MAY|MAI|MAGGIO|MAIO",
    6: "JUNIO|JUNE|JUIN|JUNI|GIUGNO|JUNHO",
    7: "JULIO|JULY|JUILLET|JULI|LUGLIO|JULHO",
    8: "AGOSTO|AUGUST|AOUT|AOÛT",
    9: "SEPTIEMBRE|SETIEMBRE|SEPTEMBER|SEPTEMBRE|SETTEMBRE|SETEMBRO",
    10: "OCTUBRE|OCTOBER|OCTOBRE|OKTOBER|OTTOBRE|OUTUBRO",
    11: "NOVIEMBRE|NOVEMBER|NOVEMBRE|NOVEMBRO",
    12: "DICIEMBRE|DECEMBER|DECEMBRE|DÉCEMBRE|DEZEMBER|DICEMBRE|DEZEMBRO",
}


def _service_month(col: str = "alltext") -> pl.Expr:
    e = pl.lit(None, pl.Int32)
    for m, names in MONTHS.items():
        e = pl.when(pl.col(col).str.contains(rf"\b(?:{names})\b")).then(pl.lit(m, pl.Int32)).otherwise(e)
    return e


def _cancelled(ctx: Ctx) -> pl.DataFrame:
    """ARI fully cancelled by an ARC that rectifies it with the same total ("anula y sustituye"): never settled by bank."""
    if "cancelled" not in ctx.__dict__.setdefault("_cache", {}):
        arc = ctx.all_items.filter((pl.col("item_type") == "ARC") & pl.col("related_document").is_not_null()).select(
            "entity", norm_expr("related_document").alias("docnorm"), pl.col("total").round(2).alias("t"))
        ari = ctx.all_items.filter(pl.col("item_type") == "ARI").select("item_id", "entity", "docnorm", pl.col("total").round(2).alias("t"))
        ctx._cache["cancelled"] = ari.join(arc, on=["entity", "docnorm", "t"], how="semi").select("item_id")
    return ctx._cache["cancelled"]


def _names(ctx: Ctx) -> pl.DataFrame:
    """customer_id -> list of 10-char normalised name keys (name, trade name) + master IBAN."""
    c = ctx.customers.select(pl.col("customer_id").alias("party_id"), "iban",
                             norm_expr("name").str.slice(0, 10).alias("k1"), norm_expr("trade_name").str.slice(0, 10).alias("k2"))
    return c


def exact_amount(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    lines = pending.filter(pl.col("is_credit") & pl.col("orig_amt").is_null())
    if lines.height == 0:
        return empty_rows()
    lines = lines.join(_acc_entity(ctx), on="account_id").select(
        "bank_txn_id", "booking_date", "currency", pl.col("acc_entity").alias("entity"), pl.col("abs_amt").round(2).alias("amt2"),
        norm_expr("alltext").alias("tnorm"), pl.col("counterparty_account").alias("cp_iban"), _service_month().alias("smonth"))
    items = ctx.items.filter((pl.col("item_type") == "ARI") & (pl.col("open_amount") > 0)).select(
        "item_id", "entity", "party_id", "currency", "docnorm", "issue_date", pl.col("open_amount").round(2).alias("amt2"),
        norm_expr("party_name").str.slice(0, 10).alias("k0")).join(_cancelled(ctx), on="item_id", how="anti")
    c = lines.join(items, on=["entity", "amt2", "currency"]).filter(pl.col("issue_date") <= pl.col("booking_date"))
    c = c.join(_names(ctx), on="party_id", how="left")

    def has(k):
        return (pl.col(k).str.len_chars() >= 5) & pl.col("tnorm").str.contains(pl.col(k), literal=True)

    digits = pl.col("docnorm").str.replace_all(r"\D", "")
    c = c.with_columns(
        ((pl.col("docnorm").str.len_chars() >= 6) & pl.col("tnorm").str.contains(pl.col("docnorm"), literal=True)).alias("doc"),
        ((digits.str.len_chars() >= 6) & pl.col("tnorm").str.contains(digits.str.slice(-6), literal=True)).alias("doc6"),
        (has("k0").fill_null(False) | has("k1").fill_null(False) | has("k2").fill_null(False)).alias("name"),
        (pl.col("cp_iban").is_not_null() & (pl.col("cp_iban") == pl.col("iban"))).fill_null(False).alias("iban_ok"),
        (pl.col("smonth").is_not_null() & (pl.col("issue_date").str.slice(5, 2).cast(pl.Int32) == (pl.col("smonth") % 12) + 1))
        .fill_null(False).alias("month_ok"),
    )
    c = c.filter(pl.col("doc") | pl.col("doc6") | pl.col("name") | pl.col("iban_ok"))
    c = c.with_columns((pl.col("doc").cast(pl.Int32) * 8 + pl.col("doc6").cast(pl.Int32) * 4 + pl.col("month_ok").cast(pl.Int32) * 2
                        + (pl.col("name") | pl.col("iban_ok")).cast(pl.Int32)).alias("score"))
    # greedy one-to-one assignment, best evidence first, then FIFO (oldest invoice) and earliest bank line
    c = c.sort(["score", "issue_date", "booking_date"], descending=[True, False, False]).select(
        "bank_txn_id", "item_id", "amt2", "score", "issue_date", "booking_date")
    chosen = []
    for _ in range(4):
        if c.height == 0:
            break
        best = c.unique("bank_txn_id", keep="first", maintain_order=True).unique("item_id", keep="first", maintain_order=True)
        chosen.append(best)
        c = c.join(best.select("bank_txn_id"), on="bank_txn_id", how="anti").join(best.select("item_id"), on="item_id", how="anti")
    if not chosen:
        return empty_rows()
    out = pl.concat(chosen)
    return mk_rows(out.with_columns(pl.col("amt2").alias("amount")), "ar_exact")


WHT_ENTITIES = ["MX01", "BR01", "MA01"]
WHT_RATES = [0.0125, 0.015, 0.10]


def _evidence_cands(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    """All (credit line, open ARI of same entity+currency issued before) pairs with customer evidence (doc / doc6 / name / iban)."""
    lines = pending.filter(pl.col("is_credit") & pl.col("orig_amt").is_null()).join(_acc_entity(ctx), on="account_id").select(
        "bank_txn_id", "booking_date", "currency", pl.col("acc_entity").alias("entity"), pl.col("abs_amt").round(2).alias("amt"),
        norm_expr("alltext").alias("tnorm"), pl.col("counterparty_account").alias("cp_iban"), "alltext")
    items = ctx.items.filter((pl.col("item_type") == "ARI") & (pl.col("open_amount") > 0)).select(
        "item_id", "entity", "party_id", "currency", "docnorm", "issue_date", "total", "open_amount", "payment_terms",
        norm_expr("party_name").str.slice(0, 10).alias("k0")).join(_cancelled(ctx), on="item_id", how="anti")
    return lines, items


def adjusted_amount(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    """One invoice, cash differs for a computable reason (POLICIES §3): discount (in/after terms), withholding, doc-cited partial
    or overpayment."""
    lines, items = _evidence_cands(ctx, pending)
    if lines.height == 0:
        return empty_rows()
    terms = ctx.terms.select(pl.col("code").alias("payment_terms"),
                             pl.col("early_payment_discount_pct").cast(pl.Float64, strict=False).alias("dpct"),
                             pl.col("discount_days").cast(pl.Int32, strict=False).alias("ddays"))
    it = items.join(terms, on="payment_terms", how="left")
    # expected cash variants per item -> (item, exp_amt, kind)
    disc = it.filter(pl.col("dpct") > 0).select("item_id", (pl.col("total") * (1 - pl.col("dpct") / 100)).round(2).alias("exp"),
                                                pl.lit("DISC").alias("kind"))
    wht = pl.concat([it.filter(pl.col("entity").is_in(WHT_ENTITIES)).select(
        "item_id", (pl.col("total") * (1 - r)).round(2).alias("exp"), pl.lit("WHT").alias("kind")) for r in WHT_RATES])
    exp = pl.concat([disc, wht]).join(it, on="item_id")
    c1 = lines.join(exp, left_on=["entity", "currency", "amt"], right_on=["entity", "currency", "exp"]).filter(
        pl.col("issue_date") <= pl.col("booking_date"))
    # doc-cited single invoice with a different amount (partial / overpayment): doc-like tokens of the text == our doc number
    toks = lines.with_columns(pl.col("alltext").str.extract_all(r"[A-Z]{0,4}\d[\d\-/]{5,}").alias("toks")).explode("toks").filter(
        pl.col("toks").is_not_null()).with_columns(norm_expr("toks").alias("tok"))
    c2 = toks.join(it, left_on=["entity", "currency", "tok"], right_on=["entity", "currency", "docnorm"]).filter(
        pl.col("issue_date") <= pl.col("booking_date")).with_columns(
        pl.when(pl.col("amt") < pl.col("open_amount") - 0.01).then(pl.lit("PART"))
        .when(pl.col("amt") > pl.col("open_amount") + 0.01).then(pl.lit("OVER")).otherwise(pl.lit("EXACT")).alias("kind"),
        (pl.col("amt") / pl.col("open_amount")).alias("ratio"))
    # a line citing several of our invoices is a 1:N case, not a partial
    ndocs = c2.group_by("bank_txn_id").agg(pl.col("item_id").n_unique().alias("nd"))
    # an overpayment is a deliberate top-up (1-2-5 nice excess, c_evidence._nice_excess); any other excess is more often a
    # multi-invoice payment that cites one invoice, so this resolver leaves it to the multi-invoice resolvers
    from .c_evidence import _nice_excess
    c2 = c2.join(ndocs, on="bank_txn_id").filter((pl.col("nd") == 1) & (
        ((pl.col("kind") == "PART") & (pl.col("ratio") > 0.02))
        | ((pl.col("kind") == "OVER") & (pl.col("ratio") <= 1.3)
           & _nice_excess((pl.col("amt") - pl.col("open_amount")).round(2), pl.col("open_amount")))))
    # ... nor is a line that also cites (by last 6 digits) another open invoice of the same customer
    others = c2.select("bank_txn_id", "tnorm", "party_id", pl.col("item_id").alias("mine")).join(
        it.select("item_id", "party_id", pl.col("docnorm").str.replace_all(r"\D", "").str.slice(-6).alias("d6")), on="party_id").filter(
        (pl.col("item_id") != pl.col("mine")) & (pl.col("d6").str.len_chars() == 6) & pl.col("tnorm").str.contains(pl.col("d6"), literal=True))
    c2 = c2.join(others.select("bank_txn_id").unique(), on="bank_txn_id", how="anti")

    def has(df, k):
        return (pl.col(k).str.len_chars() >= 5) & pl.col("tnorm").str.contains(pl.col(k), literal=True)

    names = _names(ctx)
    c1 = c1.join(names, on="party_id", how="left").with_columns(
        (has(c1, "k0").fill_null(False) | has(c1, "k1").fill_null(False) | has(c1, "k2").fill_null(False)
         | (pl.col("cp_iban").is_not_null() & (pl.col("cp_iban") == pl.col("iban"))).fill_null(False)
         | pl.col("tnorm").str.contains(pl.col("docnorm"), literal=True)).alias("ev")).filter(pl.col("ev"))
    c2 = c2.join(names, on="party_id", how="left").filter(
        has(c2, "k0").fill_null(False) | has(c2, "k1").fill_null(False) | has(c2, "k2").fill_null(False)
        | (pl.col("cp_iban").is_not_null() & (pl.col("cp_iban") == pl.col("iban"))).fill_null(False))
    cols = ["bank_txn_id", "item_id", "amt", "open_amount", "kind", "issue_date", "booking_date", "ddays"]
    cand = pl.concat([c1.with_columns(pl.lit(1).alias("prio")).select(cols + ["prio"]),
                      c2.with_columns(pl.lit(2).alias("prio")).select(cols + ["prio"])])
    late = pl.col("booking_date").str.to_date() > pl.col("issue_date").str.to_date().dt.offset_by(
        pl.concat_str([(pl.col("ddays").fill_null(0) + 3).cast(pl.Utf8), pl.lit("d")]))
    cand = cand.with_columns(
        pl.when(pl.col("kind") == "DISC").then(pl.when(late).then(pl.lit("PARTIAL")).otherwise(pl.lit("SHORT_DISCOUNT")))
        .when(pl.col("kind") == "WHT").then(pl.lit("SHORT_WITHHOLDING"))
        .when(pl.col("kind") == "PART").then(pl.lit("PARTIAL"))
        .when(pl.col("kind") == "OVER").then(pl.lit("OVERPAYMENT"))
        .otherwise(pl.lit("")).alias("exception"))
    cand = cand.with_columns(pl.when(pl.col("exception") == "PARTIAL").then(pl.col("amt")).otherwise(pl.col("open_amount")).alias("amount"))
    cand = cand.sort(["prio", "issue_date", "booking_date"])
    out = cand.unique("bank_txn_id", keep="first", maintain_order=True).unique("item_id", keep="first", maintain_order=True)
    return mk_rows(out, "ar_adjusted", pl.col("exception"))


RESOLVERS = [exact_amount, adjusted_amount]
