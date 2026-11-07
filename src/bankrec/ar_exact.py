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
        norm_expr("party_name").str.slice(0, 10).alias("k0"))
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


RESOLVERS = [exact_amount]
