"""Persona C: a receipt from one identified customer that pays several invoices without citing them.

Exact subset sum (cash-application "combination matching"): among the customer's open invoices issued in the last
MAX_AGE_DAYS before the payment, count the combinations of ANY size whose open amounts add up to the cash to the cent.
Exactly one combination -> those invoices, each fully cleared (no exception). None or several -> abstain.
Solved by dynamic programming over the amount in cents (pseudo-polynomial: invoices x cents), counting solutions up to 2.

    BANKREC_ORDER=...,c_fx,c_cited,c_bulk,ar_due_subset,c_duplicates,c_overpayment
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .c_evidence import _line_parties, _lines, _open_invoices
from .data import norm_expr
from .deterministic import empty_rows, mk_rows

MAX_AGE_DAYS = 120         # longest payment term in payment_terms.csv (90 days) + 30 days of grace
MAX_POOL = 40              # complexity bound: candidate invoices per receipt
MAX_CENTS = 10_000_000     # complexity bound: receipts up to 100,000 in currency units
MIN_INVOICES = 2           # a single invoice is the exact-amount matchers' job


def _unique_subset(vals: list[int], target: int):
    if target <= 0 or sum(vals) < target:
        return None
    tabs = [np.zeros(target + 1, np.int8)]
    tabs[0][0] = 1
    for v in vals:
        prev = tabs[-1]
        c = prev.copy()
        if v <= target:
            np.minimum(2, c[v:] + prev[:target + 1 - v], out=c[v:])
        tabs.append(c)
    if tabs[-1][target] != 1:
        return None
    sol, t = [], target
    for i in range(len(vals), 0, -1):
        if tabs[i - 1][t] == 1:
            continue
        sol.append(i - 1)
        t -= vals[i - 1]
    return sol if t == 0 else None


def bulk_subsets(ctx, pending: pl.DataFrame) -> pl.DataFrame:
    lines = _lines(ctx, pending, credit=True)
    items = _open_invoices(ctx)
    if not lines.height or not items.height:
        return empty_rows()
    lines = lines.filter(pl.col("cash") * 100 <= MAX_CENTS)
    # payer: the full counterparty name equal to one customer's name / trade name, else the IBAN, else a unique name key
    cu = ctx.customers.select(pl.col("customer_id").alias("party_id"), norm_expr("name").alias("n1"), norm_expr("trade_name").alias("n2"))
    names = pl.concat([cu.select("party_id", pl.col("n1").alias("cpn")), cu.select("party_id", pl.col("n2").alias("cpn"))]).filter(
        pl.col("cpn").str.len_chars() >= 5).unique()
    names = names.join(names.group_by("cpn").agg(pl.col("party_id").n_unique().alias("_k")).filter(pl.col("_k") == 1), on="cpn", how="semi")
    cp = pending.select("bank_txn_id", norm_expr("counterparty_name").alias("cpn")).join(lines.select("bank_txn_id"), on="bank_txn_id")
    by_name = cp.join(names, on="cpn").select("bank_txn_id", "party_id")
    lp = _line_parties(ctx, lines.join(by_name, on="bank_txn_id", how="anti"), "AR")
    lp = lp.join(lp.group_by("bank_txn_id").len().filter(pl.col("len") == 1), on="bank_txn_id", how="semi").select("bank_txn_id", "party_id")
    lp = pl.concat([by_name, lp]).unique("bank_txn_id")
    if not lp.height:
        return empty_rows()
    c = (lines.select("bank_txn_id", "entity", "currency", "cash", "booking_date").join(lp, on="bank_txn_id")
         .join(items.select("item_id", "entity", "currency", "party_id", "open", "issue_date"), on=["entity", "currency", "party_id"])
         .with_columns((pl.col("booking_date").str.slice(0, 10).str.to_date()
                        - pl.col("issue_date").str.slice(0, 10).str.to_date()).dt.total_days().alias("age"))
         .filter((pl.col("age") >= 0) & (pl.col("age") <= MAX_AGE_DAYS) & (pl.col("open") > 0)))
    # abstain when the customer could be netting a credit note (an invoices-only sum would then be a coincidence)
    cn = ctx.items.filter((pl.col("item_type") == "ARC") & (pl.col("open_amount") > 0.005)).select("entity", "currency", "party_id").unique()
    c = c.join(cn, on=["entity", "currency", "party_id"], how="anti")
    # abstain when the text cites one of our invoice numbers (the cited-invoice resolver's case: partial shares)
    cited = lines.select("bank_txn_id", "tnorm").join(c.select("bank_txn_id").unique(), on="bank_txn_id")
    docs = items.filter(pl.col("docnorm").str.len_chars() >= 6)["docnorm"].unique().to_list()
    if docs and cited.height:
        hit = cited.with_columns(pl.col("tnorm").str.contains_any(docs).alias("_c")).filter(pl.col("_c")).select("bank_txn_id")
        c = c.join(hit, on="bank_txn_id", how="anti")
    n = c.group_by("bank_txn_id").len()
    c = c.join(n.filter((pl.col("len") >= MIN_INVOICES) & (pl.col("len") <= MAX_POOL)), on="bank_txn_id", how="semi")
    out, used = [], set()
    for (tid,), g in c.sort("booking_date").group_by("bank_txn_id", maintain_order=True):
        g = g.sort("item_id")
        g = g.filter(~pl.col("item_id").is_in(list(used)))
        ids = g["item_id"].to_list()
        vals = [int(round(float(x) * 100)) for x in g["open"].to_list()]
        sol = _unique_subset(vals, int(round(float(g["cash"][0]) * 100)))
        if sol is None or len(sol) < MIN_INVOICES:
            continue
        for i in sol:
            out.append((tid, ids[i], vals[i] / 100))
            used.add(ids[i])
    if not out:
        return empty_rows()
    df = pl.DataFrame(out, schema={"bank_txn_id": pl.Utf8, "item_id": pl.Utf8, "amount": pl.Float64}, orient="row")
    return mk_rows(df, "c_bulk", "")


RESOLVERS = [bulk_subsets]
