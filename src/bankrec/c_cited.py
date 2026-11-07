"""Persona C: a receipt whose text cites several of our invoice numbers, all of one customer.

Cash is applied in the order the payer lists its invoices (POLICIES §3: an invoice that keeps an open balance after the line
is PARTIAL; one paid in full has no exception):
  - the cited open amounts add up to the cash            -> every cited invoice fully cleared, no exception;
  - the cash falls short of the cited total by d          -> every cited invoice in full except the LAST one cited, which
                                                             takes what is left (open - d, PARTIAL; the rest stays open);
  - anything else (one invoice cited, several customers, cash above the cited total, the last invoice cannot absorb d,
    an invoice number that matches two open items)        -> abstain.

    BANKREC_ORDER=...,c_fx,c_cited,ar_due_subset,c_duplicates,c_overpayment
"""
from __future__ import annotations

import polars as pl

from .c_evidence import AMT_TOL, DOC_MIN_LEN, _lines, _open_invoices
from .deterministic import empty_rows, mk_rows


def cited_invoices(ctx, pending: pl.DataFrame) -> pl.DataFrame:
    lines = _lines(ctx, pending, credit=True)
    items = _open_invoices(ctx).filter(pl.col("docnorm").str.len_chars() >= DOC_MIN_LEN)
    if not lines.height or not items.height:
        return empty_rows()
    vocab = items["docnorm"].unique().to_list()
    m = (lines.select("bank_txn_id", "entity", "currency", "cash", "booking_date",
                      pl.col("tnorm").str.extract_many(vocab).alias("doc"))
         .filter(pl.col("doc").list.len() >= 2)
         .with_columns(pl.int_ranges(0, pl.col("doc").list.len()).alias("pos"))
         .explode(["doc", "pos"]))
    if not m.height:
        return empty_rows()
    it = items.rename({"docnorm": "doc"})
    ambiguous = it.group_by("entity", "currency", "doc").len().filter(pl.col("len") > 1).select("entity", "currency", "doc")
    c = (m.join(it, on=["entity", "currency", "doc"])
         .join(ambiguous, on=["entity", "currency", "doc"], how="anti")
         .filter(pl.col("issue_date") <= pl.col("booking_date"))
         .sort("pos").unique(["bank_txn_id", "item_id"], keep="first", maintain_order=True))
    out = []
    for (tid,), g in c.group_by("bank_txn_id"):
        if g.height < 2 or g["party_id"].n_unique() != 1:
            continue
        g = g.sort("pos")
        cash = round(float(g["cash"][0]) * 100)
        opens = [round(float(x) * 100) for x in g["open"].to_list()]
        d = sum(opens) - cash
        ids = g["item_id"].to_list()
        if abs(d) <= 1:
            out += [(tid, i, o / 100, "") for i, o in zip(ids, opens)]
        elif d > 1 and opens[-1] > d:
            out += [(tid, i, o / 100, "") for i, o in zip(ids[:-1], opens[:-1])]
            out.append((tid, ids[-1], (opens[-1] - d) / 100, "PARTIAL"))
    if not out:
        return empty_rows()
    df = pl.DataFrame(out, schema={"bank_txn_id": pl.Utf8, "item_id": pl.Utf8, "amount": pl.Float64, "exc": pl.Utf8}, orient="row")
    return mk_rows(df, "c_cited", pl.col("exc"))


RESOLVERS = [cited_invoices]
