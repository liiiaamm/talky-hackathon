"""Persona C: receipts whose text cites invoice numbers of one customer (cash-application practice: references first,
then a waterfall).

1. Citations: our invoice numbers found in the text, plus cut-off references (bank fields have character limits): a text
   word of at least TRUNC_MIN_LEN characters that is the beginning of exactly one open invoice number of that customer.
2. Waterfall: a customer often pays one group of invoices with several transfers, each repeating the whole list. Its
   cited receipts are taken in booking order and each one's cash is applied to the cited invoices in the order the payer
   lists them, against what is still open (shared across those receipts). An invoice the cash fully covers is cleared
   (no exception); the one where the cash runs out keeps a balance (PARTIAL, POLICIES §3) and the next receipt continues it.
3. Cash left after every cited invoice is covered: if it equals exactly ONE other open invoice of the customer, that one
   too; otherwise the receipt is left to later resolvers.
Abstain on a receipt citing several customers, an invoice number matching two open items, or fewer than two rows.

    BANKREC_ORDER=...,c_fx,c_cited,ar_due_subset,c_duplicates,c_overpayment
"""
from __future__ import annotations

import re

import polars as pl

from .c_evidence import DOC_MIN_LEN, _lines, _open_invoices
from .deterministic import empty_rows, mk_rows
from .remittance import PARTIAL_SHARE_STEP   # unplanned partial payment = a whole-percent share of one invoice

TRUNC_MIN_LEN = 8          # a cut-off reference must keep at least this many characters of the invoice number


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def cited_invoices(ctx, pending: pl.DataFrame) -> pl.DataFrame:
    lines = _lines(ctx, pending, credit=True)
    items = _open_invoices(ctx).filter(pl.col("docnorm").str.len_chars() >= DOC_MIN_LEN)
    if not lines.height or not items.height:
        return empty_rows()
    vocab = items["docnorm"].unique().to_list()
    m = (lines.select("bank_txn_id", "entity", "currency", "cash", "booking_date",
                      pl.col("tnorm").str.extract_many(vocab).alias("doc"))
         .filter(pl.col("doc").list.len() >= 1)
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
    if not c.height:
        return empty_rows()
    one = c.group_by("bank_txn_id").agg(pl.col("party_id").n_unique().alias("_np")).filter(pl.col("_np") == 1)
    c = c.join(one, on="bank_txn_id", how="semi")
    pool = (c.select("bank_txn_id", "entity", "currency", "party_id").unique()
            .join(it.select("item_id", "entity", "currency", "party_id", "doc", "open", "issue_date"),
                  on=["entity", "currency", "party_id"]))
    pools = {tid: g for (tid,), g in pool.group_by("bank_txn_id")}
    raw = dict(zip(pending["bank_txn_id"].to_list(), pending["alltext"].to_list()))
    # per receipt: ordered cited list (exact, then cut-off references in text order)
    rec = []
    for (tid,), g in c.group_by("bank_txn_id"):
        g = g.sort("pos")
        bd = g["booking_date"][0]
        p = pools[tid].filter(pl.col("issue_date") <= bd)
        pdocs = list(zip(p["item_id"].to_list(), p["doc"].to_list(), [round(float(x) * 100) for x in p["open"].to_list()]))
        sel = g["item_id"].to_list()
        have = set(sel)
        for tok in (_norm(w) for w in re.split(r"\s+", raw.get(tid) or "")):
            if len(tok) < TRUNC_MIN_LEN:
                continue
            hits = [i for i, d, _ in pdocs if d != tok and d.startswith(tok)]
            if len(hits) == 1 and hits[0] not in have:
                sel.append(hits[0]); have.add(hits[0])
        rec.append((g["party_id"][0], bd, tid, round(float(g["cash"][0]) * 100), sel, pdocs))
    steps = round(1 / PARTIAL_SHARE_STEP)
    out = []
    for party, bd, tid, cash, sel, pdocs in rec:
        opens = {i: o for i, _, o in pdocs}
        sel = [i for i in sel if opens.get(i, 0) > 0]
        total = sum(opens[i] for i in sel)
        d = total - cash
        rows = None
        if abs(d) <= 1:                                   # every cited invoice in full
            rows = [(i, opens[i], False) for i in sel]
        elif d > 1:                                       # one cited invoice partly paid: a whole-percent share of it
            fits = [i for i in sel if opens[i] > d and
                    abs((opens[i] - d) - round(opens[i] * round((opens[i] - d) / opens[i] * steps) / steps)) <= 1]
            if len(fits) == 1:
                rows = [(i, opens[i] - d if i == fits[0] else opens[i], i == fits[0]) for i in sel]
        if not rows or len(rows) < 2:
            continue
        out += [(tid, i, take / 100, "PARTIAL" if partial else "") for i, take, partial in rows]
    if not out:
        return empty_rows()
    df = pl.DataFrame(out, schema={"bank_txn_id": pl.Utf8, "item_id": pl.Utf8, "amount": pl.Float64, "exc": pl.Utf8}, orient="row")
    return mk_rows(df, "c_cited", pl.col("exc"))


RESOLVERS = [cited_invoices]
