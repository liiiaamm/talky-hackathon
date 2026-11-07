"""EXPERIMENT: unclaimed AR receipt = a CONTIGUOUS run (>= 2) of the payer's open invoices in due-date order whose sum equals the
cash to the cent, and that run is the only such run. TRAIN H2 screen (gold payer): 103 / 104 exactly correct. Runs last."""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import re

import polars as pl

from .ar_due_subset import _automata, _cancelled, TOP_PARTIES
from .data import Ctx, norm_expr
from .deterministic import _acc_entity, empty_rows, mk_rows

BACK = 365


def contig(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    lines = pending.filter(pl.col("is_credit") & pl.col("orig_amt").is_null()).join(_acc_entity(ctx), on="account_id").select(
        "bank_txn_id", pl.col("booking_date").cast(pl.Utf8).str.slice(0, 10).str.to_date().alias("bd"), "currency",
        pl.col("acc_entity").alias("entity"), (pl.col("abs_amt") * 100).round(0).cast(pl.Int64).alias("c"),
        norm_expr("alltext").alias("t"), pl.col("counterparty_account").alias("iban"), "alltext").sort("bd", "bank_txn_id")
    if lines.height == 0:
        return empty_rows()
    items = ctx.items.filter((pl.col("item_type") == "ARI") & (pl.col("open_amount") > 0.005)).join(_cancelled(ctx), on="item_id", how="anti").select(
        "item_id", "entity", "party_id", "currency", "docnorm",
        pl.col("issue_date").cast(pl.Utf8).str.slice(0, 10).str.to_date().alias("iss"),
        pl.col("due_date").cast(pl.Utf8).str.slice(0, 10).str.to_date().alias("due"),
        (pl.col("open_amount") * 100).round(0).cast(pl.Int64).alias("c"))
    autos = _automata(ctx, ctx.items.filter(pl.col("item_type").is_in(["ARI", "ARC"])))
    iban2p = defaultdict(set)
    for ent, p, ib in ctx.customers.select("entity", "customer_id", "iban").iter_rows():
        if ib:
            iban2p[(ent, ib)].add(p)
    by_party, doc2p = defaultdict(list), {}
    for iid, ent, p, ccy, doc, iss, due, c in items.sort("due", "item_id").iter_rows():
        if c > 0 and iss is not None and due is not None:
            by_party[(ent, p)].append((iid, ccy, c, due, iss))
        if doc and len(doc) >= 6:
            doc2p[(ent, doc)] = p
    used, out = set(), []
    for txn, bd, ccy, ent, amt, t, iban, raw in lines.iter_rows():
        prio = defaultdict(int)
        if iban:
            for p in iban2p.get((ent, iban), ()):
                prio[p] = max(prio[p], 50)
        A = autos.get(ent)
        if A is not None:
            for _end, (L, ps) in A.iter(t):
                for p in ps:
                    prio[p] = max(prio[p], L)
        for tok in {re.sub(r"[^A-Z0-9]", "", x) for x in re.findall(r"[A-Z0-9][A-Z0-9\-/]{5,}", raw or "")}:
            p = doc2p.get((ent, tok))
            if p:
                prio[p] = 100
        if not prio:
            continue
        sols = []
        for p in sorted(prio, key=lambda q: (-prio[q], q))[:TOP_PARTIES]:
            pool = [r for r in by_party.get((ent, p), ()) if r[0] not in used and r[1] == ccy and r[4] <= bd and r[3] >= bd - timedelta(days=BACK)]
            P = [0]
            for r in pool:
                P.append(P[-1] + r[2])
            pos = defaultdict(list)
            for j, x in enumerate(P):
                pos[x].append(j)
            for j in range(len(P)):
                for d in (-1, 0, 1):
                    for i in pos.get(P[j] - amt + d, ()):
                        if i < j - 1:
                            sols.append(pool[i:j])
            if len(sols) > 1:
                break
        if len(sols) != 1:
            continue
        for r in sols[0]:
            used.add(r[0])
            out.append((txn, r[0], r[2] / 100))
    if not out:
        return empty_rows()
    return mk_rows(pl.DataFrame(out, schema={"bank_txn_id": pl.Utf8, "item_id": pl.Utf8, "amount": pl.Float64}, orient="row"), "c_contig")


RESOLVERS = [contig]
