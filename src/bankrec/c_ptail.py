"""EXPERIMENT: unclaimed AR receipts = k >= 1 invoices paid in full + exactly ONE invoice paid at an instalment fraction.

TRAIN audit (INT7 05 section 1c): about 650 of the ~1,200 unclaimed ar_n_to_m lines per half-year have exactly one PARTIAL gold
row, always at 0.4 / 0.5 / 0.6 of that invoice. No other resolver searches this shape.

Same payer evidence, pool and ledger view as ar_due_subset (IBAN / name automata / cited documents; open ARI of that payer,
same currency, issued <= booking, due in [booking - 120, booking + 10]). Integer cents throughout.
For each still-pending credit line:
  - abstain if the pool has more than MAX_POOL items, or if any exact full subset explains the cash (that is a full-subset case);
  - count every (j, r, S): invoice j at fraction r of its open amount (to the cent, +-1 cent) plus an exact subset S (>= 1 item)
    of the other pool items; claim only when exactly one solution exists over all candidate payers; otherwise abstain.
Rows: the full items at their open amount, j at the partial amount with exception PARTIAL.
Placed LAST (after c_overpayment) so it cannot take lines from any validated resolver.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import re

import polars as pl

from .ar_due_subset import _automata, _cancelled, _unique_subset, TOP_PARTIES
from .data import Ctx, norm_expr
from .deterministic import _acc_entity, empty_rows, mk_rows

FRACTIONS = ((2, 5), (1, 2), (3, 5))   # 0.4, 0.5, 0.6 (observed on TRAIN gold, both windows)
BACK, AHEAD = 120, 10
MAX_POOL = 24


def _solutions(pool: list, amt: int):
    """All-or-nothing count capped at 2 of (j, r, subset) solutions; returns (n, solution)."""
    n_full, _ = _unique_subset([r[2] for r in pool], amt)
    if n_full:
        return 2, None                      # an exact full subset exists: not this shape
    found, sol = 0, None
    for j, rj in enumerate(pool):
        rest = pool[:j] + pool[j + 1:]
        for a, b in FRACTIONS:
            part = (rj[2] * a + b // 2) // b
            for d in (-1, 0, 1):
                tgt = amt - (part + d)
                if tgt <= 0:
                    continue
                n, s = _unique_subset([r[2] for r in rest], tgt)
                if n:
                    found += n
                    if found >= 2:
                        return 2, None
                    sol = ([rest[i] for i in s], rj, part + d)
    return found, sol


def partial_tail(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    lines = pending.filter(pl.col("is_credit") & pl.col("orig_amt").is_null()).join(_acc_entity(ctx), on="account_id").select(
        "bank_txn_id", pl.col("booking_date").cast(pl.Utf8).str.slice(0, 10).str.to_date().alias("bd"), "currency",
        pl.col("acc_entity").alias("entity"), (pl.col("abs_amt") * 100).round(0).cast(pl.Int64).alias("c"),
        norm_expr("alltext").alias("t"), pl.col("counterparty_account").alias("iban"), "alltext").sort("bd", "bank_txn_id")
    if lines.height == 0:
        return empty_rows()
    items = ctx.items.filter((pl.col("item_type") == "ARI") & (pl.col("open_amount") > 0.005)).join(
        _cancelled(ctx), on="item_id", how="anti").select(
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
    for iid, ent, p, ccy, doc, iss, due, c in items.iter_rows():
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
        lo, hi = bd - timedelta(days=BACK), bd + timedelta(days=AHEAD)
        total, sol = 0, None
        for p in sorted(prio, key=lambda q: (-prio[q], q))[:TOP_PARTIES]:
            pool = [r for r in by_party.get((ent, p), ()) if r[0] not in used and r[1] == ccy and r[4] <= bd and lo <= r[3] <= hi]
            if len(pool) < 2:
                continue
            if len(pool) > MAX_POOL:
                total = 2
                break
            n, s = _solutions(pool, amt)
            total += n
            if total >= 2:
                break
            if n:
                sol = s
        if total != 1 or sol is None:
            continue
        full, j, part = sol
        if not full:
            continue
        for r in full:
            used.add(r[0])
            out.append((txn, r[0], r[2] / 100, False))
        used.add(j[0])
        out.append((txn, j[0], part / 100, True))
    if not out:
        return empty_rows()
    df = pl.DataFrame(out, schema={"bank_txn_id": pl.Utf8, "item_id": pl.Utf8, "amount": pl.Float64, "_p": pl.Boolean}, orient="row")
    return mk_rows(df, "c_ptail", pl.when(pl.col("_p")).then(pl.lit("PARTIAL")).otherwise(pl.lit("")))


RESOLVERS = [partial_tail]
