"""Unclaimed AR receipts that pay several RECENTLY DUE items of one payer (ar_ref_noise, 1:N text, 1:N with credit notes).

Golden-free rule found in the 2025-H1 exploration: customers pay what has just fallen due. The golden items of a receipt are
due at most ~15 days after the booking date (p99 +7), while the open items nobody pays are much older (median -54 days).
Restricting the payer's pool to that window shrinks it from ~40 to ~14 items and the exact subset sum usually becomes unique.

For each still-pending credit line (same currency) with an identified payer (IBAN / name keys / cited docs, as ar_multi):
  pool(D) = payer's open ARI with due_date in [booking - D, booking + 10] and ARC issued in that window, issued <= booking,
            not already claimed by an earlier resolver (or by an earlier line here)
  for D in 30, 45, 60, 90, 120: exactly one exact (to the cent) subset of >= 2 items -> claim it; 2+ subsets -> abstain
  (larger windows only add solutions). Several candidate payers solving the same line -> abstain.
Runs after the text/remittance/AR modules, right before the late category classifier.
Replay on the 13:00 snapshots (cat_rest/fallback lines only): H1 85.14 -> 86.58, out-of-sample H2 84.83 -> 86.39.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import polars as pl

from .ar_exact import _cancelled
from .ar_multi import _automata
from .data import Ctx, norm_expr
from .deterministic import _acc_entity, empty_rows, mk_rows

WINDOWS = (30, 45, 60, 90, 120)
AHEAD = 10            # days after booking an item may still be due
MAX_POOL = 32         # meet in the middle: 2 x 2^16 subsets at most
TOP_PARTIES = 3
_MASKS: dict[int, np.ndarray] = {}


def _masks(k: int) -> np.ndarray:
    if k not in _MASKS:
        _MASKS[k] = ((np.arange(1 << k)[:, None] >> np.arange(k)) & 1).astype(np.int64)
    return _MASKS[k]


def _unique_subset(vals: list[int], target: int):
    """(n_solutions capped at 2, index list of the unique solution or None). Meet in the middle, vectorised."""
    h = len(vals) // 2
    a, b = np.array(vals[:h], np.int64), np.array(vals[h:], np.int64)
    ma, mb = _masks(len(a)), _masks(len(b))
    sa, sb = ma @ a, mb @ b
    order = np.argsort(sa, kind="stable")
    ss = sa[order]
    need = target - sb
    lo, hi = np.searchsorted(ss, need, "left"), np.searchsorted(ss, need, "right")
    cnt = hi - lo
    total = int(cnt.sum())
    if total != 1:
        return min(total, 2), None
    j = int(np.nonzero(cnt)[0][0])
    i = int(order[lo[j]])
    return 1, [int(x) for x in np.nonzero(ma[i])[0]] + [h + int(x) for x in np.nonzero(mb[j])[0]]


def _cascade(its: list, bdate: date, amt: int):
    """its: (item_id, typ, value_cents_signed, due_or_issue: date). Returns the claimed records or None."""
    for D in WINDOWS:
        lo, hi = bdate - timedelta(days=D), bdate + timedelta(days=AHEAD)
        w = [r for r in its if lo <= r[3] <= hi]
        if len(w) < 2 or len(w) > MAX_POOL:
            continue
        n, sol = _unique_subset([r[2] for r in w], amt)
        if n == 1 and len(sol) >= 2:
            return [w[i] for i in sol]
        if n >= 2:
            return None
    return None


def _claimed_cents(ctx: Ctx) -> dict:
    """item_id -> cents already allocated by earlier resolvers (only needed when the shared ledger is off)."""
    if os.environ.get("BANKREC_LEDGER", "1") == "1":
        return {}
    prev = [r for r in ctx.__dict__.get("_claimed", []) if r.height]
    if not prev:
        return {}
    df = pl.concat([r.select("item_id", "amount", "exception") for r in prev], how="vertical_relaxed").filter(
        pl.col("item_id").is_not_null() & (pl.col("item_id") != "") & ~pl.col("exception").fill_null("").is_in(["RETURNED", "DUPLICATE_PAYMENT"]))
    df = df.group_by("item_id").agg((pl.col("amount").abs().sum() * 100).round(0).cast(pl.Int64).alias("c"))
    return dict(df.iter_rows())


def due_subset(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    lines = pending.filter(pl.col("is_credit") & pl.col("orig_amt").is_null()).join(_acc_entity(ctx), on="account_id").select(
        "bank_txn_id", pl.col("booking_date").cast(pl.Utf8).str.slice(0, 10).str.to_date().alias("bd"), "currency",
        pl.col("acc_entity").alias("entity"), (pl.col("abs_amt") * 100).round(0).cast(pl.Int64).alias("c"),
        norm_expr("alltext").alias("t"), pl.col("counterparty_account").alias("iban"), "alltext").sort("bd", "bank_txn_id")
    if lines.height == 0:
        return empty_rows()
    items = ctx.items.filter(pl.col("item_type").is_in(["ARI", "ARC"]) & (pl.col("open_amount") > 0.005)).join(
        _cancelled(ctx), on="item_id", how="anti").select(
        "item_id", "item_type", "entity", "party_id", "party_name", "currency", "docnorm",
        pl.col("issue_date").cast(pl.Utf8).str.slice(0, 10).str.to_date().alias("iss"),
        pl.col("due_date").cast(pl.Utf8).str.slice(0, 10).str.to_date().alias("due"),
        (pl.col("open_amount") * 100).round(0).cast(pl.Int64).alias("c"))
    claimed = _claimed_cents(ctx)
    autos = _automata(ctx, items)
    iban2p = defaultdict(set)
    for ent, p, ib in ctx.customers.select("entity", "customer_id", "iban").iter_rows():
        if ib:
            iban2p[(ent, ib)].add(p)
    by_party = defaultdict(list)
    doc2p = {}
    for iid, typ, ent, p, _pn, ccy, doc, iss, due, c in items.iter_rows():
        c -= claimed.get(iid, 0)
        if c <= 0 or iss is None:
            continue
        key = due if (typ == "ARI" and due is not None) else iss
        by_party[(ent, p)].append((iid, typ, -c if typ == "ARC" else c, key, iss, ccy, abs(c)))
        if doc and len(doc) >= 6:
            doc2p[(ent, doc)] = p
    used = set()
    out = []
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
            its = [r for r in by_party.get((ent, p), ()) if r[0] not in used and r[5] == ccy and r[4] <= bd]
            if len(its) < 2:
                continue
            s = _cascade(its, bd, amt)
            if s:
                sols.append(s)
        if len(sols) != 1:
            continue
        for r in sols[0]:
            used.add(r[0])
            out.append((txn, r[0], r[6] / 100))
    if not out:
        return empty_rows()
    df = pl.DataFrame(out, schema={"bank_txn_id": pl.Utf8, "item_id": pl.Utf8, "amount": pl.Float64}, orient="row")
    return mk_rows(df, "ar_due_subset")


RESOLVERS = [due_subset]
