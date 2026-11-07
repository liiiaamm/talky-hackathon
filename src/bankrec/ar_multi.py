"""AR receipts that pay SEVERAL open items of the payer (ar_1toN_text, multi-invoice ar_ref_noise, 1:N with credit notes, n_to_m…).

Lines are processed in booking order with a running set of used items (a light ledger). For each line:
  parties  = customers whose IBAN == counterparty IBAN, whose name keys appear in the text (Aho-Corasick), or whose doc numbers are cited
  then, per party, the first rule that gives an exact sum (to the cent) wins:
    1. cited items (full doc number or last 6 digits in the text), optionally plus credit notes of the party
    2. FIFO prefix of the party's open invoices (oldest first), optionally minus one credit note
    3. a unique pair / triple of open invoices (optionally with one credit note)
"""
from __future__ import annotations

import itertools
import re
from collections import defaultdict

import ahocorasick
import polars as pl

from .ar_exact import _cancelled
from .data import Ctx, norm_expr
from .deterministic import _acc_entity, empty_rows, mk_rows

MAX_FIFO = 80
MAX_PAIR = 300
MAX_TRIPLE = 60


def _automata(ctx: Ctx, items: pl.DataFrame) -> dict:
    """entity -> Aho-Corasick automaton of normalised name prefixes (10/16/24 chars) -> set(party_id). Longer hit = better."""
    src = pl.concat([
        ctx.customers.select("entity", pl.col("customer_id").alias("party_id"), norm_expr("name").alias("n")),
        ctx.customers.select("entity", pl.col("customer_id").alias("party_id"), norm_expr("trade_name").alias("n")),
        items.select("entity", "party_id", norm_expr("party_name").alias("n")),
    ])
    keys = pl.concat([src.select("entity", "party_id", pl.col("n").str.slice(0, L).alias("k")) for L in (10, 16, 24)]).filter(
        pl.col("k").str.len_chars() >= 6).unique()
    autos = {}
    for (ent,), df in keys.group_by(["entity"]):
        A = ahocorasick.Automaton()
        d = defaultdict(set)
        for k, p in zip(df["k"], df["party_id"]):
            d[k].add(p)
        for k, ps in d.items():
            A.add_word(k, (len(k), ps))
        A.make_automaton()
        autos[ent] = A
    return autos


def multi_invoice(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    lines = pending.filter(pl.col("is_credit") & pl.col("orig_amt").is_null()).join(_acc_entity(ctx), on="account_id").select(
        "bank_txn_id", "booking_date", "currency", pl.col("acc_entity").alias("entity"), (pl.col("abs_amt") * 100).round(0).cast(pl.Int64).alias("c"),
        norm_expr("alltext").alias("t"), pl.col("counterparty_account").alias("iban"), "alltext").sort("booking_date")
    if lines.height == 0:
        return empty_rows()
    items = ctx.items.filter(pl.col("item_type").is_in(["ARI", "ARC"]) & (pl.col("open_amount") > 0)).join(
        _cancelled(ctx), on="item_id", how="anti").select(
        "item_id", "item_type", "entity", "party_id", "party_name", "currency", "docnorm", "issue_date",
        (pl.col("open_amount") * 100).round(0).cast(pl.Int64).alias("c"))
    autos = _automata(ctx, items)
    iban2p = defaultdict(set)
    for ent, p, ib in ctx.customers.select("entity", "customer_id", "iban").iter_rows():
        if ib:
            iban2p[(ent, ib)].add(p)
    # party -> open items (sorted by issue date), doc -> party
    by_party = defaultdict(list)
    doc2p = {}
    for iid, typ, ent, p, _pn, ccy, doc, iss, c in items.sort("issue_date").iter_rows():
        d = re.sub(r"\D", "", doc or "")
        rec = (iid, typ, ccy, doc or "", d, iss, c)
        by_party[(ent, p)].append(rec)
        if doc and len(doc) >= 6:
            doc2p[(ent, doc)] = p
    used = set()
    out = []
    for txn, bdate, ccy, ent, amt, t, iban, raw in lines.iter_rows():
        prio = defaultdict(int)          # party -> evidence score
        if iban:
            for p in iban2p.get((ent, iban), ()):
                prio[p] = max(prio[p], 50)
        A = autos.get(ent)
        if A is not None:
            for end, (L, ps) in A.iter(t):
                for p in ps:
                    prio[p] = max(prio[p], L)
        full = {re.sub(r"[^A-Z0-9]", "", x) for x in re.findall(r"[A-Z0-9][A-Z0-9\-/]{5,}", raw)}
        toks = (full, [x for x in re.findall(r"\d+", raw) if 5 <= len(x) <= 9])
        for tok in full:
            p = doc2p.get((ent, tok))
            if p:
                prio[p] = 100
        if not prio:
            continue
        sol = None
        for p in sorted(prio, key=lambda q: -prio[q])[:20]:
            its = [r for r in by_party.get((ent, p), []) if r[0] not in used and r[2] == ccy and r[5] <= bdate]
            if not its:
                continue
            sol = _solve(its, amt, toks)
            if sol:
                break
        if sol:
            for r in sol:
                used.add(r[0])
                out.append((txn, r[0], abs(r[6]) / 100))
    if not out:
        return empty_rows()
    df = pl.DataFrame(out, schema={"bank_txn_id": pl.Utf8, "item_id": pl.Utf8, "amount": pl.Float64}, orient="row")
    return mk_rows(df, "ar_multi")


def _val(r) -> int:
    return -r[6] if r[1] == "ARC" else r[6]


def _cited(r, toks) -> bool:
    """our doc number appears as a token, or a 5-9 digit token of the text is the tail of its digits (prefix dropped / abbreviated)."""
    full, digs = toks
    if len(r[3]) >= 6 and r[3] in full:
        return True
    d = r[4]
    return len(d) >= 6 and any(d.endswith(x) and len(x) >= min(6, len(d)) for x in digs)


def _solve(its: list, amt: int, toks):
    inv = [r for r in its if r[1] == "ARI"]
    cns = [r for r in its if r[1] == "ARC"]
    # 1. cited items
    cited = [r for r in its if _cited(r, toks)]
    if len(cited) >= 2 or (cited and cns):
        s = sum(_val(r) for r in cited)
        if s == amt and len(cited) >= 2:
            return cited
        rest_cn = [r for r in cns if r not in cited]
        for k in (1, 2):
            for combo in itertools.combinations(rest_cn[:12], k):
                if s + sum(_val(r) for r in combo) == amt:
                    return cited + list(combo)
    # 2. FIFO prefix (>= 2 invoices; single invoices are ar_exact's), optionally with one credit note
    s = 0
    for k, r in enumerate(inv[:MAX_FIFO]):
        s += r[6]
        if k >= 1 and s == amt:
            return inv[:k + 1]
        if k >= 0:
            for cn in cns[:10]:
                if s - cn[6] == amt:
                    return inv[:k + 1] + [cn]
        if s > amt + sum(cn[6] for cn in cns[:10]):
            break
    # 3. unique pair / triple (optionally with one credit note)
    sols = []
    pool = inv[:MAX_PAIR]
    idx = defaultdict(list)
    for i, r in enumerate(pool):
        idx[r[6]].append(i)
    for target_off, extra in [(0, [])] + [(cn[6], [cn]) for cn in cns[:5]]:
        target = amt + target_off
        for i, r in enumerate(pool):
            for j in idx.get(target - r[6], []):
                if j > i:
                    sols.append([r, pool[j]] + extra)
        if len(sols) > 1:
            return None
    if len(sols) == 1:
        return sols[0]
    tri = inv[:MAX_TRIPLE]
    for a, b, c in itertools.combinations(tri, 3):
        if a[6] + b[6] + c[6] == amt:
            sols.append([a, b, c])
            if len(sols) > 1:
                return None
    return sols[0] if len(sols) == 1 else None


RESOLVERS = [multi_invoice]
