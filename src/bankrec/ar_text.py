"""Person B resolver: payer identification + invoice references + amount search (see src/matching/).

Pipeline contract: f(ctx, pending) -> rows [bank_txn_id, item_id, amount, category, exception, module].
Claims only lines where the matcher proposes allocations with confidence >= cfg['min_confidence'];
everything else stays pending for the late category classifier / fallback.

Balances: the pipeline has no shared ledger yet, so this resolver keeps its own chronological
balances for the lines it processes, seeded from ctx.items.open_amount (LIVE/FINAL: from
open_items_at_start). On TRAIN dev windows, ctx.items.open_amount is the invoice total, so the
opening balance at the window start is rebuilt from golden rows booked BEFORE the window
(matching/opening.py) - history only, never labels of the evaluated lines.

Config (ablations): env BANKREC_AR_TEXT_CFG='{"use_subset": false, ...}' (keys: CustomerMatcher.cfg
plus 'sides' ['AR'] / ['AR','AP'] and 'min_confidence').
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import polars as pl

from .data import PQ, Ctx
from .ap_bulk import bulk_mixed_runs
from .deterministic import ROWS_SCHEMA, empty_rows

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from matching.customer_matcher import CustomerMatcher, ItemStore  # noqa: E402
from matching.opening import cancelled_mask, cents, train_opening  # noqa: E402
from matching.party_resolver import PartyResolver  # noqa: E402

DEFAULT_CFG = {"sides": ["AR"], "min_confidence": 0.0, "node_budget": 2000, "max_pool": 40, "top_parties": 8, "use_window": True}
STATE = {}   # last run's matcher/store/ledger, for Person C and diagnostics


def _cfg():
    c = dict(DEFAULT_CFG)
    c.update(json.loads(os.environ.get("BANKREC_AR_TEXT_CFG", "{}")))
    return c


def _credit_note_reasons() -> pd.DataFrame:
    parts = []
    for p in glob.glob(os.path.join(PQ, "*", "credit_notes.parquet")):
        parts.append(pd.read_parquet(p, columns=["item_id", "reason"]))
    return pd.concat(parts).drop_duplicates("item_id") if parts else pd.DataFrame(columns=["item_id", "reason"])


def _items(ctx: Ctx, window_start: str) -> pd.DataFrame:
    it = ctx.items.select(
        "item_id", pl.col("item_type").alias("kind"), "entity", "party_id", pl.col("docnum").alias("document_number"),
        "issue_date", "currency", "po", "related_document", "payment_terms",
        pl.col("total").alias("total_amount"), "open_amount").to_pandas()
    it = it.merge(_credit_note_reasons(), on="item_id", how="left")
    for c in ("document_number", "po", "related_document", "reason", "issue_date", "party_id", "payment_terms"):
        it[c] = it[c].fillna("")
    if ctx.phase in ("train", "sample"):
        gp = os.path.join(PQ, ctx.phase, "golden_allocations.parquet")
        bp = os.path.join(PQ, ctx.phase, "bank_transactions.parquet")
        g = pd.read_parquet(gp, columns=["bank_txn_id", "item_id", "amount", "exception"])
        b = pd.read_parquet(bp, columns=["bank_txn_id", "booking_date"])
        it["open_cents"] = train_opening(it, g, b.set_index("bank_txn_id").booking_date, window_start)
        # never above what the pipeline still has open (window opening, and earlier resolvers with BANKREC_LEDGER=1)
        if os.environ.get("BANKREC_AR_TEXT_SEED", "1") == "1":
            it["open_cents"] = np.minimum(it["open_cents"].to_numpy(), cents(it.open_amount))
    else:
        it["open_cents"] = cents(it.open_amount)
    it.loc[cancelled_mask(it), "open_cents"] = 0
    return it[it.open_cents != 0].reset_index(drop=True)


class _Ledger:
    def __init__(self, open_cents):
        self.o = np.asarray(open_cents, dtype=np.int64).copy()

    def open(self, i):
        return int(self.o[i])


def _entity_scope(ctx: Ctx) -> dict:
    ents = pl.read_csv(os.path.join(os.path.dirname(PQ), "masters", "entities.csv"), infer_schema_length=0)
    allents = set(ents["entity"].to_list()) | set(ctx.bank["entity"].unique().to_list())
    hub = set(ents.filter(pl.col("role").str.to_lowercase().str.contains("in-house bank"))["entity"].to_list())
    return {e: (allents if e in hub else {e}) for e in allents}


def ar_text(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    cfg = _cfg()
    sides = set(cfg.pop("sides"))
    min_conf = cfg.pop("min_confidence")
    lines = pending
    if sides == {"AR"}:
        lines = pending.filter(pl.col("is_credit"))
    elif sides == {"AP"}:
        lines = pending.filter(~pl.col("is_credit"))
    if lines.height == 0:
        return empty_rows()
    start = ctx.bank["booking_date"].min()
    items = _items(ctx, start)
    store = ItemStore(items)
    cu, su = ctx.customers.to_pandas().fillna(""), ctx.suppliers.to_pandas().fillna("")
    matcher = CustomerMatcher(store, PartyResolver(cu, su), _entity_scope(ctx), cfg, terms=ctx.terms.to_pandas().fillna(""),
                              fx=ctx.fx.to_pandas() if ctx.fx is not None else None)
    ledger = _Ledger(items.open_cents.to_numpy())
    STATE.update(matcher=matcher, store=store, ledger=ledger)
    cols = ["bank_txn_id", "entity", "amount", "currency", "orig_amount", "orig_currency", "text", "ref1", "ref2",
            "remittance_info", "end_to_end_id", "counterparty_name", "counterparty_account", "booking_date",
            "bank_code", "statement_format"]
    df = lines.select([pl.col(c).fill_null("") for c in cols]).sort(["booking_date", "bank_txn_id"]).to_pandas()
    out = {k: [] for k in ROWS_SCHEMA}
    for t in df.itertuples(index=False):
        p = matcher.match(t, ledger)
        if p.abstained or p.confidence < min_conf or not p.allocations:
            continue
        idx, ok = [], True
        for a in p.allocations:
            i = store.pos[a["item_id"]]
            c = int(round(a["amount"] * 100))
            dup = a["exception"] == "DUPLICATE_PAYMENT"      # pays an item already settled: leaves the balance
            if c <= 0 or (not dup and c > abs(ledger.open(i))) or i in idx:
                ok = False
                break
            idx.append(i)
        if not ok:
            continue
        for a, i in zip(p.allocations, idx):
            c = int(round(a["amount"] * 100))
            if a["exception"] != "DUPLICATE_PAYMENT":
                ledger.o[i] -= c if ledger.o[i] > 0 else -c
            out["bank_txn_id"].append(t.bank_txn_id)
            out["item_id"].append(a["item_id"])
            out["amount"].append(round(c / 100, 2))
            out["category"].append(None)
            out["exception"].append(a["exception"])
            out["module"].append(f"ar_{p.match_method}")
        matcher.record(p, int(round(abs(float(t.amount)) * 100)), pd.Timestamp(t.booking_date).toordinal())
    return pl.DataFrame(out, schema=ROWS_SCHEMA)


# ap_bulk runs just before ar_text through this list, so it is active with the default BANKREC_ORDER and no pipeline change.
# If `ap_bulk` is also added to BANKREC_ORDER the second copy claims nothing (lines it owns are no longer pending).
RESOLVERS = [bulk_mixed_runs, ar_text]
