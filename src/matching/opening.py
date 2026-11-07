"""Opening balances for an evaluation window on TRAIN.

LIVE/FINAL ship `open_items_at_start`. TRAIN does not, so for a dev window starting at `lo`
we rebuild it from golden rows booked BEFORE `lo` only (history known at the boundary, the
same information the organisers used). Rules, checked against live/open_items_at_start
(99.7% of its items reproduced to the cent):
  RETURNED re-opens the amount, DUPLICATE_PAYMENT leaves the balance, anything else reduces it.
Items issued before 2021-07-01 with no golden history are treated as closed (paid before TRAIN
starts). Cancelled invoices ("Anulación de factura" credit notes) and their cancelling credit
notes are never settled by bank (POLICIES §5.7) and are dropped in every phase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cents(s) -> np.ndarray:
    return (pd.to_numeric(s, errors="coerce").fillna(0).to_numpy() * 100).round().astype(np.int64)


def replay_deltas(golden: pd.DataFrame) -> pd.Series:
    """golden rows (item rows only) -> signed change of open cents per item_id."""
    g = golden[golden.item_id.fillna("") != ""]
    c = cents(g.amount)
    exc = g.exception.fillna("").to_numpy()
    d = np.where(exc == "RETURNED", c, np.where(exc == "DUPLICATE_PAYMENT", 0, -c))
    return pd.Series(d, index=g.item_id.to_numpy()).groupby(level=0).sum()


def cancelled_mask(items: pd.DataFrame) -> np.ndarray:
    """items needs kind, entity, document_number, reason, related_document."""
    reason = items.get("reason", pd.Series("", index=items.index)).fillna("")
    cn = items[(items.kind == "ARC") & reason.str.startswith("Anul")]
    pairs = set(zip(cn.entity, cn.related_document.fillna("")))
    inv = np.fromiter(((e, d) in pairs for e, d in zip(items.entity, items.document_number)), bool, len(items))
    return inv | items.item_id.isin(cn.item_id).to_numpy()


def train_opening(items: pd.DataFrame, golden: pd.DataFrame, booking: pd.Series, lo: str) -> np.ndarray:
    """Open cents at `lo` for every row of `items` (0 = closed). booking: bank_txn_id -> booking_date."""
    bd = golden.bank_txn_id.map(booking)
    past = golden[(bd < lo).to_numpy()]
    moved = replay_deltas(past)
    total = cents(items.total_amount)
    open_c = total + items.item_id.map(moved).fillna(0).astype(np.int64).to_numpy()
    untouched_old = (items.issue_date < "2021-07-01").to_numpy() & ~items.item_id.isin(moved.index).to_numpy()
    open_c[untouched_old] = 0
    return open_c
