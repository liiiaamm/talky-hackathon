"""Chronological validation folds built from TRAIN.

A fold = (lo, hi) test window. We emit, under work/folds/<name>/:
  bank_transactions.parquet   lines booked in [lo, hi]
  items.parquet               unified items issued <= hi, with open_amount at lo
  golden.csv.gz / txns.csv.gz files for the official scorer

Opening balances replay ONLY golden rows booked before `lo` (past history, the same
information the organisers used to build open_items_at_start). Same rules validated
against live/open_items_at_start (99.7% exact): RETURNED re-opens, DUPLICATE_PAYMENT
does not move the balance, everything else reduces it. Items issued before 2021-07-01
with no golden history are dropped (paid before TRAIN started), as are cancelled
invoices and their cancelling credit notes (never settled by bank, POLICIES §5.7).
"""
import gzip, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "..", "work")
FOLDS = {"2025H1": ("2025-01-01", "2025-06-30"), "2024H2": ("2024-07-01", "2024-12-31")}


def unified_items(src):
    ar = pd.read_parquet(f"{src}/ar_invoices.parquet")
    ap = pd.read_parquet(f"{src}/ap_invoices.parquet")
    cn = pd.read_parquet(f"{src}/credit_notes.parquet")
    a = pd.DataFrame({"item_id": ar.item_id, "kind": "ARI", "entity": ar.entity, "party_id": ar.customer_id,
                      "document_number": ar.document_number, "issue_date": ar.issue_date, "due_date": ar.due_date,
                      "currency": ar.currency, "total_amount": ar.total_amount, "payment_terms": ar.payment_terms,
                      "po": ar.customer_po, "payment_method": ar.payment_method, "related_document": "", "reason": ""})
    p = pd.DataFrame({"item_id": ap.item_id, "kind": "API", "entity": ap.entity, "party_id": ap.supplier_id,
                      "document_number": ap.supplier_invoice_number, "issue_date": ap.issue_date, "due_date": ap.due_date,
                      "currency": ap.currency, "total_amount": ap.total_amount, "payment_terms": ap.payment_terms,
                      "po": ap.our_po, "payment_method": ap.payment_method, "related_document": "", "reason": ""})
    c = pd.DataFrame({"item_id": cn.item_id, "kind": np.where(cn.side == "AR", "ARC", "APC"), "entity": cn.entity,
                      "party_id": cn.party_id, "document_number": cn.document_number, "issue_date": cn.issue_date,
                      "due_date": "", "currency": cn.currency, "total_amount": cn.total_amount, "payment_terms": "",
                      "po": "", "payment_method": "", "related_document": cn.related_document, "reason": cn.reason})
    return pd.concat([a, p, c], ignore_index=True)


def cents(s):
    return (pd.to_numeric(s, errors="coerce").fillna(0) * 100).round().astype(np.int64)


def build(name):
    lo, hi = FOLDS[name]
    out = os.path.join(W, "folds", name); os.makedirs(out, exist_ok=True)
    src = os.path.join(W, "train")
    bt = pd.read_parquet(f"{src}/bank_transactions.parquet")
    g = pd.read_parquet(f"{src}/golden_allocations.parquet")
    bd = g.bank_txn_id.map(bt.set_index("bank_txn_id").booking_date)
    it = unified_items(src)
    it = it[it.issue_date <= hi].copy()
    past = g[(bd < lo).values & (g.item_id != "").values]
    c = cents(past.amount)
    delta = np.where(past.exception == "RETURNED", c, np.where(past.exception == "DUPLICATE_PAYMENT", 0, -c))
    moved = pd.Series(delta, index=past.item_id.values).groupby(level=0).sum()
    it["open_cents"] = cents(it.total_amount) + it.item_id.map(moved).fillna(0).astype(np.int64)
    touched = it.item_id.isin(moved.index)
    cancel = it[(it.kind == "ARC") & it.reason.str.startswith("Anul")]
    cancelled_docs = set(zip(cancel.entity, cancel.related_document))
    is_cancelled = [(e, d) in cancelled_docs for e, d in zip(it.entity, it.document_number)]
    drop = ((it.issue_date < "2021-07-01") & ~touched) | pd.Series(is_cancelled, index=it.index) | it.item_id.isin(cancel.item_id)
    it = it[~drop & (it.open_cents != 0)]
    it.to_parquet(f"{out}/items.parquet", index=False)
    win = bt[(bt.booking_date >= lo) & (bt.booking_date <= hi)]
    win.to_parquet(f"{out}/bank_transactions.parquet", index=False)
    gw = g[g.bank_txn_id.isin(win.bank_txn_id)]
    gw.to_csv(f"{out}/golden.csv.gz", index=False, compression="gzip")
    with gzip.open(f"{out}/txns.csv.gz", "wt") as fh:
        fh.write("bank_txn_id\n" + "\n".join(win.bank_txn_id) + "\n")
    print(f"{name}: lines={len(win):,} open items={len(it):,} golden rows={len(gw):,}")


if __name__ == "__main__":
    for n in sys.argv[1:] or FOLDS:
        build(n)
