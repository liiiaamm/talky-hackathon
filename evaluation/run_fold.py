"""Run Person B's matcher on a chronological fold (or LIVE) and score it with the official scorer.

    python evaluation/run_fold.py 2025H1 [--cfg '{"use_subset": false}'] [--tag name]
    python evaluation/run_fold.py 2025H1 --starter      # starter-equivalent baseline on the same fold

Contains a MINIMAL stand-in resolver/ledger for evaluation only (Person A owns the real one):
lines are processed in booking order, each proposal is committed if it does not over-settle.
Lines without a proposal get the starter placeholder (receipt -> UNIDENTIFIED, debit -> BANK_FEES)
so scores are comparable with the official baseline.
"""
import argparse, csv, gzip, json, os, re, sys, time
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from matching.customer_matcher import CustomerMatcher, ItemStore  # noqa: E402
from matching.party_resolver import PartyResolver  # noqa: E402
import score as SC  # noqa: E402

MASTERS = os.path.join(HERE, "..", "..", "bankrec-docs-and-masters", "masters")
W = os.path.join(HERE, "..", "work")


class SimLedger:
    def __init__(self, open_cents):
        self.o = np.asarray(open_cents, dtype=np.int64).copy()

    def open(self, i):
        return int(self.o[i])


def entity_scope(entities):
    allents = set(entities.entity)
    hub = set(entities[entities.role.str.contains("in-house bank", case=False)].entity)
    return {e: (allents if e in hub else {e}) for e in allents}


def load_masters():
    cu = pd.read_csv(os.path.join(MASTERS, "customers.csv"), dtype=str, keep_default_na=False)
    su = pd.read_csv(os.path.join(MASTERS, "suppliers.csv"), dtype=str, keep_default_na=False)
    en = pd.read_csv(os.path.join(MASTERS, "entities.csv"), dtype=str, keep_default_na=False)
    return cu, su, en


def placeholder(t):
    return [t.bank_txn_id, "", "", "UNIDENTIFIED", "UNIDENTIFIED"] if float(t.amount) > 0 else [t.bank_txn_id, "", "", "BANK_FEES", ""]


def run(fold, cfg, tag, starter=False):
    d = os.path.join(W, "folds", fold)
    t0 = time.time()
    items = pd.read_parquet(os.path.join(d, "items.parquet"))
    bt = pd.read_parquet(os.path.join(d, "bank_transactions.parquet")).sort_values(["booking_date", "bank_txn_id"])
    cu, su, en = load_masters()
    store = ItemStore(items)
    t1 = time.time()
    rows, stats = [], Counter()
    props = []
    if starter:
        # same logic as starter/starter.py: global printed-number index, exact total, first hit
        tot = {iid: int(c) for iid, c in zip(items.item_id, (items.total_amount.astype(float) * 100).round())}
        by_number = {}
        for iid, doc in zip(items.item_id, items.document_number):
            by_number.setdefault(re.sub(r"[^A-Z0-9]", "", (doc or "").upper()), []).append(iid)
        used = set()
        for t in bt.itertuples(index=False):
            amt = int(round(float(t.amount) * 100))
            text = " ".join([t.text, t.ref1, t.ref2, t.remittance_info]).upper()
            hit = None
            for tok in re.split(r"[\s|,;/]+", text):
                if len(tok) < 5:
                    continue
                for iid in by_number.get(re.sub(r"[^A-Z0-9]", "", tok), []):
                    if iid not in used and tot[iid] == abs(amt):
                        hit = iid
                        break
                if hit:
                    break
            if hit:
                used.add(hit)
                rows.append([t.bank_txn_id, hit, f"{abs(amt) / 100:.2f}", "", ""])
                stats["matched"] += 1
            else:
                rows.append(placeholder(t))
        build_s = 0
    else:
        resolver = PartyResolver(cu, su)
        matcher = CustomerMatcher(store, resolver, entity_scope(en), cfg,
                                  terms=pd.read_csv(os.path.join(MASTERS, "payment_terms.csv"), dtype=str, keep_default_na=False))
        ledger = SimLedger(items.open_cents.to_numpy())
        build_s = time.time() - t1
        t1 = time.time()
        for t in bt.itertuples(index=False):
            p = matcher.match(t, ledger)
            ok = not p.abstained
            if ok:   # stand-in resolver: reject over-settlement / duplicates
                seen = set()
                for a in p.allocations:
                    i = store.pos[a["item_id"]]
                    c = int(round(a["amount"] * 100))
                    if a["item_id"] in seen or c > abs(ledger.open(i)) or c <= 0:
                        ok = False
                    seen.add(a["item_id"])
            if ok:
                for a in p.allocations:
                    i = store.pos[a["item_id"]]
                    c = int(round(a["amount"] * 100))
                    ledger.o[i] -= c if ledger.o[i] > 0 else -c
                    rows.append([t.bank_txn_id, a["item_id"], f"{a['amount']:.2f}", "", a["exception"]])
                stats[p.match_method] += 1
            else:
                rows.append(placeholder(t))
                stats["abstain" if p.abstained else "rejected"] += 1
            props.append((t.bank_txn_id, p.match_method, p.confidence, len(p.allocations), p.payer_party_id or ""))
    run_s = time.time() - t1
    out = os.path.join(d, f"sub_{tag}.csv.gz")
    with gzip.open(out, "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["bank_txn_id", "item_id", "amount", "category", "exception"])
        w.writerows(rows)
    if props:
        pd.DataFrame(props, columns=["bank_txn_id", "method", "confidence", "n_alloc", "party"]).to_parquet(os.path.join(d, f"props_{tag}.parquet"))
    r = SC.score(fold, out)
    m = r["metrics"]
    rep = {"fold": fold, "tag": tag, "ok": r["ok"], "score": r["score"], **{k: m.get(k) for k in
           ("allocation_f1", "precision", "recall", "txn_accuracy", "exception_macro_f1", "rows", "txns_covered", "txns_total")},
           "methods": dict(stats), "load_s": round(t1 - t0 - build_s, 1) if not starter else None,
           "build_s": round(build_s, 1), "match_s": round(run_s, 1), "lines_per_s": round(len(bt) / max(run_s, 1e-9)),
           "cfg": cfg, "errors": r["errors"][:3]}
    print(json.dumps(rep, indent=1))
    with open(os.path.join(HERE, "..", "work", "results.jsonl"), "a") as fh:
        fh.write(json.dumps({**rep, "segments": r["segments"]}) + "\n")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fold")
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--tag", default="b")
    ap.add_argument("--starter", action="store_true")
    a = ap.parse_args()
    run(a.fold, json.loads(a.cfg), a.tag if not a.starter else "starter", a.starter)
