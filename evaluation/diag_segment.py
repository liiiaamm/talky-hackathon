"""Why does a segment fail? Uses golden ONLY to diagnose (never as a feature).
    python evaluation/diag_segment.py 2025H1 out/sub.csv.gz ar_ref_noise [n]"""
import sys, collections, numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "evaluation")
from run_fold import load_masters, entity_scope, SimLedger
from matching.customer_matcher import CustomerMatcher, ItemStore
from matching.party_resolver import PartyResolver
fold, sub, seg = sys.argv[1:4]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
d = f"work/folds/{fold}"
g = pd.read_csv(f"{d}/golden.csv.gz", dtype=str, keep_default_na=False)
s = pd.read_csv(sub, dtype=str, keep_default_na=False)
segl = g[g.segment == seg]
gs = segl[segl.item_id != ""].groupby("bank_txn_id").item_id.apply(frozenset)
ss = s[s.item_id != ""].groupby("bank_txn_id").item_id.apply(frozenset)
claimed = gs.index.isin(ss.index)
same = np.array([ss.get(t) == v for t, v in gs.items()])
print(f"{seg}: lines {len(gs)} | claimed {claimed.mean():.3f} | item-set exact {same.mean():.3f} | claimed-but-wrong-set {(claimed & ~same).mean():.3f}")
items = pd.read_parquet(f"{d}/items.parquet"); store = ItemStore(items)
cu, su, en = load_masters(); res = PartyResolver(cu, su); scope = entity_scope(en)
bt = pd.read_parquet(f"{d}/bank_transactions.parquet").set_index("bank_txn_id")
L = SimLedger(items.open_cents.to_numpy())
why = collections.Counter(); ex = collections.defaultdict(list)
for t in [t for t, c in zip(gs.index, claimed) if not c][:n]:
    r = bt.loc[t]; gold = gs[t]; gp = {store.party[store.pos[i]] for i in gold if i in store.pos}
    if not all(i in store.pos for i in gold): why["gold item not open at window start"] += 1; continue
    blob = " ".join((r.text, r.ref1, r.ref2, r.remittance_info, r.end_to_end_id))
    pc = res.resolve(blob, r.counterparty_name, r.counterparty_account, scope.get(r.entity, {r.entity}), "AR", top_k=10)
    ids = [c.party_id for c in pc]
    rank = min((ids.index(p) for p in gp if p in ids), default=None)
    tgt = int(round(abs(float(r.amount)) * 100)); gsum = sum(int(store.sign[store.pos[i]] * L.open(store.pos[i])) for i in gold)
    pool = sum(1 for p in gp for i in store.by_party.get(p, ()) if L.open(i))
    if len(gp) > 1: k = "gold spans >1 party"
    elif rank is None: k = "party not in top10"
    elif rank >= 3: k = "party rank>=3"
    elif pc[rank].score < 0.55: k = "party score<0.55"
    elif gsum != tgt: k = "gold open-sum != cash (deduction/FX/stale balance)"
    elif pool > 40: k = "pool>40 (truncated)"
    else: k = "subset search missed (budget/order)"
    why[k] += 1
    if len(ex[k]) < 3: ex[k].append((t, r.text[:90], [(c.party_id, round(c.score, 2)) for c in pc[:3]], sorted(gp), gsum, tgt, pool))
for k, v in why.most_common():
    print(f"  {v:6d}  {k}")
    for e in ex[k]: print("        ", e)
