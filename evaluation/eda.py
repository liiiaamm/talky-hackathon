"""Scoring-oriented EDA on a TRAIN window using golden labels for MEASUREMENT only."""
import re, sys, time, unicodedata
import numpy as np, pandas as pd
W = "work/train/"; M = "../bankrec-docs-and-masters/masters/"
lo, hi = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("2025-01-01", "2025-06-30")
t0 = time.time()
bt = pd.read_parquet(W + "bank_transactions.parquet")
bt = bt[(bt.booking_date >= lo) & (bt.booking_date <= hi)].set_index("bank_txn_id")
g = pd.read_parquet(W + "golden_allocations.parquet"); g = g[g.bank_txn_id.isin(bt.index)]
ar = pd.read_parquet(W + "ar_invoices.parquet", columns=["item_id","entity","document_number","customer_id","customer_po","issue_date","due_date","currency","total_amount"])
ap = pd.read_parquet(W + "ap_invoices.parquet", columns=["item_id","entity","supplier_invoice_number","supplier_id","our_po","issue_date","due_date","currency","total_amount"])
cn = pd.read_parquet(W + "credit_notes.parquet", columns=["item_id","entity","document_number","party_id","issue_date","currency","total_amount"])
items = pd.concat([ar.rename(columns={"customer_id":"party","customer_po":"po"}),
                   ap.rename(columns={"supplier_invoice_number":"document_number","supplier_id":"party","our_po":"po"}),
                   cn.rename(columns={"party_id":"party"})]).set_index("item_id")
cu = pd.read_csv(M + "customers.csv", dtype=str, keep_default_na=False)
su = pd.read_csv(M + "suppliers.csv", dtype=str, keep_default_na=False)
print(f"window {lo}..{hi}: lines={len(bt):,} golden rows={len(g):,} load {time.time()-t0:.0f}s")

gi = g[g.item_id != ""].copy(); n_items = gi.groupby("bank_txn_id").size()
nl = len(bt); print("\n## Allocation structure")
print(f"no-item lines {100*(1-len(n_items)/nl):.1f}% | 1 item {100*(n_items==1).sum()/nl:.1f}% | >1 {100*(n_items>1).sum()/nl:.1f}%")
print("items/line pct50/90/99/max:", np.percentile(n_items,[50,90,99]).tolist(), n_items.max())
print(f"pairs from lines with >20 items: {100*n_items[n_items>20].sum()/len(gi):.1f}% of pairs, {100*(n_items>20).sum()/nl:.2f}% of lines")
pre = gi.item_id.str[:3]; print("pair prefix share", (pre.value_counts(normalize=True)*100).round(1).to_dict())
print("exception rows", g.exception.value_counts().to_dict())

# reference coverage
N = lambda s: re.sub(r"[^A-Z0-9]", "", s.upper())
blob = (bt.text + " " + bt.ref1 + " " + bt.ref2 + " " + bt.remittance_info + " " + bt.end_to_end_id).str.upper()
nblob = blob.map(N); digblob = blob.str.replace(r"\D+", " ", regex=True).map(lambda s: " " + s + " ")
gi["doc"] = gi.item_id.map(items.document_number); gi["po"] = gi.item_id.map(items.po).fillna("")
nb = nblob.reindex(gi.bank_txn_id).values; db = digblob.reindex(gi.bank_txn_id).values
gi["lit"] = [d.upper() in b for d, b in zip(gi.doc, blob.reindex(gi.bank_txn_id).values)]
gi["norm"] = [N(d) in b for d, b in zip(gi.doc, nb)]
def core(d):
    x = re.sub(r"\D", "", d); return x.lstrip("0")
gi["digits"] = [len(core(d)) >= 4 and (core(d) in b.replace(" ", "")) for d, b in zip(gi.doc, db)]
gi["po_hit"] = [len(p) >= 5 and N(p) in b for p, b in zip(gi.po, nb)]
print("\n## Reference coverage (pairs)")
print({k: round(100*gi[k].mean(), 1) for k in ["lit","norm","digits","po_hit"]})
lines = gi.groupby("bank_txn_id").agg(anyref=("norm","max"), allref=("norm","min"), dig=("digits","max"), n=("item_id","size"))
print(f"lines: all items ref'd (norm) {100*lines.allref.mean():.1f}% | any {100*lines.anyref.mean():.1f}% | any digits-only {100*lines.dig.mean():.1f}% | none {100*(~lines.anyref & ~lines.dig).mean():.1f}%")

# payer identification (AR receipts, first item's customer)
first = gi.drop_duplicates("bank_txn_id").set_index("bank_txn_id"); first["party"] = first.item_id.map(items.party)
arl = first[first.item_id.str.startswith(("ARI","ARC"))].copy()
cuI = cu.set_index("customer_id"); arl["cp"] = bt.counterparty_account.reindex(arl.index); arl["fmt"] = bt.statement_format.reindex(arl.index)
iban2c = cu[cu.iban != ""].groupby("iban").customer_id.apply(set)
arl["iban_ok"] = [p in iban2c.get(i, set()) for p, i in zip(arl.party, arl.cp)]
arl["iban_any"] = [i in iban2c.index for i in arl.cp]
def nn(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z0-9 ]", " ", s)
arl["nm"] = arl.party.map(cuI.name).fillna("").map(nn); arl["tn"] = arl.party.map(cuI.trade_name).fillna("").map(nn)
ab = blob.reindex(arl.index).map(nn).values
arl["name_full"] = [len(n.split()) > 0 and " ".join(n.split()) in " ".join(b.split()) for n, b in zip(arl.nm, ab)]
arl["name_pref12"] = [len(n) > 0 and n.replace(" ", "")[:12] in b.replace(" ", "") for n, b in zip(arl.nm, ab)]
arl["trade"] = [len(t) > 3 and t.replace(" ", "") in b.replace(" ", "") for t, b in zip(arl.tn, ab)]
print("\n## Payer identification, AR lines by format (share %)")
print((arl.groupby("fmt")[["iban_any","iban_ok","name_full","name_pref12","trade"]].mean()*100).round(1).to_string())
print(f"any of iban_ok/name_pref12/trade: {100*(arl.iban_ok|arl.name_pref12|arl.trade).mean():.1f}%  (n={len(arl):,})")
arl["parent"] = arl.party.map(cuI.parent_customer_id).fillna("")
print(f"AR lines whose customer has a parent: {100*(arl.parent!='').mean():.1f}%")

# amount matching: single item exact; all items same party
amt = (bt.amount.astype(float).abs()*100).round().astype(np.int64)
gi["c"] = (gi.amount.astype(float)*100).round().astype(np.int64)
gi["tot"] = (gi.item_id.map(items.total_amount).astype(float)*100).round().astype(np.int64)
gi["party"] = gi.item_id.map(items.party)
one = gi[gi.bank_txn_id.isin(n_items[n_items==1].index)]
print("\n## Amount")
print(f"1-item lines: golden amt == item total {100*(one.c==one.tot).mean():.1f}% ; bank |amt| == item total {100*(amt.reindex(one.bank_txn_id).values==one.tot.values).mean():.1f}%")
multi = gi[gi.bank_txn_id.isin(n_items[n_items>1].index)]
print(f">1-item lines single party: {100*multi.groupby('bank_txn_id').party.nunique().eq(1).mean():.1f}%")

# temporal
gi["bd"] = pd.to_datetime(gi.bank_txn_id.map(bt.booking_date)); gi["iss"] = pd.to_datetime(gi.item_id.map(items.issue_date))
dd = (gi.bd - gi.iss).dt.days
print("\n## Temporal: payment - issue days pct 1/10/50/90/99:", np.percentile(dd.dropna(), [1,10,50,90,99]).round().tolist(), f"| before issue {100*(dd<0).mean():.2f}%")
print("segments (lines, top 25):", g.drop_duplicates("bank_txn_id").segment.value_counts().head(25).to_dict())
print(f"done {time.time()-t0:.0f}s")
