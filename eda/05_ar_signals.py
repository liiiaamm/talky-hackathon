"""EDA 5: AR receipt lines: which signals identify the golden items (doc number in text, digits, name, amount)."""
import os
import re

import duckdb
import pandas as pd

P = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "parquet", "train"))
con = duckdb.connect()
con.execute(f"""
CREATE VIEW g AS SELECT * FROM '{P}/golden_allocations.parquet' WHERE bank_txn_id >= 'BT-2024';
CREATE VIEW bt AS SELECT * FROM '{P}/bank_transactions.parquet' WHERE bank_txn_id >= 'BT-2024';
CREATE VIEW ar AS SELECT item_id, document_number docnum, customer_id, customer_name, total_amount, customer_po FROM '{P}/ar_invoices.parquet'
  UNION ALL SELECT item_id, document_number, party_id, party_name, total_amount, related_document FROM '{P}/credit_notes.parquet' WHERE side='AR';
""")
M = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "masters"))
cust = pd.read_csv(f"{M}/customers.csv", dtype=str).set_index("customer_id")

segs = ["ar_ref_noise", "ar_1to1", "ar_same_amount_fifo", "ar_1toN_text", "ar_short_discount", "ar_same_amount_month", "ar_instalment",
        "ar_partial", "unidentified", "ar_1toN_credit_note", "ar_n_to_m", "unidentified_lookalike", "ar_short_withholding",
        "ar_1toN_remittance", "ar_fx", "ar_short_dispute", "ar_dispute_settlement", "ar_retention", "ar_discount_late",
        "ar_third_party_confirming", "ar_duplicate_payment", "ar_overpayment", "ar_third_party_parent", "ar_new_iban"]
df = con.execute(f"""
SELECT g.segment, t.bank_txn_id, t.amount::double amt, t.statement_format fmt,
       upper(concat_ws(' ', t.text, t.ref1, t.ref2, t.remittance_info, t.counterparty_name)) txt, t.counterparty_account iban,
       list(a.docnum) docs, list(a.customer_id) custs, list(a.total_amount::double) totals, list(g.amount::double) gamts, list(a.customer_po) pos
FROM g JOIN bt t USING(bank_txn_id) LEFT JOIN ar a USING(item_id)
WHERE g.segment IN ({','.join(f"'{s}'" for s in segs)})
GROUP BY ALL""").fetchdf()

norm = lambda s: re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def sig(r):
    t = r.txt or ""
    tn = norm(t)
    digits = re.findall(r"\d{4,}", t)
    docs = [d for d in r.docs if d]
    out = {}
    out["doc_exact"] = any(norm(d) in tn for d in docs) if docs else None
    # doc number digits without prefix / leading zeros, suffix of >=5 digits
    def partial(d):
        dd = re.sub(r"\D", "", d).lstrip("0")
        return any(len(x) >= 5 and (x.lstrip("0") in dd or dd.endswith(x.lstrip("0"))) for x in digits)
    out["doc_partial"] = any(partial(d) for d in docs) if docs else None
    out["po_in_text"] = any(p and norm(p) in tn for p in r.pos) if docs else None
    names = []
    for c in set(x for x in r.custs if x):
        if c in cust.index:
            names += [cust.at[c, "name"], cust.at[c, "trade_name"]]
    nn = [norm(n)[:10] for n in names if isinstance(n, str) and n]
    out["name10_in_text"] = any(n in tn for n in nn) if docs else None
    ibans = [cust.at[c, "iban"] for c in set(x for x in r.custs if x) if c in cust.index]
    out["iban_eq_master"] = (bool(r.iban) and r.iban in ibans) if (docs and r.fmt == "camt053") else None
    out["sum_eq_amt"] = abs(sum(r.gamts) - abs(r.amt)) < 0.01 if docs else None
    out["single_total_eq_amt"] = (len(docs) == 1 and abs(r.totals[0] - abs(r.amt)) < 0.01) if docs else None
    out["no_digits"] = len(digits) == 0
    return pd.Series(out)


s = pd.concat([df[["segment"]], df.apply(sig, axis=1)], axis=1)
res = s.groupby("segment").agg(lambda x: round(100 * x.dropna().astype(float).mean(), 1) if x.dropna().size else None)
res.insert(0, "lines", s.groupby("segment").size())
pd.set_option("display.width", 250)
print(res.sort_values("lines", ascending=False).to_string())
