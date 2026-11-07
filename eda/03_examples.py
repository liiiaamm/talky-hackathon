"""EDA 3: real examples per segment: bank line fields vs the golden items (doc number, party, amounts)."""
import os
import sys

import duckdb

P = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "parquet"))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
con = duckdb.connect()
con.execute(f"""
CREATE VIEW g AS SELECT * FROM '{P}/train/golden_allocations.parquet';
CREATE VIEW bt AS SELECT * FROM '{P}/train/bank_transactions.parquet';
CREATE VIEW items AS
  SELECT item_id, document_number docnum, customer_name party, customer_id party_id, issue_date, currency, total_amount, customer_po po FROM '{P}/train/ar_invoices.parquet'
  UNION ALL SELECT item_id, supplier_invoice_number, supplier_name, supplier_id, issue_date, currency, total_amount, our_po FROM '{P}/train/ap_invoices.parquet'
  UNION ALL SELECT item_id, document_number, party_name, party_id, issue_date, currency, total_amount, related_document FROM '{P}/train/credit_notes.parquet';
""")
segs = [r[0] for r in con.execute("SELECT segment FROM g GROUP BY 1 ORDER BY count(DISTINCT bank_txn_id) DESC").fetchall()]
for seg in segs:
    txns = [r[0] for r in con.execute(
        f"SELECT bank_txn_id FROM (SELECT DISTINCT bank_txn_id FROM g WHERE segment='{seg}' AND bank_txn_id >= 'BT-2024') USING SAMPLE {N} ROWS (reservoir, 42)").fetchall()]
    print(f"\n{'#' * 20} {seg}")
    for t in txns:
        b = con.execute(f"SELECT * FROM bt WHERE bank_txn_id='{t}'").fetchdf().iloc[0]
        f = lambda k: '' if b[k] is None or str(b[k]) in ('None', 'nan') else str(b[k])
        print(f"  {t} {f('entity')} {f('statement_format')} {f('booking_date')} {f('amount')} {f('currency')} code={f('bank_code')}/{f('bank_subcode')} ref1={f('ref1')} ref2={f('ref2')}"
              + (f" orig={f('orig_amount')} {f('orig_currency')}" if f('orig_amount') else ''))
        print(f"     text: {f('text')[:230]}")
        extra = ' | '.join(x for x in [f('counterparty_name'), f('counterparty_account'), f('end_to_end_id'), f('remittance_info')[:120]] if x)
        if extra:
            print(f"     camt: {extra}")
        rows = con.execute(f"""SELECT g.item_id, g.amount, g.category, g.exception, i.docnum, i.party, i.issue_date, i.currency, i.total_amount, i.po
                              FROM g LEFT JOIN items i USING(item_id) WHERE bank_txn_id='{t}' LIMIT 4""").fetchall()
        n = con.execute(f"SELECT count(*) FROM g WHERE bank_txn_id='{t}'").fetchone()[0]
        for r in rows:
            if r[0]:
                print(f"     -> {r[0]} amt={r[1]} exc={r[3] or ''} | doc={r[4]} {r[5]} {r[6]} {r[7]} total={r[8]} po/rel={r[9]}")
            else:
                print(f"     -> category={r[2]} exc={r[3] or ''}")
        if n > 4:
            print(f"     ... {n} rows")
