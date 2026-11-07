"""EDA 4: how the structured files link to bank lines, and how well that reproduces the golden."""
import os

import duckdb

P = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "parquet", "train"))
con = duckdb.connect()
con.execute(f"""
CREATE VIEW g AS SELECT * FROM '{P}/golden_allocations.parquet';
CREATE TABLE bt AS SELECT *, upper(concat_ws(' ', text, ref1, ref2, end_to_end_id, remittance_info, counterparty_name)) alltext FROM '{P}/bank_transactions.parquet';
CREATE VIEW dd AS SELECT * FROM '{P}/direct_debit_batches.parquet';
CREATE VIEW cs AS SELECT * FROM '{P}/card_settlements.parquet';
CREATE VIEW cq AS SELECT * FROM '{P}/cheque_deposits.parquet';
CREATE VIEW pr AS SELECT * FROM '{P}/payment_runs.parquet';
CREATE VIEW ar AS SELECT item_id, document_number, entity FROM '{P}/ar_invoices.parquet';
CREATE VIEW items AS SELECT item_id, document_number docnum, entity FROM '{P}/ar_invoices.parquet'
   UNION ALL SELECT item_id, supplier_invoice_number, entity FROM '{P}/ap_invoices.parquet'
   UNION ALL SELECT item_id, document_number, entity FROM '{P}/credit_notes.parquet';
""")


def q(title, sql):
    print(f"\n## {title}")
    print(con.execute(sql).fetchdf().to_string(index=False))


# --- DD batches: batch_id in bank text? amount = batch_total?
q("DD batches: batch_id found in bank text of a line with amount == batch_total (same account)", """
WITH b AS (SELECT batch_id, account_id, any_value(batch_total)::double tot, any_value(collection_date) d, count(*) n FROM dd GROUP BY 1,2)
SELECT count(*) batches, count(t.bank_txn_id) linked_by_id, count(*) FILTER (WHERE abs(t.amount::double - b.tot) < 0.005) amount_eq
FROM b LEFT JOIN bt t ON t.account_id=b.account_id AND contains(t.alltext, b.batch_id)""")
q("DD rows: remittance_info == item document_number?  (and golden agreement)", """
SELECT count(*) dd_rows, count(a.item_id) doc_found FROM dd JOIN (SELECT DISTINCT account_id, entity FROM bt) e USING(account_id) LEFT JOIN ar a ON a.document_number = dd.remittance_info AND a.entity=e.entity""")

# --- cards: MID + settlement date in text?
q("Card settlements: merchant_id in text and sum(gross) vs net bank amount", """
WITH s AS (SELECT settlement_id, account_id, merchant_id, settlement_date, sum(gross::double) gross, sum(net::double) net, count(*) n FROM cs GROUP BY ALL)
SELECT count(*) settlements, count(t.bank_txn_id) linked_mid_date, count(*) FILTER (WHERE abs(t.amount::double-s.net)<0.02) amt_eq_net,
       count(*) FILTER (WHERE abs(t.amount::double-s.gross)<0.02) amt_eq_gross
FROM s LEFT JOIN bt t ON t.account_id=s.account_id AND contains(t.alltext, s.merchant_id) AND t.booking_date::date BETWEEN s.settlement_date::date AND s.settlement_date::date + 3""")

# --- cheques: deposit_id in text?
q("Cheque deposits: deposit_id in text", """
WITH s AS (SELECT deposit_id, account_id, any_value(deposit_total)::double tot FROM cq GROUP BY ALL)
SELECT count(*) deposits, count(t.bank_txn_id) linked, count(*) FILTER (WHERE abs(t.amount::double-s.tot)<0.02) amt_eq
FROM s LEFT JOIN bt t ON t.account_id=s.account_id AND contains(t.alltext, s.deposit_id)""")

# --- payment runs: e2e id or run id in bank fields
q("Payment runs by type: transfer (e2e) found in bank fields; run_id found (aggregated/confirming)", """
WITH e AS (SELECT run_type, end_to_end_id, run_id, debtor_account_id acc, any_value(transfer_amount)::double amt FROM pr GROUP BY ALL)
SELECT run_type, count(*) transfers,
       count(*) FILTER (WHERE EXISTS (SELECT 1 FROM bt t WHERE t.account_id=e.acc AND contains(t.alltext, e.end_to_end_id))) e2e_full,
       count(*) FILTER (WHERE EXISTS (SELECT 1 FROM bt t WHERE t.account_id=e.acc AND contains(t.alltext, e.run_id))) run_id_found
FROM e GROUP BY 1""")

# --- golden agreement for payment runs: golden pairs of AP segments covered by a payment_runs document
q("AP golden pairs whose item doc number appears in payment_runs (same entity) — by segment", """
SELECT g.segment, count(*) pairs, count(*) FILTER (WHERE EXISTS (SELECT 1 FROM pr WHERE pr.document_number=i.docnum)) in_runs
FROM g JOIN items i USING(item_id) WHERE g.item_id LIKE 'AP%' GROUP BY 1 ORDER BY 2 DESC""")

# --- what is in ref fields for AP lines (pattern of run/e2e ids)
q("AP lines: does bank text carry an id like PR...-NNNN?", """
SELECT g.segment, t.statement_format, count(DISTINCT t.bank_txn_id) lines,
       round(100*avg(regexp_matches(t.alltext, 'PR\\d{12}')::int),1) pct_PRid,
       round(100*avg(regexp_matches(t.alltext, 'CF\\d{12}')::int),1) pct_CFid
FROM g JOIN bt t USING(bank_txn_id) WHERE g.segment LIKE 'ap_%' GROUP BY 1,2 ORDER BY 1,2""")
