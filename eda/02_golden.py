"""EDA 2: the TRAIN golden. Where the score lives: segments by lines and by pairs, exceptions, categories, trends."""
import os

import duckdb

P = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "parquet"))
con = duckdb.connect()
con.execute(f"CREATE VIEW g AS SELECT * FROM '{P}/train/golden_allocations.parquet'")
con.execute(f"CREATE VIEW bt AS SELECT * FROM '{P}/train/bank_transactions.parquet'")


def q(title, sql):
    print(f"\n## {title}")
    df = con.execute(sql).fetchdf()
    with __import__("pandas").option_context("display.width", 250, "display.max_rows", 200, "display.max_columns", 30):
        print(df.to_string(index=False))


q("Coverage: bank lines vs golden lines", """
SELECT (SELECT count(*) FROM bt) bank_lines, (SELECT count(DISTINCT bank_txn_id) FROM g) golden_lines,
       (SELECT count(*) FROM g) golden_rows, (SELECT count(*) FROM g WHERE item_id IS NOT NULL) pairs""")

q("Segment per line: lines, % lines, % pairs, rows/line, difficulty, sign", """
WITH t AS (SELECT bank_txn_id, any_value(segment) seg, any_value(difficulty) diff, count(item_id) npairs FROM g GROUP BY 1)
SELECT seg, count(*) lines, round(100*count(*)/sum(count(*)) over(),2) pct_lines,
       sum(npairs) pairs, round(100*sum(npairs)/sum(sum(npairs)) over(),2) pct_pairs,
       round(avg(npairs),1) avg_pairs, max(npairs) max_pairs, round(avg(diff::int),2) diff,
       round(100*avg(CASE WHEN bt.amount::double>0 THEN 1 ELSE 0 END),0) pct_credit
FROM t JOIN bt USING(bank_txn_id) GROUP BY 1 ORDER BY lines DESC""")

q("Exceptions per row (and by segment top)", """
SELECT coalesce(exception,'(empty)') exc, count(*) rows, count(DISTINCT bank_txn_id) lines,
       string_agg(DISTINCT segment, ', ') FILTER (WHERE segment IS NOT NULL) segs
FROM g GROUP BY 1 ORDER BY rows DESC""")

q("Categories (no-item rows) with segment", """
SELECT category, exception, count(*) n, string_agg(DISTINCT segment, ', ') segs FROM g WHERE item_id IS NULL GROUP BY 1,2 ORDER BY n DESC""")

q("Item prefix mix in pairs", """SELECT substr(item_id,1,3) pref, coalesce(exception,'') exc, count(*) n FROM g WHERE item_id IS NOT NULL GROUP BY 1,2 ORDER BY n DESC LIMIT 30""")

q("Mixed lines: lines with >1 distinct exception / with credit notes", """
SELECT count(*) FILTER (WHERE nexc>1) lines_multi_exc, count(*) FILTER (WHERE has_arc) lines_with_credit_note, count(*) total
FROM (SELECT bank_txn_id, count(DISTINCT coalesce(exception,'')) nexc, bool_or(item_id LIKE '%C-%') has_arc FROM g GROUP BY 1)""")

q("Segment share by year (lines %) — trend / drift", """
WITH t AS (SELECT bank_txn_id, any_value(segment) seg FROM g GROUP BY 1)
PIVOT (SELECT seg, substr(bt.booking_date,1,4) yr, count(*) n FROM t JOIN bt USING(bank_txn_id) GROUP BY 1,2)
ON yr USING sum(n) ORDER BY "2024" DESC NULLS LAST LIMIT 45""")

q("Lines per month (volume)", """SELECT substr(booking_date,1,7) m, count(*) n FROM bt GROUP BY 1 ORDER BY 1""")
