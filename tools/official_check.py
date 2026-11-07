"""Pre-submission checks for a LIVE/FINAL prediction file. Read-only; never uploads anything.

    python tools/official_check.py <phase> <file.csv.gz>

1. Official format validation: data/starter/bankrec_score.py (the server's scorer) with an empty golden, so every rule the
   server applies to the file itself is checked (header, ids of the phase, item id format, categories, exceptions, amounts,
   duplicates, a line with both item and category rows, size and row limits).
2. Coverage: every bank line of the phase has at least one row.
3. Integrity against open_items_at_start: items allocated above their open amount without OVERPAYMENT / DUPLICATE_PAYMENT /
   RETURNED, and items settled by several lines with no RETURNED / PARTIAL / DUPLICATE_PAYMENT (reported, not fatal).
Exit code 1 on any fatal problem.
"""
import csv
import gzip
import os
import sys
import tempfile

import polars as pl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "data", "starter"))
import bankrec_score as S  # noqa: E402

EXC = ["PARTIAL", "OVERPAYMENT", "SHORT_DISCOUNT", "SHORT_BANK_CHARGES", "SHORT_WITHHOLDING", "SHORT_DISPUTE",
       "DUPLICATE_PAYMENT", "UNIDENTIFIED", "RETURNED", "FX_DIFFERENCE"]


def main():
    phase, path = sys.argv[1], sys.argv[2]
    pq = os.path.join(ROOT, "data", "parquet", phase)
    bank = pl.read_parquet(os.path.join(pq, "bank_transactions.parquet"), columns=["bank_txn_id"])
    masters = os.path.join(ROOT, "data", phase, "masters_final")
    cat_csv = os.path.join(masters if os.path.isdir(masters) else os.path.join(ROOT, "data", "masters"), "categories.csv")
    cats = [r["category"] for r in csv.DictReader(open(cat_csv, encoding="utf-8"))]
    fatal = []
    with tempfile.TemporaryDirectory() as d:
        g, t = os.path.join(d, "g.csv"), os.path.join(d, "t.csv")
        open(g, "w").write("bank_txn_id,item_id,amount,category,exception\n")
        bank.write_csv(t)
        res = S.score_submission(S.load_golden(g, t, EXC, cats), open(path, "rb").read(), os.path.basename(path))
    print(f"official format validation: ok={res['ok']} errors={res['errors'][:5]}")
    if not res["ok"]:
        fatal.append("official validation failed")
    raw = open(path, "rb").read()
    sub = pl.read_csv(gzip.decompress(raw) if path.endswith(".gz") else raw, infer_schema_length=0)
    missing = bank.join(sub.select("bank_txn_id").unique(), on="bank_txn_id", how="anti").height
    print(f"rows {sub.height:,}; bank lines {bank.height:,}; missing lines {missing:,}")
    if missing:
        fatal.append(f"{missing} bank lines without a row")
    oi_path = os.path.join(pq, "open_items_at_start.parquet")
    if os.path.exists(oi_path):
        oi = pl.read_parquet(oi_path, columns=["item_id", "open_amount"]).with_columns(pl.col("open_amount").cast(pl.Float64))
        it = sub.filter(pl.col("item_id").fill_null("") != "").with_columns(pl.col("amount").cast(pl.Float64))
        agg = it.group_by("item_id").agg(pl.col("amount").sum(), pl.col("bank_txn_id").n_unique().alias("nl"),
                                         pl.col("exception").fill_null("").unique().str.join("+").alias("ex")).join(oi, on="item_id", how="left")
        over = agg.filter(pl.col("open_amount").is_not_null() & (pl.col("amount") > pl.col("open_amount") * 1.005 + 0.01)
                          & ~pl.col("ex").str.contains("OVERPAYMENT|DUPLICATE|RETURNED")).height
        multi = agg.filter((pl.col("nl") > 1) & ~pl.col("ex").str.contains("RETURNED|PARTIAL|DUPLICATE")).height
        print(f"over-allocated items (no OVERPAYMENT/DUPLICATE/RETURNED): {over:,}; items on >1 line unexplained: {multi:,}")
    if fatal:
        print("NOT READY:", "; ".join(fatal))
        sys.exit(1)
    print("checks OK (nothing was submitted)")


if __name__ == "__main__":
    main()
