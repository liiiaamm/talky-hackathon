"""EDA 1: shape, columns and a few rows of every main table (TRAIN + LIVE) and masters."""
import glob
import os

import duckdb

D = os.path.join(os.path.dirname(__file__), "..", "data")
con = duckdb.connect()
con.execute("SET threads=8")


def show(path, n=3):
    rel = f"read_csv_auto('{path}', all_varchar=true, sample_size=-1)"
    cnt = con.execute(f"SELECT count(*) FROM {rel}").fetchone()[0]
    df = con.execute(f"SELECT * FROM {rel} LIMIT {n}").fetchdf()
    print(f"\n### {os.path.relpath(path, D)}  rows={cnt:,}  cols={len(df.columns)}")
    print("   ", list(df.columns))
    with __import__("pandas").option_context("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60):
        print(df.to_string(index=False))


for ph in ["train", "live"]:
    print(f"\n{'=' * 30} {ph.upper()} {'=' * 30}")
    for f in sorted(glob.glob(f"{D}/{ph}/*.csv.gz")):
        show(f)
    for sub in ["direct_debit_batches", "card_settlements", "cheque_deposits", "payment_runs", "remittance_advices", "pdfs", "bank_statements"]:
        files = glob.glob(f"{D}/{ph}/{sub}/**/*", recursive=True)
        files = [x for x in files if os.path.isfile(x)]
        exts = {}
        for x in files:
            e = os.path.splitext(x)[1]
            exts[e] = exts.get(e, 0) + 1
        print(f"\n--- {ph}/{sub}: {len(files)} files {exts}")

print(f"\n{'=' * 30} MASTERS {'=' * 30}")
for f in sorted(glob.glob(f"{D}/masters/*.csv")):
    show(f, 4)
