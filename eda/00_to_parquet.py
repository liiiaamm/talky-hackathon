"""Convert the big csv.gz tables + batch/payment folders of each phase to parquet (all strings, typed later)."""
import argparse
import glob
import os

import duckdb

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--phase", action="append", choices=["sample", "train", "live", "final"], help="convert only this phase; repeat to select several")
args = parser.parse_args()

D = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
OUT = os.path.join(D, "parquet")
os.makedirs(OUT, exist_ok=True)
con = duckdb.connect()
con.execute("SET threads=12")

for ph in args.phase or ["sample", "train", "live", "final"]:
    if not os.path.isdir(f"{D}/{ph}"):
        if args.phase:
            parser.error(f"phase directory does not exist: {D}/{ph}")
        continue
    os.makedirs(f"{OUT}/{ph}", exist_ok=True)
    for f in sorted(glob.glob(f"{D}/{ph}/*.csv.gz")):
        name = os.path.basename(f).replace(".csv.gz", "")
        con.execute(f"COPY (SELECT * FROM read_csv('{f}', all_varchar=true, header=true)) TO '{OUT}/{ph}/{name}.parquet' (FORMAT parquet)")
        print(ph, name)
    for sub in ["direct_debit_batches", "card_settlements", "cheque_deposits", "payment_runs"]:
        files = glob.glob(f"{D}/{ph}/{sub}/*.csv")
        if files:
            con.execute(f"COPY (SELECT *, filename AS src_file FROM read_csv('{D}/{ph}/{sub}/*.csv', all_varchar=true, header=true, filename=true, union_by_name=true)) TO '{OUT}/{ph}/{sub}.parquet' (FORMAT parquet)")
            print(ph, sub, len(files))
    for idx in ["remittance_advices/index.csv", "pdfs/index.csv"]:
        p = f"{D}/{ph}/{idx}"
        if os.path.exists(p):
            name = idx.split("/")[0] + "_index"
            con.execute(f"COPY (SELECT * FROM read_csv('{p}', all_varchar=true, header=true)) TO '{OUT}/{ph}/{name}.parquet' (FORMAT parquet)")
            print(ph, name)
