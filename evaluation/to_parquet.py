"""Cache phase CSVs as parquet (all columns as string) under work/<phase>/. Idempotent."""
import os, sys, time
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK = os.path.join(os.path.dirname(__file__), "..", "work")
TABLES = ["bank_transactions", "ar_invoices", "ap_invoices", "credit_notes", "open_items_at_start", "golden_allocations"]


def main(phases):
    for ph in phases:
        os.makedirs(os.path.join(WORK, ph), exist_ok=True)
        for t in TABLES:
            src = os.path.join(ROOT, ph, f"{t}.csv.gz")
            dst = os.path.join(WORK, ph, f"{t}.parquet")
            if not os.path.exists(src) or os.path.exists(dst):
                continue
            t0 = time.time()
            df = pd.read_csv(src, dtype=str, keep_default_na=False)
            df.to_parquet(dst, index=False)
            print(f"{ph}/{t}: {len(df):,} rows {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or ["train", "live"])
