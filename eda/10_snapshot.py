"""Error-analysis snapshot: full pipeline output + standalone proposals of every optional module on the same pool.

    python eda/10_snapshot.py 2025-01-01 2025-06-30 h1
Writes out/snap_<tag>_sub.parquet (pipeline rows with module), out/snap_<tag>_alt.parquet (standalone proposals, column 'module'
= resolver name), out/snap_<tag>_gold.parquet.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import polars as pl  # noqa: E402

from bankrec import pipeline as P  # noqa: E402
from bankrec.data import load  # noqa: E402
from bankrec.score import score  # noqa: E402

lo, hi, tag = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = os.path.join(os.path.dirname(__file__), "..", "out")
t0 = time.time()
ctx = load("train", lo, hi)
sub = P.run(ctx)
sub.write_parquet(f"{OUT}/snap_{tag}_sub.parquet")
ctx.golden.write_parquet(f"{OUT}/snap_{tag}_gold.parquet")
r = score(sub, ctx.golden, verbose=False)
print("SCORE", r["score"], r["f1"], r["txn_acc"], r["exc_macro"], f"({time.time() - t0:.0f}s)", flush=True)

# standalone proposals on the pool left after deterministic + strong categories + AP extras (what AR modules compete for)
ctx = load("train", lo, hi)
early = [f for f in P.MODULES if f.__module__.endswith(("deterministic", "categories")) and f.__name__ != "all_categories"]
pending = ctx.bank
for f in early:
    rows = f(ctx, pending)
    if rows.height:
        pending = pending.join(rows.select("bank_txn_id").unique(), on="bank_txn_id", how="anti")
alts = []
for f in P.MODULES:
    if f in early or f.__name__ == "all_categories":
        continue
    t1 = time.time()
    rows = f(ctx, pending)
    alts.append(rows.with_columns(pl.lit(f.__name__).alias("resolver")))
    print(f"  standalone {f.__name__:<20} lines {rows['bank_txn_id'].n_unique() if rows.height else 0:>8,} ({time.time() - t1:.0f}s)", flush=True)
pl.concat(alts, how="vertical_relaxed").write_parquet(f"{OUT}/snap_{tag}_alt.parquet")
pending.select("bank_txn_id").write_parquet(f"{OUT}/snap_{tag}_pool.parquet")
print(f"done {time.time() - t0:.0f}s")
