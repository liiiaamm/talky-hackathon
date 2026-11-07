"""Orchestrator: run resolvers in priority order; the first resolver that claims a bank line owns it.

    python -m bankrec.pipeline --phase train --from 2025-01-01 --to 2025-06-30 --score
    python -m bankrec.pipeline --phase live --out out/live.csv.gz

Adding a module (teammates): write f(ctx, pending) -> rows (see deterministic.mk_rows) and add it to MODULES below.
"""
from __future__ import annotations

import argparse
import os
import time

import polars as pl

from . import deterministic as det
from .data import EXCEPTIONS, SUBMIT_COLS, Ctx, load

MODULES = list(det.ALL)
try:  # optional modules from teammates / later stages
    from . import categories as cat
    MODULES.append(cat.strong_categories)   # obvious non-item lines (fees, payroll, cash pool…) right after the exact joins
    LATE = [cat.all_categories]
except ImportError:
    LATE = []
MODULES.extend(det.LATE_AP)
for _name in ("remittance", "ar_exact", "ar_text"):       # teammates' modules: expose RESOLVERS = [f, ...]
    try:
        _m = __import__(f"bankrec.{_name}", fromlist=["RESOLVERS"])
        MODULES.extend(_m.RESOLVERS)
    except ImportError:
        pass
MODULES.extend(LATE)


def fallback(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    return pending.select(
        "bank_txn_id", pl.lit(None, pl.Utf8).alias("item_id"), pl.lit(None, pl.Float64).alias("amount"),
        pl.when(pl.col("is_credit")).then(pl.lit("UNIDENTIFIED")).otherwise(pl.lit("BANK_FEES")).alias("category"),
        pl.when(pl.col("is_credit")).then(pl.lit("UNIDENTIFIED")).otherwise(pl.lit("")).alias("exception"),
        pl.lit("fallback").alias("module"))


def run(ctx: Ctx, modules=None, verbose=True) -> pl.DataFrame:
    modules = MODULES if modules is None else modules
    pending = ctx.bank
    out = []
    for f in modules:
        t0 = time.time()
        rows = f(ctx, pending)
        if rows.height:
            rows = rows.join(pending.select("bank_txn_id"), on="bank_txn_id", how="semi")   # never touch lines already owned
            out.append(rows)
            pending = pending.join(rows.select("bank_txn_id").unique(), on="bank_txn_id", how="anti")
        if verbose:
            print(f"  {f.__name__:<22} lines {rows['bank_txn_id'].n_unique() if rows.height else 0:>8,}  rows {rows.height:>9,}"
                  f"  pending {pending.height:>8,}  ({time.time() - t0:.1f}s)")
    out.append(fallback(ctx, pending))
    return pl.concat(out, how="vertical_relaxed")


def validate(sub: pl.DataFrame, ctx: Ctx) -> list[str]:
    errs = []
    ids = set(ctx.bank["bank_txn_id"].to_list())
    if not set(sub["bank_txn_id"].unique().to_list()) <= ids:
        errs.append("bank_txn_id outside the phase")
    missing = len(ids - set(sub["bank_txn_id"].unique().to_list()))
    if missing:
        errs.append(f"{missing} bank lines missing (they count as wrong)")
    it = sub.filter(pl.col("item_id").is_not_null() & (pl.col("item_id") != ""))
    if it.select(pl.struct("bank_txn_id", "item_id").is_duplicated().any()).item():
        errs.append("duplicated (bank_txn_id, item_id)")
    if not it["item_id"].str.contains(r"^(ARI|ARC|API|APC)-\d{4}-\d{7}$").all():
        errs.append("bad item_id")
    if (it["amount"].is_null() | (it["amount"] < 0)).any():
        errs.append("item rows need a non-negative amount")
    noit = sub.filter(pl.col("item_id").is_null() | (pl.col("item_id") == ""))
    if noit["bank_txn_id"].is_duplicated().any():
        errs.append("more than one no-item row for a bank line")
    if noit.join(it.select("bank_txn_id"), on="bank_txn_id", how="semi").height:
        errs.append("bank line with both item rows and a category row")
    if not noit["category"].is_in(ctx.categories).all():
        errs.append("unknown category")
    if not sub["exception"].fill_null("").is_in(EXCEPTIONS + [""]).all():
        errs.append("unknown exception")
    if it["category"].fill_null("").ne("").any():
        errs.append("category must be empty on item rows")
    return errs


def write(sub: pl.DataFrame, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df = sub.select(
        "bank_txn_id", pl.col("item_id").fill_null(""),
        pl.when(pl.col("item_id").is_not_null() & (pl.col("item_id") != "")).then(pl.col("amount").round(2).map_elements(lambda x: f"{x:.2f}", return_dtype=pl.Utf8)).otherwise(pl.lit("")).alias("amount"),
        pl.col("category").fill_null(""), pl.col("exception").fill_null(""))
    if path.endswith(".gz"):
        import gzip
        with gzip.open(path, "wb", compresslevel=6) as fh:
            df.write_csv(fh)
    else:
        df.write_csv(path)
    size = os.path.getsize(path)
    print(f"wrote {path}: {df.height:,} rows, {size / 1e6:.1f} MB on disk")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="train", choices=["sample", "train", "live", "final"])
    ap.add_argument("--from", dest="dfrom")
    ap.add_argument("--to", dest="dto")
    ap.add_argument("--out")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    ctx = load(a.phase, a.dfrom, a.dto)
    print(f"loaded {a.phase}: {ctx.bank.height:,} bank lines, {ctx.items.height:,} items ({time.time() - t0:.1f}s)")
    sub = run(ctx)
    errs = validate(sub, ctx)
    print("VALIDATION:", "OK" if not errs else errs)
    if a.score and ctx.golden is not None:
        from .score import score
        score(sub, ctx.golden)
    if a.out:
        write(sub, a.out)
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
