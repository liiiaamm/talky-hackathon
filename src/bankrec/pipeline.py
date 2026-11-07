"""Orchestrator: run resolvers in priority order; the first resolver that claims a bank line owns it.

    python -m bankrec.pipeline --phase train --from 2025-01-01 --to 2025-06-30 --score
    python -m bankrec.pipeline --phase live --out out/live.csv.gz

Adding a module (teammates): write f(ctx, pending) -> rows (see deterministic.mk_rows) and add it to MODULES below.
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
import time
import traceback

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
_SKIP = set(filter(None, __import__("os").environ.get("BANKREC_SKIP", "").split(",")))   # e.g. BANKREC_SKIP=ra_advices,remittance
_ORDER = __import__("os").environ.get("BANKREC_ORDER", "remittance,pdf_clauses,ar_text,ar_exact,ar_multi,ar_due_subset").split(",")   # optional modules, in order
for _name in _ORDER:       # teammates' modules: expose RESOLVERS = [f, ...]
    if _name in _SKIP:
        continue
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


LEDGER = os.environ.get("BANKREC_LEDGER", "0") == "1"


def apply_ledger(ctx: Ctx, rows: pl.DataFrame):
    """Replay accepted allocations in bank-date order against the opening balances.

    Keep closed items in the private ledger so a later RETURNED can reopen them.
    Policies require fully clearing exceptions to carry the full open amount;
    subtract that amount rather than a sentinel that would swallow future returns.
    Resolver priority still decides which proposal owns a bank line.
    """
    it = rows.filter(pl.col("item_id").is_not_null() & (pl.col("item_id") != ""))
    if it.height == 0:
        return
    state = ctx.__dict__.get("_ledger")
    if state is None:
        roster = ctx.all_items if ctx.all_items is not None else ctx.items
        # all_items includes paid documents needed by returns. The active view is
        # authoritative for opening balances when callers construct a context.
        roster = roster.drop("open_amount").join(
            ctx.items.select("item_id", "open_amount"), on="item_id", how="left"
        ).with_columns(pl.col("open_amount").fill_null(0.0))
        state = ctx.__dict__["_ledger"] = {"opening": roster, "events": []}
    state["events"].append(it.select("bank_txn_id", "item_id", "amount", "exception"))
    events = pl.concat(state["events"], how="vertical_relaxed").join(
        ctx.bank.select("bank_txn_id", "booking_date"), on="bank_txn_id", how="left"
    ).sort("item_id", "booking_date", "bank_txn_id").with_columns(
        pl.when(pl.col("exception") == "RETURNED").then(pl.col("amount"))
        .when(pl.col("exception") == "DUPLICATE_PAYMENT").then(0.0)
        .otherwise(-pl.col("amount")).alias("_change")
    ).with_columns(pl.col("_change").cum_sum().over("item_id").alias("_sum"))
    balances = events.join(
        state["opening"].select("item_id", "open_amount"), on="item_id", how="inner"
    ).sort("item_id", "booking_date", "bank_txn_id").with_columns((pl.col("open_amount") + pl.col("_sum")).alias("_raw"))
    # Reflection at zero reproduces clipping after each chronological event,
    # including a return following an over-applied/fully cleared allocation.
    balances = balances.with_columns(
        (pl.col("_raw") - pl.col("_raw").cum_min().over("item_id").clip(upper_bound=0.0))
        .round(2).alias("_balance")
    ).group_by("item_id").agg(pl.col("_balance").last())
    ctx.items = state["opening"].join(balances, on="item_id", how="left").with_columns(
        pl.coalesce("_balance", "open_amount").alias("open_amount")
    ).drop("_balance").filter(pl.col("open_amount") > 0.005)


def _memory_mb() -> tuple[float, float]:
    """Current RSS and process high-water RSS on Linux; peak RSS elsewhere."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    try:
        with open("/proc/self/statm") as fh:
            rss = int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024 ** 2
    except (OSError, ValueError, IndexError):
        rss = peak
    return rss, peak


def run(ctx: Ctx, modules=None, verbose=True, *, safe=False, profile=False) -> pl.DataFrame:
    modules = MODULES if modules is None else modules
    pending = ctx.bank
    out = []
    ctx.__dict__["_claimed"] = out      # read-only view of the rows accepted so far (ar_due_subset)
    for f in modules:
        t0 = time.perf_counter()
        old_items, old_all_items = ctx.items, ctx.all_items
        had_ledger = hasattr(ctx, "_ledger")
        old_ledger = getattr(ctx, "_ledger", None)
        if isinstance(old_ledger, dict):
            old_ledger = {key: list(value) if isinstance(value, list) else value for key, value in old_ledger.items()}
        rows = None
        status = "ok"
        try:
            rows = f(ctx, pending)
            if rows.height:
                rows = rows.join(pending.select("bank_txn_id"), on="bank_txn_id", how="semi")
                next_pending = pending.join(rows.select("bank_txn_id").unique(), on="bank_txn_id", how="anti")
                if LEDGER:
                    apply_ledger(ctx, rows)
                # Commit ownership only after all processing of this resolver succeeds.
                out.append(rows)
                pending = next_pending
        except Exception:
            ctx.items, ctx.all_items = old_items, old_all_items
            if had_ledger:
                ctx._ledger = old_ledger
            elif hasattr(ctx, "_ledger"):
                del ctx._ledger
            status = "failed"
            if not safe:
                raise
            rows = None
            print(f"WARNING: resolver {f.__name__} failed; its bank lines remain pending", file=sys.stderr, flush=True)
            traceback.print_exc()
        finally:
            elapsed = time.perf_counter() - t0
            if profile:
                rss, peak = _memory_mb()
                print(f"PROFILE {f.__name__} status={status} elapsed_s={elapsed:.3f} rss_mb={rss:.1f} peak_rss_mb={peak:.1f}", flush=True)
        if verbose:
            print(f"  {f.__name__:<22} lines {rows['bank_txn_id'].n_unique() if rows is not None and rows.height else 0:>8,}"
                  f"  rows {rows.height if rows is not None else 0:>9,}  pending {pending.height:>8,}  ({elapsed:.1f}s)", flush=True)
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
    if (it["amount"].is_null() | ~it["amount"].is_finite() | (it["amount"] < 0)).any():
        errs.append("item rows need a finite non-negative amount")
    noit = sub.filter(pl.col("item_id").is_null() | (pl.col("item_id") == ""))
    if noit["amount"].is_not_null().any():
        errs.append("amount must be empty on no-item rows")
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


ITEM_ID_RE = r"^(ARI|ARC|API|APC)-\d{4}-\d{7}$"


def repair(sub: pl.DataFrame, ctx: Ctx) -> pl.DataFrame:
    """FINAL safety net (--repair): make an invalid submission valid by dropping only the offending rows.
    Item rows win over no-item rows; bank lines left without any row get the fallback row. Prints what it changed."""
    n0 = sub.height
    sub = sub.join(ctx.bank.select("bank_txn_id"), on="bank_txn_id", how="semi")
    has_item = pl.col("item_id").is_not_null() & (pl.col("item_id") != "")
    it = sub.filter(has_item)
    good = it["item_id"].str.contains(ITEM_ID_RE) & it["amount"].is_not_null() & it["amount"].is_finite() & (it["amount"] >= 0)
    it = it.filter(good).unique(["bank_txn_id", "item_id"], keep="first", maintain_order=True).with_columns(
        pl.lit(None, pl.Utf8).alias("category"))
    noit = sub.filter(~has_item).join(it.select("bank_txn_id"), on="bank_txn_id", how="anti").unique(
        "bank_txn_id", keep="first", maintain_order=True).with_columns(pl.lit(None, pl.Float64).alias("amount"))
    noit = noit.filter(pl.col("category").is_in(ctx.categories))
    out = pl.concat([it, noit], how="vertical_relaxed").with_columns(
        pl.when(pl.col("exception").fill_null("").is_in(EXCEPTIONS)).then(pl.col("exception")).otherwise(pl.lit("")).alias("exception"))
    missing = ctx.bank.join(out.select("bank_txn_id").unique(), on="bank_txn_id", how="anti")
    out = pl.concat([out, fallback(ctx, missing)], how="vertical_relaxed")
    print(f"REPAIR: {n0:,} -> {out.height:,} rows; dropped {n0 - (out.height - missing.height):,}, "
          f"fallback for {missing.height:,} bank lines", flush=True)
    return out


def write(sub: pl.DataFrame, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df = sub.select(
        "bank_txn_id", pl.col("item_id").fill_null(""),
        pl.when(pl.col("item_id").is_not_null() & (pl.col("item_id") != "")).then(pl.col("amount").round(2).map_elements(lambda x: f"{x:.2f}", return_dtype=pl.Utf8)).otherwise(pl.lit("")).alias("amount"),
        pl.col("category").fill_null(""), pl.col("exception").fill_null("")).sort(SUBMIT_COLS)
    if path.endswith(".gz"):
        import gzip
        with open(path, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=raw, mtime=0) as fh:
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
    ap.add_argument("--safe", action="store_true", help="continue after a failed resolver, with a visible warning")
    ap.add_argument("--profile", action="store_true", help="print per-resolver elapsed time, RSS and process peak RSS")
    ap.add_argument("--repair", action="store_true",
                    help="if validation fails, drop the offending rows, give uncovered lines the fallback row, validate again")
    a = ap.parse_args()
    t0 = time.time()
    ctx = load(a.phase, a.dfrom, a.dto)
    print(f"loaded {a.phase}: {ctx.bank.height:,} bank lines, {ctx.items.height:,} items ({time.time() - t0:.1f}s)")
    sub = run(ctx, safe=a.safe, profile=a.profile)
    errs = validate(sub, ctx)
    print("VALIDATION:", "OK" if not errs else errs)
    if errs and a.repair:
        sub = repair(sub, ctx)
        errs = validate(sub, ctx)
        print("VALIDATION after repair:", "OK" if not errs else errs)
    if errs:
        raise SystemExit(1)
    if a.score and ctx.golden is not None:
        from .score import score
        score(sub, ctx.golden)
    if a.out:
        write(sub, a.out)
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
