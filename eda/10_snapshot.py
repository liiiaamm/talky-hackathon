"""Save pipeline predictions and independent resolver proposals without retaining multiple contexts.

    python eda/10_snapshot.py 2025-01-01 2025-06-30 h1
    python eda/10_snapshot.py 2025-01-01 2025-06-30 h1 --standalone-only --resume

Run under tools/heavy.sh: workers run sequentially and each releases its memory on exit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


def worker(args):
    sys.path.insert(0, str(ROOT / "src"))
    import polars as pl
    from bankrec import pipeline as P
    from bankrec.data import load
    from bankrec.score import score

    prefix = OUT / f"snap_{args.tag}"
    if args.worker == "merge":
        frames = [pl.read_parquet(f"{prefix}_proposal_{name}.parquet") for name in args.resolvers.split(",") if name]
        if frames:
            pl.concat(frames, how="vertical_relaxed").write_parquet(f"{prefix}_alt.parquet")
        return
    ctx = load("train", args.lo, args.hi)
    if args.worker == "pipeline":
        sub = P.run(ctx)
        sub.write_parquet(f"{prefix}_sub.parquet")
        ctx.golden.write_parquet(f"{prefix}_gold.parquet")
        result = score(sub, ctx.golden, verbose=False)
        metrics = {k: v for k, v in result.items() if k not in ("segments", "txn_table")}
        Path(f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2))
        print("SCORE", metrics, flush=True)
        return
    early = [f for f in P.MODULES if f.__module__.endswith(("deterministic", "categories")) and f.__name__ != "all_categories"]
    if args.worker == "pool":
        pending = ctx.bank
        for f in early:
            rows = f(ctx, pending)
            if rows.height:
                pending = pending.join(rows.select("bank_txn_id").unique(), on="bank_txn_id", how="anti")
                if P.LEDGER:
                    P.apply_ledger(ctx, rows)
        pending.select("bank_txn_id").write_parquet(f"{prefix}_pool.parquet")
        # Preserve ledger state so standalone resolvers see exactly the same pool/context.
        if P.LEDGER:
            ctx.items.write_parquet(f"{prefix}_pool_items.parquet")
            ctx.all_items.write_parquet(f"{prefix}_pool_all_items.parquet")
        return
    pending = ctx.bank.join(pl.read_parquet(f"{prefix}_pool.parquet"), on="bank_txn_id", how="semi")
    if P.LEDGER:
        ctx.items = pl.read_parquet(f"{prefix}_pool_items.parquet")
        ctx.all_items = pl.read_parquet(f"{prefix}_pool_all_items.parquet")
    candidates = [f for f in P.MODULES if f not in early and f.__name__ == args.worker]
    if len(candidates) != 1:
        raise ValueError(f"Expected one resolver named {args.worker}, got {len(candidates)}")
    rows = candidates[0](ctx, pending).with_columns(pl.lit(args.worker).alias("resolver"))
    rows.write_parquet(f"{prefix}_proposal_{args.worker}.parquet")
    print(f"standalone {args.worker}: {rows['bank_txn_id'].n_unique():,} lines, {rows.height:,} rows", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lo")
    ap.add_argument("hi")
    ap.add_argument("tag")
    ap.add_argument("--standalone-only", action="store_true", help="Keep existing pipeline sub/gold snapshots")
    ap.add_argument("--resume", action="store_true", help="Reuse completed workers only when inputs and source match")
    ap.add_argument("--resolvers", help="Comma-separated standalone resolver function names")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.worker:
        worker(args)
        return
    # Discover modules in a short-lived process; the coordinator never imports the heavy libraries.
    code = ("import json,sys;sys.path.insert(0,sys.argv[1]);from bankrec import pipeline as p;"
            "print(json.dumps([f.__name__ for f in p.MODULES if not f.__module__.endswith(('deterministic','categories'))]))")
    names = json.loads(subprocess.check_output([sys.executable, "-c", code, str(ROOT / "src")], text=True))
    selected = args.resolvers.split(",") if args.resolvers else names
    if len(set(selected)) != len(selected) or any(n not in names for n in selected):
        ap.error(f"--resolvers must be unique names from {','.join(names)}")
    prefix = OUT / f"snap_{args.tag}"
    fingerprint = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve()]:
        fingerprint.update(str(path.relative_to(ROOT)).encode())
        fingerprint.update(path.read_bytes())
    model = Path(os.environ.get("BANKREC_CAT_MODEL", str(ROOT / "models/categories.joblib"))).resolve()
    if model.exists():
        fingerprint.update(model.read_bytes())
    input_files = list((ROOT / "data/parquet/train").glob("*.parquet")) + list((ROOT / "data/masters").glob("*.csv"))
    input_files += [ROOT / "data/train" / name / "index.csv" for name in ("remittance_advices", "pdfs")]
    inputs = {str(path.resolve()): [path.stat().st_size, path.stat().st_mtime_ns]
              for path in sorted(input_files) if path.is_file()}
    signature = dict(lo=args.lo, hi=args.hi, source=fingerprint.hexdigest(), model=str(model), inputs=inputs,
                     env={k: v for k, v in os.environ.items() if k.startswith("BANKREC_")})
    manifest_path = Path(f"{prefix}_workers.json")
    previous = json.loads(manifest_path.read_text()) if args.resume and manifest_path.exists() else {}
    completed = set(previous.get("completed", [])) if previous.get("signature") == signature else set()
    t0 = time.monotonic()

    def launch(name, outputs):
        if args.resume and name in completed and all(Path(p).exists() for p in outputs):
            print(f"reusing {name}", flush=True)
            return
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), args.lo, args.hi, args.tag,
               "--worker", name, "--resolvers", ",".join(selected)]
        subprocess.run(cmd, check=True)
        completed.add(name)
        manifest_path.write_text(json.dumps(dict(signature=signature, completed=sorted(completed)), indent=2))

    if not args.standalone_only:
        launch("pipeline", [f"{prefix}_sub.parquet", f"{prefix}_gold.parquet", f"{prefix}_metrics.json"])
    pool_outputs = [f"{prefix}_pool.parquet"]
    if os.environ.get("BANKREC_LEDGER", "0") == "1":
        pool_outputs += [f"{prefix}_pool_items.parquet", f"{prefix}_pool_all_items.parquet"]
    launch("pool", pool_outputs)
    for name in selected:
        launch(name, [f"{prefix}_proposal_{name}.parquet"])
    # Always merge the current selection, including when resuming a subset of resolvers.
    completed.discard("merge")
    launch("merge", [f"{prefix}_alt.parquet"])
    print(f"done {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
