"""Read-only preflight for a LIVE/FINAL run with the validated INT6 configuration (tools/final_config.sh). Never uploads.

    source tools/final_config.sh && python tools/preflight.py <phase>

Fails (exit 1) on: hash randomisation active, configuration drift, missing dependency or resolver, model file missing or not
matching its pinned sha256, missing phase data, output directory not writable, too little free disk. Set
BANKREC_ALLOW_UNPINNED_MODELS=1 only for a deliberate, approved model change.
"""
import hashlib
import importlib
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORDER = "remittance,pdf_clauses,ar_text,ar_exact,ar_multi,c_partial,c_discount,c_bank_charges,c_fx,ar_due_subset"
DEPS = ("polars", "numpy", "pandas", "sklearn", "joblib", "ahocorasick", "pypdf", "openpyxl", "lightgbm")
RESOLVERS = ("remittances", "pdf_clauses", "bulk_mixed_runs", "ar_text", "exact_amount", "multi_invoice", "partial_payments",
             "discount_receipts", "bank_charges", "fx_receipts", "due_subset", "strong_categories", "all_categories")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "final"
    bad = []

    def check(ok, msg):
        print(("ok    " if ok else "FAIL  ") + msg)
        if not ok:
            bad.append(msg)

    check(os.environ.get("PYTHONHASHSEED") == "0" and sys.flags.hash_randomization == 0,
          f"PYTHONHASHSEED=0 inherited at interpreter start (env={os.environ.get('PYTHONHASHSEED')}, "
          f"hash_randomization={sys.flags.hash_randomization})")
    check(os.environ.get("BANKREC_LEDGER") == "1", "BANKREC_LEDGER=1")
    check(os.environ.get("BANKREC_ORDER") == ORDER, "BANKREC_ORDER = validated INT6 order")
    check(not os.environ.get("BANKREC_SKIP"), "BANKREC_SKIP unset")
    for mod in DEPS:
        try:
            importlib.import_module(mod)
            ok = True
        except Exception as e:  # noqa: BLE001
            ok, mod = False, f"{mod} ({type(e).__name__}: {e})"
        check(ok, f"dependency {mod}")
    allow = os.environ.get("BANKREC_ALLOW_UNPINNED_MODELS") == "1"
    for var, pin in (("BANKREC_CAT_MODEL", "BANKREC_CAT_SHA256"), ("BANKREC_PARTIAL_MODEL", "BANKREC_PARTIAL_SHA256")):
        path = os.environ.get(var, "")
        exists = bool(path) and os.path.isabs(path) and os.path.isfile(path) and os.access(path, os.R_OK)
        check(exists, f"{var} is an absolute, readable file: {path or '<unset>'}")
        if exists:
            got = sha256(path)
            check(got == os.environ.get(pin) or allow, f"{var} sha256 {got[:16]}... matches pin {os.environ.get(pin, '')[:16]}..."
                  + (" (UNPINNED ALLOWED)" if allow and got != os.environ.get(pin) else ""))
    sys.path.insert(0, os.path.join(ROOT, "src"))
    try:
        pipeline = importlib.import_module("bankrec.pipeline")
        names = {f.__name__ for f in pipeline.MODULES}
        missing = [r for r in RESOLVERS if r not in names]
        check(not missing, f"all resolvers loaded{'' if not missing else ' (missing ' + ', '.join(missing) + ')'}")
        check(pipeline.LEDGER, "pipeline sees the ledger on")
    except Exception as e:  # noqa: BLE001
        check(False, f"import bankrec.pipeline ({type(e).__name__}: {e})")
    pq = os.path.join(ROOT, "data", "parquet", phase)
    for name in ("bank_transactions.parquet", "open_items_at_start.parquet"):
        check(os.path.isfile(os.path.join(pq, name)), f"phase data {os.path.relpath(os.path.join(pq, name), ROOT)}")
    check(os.path.isfile(os.path.join(ROOT, "data", "starter", "bankrec_score.py")), "official scorer data/starter/bankrec_score.py")
    out = os.path.join(ROOT, "out")
    os.makedirs(out, exist_ok=True)
    check(os.access(out, os.W_OK), "out/ writable")
    free = shutil.disk_usage(ROOT).free / 1e9
    check(free > 5, f"free disk {free:.1f} GB > 5 GB")
    for tool in ("shasum", "tar", "curl"):
        check(shutil.which(tool) is not None, f"command {tool}")
    if bad:
        print(f"PREFLIGHT FAILED ({len(bad)} problem(s)); do not submit")
        sys.exit(1)
    print("PREFLIGHT OK")


if __name__ == "__main__":
    main()
