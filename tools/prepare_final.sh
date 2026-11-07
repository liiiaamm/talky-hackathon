#!/usr/bin/env bash
# Download and prepare FINAL; this script never submits.
# Takes the global heavy lock once (after extraction) for all heavy steps; do not wrap it in heavy.sh. Re-run after a failure with
#   bash tools/prepare_final.sh <SHA256>      (download is skipped when the tarball already matches) or --skip-download.
# Offline rehearsal: PYTHON_BIN=/absolute/python BANKREC_CAT_MODEL=/absolute/model \
#   bash tools/prepare_final.sh --skip-download --skip-train
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON_BIN:-$PWD/.venv/bin/python}"
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export BANKREC_CAT_MODEL="${BANKREC_CAT_MODEL:-$PWD/out/final_categories.joblib}"
# Relative model paths must keep their meaning when the pipeline runs from src/.
BANKREC_CAT_MODEL="$("$PY" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$BANKREC_CAT_MODEL")"
export BANKREC_CAT_MODEL
# Validated INT6 configuration (PYTHONHASHSEED=0, ledger, order, pinned model hashes), exported before any later Python
# starts so every step inherits it. The pins expect the LIVE category model: run with --skip-train and
# BANKREC_CAT_MODEL=/abs/path/cat_all.joblib (a freshly trained model fails preflight unless BANKREC_ALLOW_UNPINNED_MODELS=1).
# shellcheck source=tools/final_config.sh
source tools/final_config.sh
skip_download=0
skip_train=0
sha256=""
for arg in "$@"; do
    case "$arg" in
        --skip-download) skip_download=1 ;;
        --skip-train) skip_train=1 ;;
        --help) sed -n '2,4p' "$0"; exit 0 ;;
        --*) echo "Unknown option: $arg" >&2; exit 2 ;;
        *) if [[ -n "$sha256" ]]; then echo "Only one SHA256 is accepted" >&2; exit 2; fi; sha256="$arg" ;;
    esac
done
if [[ "$skip_train" == 0 && "${BANKREC_ALLOW_UNPINNED_MODELS:-}" != 1 ]]; then
    echo "The validated configuration pins cat_all.joblib: re-run with --skip-train and BANKREC_CAT_MODEL=/abs/path/cat_all.joblib" >&2
    echo "(or set BANKREC_ALLOW_UNPINNED_MODELS=1 for an approved retrain)" >&2; exit 2
fi
mkdir -p data/_dl out
T0=$SECONDS
step() { echo "== [$(date +%H:%M:%S) +$((SECONDS - T0))s] $*"; }
if [[ "$skip_download" == 0 ]]; then
    # A rehearsal mount (symlinks into data/live) must never receive the real tarball: tar would write through it.
    if [[ -e data/final ]] && [[ -L data/final || -n "$(find data/final -maxdepth 2 -type l -print -quit)" ]]; then
        echo "data/final contains symlinks (rehearsal mount?): move it away before extracting FINAL" >&2; exit 1
    fi
    tgz=data/_dl/bankrec-final.tar.gz
    if [[ -n "$sha256" && -f "$tgz" ]] && echo "$sha256  $tgz" | shasum -a 256 -c --status; then
        step "download skipped: $tgz already matches the SHA256"
    else
        step "download"
        curl -sSfL --retry 3 --retry-delay 5 -C - -o "$tgz" https://hackathon.usetalky.com/data/v1/bankrec-final.tar.gz
    fi
    if [[ -n "$sha256" ]]; then
        echo "$sha256  data/_dl/bankrec-final.tar.gz" | shasum -a 256 -c
    else
        shasum -a 256 data/_dl/bankrec-final.tar.gz
    fi
    step "extract"
    tar -tzf data/_dl/bankrec-final.tar.gz 2>/dev/null | grep -v '/\._\|^\._' | awk -F/ '{print $1}' | sort | uniq -c
    tar -xzf data/_dl/bankrec-final.tar.gz -C data 2> >(grep -v 'Ignoring unknown extended header' >&2)
fi
[[ -d data/final && ! -L data/final ]] || { echo "Need a real data/final directory" >&2; exit 1; }
# Take the global heavy lock ONCE for every remaining step: nobody can cut in between parquet, parsing and pipeline.
# The inner heavy.sh calls see HEAVY_LOCK_HELD and do not lock again (no deadlock).
if [[ -z "${HEAVY_LOCK_HELD:-}" ]]; then
    mkdir -p "${BANKREC_LOCK_DIR:-/tmp/bankrec_locks}"
    exec 8>"${BANKREC_LOCK_DIR:-/tmp/bankrec_locks}"/run.lock
    step "waiting for the global lock ... holder: $(cat "${BANKREC_LOCK_DIR:-/tmp/bankrec_locks}"/holder 2>/dev/null || true)"
    # macOS has no flock(1): take the same flock(2) lock on the inherited fd with Python (held until the fd closes)
    if command -v flock >/dev/null; then flock 8; else "$PY" -c 'import fcntl; fcntl.flock(8, fcntl.LOCK_EX)'; fi
    echo "$$ final-runbook $(date +%H:%M:%S) $PWD :: prepare_final.sh $*" > "${BANKREC_LOCK_DIR:-/tmp/bankrec_locks}"/holder
    trap ': > "${BANKREC_LOCK_DIR:-/tmp/bankrec_locks}"/holder' EXIT
    export HEAVY_LOCK_HELD=1
    step "got the global lock"
fi
[[ -d data/final/masters_final ]] || { echo "Missing data/final/masters_final" >&2; exit 1; }
# macOS AppleDouble junk: inside final/ and at the top level (the LIVE tarball shipped data/._live).
find data/final -name '._*' -type f -delete
find data -maxdepth 1 -name '._*' -type f -delete
# Invalidate only FINAL caches when new raw data was downloaded.
# The whole data/parquet/final is derived from data/final: drop it so no table or cache can survive from a rehearsal.
if [[ "$skip_download" == 0 ]]; then
    [[ ! -L data/parquet/final ]] || { echo "data/parquet/final is a symlink: refusing to clear it" >&2; exit 1; }
    rm -rf data/parquet/final
fi
echo "junk left: $(find data/final -name '._*' | wc -l)  files: $(find data/final -type f | wc -l)  masters_final: $(ls data/final/masters_final | tr '\n' ' ')"
step "parquet (FINAL only)"
bash tools/heavy.sh quick "$PY" eda/00_to_parquet.py --phase final
if [[ "$skip_train" == 0 ]]; then
    step "classifier (all TRAIN, saved separately from development model)"
    (cd src && bash ../tools/heavy.sh quick "$PY" -m bankrec.categories --train --all)
fi
[[ -f "$BANKREC_CAT_MODEL" ]] || { echo "Missing classifier: $BANKREC_CAT_MODEL" >&2; exit 1; }
step "parse FINAL documents with bounded workers"
(cd src && bash ../tools/heavy.sh quick "$PY" - <<'PY'
import os
from bankrec.data import DATA, PQ
from bankrec.remittance_parse import parse_phase
from bankrec.pdf_clauses import parse_phase as parse_pdf_clauses
workers = int(os.environ.get("BANKREC_PARSE_WORKERS", "4"))
if os.path.isdir(os.path.join(DATA, 'final', 'remittance_advices')):
    path = os.path.join(PQ, 'final', 'remittance_lines.parquet')
    if not os.path.exists(path):
        parse_phase('final', workers=workers).write_parquet(path)
parse_pdf_clauses('final', workers=workers)
PY
)
step "preflight (configuration, dependencies, resolvers, pinned models, data, paths)"
"$PY" tools/preflight.py final
step "pipeline"
# Never let a stale file look READY after a failed run. Emergency: BANKREC_PIPELINE_ARGS="--safe --repair".
rm -f out/final.csv.gz
# shellcheck disable=SC2086
(cd src && bash ../tools/heavy.sh full "$PY" -m bankrec.pipeline --phase final --profile ${BANKREC_PIPELINE_ARGS:-} --out ../out/final.csv.gz)
step "size and distributions (vs ${BANKREC_REF:-out/live.csv.gz} if present)"
BANKREC_REF="${BANKREC_REF:-out/live.csv.gz}" "$PY" - <<'PY'
import gzip
import os
import polars as pl
p = 'out/final.csv.gz'
with gzip.open(p, 'rb') as f:
    nbytes = sum(len(block) for block in iter(lambda: f.read(1024 * 1024), b''))
d = pl.read_csv(p, infer_schema_length=0)
if nbytes > 60_000_000 or d.height > 3_000_000:
    raise SystemExit(f'Submission exceeds limits: {nbytes} bytes, {d.height} rows')
bank = pl.read_parquet('data/parquet/final/bank_transactions.parquet', columns=['bank_txn_id', 'entity'])
missing = bank.join(d.select('bank_txn_id').unique(), on='bank_txn_id', how='anti').height
if missing:
    raise SystemExit(f'{missing} FINAL bank lines are not in {p}')
print(f'{p}: lines={d["bank_txn_id"].n_unique():,} rows={d.height:,} uncompressed_bytes={nbytes:,} (limits 60 MB / 3 M rows)')


def summary(df):
    n = df['bank_txn_id'].n_unique()
    line = df.group_by('bank_txn_id').agg((pl.col('item_id').fill_null('') != '').any().alias('it'),
                                         pl.col('category').fill_null('').max().alias('cat'))
    kind = line.select(pl.when(pl.col('it')).then(pl.lit('<ITEMS>')).otherwise(pl.col('cat')).alias('k'))['k'].value_counts()
    exc = df.filter(pl.col('exception').fill_null('') != '')['exception'].value_counts()
    return n, kind.with_columns((pl.col('count') / n * 100).round(2).alias('pct')), exc


ref = os.environ.get('BANKREC_REF')
ref = ref if ref and os.path.exists(ref) and os.path.abspath(ref) != os.path.abspath(p) else None
n, kind, exc = summary(d)
if ref:
    rn, rkind, rexc = summary(pl.read_csv(ref, infer_schema_length=0))
    kind = kind.join(rkind.rename({'count': 'ref_count', 'pct': 'ref_pct'}), on='k', how='full', coalesce=True)
    exc = exc.join(rexc.rename({'count': 'ref_count'}), on='exception', how='full', coalesce=True)
    print(f'reference {ref}: lines={rn:,}')
with pl.Config(tbl_rows=60, tbl_hide_dataframe_shape=True):
    print('per bank line (share of lines):')
    print(kind.sort('count', descending=True, nulls_last=True))
    print('exceptions (rows):')
    print(exc.sort('count', descending=True, nulls_last=True))
    print('bank lines per entity:')
    print(bank['entity'].value_counts().sort('entity'))
PY
shasum -a 256 out/final.csv.gz
step "READY: out/final.csv.gz (not submitted)"
