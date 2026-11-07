#!/usr/bin/env bash
# Download and prepare FINAL; this script never submits.
# Uses internal heavy.sh locks: do not wrap the whole script in heavy.sh.
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
mkdir -p data/_dl out
if [[ "$skip_download" == 0 ]]; then
    echo "== download"
    curl -sSfL --retry 3 -C - -o data/_dl/bankrec-final.tar.gz https://hackathon.usetalky.com/data/v1/bankrec-final.tar.gz
    if [[ -n "$sha256" ]]; then
        echo "$sha256  data/_dl/bankrec-final.tar.gz" | sha256sum -c
    else
        sha256sum data/_dl/bankrec-final.tar.gz
    fi
    echo "== extract"
    tar -xzf data/_dl/bankrec-final.tar.gz -C data
fi
[[ -d data/final && ! -L data/final ]] || { echo "Need a real data/final directory" >&2; exit 1; }
[[ -d data/final/masters_final ]] || { echo "Missing data/final/masters_final" >&2; exit 1; }
find data/final -name '._*' -type f -delete
# Invalidate only FINAL caches when new raw data was downloaded.
if [[ "$skip_download" == 0 ]]; then
    rm -f data/parquet/final/remittance_lines.parquet data/parquet/final/pdf_clauses.parquet
fi
echo "== parquet (FINAL only)"
bash tools/heavy.sh quick "$PY" eda/00_to_parquet.py --phase final
if [[ "$skip_train" == 0 ]]; then
    echo "== classifier (all TRAIN, saved separately from development model)"
    (cd src && bash ../tools/heavy.sh quick "$PY" -m bankrec.categories --train --all)
fi
[[ -f "$BANKREC_CAT_MODEL" ]] || { echo "Missing classifier: $BANKREC_CAT_MODEL" >&2; exit 1; }
echo "== parse FINAL documents with bounded workers"
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
echo "== pipeline"
(cd src && bash ../tools/heavy.sh full "$PY" -m bankrec.pipeline --phase final --profile --out ../out/final.csv.gz)
echo "== size and distributions"
"$PY" - <<'PY'
import gzip
import os
import polars as pl
p = 'out/final.csv.gz'
with gzip.open(p, 'rb') as f:
    nbytes = sum(len(block) for block in iter(lambda: f.read(1024 * 1024), b''))
d = pl.read_csv(p, infer_schema_length=0)
if nbytes > 60_000_000 or d.height > 3_000_000:
    raise SystemExit(f'Submission exceeds limits: {nbytes} bytes, {d.height} rows')
print(f'{p}: lines={d["bank_txn_id"].n_unique():,} rows={d.height:,} uncompressed_bytes={nbytes:,}')
print('categories:', d.filter(pl.col('category').fill_null('') != '')['category'].value_counts().sort('count', descending=True).rows())
print('exceptions:', d['exception'].fill_null('').value_counts().sort('count', descending=True).rows())
PY
sha256sum out/final.csv.gz
echo "READY: out/final.csv.gz (not submitted)"
