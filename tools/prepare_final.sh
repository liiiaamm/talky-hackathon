#!/usr/bin/env bash
# 18:00 runbook: download FINAL, verify, extract, clean macOS junk, parquet, run the pipeline, validate. Does NOT submit.
#   bash tools/prepare_final.sh [SHA256]        (SHA256 from https://hackathon.usetalky.com/config.json, optional)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p data/_dl out
echo "== config.json (FINAL entry)"
curl -sS https://hackathon.usetalky.com/config.json | $PY -c "import json,sys; d=json.load(sys.stdin); print([x for x in d['downloads'] if x['id']=='final'])"
echo "== download"
curl -sSfL --retry 3 -C - -o data/_dl/bankrec-final.tar.gz https://hackathon.usetalky.com/data/v1/bankrec-final.tar.gz
if [ "${1:-}" != "" ]; then echo "$1  data/_dl/bankrec-final.tar.gz" | sha256sum -c; else sha256sum data/_dl/bankrec-final.tar.gz; fi
echo "== extract"
tar -xzf data/_dl/bankrec-final.tar.gz -C data 2>/dev/null || tar -xzf data/_dl/bankrec-final.tar.gz -C data
find data -name '._*' -type f -delete
ls data/final; ls data/final/masters_final 2>/dev/null || echo "WARNING: no masters_final"
echo "== parquet"
$PY eda/00_to_parquet.py | grep -E '^final' || true
echo "== pipeline"
cd src && ../.venv/bin/python -m bankrec.pipeline --phase final --out ../out/final.csv.gz
cd .. && ls -la out/final.csv.gz
echo "== sanity: distributions FINAL vs LIVE"
$PY - <<'EOF'
import polars as pl
for p in ["out/final.csv.gz"] + sorted(__import__("glob").glob("out/live*.csv.gz"))[-1:]:
    d = pl.read_csv(p, infer_schema_length=0)
    n = d["bank_txn_id"].n_unique()
    print(p, f"lines={n:,} rows={d.height:,} item_rows={(d['item_id'].fill_null('')!='').sum():,}")
    print("  categories:", d.filter(pl.col("category").fill_null("") != "")["category"].value_counts().sort("count", descending=True).head(8).rows())
    print("  exceptions:", d["exception"].fill_null("").value_counts().sort("count", descending=True).rows())
EOF
echo "READY: out/final.csv.gz  ->  python tools/submit.py FINAL out/final.csv.gz --i-am-sure -m '...'"
