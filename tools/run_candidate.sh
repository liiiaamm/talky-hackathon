#!/usr/bin/env bash
# Produce a LIVE/FINAL prediction file with the team's validated INT6 configuration and check it. NEVER uploads or submits.
#   PYTHON_BIN=/path/to/venv/bin/python BANKREC_CAT_MODEL=/abs/path/cat_all.joblib \
#     bash tools/run_candidate.sh final out/final_candidate.csv.gz
# Configuration comes from tools/final_config.sh (sourced before any Python starts, so PYTHONHASHSEED=0 is inherited).
# Fails fast via tools/preflight.py (dependencies, resolvers, pinned model hashes, data, paths) and when the official
# validation fails. No --repair: an invalid file stops here for a human decision.
set -euo pipefail
cd "$(dirname "$0")/.."
PHASE="${1:?phase: live|final}"
OUT="${2:?output path}"
PY="${PYTHON_BIN:?set PYTHON_BIN to the python of a venv with requirements.txt installed}"
: "${BANKREC_CAT_MODEL:?set BANKREC_CAT_MODEL to the absolute path of cat_all.joblib}"
[[ -e "$OUT" ]] && { echo "refusing to overwrite $OUT" >&2; exit 1; }
export BANKREC_CAT_MODEL
# shellcheck source=tools/final_config.sh
source tools/final_config.sh
echo "== $(date +%T) commit $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src || echo -dirty)  phase=$PHASE"
echo "== $(date +%T) preflight"
"$PY" tools/preflight.py "$PHASE"
"$PY" -m unittest tests.test_exceptions_b.TestEnvironment
echo "== $(date +%T) pipeline"
mkdir -p "$(dirname "$OUT")"
OUT_ABS="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
(cd src && /usr/bin/time -l "$PY" -m bankrec.pipeline --phase "$PHASE" --out "$OUT_ABS") 2>&1 | \
  grep -E "loaded|lines .* rows|VALIDATION|wrote|^total|maximum resident" || true
[[ -s "$OUT_ABS" ]] || { echo "pipeline produced no file" >&2; exit 1; }
echo "== $(date +%T) official checks"
"$PY" tools/official_check.py "$PHASE" "$OUT_ABS"
shasum -a 256 "$OUT_ABS"
echo "READY: $OUT_ABS  (NOT submitted; submission needs explicit approval)"
