#!/usr/bin/env bash
# Global lock for heavy runs (pipeline / snapshot) shared by every agent on this machine (15 GB RAM).
#   tools/heavy.sh quick <cmd ...>   # e.g. BANKREC_SKIP=ar_text runs (~1.5 min, ~4 GB)
#   tools/heavy.sh full  <cmd ...>   # full pipeline incl. ar_text (~6 min, ~8 GB)
# One heavy run at a time; waits in line. Logs who holds it in /tmp/bankrec_locks/holder.
set -euo pipefail
L="${BANKREC_LOCK_DIR:-/tmp/bankrec_locks}"
mkdir -p "$L"
mode="${1:?quick|full}"; shift
# A parent (tools/prepare_final.sh) already holds the lock for the whole runbook: run directly, locking again would deadlock.
if [[ -n "${HEAVY_LOCK_HELD:-}" ]]; then
    echo "[heavy.sh] lock already held by the parent ($mode) at $(date +%H:%M:%S)" >&2
    exec "$@"
fi
exec 9>"$L/run.lock"
echo "[heavy.sh] waiting for the lock ($mode) ... holder: $(cat "$L/holder" 2>/dev/null || echo none)" >&2
# macOS has no flock(1): take the same flock(2) lock on the inherited fd with Python (held until the fd closes)
if command -v flock >/dev/null; then flock 9; else "${PYTHON_BIN:-python3}" -c 'import fcntl; fcntl.flock(9, fcntl.LOCK_EX)'; fi
echo "$$ $mode $(date +%H:%M:%S) $PWD :: $*" > "$L/holder"
echo "[heavy.sh] got the lock at $(date +%H:%M:%S)" >&2
set +e; "$@"
rc=$?
: > "$L/holder"
exit $rc
