#!/usr/bin/env python3
"""Submit a file to the Talky hackathon and wait for the result (stdlib only).

Same 4-step flow as the official script on the Submit page (upload-url -> S3 -> score -> poll), plus:
  * local sanity checks BEFORE reserving a slot (asking for the upload URL already spends one),
  * FINAL requires --i-am-sure (one scored FINAL submission per team, no re-submits),
  * every result is appended to submissions/log.jsonl (commit it so the team sees the history).

Usage:
    read -rs HK_TOKEN && export HK_TOKEN
    python tools/submit.py LIVE out/live.csv.gz -m "baseline deterministic"
    python tools/submit.py FINAL out/final.csv.gz -m "..." --i-am-sure

Limits (from the site): 1 submission every 180 s, 20 per day per phase, <= 60 MB uncompressed, <= 3 M rows.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

API = "https://hackathon.usetalky.com/api"
FINAL_STATES = {"scored", "received", "invalid", "error"}
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(ROOT, "submissions", "log.jsonl")
COLS = {"bank_txn_id", "item_id", "amount", "category", "exception"}


def http(method: str, url: str, token: str | None = None, body: dict | None = None,
         data: bytes | None = None, headers: dict | None = None, timeout: int = 60):
    h = {"Accept": "application/json", **(headers or {})}
    if token:
        h["Authorization"] = f"Bearer {token}"
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(txt) if txt.strip().startswith(("{", "[")) else txt)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def multipart(fields: dict, name: str, payload: bytes) -> tuple[bytes, str]:
    """S3 presigned POST: every field first, the file last."""
    b = uuid.uuid4().hex
    buf = io.BytesIO()
    for k, v in fields.items():
        buf.write(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    buf.write(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
              f"Content-Type: application/octet-stream\r\n\r\n".encode())
    buf.write(payload)
    buf.write(f"\r\n--{b}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={b}"


def local_checks(path: str, phase: str) -> dict:
    raw = open(path, "rb").read()
    text = gzip.decompress(raw) if path.endswith(".gz") else raw
    mb = len(text) / 1e6
    rows = list(csv.DictReader(io.StringIO(text.decode("utf-8-sig"))))
    errs = []
    if mb > 60:
        errs.append(f"{mb:.1f} MB uncompressed > 60 MB")
    if len(rows) > 3_000_000:
        errs.append(f"{len(rows):,} rows > 3,000,000")
    if not rows or set(rows[0].keys()) != COLS:
        errs.append(f"header must be exactly {sorted(COLS)}")
    covered = {r["bank_txn_id"] for r in rows} if rows else set()
    bank = os.path.join(ROOT, "data", phase.lower(), "bank_transactions.csv.gz")
    missing = None
    if os.path.exists(bank) and rows:
        with gzip.open(bank, "rt", encoding="utf-8") as fh:
            ids = {r["bank_txn_id"] for r in csv.DictReader(fh)}
        missing = len(ids - covered)
        extra = len(covered - ids)
        if extra:
            errs.append(f"{extra:,} bank_txn_id not in {phase}")
        if missing:
            print(f"WARNING: {missing:,} of {len(ids):,} bank lines missing (they count as wrong)")
    if errs:
        sys.exit("local checks failed, NOT submitting (no slot spent):\n  " + "\n  ".join(errs))
    return {"rows": len(rows), "mb": round(mb, 1), "lines_covered": len(covered), "missing_lines": missing,
            "sha256": hashlib.sha256(raw).hexdigest()[:16], "payload": raw}


def git_rev() -> str:
    try:
        rev = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", ROOT, "diff", "--quiet", "HEAD"]) != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["LIVE", "FINAL"])
    ap.add_argument("file")
    ap.add_argument("-m", "--message", default="", help="what changed (goes to submissions/log.jsonl)")
    ap.add_argument("--i-am-sure", action="store_true", help="required for FINAL")
    a = ap.parse_args()

    token = os.environ.get("HK_TOKEN")
    if not token:
        sys.exit("HK_TOKEN not set:  read -rs HK_TOKEN && export HK_TOKEN")
    if a.phase == "FINAL" and not a.i_am_sure:
        sys.exit("FINAL is ONE scored submission per team, no re-submits. Re-run with --i-am-sure.")

    info = local_checks(a.file, a.phase)
    payload = info.pop("payload")
    print(f"local checks OK: {info}")
    name = os.path.basename(a.file)

    # 1) Reserve a slot (this already spends one) and get the presigned S3 upload
    st, up = http("POST", f"{API}/submissions/upload-url", token, {"phase": a.phase, "filename": name})
    if st != 201:
        sys.exit(f"upload-url failed: {st} {up}")
    sub_id = up["sub_id"]
    print(f"slot reserved: {sub_id}")

    # 2) Upload to S3
    body, ctype = multipart(up["upload"]["fields"], name, payload)
    st, resp = http("POST", up["upload"]["url"], data=body, headers={"Content-Type": ctype}, timeout=900)
    if st not in (200, 201, 204):
        sys.exit(f"S3 upload failed: {st} {str(resp)[:500]}")

    # 3) Queue scoring
    st, sub = http("POST", f"{API}/submissions/{sub_id}/score", token, {})
    if st >= 400:
        sys.exit(f"score failed: {st} {sub}")

    # 4) Poll every 5 s (the whole venue shares one IP), up to 10 min
    deadline = time.time() + 600
    while (not isinstance(sub, dict) or sub.get("status") not in FINAL_STATES) and time.time() < deadline:
        time.sleep(5)
        _, sub = http("GET", f"{API}/submissions/{sub_id}", token)
    print(json.dumps(sub, indent=2, ensure_ascii=False))

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    rec = {"ts": dt.datetime.now().isoformat(timespec="seconds"), "phase": a.phase, "sub_id": sub_id, "file": name,
           "git": git_rev(), "message": a.message, **info,
           "status": sub.get("status") if isinstance(sub, dict) else None,
           "score": sub.get("score") if isinstance(sub, dict) else None,
           "metrics": sub.get("metrics") if isinstance(sub, dict) else None,
           "errors": sub.get("errors") if isinstance(sub, dict) else None}
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"logged to {os.path.relpath(LOG, ROOT)}  ->  score {rec['score']}")


if __name__ == "__main__":
    main()
