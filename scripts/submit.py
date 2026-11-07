#!/usr/bin/env python3
"""Submit a file to the Talky hackathon and wait for the result (same flow as https://hackathon.usetalky.com/enviar).

Usage:  read -rs HK_TOKEN && export HK_TOKEN
        python scripts/submit.py LIVE out/live_v1.csv.gz
FINAL is ONE shot: the script asks for confirmation.
"""
import json
import os
import sys
import time

import requests

API = os.environ.get("HK_API", "https://hackathon.usetalky.com/api")
FINAL = {"scored", "received", "invalid", "error"}


def main(phase: str, path: str) -> None:
    if phase == "FINAL" and input("FINAL is your ONLY scored submission. Type FINAL to confirm: ").strip() != "FINAL":
        sys.exit("aborted")
    headers = {"Authorization": f"Bearer {os.environ['HK_TOKEN']}"}
    name = os.path.basename(path)
    r = requests.post(f"{API}/submissions/upload-url", headers=headers, json={"phase": phase, "filename": name}, timeout=30)
    if r.status_code != 201:
        sys.exit(f"upload-url failed: {r.status_code} {r.text}")
    up = r.json()
    sub_id = up["sub_id"]
    with open(path, "rb") as f:
        s3 = requests.post(up["upload"]["url"], data=up["upload"]["fields"], files={"file": (name, f)}, timeout=900)
    if s3.status_code not in (200, 201, 204):
        sys.exit(f"S3 upload failed: {s3.status_code} {s3.text[:500]}")
    r = requests.post(f"{API}/submissions/{sub_id}/score", headers=headers, json={}, timeout=30)
    if r.status_code >= 400:
        sys.exit(f"score failed: {r.status_code} {r.text}")
    sub = r.json()
    deadline = time.time() + 600
    while sub.get("status") not in FINAL and time.time() < deadline:
        time.sleep(5)   # the whole venue shares one IP: never poll faster
        sub = requests.get(f"{API}/submissions/{sub_id}", headers=headers, timeout=30).json()
    print(json.dumps(sub, indent=2, ensure_ascii=False))
    os.makedirs("out/results", exist_ok=True)
    with open(f"out/results/{phase}_{name}_{int(time.time())}.json", "w") as fh:
        json.dump(sub, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("LIVE", "FINAL"):
        sys.exit("usage: python scripts/submit.py LIVE|FINAL file.csv[.gz]  (with HK_TOKEN exported)")
    main(sys.argv[1], sys.argv[2])
