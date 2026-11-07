# FINAL readiness checklist (18:00 – 19:30)

Primary candidate: **DC7**, branch `final/dc7`, commit `f1e6ec7` (LIVE 93.80).
Fallback 1: INT6 + C + gate, commit `82f4364` (local branch `local/int7-final`, Jack's machine).
Fallback 2: INT6, branch `local/int6-final`, commit `7b56c20`.
FINAL = **one scored submission per team**. Nothing is submitted without explicit team approval.

## 0. Before 18:00
- [ ] Checkout of `f1e6ec7` in its own directory (never reuse a rehearsal folder with symlinked `data/final`).
- [ ] Venv with `requirements.txt` installed; `python -c "import ahocorasick, lightgbm, sklearn, pypdf, openpyxl"` works.
- [ ] Models available and hashes match `tools/final_config.sh`:
      `cat_all.joblib` = `b38a54f1f4af6efb006d18eeb29e92208e22ca7aca2ee116c9f33ef61e313220`,
      `c_partial_2025-07-01.joblib` = `acfb6c2465ece2183e4b9006aa2b72034396f66b6a6ae296788eeaac5ccea7e5`
      (`shasum -a 256 <file>`).
- [ ] At least 8 GB free RAM and 5 GB free disk; no other heavy job running.

## 1. Run (from the repository root)
```bash
export PYTHON_BIN=/ABS/PATH/.venv/bin/python
export BANKREC_CAT_MODEL=/ABS/PATH/cat_all.joblib
export BANKREC_PARTIAL_MODEL=/ABS/PATH/c_partial_2025-07-01.joblib
bash tools/prepare_final.sh --skip-train 2>&1 | tee out/final_run.log
```
- [ ] Log shows `PREFLIGHT OK` and ends with `READY: out/final.csv.gz`.
- [ ] Exit code 0. Runtime noted (LIVE took ~187 s for the pipeline; allow up to ~10 min including download and parsing).

## 2. Inspect inputs (data checks)
`python <path>/final_checks.py "$PWD" final out/final_run.log` (read-only script; kept outside the repo by Jack). It must
report **0 FAIL**. Check in particular:
- [ ] `data/final/masters_final/` present and used.
- [ ] Every bank entity is in `entities.csv`, including the anticipated new entity **ZA01** (announced in POLICIES §6; not yet
      verified on FINAL data); account-id prefix equals the bank entity for every account
      (the pipeline derives the entity from the account-id prefix, e.g. `ZA01-…-ZAR`).
- [ ] FX rates exist for every non-EUR currency in the phase, **including ZAR**; last FX date ≥ last booking date.
- [ ] No expected table missing (payment runs, DD batches, cards, cheques, remittance index, PDF index, open items).
- [ ] Remittance advices: count by format (csv / txt / pdf / xlsx / new?) and share of files that produced parsed lines.
      **Any format below 90 %, or a format with 0 parsed rows, is a FAIL** (new FINAL layout).
- [ ] No resolver with zero output in the pipeline log.

## 3. Validate the output
```bash
"$PYTHON_BIN" tools/official_check.py final out/final.csv.gz
shasum -a 256 out/final.csv.gz
```
- [ ] `official format validation: ok=True`, `missing lines 0` (every FINAL bank line represented).
- [ ] No duplicate rows; over-allocated items in the same range as LIVE (63).
- [ ] Module line shares comparable with the LIVE run (remittance, pdf_clauses, ar_text, categories).
- [ ] Record: file name, rows, runtime, commit, model hashes, SHA-256 (in the team channel and the handoff).

## STOP conditions (do not submit; investigate)
- Preflight FAIL, non-zero exit, Python traceback, or no `READY` line.
- `final_checks.py` reports FAIL (missing masters / ZA01 / FX currency / table, remittance coverage < 90 % for a format).
- Official validation not ok, or missing lines > 0.
- Model hash mismatch, or a different commit than `f1e6ec7`.

## Rollback (never overwrite the validated candidate's output)
1. If the failure is **specific to DC7 code** (traceback in `c_evidence`, `c_duplicates`, `pdf_clauses` v2, or a DC7-only
   module with zero output): run fallback 1 (`82f4364`) in a **separate checkout** with the same data and the same checks,
   writing to a different file (e.g. `out/final_int7.csv.gz`).
2. If the failure is **common to all candidates** (schema, masters, parser): fallbacks will not help. Fix the data issue
   narrowly, re-run the checks, and keep the previous files.
3. If no validated DC7 file exists by **19:00**, start the fallback in parallel. Never submit an unvalidated file.

## Submission (only after explicit team approval)
- [ ] Approval recorded (who, when, which file and SHA-256).
- [ ] `python tools/submit.py FINAL out/final.csv.gz -m "..." --i-am-sure` (needs `HK_TOKEN` in the environment; never commit it).
- [ ] Record the submission ID. One submission only; no retries with other files.

## Separate deadline: 19:45
- [ ] Project video uploaded.
- [ ] README and write-up final; repository access confirmed for the jury (repository stays private unless the rules require otherwise).
