# How to run the 93.58 version (branch `integ/a-due-on-c`)

Diego's `feat/c-on-integ` (9760482) + Persona A's `ar_due_subset`. The defaults already contain the winning setup:
`BANKREC_LEDGER=1` and the order `remittance,pdf_clauses,ar_text,ar_exact,ar_multi,c_partial,c_discount,c_bank_charges,c_fx,ar_due_subset`.

| Window | Without ar_due_subset | With ar_due_subset |
|---|---|---|
| TRAIN 2025-H1 (categories model `categories_to2024.joblib`) | 92.24 | **93.58** |
| TRAIN 2024-H2, out of sample (`categories_to2024h1_research.joblib`) | 91.51 | **92.92** |

Pipeline with a free machine: ~315 s, peak RSS ~7.8 GB. **Never run two pipelines at once** (15 GB of RAM): use `tools/heavy.sh`.

## 1. `c_partial` model (`models/` is gitignored)

`c_partial` picks the newest `models/c_partial_<date>.joblib` whose date is <= the window start. Without one it claims nothing.

```bash
PYTHONPATH=src python -m bankrec.c_partial --train --to 2024-12-31   # for TRAIN 2025-H1
PYTHONPATH=src python -m bankrec.c_partial --train --to 2024-06-30   # for TRAIN 2024-H2
PYTHONPATH=src python -m bankrec.c_partial --train --to 2025-06-30   # LIVE / FINAL (all of TRAIN)
```

## 2. TRAIN evaluation

```bash
bash tools/heavy.sh full env PYTHONPATH=src POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  BANKREC_CAT_MODEL=$PWD/models/categories_to2024.joblib \
  python -u -m bankrec.pipeline --phase train --from 2025-01-01 --to 2025-06-30 --score --profile
```

## 3. LIVE

```bash
bash tools/heavy.sh full env PYTHONPATH=src POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  BANKREC_CAT_MODEL=$PWD/models/categories.joblib BANKREC_PARTIAL_MODEL=$PWD/models/c_partial_2025-06-30.joblib \
  python -u -m bankrec.pipeline --phase live --profile --out out/live.csv.gz
```

## What `ar_due_subset` does

It acts on AR receipts that no module claimed (they used to end up in `cat_rest`/`fallback`). For an identified payer, it looks
among their open invoices due in [booking - D, booking + 10] days, with D = 30, 45, 60, 90 and 120. Credit notes count by issue
date. It claims only when there is a single exact (to the cent) combination of >= 2 items, and abstains when the match is
ambiguous or more than one payer fits. It is the last AR module, right before the category classifier. On TRAIN it claims
~4,000 receipts per half year.
