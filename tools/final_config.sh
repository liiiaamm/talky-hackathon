# Validated INT6 configuration (LIVE s_daffac9d066561d822de = 93.17; clean TRAIN 2025-H1 93.62 / 2024-H2 93.29).
# Sourced by tools/run_candidate.sh and tools/prepare_final.sh BEFORE any Python process starts, so every interpreter
# inherits PYTHONHASHSEED=0 at startup (setting it inside Python is too late: the hash seed is fixed at launch).
export PYTHONHASHSEED=0
export BANKREC_LEDGER=1
export BANKREC_ORDER=remittance,pdf_clauses,ar_text,ar_exact,ar_multi,c_partial,c_discount,c_bank_charges,c_fx,ar_due_subset
unset BANKREC_SKIP BANKREC_AR_TEXT_CFG
# Models are local files (models/ and *.joblib are gitignored); their sha256 is pinned here and checked by tools/preflight.py.
# BANKREC_CAT_MODEL: the category model trained with --all on TRAIN (cat_all.joblib), the one used for the LIVE submission.
# c_partial: all labelled TRAIN data (TRAIN ends 2025-06-30, so any cutoff >= 2025-07-01 trains on identical rows).
export BANKREC_PARTIAL_MODEL="${BANKREC_PARTIAL_MODEL:-$PWD/models/c_partial_2025-07-01.joblib}"
export BANKREC_CAT_SHA256=b38a54f1f4af6efb006d18eeb29e92208e22ca7aca2ee116c9f33ef61e313220
export BANKREC_PARTIAL_SHA256=acfb6c2465ece2183e4b9006aa2b72034396f66b6a6ae296788eeaac5ccea7e5
