# Token Co. | Close the Books: project write-up

## Submission text (1,460 characters)

Token Co. reconciles every bank movement of a multi-entity group: each line is matched to the invoices it settles, or
explained with a category, and every difference carries the right accounting exception.

Our pipeline is a precision-first cascade of 15 resolvers. Exact structural joins come first (direct debits, cards,
cheques, payment runs), then document evidence (remittance advices in csv/txt/pdf/xlsx, clauses printed on invoice PDFs),
then customer matching by name, IBAN and cited documents with exact subset search, discounts, FX and bank charges.
A shared ledger replays every accepted allocation, so later stages only see what is still open. Two small pinned models
help: TF-IDF + logistic regression for categories, LightGBM for unreferenced partial payments. No LLM runs at inference.

Two new resolvers target multi-invoice payments without references: c_ptail finds bundles of full invoices plus one
invoice paid at 40/50/60 %, and c_contig finds the only run of consecutive due invoices that sums to the cash. Both claim
only a unique solution and abstain otherwise.

Official FINAL: 94.45 (allocation F1 0.944, precision 0.988, recall 0.904; transaction accuracy 0.962; exception
macro-F1 0.918), covering all 220,256 bank lines. Clean time-separated holdouts: 94.58 (2025-H1), 94.33 (2024-H2).

## What is new

- **c_ptail**: about half of the unclaimed multi-invoice lines in TRAIN are "k invoices in full + one invoice at 0.4,
  0.5 or 0.6". Integer-cent meet-in-the-middle search over the payer's recently due invoices; abstains when a full
  subset also explains the cash or when several solutions exist. Holdouts +0.14 / +0.15.
- **c_contig**: when a contiguous run of the payer's open invoices in due-date order sums exactly to the payment, it is
  the right bundle 95-96 % of the time on the holdouts. Prefix sums make this linear-time. Holdouts +0.08 / +0.08.

## Validation

Every change was measured with the official scorer on two time-separated TRAIN windows, with models trained only on
earlier data and documents restricted to the current and earlier phases. Preflight checks the pinned model hashes and
the resolver order; the official validator and a coverage check run on every output.

## Limitations

- Large customer sweeps (more than 30 invoices) and ambiguous subsets remain the main source of missed allocations:
  several exact combinations fit and the data gives no evidence to choose.
- Allocation precision fell slightly (0.9890 to 0.9884) in exchange for recall.
- Not production-ready: synthetic data, a local machine (~30 GB peak memory) and thresholds fixed on TRAIN.

## Demo video script (60-90 s; real outputs only)

0:00 "Every month a finance team matches thousands of bank lines to invoices. One payment can cover twenty invoices
minus a credit note, short by a bank fee. We are Token Co."
0:12 Screen: the README pipeline diagram. "Fifteen resolvers in order: exact joins, remittance and PDF evidence,
customer matching, and a shared ledger so nothing is allocated twice."
0:30 Screen: terminal running `bash tools/run_candidate.sh` or the FINAL log (`PREFLIGHT OK`, resolver line counts,
`VALIDATION: OK`, `ok=True`, 220,256 / 220,256 lines).
0:45 "Our new resolvers find multi-invoice payments with no reference: full invoices plus one paid at half, or a run of
consecutive due invoices that sums to the cash, and they abstain when the answer is not unique."
1:00 Screen: FINAL result 94.45 and metrics table. "94.45 on FINAL, every bank line explained, precision 0.988."
1:15 "Precision first, evidence always: fewer hours closing the books, and every match explainable."
