"""Early entry point for Persona C's SHORT_BANK_CHARGES resolver (c_evidence.bank_charges).

Place it BEFORE the generic AR matchers so a cross-border receipt short by an intermediary fee is not read as a PARTIAL payment:

    BANKREC_ORDER=remittance,pdf_clauses,c_bank_charges_early,ar_text,ar_exact,ar_multi,c_discount,c_fx,c_duplicates

It only claims a receipt when the statement marks it as cross-border (XBCT, N43 code 13 or a SWIFT charges field), the payer's
country differs from the entity's, and exactly one open invoice of the best-evidenced payer fits (POLICIES §3: shortfall
<= 60 EUR equivalent and <= 2 % of the item, not that invoice's early-payment discount). Otherwise it abstains.
"""
from .c_evidence import bank_charges

RESOLVERS = [bank_charges]
