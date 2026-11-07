"""Late entry point for Persona C's SHORT_BANK_CHARGES resolver.

It runs AFTER the AR matching modules (ar_text / ar_exact / ar_multi) and only takes the cross-border receipts they leave: a receipt a few euros
short of one open invoice, which POLICIES §3 explains as the intermediary bank's fee when the shortfall is ≤ 60 EUR equivalent and ≤ 2 %
(SHORT_BANK_CHARGES, invoice fully cleared). Logic lives in c_exceptions.bank_charges;
this module only gives it its own name in BANKREC_ORDER.
"""
from .c_exceptions import bank_charges

RESOLVERS = [bank_charges]
