"""Early entry point for Persona C's SHORT_BANK_CHARGES resolver.

It must run before the AR matching modules (ar_text / ar_exact / ar_adjusted): those see a cross-border receipt a few euros
short of an invoice and label it PARTIAL, while POLICIES §3 says a shortfall ≤ 60 EUR equivalent and ≤ 2 % on a cross-border
receipt is the intermediary bank's fee (SHORT_BANK_CHARGES, invoice fully cleared). Logic lives in c_exceptions.bank_charges;
this module only gives it its own name in BANKREC_ORDER. The late resolvers (duplicates, overpayments) stay in c_exceptions.
"""
from .c_exceptions import bank_charges

RESOLVERS = [bank_charges]
