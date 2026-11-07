"""Last entry point for Persona C's OVERPAYMENT resolver (c_evidence.overpayments).

Place it LAST among the matchers (after ar_due_subset, before the catch-all categories): it only looks at receipts nobody
explained, and claims one when it exceeds a single open invoice of the best-evidenced payer by a deliberate top-up (a 1-2-5
"nice number" in currency units or in percent of the invoice) and no combination of the payer's open documents explains the
cash exactly. Otherwise it abstains.

    BANKREC_ORDER=...,c_fx,c_duplicates,ar_due_subset,c_overpayment
"""
from .c_evidence import overpayments

RESOLVERS = [overpayments]
