"""Late entry point for Persona C's DUPLICATE_PAYMENT resolver (c_evidence.duplicates).

Place it AFTER every matcher: it only looks at lines nobody explained, and claims one that repeats the total of an item that was
open at the window start and is already settled (shared ledger), or whose earlier same-amount, same-payer line an earlier
resolver accepted (POLICIES §3). It abstains when the payer has several open items of that amount (same-amount FIFO, POLICIES §5.6).
"""
from .c_evidence import duplicates

RESOLVERS = [duplicates]
