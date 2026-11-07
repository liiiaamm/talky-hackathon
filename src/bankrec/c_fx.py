"""Entry point for Persona C's FX-receipt pass (c_exceptions.fx_receipts), so it can be placed on its own in BANKREC_ORDER.
Run it after the AR matchers: it only sees receipts in another currency (orig_amount set), which every AR matcher skips."""
from .c_exceptions import fx_receipts

RESOLVERS = [fx_receipts]
