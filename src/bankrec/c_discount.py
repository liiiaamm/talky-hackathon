"""Entry point for Persona C's 1:N early-payment-discount pass (c_exceptions.discount_receipts).
Place it after the AR matchers (ar_text / ar_exact / ar_multi): it only re-tries the receipts they left, on a view of the open
items where untouched discount-term invoices are valued at their discounted amount."""
from .c_exceptions import discount_receipts

RESOLVERS = [discount_receipts]
