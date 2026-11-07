"""A1: unique small-structure matching (pairs, oldest-window sets netted with one credit note). Synthetic data only."""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from matching.customer_matcher import CustomerMatcher, ItemStore  # noqa: E402
from matching.party_resolver import PartyResolver  # noqa: E402
from matching.subset_sum import mitm_all, pair_solutions, windowed_solutions  # noqa: E402

CUST = pd.DataFrame([("C1", "ES01", "Riegos Oasis S.L.", "", "ES", "", "")],
                    columns=["customer_id", "entity", "name", "trade_name", "country", "iban", "parent_customer_id"])
SUPP = pd.DataFrame(columns=["supplier_id", "entity", "name", "country", "iban"])
AMTS = [123457, 98765, 45613, 777701, 31234, 560019, 88801, 241117, 19903, 402261, 65537, 310009, 150031, 72211, 509903, 28817]


def build(extra=()):
    rows = []
    for k, a in enumerate(AMTS):
        rows.append((f"ARI-2025-{k+1:07d}", "ARI", "ES01", "C1", f"FV25-{k+1:07d}", f"2025-01-{k+1:02d}", "EUR", a / 100, "T30", "", "", a))
    rows += list(extra)
    cols = ["item_id", "kind", "entity", "party_id", "document_number", "issue_date", "currency", "total_amount", "payment_terms", "po",
            "related_document", "open_cents"]
    return pd.DataFrame(rows, columns=cols)


class Ledger:
    def __init__(self, o):
        self.o = np.asarray(o, dtype=np.int64).copy()

    def open(self, i):
        return int(self.o[i])


def line(amount):
    return SimpleNamespace(bank_txn_id="BT-1", entity="ES01", amount=str(amount), currency="EUR", orig_amount="", orig_currency="",
                           text="RIEGOS OASIS S.L. PAGO", ref1="", ref2="", remittance_info="", end_to_end_id="", counterparty_name="",
                           counterparty_account="", booking_date="2025-03-05")


def run(items, amount, cfg=None):
    c = {"use_window": True, "use_subset": False}
    c.update(cfg or {})
    m = CustomerMatcher(ItemStore(items), PartyResolver(CUST, SUPP), {"ES01": {"ES01"}}, c)
    return m.match(line(amount), Ledger(items.open_cents))


def ids(p):
    return sorted(a["item_id"] for a in p.allocations)


class TestPrimitives(unittest.TestCase):
    def test_mitm_all(self):
        self.assertEqual(sorted(mitm_all([5, 3, 2, 8], 10)), [(0, 1, 2), (2, 3)])
        self.assertEqual(mitm_all([5, 3], 100), [])

    def test_pairs(self):
        self.assertEqual(pair_solutions([5, 3, 2, 8], 10), [(2, 3)])
        self.assertEqual(pair_solutions([5], 5), [])

    def test_windowed_with_credit_note(self):
        sols = windowed_solutions([100, 200, 300, 400], [50], 450)   # 100+400 or 200+300-50?  (+50 -> 500 = 100+400, 200+300)
        self.assertTrue(all(s[1] == 0 for s in sols if s[1] is not None))
        self.assertEqual(sols[0][0], 3)


class TestWindowMatcher(unittest.TestCase):
    def test_unique_pair_claimed(self):
        p = run(build(), (AMTS[2] + AMTS[7]) / 100)
        self.assertEqual(p.match_method, "window")
        self.assertEqual(ids(p), ["ARI-2025-0000003", "ARI-2025-0000008"])

    def test_oldest_window_set_netted_with_related_credit_note(self):
        cn = ("ARC-2025-0000101", "ARC", "ES01", "C1", "AB25-0000101", "2025-02-01", "EUR", 600.00, "", "", "FV25-0000002", 60000)
        items = build([cn])
        paid = AMTS[0] + AMTS[1] + AMTS[3] + AMTS[4] + AMTS[6] - 60000
        p = run(items, paid / 100)
        self.assertEqual(p.match_method, "window")
        self.assertEqual(ids(p), ["ARC-2025-0000101", "ARI-2025-0000001", "ARI-2025-0000002", "ARI-2025-0000004", "ARI-2025-0000005",
                                  "ARI-2025-0000007"])

    def test_credit_note_not_related_to_set_abstains(self):
        cn = ("ARC-2025-0000101", "ARC", "ES01", "C1", "AB25-0000101", "2025-02-01", "EUR", 600.00, "", "", "FV25-0000016", 60000)
        paid = AMTS[0] + AMTS[1] + AMTS[3] + AMTS[4] + AMTS[6] - 60000
        p = run(build([cn]), paid / 100)
        self.assertEqual(p.match_method, "abstain")

    def test_ambiguous_pairs_abstain(self):
        items = build()
        items.loc[len(items)] = ("ARI-2025-0000099", "ARI", "ES01", "C1", "FV25-0000099", "2025-01-20", "EUR", AMTS[0] / 100, "T30", "", "", AMTS[0])
        items.loc[len(items)] = ("ARI-2025-0000098", "ARI", "ES01", "C1", "FV25-0000098", "2025-01-21", "EUR", AMTS[1] / 100, "T30", "", "", AMTS[1])
        # AMTS[0]+AMTS[2] is hit by two pairs? (0,2) and (99,2): same amount -> ambiguous
        p = run(items, (AMTS[0] + AMTS[2]) / 100)
        self.assertEqual(p.match_method, "abstain")

    def test_flag_off(self):
        p = run(build(), (AMTS[2] + AMTS[7]) / 100, {"use_window": False})
        self.assertEqual(p.match_method, "abstain")


if __name__ == "__main__":
    unittest.main()
