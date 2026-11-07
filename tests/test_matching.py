"""Person B unit tests on synthetic fixtures (no competition data).  python -m unittest discover -s tests"""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from matching.customer_matcher import CustomerMatcher, ItemStore  # noqa: E402
from matching.normalize import doc_keys, service_month, text_reference_tokens  # noqa: E402
from matching.party_resolver import PartyResolver  # noqa: E402
from matching.subset_sum import find_subsets  # noqa: E402

CUST = pd.DataFrame([
    # customer_id, entity, name, trade_name, country, iban, parent
    ("C1", "ES01", "Riegos Oasis S.L.", "", "ES", "ES9100000000000000000001", ""),
    ("C2", "ES01", "Riegos Oasis Norte S.L.", "", "ES", "", ""),
    ("C3", "ES01", "Ibáñez Riegos S.Coop.", "Aguas Ibañez", "ES", "", ""),
    ("C4", "MX01", "Albercas del Golfo S.A. de C.V.", "", "MX", "", ""),
    ("P1", "ES01", "Grupo Hotelero Sol S.A.", "", "ES", "", ""),
    ("S1", "ES01", "Hotel Sol Playa S.L.", "", "ES", "", "P1"),
], columns=["customer_id", "entity", "name", "trade_name", "country", "iban", "parent_customer_id"])
SUPP = pd.DataFrame(columns=["supplier_id", "entity", "name", "country", "iban"])
TERMS = pd.DataFrame([("T30", "", ""), ("D2N30", "2.00", "10")], columns=["code", "early_payment_discount_pct", "discount_days"])


def items(rows):
    cols = ["item_id", "kind", "entity", "party_id", "document_number", "issue_date", "currency", "total_amount", "payment_terms", "po", "open_cents"]
    return pd.DataFrame(rows, columns=cols)


ITEMS = items([
    ("ARI-2025-0000001", "ARI", "ES01", "C1", "FV25-0100001", "2025-01-10", "EUR", "1210.00", "T30", "", 121000),
    ("ARI-2025-0000002", "ARI", "ES01", "C1", "FV25-0100002", "2025-01-12", "EUR", "605.00", "T30", "", 60500),
    ("ARC-2025-0000003", "ARC", "ES01", "C1", "AB25-0000003", "2025-01-15", "EUR", "145.20", "", "", 14520),
    ("ARI-2025-0000004", "ARI", "ES01", "C1", "FV25-0100004", "2025-01-20", "EUR", "2330.00", "T30", "", 233000),
    ("ARI-2025-0000005", "ARI", "ES01", "C2", "FV25-0100005", "2025-01-11", "EUR", "605.00", "T30", "", 60500),
    ("ARI-2025-0000006", "ARI", "ES01", "C3", "FV25-0100006", "2025-02-01", "EUR", "500.00", "D2N30", "", 50000),
    ("ARI-2025-0000007", "ARI", "MX01", "C4", "F-7", "2025-02-01", "MXN", "10000.00", "T30", "", 1000000),
    ("ARI-2025-0000008", "ARI", "ES01", "S1", "FV25-0100008", "2025-02-01", "EUR", "777.00", "T30", "", 77700),
    ("ARI-2025-0000009", "ARI", "ES01", "C3", "FV25-0100009", "2025-03-01", "EUR", "300.00", "T30", "", 30000),
    ("ARI-2025-0000010", "ARI", "ES01", "C3", "FV25-0100010", "2025-04-01", "EUR", "300.00", "T30", "", 30000),
])


class Ledger:
    def __init__(self, o):
        self.o = np.asarray(o, dtype=np.int64).copy()

    def open(self, i):
        return int(self.o[i])


def line(amount, text="", date="2025-02-05", entity="ES01", iban="", name="", ccy="EUR", orig_amt="", orig_ccy=""):
    return SimpleNamespace(bank_txn_id="BT-1", entity=entity, amount=str(amount), currency=ccy, orig_amount=orig_amt,
                           orig_currency=orig_ccy, text=text, ref1="", ref2="", remittance_info="", end_to_end_id="",
                           counterparty_name=name, counterparty_account=iban, booking_date=date)


def matcher(cfg=None):
    store = ItemStore(ITEMS)
    m = CustomerMatcher(store, PartyResolver(CUST, SUPP), {"ES01": {"ES01"}, "MX01": {"MX01"}}, cfg, terms=TERMS)
    return m, Ledger(ITEMS.open_cents)


def ids(p):
    return sorted(a["item_id"] for a in p.allocations)


class TestNormalize(unittest.TestCase):
    def test_doc_keys(self):
        k = doc_keys("FV25-0108480")
        self.assertEqual(k.strong, "FV250108480")
        self.assertEqual(k.digits, "250108480")
        self.assertIn("108480", k.tails)

    def test_reference_tokens_keep_joined_and_digit_runs(self):
        strong, digit = text_reference_tokens("PAGO FRA. FV25-0108480 Y 0100002")
        self.assertIn("FV250108480", strong)
        self.assertIn("100002", digit)

    def test_service_month(self):
        self.assertEqual(service_month("MANT. MAYO 25"), (2025, 5))
        self.assertEqual(service_month("SERVICE APRIL 2025"), (2025, 4))
        self.assertIsNone(service_month("PAGO FACTURA"))


class TestSubsetSum(unittest.TestCase):
    def test_exact_with_credit_note(self):
        sols, _ = find_subsets([121000, 60500, -14520, 233000], 121000 + 60500 - 14520 + 233000)
        self.assertEqual(sols[0], (0, 1, 2, 3))

    def test_required_and_no_solution(self):
        sols, _ = find_subsets([100, 200, 300], 500, required=[0])
        self.assertTrue(all(0 in s for s in sols))
        self.assertEqual(find_subsets([100, 200], 50)[0], [])

    def test_ambiguity_detected(self):
        sols, _ = find_subsets([100, 100, 200], 200)
        self.assertGreaterEqual(len(sols), 2)


class TestPartyResolver(unittest.TestCase):
    def setUp(self):
        self.r = PartyResolver(CUST, SUPP)

    def test_accents_and_suffix(self):
        self.assertEqual(self.r.resolve("TRANSFERENCIA IBANEZ RIEGOS SCOOP", "", "", {"ES01"}, "AR")[0].party_id, "C3")

    def test_trade_name(self):
        self.assertEqual(self.r.resolve("AGUAS IBANEZ PAGO", "", "", {"ES01"}, "AR")[0].party_id, "C3")

    def test_longer_name_wins_tie(self):
        self.assertEqual(self.r.resolve("RIEGOS OASIS NORTE SL", "", "", {"ES01"}, "AR")[0].party_id, "C2")

    def test_truncated_name(self):
        self.assertEqual(self.r.resolve("RIEGOS OASIS NOR", "", "", {"ES01"}, "AR")[0].party_id, "C2")

    def test_iban_exact(self):
        c = self.r.resolve("", "", "ES91 0000 0000 0000 0000 0001", {"ES01"}, "AR")[0]
        self.assertEqual((c.party_id, c.match_method), ("C1", "exact_iban"))

    def test_entity_filter_and_unknown(self):
        self.assertEqual(self.r.resolve("RIEGOS OASIS", "", "", {"MX01"}, "AR"), [])
        self.assertEqual(self.r.resolve("ZZZ UNKNOWN PAYER", "", "", {"ES01"}, "AR"), [])

    def test_group(self):
        self.assertEqual(self.r.group_members("P1"), ["P1", "S1"])


class TestMatcher(unittest.TestCase):
    def test_exact_reference(self):
        m, L = matcher()
        p = m.match(line(1210.00, "PAGO FRA FV25-0100001"), L)
        self.assertEqual((p.match_method, ids(p)), ("ref_exact", ["ARI-2025-0000001"]))

    def test_multi_invoice_with_credit_note_by_party(self):
        m, L = matcher()
        p = m.match(line(1210 + 605 - 145.20 + 2330, "TRANSF RIEGOS OASIS SL"), L)
        self.assertEqual(ids(p), ["ARC-2025-0000003", "ARI-2025-0000001", "ARI-2025-0000002", "ARI-2025-0000004"])
        self.assertTrue(all(a["amount"] > 0 for a in p.allocations))          # positive convention, credit note too

    def test_same_amount_fifo_and_month(self):
        m, L = matcher()
        self.assertEqual(ids(m.match(line(300.00, "IBANEZ RIEGOS"), L)), ["ARI-2025-0000009"])            # oldest
        self.assertEqual(ids(m.match(line(300.00, "IBANEZ RIEGOS SERVICIOS MARZO 2025"), L)), ["ARI-2025-0000010"])  # invoiced next month

    def test_discount_within_terms(self):
        m, L = matcher()
        p = m.match(line(490.00, "IBANEZ RIEGOS", date="2025-02-12"), L)   # 500 - 2%, issued 02-01, 10d + 3 grace
        self.assertEqual([(a["item_id"], a["amount"], a["exception"]) for a in p.allocations],
                         [("ARI-2025-0000006", 500.0, "SHORT_DISCOUNT")])

    def test_discount_after_deadline_not_short_discount(self):
        m, L = matcher()
        p = m.match(line(490.00, "IBANEZ RIEGOS", date="2025-03-20"), L)
        self.assertNotIn("SHORT_DISCOUNT", [a["exception"] for a in p.allocations])

    def test_withholding_country_gate(self):
        m, L = matcher()
        p = m.match(line(9000.00, "ALBERCAS DEL GOLFO", entity="MX01", ccy="MXN"), L)  # 10% withheld, MX customer
        self.assertEqual([(a["amount"], a["exception"]) for a in p.allocations], [(10000.0, "SHORT_WITHHOLDING")])

    def test_partial_hint(self):
        m, L = matcher()
        p = m.match(line(400.00, "PAGO A CUENTA FV25-0100004"), L)
        self.assertEqual([(a["amount"], a["exception"]) for a in p.allocations], [(400.0, "PARTIAL")])

    def test_fx_uses_orig_amount(self):
        it = ITEMS.copy()
        it.loc[0, "currency"] = "USD"
        store = ItemStore(it)
        m = CustomerMatcher(store, PartyResolver(CUST, SUPP), {"ES01": {"ES01"}}, terms=TERMS)
        p = m.match(line(1100.00, "FV25-0100001", orig_amt="1210.00", orig_ccy="USD"), Ledger(it.open_cents))
        self.assertEqual([(a["amount"], a["exception"]) for a in p.allocations], [(1210.0, "FX_DIFFERENCE")])

    def test_parent_pays_subsidiary(self):
        m, L = matcher()
        self.assertEqual(ids(m.match(line(777.00, "GRUPO HOTELERO SOL"), L)), ["ARI-2025-0000008"])

    def test_already_settled_and_unknown(self):
        m, L = matcher()
        L.o[0] = 0
        p = m.match(line(1210.00, "PAGO FRA FV25-0100001"), L)
        self.assertNotIn("ARI-2025-0000001", ids(p))
        self.assertTrue(m.match(line(999.99, "UNKNOWN PAYER XYZ"), L).abstained)


if __name__ == "__main__":
    unittest.main()
