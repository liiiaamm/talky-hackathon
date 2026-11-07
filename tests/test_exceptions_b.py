"""Person B exception rules and environment checks (synthetic fixtures, no competition data).
python -m unittest discover -s tests"""
import importlib
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from test_matching import CUST, ITEMS, SUPP, TERMS, Ledger, ids, line  # noqa: E402
from matching.customer_matcher import CustomerMatcher, ItemStore, _round_units  # noqa: E402
from matching.party_resolver import PartyResolver  # noqa: E402

FX = pd.DataFrame([("2025-01-31", "USD", 1.04), ("2025-02-03", "USD", 1.02)], columns=["date", "currency", "units_per_eur"])


def matcher(cfg=None):
    store = ItemStore(ITEMS)
    m = CustomerMatcher(store, PartyResolver(CUST, SUPP), {"ES01": {"ES01"}, "MX01": {"MX01"}}, cfg, terms=TERMS, fx=FX)
    return m, Ledger(ITEMS.open_cents)


def xline(amount, text="", code="", fmt="camt053", **kw):
    ln = line(amount, text, **kw)
    ln.bank_code, ln.statement_format = code, fmt
    return ln


class TestBankCharges(unittest.TestCase):
    def test_cross_border_fee_clears_invoice(self):
        m, led = matcher()
        p = m.match(xline("2315.00", "WIRE IN RIEGOS OASIS SL", code="PMNT/RCDT/XBCT"), led)   # 2330.00 - 15.00 fee
        self.assertEqual(p.match_method, "bank_charges")
        self.assertEqual(ids(p), ["ARI-2025-0000004"])
        self.assertEqual(p.allocations[0]["exception"], "SHORT_BANK_CHARGES")
        self.assertAlmostEqual(p.allocations[0]["amount"], 2330.00)

    def test_no_cross_border_evidence_no_charges(self):
        m, led = matcher()
        p = m.match(xline("2315.00", "TRANSFERENCIA RIEGOS OASIS SL", code="PMNT/RCDT/ESCT"), led)
        self.assertNotEqual(p.match_method, "bank_charges")

    def test_shortfall_above_cap_is_not_charges(self):
        m, led = matcher()
        p = m.match(xline("2250.00", "WIRE IN RIEGOS OASIS SL", code="PMNT/RCDT/XBCT"), led)   # 80.00 short > 60 EUR
        self.assertNotEqual(p.match_method, "bank_charges")

    def test_n43_code_13_and_sha_text(self):
        self.assertTrue(CustomerMatcher._cross_border(xline("1", "X", code="13", fmt="n43")))
        self.assertTrue(CustomerMatcher._cross_border(xline("1", "TRANSFER. EN DIV. | GASTOS SHA", code="02", fmt="n43")))
        self.assertFalse(CustomerMatcher._cross_border(xline("1", "SHARED SERVICES", code="02", fmt="n43")))

    def test_cap_in_foreign_currency_uses_last_rate(self):
        m, _ = matcher()
        self.assertEqual(m._cap_cents("USD", "2025-02-02"), int(60 * 1.04 * 100))
        self.assertEqual(m._cap_cents("EUR", "2025-02-02"), 6000)


class TestDuplicates(unittest.TestCase):
    def test_repeat_of_a_full_single_payment(self):
        m, led = matcher()
        first = m.match(line("2330.00", "TRANSFERENCIA RIEGOS OASIS SL", date="2025-02-05"), led)
        self.assertEqual(ids(first), ["ARI-2025-0000004"])
        i = m.s.pos["ARI-2025-0000004"]
        led.o[i] = 0
        m.record(first, 233000, pd.Timestamp("2025-02-05").toordinal())
        p = m.match(line("2330.00", "TRANSFERENCIA RIEGOS OASIS SL", date="2025-02-12"), led)
        self.assertEqual(p.match_method, "duplicate")
        self.assertEqual(p.allocations[0]["exception"], "DUPLICATE_PAYMENT")
        self.assertAlmostEqual(p.allocations[0]["amount"], 2330.00)

    def test_same_amount_with_another_open_invoice_is_not_a_duplicate(self):
        m, led = matcher()
        first = m.match(line("300.00", "TRANSFERENCIA AGUAS IBANEZ", date="2025-03-05"), led)
        led.o[m.s.pos[first.allocations[0]["item_id"]]] = 0
        m.record(first, 30000, pd.Timestamp("2025-03-05").toordinal())
        p = m.match(line("300.00", "TRANSFERENCIA AGUAS IBANEZ", date="2025-04-05"), led)   # FV25-0100010 still open
        self.assertNotEqual(p.match_method, "duplicate")

    def test_too_late_is_not_a_duplicate(self):
        m, led = matcher()
        first = m.match(line("2330.00", "TRANSFERENCIA RIEGOS OASIS SL", date="2025-02-05"), led)
        led.o[m.s.pos["ARI-2025-0000004"]] = 0
        m.record(first, 233000, pd.Timestamp("2025-02-05").toordinal())
        p = m.match(line("2330.00", "TRANSFERENCIA RIEGOS OASIS SL", date="2025-04-05"), led)
        self.assertNotEqual(p.match_method, "duplicate")


class TestOverpayment(unittest.TestCase):
    def test_round_units(self):
        self.assertEqual([c for c in (100, 200, 500, 1000, 5000, 900, 1400, 150, 0) if _round_units(c)], [100, 200, 500, 1000, 5000])

    def test_round_top_up_clears_at_open_amount(self):
        m, led = matcher()
        p = m.match(line("2380.00", "TRANSFERENCIA RIEGOS OASIS SL"), led)     # 2330.00 + 50.00
        self.assertEqual(p.match_method, "overpayment")
        self.assertEqual(p.allocations[0]["exception"], "OVERPAYMENT")
        self.assertAlmostEqual(p.allocations[0]["amount"], 2330.00)

    def test_odd_excess_abstains(self):
        m, led = matcher()
        p = m.match(line("2343.17", "TRANSFERENCIA RIEGOS OASIS SL"), led)
        self.assertNotEqual(p.match_method, "overpayment")


class TestSuffixIndex(unittest.TestCase):
    def test_six_digit_suffix_is_indexed(self):
        s = ItemStore(ITEMS)
        self.assertIn(s.pos["ARI-2025-0000004"], s.tails[("ES01", "100004")])


class TestApBulk(unittest.TestCase):
    def test_c34_split_run_id_and_count(self):
        import polars as pl
        from bankrec.ap_bulk import C34_SPLIT, COUNT
        t = pl.Series(["TRANSF. EMITIDAS FICHERO C34 PR0200250 | 11085 | (357 ORD.)"]).str.replace_all(C34_SPLIT, "${1}${2}")
        self.assertEqual(t.str.extract(r"((?:PR|CF)\d{12})")[0], "PR020025011085")
        self.assertEqual(t.str.extract(COUNT, 2)[0], "357")
        self.assertEqual(pl.Series(["BULK PAYMENT PR010125010963 382 TRANSFERS"]).str.extract(COUNT, 1)[0], "382")


class TestEnvironment(unittest.TestCase):
    """Optional resolvers are imported with `except ImportError: pass` in pipeline.py: a missing dependency silently
    drops a resolver (pyahocorasick -> ar_multi cost 1.7 TRAIN points). Fail loudly here instead."""

    def test_required_dependencies(self):
        for mod in ("polars", "numpy", "pandas", "sklearn", "joblib", "ahocorasick", "pypdf", "openpyxl"):
            importlib.import_module(mod)

    def test_all_resolvers_load(self):
        os.environ.pop("BANKREC_SKIP", None)
        pipeline = importlib.import_module("bankrec.pipeline")
        names = {f.__name__ for f in pipeline.MODULES}
        for need in ("remittances", "pdf_clauses", "bulk_mixed_runs", "ar_text", "exact_amount", "multi_invoice",
                     "strong_categories", "all_categories"):
            self.assertIn(need, names)


if __name__ == "__main__":
    unittest.main()
