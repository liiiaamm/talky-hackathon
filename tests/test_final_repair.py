"""FINAL safety nets: --repair makes an invalid submission valid; a missing category model is fatal in FINAL."""
import contextlib
import io
import unittest
from unittest.mock import patch

import polars as pl

from bankrec import categories, pipeline
from bankrec.data import Ctx


def ctx(phase="final"):
    c = Ctx(phase=phase, bank=pl.DataFrame({"bank_txn_id": ["b1", "b2", "b3", "b4"], "is_credit": [True, False, True, True]}),
            items=pl.DataFrame({"item_id": ["ARI-2026-0000001"], "open_amount": [10.0]}))
    c.categories = ["BANK_FEES", "UNIDENTIFIED", "PAYROLL"]
    return c


def rows(data):
    return pl.DataFrame(data, schema={"bank_txn_id": pl.Utf8, "item_id": pl.Utf8, "amount": pl.Float64,
                                      "category": pl.Utf8, "exception": pl.Utf8, "module": pl.Utf8}, orient="row")


class RepairTests(unittest.TestCase):
    def test_repair_drops_offenders_and_fills_missing(self):
        c = ctx()
        sub = rows([
            ("b1", "ARI-2026-0000001", 10.0, None, "", "m"),
            ("b1", "ARI-2026-0000001", 10.0, None, "", "m"),        # duplicated pair
            ("b1", None, None, "PAYROLL", "", "m"),                 # no-item row on an item line
            ("b2", "BAD-ID", 5.0, None, "", "m"),                    # bad item id -> b2 uncovered
            ("b3", None, None, "NOT_A_CATEGORY", "WEIRD", "m"),     # unknown category -> b3 uncovered
            ("bX", None, None, "PAYROLL", "", "m"),                 # outside the phase
        ])
        self.assertTrue(pipeline.validate(sub, c))
        with contextlib.redirect_stdout(io.StringIO()):
            fixed = pipeline.repair(sub, c)
        self.assertEqual(pipeline.validate(fixed, c), [])
        self.assertEqual(sorted(fixed["bank_txn_id"].to_list()), ["b1", "b2", "b3", "b4"])
        self.assertEqual(fixed.filter(pl.col("bank_txn_id") == "b1")["item_id"].to_list(), ["ARI-2026-0000001"])
        self.assertEqual(fixed.filter(pl.col("bank_txn_id") == "b2")["category"].to_list(), ["BANK_FEES"])

    def test_valid_submission_is_unchanged_by_repair(self):
        c = ctx()
        sub = pipeline.fallback(c, c.bank)
        with contextlib.redirect_stdout(io.StringIO()):
            fixed = pipeline.repair(sub, c)
        self.assertTrue(fixed.sort("bank_txn_id").equals(sub.sort("bank_txn_id")))


class MissingModelTests(unittest.TestCase):
    def test_missing_model_is_fatal_in_final_only(self):
        with patch.object(categories, "MODEL", "/nonexistent/categories.joblib"):
            with self.assertRaises(FileNotFoundError):
                categories.strong_categories(ctx("final"), ctx("final").bank)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(categories.all_categories(ctx("live"), ctx("live").bank).height, 0)
            self.assertIn("WARNING", out.getvalue())


if __name__ == "__main__":
    unittest.main()
