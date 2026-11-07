"""Ledger invariants independent of resolver recall and trained models."""
from pathlib import Path
import sys
import unittest

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bankrec.data import Ctx
from bankrec.pipeline import apply_ledger


class LedgerTests(unittest.TestCase):
    def context(self, opening=100.0):
        bank = pl.DataFrame({"bank_txn_id": ["early", "late"], "booking_date": ["2025-01-01", "2025-01-02"]})
        all_items = pl.DataFrame({"item_id": ["i", "untouched"], "open_amount": [opening, 15.0]})
        return Ctx("sample", bank, all_items.filter(pl.col("open_amount") > 0), all_items=all_items)

    def rows(self, txn, amount, exception=""):
        return pl.DataFrame({"bank_txn_id": [txn], "item_id": ["i"], "amount": [amount], "exception": [exception]})

    def balance(self, ctx):
        found = ctx.items.filter(pl.col("item_id") == "i")
        return found["open_amount"][0] if found.height else 0.0

    def test_closed_item_reopens_after_return(self):
        ctx = self.context()
        apply_ledger(ctx, self.rows("early", 100.0))
        self.assertEqual(self.balance(ctx), 0.0)
        apply_ledger(ctx, self.rows("late", 100.0, "RETURNED"))
        self.assertEqual(self.balance(ctx), 100.0)

    def test_item_closed_before_phase_can_reopen(self):
        ctx = self.context(0.0)
        apply_ledger(ctx, self.rows("early", 20.0, "RETURNED"))
        self.assertEqual(self.balance(ctx), 20.0)
        self.assertEqual(ctx.items.filter(pl.col("item_id") == "untouched")["open_amount"][0], 15.0)

    def test_explained_difference_does_not_swallow_return(self):
        for exception in ["SHORT_DISCOUNT", "SHORT_BANK_CHARGES", "SHORT_WITHHOLDING", "FX_DIFFERENCE", "OVERPAYMENT"]:
            ctx = self.context()
            apply_ledger(ctx, self.rows("early", 100.0, exception))
            apply_ledger(ctx, self.rows("late", 25.0, "RETURNED"))
            self.assertEqual(self.balance(ctx), 25.0, exception)

    def test_duplicate_does_not_reduce_balance(self):
        ctx = self.context()
        apply_ledger(ctx, self.rows("early", 100.0, "DUPLICATE_PAYMENT"))
        self.assertEqual(self.balance(ctx), 100.0)

    def test_replay_uses_dates_even_if_modules_arrive_in_reverse(self):
        ctx = self.context(0.0)
        apply_ledger(ctx, self.rows("late", 30.0, "PARTIAL"))
        apply_ledger(ctx, self.rows("early", 100.0, "RETURNED"))
        self.assertEqual(self.balance(ctx), 70.0)

    def test_dispute_leaves_unpaid_amount_open(self):
        ctx = self.context()
        apply_ledger(ctx, self.rows("early", 80.0, "SHORT_DISPUTE"))
        self.assertEqual(self.balance(ctx), 20.0)


if __name__ == "__main__":
    unittest.main()
