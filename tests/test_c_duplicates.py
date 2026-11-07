"""Persona C duplicates: only an earlier line that paid the SAME item in full by itself makes a later equal payment a repeat
(POLICIES §3: the later line gets the exception).   python -m unittest tests.test_c_duplicates"""
import os
import sys
import unittest
from types import SimpleNamespace

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bankrec.c_evidence import _prior_full_payments  # noqa: E402


def ctx_with(claimed_rows: dict) -> SimpleNamespace:
    ctx = SimpleNamespace(bank=pl.DataFrame({
        "bank_txn_id": ["B1", "B2", "B3", "B4"],
        "booking_date": ["2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"],
        "abs_amt": [100.0, 250.0, 80.0, 40.0]}))
    ctx.__dict__["_claimed"] = [pl.DataFrame(claimed_rows)]
    return ctx


class TestPriorFullPayments(unittest.TestCase):
    def test_only_single_item_full_payments_count(self):
        ctx = ctx_with({
            "bank_txn_id": ["B1", "B2", "B2", "B3", "B4"],
            "item_id": ["ARI-2025-0000001", "ARI-2025-0000002", "ARI-2025-0000003", "ARI-2025-0000004", None],
            "amount": [100.0, 150.0, 100.0, 50.0, None]})
        got = _prior_full_payments(ctx)
        # B1 paid one item in full; B2 paid two items; B3 paid part of one item; B4 is a category row
        self.assertEqual(got["item_id"].to_list(), ["ARI-2025-0000001"])
        self.assertEqual(got["p_date"].to_list(), ["2025-01-02"])

    def test_no_claimed_rows(self):
        ctx = SimpleNamespace(bank=pl.DataFrame({"bank_txn_id": ["B1"], "booking_date": ["2025-01-02"], "abs_amt": [1.0]}))
        self.assertEqual(_prior_full_payments(ctx).height, 0)


if __name__ == "__main__":
    unittest.main()
