"""Persona C overpayment rule: the deliberate-top-up test is a scale-free convention (1-2-5 nice numbers, one minor unit),
not a list of amounts read off labels.   python -m unittest tests.test_c_overpayment"""
import os
import sys
import unittest

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bankrec.c_evidence import _nice_excess  # noqa: E402


def nice(excess: float, open_amount: float) -> bool:
    df = pl.DataFrame({"ex": [excess], "o": [open_amount]})
    return bool(df.select(_nice_excess(pl.col("ex"), pl.col("o")).alias("n"))["n"][0])


class TestNiceExcess(unittest.TestCase):
    def test_round_unit_top_ups(self):
        for ex in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0):
            self.assertTrue(nice(ex, 1065.45), ex)

    def test_amounts_outside_the_one_two_five_series(self):
        for ex in (3.0, 7.0, 15.0, 150.0, 123.45):
            self.assertFalse(nice(ex, 2359.66), ex)

    def test_percentages_within_one_minor_unit(self):
        self.assertTrue(nice(23.59, 2359.66))    # 1 % = 23.5966 (cash truncated to the cent)
        self.assertTrue(nice(52.70, 527.08))     # 10 % = 52.708
        self.assertTrue(nice(47.19, 2359.66))    # 2 %
        self.assertFalse(nice(70.79, 2359.66))   # 3 % is not a 1-2-5 percentage

    def test_unit_top_up_never_above_the_invoice(self):
        self.assertFalse(nice(500.0, 300.0))

    def test_rounding_the_cash_up_is_not_a_top_up(self):
        self.assertFalse(nice(0.55, 1065.45))


if __name__ == "__main__":
    unittest.main()
