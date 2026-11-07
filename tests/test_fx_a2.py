"""A2: anchor_complete (cited invoices + uncited completing ones). Synthetic fixtures only."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from test_matching import ids, line, matcher  # noqa: E402


class TestAnchorComplete(unittest.TestCase):
    def test_cited_plus_one_uncited(self):
        m, led = matcher()
        p = m.match(line("3540.00", "TRANSFERENCIA RIEGOS OASIS SL FV25-0100001"), led)   # 1210.00 cited + 2330.00 uncited
        self.assertEqual(sorted(ids(p)), ["ARI-2025-0000001", "ARI-2025-0000004"])

    def test_off_amount_does_not_complete(self):
        m, led = matcher()
        p = m.match(line("3541.00", "TRANSFERENCIA RIEGOS OASIS SL FV25-0100001"), led)
        self.assertNotEqual(p.match_method, "anchor_complete")

    def test_switch_off(self):
        m, led = matcher({"anchor_complete": False})
        p = m.match(line("3540.00", "TRANSFERENCIA RIEGOS OASIS SL FV25-0100001"), led)
        self.assertNotEqual(p.match_method, "anchor_complete")


if __name__ == "__main__":
    unittest.main()
