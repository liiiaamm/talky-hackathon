"""Failure isolation and stable submission serialization."""
import contextlib
import gzip
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import polars as pl
from bankrec import pipeline
from bankrec.data import Ctx

class RuntimeTests(unittest.TestCase):
    def context(self):
        return Ctx(phase="sample", bank=pl.DataFrame({"bank_txn_id": ["b2", "b1"], "is_credit": [True, False]}),
                   items=pl.DataFrame({"item_id": ["i1"], "open_amount": [10.]}))
    def test_safe_failure_preserves_pending_and_balances(self):
        ctx = self.context()
        original = ctx.items
        def broken(ctx, pending):
            ctx.items = ctx.items.head(0)
            raise RuntimeError("injected failure")
        seen = []
        def later(ctx, pending):
            seen.extend(pending["bank_txn_id"].to_list())
            return pipeline.fallback(ctx, pending)
        stderr, stdout = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
            rows = pipeline.run(ctx, [broken, later], verbose=False, safe=True, profile=True)
        self.assertEqual(sorted(seen), ["b1", "b2"])
        self.assertTrue(ctx.items.equals(original))
        self.assertEqual(rows.height, 2)
        self.assertIn("injected failure", stderr.getvalue())
        self.assertIn("status=failed", stdout.getvalue())
        self.assertIn("status=ok", stdout.getvalue())
    def test_default_failure_stops(self):
        def broken(ctx, pending):
            raise RuntimeError("injected failure")
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            pipeline.run(self.context(), [broken], verbose=False)
    def test_failed_ledger_does_not_commit_rows(self):
        ctx = self.context()
        def claim(ctx, pending):
            return pipeline.fallback(ctx, pending.head(1))
        with patch.object(pipeline, "LEDGER", True), patch.object(pipeline, "apply_ledger", side_effect=RuntimeError("ledger")):
            with contextlib.redirect_stderr(io.StringIO()):
                rows = pipeline.run(ctx, [claim], verbose=False, safe=True)
        self.assertEqual(rows["bank_txn_id"].n_unique(), 2)
        self.assertEqual(rows.height, 2)
    def test_safe_failure_restores_chronological_ledger_events(self):
        ctx = self.context()
        ctx._ledger = {"opening": ctx.items, "events": [ctx.items]}
        def broken(ctx, pending):
            ctx._ledger["events"].append(ctx.items.head(0))
            raise RuntimeError("injected event")
        with contextlib.redirect_stderr(io.StringIO()):
            pipeline.run(ctx, [broken], verbose=False, safe=True)
        self.assertEqual(len(ctx._ledger["events"]), 1)
        del ctx._ledger
        def newly_created(ctx, pending):
            ctx._ledger = {"opening": ctx.items, "events": []}
            raise RuntimeError("new event")
        with contextlib.redirect_stderr(io.StringIO()):
            pipeline.run(ctx, [newly_created], verbose=False, safe=True)
        self.assertFalse(hasattr(ctx, "_ledger"))

    def test_invalid_output_is_not_written_by_cli(self):
        ctx = self.context()
        rows = pipeline.fallback(ctx, ctx.bank)
        with patch.object(pipeline, "load", return_value=ctx), patch.object(pipeline, "run", return_value=rows), patch.object(pipeline, "validate", return_value=["bad amount"]), patch.object(pipeline, "write") as writer, patch("sys.argv", ["pipeline", "--out", "invalid.csv"]):
            with self.assertRaises(SystemExit) as exc:
                pipeline.main()
        self.assertEqual(exc.exception.code, 1)
        writer.assert_not_called()

    def test_writer_same_bytes_for_row_order_and_output_name(self):
        rows = pipeline.fallback(self.context(), self.context().bank)
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.csv.gz", Path(d) / "b.csv.gz"
            pipeline.write(rows, str(a))
            pipeline.write(rows.reverse(), str(b))
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(gzip.decompress(a.read_bytes()).decode().splitlines()[1].split(",")[0], "b1")
if __name__ == "__main__":
    unittest.main()
