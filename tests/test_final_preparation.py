"""The FINAL conversion must never overwrite existing TRAIN/LIVE parquet."""
import gzip
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class FinalConversionTests(unittest.TestCase):
    def test_final_only_preserves_other_phases_and_reports_errors(self):
        source = Path(__file__).resolve().parents[1] / 'eda/00_to_parquet.py'
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'eda').mkdir()
            shutil.copy2(source, root/'eda/00_to_parquet.py')
            for phase in ['final','train']:
                (root/'data'/phase).mkdir(parents=True)
                with gzip.open(root/'data'/phase/'bank_transactions.csv.gz', 'wt') as f:
                    f.write('bank_txn_id,amount\nBT-2026-0000001,10.00\n')
            keep = root/'data/parquet/train/bank_transactions.parquet'
            keep.parent.mkdir(parents=True)
            keep.write_bytes(b'preserve existing development parquet')
            before = keep.stat().st_mtime_ns
            result = subprocess.run([sys.executable, str(root/'eda/00_to_parquet.py'), '--phase', 'final'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root/'data/parquet/final/bank_transactions.parquet').is_file())
            self.assertEqual(keep.read_bytes(), b'preserve existing development parquet')
            self.assertEqual(keep.stat().st_mtime_ns, before)
            self.assertNotIn('train ', result.stdout)
            result = subprocess.run([sys.executable, str(root/'eda/00_to_parquet.py'), '--phase', 'live'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('phase directory does not exist', result.stderr)

if __name__ == '__main__':
    unittest.main()
