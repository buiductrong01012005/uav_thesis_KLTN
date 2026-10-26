"""Integrity regressions for the imported seven-method benchmark; no rollout."""
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.evaluation import run_bs5_benchmark7_best120k as benchmark


class BenchmarkSevenIntegrityTests(unittest.TestCase):
    def test_metadata_paths_relocate_and_cannot_escape_checkpoint_directory(self):
        original = 'benchmark_assets/bs5_v4_best120k/C-R_seed551_best120k.pt'
        self.assertEqual(benchmark.model_asset_path(original),
                         (benchmark.ASSETS / 'C-R_seed551_best120k.pt').resolve())
        for invalid in ['../model.pt', 'benchmark_assets/bs5_v4_best120k/../other.pt']:
            with self.assertRaises(AssertionError):
                benchmark.model_asset_path(invalid)

    @unittest.skipUnless(benchmark.ASSETS.is_dir(), 'Local checkpoint assets not installed')
    def test_scientific_source_change_is_rejected_after_import(self):
        real_hash = benchmark.file_hash
        protected = benchmark.ROOT / 'src/envs/bs5_return_commit_env.py'

        def changed_hash(path):
            return '0' * 64 if Path(path).resolve() == protected.resolve() else real_hash(path)

        with patch.object(benchmark, 'file_hash', side_effect=changed_hash):
            with self.assertRaisesRegex(RuntimeError, 'bs5_return_commit_env.py'):
                benchmark.check_core_source()

    @unittest.skipUnless(benchmark.ASSETS.is_dir(), 'Local checkpoint assets not installed')
    def test_checkpoint_corruption_is_rejected(self):
        real_hash = benchmark.file_hash
        protected = benchmark.ASSETS / 'C-R_seed551_best120k.pt'

        def changed_hash(path):
            return '0' * 64 if Path(path).resolve() == protected.resolve() else real_hash(path)

        with patch.object(benchmark, 'file_hash', side_effect=changed_hash):
            with self.assertRaises(AssertionError):
                benchmark.check_inputs()


if __name__ == '__main__':
    unittest.main()
