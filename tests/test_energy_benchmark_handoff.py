"""CPU environment/commit/merge tests; no training and no test-seed rollouts."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from scripts.evaluation import run_energy_collision_benchmark as bench
from scripts.analysis.report_energy_collision_benchmark import scan_archives, comparison, report


class FixturePolicy:
    def act(self, obs, mask, epsilon, rng):
        assert epsilon == 0.0
        return [int(rng.choice(np.flatnonzero(row))) for row in mask]


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = bench.make_protocol("P03", 1, True)

    def test_original_science_and_partition_locks(self):
        self.assertEqual(set(bench.verify_assets()), {"P03", "P05"})
        a = bench.make_protocol("P03", 1)
        b = bench.make_protocol("P05", 2)
        self.assertEqual(a["comparison_fingerprint"], b["comparison_fingerprint"])
        self.assertEqual(a["target_seeds"], list(range(920101, 920106)))
        self.assertEqual(b["target_seeds"], list(range(920106, 920111)))
        self.assertFalse(set(a["config"]["test_seeds"]) & set(a["training_protocol"]["validation_seeds"]))

    def test_scalar_observer_copied_verbatim(self):
        old = ast.parse((bench.ROOT / "scripts/training/run_bs5_energy_collision_train.py").read_text())
        new = ast.parse((bench.ROOT / "scripts/evaluation/energy_benchmark_metrics.py").read_text())
        get = lambda tree: ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "observe_step"))
        self.assertEqual(get(old), get(new))

    def test_midseed_resume_and_battery_accounting(self):
        p = self.p
        policy = FixturePolicy()
        a = bench.start_state(p, 940001)
        b = bench.start_state(p, 940001)
        for _ in range(96):
            bench.advance(a, policy, p)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for _ in range(31):
                bench.advance(b, policy, p)
            bench.save_state(folder, p, b)
            # An uncommitted inactive generation must not destroy the active commit.
            progress = json.loads((folder / "progress.json").read_text())
            (folder / f"resume_{1-progress['generation']}.pkl").write_bytes(b"interrupted write")
            b = bench.load_state(folder, p)
            self.assertEqual(len(b["rows"]), 31)
            for _ in range(65):
                bench.advance(b, policy, p)
            for key in a["arrays"]:
                np.testing.assert_array_equal(a["arrays"][key], b["arrays"][key])
            self.assertEqual(a["energy"], b["energy"])
            self.assertEqual(a["collision"], b["collision"])
            self.assertEqual(a["env"].energy_events, b["env"].energy_events)
            self.assertEqual(a["ue_trace_sha256"], b["ue_trace_sha256"])
            aa = bench.summarize(a, p, True)
            bb = bench.summarize(b, p, True)
            for key in aa:
                if key not in ("actor_ms", "actor_ms_p95", "actor_ms_max", "radio_ms", "simulation_wall_seconds"):
                    self.assertEqual(aa[key], bb[key], key)
            self.assertLess(aa["energy_balance_max_abs_j"], 1e-7)
            self.assertAlmostEqual(aa["energy_consumed_wh"], sum(aa[k] for k in ("flight_wh", "circuit_wh", "access_pa_wh", "backhaul_pa_wh")), places=8)
            bench.save_state(folder, p, b)
            bench.save_seed(folder, p, b, True)
            self.assertEqual(json.loads((folder / "summary.json").read_text())["slots"], 96)
            with np.load(folder / "trace.npz", allow_pickle=False) as d:
                self.assertEqual(d["ue_rates"].shape, (96, 90))
            current = json.loads((folder / "progress.json").read_text())["resume_file"]
            with (folder / current).open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaises(ValueError):
                bench.load_state(folder, p)

    def fixture_archive(self, folder, label, part, rows, suffix="REVIEW"):
        p = bench.make_protocol(label, part)
        name = f"{label}_TEST_PART{part}"
        path = folder / f"{name}_{suffix}.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr(name+"/protocol.json", json.dumps(p))
            for row in rows:
                z.writestr(f"{name}/seed{row['seed']}/summary.json", json.dumps(row))
        return path

    def test_export_review_excludes_resume_and_temp(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "P03_TEST_PART1"
            seed = root / "seed940001"
            seed.mkdir(parents=True)
            bench.atomic_json(root / "protocol.json", self.p)
            (seed / "resume_0.pkl").write_bytes(b"fixture")
            (seed / "resume_1.pkl").write_bytes(b"fixture-old")
            (seed / "progress.json").write_text("{}")
            (root / "worker.lock").write_text("")
            (seed / "summary.json.tmp").write_text("partial-write")
            bench.export(root)
            with zipfile.ZipFile(root.parent / (root.name+"_REVIEW.zip")) as z:
                self.assertTrue(all(not n.endswith((".pkl", ".tmp", ".lock")) for n in z.namelist()))
                self.assertIn(root.name+"/protocol.json", z.namelist())
            with zipfile.ZipFile(root.parent / (root.name+"_RESUME.zip")) as z:
                self.assertIn(root.name+"/seed940001/resume_0.pkl", z.namelist())
                self.assertIn(root.name+"/seed940001/resume_1.pkl", z.namelist())

    def test_merge_completeness_duplicate_trace_and_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            paths = []
            for label in ("P03", "P05"):
                for part in (1, 2):
                    p = bench.make_protocol(label, part)
                    rows = []
                    for seed in p["target_seeds"]:
                        row = dict(label=label, seed=seed, slots=10000, target_slots=10000, complete=True,
                                   diagnostic=False, comparison_fingerprint=p["comparison_fingerprint"],
                                   ue_trace_sha256=str(seed), energy_consumed_wh=3000., delivered_mbit=600000.,
                                   active_blocked_total=20., policy_move_requests_total=1000.)
                        from scripts.analysis.report_energy_collision_benchmark import METRICS
                        row.update({key: 1. for key in METRICS if key not in row})
                        row["throughput_mbps"] = 100.+(5. if label=="P05" else 0.)
                        if seed == 920101:
                            row["blocked_per_policy_move"] = None
                            row["policy_move_requests_total"] = 0.
                            row["active_blocked_total"] = 0.
                        rows.append(row)
                    paths.append(self.fixture_archive(folder, label, part, rows))
            meta = report(paths[:-1], folder / "partial")
            self.assertEqual(meta["status"], "partial")
            self.assertFalse((folder / "partial/paired_comparison.csv").exists())
            complete = report(paths, folder / "complete")
            self.assertEqual(complete["status"], "complete")
            self.assertTrue((folder / "complete/paired_metrics.png").exists())
            result, p, _ = scan_archives(paths)
            _, paired = comparison(result, p["config"]["test_seeds"])
            throughput = next(x for x in paired if x["metric"] == "throughput_mbps")
            self.assertEqual(throughput["p05_minus_p03"], 5.)
            self.assertEqual(throughput["delta_ci95_low"], 5.)
            self.assertEqual(throughput["delta_ci95_high"], 5.)
            # An exact repeated review archive is harmless.
            self.assertEqual(len(scan_archives(paths+[paths[0]])[0]), 20)
            bad = deepcopy(rows)
            bad[0]["ue_trace_sha256"] = "not-the-same-trace"
            corrupt = self.fixture_archive(folder, "P05", 2, bad, "BAD")
            with self.assertRaisesRegex(ValueError, "UE trace mismatch"):
                scan_archives(paths[:-1]+[corrupt])
            with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                scan_archives(paths+[corrupt])


if __name__ == "__main__":
    unittest.main(verbosity=2)
