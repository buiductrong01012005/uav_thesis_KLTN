"""Bounded end-to-end science/resume checks; unittest, no pytest dependency."""
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
import zipfile

import numpy as np
import torch
from scripts.experiments.bs5_v4_fresh_common import (
    atomic_json, env_for, full_condition, read_json, setup, file_hash,
)
from scripts.training import run_bs5_v4_fresh_train as train
from scripts.evaluation import run_bs5_v4_benchmark as benchmark
from scripts.analysis.bs5_v4_fresh_report import export_training, export_benchmark
from src.baselines.return_v4_safe_greedy import make_greedy


def equal_tree(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a.cpu(), b.cpu()))
    elif isinstance(a, np.ndarray):
        test.assertTrue(np.array_equal(a, b))
    elif isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for k in a:
            equal_tree(test, a[k], b[k])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            equal_tree(test, x, y)
    else:
        test.assertEqual(a, b)


class FreshV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup("cpu")
        parent = os.environ.get("BS5_V4_TEST_ROOT")
        cls.temp = None if parent else tempfile.TemporaryDirectory(prefix="bs5_v4_tests_")
        cls.base = Path(parent or cls.temp.name)
        cls.base.mkdir(parents=True, exist_ok=True)
        cls.training = cls.base / "diagnostic_training"
        train.prepare(cls.training, "cpu", diagnostic=True)
        # Pause during initial validation and resume; training has not started.
        train.run(cls.training, max_units=1)
        cls.first_progress = read_json(cls.training / "runs/C-R_seed955551/progress.json")
        train.run(cls.training)

    @classmethod
    def tearDownClass(cls):
        if cls.temp:
            cls.temp.cleanup()

    def test_01_counts_and_shared_training_tapes(self):
        p = read_json(self.training / "protocol.json")
        self.assertTrue(train.all_complete(self.training, p))
        self.assertEqual(self.first_progress["steps"], 0)
        self.assertEqual(len(self.first_progress["validations"]), 1)
        for seed in p["training_seeds"]:
            runs = [self.training / "runs" / f"{arm}_seed{seed}" for arm in p["arms"]]
            a, b = [read_json(run / "progress.json") for run in runs]
            self.assertEqual(a["initial_weights_sha256"], b["initial_weights_sha256"])
            self.assertGreater(a["updates"], 0)
            for ga, gb in zip(a["games"], b["games"]):
                ia, ib = [read_json(run / g["index"]) for run, g in zip(runs, [ga, gb])]
                self.assertEqual(ia["trace_sha256"], ib["trace_sha256"])
                self.assertEqual(ia["spec"], ib["spec"])

    def test_02_exact_resume_weights_optimizer_replay_rng(self):
        roots = [self.base / "uninterrupted", self.base / "interrupted"]
        for root in roots:
            p = train.prepare(root, "cpu", diagnostic=True)
            p.update(arms=["C-R"], training_seeds=[955551])
            p["planned_training_steps"], p["planned_validation_steps"] = 96, 144
            atomic_json(root / "protocol.json", p)
        train.run(roots[0])
        train.run(roots[1], max_units=4)  # two validation games + two train games
        self.assertEqual(read_json(roots[1] / "status.json")["status"], "paused_at_game_boundary")
        train.run(roots[1])
        a, b = [torch.load(root / "runs/C-R_seed955551" / train.read_progress(root / "runs/C-R_seed955551")["resume_file"],
                          map_location="cpu", weights_only=False) for root in roots]
        for key in ["models", "optimizer", "replay", "exploration", "rng"]:
            equal_tree(self, a[key], b[key])
        for key in ["steps", "updates", "episodes", "lineage"]:
            self.assertEqual(a["progress"][key], b["progress"][key])

    def test_03_mask_terminals_and_no_relay(self):
        for path in self.training.glob("runs/*/training/*/index.json"):
            index = read_json(path)
            with np.load(path.parent / "slots.npz") as z:
                self.assertEqual(bool(z["learning_done"][-1]), index["condition"]["stage"] != "S4")
                if index["arm"] == "C-N":
                    self.assertFalse(z["mask"][:, :, 1::2].any())
                    self.assertFalse((z["actions"] % 2).any())
                    self.assertTrue((z["routes"] == -1).all())

    def test_04_isolated_preview_energy_and_return_memory(self):
        p = read_json(self.training / "protocol.json")
        spec = dict(condition=full_condition(p), horizon=12, scenario_seed=284001,
                    assignment_seed=284001, residual_battery=30., diagnostic=True)
        env = env_for(p, spec, self.base / "preview_tapes", "C-R")
        env.reset(seed=spec["scenario_seed"], assignment_seed=spec["assignment_seed"], diagnostic=True)
        # Exercise both existing Return routing and Active energy-shortfall override.
        env.status[3] = 1
        env.battery[4] = 0.01
        before = pickle.dumps(env, protocol=5)
        roles = np.array([0, 1, 0, 0, 0])
        heuristic = make_greedy("Dynamic-Recovery", env)
        actual = heuristic._score(roles, {0: 2})
        self.assertEqual(pickle.dumps(env, protocol=5), before)
        clone = deepcopy(env)
        actions = [clone.encode_action(2 if i == 0 else 4, int(roles[i])) if clone.status[i] == 0 else 8 for i in range(5)]
        expected = clone.step(actions)["reward"]
        self.assertEqual(actual, expected)
        self.assertIsNotNone(clone.last_return_control)

    def test_05_all_greedy_actions_leave_executed_env_untouched(self):
        p = read_json(self.training / "protocol.json")
        spec = dict(condition=full_condition(p), horizon=12, scenario_seed=284002,
                    assignment_seed=284002, residual_battery=30., diagnostic=True)
        for label in ["5A-Recovery", "4A1R-Recovery", "3A2R-Recovery", "Dynamic-Recovery"]:
            env = env_for(p, spec, self.base / "preview_tapes", label)
            out = env.reset(seed=284002, assignment_seed=284002, diagnostic=True)
            heuristic = make_greedy(label, env)
            for _ in range(3):
                heuristic.observe_service(0.)
            before = pickle.dumps(env, protocol=5)
            actions = heuristic.act(out["action_mask"])
            self.assertEqual(pickle.dumps(env, protocol=5), before)
            self.assertTrue(all(out["action_mask"][i, a] for i, a in enumerate(actions)))
            self.assertGreater(heuristic.preview_calls, 0)

    def test_06_full_export_handoff_and_separate_benchmark(self):
        export = export_training(self.training)
        self.assertTrue((export / "training_resume.zip").exists())
        self.assertTrue((export / "training_for_benchmark.zip").exists())
        self.assertTrue((export / "analysis/learning_curves.png").exists())
        handoff = self.base / "handoff"
        with zipfile.ZipFile(export / "training_for_benchmark.zip") as z:
            self.assertIsNone(z.testzip())
            z.extractall(handoff)
        root = self.base / "diagnostic_benchmark"
        protocol = benchmark.prepare(root, handoff / "bs5_v4_training")
        self.assertEqual(len(protocol["jobs"]), 9)
        self.assertEqual(sum(x["label"] == "Random-Relay" for x in protocol["policies"]), 1)
        benchmark.run(root, max_games=2)
        self.assertEqual(read_json(root / "status.json")["completed"], 2)
        benchmark.run(root)
        self.assertEqual(read_json(root / "status.json")["status"], "complete")
        exported = export_benchmark(root, deep=True)
        receipt = read_json(exported / "analysis/audit_receipt.json")
        self.assertTrue(receipt["deep_planner_recomputed"])
        self.assertEqual(receipt["audited_committed_slots"], 108)
        self.assertEqual(len(receipt["balanced_scenarios"]), 1)

    def test_07_uncommitted_compute_is_never_replayed(self):
        root = self.base / "corrupt_budget"
        p = train.prepare(root, "cpu", diagnostic=True)
        p.update(arms=["C-R"], training_seeds=[955551])
        atomic_json(root / "protocol.json", p)
        train.run(root, max_units=0)
        path = root / "runs/C-R_seed955551/budget_ledger.json"
        ledger = read_json(path)
        ledger["train"]["charged"] += 1
        atomic_json(path, ledger)
        with self.assertRaisesRegex(AssertionError, "Uncommitted"):
            train.run(root)
        self.assertEqual(read_json(path)["train"]["charged"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable on this host")
    def test_08_cuda_forward_backward_and_resume(self):
        root = self.base / "cuda_smoke"
        p = train.prepare(root, "cuda", diagnostic=True)
        p.update(arms=["C-R"], training_seeds=[955551], planned_training_steps=96, planned_validation_steps=144)
        atomic_json(root / "protocol.json", p)
        train.run(root, max_units=4)
        self.assertGreater(read_json(root / "runs/C-R_seed955551/progress.json")["updates"], 0)
        train.run(root)
        self.assertTrue(train.all_complete(root, p))
        setup("cpu")

    def test_09_notebook_source_discovery_and_resume_transport(self):
        project = Path(__file__).resolve().parents[1]
        simulation = self.base / "kaggle_sim"
        inputs, work = simulation / "input", simulation / "working"
        inputs.mkdir(parents=True)
        work.mkdir()
        manifest = project / "DATASET_MANIFEST.json"
        self.assertTrue(manifest.exists(), "Build the Dataset manifest before preflight.")
        data = read_json(manifest)
        archive = inputs / "source_dataset.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for rel in data:
                z.write(project / rel, "uav_thesis/" + rel)
            z.write(manifest, "uav_thesis/DATASET_MANIFEST.json")
        notebook = read_json(project / "scripts/notebooks/bs5_return_v4_fresh_train_kaggle.ipynb")
        code = "".join(notebook["cells"][2]["source"]).replace("/kaggle", str(simulation))
        # Kaggle has Torch in the kernel environment. This CPU CI host supplies
        # the same installed test runtime via PYTHONPATH instead.
        runtime_home = str(Path(torch.__file__).resolve().parent.parent)
        code = code.replace("RUN_ENV['PYTHONPATH']=str(SOURCE_ROOT)",
                            "RUN_ENV['PYTHONPATH']=" + repr(runtime_home) + "+os.pathsep+str(SOURCE_ROOT)")
        namespace = dict(SOURCE_PATH=None)
        exec(compile(code, "notebook-bootstrap-simulation", "exec"), namespace)
        self.assertEqual(namespace["SOURCE_PATH"], archive)
        self.assertEqual(read_json(namespace["SOURCE_ROOT"] / "DATASET_MANIFEST.json"), data)
        exports = sorted(self.training.parent.glob(self.training.name + "_export*"))
        exported = next(x for x in exports if (x / "training_for_benchmark.zip").exists())
        self.assertFalse(namespace["is_main_archive"](exported / "training_resume.zip", "bs5_return_v4_fresh_crcn_v1"))
        marker = inputs / "main_marker.zip"
        with zipfile.ZipFile(marker, "w") as z:
            z.writestr("main/protocol.json", json.dumps(dict(version="bs5_return_v4_fresh_crcn_v1", diagnostic=False)))
        self.assertTrue(namespace["is_main_archive"](marker, "bs5_return_v4_fresh_crcn_v1"))
        restored = work / "resumed_training"
        namespace["restore_archive_or_directory"](exported / "training_resume.zip", restored, "training_resume.zip")
        resume_file = train.read_progress(self.training / "runs/C-R_seed955551")["resume_file"]
        self.assertEqual(file_hash(restored / "runs/C-R_seed955551" / resume_file),
                         file_hash(self.training / "runs/C-R_seed955551" / resume_file))
        before = read_json(restored / "runs/C-R_seed955551/budget_ledger.json")
        rc = namespace["run_stream"](["scripts/training/run_bs5_v4_fresh_train.py", "run", "--output", restored, "--max-seconds", "0"], work / "restore_check.log")
        self.assertEqual(rc, 0)
        self.assertEqual(read_json(restored / "runs/C-R_seed955551/budget_ledger.json"), before)

    def test_10_parallel_partitions_match_sequential_and_merge(self):
        from scripts.experiments.merge_bs5_v4_training import merge
        import shutil
        inputs = []
        for arm in ["C-R", "C-N"]:
            for seed in [955551, 955552]:
                root = self.base / f"partition_{arm}_{seed}"
                p = train.prepare(root, "cpu", diagnostic=True, arms=[arm], seeds=[seed])
                self.assertEqual(p["planned_training_steps"], 96)
                self.assertEqual(p["planned_validation_steps"], 144)
                train.run(root, max_units=4)
                train.run(root)
                run = root / "runs" / f"{arm}_seed{seed}"
                baseline = self.training / "runs" / f"{arm}_seed{seed}"
                a, b = [torch.load(r / train.read_progress(r)["resume_file"], map_location="cpu", weights_only=False) for r in [run, baseline]]
                for key in ["models", "optimizer", "replay", "exploration", "rng"]:
                    equal_tree(self, a[key], b[key])
                export = export_training(root)
                self.assertTrue((export / "training_for_benchmark.zip").exists())
                inputs.append(export / "training_for_benchmark.zip")
        merged = merge(self.base / "merged_four", inputs)
        self.assertTrue(train.all_complete(merged, read_json(merged / "protocol.json")))
        self.assertEqual(len(read_json(merged / "TRAINING_PROVENANCE.json")), 4)
        self.assertTrue((merged / "analysis/learning_curves.png").exists())
        bp = benchmark.prepare(self.base / "merged_benchmark", merged)
        self.assertEqual(len(bp["policies"]), 9)
        self.assertIn("training_job_provenance", bp)
        self.assertTrue((self.base / "merged_benchmark/training_analysis/learning_curves.png").exists())
        benchmark.run(self.base / "merged_benchmark", max_games=1)
        with self.assertRaisesRegex(AssertionError, "four jobs"):
            merge(self.base / "missing_partition", inputs[:3])
        with self.assertRaisesRegex(AssertionError, "Duplicate"):
            merge(self.base / "duplicate_partition", inputs + [inputs[0]])
        with self.assertRaisesRegex(AssertionError, "Merge all four"):
            benchmark.prepare(self.base / "reject_single_benchmark", self.base / "partition_C-R_955551")
        bad = self.base / "tampered_handoff"
        with zipfile.ZipFile(inputs[0]) as z:
            z.extractall(bad)
        bad = bad / "bs5_v4_training"
        tape_path = bad / "analysis/MATCHED_TRACE_RECEIPT.json"
        receipt = read_json(tape_path)
        receipt[next(iter(receipt))]["position_sha256"] = "f" * 64
        atomic_json(tape_path, receipt)
        manifest_path = bad / "HANDOFF_SHA256SUMS.json"
        manifest = read_json(manifest_path)
        manifest["analysis/MATCHED_TRACE_RECEIPT.json"] = file_hash(tape_path)
        atomic_json(manifest_path, manifest)
        with self.assertRaisesRegex(AssertionError, "UE tape"):
            merge(self.base / "reject_mismatched_tapes", [bad, *inputs[1:]])

    def test_11_two_partition_route_and_benchmark_profile_no_training(self):
        from scripts.experiments.merge_bs5_v4_training import merge
        from scripts.verification.profile_bs5_v4_benchmark import profile
        inputs = []
        for arm in ["C-R", "C-N"]:
            root = self.base / ("two_partition_" + arm)
            p = train.prepare(root, "cpu", diagnostic=True, arms=[arm])
            self.assertEqual(p["planned_training_steps"], 192)
            train.run(root)
            export = export_training(root)
            inputs.append(export / "training_for_benchmark.zip")
        merged = merge(self.base / "merged_two", inputs)
        self.assertEqual(len(read_json(merged / "TRAINING_PROVENANCE.json")), 4)
        receipt = profile(self.base / "benchmark_cpu_profile")
        self.assertEqual(receipt["training_updates"], 0)
        self.assertEqual(receipt["final_test_steps"], 0)
        self.assertEqual(receipt["diagnostic_slots"], 56)

    def test_12_partition_preflight_and_notebook_scopes(self):
        from scripts.verification.preflight_bs5_v4_partition import check
        receipt = check(self.base / "partition_preflight", ["C-N"], [552], "cpu")
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["main_training_steps"], 0)
        self.assertEqual(receipt["benchmark_steps"], 0)
        project = Path(__file__).resolve().parents[1]
        seen = set()
        for arm in ["CR", "CN"]:
            for seed in [551, 552]:
                path = project / "scripts/notebooks" / f"bs5_return_v4_train_{arm}_{seed}_kaggle.ipynb"
                notebook = read_json(path)
                settings = {}
                exec("".join(notebook["cells"][1]["source"]), settings)
                key = (settings["TRAIN_ARMS"][0], settings["TRAINING_SEEDS"][0])
                self.assertNotIn(key, seen)
                seen.add(key)
                self.assertTrue(settings["REQUIRE_GPU"])
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        compile("".join(cell["source"]), str(path), "exec")
        self.assertEqual(seen, {(a, s) for a in ["C-R", "C-N"] for s in [551, 552]})


if __name__ == "__main__":
    unittest.main()
