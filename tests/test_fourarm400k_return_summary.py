"""Regression: the last simulation slot is saved before Returning summary."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts.analysis.report_fourarm400k_multiseed import scan
from scripts.evaluation import run_fourarm400k_heuristics as core
from scripts.evaluation.fourarm400k_return_summary import return_metrics


class ReturnSummaryTests(unittest.TestCase):
    def test_missing_plan_preserves_measured_trip_and_slot_boundary(self):
        events = [
            dict(event="return_start", agent=0, slot=1, route_m=None),
            dict(event="return_start", agent=1, slot=1, route_m=50),
            dict(event="swap_start", agent=0, slot=3),
            dict(event="swap_start", agent=1, slot=3),
        ]
        distance = [[50, 50], [0, 50], [50, 50]]
        result = return_metrics(events, distance)
        self.assertEqual(result["mean_return_planned_m"], 50)
        self.assertEqual(result["mean_return_actual_m"], 75)
        self.assertEqual(result["mean_return_duration_slots"], 2)
        self.assertEqual(result["return_missing_plan_events"], 1)
        self.assertEqual(result["return_detour_events"], 1)
        unknown = return_metrics([events[0], events[2]], distance)
        self.assertIsNone(unknown["mean_return_planned_m"])
        self.assertEqual(unknown["mean_return_actual_m"], 50)

    def test_old_final_snapshot_can_commit_without_replaying_slots(self):
        protocol = core.make_protocol("RANDOM", 1, diagnostic=True)
        protocol["runtime"] = core.runtime()
        seed = protocol["target_seeds"][0]
        state = core.start_state(protocol, seed)
        for _ in range(protocol["horizon"]):
            core.advance(state, protocol)
        # Reproduce a valid missing plan on one completed trip, alongside a
        # known plan. The rest of the state comes from the real simulator.
        state["env"].energy_events = [
            dict(event="return_start", agent=0, slot=5, route_m=None),
            dict(event="return_start", agent=1, slot=5, route_m=50),
            dict(event="swap_start", agent=0, slot=9),
            dict(event="swap_start", agent=1, slot=9),
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "random"
            folder = output / f"seed{seed}"
            folder.mkdir(parents=True)
            core.atomic_json(output / "protocol.json", protocol)
            core.save_state(folder, protocol, state)
            with self.assertRaises(TypeError):
                core.save_seed(folder, protocol, state, True)
            self.assertFalse((folder / "COMMIT.json").exists())
            entry = core.ROOT / "scripts/evaluation/run_fourarm400k_returnfix.py"
            result = subprocess.run([
                sys.executable, str(entry), "baseline", "--label", "RANDOM",
                "--part", "1", "--diagnostic", "--output", str(output),
            ], cwd=core.ROOT, capture_output=True, text=True, check=True)
            self.assertIn(f"slot {protocol['horizon']}", result.stdout)
            self.assertEqual(json.loads((output / "protocol.json").read_text()), protocol)
            summary = core.verify_seed_commit(folder, protocol)
            self.assertEqual(summary["ue_trace_sha256"], state["ue_trace_sha256"])
            self.assertEqual(summary["return_missing_plan_events"], 1)
            self.assertFalse((folder / "progress.json").exists())
            with np.load(folder / "trace.npz", allow_pickle=False) as trace:
                np.testing.assert_array_equal(trace["ue_rates"], state["arrays"]["ue_rates"])
            rows, _, _ = scan(root, allow_diagnostic=True, include_random=True)
            self.assertEqual(rows[("RANDOM", None, seed)]["return_missing_plan_events"], 1)
            self.assertEqual(rows[("RANDOM", None, seed)]["mean_return_duration_slots"], 4)
