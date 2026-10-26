"""CPU startup timing/physics check on diagnostic controls; zero training updates."""
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.experiments.bs5_v4_fresh_common import (
    Budget, BS5CurriculumQMIX, act, atomic_json, env_for, full_condition,
    read_json, rollout, seed_all, setup, source_identity,
)
from scripts.analysis.bs5_v4_fresh_report import checked_game
from src.baselines.return_v4_safe_greedy import make_greedy


def profile(output):
    out = Path(output).resolve()
    identity = dict(source_sha256=source_identity()["sha256"], runtime=setup("cpu"))
    if (out / "receipt.json").exists():
        receipt = read_json(out / "receipt.json")
        assert receipt["identity"] == identity and receipt["status"] == "passed"
        return receipt
    assert not out.exists(), "Partial profile: preserve/inspect before retry."
    out.mkdir(parents=True)
    p = read_json(ROOT / "configs/bs5_return_v4_fresh_train.json")
    labels = ["C-R", "C-N", "Random-Relay", "5A-Recovery", "4A1R-Recovery", "3A2R-Recovery", "Dynamic-Recovery"]
    budget = Budget(out / "budget_ledger.json", {x: 8 for x in labels})
    spec = dict(condition=full_condition(p), horizon=8, scenario_seed=284901, assignment_seed=284901,
                residual_battery=30., diagnostic=True)
    profiles = {}
    started = time.monotonic()
    for label in labels:
        seed_all(955551)
        rng = np.random.default_rng(955901)
        learner = BS5CurriculumQMIX(env_for(p, spec, out / "tapes", label), device="cpu") if label in ["C-R", "C-N"] else None
        heuristic = None
        def choose(env, data):
            nonlocal heuristic
            if learner:
                return act(learner, data, 0., rng), 0., 0
            if label == "Random-Relay":
                return [int(rng.choice(np.flatnonzero(m))) for m in data["action_mask"]], 0., 0
            if heuristic is None:
                heuristic = make_greedy(label, env)
            elif env.t:
                heuristic.observe_service(float(env.last_service_snapshot["network"]["utility"]))
            n = heuristic.preview_calls
            actions = heuristic.act(data["action_mask"])
            return actions, 0., heuristic.preview_calls-n
        directory = out / "controls" / label
        index = rollout(directory, p, spec, label, out / "tapes", budget, label, choose)
        checked_game(directory, deep=True)
        profiles[label] = dict(short_diagnostic_slots=8, runtime_s=index["runtime_s"],
                               decision_median_ms=index["decision_median_ms"], decision_p95_ms=index["decision_p95_ms"],
                               extrapolated_10k_s=index["runtime_s"]/8*10000,
                               note="Short startup control only, not a whole-horizon guarantee; random initial actor weights, no learning")
    receipt = dict(status="passed", identity=identity, benchmark_startup_timing=profiles,
                   wall_s=time.monotonic()-started, diagnostic_slots=56,
                   training_updates=0, main_training_steps=0, final_test_steps=0)
    atomic_json(out / "receipt.json", receipt)
    print("Benchmark CPU controls PASS; no training updates, no final test.")
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    profile(ap.parse_args().output)
