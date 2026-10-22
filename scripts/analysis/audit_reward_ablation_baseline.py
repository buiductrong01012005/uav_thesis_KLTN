"""Re-evaluate existing A0 checkpoints on original, information-matched utility KPIs.

No training. Incrementally persist after each evaluation episode.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.training.run_thesis_no_curriculum_fulltask import (
    make_env,
    make_learner,
    evaluate,
    TEST_SEEDS,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[91, 92, 93])
    p.add_argument("--test-seeds", type=int, nargs="+", default=list(TEST_SEEDS))
    p.add_argument("--outdir", default="results/reward_ablation")
    p.add_argument(
        "--checkpoint-dir",
        default="results/thesis_curriculum_ablation/direct_full_task_randomized",
    )
    args = p.parse_args()
    torch.set_num_threads(1)
    dest = Path(args.outdir)
    if not dest.is_absolute():
        dest = ROOT / dest
    dest.mkdir(parents=True, exist_ok=True)
    cpdir = Path(args.checkpoint_dir)
    if not cpdir.is_absolute():
        cpdir = ROOT / cpdir
    report = {
        "description": "Existing QMIX A0 original checkpoint; original utility evaluation; no training",
        "test_seeds": args.test_seeds,
        "training_seeds": args.seeds,
        "rows": [],
    }
    path = dest / "a0_reference_exact_kpis.json"
    env = make_env(args.test_seeds[0], 150)
    for seed in args.seeds:
        cp = cpdir / f"seed{seed}_best_final_block.pt"
        if not cp.exists():
            raise FileNotFoundError(cp)
        learner = make_learner(env, "cpu", 3e-4)
        learner.load(str(cp))
        ev = evaluate(learner, args.test_seeds, 150)
        row = {
            "seed": seed,
            **{
                k: ev[k]
                for k in [
                    "mean_utility",
                    "std_utility",
                    "mean_battery_drain_per_step",
                    "mean_role_switches_per_step",
                    "mean_relays",
                ]
            },
        }
        report["rows"].append(row)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("A0_RESULT", json.dumps(row), flush=True)
    report["aggregate"] = {
        k: {
            "mean": float(np.mean([row[k] for row in report["rows"]])),
            "std": float(np.std([row[k] for row in report["rows"]])),
        }
        for k in [
            "mean_utility",
            "mean_battery_drain_per_step",
            "mean_role_switches_per_step",
            "mean_relays",
        ]
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("A0_AGGREGATE", json.dumps(report["aggregate"]), flush=True)


if __name__ == "__main__":
    main()
