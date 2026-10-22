from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_v4_final_evidence import learner_for, a0_episode
from src.envs.v4_final_fixed_map_env import V4FinalFixedMapConfig, V4FinalFixedMapEnv


def stats(xs):
    a = np.asarray(xs, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(len(a)),
    }


def parse_ckpt(spec: str):
    if "=" not in spec:
        raise ValueError(f"Checkpoint spec must be seed=path, got: {spec}")
    seed_s, path_s = spec.split("=", 1)
    return int(seed_s), Path(path_s).resolve()


def main():
    ap = argparse.ArgumentParser(
        description="Locked final A0 held-out evaluation across training seeds"
    )
    ap.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Repeatable seed=checkpoint_path, e.g. 81=results/.../seed81_best.pt",
    )
    ap.add_argument("--layout-start", type=int, default=1201)
    ap.add_argument("--num-layouts", type=int, default=20)
    ap.add_argument("--output", default="results/final_a0_multiseed/heldout_aggregate.json")
    args = ap.parse_args()

    # Evaluation-only runtime setting. Does not change policy/training.
    torch.set_num_threads(1)

    ckpts = [parse_ckpt(x) for x in args.checkpoint]
    layouts = list(range(args.layout_start, args.layout_start + args.num_layouts))
    per_seed = []

    for train_seed, ckpt in ckpts:
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        env = V4FinalFixedMapEnv(
            V4FinalFixedMapConfig(
                seed=1101,
                coarse_obs_size=5,
                enable_ue_mobility=False,
                enable_battery_cycle=False,
            )
        )
        learner = learner_for(env, ckpt)
        rows = []
        for layout_seed in layouts:
            rl = a0_episode(learner, layout_seed, False)
            frozen = a0_episode(learner, layout_seed, True)
            rows.append(
                {
                    "layout_seed": layout_seed,
                    "rl": rl,
                    "frozen": frozen,
                    "gain_vs_frozen": rl["mean_utility"] - frozen["mean_utility"],
                }
            )
        per_seed.append(
            {
                "training_seed": train_seed,
                "checkpoint": str(ckpt),
                "rl_mean_utility": stats([r["rl"]["mean_utility"] for r in rows]),
                "frozen_mean_utility": stats([r["frozen"]["mean_utility"] for r in rows]),
                "gain_vs_frozen": stats([r["gain_vs_frozen"] for r in rows]),
                "win_fraction_vs_frozen": float(np.mean([r["gain_vs_frozen"] > 0 for r in rows])),
                "rows": rows,
            }
        )
        print(
            f"seed{train_seed}: heldout RL={per_seed[-1]['rl_mean_utility']['mean']:.6f}, "
            f"gain={per_seed[-1]['gain_vs_frozen']['mean']:.6f}",
            flush=True,
        )

    seed_means = [x["rl_mean_utility"]["mean"] for x in per_seed]
    seed_gains = [x["gain_vs_frozen"]["mean"] for x in per_seed]
    out = {
        "protocol": {
            "phase": "A0_final_heldout",
            "training_seeds": [s for s, _ in ckpts],
            "layout_seeds": layouts,
            "num_layouts_per_training_seed": len(layouts),
            "roles": "fixed 4 Access + 1 Relay",
            "battery_cycle": False,
            "ue_mobility": False,
            "note": "Held-out environment evaluation of each best training-seed checkpoint; no tuning.",
        },
        "per_training_seed": per_seed,
        "across_training_seeds": {
            "heldout_rl_seed_mean": stats(seed_means),
            "heldout_gain_seed_mean": stats(seed_gains),
            "mean_win_fraction": float(np.mean([x["win_fraction_vs_frozen"] for x in per_seed])),
        },
    }
    outp = Path(args.output)
    if not outp.is_absolute():
        outp = (ROOT / outp).resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(
        json.dumps({"output": str(outp), **out["across_training_seeds"]}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
