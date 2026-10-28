"""Rescore inference checkpoints with the full TCF metrics, without training."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.envs.bs5_tcf_env import TCFWeights
from src.rl.bs5_service_v2_qmix import ServiceV2QMIX
from scripts.training import run_bs5_tcf120k as training


def run(args):
    original = json.loads(args.protocol.read_text())
    for name, expected in original["source_sha256"].items():
        if training.sha(ROOT / name) != expected:
            raise RuntimeError("Original source differs: " + name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("protocol_sha256") != training.digest(original):
        raise RuntimeError("Checkpoint/protocol lineage mismatch")
    if checkpoint.get("obs_dim") != 148 or checkpoint.get("state_dim") != 66:
        raise ValueError("Expected service-v2 inference checkpoint")
    p = dict(original)
    presets = json.loads(training.CONFIG.read_text())["presets"]
    weights = dict(presets[args.preset])
    if args.weights is not None:
        weights = dict(zip(["throughput", "coverage", "fairness"], args.weights))
    TCFWeights(**weights)
    p.update(reward_weights=weights, coverage_floor_mbps=args.coverage_floor, device="cpu")
    if args.slots:
        p["full_horizon"] = args.slots
    if args.seeds:
        p["validation_seeds"] = args.seeds
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    learner = ServiceV2QMIX(device="cpu")
    learner.agent.load_state_dict(checkpoint["agent"])
    learner.mixer.load_state_dict(checkpoint["mixer"])
    rows = training.evaluate(learner, p, checkpoint["selected_step"], args.output)
    summary = {
        k: float(np.mean([r[k] for r in rows]))
        for k in rows[0]
        if k not in {"seed", "step", "slots"}
    }
    summary.update(
        arm=original["arm"],
        selected_step=checkpoint["selected_step"],
        slots=p["full_horizon"],
        seeds=p["validation_seeds"],
        checkpoint_sha256=training.sha(args.checkpoint),
        reward_weights=weights,
        coverage_floor_mbps=args.coverage_floor,
        score_is_posthoc=True,
        original_checkpoint_selection=checkpoint.get("selection_metric", "utility"),
    )
    training.atomic_json(args.output / "summary.json", summary)
    training.write_csv(args.output / "validation.csv", rows)
    training.atomic_json(
        args.output / "evaluation_receipt.json",
        dict(
            original_protocol=original,
            original_checkpoint_sha256=training.sha(args.checkpoint),
            evaluation=p,
            evaluation_source_sha256=training.source_lock(),
            current_runtime=dict(
                python=sys.version.split()[0], numpy=np.__version__, torch=str(torch.__version__)
            ),
            scope="Fixed validation rescore only; no parameter updates or checkpoint selection",
        ),
    )
    print(json.dumps(summary, indent=2), flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", default="balanced")
    parser.add_argument("--weights", type=float, nargs=3)
    parser.add_argument("--coverage-floor", type=float, default=1.0)
    parser.add_argument("--slots", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+")
    run(parser.parse_args())
