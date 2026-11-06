"""Evaluate a frozen checkpoint under multi-hop physics; no optimizer/replay.

Only hash-pinned C-N reference checkpoints may use the legacy-checkpoint path.
This reuses historical C-N training, never pretends it was trained multi-hop.
"""

import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.training import run_bs5_multihop_train as train
from src.rl.bs5_multihop_qmix import make_learner


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(a):
    args = train.parser().parse_args(
        ["--arm", a.arm, "--preset", a.preset, "--output", str(a.output), "--device", a.device]
    )
    p = train.load_protocol(args)
    p.update(full_horizon=a.horizon, validation_seeds=a.seeds, diagnostic=a.diagnostic)
    cp = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    reference = None
    if cp.get("training_env_version") != "bs5_multihop_training_1":
        if a.arm != "C-N" or a.reference_manifest is None:
            raise ValueError("Legacy reuse is only supported for pinned C-N references")
        refs = json.loads(a.reference_manifest.read_text())["checkpoints"]
        reference = next(
            (r for r in refs if r["sha256"] == sha(a.checkpoint) and r["preset"] == a.preset), None
        )
        if reference is None:
            raise ValueError("Reference checkpoint hash/preset not approved")
        if reference["arm"] != "C-N" or cp["selected_step"] != reference["selected_step"]:
            raise ValueError("Wrong reference identity")
        if reference["reward_weights"] != p["reward_weights"]:
            raise ValueError("Reference reward mismatch")
    else:
        if cp["arm"] != a.arm or cp["preset"] != a.preset or cp["n_actions"] != p["policy_actions"]:
            raise ValueError("Checkpoint arm/preset/actions mismatch")
        # Same-science verification; runtime and training target may differ at evaluation.
        if cp["reward_version"] != p["reward_version"]:
            raise ValueError("Reward version mismatch")
    identity = dict(
        checkpoint_sha256=sha(a.checkpoint),
        arm=a.arm,
        preset=a.preset,
        horizon=a.horizon,
        seeds=a.seeds,
        diagnostic=a.diagnostic,
        source_sha256=p["source_sha256"],
    )
    if a.output.exists():
        receipt = a.output / "evaluation_receipt.json"
        if receipt.is_file() and json.loads(receipt.read_text())["identity"] == identity:
            print("Reuse completed identical evaluation", a.output)
            return
        raise ValueError("Evaluation output exists with a different or incomplete identity")
    a.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    learner = make_learner(a.arm, device=a.device)
    learner.agent.load_state_dict(cp["agent"])
    learner.mixer.load_state_dict(cp["mixer"])
    rows = train.evaluate(learner, p, cp["selected_step"], a.output)
    train.write_csv(a.output / "validation.csv", rows)
    train.atomic_json(a.output / "protocol.json", p)
    train.atomic_json(
        a.output / "status.json",
        dict(
            status="reference_evaluated" if reference else "checkpoint_evaluated",
            training_env_version="bs5_multihop_training_1",
            arm=a.arm,
            preset=a.preset,
            committed_training_steps=0,
            target_steps=0,
            new_training_performed=False,
            source_training_budget=reference["training_budget"] if reference else None,
            reference_selected_step=cp["selected_step"],
            best_by_metric={},
            diagnostic=a.diagnostic,
            device=a.device,
        ),
    )
    train.atomic_json(
        a.output / "evaluation_receipt.json",
        dict(
            identity=identity,
            reference=reference,
            selected_step=cp["selected_step"],
            mean_metrics={
                k: float(np.mean([r[k] for r in rows]))
                for k in [
                    "throughput_mbps",
                    "service_coverage_ues",
                    "qos_ues",
                    "jain_time_average",
                    "unique_served_ues",
                ]
            },
            scope="Frozen checkpoint evaluation; no new training; validation seeds are not a held-out benchmark",
        ),
    )
    print(
        json.dumps(
            json.loads((a.output / "evaluation_receipt.json").read_text())["mean_metrics"], indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["C-N", "C-R", "C-H", "H-only"], required=True)
    parser.add_argument("--preset", choices=["balanced", "tc_priority"], default="balanced")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[620001, 620002])
    parser.add_argument("--diagnostic", action="store_true")
    a = parser.parse_args()
    if a.horizon <= 0:
        parser.error("horizon must be positive")
    run(a)
