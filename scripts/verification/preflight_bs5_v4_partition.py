"""Small partition-specific train/resume check; never benchmark or train main seeds."""
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.experiments.bs5_v4_fresh_common import atomic_json, read_json, setup, source_identity, env_for, full_condition
from scripts.training import run_bs5_v4_fresh_train as train


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def check(output, arms, seeds, device):
    out = Path(output).resolve()
    identity = dict(source_sha256=source_identity()["sha256"], runtime=setup(device), arms=arms, seeds=seeds)
    if (out / "receipt.json").exists():
        receipt = read_json(out / "receipt.json")
        assert receipt["identity"] == identity and receipt["status"] == "passed"
        print("Partition preflight already passed.")
        return receipt
    assert not out.exists(), "Partial preflight exists; preserve and inspect it."
    out.mkdir(parents=True)
    start = time.monotonic()
    try:
        roots = [out / "continuous", out / "resumed"]
        diagnostic_seed = 955000 + seeds[0]
        for root in roots:
            train.prepare(root, identity["runtime"]["device"], diagnostic=True, arms=[arms[0]], seeds=[diagnostic_seed])
        train.run(roots[0])
        train.run(roots[1], max_units=4)
        assert read_json(roots[1] / "status.json")["status"] == "paused_at_game_boundary"
        train.run(roots[1])
        states = []
        for root in roots:
            run = root / "runs" / f"{arms[0]}_seed{diagnostic_seed}"
            progress = train.read_progress(run)
            assert progress["updates"] > 0
            states.append(torch.load(run / progress["resume_file"], map_location="cpu", weights_only=False))
        for key in ["models", "optimizer", "replay", "exploration", "rng"]:
            equal(states[0][key], states[1][key])
        p = read_json(roots[0] / "protocol.json")
        spec = dict(condition=full_condition(p), horizon=6, scenario_seed=284501, assignment_seed=284501,
                    residual_battery=30., diagnostic=True)
        envs = [env_for(p, spec, out / "control_tapes", arm) for arm in ["C-R", "C-N"]]
        outputs = [e.reset(seed=284501, assignment_seed=284501, diagnostic=True) for e in envs]
        assert envs[0].trace_receipt == envs[1].trace_receipt
        for slot in range(6):
            for i, env in enumerate(envs):
                actions = [int(np.flatnonzero(m)[-1]) for m in outputs[i]["action_mask"]]
                outputs[i] = env.step(actions)
            assert np.array_equal(envs[0].ue_positions, envs[1].ue_positions)
        ledgers = {str(root.name): read_json(next(root.glob("runs/*/budget_ledger.json"))) for root in roots}
        receipt = dict(status="passed", identity=identity, wall_s=time.monotonic()-start,
                       diagnostic_ledgers=ledgers, extra_control_slots=12,
                       main_training_steps=0, final_test_steps=0, benchmark_steps=0,
                       checks=["forward/backward", "exact resume models/Adam/replay/all RNG", "paired UE tapes", "no-Relay masks/router", "S4 bootstrap"],
                       scope="Only diagnostic training seed955xxx and control scenario284501; no main training or benchmark")
        atomic_json(out / "receipt.json", receipt)
        print("Partition preflight PASS:", identity["runtime"]["device"], round(receipt["wall_s"], 1), "seconds")
        return receipt
    except BaseException as exc:
        atomic_json(out / "receipt.json", dict(status="failed", identity=identity, error=repr(exc)))
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = ap.parse_args()
    check(args.output, args.arms.split(","), [int(x) for x in args.seeds.split(",")], args.device)
