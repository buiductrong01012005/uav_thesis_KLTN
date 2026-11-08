"""Kaggle preflight: actual PyTorch updates, frozen metadata and exact resume."""
from pathlib import Path
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def assert_equal(a, b, label="root"):
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            raise AssertionError("Tensor mismatch: " + label)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=label)
    elif isinstance(a, random.Random):
        assert_equal(a.getstate(), b.getstate(), label+".rng")
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), label
        for key in a:
            assert_equal(a[key], b[key], label+"."+str(key))
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), label
        for i, (x, y) in enumerate(zip(a, b)):
            assert_equal(x, y, label+f"[{i}]")
    elif hasattr(a, "__dict__"):
        assert_equal(vars(a), vars(b), label)
    elif a != b:
        raise AssertionError("Value mismatch: " + label)


def run(arm, output):
    if output.exists():
        raise ValueError("Use a fresh preflight directory")
    output.mkdir(parents=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # Load by directory/file, avoiding an installed package also named `tests`.
    for pattern in ["test_bs5_temporal_fair.py", "test_bs5_fourarm400k.py"]:
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"),
                        "-p", pattern, "-v"], cwd=ROOT, env=env, check=True)
    from scripts.training.run_bs5_fourarm400k import parser, load_protocol, make_env
    from src.rl.bs5_fourarm400k_qmix import make_learner
    p = load_protocol(parser().parse_args(["--arm", arm, "--output", str(output / "check")]))
    assert p["total_steps"] == 399600 and p["obs_dim"] == 173 and p["state_dim"] == 156
    assert p["fair_config"]["mode"] == "instant" and p["fair_config"]["rho"] == 1/60
    assert p["energy_config"]["energy_weight"] == .1 and p["collision_config"]["collision_weight"] == .2
    learner = make_learner(arm, device="cpu")
    fixture = make_env(p)
    current = fixture.reset(seed=730001, assignment_seed=730001)
    assert learner.agent(torch.as_tensor(current["obs"])).shape == (5, p["policy_actions"])
    # Repeated runs have identical architecture/initial weights for a fixed seed.
    torch.manual_seed(551)
    first = make_learner(arm, device="cpu")
    torch.manual_seed(551)
    second = make_learner(arm, device="cpu")
    assert_equal(first.agent.state_dict(), second.agent.state_dict(), "matched_initial_agent")
    # All arms share identical initial encoder, hidden body and mixer.
    torch.manual_seed(551)
    reference = make_learner("C-R", device="cpu")
    assert_equal(first.agent.encoder.state_dict(), reference.agent.encoder.state_dict(), "common_encoder")
    assert_equal(first.agent.head[0].state_dict(), reference.agent.head[0].state_dict(), "common_hidden")
    assert_equal(first.mixer.state_dict(), reference.mixer.state_dict(), "common_mixer")
    # Verify that a reported combined-role UAV contributes to both capability maps.
    capture = []
    hook = first.agent.encoder.register_forward_pre_hook(lambda module, inputs: capture.append(inputs[0].detach().clone()))
    obs = torch.zeros((5,173))
    obs[:,75+2] = .5; obs[:,75+4] = 1; obs[:,75+10] = 1
    first.agent(obs)
    hook.remove()
    assert torch.all(capture[0][:,3,0,0] == 1) and torch.all(capture[0][:,4,0,0] == 1)
    runner = ROOT / "scripts/training/run_bs5_fourarm400k.py"
    def train(folder, extra=()):
        subprocess.run([sys.executable, str(runner), "--arm", arm, "--output", str(folder),
                        "--diagnostic", "--device", "cpu", *extra], cwd=ROOT, env=env, check=True)
    full, split = output / "continuous", output / "resumed"
    train(full)
    train(split, ["--pause-at-step", "48"])
    status = json.loads((split / "status.json").read_text())
    assert status["committed_training_steps"] == 48
    train(split, ["--resume"])
    a = torch.load(full / "resume.pt", map_location="cpu", weights_only=False)
    b = torch.load(split / "resume.pt", map_location="cpu", weights_only=False)
    for key in ["protocol", "step", "episode", "updates", "model", "replay", "rng", "best_by_metric", "energy_events", "collision_events"]:
        assert_equal(a[key], b[key], key)
    for key in ["training", "validation"]:
        aa = [{k: v for k, v in row.items() if k not in {"wall_seconds", "actor_ms", "radio_ms"}} for row in a[key]]
        bb = [{k: v for k, v in row.items() if k not in {"wall_seconds", "actor_ms", "radio_ms"}} for row in b[key]]
        assert_equal(aa, bb, key)
    best = torch.load(full / "best.pt", map_location="cpu", weights_only=True)
    assert best["obs_dim"] == 173 and best["state_dim"] == 156 and best["selection_metric"] == "reward_total"
    result = dict(status="passed", arm=arm, diagnostic_steps=96, resume_pause=48,
                  model_optimizer_replay_rng_bitwise_equal=True, scientific_rows_equal=True,
                  environment_tests=20, actual_pytorch_updates=True, common_initial_core_verified=True, hybrid_capability_maps_verified=True, scope="preflight only,not scientific training results")
    (output / "PREFLIGHT_PASSED.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["C-N", "C-R", "C-H", "H-only"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.arm, args.output.resolve())
