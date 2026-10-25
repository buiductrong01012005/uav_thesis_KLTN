"""Bounded, persistent preflight. Never trains the main experiment seed."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.training import run_bs5_cr_h1200_return as runner
from src.envs.v4_fixed_mobile_battery_env import V4FixedMobileBatteryEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    runner.setup_runtime(args.device, 1)
    fingerprint = dict(
        source=runner.source_identity()["sha256"], runtime=runner.runtime(args.device)
    )
    receipt = root / "receipt.json"
    if receipt.exists():
        previous = runner.read_json(receipt)
        if previous.get("fingerprint") == fingerprint and previous.get("status") == "passed":
            print(json.dumps(previous))
            return
        raise RuntimeError(
            "Preflight exists with different source/runtime or incomplete/failed status. Preserve evidence; no automatic retry."
        )
    ledger = dict(
        status="started",
        fingerprint=fingerprint,
        caps=dict(diagnostic_train=512, control_eval=4096),
        charged=dict(diagnostic_train=0, control_eval=0),
        main_train_steps=0,
        note="reserved before every physical env step, includes legacy fixture comparator",
    )
    runner.atomic_json(receipt, ledger)
    original = V4FixedMobileBatteryEnv.step

    def counted(env, actions):
        kind = (
            "diagnostic_train"
            if getattr(env, "_audit_phase", "control") == "train"
            else "control_eval"
        )
        if ledger["charged"][kind] >= ledger["caps"][kind]:
            raise RuntimeError("Preflight compute cap reached")
        ledger["charged"][kind] += 1
        runner.atomic_json(receipt, ledger)
        return original(env, actions)

    V4FixedMobileBatteryEnv.step = counted
    try:
        import pytest

        status = pytest.main(
            [
                "-q",
                "-p",
                "no:cacheprovider",
                str(ROOT / "tests/test_bs5_cr_h1200_return.py"),
                str(ROOT / "tests/test_bs5_energy_return.py"),
                str(ROOT / "tests/test_bs5_collision_contract.py"),
                "--basetemp",
                str(root / "test_data"),
            ]
        )
        if status:
            raise RuntimeError("Preflight pytest failed: " + str(status))
        if args.device == "cuda":
            gpu = root / "cuda_compatibility"
            runner.prepare(gpu, "cuda", diagnostic=True)
            runner.run(gpu, stop_after=24)
        ledger["status"] = "passed"
        ledger["gpu_exact_resume_tested"] = False
    except BaseException as error:
        ledger.update(status="failed", error=repr(error))
        raise
    finally:
        V4FixedMobileBatteryEnv.step = original
        runner.atomic_json(receipt, ledger)
    print(json.dumps(ledger))


if __name__ == "__main__":
    main()
