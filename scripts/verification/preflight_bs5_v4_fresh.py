"""Bounded diagnostic checks, separate from all main training/test budgets."""
from pathlib import Path
import argparse
import os
import sys
import time
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.experiments.bs5_v4_fresh_common import atomic_json, read_json, setup, source_identity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    out = a.output.resolve()
    key = dict(source_sha256=source_identity()["sha256"], runtime=setup("cpu"),
               cuda_smoke_hardware=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    if (out / "receipt.json").exists():
        receipt = read_json(out / "receipt.json")
        assert receipt["identity"] == key and receipt["status"] == "passed", "Preflight identity changed; use a fresh diagnostic output directory."
        print("Preflight already passed for this exact source/runtime.")
        return
    assert not out.exists(), "Partial preflight exists; preserve it and inspect logs, do not silently rerun."
    out.mkdir(parents=True)
    os.environ["BS5_V4_TEST_ROOT"] = str(out / "diagnostic_outputs")
    started = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_bs5_v4_fresh.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    charged = {}
    for path in out.rglob("budget_ledger.json"):
        charged[path.relative_to(out).as_posix()] = read_json(path)
    profiles = {}
    for path in (out / "diagnostic_outputs/diagnostic_benchmark/games").glob("*/*/index.json"):
        index = read_json(path)
        label = index["policy"].split("_seed")[0]
        profiles[label] = dict(short_diagnostic_slots=index["horizon"],
                              decision_median_ms=index["decision_median_ms"], decision_p95_ms=index["decision_p95_ms"],
                              runtime_s=index["runtime_s"], extrapolated_10k_s=index["runtime_s"] / index["horizon"] * 10000,
                              note="Startup-only diagnostic timing; not a guaranteed whole-horizon runtime.")
    atomic_json(out / "receipt.json", dict(identity=key, status="passed" if result.wasSuccessful() else "failed",
                tests=result.testsRun, errors=len(result.errors), failures=len(result.failures),
                skipped=len(result.skipped), benchmark_startup_timing=profiles,
                wall_s=time.monotonic()-started, diagnostic_ledgers=charged,
                main_training_steps=0, final_test_steps=0,
                scope="Diagnostic seeds only; tests include fresh training, exact resume, shared tapes, no-Relay masks, isolated previews, export/handoff and separate benchmark."))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
