"""Run the real W00 pipeline with 96 training steps and 40 evaluation slots."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def compare_training(first, resumed):
    import torch
    from scripts.verification.verify_bs5_fourarm400k import assert_equal

    a = torch.load(first / "resume.pt", map_location="cpu", weights_only=False)
    b = torch.load(resumed / "resume.pt", map_location="cpu", weights_only=False)
    for key in ("protocol", "step", "episode", "updates", "model", "replay", "rng",
                "best_by_metric", "energy_events", "collision_events"):
        assert_equal(a[key], b[key], key)
    for key in ("training", "validation"):
        def scientific(rows):
            return [{k: v for k, v in row.items()
                     if k not in {"wall_seconds", "actor_ms", "radio_ms"}} for row in rows]
        assert_equal(scientific(a[key]), scientific(b[key]), key)


def verify_benchmarks(output):
    import numpy as np
    import pandas as pd
    if sys.platform == "win32":
        sys.path.insert(0, str(ROOT / "scripts/compat/windows"))
    from scripts.evaluation import run_fourarm400k_heuristics as core

    rows, traces, hashes = [], [], []
    for name in ("benchmark", "benchmark_repeat", "baseline"):
        folder = output / name
        p = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
        seed_folder = folder / "seed940201"
        summary = core.verify_seed_commit(seed_folder, p)
        frame = pd.read_csv(seed_folder / "slots.csv")
        assert p["horizon"] == summary["slots"] == summary["target_slots"] == 40
        assert summary["seed"] == 940201 and summary["complete"]
        assert len(frame) == 40 and frame.slot.tolist() == list(range(1, 41))
        assert np.isfinite(frame.select_dtypes(include="number").to_numpy()).all()
        for metric in ("qos_fraction", "service_coverage_fraction", "positive_fraction"):
            assert frame[metric].between(-1e-12, 1 + 1e-12).all(), metric
        for metric in ("throughput_mbps", "qos_fraction", "reward_total"):
            np.testing.assert_allclose(summary[metric], frame[metric].mean(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(summary["delivered_mbit"], frame.throughput_mbps.sum() * 5, rtol=1e-12)
        hashes.append(summary["ue_trace_sha256"])
        rows.append(frame.drop(columns=["actor_ms", "radio_ms", "actor_exceeds_slot"]))
        with np.load(seed_folder / "trace.npz", allow_pickle=False) as z:
            traces.append({k: z[k].copy() for k in z.files})
    np.testing.assert_allclose(rows[0], rows[1], rtol=1e-12, atol=1e-12)
    for key in traces[0]:
        np.testing.assert_array_equal(traces[0][key], traces[1][key], err_msg=key)
    assert len(set(hashes)) == 1, "Same-horizon UE traces differ between configurations"


def run(output, *, with_ablation=False, with_tests=False):
    output = output.resolve()
    if not output.is_relative_to((ROOT / "runs/local").resolve()) or output.exists():
        raise ValueError("Use a fresh output directory under runs/local/")
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", MPLBACKEND="Agg",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "scripts/compat/windows"), str(ROOT)] if sys.platform == "win32" else [str(ROOT)])
    receipt = dict(status="running", training_steps=96, evaluation_slots=40, commands=[])

    def save():
        (output / "smoke_report.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    def command(name, *args):
        cmd = [sys.executable, "-B", *map(str, args)]
        started = time.monotonic()
        with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt["commands"].append(dict(name=name, argv=cmd, exit_code=result.returncode,
                                        seconds=round(time.monotonic() - started, 2)))
        save()
        print(f"{name}: exit {result.returncode}", flush=True)
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {logs / (name + '.log')}")

    try:
        command("assets", "-m", "scripts.verification.verify_w00_import", "--output", output / "assets.json")
        train = ["-m", "scripts.w00", "train", "--arm", "C-R", "--training-seed", "551", "--diagnostic"]
        command("train_full", *train, "--output", output / "train_full")
        command("train_pause", *train, "--pause-at-step", "48", "--output", output / "train_resume")
        paused = json.loads((output / "train_resume/status.json").read_text(encoding="utf-8"))
        assert paused["committed_training_steps"] == 48
        command("train_resume", *train, "--resume", "--output", output / "train_resume")
        compare_training(output / "train_full", output / "train_resume")
        receipt["training_resume_bitwise_equal"] = True
        if with_ablation:
            for variant in ("mlp", "no_curriculum"):
                command(f"train_{variant}", *train, "--variant", variant, "--output", output / f"train_{variant}")
                status = json.loads((output / f"train_{variant}/status.json").read_text(encoding="utf-8"))
                assert status["status"] == "complete" and status["committed_training_steps"] == 96
        benchmark = ["-m", "scripts.w00", "benchmark", "drl", "--arm", "C-R", "--training-seed", "551",
                     "--diagnostic", "--seeds", "940201", "--slots", "40", "--run-id", output.name]
        command("benchmark_pause", *benchmark, "--stop-after-slots", "17", "--output", output / "benchmark")
        paused = json.loads((output / "benchmark/seed940201/summary.json").read_text(encoding="utf-8"))
        assert paused["slots"] == 17 and not paused["complete"]
        command("benchmark_resume", *benchmark, "--output", output / "benchmark")
        command("benchmark_repeat", *benchmark, "--output", output / "benchmark_repeat")
        command("baseline", "-m", "scripts.w00", "benchmark", "baseline", "--label", "RANDOM",
                "--diagnostic", "--seeds", "940201", "--slots", "40", "--run-id", output.name,
                "--output", output / "baseline")
        verify_benchmarks(output)
        receipt["benchmark_repeat_and_resume_equal"] = True
        command("figures", "-m", "scripts.plotting.plot_final400k", "--input", "results/w00_main/benchmark",
                "--training", "results/w00_main/training", "--output", output / "figures")
        (output / "figures_v12").mkdir()
        command("figures_v12", "scripts/plotting/plot_w00_7cfg.py", "results/w00_main/benchmark", output / "figures_v12")
        command("metric", "-m", "scripts.plotting.plot_w00_metric", "--input", "results/w00_main/benchmark/per_run.csv",
                "--metric", "joules_per_mbit", "--unit", "J/Mbit", "--by", "seed", "--output", output / "metric")
        for name in ("learning", "benchmark", "waterfall"):
            assert (output / f"figures/{name}.png").stat().st_size > 0
        if with_tests:
            command("pytest", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "tests/test_bs5_lp_roundoff.py", "tests/test_bs5_lpfix_io.py",
                    "tests/test_fourarm400k_return_summary.py", "tests/test_final400k_delivery.py",
                    "tests/test_fourarm400k_multiseed.py", "tests/test_fourarm400k_heuristic_io.py",
                    "tests/test_bs5_fourarm400k_io.py")
        receipt["status"] = "passed"
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        save()
    print(f"Passed: {output / 'smoke_report.json'}", flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-ablation", action="store_true", help="Requires the separate restored ablation snapshot")
    parser.add_argument("--with-tests", action="store_true", help="Also run the focused existing regression tests")
    args = parser.parse_args()
    run(args.output, with_ablation=args.with_ablation, with_tests=args.with_tests)
