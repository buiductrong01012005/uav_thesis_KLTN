"""Short Windows train/eval, three baselines and six CSV figure exports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-train-eval", action="store_true",
                        help="Verify and retain an already completed diagnostic train/eval")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs") or output == ROOT / "runs":
        parser.error("Use an output below runs/")
    if output.exists() and not args.reuse_train_eval:
        parser.error("Use a fresh output directory")
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", MPLBACKEND="Agg",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    receipt = dict(commands=[], reused_train_eval=args.reuse_train_eval)

    def save():
        (output / "smoke_report.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    def command(name, *options):
        cmd = [sys.executable, "-B", *map(str, options)]
        start = time.monotonic()
        with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
            done = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt["commands"].append(dict(name=name, argv=cmd, exit_code=done.returncode,
                                        seconds=round(time.monotonic() - start, 2)))
        save()
        print(name, done.returncode, flush=True)
        if done.returncode:
            raise RuntimeError(f"Failed: {logs / (name + '.log')}")

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    try:
        if not args.reuse_train_eval:
            command("train", "scripts/train.py", "--arm", "C-R", "--seed", "551", "--steps", "96",
                    "--diagnostic", "--output", output / "train")
            command("eval", "run.py", "eval", "--checkpoint", output / "train/best.pt",
                    "--test-seed", "930201", "--slots", "40", "--output", output / "eval")
        status = read(output / "train/status.json")
        assert status["status"] == "complete" and status["committed_training_steps"] == 96
        evaluation = read(output / "eval/seed930201/summary.json")
        assert evaluation["complete"] and evaluation["slots"] == 40
        protocol = read(output / "eval/protocol.json")
        model = output / "train/best.pt"
        assert protocol["selection"]["checkpoint_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
        receipt.update(training_steps=96, training_updates=status["training_updates"], evaluation_slots=40,
                       checkpoint_sha256=protocol["selection"]["checkpoint_sha256"])
        for label in ("RANDOM", "DYNAMIC_AR", "HEURISTIC_HO"):
            command(label, "-m", "scripts.w00", "benchmark", "baseline", "--label", label,
                    "--part", "1", "--seeds", "930201", "--slots", "8", "--run-id", "compact-smoke",
                    "--output", output / "baselines" / label.lower())
            summary = read(output / "baselines" / label.lower() / "seed930201/summary.json")
            assert summary["complete"] and summary["slots"] == 8
        for name in ("learning", "demand_breakdown", "throughput_blocks", "ue_service", "operation", "throughput_cost"):
            command(name, "-m", "scripts.plotting.chapter5.plot_" + name,
                    "--output", output / "figures")
        products = sorted(p for p in (output / "figures").rglob("*") if p.is_file())
        counts = {suffix: sum(p.suffix == suffix for p in products) for suffix in (".png", ".pdf", ".csv", ".json")}
        assert all(n == 6 for n in counts.values()) and all(p.stat().st_size for p in products)
        receipt.update(figure_products=counts, status="passed")
    except BaseException as exc:
        receipt.update(status="failed", error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        save()
    print(output / "smoke_report.json", flush=True)


if __name__ == "__main__":
    main()
