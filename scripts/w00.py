"""Windows entry point for W00; scientific logic stays in the existing runners."""
from pathlib import Path
import argparse
import hashlib
import os
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VERSION = "w00_native_windows_v1"


def enable_windows():
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    if sys.platform == "win32":
        sys.path.insert(0, str(ROOT / "scripts/compat/windows"))


def local_output(path):
    path = Path(path).resolve()
    runs = (ROOT / "runs").resolve()
    if path == runs or not path.is_relative_to(runs):
        raise ValueError("New W00 runs must use an output directory under runs/")
    return path


def benchmark(argv):
    from scripts.evaluation import run_fourarm400k_heuristics as core
    from scripts.evaluation import run_fourarm400k_drl_benchmark as drl
    from scripts.evaluation.fourarm400k_return_summary import extend_summary

    parser = argparse.ArgumentParser(description="W00 native benchmark, optional separate custom horizon")
    parser.add_argument("mode", choices=("drl", "baseline"))
    parser.add_argument("--arm", choices=drl.ARMS)
    parser.add_argument("--training-seed", type=int, choices=drl.SEEDS, default=551)
    parser.add_argument("--models", type=Path, default=ROOT / "checkpoints/w00_main/FOURARM400K_FROZEN_MODELS.zip")
    parser.add_argument("--label", choices=tuple(core.LABELS))
    parser.add_argument("--part", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--slots", type=int, help="Complete custom horizon per seed; separate from frozen 10000")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=local_output, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--stop-after-slots", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=36000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args(argv)
    if args.slots is not None and args.slots < 1:
        parser.error("--slots must be positive; no rounding")
    if args.save_every < 1 or args.log_every < 1 or args.stop_after_slots < 0:
        parser.error("Positive save/log intervals and nonnegative stop-after-slots required")
    if not args.run_id.strip() or any(c in args.run_id for c in '/\\'):
        parser.error("--run-id must be a nonempty name")
    if args.mode == "drl" and args.arm is None:
        parser.error("drl requires --arm")
    if args.mode == "baseline" and args.label is None:
        parser.error("baseline requires --label")

    original = core.make_protocol
    adapter_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def protocol(label, part, diagnostic=False, seeds=None):
        p = original(label, part, diagnostic, seeds)
        if args.slots is not None:
            p["horizon"] = args.slots
            p["config"]["horizon"] = args.slots
            p["environment_contract"]["horizon"] = args.slots
        p["native_adapter"] = dict(version=VERSION, sha256=adapter_hash, run_id=args.run_id,
                                   custom_horizon=args.slots is not None,
                                   trace_comparison="same horizon and scenario seed only")
        # Include native provenance in the environment fingerprint, including for DRL.
        p["environment_contract"]["native_adapter"] = p["native_adapter"]
        p["environment_fingerprint"] = core.digest(p["environment_contract"])
        p["comparison_fingerprint"] = core.digest({k: v for k, v in p.items()
                                                   if k != "comparison_fingerprint"})
        return p

    core.make_protocol = protocol
    core.extend_summary = extend_summary
    signal.signal(signal.SIGTERM, core.stop_handler)
    signal.signal(signal.SIGINT, core.stop_handler)
    try:
        return drl.run(args) if args.mode == "drl" else core.run(args)
    finally:
        core.make_protocol = original


def train(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", choices=("main", "mlp", "no_curriculum"), default="main")
    args, remaining = parser.parse_known_args(argv)
    if args.variant == "main":
        from scripts.training import run_bs5_fourarm400k as runner
        options = runner.parser().parse_args(remaining)
        options.output = local_output(options.output)
        signal.signal(signal.SIGTERM, runner.stop_handler)
        signal.signal(signal.SIGINT, runner.stop_handler)
        return runner.run(options)
    # A fresh interpreter keeps ablation src/ imports isolated from the main model.
    source = ROOT / "archive/frozen_sources/w00_ablation/extracted/uav_thesis"
    if not source.is_dir():
        raise FileNotFoundError("Restore the separate W00 ablation snapshot first")
    command = [sys.executable, "-B", str(ROOT / "scripts/compat/train_w00_snapshot.py"),
               "--snapshot", str(source), "--variant", args.variant, *remaining]
    subprocess.run(command, check=True, cwd=ROOT)


def evaluate(argv):
    from scripts.evaluation.eval_w00_checkpoint import main as run_eval

    return run_eval(argv)


def main(argv=None):
    enable_windows()
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "eval", "benchmark"))
    # Only parse the command here, leaving --help and flags to its own parser.
    args = parser.parse_args(argv[:1])
    return {"train": train, "eval": evaluate, "benchmark": benchmark}[args.command](argv[1:])


if __name__ == "__main__":
    main()
