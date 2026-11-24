"""Train a fresh QMIX run from JSON config; explicit CLI values override config."""

from pathlib import Path
import argparse
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
# Direct execution adds scripts/ to sys.path. Keep the package parent only:
# scripts/packaging is our exporter, while matplotlib needs PyPI packaging.
sys.path[:] = [str(ROOT)] + [
    entry for entry in sys.path
    if Path(entry or ".").resolve() not in {ROOT, ROOT / "scripts"}
]
sys.dont_write_bytecode = True

from scripts.w00 import enable_windows, local_output
from src.configuration import resolve_training
from src.envs.config import ARM_ACTIONS


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/training.json")
    p.add_argument("--arm", choices=ARM_ACTIONS)
    p.add_argument("--seed", "--training-seed", dest="training_seed", type=int)
    p.add_argument("--steps", "--total-steps", dest="total_steps", type=int)
    p.add_argument("--output", type=Path)
    p.add_argument("--lr", type=float)
    p.add_argument("--gamma", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--device", choices=["cpu"])
    p.add_argument("--lr-schedule", choices=["linear", "constant"])
    p.add_argument("--diagnostic", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--set", dest="assignments", action="append", default=[], metavar="SECTION.KEY=JSON"
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--pause-at-step", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=0.0)
    p.add_argument("--stop-after-episodes", type=int, default=0)
    return p


def prepare(args):
    overrides = {
        key: getattr(args, key)
        for key in [
            "arm",
            "training_seed",
            "total_steps",
            "lr",
            "gamma",
            "batch_size",
            "device",
            "lr_schedule",
            "diagnostic",
        ]
    }
    if args.resume:
        if args.output is None:
            raise ValueError("--resume requires --output for the existing run")
        from src.configuration import read_json, set_value

        config = read_json(args.output / "resolved_config.json")
        expected = config["training"]
        for key, value in overrides.items():
            if value is not None and expected[key] != value:
                raise ValueError("Resume configuration changed: " + key)
        for assignment in args.assignments:
            import copy

            candidate = copy.deepcopy(config)
            set_value(candidate, assignment)
            if candidate != config:
                raise ValueError("Resume configuration changed: " + assignment)
    else:
        config = resolve_training(args.config, overrides, args.assignments)
    args.output = local_output(args.output or config["training"]["output"])
    if not args.resume:
        config["training"]["output"] = args.output.as_posix()
    args.resolved_config = config
    if not 0 <= args.pause_at_step <= config["training"]["total_steps"]:
        raise ValueError("Pause point must be inside the training budget")
    return args


def main(argv=None):
    enable_windows()
    args = prepare(parser().parse_args(argv))
    from scripts.training import run_bs5_fourarm400k as runner

    signal.signal(signal.SIGINT, runner.stop_handler)
    signal.signal(signal.SIGTERM, runner.stop_handler)
    return runner.run(args)


if __name__ == "__main__":
    main()
