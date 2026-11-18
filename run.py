"""Short Windows commands for the existing W00 runners."""
import argparse
import sys

# Set this before importing repository modules, including in the parent process.
sys.dont_write_bytecode = True
from scripts import w00  # noqa: E402 -- prevent bytecode before local imports

ARMS = {"qmix-a": "C-N", "qmix-ar": "C-R", "qmix-arh": "C-H", "qmix-h": "H-only"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train a fresh W00 model")
    train.add_argument("configuration", choices=ARMS)
    train.add_argument("--seed", type=int, default=551)
    train.add_argument("--steps", type=int, default=399600)
    train.add_argument("--output", required=True)
    train.add_argument("--diagnostic", action="store_true", help="Existing 96-step smoke protocol")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--pause-at-step", type=int, default=0)
    # The evaluation parser lives in scripts.w00; keep one CLI definition.
    commands.add_parser("eval", add_help=False, help="Evaluate the exact checkpoint path")
    args, remaining = parser.parse_known_args(argv)
    if args.command == "eval":
        return w00.main(["eval", *remaining])
    if remaining:
        parser.error("unrecognized arguments: " + " ".join(remaining))
    forwarded = ["train", "--arm", ARMS[args.configuration], "--training-seed", str(args.seed),
                 "--total-steps", str(args.steps), "--output", args.output,
                 "--pause-at-step", str(args.pause_at_step)]
    if args.diagnostic:
        forwarded.append("--diagnostic")
    if args.resume:
        forwarded.append("--resume")
    return w00.main(forwarded)


if __name__ == "__main__":
    main()
