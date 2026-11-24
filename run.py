"""Short aliases for the shared train and single-checkpoint evaluation entry points."""

import sys

sys.dont_write_bytecode = True
ARMS = {"qmix-a": "C-N", "qmix-ar": "C-R", "qmix-arh": "C-H", "qmix-h": "H-only"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ["-h", "--help"]:
        print("python run.py train [qmix-a|qmix-ar|qmix-arh|qmix-h] [--config ...] [train options]")
        print("python run.py eval --checkpoint best.pt [--config ...] [eval options]")
        return
    command, args = argv[0], argv[1:]
    if command == "train":
        if args and args[0] in ARMS:
            arm = ARMS[args.pop(0)]
            args = ["--arm", arm, *args]
        from scripts.train import main as run_train

        return run_train(args)
    if command == "eval":
        from scripts.evaluate import main as run_eval

        return run_eval(args)
    raise ValueError("Use train or eval")


if __name__ == "__main__":
    main()
