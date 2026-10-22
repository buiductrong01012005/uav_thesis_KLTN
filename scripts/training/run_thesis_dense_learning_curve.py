from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_command(
    root: Path,
    seed: int,
    total_transitions: int,
    eval_every: int,
    torch_threads: int,
    outdir: Path,
) -> list[str]:
    train_script = root / "scripts" / "training" / "run_v9_r6_battery_transition_curriculum.py"
    init_ckpt = root / "results" / "v8_r6_macro_relay_battery_curriculum" / f"seed{seed}_best.pt"

    if not train_script.exists():
        raise FileNotFoundError(
            f"Missing training script: {train_script}\n"
            "Run this command from the UAV thesis project root."
        )
    if not init_ckpt.exists():
        raise FileNotFoundError(
            f"Missing same-seed battery-aware checkpoint: {init_ckpt}\n"
            "Do not fall back to another training seed."
        )

    return [
        sys.executable,
        str(train_script),
        "--seed",
        str(seed),
        "--total-transitions",
        str(total_transitions),
        "--eval-every",
        str(eval_every),
        "--init-checkpoint",
        str(init_ckpt),
        "--outdir",
        str(outdir),
        "--torch-threads",
        str(torch_threads),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run the final transition-focused Spatial-QMIX stage with dense "
            "validation logging for thesis learning-curve figures."
        )
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[91, 92, 93])
    parser.add_argument("--total-transitions", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--outdir",
        default="results/thesis_dense_learning_curve",
        help="New directory only; the locked final result directory is not touched.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run a seed even if its dense history and summary already exist.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (root / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("Project root :", root)
    print("Output dir   :", outdir)
    print("Seeds        :", args.seeds)
    print("Eval every   :", args.eval_every, "transitions")
    print("IMPORTANT    : existing final held-out results are NOT overwritten.")
    print()

    for seed in args.seeds:
        hist = outdir / f"seed{seed}_history.json"
        summ = outdir / f"seed{seed}_summary.json"

        if hist.exists() and summ.exists() and not args.force:
            print(f"[seed {seed}] already complete -> skip")
            continue

        cmd = build_command(
            root=root,
            seed=seed,
            total_transitions=args.total_transitions,
            eval_every=args.eval_every,
            torch_threads=args.torch_threads,
            outdir=outdir,
        )

        print(f"[seed {seed}]")
        print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
        if args.dry_run:
            print()
            continue

        subprocess.run(cmd, cwd=root, check=True)
        if not hist.exists() or not summ.exists():
            raise RuntimeError(
                f"Seed {seed} finished but expected history/summary was not created."
            )
        print(f"[seed {seed}] complete\n")

    if args.dry_run:
        print("Dry run only. No training was started.")
    else:
        print("Dense learning-curve runs complete.")
        print("Send back these files from:")
        print(outdir)
        for seed in args.seeds:
            print(f"  seed{seed}_history.json")
            print(f"  seed{seed}_summary.json")


if __name__ == "__main__":
    main()
