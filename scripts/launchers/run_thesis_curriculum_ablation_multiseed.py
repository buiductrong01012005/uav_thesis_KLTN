from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[91, 92, 93])
    ap.add_argument("--block-transitions", type=int, default=30000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument(
        "--outdir",
        default="results/thesis_curriculum_ablation/direct_full_task",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    script = ROOT / "scripts" / "training" / "run_thesis_curriculum_ablation.py"

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir

    for seed in args.seeds:
        summary = outdir / f"seed{seed}_summary.json"
        if summary.exists() and not args.force:
            print(f"SKIP seed{seed}: {summary} exists", flush=True)
            continue

        cmd = [
            sys.executable,
            str(script),
            "--seed",
            str(seed),
            "--block-transitions",
            str(args.block_transitions),
            "--eval-every",
            str(args.eval_every),
            "--torch-threads",
            str(args.torch_threads),
            "--outdir",
            args.outdir,
        ]
        print("RUN", " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
