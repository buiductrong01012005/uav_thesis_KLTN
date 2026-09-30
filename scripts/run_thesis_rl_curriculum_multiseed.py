from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[91, 92, 93])
    ap.add_argument("--methods", nargs="+", choices=["cartesian", "target-aware"],
                    default=["cartesian", "target-aware"])
    ap.add_argument("--stage-transitions", type=int, default=30000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--outdir", default="results/thesis_full_curriculum")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    script = ROOT / "scripts" / "run_thesis_rl_curriculum.py"
    for method in args.methods:
        method_dir = Path(args.outdir) / method.replace("-", "_")
        if not method_dir.is_absolute():
            method_dir = ROOT / method_dir
        for seed in args.seeds:
            summary = method_dir / f"seed{seed}_summary.json"
            if summary.exists() and not args.force:
                print(f"SKIP {method} seed{seed}: {summary} exists", flush=True)
                continue
            cmd = [
                sys.executable, str(script),
                "--method", method,
                "--seed", str(seed),
                "--stage-transitions", str(args.stage_transitions),
                "--eval-every", str(args.eval_every),
                "--torch-threads", str(args.torch_threads),
                "--outdir", args.outdir,
            ]
            print("RUN", " ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
