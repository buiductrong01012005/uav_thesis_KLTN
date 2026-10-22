"""Safe launcher for A5 relay-required-service reward on R4.

Default deliberately runs ONLY seed 91.  A5 uses
    r_train = U + lambda_G * G_req
with lambda_G=0.10 by default.  Validation/test utility remains the original U.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="+", type=int, default=[91])
    ap.add_argument("--relay-required-lambda", type=float, default=0.10)
    ap.add_argument("--block-transitions", type=int, default=30000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--outdir", default="results/relay_stress_r4_qmix/a5_relay_required_lam0p10")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--restart-incomplete",
        action="store_true",
        help="restart an interrupted seed FROM ZERO; this is not resume",
    )
    args = ap.parse_args()
    if args.block_transitions <= 0 or args.eval_every <= 0:
        ap.error("block-transitions/eval-every must be positive")
    if not (0.0 <= args.relay_required_lambda <= 1.0):
        ap.error("relay-required-lambda must lie in [0,1]")
    if len(set(args.seeds)) != len(args.seeds):
        ap.error("duplicate seed")

    out = Path(args.outdir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        config = out / f"seed{seed}_run_config.json"
        summary = out / f"seed{seed}_summary.json"
        if config.exists():
            c = json.loads(config.read_text(encoding="utf-8"))
            expected = (
                "thesis_cartesian_r4_relay_stress",
                float(args.relay_required_lambda),
                int(3 * args.block_transitions),
                int(args.eval_every),
            )
            seen = (
                c.get("environment_profile"),
                float(c.get("relay_required_lambda", -1.0)),
                int(c.get("total_transitions", -1)),
                int(c.get("eval_every", -1)),
            )
            if seen != expected:
                raise RuntimeError(
                    f"Refusing incompatible existing A5 run: {config}; expected {expected}, saw {seen}"
                )
        if summary.exists():
            d = json.loads(summary.read_text(encoding="utf-8"))
            if d.get("total_transitions") == 3 * args.block_transitions and config.exists():
                print("SKIP COMPLETE A5 R4", seed, flush=True)
                continue
            raise RuntimeError(f"Partial/incompatible summary: {summary}")

        partial = sorted(out.glob(f"seed{seed}_*"))
        if partial and not args.restart_incomplete:
            raise RuntimeError(
                f"INCOMPLETE A5 R4 seed {seed} in {out}. Back it up, then add "
                "--restart-incomplete to restart this seed FROM ZERO. Mid-seed resume is not implemented."
            )
        if partial:
            print("RESTART FROM ZERO (NOT RESUME) A5 R4", seed, flush=True)

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "training" / "run_relay_stress_r4_a5_qmix.py"),
            "--seed",
            str(seed),
            "--block-transitions",
            str(args.block_transitions),
            "--eval-every",
            str(args.eval_every),
            "--relay-required-lambda",
            str(args.relay_required_lambda),
            "--outdir",
            str(out),
        ]
        print("RUN A5 R4", seed, " ".join(cmd), flush=True)
        if args.dry_run:
            continue

        log_path = out / f"seed{seed}_stdout.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if line.startswith(("INITIAL", "BLOCK", "VAL", "DONE")):
                    print(f"[A5 R4 seed{seed}] {line}", end="", flush=True)
            rc = proc.wait()
        if rc:
            raise SystemExit(f"A5 R4 seed {seed} failed with code {rc}; see {log_path}")
        if not summary.exists():
            raise RuntimeError(f"Runner exited without summary: {summary}")

    if not args.dry_run:
        print("A5 R4 REQUESTED TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
