"""Run reward ablations A1-A4 using the existing Spatial-QMIX runner.

A0 is the existing original checkpoint set and is NOT retrained.
A1 Energy:          U - 0.05 E
A2 Role:            U + 0.02 S
A3 Energy + Role:   U - 0.05 E + 0.02 S
A4 + Spatial:       U - 0.05 E + 0.02 S - 0.02 D, d_min=100 m

From repo root on Windows PowerShell:
    python scripts/launchers/run_reward_ablation_multiseed.py --dry-run
    python scripts/launchers/run_reward_ablation_multiseed.py --variants spatial --seeds 91
    python scripts/launchers/run_reward_ablation_multiseed.py --variants spatial

Existing completed A1-A3 runs are detected and skipped. A4 is written to
results/reward_ablation/spatial. Interrupted seeds are never silently resumed.
"""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    "energy": {"energy": 0.05, "role": 0.0, "spatial": 0.0, "dmin": 100.0},
    "role": {"energy": 0.0, "role": 0.02, "spatial": 0.0, "dmin": 100.0},
    "combined": {"energy": 0.05, "role": 0.02, "spatial": 0.0, "dmin": 100.0},
    "spatial": {"energy": 0.05, "role": 0.02, "spatial": 0.02, "dmin": 100.0},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[91, 92, 93])
    ap.add_argument("--block-transitions", type=int, default=30000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--outdir", default="results/reward_ablation")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--restart-incomplete",
        action="store_true",
        help="Explicitly restart an interrupted seed from zero; NOT checkpoint resume",
    )
    args = ap.parse_args()
    if args.block_transitions <= 0 or args.eval_every <= 0:
        ap.error("positive blocks/eval required")
    if len(set(args.seeds)) != len(args.seeds):
        ap.error("duplicate seed")

    out = Path(args.outdir)
    if not out.is_absolute():
        out = ROOT / out

    for label in args.variants:
        cfg = VARIANTS[label]
        for seed in args.seeds:
            folder = out / label
            conf = folder / f"seed{seed}_run_config.json"
            summ = folder / f"seed{seed}_summary.json"
            if conf.exists():
                existing = json.loads(conf.read_text(encoding="utf-8"))
                expected = (
                    cfg["energy"],
                    cfg["role"],
                    cfg["spatial"],
                    cfg["dmin"],
                    3 * args.block_transitions,
                    args.eval_every,
                )
                # Backward-compatible with A1-A3 configs created before A4 existed.
                seen = (
                    float(existing.get("energy_lambda", -1)),
                    float(existing.get("role_lambda", -1)),
                    float(existing.get("spatial_lambda", 0.0)),
                    float(existing.get("min_separation_m", 100.0)),
                    int(existing.get("total_transitions", -1)),
                    int(existing.get("eval_every", -1)),
                )
                if seen != expected:
                    raise RuntimeError(
                        f"Refusing incompatible existing run: {conf}; expected {expected}, saw {seen}"
                    )
            if summ.exists():
                if not conf.exists():
                    raise RuntimeError(
                        f"Completed summary exists but run config is missing: {summ}"
                    )
                d = json.loads(summ.read_text(encoding="utf-8"))
                if d.get("total_transitions") == 3 * args.block_transitions:
                    print("SKIP COMPLETE", label, seed, flush=True)
                    continue
                raise RuntimeError(f"Partial/incompatible summary, check manually: {summ}")

            partial_files = sorted(folder.glob(f"seed{seed}_*")) if folder.exists() else []
            if partial_files and not args.restart_incomplete:
                raise RuntimeError(
                    f"INCOMPLETE seed {seed} in {folder}. Do not overwrite a partial run. "
                    "Back up the directory and rerun with --restart-incomplete to restart "
                    "this seed FROM ZERO; mid-seed checkpoint resume is not implemented."
                )
            if partial_files and args.restart_incomplete:
                print("RESTART FROM ZERO (NOT RESUME)", label, seed, flush=True)

            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "training" / "run_thesis_no_curriculum_fulltask.py"),
                "--seed",
                str(seed),
                "--block-transitions",
                str(args.block_transitions),
                "--eval-every",
                str(args.eval_every),
                "--energy-lambda",
                str(cfg["energy"]),
                "--role-lambda",
                str(cfg["role"]),
                "--spatial-lambda",
                str(cfg["spatial"]),
                "--min-separation-m",
                str(cfg["dmin"]),
                "--outdir",
                str(folder),
            ]
            print("RUN", label, seed, " ".join(cmd), flush=True)
            if args.dry_run:
                continue

            folder.mkdir(parents=True, exist_ok=True)
            logfile = folder / f"seed{seed}_stdout.log"
            with logfile.open("w", encoding="utf-8") as log:
                p = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                assert p.stdout is not None
                for line in p.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(("INITIAL", "BLOCK", "VAL", "DONE")):
                        print(f"[{label} seed{seed}] {line}", end="", flush=True)
                rc = p.wait()
            if rc:
                raise SystemExit(f"{label} seed{seed} failed (code {rc}). Check {logfile}")
            if not summ.exists():
                raise RuntimeError(f"Missing final summary after zero exit: {summ}")

    if not args.dry_run:
        print(
            "ALL REQUESTED RUNS FINISHED; run scripts/analysis/summarize_reward_ablation.py",
            flush=True,
        )


if __name__ == "__main__":
    main()
