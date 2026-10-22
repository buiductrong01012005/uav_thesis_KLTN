"""Safe pilot launcher for R4 A5 with lambda_G = 1.0.

This wrapper intentionally runs only seed 91 by default and writes to a new
folder, so the existing lambda=0.10 experiment is never overwritten.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAMBDA_G = 1.0
DEFAULT_OUTDIR = "results/relay_stress_r4_qmix/a5_relay_required_lam1p00"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', nargs='+', type=int, default=[91])
    ap.add_argument('--block-transitions', type=int, default=30000)
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--outdir', default=DEFAULT_OUTDIR)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--restart-incomplete', action='store_true')
    args = ap.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / 'scripts' / 'run_relay_stress_r4_a5_multiseed.py'),
        '--seeds', *[str(x) for x in args.seeds],
        '--relay-required-lambda', str(LAMBDA_G),
        '--block-transitions', str(args.block_transitions),
        '--eval-every', str(args.eval_every),
        '--outdir', str(args.outdir),
    ]
    if args.dry_run:
        cmd.append('--dry-run')
    if args.restart_incomplete:
        cmd.append('--restart-incomplete')

    print('A5 LAMBDA=1.0:', ' '.join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == '__main__':
    main()
