"""Safe A5 R4 pilot launcher with relay-required lambda_G = 0.30.

Uses the existing general R4 A5 trainer; does not alter model, task,
reward definition, checkpoint selection or R4 topology. Default: seed 91 only.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAMBDA_G = 0.30
DEFAULT_OUTDIR = 'results/relay_stress_r4_qmix/a5_relay_required_lam0p30'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[91])
    parser.add_argument('--block-transitions', type=int, default=30000)
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--restart-incomplete', action='store_true',
                        help='Restart interrupted seed from scratch; NOT checkpoint resume')
    args = parser.parse_args()

    if args.block_transitions <= 0 or args.eval_every <= 0:
        parser.error('--block-transitions and --eval-every must be > 0')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('duplicate training seeds')

    cmd = [
        sys.executable,
        str(ROOT / 'scripts' / 'run_relay_stress_r4_a5_multiseed.py'),
        '--seeds', *[str(seed) for seed in args.seeds],
        '--relay-required-lambda', str(LAMBDA_G),
        '--block-transitions', str(args.block_transitions),
        '--eval-every', str(args.eval_every),
        '--outdir', DEFAULT_OUTDIR,
    ]
    if args.dry_run:
        cmd.append('--dry-run')
    if args.restart_incomplete:
        cmd.append('--restart-incomplete')
    print('A5 R4 LAMBDA=0.30:', ' '.join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == '__main__':
    main()
