"""Held-out KPI replay for R4 A5 lambda_G=1.0 checkpoint."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAMBDA_G = 1.0
TRAIN_DIR = ROOT / 'results' / 'relay_stress_r4_qmix' / 'a5_relay_required_lam1p00'
DEFAULT_EVAL_DIR = ROOT / 'results' / 'relay_stress_r4_qmix' / 'a5_lam1_eval'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=91)
    ap.add_argument('--test-seeds', type=int, nargs='+', default=None)
    ap.add_argument('--horizon', type=int, default=150)
    ap.add_argument('--torch-threads', type=int, default=1)
    ap.add_argument('--outdir', type=Path, default=DEFAULT_EVAL_DIR)
    args = ap.parse_args()

    checkpoint = TRAIN_DIR / f'seed{args.seed}_best_final_block.pt'
    if not checkpoint.exists():
        raise FileNotFoundError(
            f'Missing lambda=1.0 checkpoint: {checkpoint}. Run '
            'scripts/run_relay_stress_r4_a5_lam1_pilot.py first.'
        )

    cmd = [
        sys.executable,
        str(ROOT / 'scripts' / 'eval_relay_stress_r4_a5_qmix.py'),
        '--seed', str(args.seed),
        '--checkpoint', str(checkpoint),
        '--horizon', str(args.horizon),
        '--torch-threads', str(args.torch_threads),
        '--outdir', str(args.outdir),
    ]
    if args.test_seeds is not None:
        cmd += ['--test-seeds', *[str(x) for x in args.test_seeds]]

    print('EVAL A5 LAMBDA=1.0:', ' '.join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == '__main__':
    main()
