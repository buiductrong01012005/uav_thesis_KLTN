"""Evaluate the independently trained A5 R4 lambda_G=0.30 checkpoint."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAMBDA_G = 0.30
TRAIN_DIR = ROOT / 'results' / 'relay_stress_r4_qmix' / 'a5_relay_required_lam0p30'
DEFAULT_EVAL_DIR = ROOT / 'results' / 'relay_stress_r4_qmix' / 'a5_lam0p30_eval'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=91)
    parser.add_argument('--test-seeds', type=int, nargs='+', default=None)
    parser.add_argument('--horizon', type=int, default=150)
    parser.add_argument('--torch-threads', type=int, default=1)
    args = parser.parse_args()

    checkpoint = TRAIN_DIR / f'seed{args.seed}_best_final_block.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f'Missing lambda=0.30 checkpoint: {checkpoint}. '
            'Train with scripts/run_relay_stress_r4_a5_lam03_pilot.py first.'
        )

    cmd = [
        sys.executable,
        str(ROOT / 'scripts' / 'eval_relay_stress_r4_a5_qmix.py'),
        '--seed', str(args.seed),
        '--checkpoint', str(checkpoint),
        '--horizon', str(args.horizon),
        '--torch-threads', str(args.torch_threads),
        '--outdir', str(DEFAULT_EVAL_DIR),
    ]
    if args.test_seeds is not None:
        cmd += ['--test-seeds', *[str(seed) for seed in args.test_seeds]]

    print('EVAL A5 R4 LAMBDA=0.30:', ' '.join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == '__main__':
    main()
