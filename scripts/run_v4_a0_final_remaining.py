from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_MULTI = ROOT / 'scripts' / 'run_v4_a0_final_multiseed.py'
HELDOUT = ROOT / 'scripts' / 'eval_v4_a0_heldout_multiseed.py'


def main():
    ap = argparse.ArgumentParser(description='One-command final A0: train remaining seeds 82/83 + held-out all 3 seeds')
    ap.add_argument('--total-transitions', type=int, default=100000)
    ap.add_argument('--output-root', default='results/final_a0_multiseed')
    ap.add_argument('--seed81-checkpoint', default='results/v4_a0_kaggle_long/seed81_best.pt')
    ap.add_argument('--num-heldout-layouts', type=int, default=20)
    ap.add_argument('--resume', action='store_true', help='Resume seed82/83 from latest_full if present')
    args = ap.parse_args()

    root = Path(args.output_root)
    if not root.is_absolute(): root = (ROOT / root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable, str(TRAIN_MULTI),
        '--seeds', '82', '83',
        '--total-transitions', str(args.total_transitions),
        '--output-root', str(root),
    ]
    if args.resume:
        train_cmd.append('--resume')
    print('STEP 1/2 TRAIN:', ' '.join(train_cmd), flush=True)
    subprocess.run(train_cmd, check=True)

    seed81 = Path(args.seed81_checkpoint)
    if not seed81.is_absolute(): seed81 = (ROOT / seed81).resolve()
    ckpts = {
        81: seed81,
        82: root / 'seed82' / 'seed82_best.pt',
        83: root / 'seed83' / 'seed83_best.pt',
    }
    for seed, p in ckpts.items():
        if not p.exists():
            raise FileNotFoundError(f'Missing seed{seed} best checkpoint: {p}')

    heldout_out = root / 'heldout_aggregate.json'
    eval_cmd = [
        sys.executable, str(HELDOUT),
        '--num-layouts', str(args.num_heldout_layouts),
        '--output', str(heldout_out),
    ]
    for seed, p in ckpts.items():
        eval_cmd += ['--checkpoint', f'{seed}={p}']
    print('STEP 2/2 HELDOUT:', ' '.join(eval_cmd), flush=True)
    subprocess.run(eval_cmd, check=True)

    print('\nFINAL_A0_DONE')
    print(f'Training aggregate: {root / "aggregate.json"}')
    print(f'Held-out aggregate: {heldout_out}')


if __name__ == '__main__':
    main()
