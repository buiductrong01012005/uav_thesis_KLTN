"""Held-out KPI replay for the A5 R4 checkpoint; evaluation uses original utility."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_relay_stress_r4_qmix import TEST_SEEDS, evaluate_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=91)
    ap.add_argument('--checkpoint', type=Path, default=None)
    ap.add_argument('--test-seeds', type=int, nargs='+', default=list(TEST_SEEDS))
    ap.add_argument('--horizon', type=int, default=150)
    ap.add_argument('--torch-threads', type=int, default=1)
    ap.add_argument('--outdir', type=Path, default=Path('results/relay_stress_r4_qmix/a5_eval'))
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.torch_threads))

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = ROOT / 'results/relay_stress_r4_qmix/a5_relay_required_lam0p10' / f'seed{args.seed}_best_final_block.pt'
    elif not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    rows, summary = evaluate_checkpoint(checkpoint, args.test_seeds, args.horizon)
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f'seed{args.seed}_heldout_kpis.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    train_summary_path = checkpoint.parent / f'seed{args.seed}_summary.json'
    cfg_path = checkpoint.parent / f'seed{args.seed}_run_config.json'
    stored_utility = None
    run_config = None
    if cfg_path.exists():
        run_config = json.loads(cfg_path.read_text(encoding='utf-8'))
    if train_summary_path.exists() and list(args.test_seeds) == list(TEST_SEEDS) and args.horizon == 150:
        train_summary = json.loads(train_summary_path.read_text(encoding='utf-8'))
        stored_utility = float(train_summary['heldout_test']['mean_utility'])
        if not np.isclose(summary['mean_utility'], stored_utility, rtol=0, atol=5e-7):
            raise AssertionError(f"Replay utility {summary['mean_utility']} != stored {stored_utility}")

    payload = {
        'method': 'Spatial-QMIX R4 A5 relay-required-service',
        'train_seed': int(args.seed), 'checkpoint': str(checkpoint),
        'backhaul_radius_m': 200.0,
        'training_reward': (run_config or {}).get('reward_formula', 'U_t + 0.10 * G_req_t'),
        'relay_required_lambda': (run_config or {}).get('relay_required_lambda', 0.10),
        'evaluation_reward': 'original environment utility U_t',
        'test_seeds': list(args.test_seeds), 'horizon': int(args.horizon),
        'stored_heldout_utility_check': stored_utility,
        'metrics': summary, 'rows': rows,
    }
    json_path = outdir / f'seed{args.seed}_heldout_kpis.json'
    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('A5_R4_QMIX_EVAL', json.dumps(summary), flush=True)
    print('WROTE', csv_path, json_path, flush=True)


if __name__ == '__main__':
    main()
