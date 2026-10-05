"""Summarize heldout KPIs for A0-A4 without comparing shaped rewards directly.

A0 is read from the existing audited reference. A1-A4 must have completed
90k summaries. For served-UE / Relay-UE / spatial coverage KPIs, additionally
run scripts/eval_reward_ablation_relay_kpi.py after training A4.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('mean_utility', 'mean_battery_drain_per_step', 'mean_role_switches_per_step', 'mean_relays')
LABELS = {
    'a0': 'Original QMIX',
    'energy': 'Energy-aware',
    'role': 'Role-bonus',
    'combined': 'Energy + Role',
    'spatial': 'Energy + Role + Spatial',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir', default='results/reward_ablation')
    ap.add_argument('--seeds', type=int, nargs='+', default=[91, 92, 93])
    args = ap.parse_args()
    folder = Path(args.outdir)
    if not folder.is_absolute():
        folder = ROOT / folder

    ref = folder / 'a0_reference_exact_kpis.json'
    if not ref.exists():
        raise FileNotFoundError(f'Run audit_reward_ablation_baseline.py first: {ref}')
    a0 = json.loads(ref.read_text(encoding='utf-8'))
    if sorted(a0.get('training_seeds', [])) != sorted(args.seeds):
        raise RuntimeError('A0 reference uses a different training seed set')

    rows = []
    for row in a0['rows']:
        rows.append({'variant': 'a0', 'seed': row['seed'], **{k: row[k] for k in KEYS}})

    for v in ('energy', 'role', 'combined', 'spatial'):
        for seed in args.seeds:
            p = folder / v / f'seed{seed}_summary.json'
            if not p.exists():
                raise FileNotFoundError(f'Missing full train output: {p}')
            summ = json.loads(p.read_text(encoding='utf-8'))
            if summ['total_transitions'] != 90000:
                raise RuntimeError(f'Not a 90k run: {p}')
            test = summ['heldout_test']
            rows.append({'variant': v, 'seed': seed, **{k: test[k] for k in KEYS}})

    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'reward_ablation_heldout_per_seed.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['variant', 'seed'] + list(KEYS))
        w.writeheader()
        w.writerows(rows)

    totals = []
    for v in LABELS:
        rr = [x for x in rows if x['variant'] == v]
        d = {'variant': v, 'label': LABELS[v]}
        for k in KEYS:
            d[f'{k}_mean'] = float(np.mean([x[k] for x in rr]))
            d[f'{k}_std'] = float(np.std([x[k] for x in rr]))
        totals.append(d)

    with (folder / 'reward_ablation_heldout_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(totals[0]))
        w.writeheader()
        w.writerows(totals)
    print(json.dumps(totals, indent=2))
    print('WROTE', folder / 'reward_ablation_heldout_summary.csv')


if __name__ == '__main__':
    main()
