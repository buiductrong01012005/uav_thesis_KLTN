"""Small read-only R4 Pareto-role diagnostic (U, G_req) on fixed UAV geometry.

Replays trained checkpoints on three previously studied scenarios. Every N slots,
tries the active UAV's binary role combinations using the SAME radio scorer,
without changing UAV/UE positions, actions, routes, mobility or battery lifecycle.
Diagnostic only: privileged, offline, no future policy execution or training.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_r4_relay_bottlenecks import (
    PATHS, checkpoint_path, radio_with_roles_exact,
)
from scripts.eval_relay_stress_r4_qmix import (
    direct_access_cells, make_env, make_learner, relay_required_ue_mask,
)
from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_fixed_mobile_battery_env import ACTIVE

EPS = 1e-9


def measures(net, required_mask, demand):
    rates = np.asarray(net['ue_rate_mbps'], dtype=float)
    n_required = int(np.count_nonzero(required_mask))
    score = (float(np.sum(np.clip(rates[required_mask] / demand, 0.0, 1.0))) / n_required
             if n_required else 0.0)
    return {
        'utility': float(net['utility']),
        'req_score': score,
        'req_served': int(np.count_nonzero(required_mask & (rates > EPS))),
        'req_total': n_required,
        'full_ues': int(np.count_nonzero(rates >= demand-EPS)),
        'served_ues': int(np.count_nonzero(rates > EPS)),
    }


def is_dominant(a, b, eps=EPS):
    """a weakly Pareto dominates b with a strict gain on >=1 axis."""
    return ((a['utility'] >= b['utility'] - eps and a['req_score'] >= b['req_score'] - eps)
            and (a['utility'] > b['utility'] + eps or a['req_score'] > b['req_score'] + eps))


def pareto_indices(rows, eps=EPS):
    """Keep unique nondominated objective points; preference to fewer Relay UAVs on ties."""
    kept = []
    for i, row in enumerate(rows):
        dominated = any(i != j and is_dominant(other, row, eps) for j, other in enumerate(rows))
        if dominated:
            continue
        equivalent = any((abs(row['utility'] - rows[j]['utility']) <= eps and
                          abs(row['req_score'] - rows[j]['req_score']) <= eps)
                         for j in kept)
        if not equivalent:
            kept.append(i)
    return kept


def enumerate_role_pareto(env, snap, direct_cells, utility_slack=0.02):
    if not (0.0 <= utility_slack <= 1.0):
        raise ValueError('utility_slack must lie within [0,1]')
    current_roles = snap['roles'].copy()
    active = np.flatnonzero(snap['status'] == ACTIVE)
    required = relay_required_ue_mask(env, snap['ue_positions'], direct_cells)
    original_net = radio_with_roles_exact(env, snap, current_roles)
    actual = measures(original_net, required, env.cfg.ue_demand_mbps)
    if abs(actual['utility'] - float(snap['utility'])) > 1e-8:
        raise AssertionError('Same-slot snapshot/role scorer mismatch')
    if not np.allclose(original_net['ue_rate_mbps'], snap['ue_rate_mbps'], atol=1e-7):
        raise AssertionError('Same-slot per-UE rates mismatch')
    candidates = []
    for flags in itertools.product((ACCESS, RELAY), repeat=len(active)):
        roles = current_roles.copy()
        roles[active] = flags
        net = radio_with_roles_exact(env, snap, roles)
        cand = {
            **measures(net, required, env.cfg.ue_demand_mbps),
            'role_vector': ''.join('R' if r == RELAY else 'A' for r in roles),
            'num_relays': int(np.count_nonzero(roles[active] == RELAY)),
            'same_as_policy': bool(np.array_equal(roles, current_roles)),
        }
        candidates.append(cand)
    same = [x for x in candidates if x['same_as_policy']]
    if len(same) != 1 or abs(same[0]['utility']-actual['utility']) > 1e-8:
        raise AssertionError('Executed policy is missing from candidate set')
    # Sort to deduplicate tied objective points deterministically: prefer fewer relay UAVs.
    candidates.sort(key=lambda x: (x['num_relays'], x['role_vector']))
    frontier = [candidates[i] for i in pareto_indices(candidates)]
    best_u = max(candidates, key=lambda c: c['utility'])
    best_req = max(candidates, key=lambda c: (c['req_score'], c['utility']))
    permitted = [c for c in candidates if c['utility'] >= actual['utility'] - utility_slack - EPS]
    balanced = max(permitted, key=lambda c: (c['req_score'], c['utility']))
    weak_dominators = [c for c in candidates if is_dominant(c, actual)]
    both_gain = [c for c in candidates if (c['utility'] > actual['utility'] + EPS and
                                          c['req_score'] > actual['req_score'] + EPS)]
    outcome = {
        'current_role_vector': ''.join('R' if r == RELAY else 'A' for r in current_roles),
        'current_nrelay': int(np.count_nonzero(current_roles[active] == RELAY)),
        'current_utility': actual['utility'],
        'current_req_score': actual['req_score'],
        'current_req_served': actual['req_served'],
        'current_full_ues': actual['full_ues'],
        'required_ues': actual['req_total'],
        'best_role_utility': best_u['utility'],
        'max_req_score': best_req['req_score'],
        'max_req_utility': best_req['utility'],
        'pareto_npoints': len(frontier),
        'pareto_dominated_policy': int(bool(weak_dominators)),
        'strict_both_gain_exists': int(bool(both_gain)),
        'n_role_candidates': len(candidates),
        'best_req_with_utility_slack': balanced['req_score'],
        'balanced_utility': balanced['utility'],
        'balanced_req_served': balanced['req_served'],
        'balanced_num_relays': balanced['num_relays'],
        'gain_req_score_at_slack': balanced['req_score'] - actual['req_score'],
        'role_only_utility_headroom': best_u['utility'] - actual['utility'],
        'role_only_req_headroom': best_req['req_score'] - actual['req_score'],
    }
    for candidate in frontier:
        candidate['delta_utility_from_policy'] = candidate['utility']-actual['utility']
        candidate['delta_req_score_from_policy'] = candidate['req_score']-actual['req_score']
    return outcome, frontier


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=tuple(PATHS),
                        default=['original', 'lam0p30', 'lam1p00'])
    parser.add_argument('--train-seed', type=int, default=91)
    parser.add_argument('--scenarios', nargs='+', type=int, default=[3102, 3106, 3117])
    parser.add_argument('--horizon', type=int, default=150)
    parser.add_argument('--sample-every', type=int, default=25,
                        help='At steps 1, N, 2N, ...; 0 is invalid')
    parser.add_argument('--utility-slack', type=float, default=0.02,
                        help='Maximum absolute utility drop permitted to maximize G_req')
    parser.add_argument('--outdir', type=Path,
                        default=Path('results/relay_stress_r4_qmix/pareto_role_audit'))
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.scenarios or args.sample_every <= 0 or args.horizon <= 0:
        parser.error('Need >=1 scenario and positive sample interval/horizon')
    if not 0 <= args.utility_slack <= 1:
        parser.error('--utility-slack must be between 0 and 1')
    paths = {name: checkpoint_path(name, args.train_seed) for name in args.variants}
    steps = [i for i in range(1,args.horizon+1) if i==1 or i%args.sample_every==0]
    print('Variants:', args.variants, 'scenarios:', args.scenarios, 'steps sampled:', steps,
          'role combinations per sample: up to 32; no training', flush=True)
    for name, path in paths.items():
        print('Checkpoint', name, ':', path, 'exists=',path.exists(),flush=True)
    if args.dry_run:
        return
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing trained R4 checkpoints:\n'+'\n'.join(missing))
    torch.set_num_threads(max(args.torch_threads,1))
    outdir = args.outdir if args.outdir.is_absolute() else ROOT/args.outdir
    per_slot, front_rows = [], []
    for name, path in paths.items():
        env0=make_env(args.scenarios[0], args.horizon)
        learner=make_learner(env0)
        learner.load(str(path))
        learner.agent.eval(); learner.mixer.eval()
        summary_path = path.parent / f'seed{args.train_seed}_summary.json'
        expected = {}
        if args.horizon==150 and summary_path.exists():
            summary=json.loads(summary_path.read_text(encoding='utf-8'))
            expected={int(x['seed']):float(x['mean_utility'])
                      for x in summary.get('heldout_test',{}).get('rows',[])}
        for seed in args.scenarios:
            env=make_env(seed,args.horizon)
            direct_cells=direct_access_cells(env)
            out=env.reset(seed=seed)
            rng=np.random.default_rng(900000+seed)
            replay_util=[]
            for t in range(1, args.horizon+1):
                actions=learner.act(out['obs'],out['action_mask'],0.0,rng)
                env.last_network=None
                out=env.step(actions)
                snap=env.last_network
                if snap is None or abs(float(out['reward'])-float(snap['utility']))>1e-8:
                    raise AssertionError('Captured current network != environment step reward')
                replay_util.append(float(snap['utility']))
                if t==1 or t%args.sample_every==0:
                    values, front=enumerate_role_pareto(env,snap,direct_cells,args.utility_slack)
                    slot={'model':name,'scenario_seed':seed,'step':t,**values}
                    per_slot.append(slot)
                    for cand in front:
                        front_rows.append({'model':name,'scenario_seed':seed,'step':t,**cand})
                if out['terminated'] or out['truncated']:
                    break
            if seed in expected and not np.isclose(np.mean(replay_util),expected[seed],rtol=0,atol=5e-7):
                raise AssertionError(f'{name} {seed} heldout utility mismatch; refusing invalid audit')
            print(f'DONE {name} scenario {seed}: original utility {np.mean(replay_util):.5f}',flush=True)
    write_csv(outdir/'per_slot.csv',per_slot)
    write_csv(outdir/'pareto_frontier.csv',front_rows)
    summary_rows=[]
    for name in args.variants:
        rs=[r for r in per_slot if r['model']==name]
        summary_rows.append({
            'model':name, 'n_samples':len(rs),
            'frac_policy_pareto_dominated':float(np.mean([r['pareto_dominated_policy'] for r in rs])),
            'frac_strict_both_gain_exists':float(np.mean([r['strict_both_gain_exists'] for r in rs])),
            'mean_current_utility':float(np.mean([r['current_utility'] for r in rs])),
            'mean_current_req_score':float(np.mean([r['current_req_score'] for r in rs])),
            'mean_best_utility_headroom':float(np.mean([r['role_only_utility_headroom'] for r in rs])),
            'mean_req_headroom_at_same_geometry':float(np.mean([r['role_only_req_headroom'] for r in rs])),
            'mean_req_gain_at_utility_slack':float(np.mean([r['gain_req_score_at_slack'] for r in rs])),
        })
    write_csv(outdir/'summary.csv',summary_rows)
    (outdir/'meta.json').write_text(json.dumps({
        'objective':['utility U (normalized throughput)','G_req mean satisfied fraction on geometry-required UEs'],
        'seed':args.train_seed,'scenarios':args.scenarios,'horizon':args.horizon,
        'sample_every':args.sample_every,'utility_slack':args.utility_slack,
        'disclaimer':'Privileged exhaustive role-only OFFLINE audit; not a deployable policy, no movement/trajectory changes.',
        'interpretation':'A candidate with U>=actual and G_req>=actual (one strict) Pareto-dominates the current role.\n'
                         'If no same-geometry improvement, it does not prove movement is the only cause.',
    },indent=2,ensure_ascii=False),encoding='utf-8')
    print('WROTE',outdir,flush=True)
    for r in summary_rows:
        print('SUMMARY',json.dumps(r),flush=True)


if __name__=='__main__':
    main()
