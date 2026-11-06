"""Probe legacy-radio Hybrid FDM before changing RL action/observation/energy."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.diagnostics.bs5_hybrid_fdm import HybridFDMConfig, score_hybrid_fdm, VERSION
from src.envs.bs5_tcf_env import BS5TCFEnv, TCFWeights
from src.envs.bs5_service_v2 import ServiceV2Config


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fixture(local_users, distance_cells):
    env = BS5TCFEnv()
    env.surviving_bs_xy = np.asarray([[125., 125.]])
    env.uav_positions[:] = [[2, 2+distance_cells], [2, 8+distance_cells], [19, 0], [18, 0], [17, 0]]
    env.status[:] = 3
    env.status[:2] = 0
    env.roles[:] = [1, 0, 0, 0, 0]
    env.ue_positions[:] = [19., 0.]
    env.ue_positions[:local_users] = env.uav_positions[0]
    env.ue_positions[local_users:local_users+25] = env.uav_positions[1]
    return env


def run(args):
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.config.read_text())
    hybrid = HybridFDMConfig(**protocol['hybrid'])
    weights = TCFWeights(**protocol['reward_weights'])
    json_write(args.output/'protocol.json', protocol)
    with np.load(args.corpus, allow_pickle=False) as data:
        arrays = {k:data[k] for k in data.files}
    np.savez_compressed(args.output/'development_frames.npz', **arrays)
    env = BS5TCFEnv(service_config=ServiceV2Config(**protocol['service']), reward_weights=weights)
    env.reset(seed=620001, assignment_seed=620001)
    env._in_step = False
    trace_rows, selected, assignments = [], [], []
    max_violation, max_airtime = 0., 0.
    scoring_calls = 0

    def evaluate(roles):
        nonlocal scoring_calls, max_violation, max_airtime
        result = score_hybrid_fdm(env, roles, hybrid, weights)
        scoring_calls += 1
        max_violation = max(max_violation, result['metrics']['flow_violation_mbps'])
        max_airtime = max(max_airtime, result['metrics']['schedule_airtime'])
        assert result['metrics']['flow_violation_mbps'] < 1e-6 and max_airtime <= 1+1e-6
        return result['metrics']

    def improves(a, b, metric):
        if b is None or a[metric] > b[metric]+1e-10:
            return True
        if abs(a[metric]-b[metric]) <= 1e-10 and 'roles' in a and 'roles' in b:
            return (a['roles'].count('2'),a['roles']) < (b['roles'].count('2'),b['roles'])
        return False

    for i in range(len(arrays['seed'])):
        for dst, src in [('uav_positions','positions'),('ue_positions','ue_positions'),
                         ('status','status'),('roles','roles'),('battery','battery')]:
            setattr(env, dst, arrays[src][i].copy())
        frame = dict(frame=i, seed=int(arrays['seed'][i]), slot=int(arrays['slot'][i]),
                     origin=str(arrays['origin'][i]) if 'origin' in arrays else 'old_utility_C-R')
        original = env.roles.copy()
        state = {k:getattr(env,k).copy() for k in ['roles','status','battery','uav_positions','ue_positions']}
        base = evaluate(original)
        all_a = evaluate(np.zeros(5, dtype=int))
        all_h = evaluate(np.where(original == 1, 2, original))
        best = base
        best_roles = original.copy()
        for aid in np.flatnonzero((original == 1) & (env.status == 0)):
            roles = original.copy()
            roles[aid] = 2
            m = evaluate(roles)
            if improves(m,best,'tcf_score'):
                best, best_roles = m, roles.copy()
        for label, roles, m in [('recorded_AR',original,base),
                               ('all_A',np.zeros(5,dtype=int),all_a),
                               ('all_recorded_R_to_H',np.where(original == 1,2,original),all_h),
                               ('best_optional_one_R_to_H',best_roles,best)]:
            trace_rows.append(dict(**frame,strategy=label,roles=''.join(map(str,roles)),**m))
        if i % args.search_stride == 0:
            active = np.flatnonzero(env.status == 0)
            winners = {(arm,criterion):None for arm in ['C-N','C-R','C-H']
                       for criterion in ['tcf_score','throughput_mbps']}
            for combo in itertools.product([0,1,2],repeat=len(active)):
                roles = np.zeros(5,dtype=int)
                roles[active] = combo
                m = evaluate(roles)
                row = dict(**frame,roles=''.join(map(str,roles)),**m)
                assignments.append(row)
                legal = ['C-H']
                if 2 not in combo:
                    legal.append('C-R')
                if all(r == 0 for r in combo):
                    legal.append('C-N')
                for arm, criterion in winners:
                    previous = winners[arm,criterion]
                    if arm in legal and improves(row,previous,criterion):
                        winners[arm,criterion] = row
            for arm,criterion in winners:
                selected.append(dict(arm=arm,criterion=criterion,**winners[arm,criterion]))
            print(f'frame {i+1}/{len(arrays["seed"])} role search complete; calls={scoring_calls}',flush=True)
        for k,value in state.items():
            np.testing.assert_array_equal(getattr(env,k),value)

    # Numeric summaries are computed by type rather than relying on key order.
    metrics = [k for k,v in base.items() if isinstance(v,(float,int,np.floating,np.integer))]
    aggregates = []
    for strategy in protocol['audit']['trace_strategies']:
        rows = [r for r in trace_rows if r['strategy'] == strategy]
        aggregates.append(dict(group='trace',arm=strategy,criterion='fixed_or_optional_TCF',frames=len(rows),
            **{k:float(np.mean([r[k] for r in rows])) for k in metrics}))
    for arm, criterion in itertools.product(['C-N','C-R','C-H'],['tcf_score','throughput_mbps']):
        rows = [r for r in selected if r['arm'] == arm and r['criterion'] == criterion]
        aggregates.append(dict(group='role_search',arm=arm,criterion=criterion,frames=len(rows),
            **{k:float(np.mean([r[k] for r in rows])) for k in metrics}))

    micro = []
    for label, users, distance in [('short_gateway_local10',10,2),
                                   ('edge_gateway_local1',1,6),
                                   ('edge_gateway_local25',25,6)]:
        fenv = fixture(users,distance)
        for role in [0,1,2]:
            roles = np.asarray([role,0,0,0,0])
            result = score_hybrid_fdm(fenv,roles,hybrid,weights)
            micro.append(dict(fixture=label,gateway_distance_m=distance*50,local_users=users,
                              remote_users=25,first_role=role,**result['metrics']))

    paired = []
    for frame in sorted({r['frame'] for r in selected}):
        rows = {r['arm']:r for r in selected if r['frame'] == frame and r['criterion'] == 'tcf_score'}
        paired.append(dict(frame=frame,delta_tcf=rows['C-H']['tcf_score']-rows['C-R']['tcf_score'],
            delta_mbps=rows['C-H']['throughput_mbps']-rows['C-R']['throughput_mbps'],
            selected_hybrids=rows['C-H']['active_hybrid']))
    summary = dict(version=VERSION,trace_frames=len(arrays['seed']),
        search_frames=len(paired),scoring_calls=scoring_calls,
        search_hybrid_selected_frames=sum(r['selected_hybrids']>0 for r in paired),
        search_strict_tcf_improvement_frames=sum(r['delta_tcf']>1e-9 for r in paired),
        max_flow_violation_mbps=max_violation,max_airtime=max_airtime,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),args.config,
            ROOT/'src/diagnostics/bs5_hybrid_fdm.py',ROOT/'src/diagnostics/bs5_spectrum_airtime.py',
            ROOT/'src/envs/bs5_service_v2.py',ROOT/'src/envs/bs5_tcf_env.py']},
        corpus_sha256=sha(args.corpus),wall_seconds=time.perf_counter()-start,
        scope=protocol['audit']['limitations'],energy='Static only; no Hybrid charging cycle or training claim',
        old_trace_input=args.corpus_label)
    for name,rows in [('trace_comparisons',trace_rows),('selected_roles',selected),
                      ('all_role_assignments',assignments),('aggregate',aggregates),
                      ('microcases',micro),('paired_role_search',paired)]:
        csv_write(args.output/(name+'.csv'),rows)
    json_write(args.output/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/bs5_hybrid_fdm_audit.json')
    p.add_argument('--search-stride',type=int,default=5)
    p.add_argument('--corpus-label',default='Verified old-policy development frames, not Hybrid policy')
    args = p.parse_args()
    if args.search_stride < 1:
        p.error('Positive search stride required')
    run(args)
