"""CPU lifecycle audit. Scripted is a mechanics fixture, never a benchmark baseline."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import os
for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.envs.bs5_energy_swap import BS5EnergySwapEnv, EnergyConfig
from src.envs.v0_env import MOVE_DELTAS
from src.baselines.bs5_delayed_multihop import RandomAR, DelayedDynamicRole
from src.baselines.bs5_delayed_h_only import DelayedHybridOnly
from src.diagnostics.bs5_episode_service_metrics import episode_service_metrics


def run(label, horizon, seed, output):
    arm='H-only' if label=='heuristic_h' else 'C-R'
    e=BS5EnergySwapEnv(arm=arm,horizon=horizon,energy_config=EnergyConfig(energy_weight=.05))
    x=e.reset(seed=seed,assignment_seed=seed)
    policy={'random':lambda:RandomAR(seed),'dynamic':DelayedDynamicRole,
            'heuristic_h':DelayedHybridOnly}.get(label,lambda:None)()
    rows=[]; rates=[]; audits=[]; trace=bytes(32); times=[]
    goals=np.array([[12,8],[10,8],[17,8],[14,4],[14,12]])
    for _ in range(horizon):
        tic=time.perf_counter()
        if policy is not None:
            actions=policy.act(x['obs'],x['action_mask'])
        else:
            actions=[]
            for i,m in enumerate(x['action_mask']):
                legal=np.flatnonzero(m)
                actions.append(int(min(legal,key=lambda a:(np.abs(e.uav_positions[i]+MOVE_DELTAS[e.decode_action(a)[0]]-goals[i]).sum(),-int(e.decode_action(a)[0]==4),a))))
        times.append((time.perf_counter()-tic)*1000)
        x=e.step(actions)
        rates.append(e.last_v2_network['ue_rate_mbps'].copy())
        trace=hashlib.sha256(trace+np.asarray(e.last_service_snapshot['ue_positions'],dtype='<f8').tobytes()).digest()
        rows.append(dict(slot=e.t,**e.last_energy_probe,throughput_mbps=x['info']['throughput_mbps'],
            returning_uavs=int((np.array(e.last_energy_audit['effective_status'])==1).sum()),
            associated_ues=sum(map(len,e.last_v2_network['association'].values())),
            access_offer_mbps=float(e.last_v2_network['access_raw_ue_rates'].sum()),
            schedule_airtime=e.last_v2_network['schedule_time_sum']))
        audits.append(e.last_energy_audit)
        if e.t % 200 == 0: print(f'{label}: {e.t}/{horizon} slots', flush=True)
    destination=output/label;destination.mkdir(parents=True,exist_ok=True)
    (destination/'events.json').write_text(json.dumps(e.energy_events,indent=2)+'\n')
    import csv
    with (destination/'slots.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (destination/'energy.npz').open('wb') as f:
        np.savez_compressed(f,**{k:np.asarray([a[k] for a in audits]) for k in audits[0]},ue_rates=rates)
    totals={k:sum(r[k] for r in rows) for k in ['consumed_wh','flight_j','circuit_j','access_pa_j','backhaul_pa_j','energy_shortfall_j']}
    event_counts={str(i):sum(x['event']=='swap_complete' and x['agent']==i for x in e.energy_events) for i in range(5)}
    summary=dict(label=label,arm=arm,seed=seed,slots=horizon,config=asdict(e.energy_config),**totals,
        throughput_mbps=float(np.mean(np.sum(rates,axis=1))),
        energy_mbit_per_joule=float(np.sum(rates)*e.cfg.slot_s/(totals['consumed_wh']*3600)),
        mean_decision_ms=float(np.mean(times)),swap_counts_per_uav=event_counts,
        returning_uav_slot_fraction=sum(r['returning_uavs'] for r in rows)/(5*horizon),
        swapping_uav_slot_fraction=sum(r['swapping_uavs'] for r in rows)/(5*horizon),
        max_balance_residual_j=float(np.max(np.abs([a['balance_residual_j'] for a in audits]))),
        ue_trace_sha256=trace.hex(),**episode_service_metrics(rates))
    (destination/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['label','slots','throughput_mbps','consumed_wh','swap_counts_per_uav','energy_shortfall_j']}),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',nargs='+',choices=['scripted','random','dynamic','heuristic_h'],default=['scripted','random'])
    p.add_argument('--slots',type=int,default=1200)
    p.add_argument('--seed',type=int,default=620001)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.slots<=0:raise ValueError('Positive slots required')
    result=[run(label,args.slots,args.seed,args.output) for label in args.labels]
    assert len({r['ue_trace_sha256'] for r in result})==1,'UE trajectories changed between policies'
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
