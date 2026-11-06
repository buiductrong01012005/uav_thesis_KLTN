"""Paired Random audit: penalty weights change reward only, not dynamics."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.envs.bs5_energy_collision import BS5EnergyCollisionEnv, CollisionConfig,EnergyConfig


def run(output,slots):
    output.mkdir(parents=True,exist_ok=False)
    envs=[BS5EnergyCollisionEnv(arm='C-R',horizon=slots,energy_config=EnergyConfig(energy_weight=.05),collision_config=CollisionConfig(w)) for w in [0.,.1]]
    xs=[e.reset(seed=620001,assignment_seed=620001) for e in envs]
    rng=np.random.default_rng(620001);rows=[];rates=[];trace=bytes(32);components=[]
    for _ in range(slots):
        acts=[int(rng.choice(np.flatnonzero(m))) for m in xs[0]['action_mask']]
        xs=[e.step(acts) for e in envs]
        a,b=envs
        for key in ['obs','state','action_mask']:np.testing.assert_array_equal(xs[0][key],xs[1][key])
        for key in ['energy_j','uav_positions','ue_positions','status']:np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
        np.testing.assert_array_equal(a.last_v2_network['ue_rate_mbps'],b.last_v2_network['ue_rate_mbps'])
        assert abs(xs[0]['reward']-xs[1]['reward']+.0-xs[1]['info']['blocked_move_cost_norm']*.1)<1e-12
        info=xs[1]['info'];rows.append(dict(slot=b.t,**{k:info[k] for k in ['reward_total','reward_energy','reward_collision','active_blocked_uavs','returning_blocked_uavs','policy_move_requests','consumed_wh','swaps_completed','energy_shortfall_j','throughput_mbps']}))
        rates.append(b.last_v2_network['ue_rate_mbps'])
        components.append(b.last_energy_audit)
        trace=hashlib.sha256(trace+np.asarray(b.last_service_snapshot['ue_positions'],dtype='<f8').tobytes()).digest()
    with (output/'slots.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output/'collision_events.json').write_text(json.dumps(b.collision_events,indent=2)+'\n')
    (output/'energy_events.json').write_text(json.dumps(b.energy_events,indent=2)+'\n')
    with (output/'energy.npz').open('wb') as f:np.savez_compressed(f,**{k:np.asarray([x[k] for x in components]) for k in components[0]},ue_rates=rates)
    result=dict(slots=slots,seed=620001,paired_dynamics_network_energy_exact=True,
        active_blocked_total=sum(x['active_blocked_uavs'] for x in rows),
        returning_blocked_total=sum(x['returning_blocked_uavs'] for x in rows),
        policy_move_requests_total=sum(x['policy_move_requests'] for x in rows),
        mean_throughput_mbps=float(np.sum(rates,axis=1).mean()),
        consumed_wh=sum(x['consumed_wh'] for x in rows),
        swaps_per_uav={str(i):sum(x['event']=='swap_complete' and x['agent']==i for x in b.energy_events) for i in range(5)},
        energy_shortfall_j=sum(x['energy_shortfall_j'] for x in rows),ue_trace_sha256=trace.hex())
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--slots',type=int,default=1200)
    a=p.parse_args()
    if a.slots<=0:raise ValueError('Positive slot count required')
    run(a.output.resolve(),a.slots)
