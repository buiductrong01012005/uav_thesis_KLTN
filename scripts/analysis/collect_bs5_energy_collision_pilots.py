"""Collect REVIEW ZIPs at a common validation step; never rank different rewards."""
import argparse
from copy import deepcopy
import csv
import io
import json
from pathlib import Path,PurePosixPath
import zipfile
import numpy as np

FIELDS=['throughput_mbps','service_coverage_fraction','qos_fraction',
    'jain_time_average','unique_served_ues','energy_consumed_wh','joules_per_mbit',
    'active_blocked_total','returning_blocked_total','policy_move_requests_total',
    'blocked_per_policy_move','energy_shortfall_total_j']


def collect(folder,output,step):
    rows=[];seen={};incomplete=[];signatures=set()
    for path in sorted(folder.rglob('*REVIEW.zip')):
        with zipfile.ZipFile(path) as z:
            ps=[n for n in z.namelist() if n.endswith('/protocol.json')]
            for member in ps:
                p=json.loads(z.read(member))
                if p.get('training_env_version')!='bs5_energy_collision_v2' or p['arm']!='C-R':continue
                base=PurePosixPath(member).parent
                data=list(csv.DictReader(io.StringIO(z.read(str(base/'validation.csv')).decode())))
                selected=[r for r in data if int(r['step'])==step]
                label=base.name
                if len(selected)!=len(p['validation_seeds']) or {int(r['seed']) for r in selected}!=set(p['validation_seeds']):
                    incomplete.append(dict(run=label,reason=f'Missing complete validation at step {step}'));continue
                common=deepcopy(p)
                common['energy_config'].pop('energy_weight')
                common['collision_config'].pop('collision_weight')
                for key in ['runtime','device','torch_version']:common.pop(key,None)
                signatures.add(json.dumps(common,sort_keys=True))
                row=dict(run=label,step=step,training_seed=p['training_seed'],
                    energy_weight=p['energy_config']['energy_weight'],
                    collision_weight=p['collision_config']['collision_weight'],
                    gpu_name=p['runtime']['gpu_name'],torch_version=p['torch_version'])
                for field in FIELDS:
                    values=[float(r[field]) for r in selected if r.get(field) not in [None,'']]
                    row[field]=float(np.mean(values)) if values else None
                key=(row['training_seed'],row['energy_weight'],row['collision_weight'])
                if key in seen:
                    if seen[key]!=row:raise ValueError('Conflicting duplicate REVIEW: '+label)
                    continue
                seen[key]=row;rows.append(row)
    if len(signatures)>1:raise ValueError('Source/seeds/budget/LR/service differ; compare separate experiment groups')
    output.mkdir(parents=True,exist_ok=True)
    if rows:
        with (output/'pilot_summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(validation_step=step,complete_runs=len(rows),incomplete=incomplete,
        rows=rows,scope='one-training-seed screen; compare service, energy and blocked movement, not reward_total; GPU/runtime differences are displayed')
    (output/'pilot_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete_runs=len(rows),incomplete=incomplete),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--step',type=int,default=24000)
    a=p.parse_args();collect(a.input,a.output,a.step)
