"""Deterministic CPU training/resume checks; no long training."""
import argparse
import dataclasses
import random
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
RUNNER=ROOT/'scripts/training/run_bs5_energy_collision_train.py'


def equal(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b),path
    elif isinstance(a,np.ndarray):
        assert np.array_equal(a,b),path
    elif dataclasses.is_dataclass(a):
        equal(vars(a),vars(b),path)
    elif isinstance(a,random.Random):
        equal(a.getstate(),b.getstate(),path)
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+f'[{i}]')
    else:assert a==b,path


def main(out):
    out.mkdir(parents=True,exist_ok=True)
    def launch(folder,arm='C-R',extra=(),expected=0):
        result=subprocess.run([sys.executable,str(RUNNER),'--arm',arm,'--output',str(out/folder),
            '--diagnostic','--energy-weight','.05','--collision-weight','.1',*extra],cwd=ROOT,capture_output=True,text=True)
        (out/(folder+('_resume' if '--resume' in extra else '')+'.log')).write_text(result.stdout+result.stderr)
        assert (result.returncode==0)==(expected==0),result.stdout+result.stderr
        return result
    launch('full')
    launch('split',extra=['--pause-at-step','48'])
    # Emulate an interrupted/uncommitted update of the convenience best alias.
    torch.save({'uncommitted_alias':True},out/'split/best.pt')
    launch('split',extra=['--resume'])
    a=torch.load(out/'full/resume.pt',weights_only=False,map_location='cpu')
    b=torch.load(out/'split/resume.pt',weights_only=False,map_location='cpu')
    for key in ['model','rng','step','updates','task_signature','best_by_metric','energy_events','collision_events']:
        equal(a[key],b[key],key)
    equal(a['replay'].__dict__,b['replay'].__dict__,'replay')
    ignored={'actor_ms','radio_ms','wall_seconds'}
    def filtered(rows):return [{k:v for k,v in row.items() if k not in ignored} for row in rows]
    equal(filtered(a['training']),filtered(b['training']),'training rows')
    # Validation includes measured runtime only in sampled files, not model state.
    assert a['step']==96 and a['updates']>0
    assert a['training'][-1]['learning_rate']==3e-5 or np.isclose(a['training'][-1]['learning_rate'],3e-5)
    for arm in ['C-N','C-H','H-only']:launch(arm,arm)
    refusal=launch('split',extra=['--resume','--lr-schedule','constant'],expected=1)
    assert 'resume refused' in refusal.stderr
    refusal=launch('split',extra=['--resume','--collision-weight','.2'],expected=1)
    assert 'resume refused' in refusal.stderr
    summary=dict(public_best_alias_recovered=True,collision_weight=.1,diagnostic_arms=['C-N','C-R','C-H','H-only'],steps_per_arm=96,
        cpu_only=True,resume_split_step=48,resume_exact_model_optimizer_replay_rng=True,
        resume_exact_scientific_training_rows=True,changed_schedule_resume_refused=True,
        updates=a['updates'],final_lr=a['training'][-1]['learning_rate'])
    (out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output.resolve())
