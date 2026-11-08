"""Actual frozen PyTorch actor and exact mid-scenario restart; diagnostic scenario only."""
from pathlib import Path
from copy import deepcopy
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.evaluation import run_fourarm400k_drl_benchmark as drl
from scripts.evaluation import run_fourarm400k_heuristics as core
from scripts.analysis.report_fourarm400k_heuristics import validate_raw

def run(models,arm,training_seed,output):
 import io,torch
 from src.rl.bs5_fourarm400k_qmix import make_learner
 if output.exists():raise ValueError('Use a fresh diagnostic output')
 output.mkdir(parents=True)
 rec,tp,payload,bundle_hash=drl.read_model(models,arm,training_seed)
 actor=drl.FrozenActor(payload,tp,rec)
 p=drl.make_protocol(arm,training_seed,1,rec,tp,bundle_hash,True);p['runtime']=drl.runtime();seed=p['target_seeds'][0]
 full=core.start_state(p,seed,controller=deepcopy(actor));split=core.start_state(p,seed,controller=deepcopy(actor))
 # Verify the actor equals the original learner's greedy action API.
 saved=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True);learner=make_learner(arm,device='cpu');learner.agent.load_state_dict(saved['agent']);learner.agent.eval();rng=np.random.default_rng(640001)
 for _ in range(p['horizon']):
  obs,mask=full['current']['obs'],full['current']['action_mask']
  assert full['controller'].act(obs,mask)==learner.act(obs,mask,0.,rng)
  core.advance(full,p)
 folder=output/'restart';folder.mkdir()
 for _ in range(37):core.advance(split,p)
 core.save_state(folder,p,split);split=core.load_state(folder,p)
 for _ in range(p['horizon']-37):core.advance(split,p)
 for key in full['arrays']:np.testing.assert_array_equal(full['arrays'][key],split['arrays'][key],err_msg=key)
 for a,b in zip(full['rows'],split['rows']):
  for k in a:
   if k not in {'actor_ms','radio_ms','actor_exceeds_slot'}:assert a[k]==b[k],k
 core.save_seed(folder,p,split,True);row=json.loads((folder/'summary.json').read_text());validate_raw(lambda name:(output/name).read_bytes(),'restart',row,p)
 result=dict(status='passed',arm=arm,training_seed=training_seed,selected_step=rec['selected_step'],checkpoint_sha256=rec['checkpoint_sha256'],greedy_actor_matches_original_learner=True,actual_frozen_model_resume_exact=True,raw_audit=True,diagnostic_seed=seed,slots=p['horizon'],scope='No held-out test scenario executed')
 (output/'PREFLIGHT_PASSED.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,required=True);p.add_argument('--arm',choices=drl.ARMS,required=True);p.add_argument('--training-seed',type=int,choices=drl.SEEDS,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.models,a.arm,a.training_seed,a.output)
