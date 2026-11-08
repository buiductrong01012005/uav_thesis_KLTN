"""Actual Kaggle/PyTorch verification for the chosen new training seed."""
from pathlib import Path
import argparse,json,os,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def run(arm,seed,output):
 import torch
 from scripts.verification.verify_bs5_fourarm400k import run as original,assert_equal
 from scripts.training.run_bs5_fourarm400k import load_protocol,parser
 from src.rl.bs5_fourarm400k_qmix import make_learner
 if output.exists():raise ValueError('Use a fresh preflight path')
 output.mkdir(parents=True)
 original(arm,output/'original_contract')
 if seed not in (551,552,553):raise ValueError('Unexpected training seed')
 p=load_protocol(parser().parse_args(['--arm',arm,'--output','fixture','--training-seed',str(seed)]))
 assert p['training_seed']==seed and p['total_steps']==399600
 initial=[]
 for a in ['C-N','C-R','C-H','H-only']:
  torch.manual_seed(seed);model=make_learner(a,device='cpu')
  initial.append((model.agent.encoder.state_dict(),model.agent.head[0].state_dict(),model.mixer.state_dict()))
 for body in initial[1:]:assert_equal(initial[0],body,'common_body_same_seed')
 torch.manual_seed(551);a=make_learner(arm,device='cpu').agent.encoder.state_dict()
 torch.manual_seed(552 if seed==551 else seed);b=make_learner(arm,device='cpu').agent.encoder.state_dict()
 assert any(not torch.equal(a[k],b[k]) for k in a),'Additional seed must change initialization'
 runner=ROOT/'scripts/training/run_bs5_fourarm400k.py'
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 def train(name,extra=()):
  subprocess.run([sys.executable,str(runner),'--arm',arm,'--training-seed',str(seed),'--output',str(output/name),'--diagnostic','--device','cpu',*extra],cwd=ROOT,env=env,check=True)
 train('continuous');train('resumed',['--pause-at-step','48']);train('resumed',['--resume'])
 aa=torch.load(output/'continuous/resume.pt',map_location='cpu',weights_only=False);bb=torch.load(output/'resumed/resume.pt',map_location='cpu',weights_only=False)
 for key in ['protocol','step','episode','updates','model','replay','rng','best_by_metric','energy_events','collision_events']:assert_equal(aa[key],bb[key],key)
 result=dict(status='passed',arm=arm,training_seed=seed,actual_pytorch_updates=True,additional_seed_changes_initialization=True,matched_bodies_across_arms=True,optimizer_replay_rng_resume_exact=True,scope='Diagnostic seeds only, not full training or held-out test')
 (output/'PREFLIGHT_PASSED.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=['C-N','C-R','C-H','H-only'],required=True);p.add_argument('--training-seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.arm,a.training_seed,a.output)
