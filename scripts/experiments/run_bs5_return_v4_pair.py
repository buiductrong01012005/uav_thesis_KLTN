"""Frozen CR60k/CN36k controller sensitivity on the original20 tapes; no training."""
from pathlib import Path
import argparse,copy,gzip,hashlib,io,json,shutil,sys,time,zipfile
from dataclasses import asdict
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.experiments.bs5_return_v4_common import ASSETS,verify_assets,new_env,file_hash
from scripts.analysis.bs5_bundle import atomic_json,read_json,canonical_hash
from scripts.training.run_bs5_launch_ab import source_identity,freeze_source,experiment_lock,weights_hash
from src.rl.episode_failure_spatial_qmix import EpisodeFailureSpatialQMIX
from src.rl.qmix import QMixer
from src.envs.bs5_return_commit_env import RETURN_VERSION,RETURN_CONTRACT
from scripts.analysis.bs5_return_v4_results import summarize_game


def setup():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 assert torch.__version__=='2.11.0+cpu' and np.__version__=='2.1.3','Use pinned CPU runtime'


def runtime():return dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,device='cpu',threads=1,deterministic=True)


def actor(env,arm):
 cp=verify_assets()['checkpoints'][arm];d=torch.load(ASSETS/cp['path'],map_location='cpu',weights_only=True)
 assert d['step']==cp['step']
 model=SimpleNamespace(agent=EpisodeFailureSpatialQMIX._agent(env),mixer=QMixer(5,111))
 for k in ['agent','mixer']:
  obj=getattr(model,k);obj.load_state_dict(d[k],strict=True);obj.eval()
  for p in obj.parameters():p.requires_grad_(False)
 selection=read_json(ASSETS/'baseline_v3'/('cr_selection.json' if arm=='C-R' else 'cn_selection.json'))
 expected=next(x for x in selection['curve'] if x['step']==cp['step'])
 assert weights_hash(model)==expected['weights_sha256']
 return model


def choose(model,out):
 with torch.inference_mode():q=model.agent(torch.as_tensor(out['obs'])).numpy()
 return [int((v:=np.flatnonzero(mask))[np.argmax(q[i,v])]) for i,mask in enumerate(out['action_mask'])]


def prepare(root,case_only=False):
 setup();root=Path(root).resolve();assert not root.exists(),'Use run on existing output; never overwrite'
 lock=verify_assets();identity=source_identity();root.mkdir(parents=True)
 jobs=[dict(arm='C-N',scenario=190006,group='heldout')] if case_only else lock['jobs']
 p=dict(version='bs5_return_v4_frozen_pair_v1',return_version=RETURN_VERSION,contract=RETURN_CONTRACT,training_steps=0,horizon=10000,
  planned_steps=len(jobs)*10000,jobs=jobs,diagnostic=case_only,runtime=runtime(),source_sha256=identity['sha256'],asset_manifest_sha256=file_hash(ASSETS/'SHA256SUMS.json'),
  checkpoints=lock['checkpoints'],scope=lock['scope'],selection='Frozen CR60k/CN36k selected before v4; no checkpoint reselection')
 atomic_json(root/'protocol.json',p);freeze_source(root,identity)
 (root/'checkpoints').mkdir()
 for arm,c in lock['checkpoints'].items():shutil.copy2(ASSETS/c['path'],root/'checkpoints'/f'{arm}.pt')
 shutil.copy2(ASSETS/'LOCK.json',root/'input_lock.json')
 for name in ['games','aggregate','cycles','windows1000','paired']:shutil.copy2(ASSETS/'baseline_v3'/f'{name}.json',root/f'baseline_v3_{name}.json')
 atomic_json(root/'commit.json',dict(completed=[]));atomic_json(root/'budget_ledger.json',dict(training_steps=0,phases={arm:dict(cap=sum(j['arm']==arm for j in jobs)*10000,charged=0,observed=0) for arm in ['C-R','C-N']}))
 atomic_json(root/'status.json',dict(status='prepared',completed=0,total=len(jobs)))
 return p


class Budget:
 def __init__(self,root):self.path=Path(root)/'budget_ledger.json';self.data=read_json(self.path)
 def flush(self):atomic_json(self.path,self.data)
 def reserve(self,arm,n):
  row=self.data['phases'][arm];assert row['charged']+n<=row['cap'];row['charged']+=n;self.flush()
 def observed(self,arm):self.data['phases'][arm]['observed']+=1


def allocate(h):
 a={k:np.empty((h,5,2),np.int16) for k in ['positions_before','positions_service','positions_post']}
 for phase in ['before','service','post']:
  for field,dtype in [('status',np.int8),('roles',np.int8),('battery',np.float32)]:a[field+'_'+phase]=np.empty((h,5),dtype)
 a.update(actions=np.empty((h,5),np.int8),rates=np.empty((h,90),np.float64),association=np.full((h,90),-1,np.int8),routes=np.full((h,5),-1,np.int8),
  mask=np.empty((h+1,5,10),bool),obs=np.empty((h+1,5,116),np.float32),state=np.empty((h+1,111),np.float32),
  stranded=np.zeros((h,5),bool),wait=np.zeros((h,5),bool),reversals=np.zeros((h,5),bool))
 return a


def save_npz(path,a):
 b=io.BytesIO();np.savez_compressed(b,**a);tmp=path.with_suffix('.npz.tmp');tmp.write_bytes(b.getvalue());tmp.replace(path)


def game(root,p,job,model,budget):
 arm,scenario=job['arm'],job['scenario'];env=new_env(arm,p['horizon']);out=env.reset(seed=scenario,assignment_seed=scenario,diagnostic=True);h=p['horizon'];a=allocate(h)
 directory=root/'games'/arm/str(scenario);assert not directory.exists();directory.mkdir(parents=True)
 initial=dict(battery=env.battery.tolist(),residual_uav=int(env.residual_uav_idx),positions=env.uav_positions.tolist(),charging_station_cells=env.charging_stations.tolist())
 controls=[];chunks=[];before_weights=weights_hash(model);last_old=None;last_effective=None
 for t in range(h):
  if t%250==0:budget.reserve(arm,min(250,h-t))
  for key in ['obs','state']:a[key][t]=out[key]
  a['mask'][t]=out['action_mask'];actions=choose(model,out);a['actions'][t]=actions
  for field,attr in [('positions','uav_positions'),('status','status'),('roles','roles'),('battery','battery')]:a[field+'_before'][t]=getattr(env,attr)
  old=env.uav_positions.copy();out=env.step(actions);budget.observed(arm)
  assert not out['terminated'] and out['truncated']==(t+1==h)
  service=env.last_service_snapshot;net=service['network'];assert np.all(np.asarray(service['ue_demand_mbps'])==4.)
  for field,attr in [('positions','uav_positions'),('status','status'),('roles','roles'),('battery','battery')]:
   a[field+'_service'][t]=service[attr];a[field+'_post'][t]=getattr(env,attr)
  a['rates'][t]=net['ue_rate_mbps'];assert abs(float(a['rates'][t].sum()/360)-out['reward'])<1e-10
  for aid,users in net['association'].items():a['association'][t,users]=int(aid)
  for aid,rid in net['routes'].items():
   if rid is not None:a['routes'][t,int(aid)]=int(rid)
  d=env.last_return_control;a['stranded'][t,d['stranded_ids_post']]=True;a['wait'][t,d['wait_ids']]=True
  effective=np.asarray(d['effective_status'])
  if t:
   a['reversals'][t]=(effective==1)&(last_effective==1)&(env.uav_positions==last_old).all(1)&(old!=env.uav_positions).any(1)
  last_old=old;last_effective=effective
  controls.append(dict(slot=t+1,return_control=copy.deepcopy(d),collision=copy.deepcopy(env.last_collision_resolution)))
  if len(controls)==250 or t+1==h:
   f=directory/f'control_{len(chunks):04d}.json.gz';tmp=f.with_suffix('.gz.tmp')
   with gzip.open(tmp,'wt') as stream:json.dump(controls,stream,separators=(',',':'),allow_nan=False)
   tmp.replace(f);chunks.append(dict(path=f.name,sha256=file_hash(f)));controls=[];budget.flush()
  if (t+1)%1000==0:print(json.dumps(dict(phase='eval',**job,slot=t+1,horizon=h)),flush=True)
 for key in ['obs','state']:a[key][h]=out[key]
 a['mask'][h]=out['action_mask'];assert weights_hash(model)==before_weights
 save_npz(directory/'slots.npz',a)
 idx=dict(**job,horizon=h,initial=initial,condition=env.condition,resolved_config=asdict(env.cfg),data='slots.npz',data_sha256=file_hash(directory/'slots.npz'),control_chunks=chunks,
  trace_sha256=env.trace_entry['position_sha256'],checkpoint_sha256=p['checkpoints'][arm]['sha256'],weights_sha256_before=before_weights,weights_sha256_after=weights_hash(model),return_version=RETURN_VERSION,
  encoding='float64 delivered rates; float32 exact obs/state/battery; integer geometry/actions; UE tape referenced by immutable hash in input Dataset')
 metrics,windows,cycles=summarize_game(a,idx);idx['metrics']=metrics
 atomic_json(directory/'index.json',idx)
 return dict(**job,index=(directory/'index.json').relative_to(root).as_posix())


def run(root,max_seconds=None,stop_after_games=None):
 setup();root=Path(root).resolve();p=read_json(root/'protocol.json');verify_assets()
 assert source_identity()['sha256']==p['source_sha256'] and runtime()==p['runtime']
 assert file_hash(ASSETS/'SHA256SUMS.json')==p['asset_manifest_sha256']
 with experiment_lock(root):
  if (root/'SHA256SUMS.json').exists():
   for rel,h in read_json(root/'SHA256SUMS.json').items():assert file_hash(root/rel)==h
   (root/'SHA256SUMS.json').unlink()
  commit=read_json(root/'commit.json');budget=Budget(root)
  assert [{k:x[k] for k in ['arm','scenario','group']} for x in commit['completed']]==p['jobs'][:len(commit['completed'])]
  try:
   for arm,row in budget.data['phases'].items():assert row['charged']==row['observed']==sum(x['arm']==arm for x in commit['completed'])*p['horizon'],'Uncommitted eval compute; preserve evidence, never reset ledger to repeat'
   start=time.monotonic();initial=len(commit['completed']);models={}
   for job in p['jobs'][len(commit['completed']):]:
    if (max_seconds is not None and time.monotonic()-start>=max_seconds) or (stop_after_games is not None and len(commit['completed'])-initial>=stop_after_games):break
    arm=job['arm'];assert file_hash(root/'checkpoints'/f'{arm}.pt')==p['checkpoints'][arm]['sha256']
    if arm not in models:models[arm]=actor(new_env(arm),arm)
    atomic_json(root/'status.json',dict(status='running',completed=len(commit['completed']),total=len(p['jobs']),current=job))
    item=game(root,p,job,models[arm],budget);commit['completed'].append(item);atomic_json(root/'commit.json',commit)
   done=len(commit['completed'])==len(p['jobs']);atomic_json(root/'status.json',dict(status='complete' if done else 'paused_at_game_boundary',completed=len(commit['completed']),total=len(p['jobs'])))
  except BaseException as exc:
   budget.flush();atomic_json(root/'status.json',dict(status='partial_error',completed=len(commit['completed']),error=repr(exc),instruction='Preserve all output and ledger. No silent replay of in-flight game.'));raise


def export(root):
 from scripts.analysis.bs5_return_v4_results import analyze_run
 root=Path(root).resolve();parent=root.parent;out=parent/(root.name+'_export');n=1
 while out.exists():n+=1;out=parent/(root.name+f'_export_{n}')
 out.mkdir();analysis=out/'analysis';analyze_run(root,analysis)
 report=root/'analysis';shutil.copytree(analysis,report,dirs_exist_ok=True)
 # Offline analyzer uses NumPy + stdlib only; no Torch/env import required.
 for name in ['bs5_return_v4_results.py','bs5_return_commit_audit.py','bs5_return_audit.py','bs5_bundle.py','bs5_continuous_common.py','analyze_bs5_cn_h1200_return.py','analyze_bs5_curriculum_bundle.py','analyze_bs5_curriculum_extend.py','bs5_curve_plot.py']:
  shutil.copy2(ROOT/'scripts/analysis'/name,root/name)
 excluded={'SHA256SUMS.json','run.lock','.experiment.lock'}
 files=[f for f in sorted(root.rglob('*')) if f.is_file() and f.name not in excluded and not f.name.endswith('.tmp')]
 atomic_json(root/'SHA256SUMS.json',{f.relative_to(root).as_posix():file_hash(f) for f in files});files.append(root/'SHA256SUMS.json')
 receipts=[];chunk=[];size=0
 def pack(items,number):
  path=out/f'evidence_part{number:03d}.zip';tmp=path.with_suffix('.zip.tmp')
  with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
   for f in items:z.write(f,f.relative_to(root).as_posix())
  tmp.replace(path)
  with zipfile.ZipFile(path) as z:assert z.testzip() is None
  receipts.append(dict(path=path.name,sha256=file_hash(path),bytes=path.stat().st_size))
 for f in files:
  if chunk and size+f.stat().st_size>70*1024**2:pack(chunk,len(receipts)+1);chunk=[];size=0
  chunk.append(f);size+=f.stat().st_size
 if chunk:pack(chunk,len(receipts)+1)
 atomic_json(out/'EXPORT_INDEX.json',dict(version=p_version(),status=read_json(root/'status.json'),files=receipts,instruction='Merge ALL evidence ZIPs to resume with the same Return-v4 Dataset/runtime; no training.'))
 with zipfile.ZipFile(out/'review_small.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(analysis.rglob('*')):
   if f.is_file():z.write(f,'analysis/'+f.relative_to(analysis).as_posix())
  z.write(root/'protocol.json','protocol.json');z.write(root/'status.json','status.json');z.write(out/'EXPORT_INDEX.json','EXPORT_INDEX.json')
 print('EXPORT',out,flush=True);return out


def p_version():return 'bs5_return_v4_compact_transport_v1'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','run','export']);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--case-only',action='store_true');ap.add_argument('--max-seconds',type=float);ap.add_argument('--stop-after-games',type=int);a=ap.parse_args()
 if a.command=='prepare':prepare(a.output,a.case_only)
 elif a.command=='run':run(a.output,a.max_seconds,a.stop_after_games)
 else:export(a.output)
if __name__=='__main__':main()
