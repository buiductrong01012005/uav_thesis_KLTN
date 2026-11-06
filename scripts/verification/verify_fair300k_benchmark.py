"""Before test: real checkpoint metadata/forward, validation replay and exact diagnostic resume."""
from pathlib import Path
import argparse,csv,json,os,subprocess,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.evaluation import run_fair300k_benchmark as bench

def equal(a,b):
 if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b,(a,b)

def verify(label,output):
 if output.exists():raise ValueError('Preflight requires a fresh output directory')
 output.mkdir(parents=True)
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 for pattern in ['test_bs5_temporal_fair.py','test_bs5_fair300k_benchmark.py']:
  subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p',pattern,'-v'],cwd=ROOT,env=env,check=True)
 bench.configure_torch();p=bench.make_protocol(label,1,True);p['runtime']=bench.runtime();policy=bench.load_policy(p)
 seed=p['target_seeds'][0];continuous=bench.start_state(p,seed)
 for _ in range(96):bench.advance(continuous,policy,p)
 split=bench.start_state(p,seed)
 for _ in range(48):bench.advance(split,policy,p)
 folder=output/'diagnostic_resume';folder.mkdir();bench.save_state(folder,p,split);split=bench.load_state(folder,p)
 for _ in range(48):bench.advance(split,policy,p)
 for k in ['arrays','energy','collision','network','cycles','pending','rejoins','ue_trace_sha256','current']:
  # current info includes wall-clock radio profiling, so compare actual policy inputs separately.
  if k=='current':
   for field in ['obs','state','action_mask','terminated','truncated','reward']:equal(continuous[k][field],split[k][field])
  else:equal(continuous[k],split[k])
 equal(continuous['rng'].bit_generator.state,split['rng'].bit_generator.state)
 ignored={'actor_ms','radio_ms'}
 equal([{k:v for k,v in r.items() if k not in ignored} for r in continuous['rows']],
       [{k:v for k,v in r.items() if k not in ignored} for r in split['rows']])
 bench.save_seed(folder,p,split,True);bench.verify_seed_commit(folder,p)
 # Reproduce the actual validation used for the frozen model before touching reserved test seeds.
 q=bench.make_protocol(label,1,True);q['horizon']=1200
 state=bench.start_state(q,720001)
 for i in range(1200):
  bench.advance(state,policy,q)
  if (i+1)%300==0:print(f'Validation replay {label}: {i+1}/1200',flush=True)
 actual=bench.summarize(state,q,True)
 with bench.benchmark_asset_path(bench.ASSETS,p['selection']['validation_file']).open(newline='') as stream:
  expected=next(r for r in csv.DictReader(stream) if int(r['step'])==p['selection']['selected_step'] and int(r['seed'])==720001)
 checked=['throughput_mbps','qos_ues','service_coverage_ues','jain_ema','jain_instant_qualified','reward_total','energy_consumed_wh','jain_time_average','active_blocked_total']
 for key in checked:
  if not np.isclose(actual[key],float(expected[key]),atol=1e-7,rtol=1e-9):
   raise AssertionError(f'Frozen validation replay mismatch {key}: actual={actual[key]}, expected={expected[key]}. Test has not started.')
 result=dict(status='passed',label=label,selected_step=p['selection']['selected_step'],diagnostic_seed=seed,diagnostic_slots=96,resume_split=48,resume_scientific_trace_equal=True,validation_replayed=dict(seed=720001,slots=1200,metrics=checked),test_seeds_used=False,runtime=p['runtime'])
 bench.atomic_json(output/'PREFLIGHT_PASSED.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--label',choices=['EMA','OLD'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(a.label,a.output.resolve())
