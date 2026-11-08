"""Metadata/IO/bootstrap/statistics tests. Synthetic numbers are fixtures, not DRL results."""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import ast,csv,io,json,tempfile,unittest,zipfile,hashlib
import numpy as np
from fourarm400k_fixture import load_api,ROOT
from scripts.analysis.fourarm400k_multiseed import collect_training,make_bundle,ARMS,SEEDS,digest
from scripts.analysis.report_fourarm400k_multiseed import aggregate,expected_keys,report,LABELS
from scripts.evaluation import run_fourarm400k_drl_benchmark as drl
from scripts.evaluation import run_fourarm400k_heuristics as core
from scripts.notebooks.build_fourarm400k_multiseed import build,BOOTSTRAP,DRL_RESUME,TRAIN_RESUME
API=load_api()
def csv_bytes(rows):
 f=io.StringIO();w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows);return f.getvalue().encode()
def fixture_training(arm,seed):
 p=API['load_protocol'](API['parser']().parse_args(['--arm',arm,'--training-seed',str(seed),'--output','fixture']))
 best=dict(step=399600,score=.5,file='selected/best_step399600.pt')
 st=dict(arm=arm,training_env_version=p['version'],target_steps=399600,status='complete',committed_training_steps=399600,best_by_metric={'reward_total':best})
 v=[dict(step=step,seed=s,reward_total=.5 if step==399600 else .1,throughput_mbps=100+seed%10,qos_ues=20,service_coverage_ues=40,jain_time_average=.9,joules_per_mbit=20) for step in p['evaluation_steps'] for s in p['validation_seeds']]
 payload=b'FIXTURE-NOT-TRAINED-'+arm.encode()+str(seed).encode()
 return {'protocol.json':json.dumps(p).encode(),'status.json':json.dumps(st).encode(),'training.csv':csv_bytes([dict(step=399600)]),'validation.csv':csv_bytes(v),'best.pt':payload,'selected/best_step399600.pt':payload}
def put_zip(path,files):
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
  for k,v in files.items():z.writestr(k,v)
class MultiseedTests(unittest.TestCase):
 def test_catalog_keeps_twelve_runs_and_freezes_all_not_best_seed(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for a in ARMS:
    for s in SEEDS:put_zip(root/f'{a}_{s}.zip',{'run/'+k:v for k,v in fixture_training(a,s).items()})
   with zipfile.ZipFile(root/'C-R_551.zip') as z:z.extractall(root/'also_extracted')
   runs,_=collect_training(root,True);self.assertEqual(len(runs),12)
   m=make_bundle(runs,root/'FOURARM400K_FROZEN_MODELS.zip');self.assertEqual(len(m['models']),12)
   rec,p,payload,bh=drl.read_model(root,'C-R',553);self.assertEqual(p['training_seed'],553);self.assertTrue(payload.startswith(b'FIXTURE'))
   testp=drl.make_protocol('C-R',553,1,rec,p,bh)
   self.assertEqual(testp['environment_fingerprint'],core.make_protocol('DYNAMIC_AR',1)['environment_fingerprint']);self.assertEqual(testp['target_seeds'],list(range(930201,930206)))
   files=fixture_training('C-R',552);v=files['validation.csv'].decode().replace('0.5','0.4');files['validation.csv']=v.encode();put_zip(root/'C-R_552.zip',{'run/'+k:v for k,v in files.items()})
   with self.assertRaises(ValueError):collect_training(root,True)
 def test_missing_duplicate_and_science_change_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);files=fixture_training('C-N',551);put_zip(root/'one.zip',files)
   runs,_=collect_training(root,True)
   with self.assertRaises(ValueError):make_bundle(runs,root/'bundle.zip')
   p=json.loads(files['protocol.json']);p['lr']=.123;p['comparison_fingerprint']=digest({k:v for k,v in p.items() if k not in {'arm','policy_actions','runtime','torch_version','comparison_fingerprint'}})
   files['protocol.json']=json.dumps(p).encode();put_zip(root/'conflict.zip',files)
   with self.assertRaises(ValueError):collect_training(root)
 def test_crossed_pairing_and_no_heuristic_pseudoreplication(self):
  scenarios=list(range(930201,930211));results={}
  for label,ts,sc in expected_keys(LABELS,scenarios):
   value=(ts-551)*10+(sc-930201) if ts is not None else sc-930201
   if label=='C-R':value+=2
   results[label,ts,sc]=dict(throughput_mbps=value,energy_consumed_wh=1,delivered_mbit=2,active_blocked_total=1,policy_move_requests_total=10)
  rows,pairs=aggregate(results,LABELS,scenarios,replicates=200)
  self.assertEqual(len(results),140);self.assertEqual(rows[-1]['actual_evaluation_runs'],10);self.assertEqual(rows[0]['actual_evaluation_runs'],30)
  diff=next(r for r in pairs if r['reference']=='C-N' and r['comparison']=='C-R' and r['metric']=='throughput_mbps')
  self.assertAlmostEqual(diff['difference'],2);self.assertAlmostEqual(diff['ci95_low'],2);self.assertAlmostEqual(diff['ci95_high'],2)
  self.assertAlmostEqual(rows[0]['joules_per_mbit_pooled'],1800)
 def test_complete_six_label_report_plots_from_explicit_synthetic_fixture(self):
  from scripts.analysis.report_fourarm400k_heuristics import METRICS
  from scripts.analysis.report_fourarm400k_multiseed import LABELS,expected_keys
  scenarios=[940201];results={}
  for label,ts,sc in expected_keys(LABELS,scenarios):
   row={k:.1 for k in METRICS};row.update(configuration=label,training_seed=ts,seed=sc,throughput_mbps=120.,qos_ues=25.,service_coverage_ues=45.,jain_time_average=.95,p05_ue_mean_rate_mbps=1.1,energy_consumed_wh=3750.,delivered_mbit=720000.,active_blocked_total=200.,policy_move_requests_total=2000.,actor_ms=2. if ts is not None else 300.)
   results[label,ts,sc]=row
  proto=core.make_protocol('DYNAMIC_AR',1,True)
  with tempfile.TemporaryDirectory() as tmp,patch('scripts.analysis.report_fourarm400k_multiseed.scan',return_value=(results,[proto],[])):
   meta=report('fixture-never-read',Path(tmp),False,True);self.assertEqual(meta['status'],'diagnostic_only');self.assertEqual(meta['completed_runs'],14)
   self.assertTrue((Path(tmp)/'training_seed_sensitivity.png').is_file())
   text=(Path(tmp)/'aggregate.csv').read_text();self.assertIn('HEURISTIC_HO',text)
 def test_forty_independent_notebooks_compile_offline_cpu(self):
  with tempfile.TemporaryDirectory() as tmp:
   build(tmp);files=list(Path(tmp).rglob('*.ipynb'));self.assertEqual(len(files),40)
   self.assertEqual(len(list((Path(tmp)/'01_train_more').glob('*.ipynb'))),8)
   self.assertEqual(len(list((Path(tmp)/'02_heuristics_now').glob('*.ipynb'))),4)
   self.assertEqual(len(list((Path(tmp)/'03_drl_test_after_train').glob('*.ipynb'))),24)
   for path in files:
    nb=json.loads(path.read_text());code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code');compile(code,str(path),'exec')
    self.assertNotIn('pip install',code);self.assertNotIn('--device","cuda',code)
    if path.name.startswith('T'):self.assertIn('TOTAL_STEPS=399600',code);self.assertTrue('TRAINING_SEED=552' in code or 'TRAINING_SEED=553' in code)
    if path.name.startswith(('T','H','B')):self.assertIn('RUN_PREFLIGHT=True',code)
 def test_new_dataset_zip_extracted_duplicate_and_tampering(self):
  import shutil
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);inp=root/'input';work=root/'work';inp.mkdir();work.mkdir();ns=dict(Path=Path,json=json,hashlib=hashlib,zipfile=zipfile,shutil=shutil,tempfile=tempfile)
   tree=ast.parse(BOOTSTRAP);exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'notebook','exec'),ns)
   files={'uav_thesis/'+rel:(ROOT/rel).read_bytes() for rel in ['scripts/training/run_bs5_fourarm400k.py','configs/bs5_fourarm400k.json','scripts/analysis/compare_bs5_fourarm400k_training.py']}
   m=dict(version='fourarm400k_multiseed_dataset_v1',files_sha256={k:hashlib.sha256(v).hexdigest() for k,v in files.items()});files['MULTI400K_DATASET_MANIFEST.json']=json.dumps(m).encode();z=inp/'UAV_MULTI400K_CPU_DATASET.zip';put_zip(z,files)
   source=ns['load_source'](inp,work)
   with zipfile.ZipFile(z) as a:a.extractall(inp/'auto')
   self.assertEqual(source,ns['load_source'](inp,work));z.unlink();self.assertEqual(source,ns['load_source'](inp,work))
   (source/'configs/bs5_fourarm400k.json').write_text('bad')
   with self.assertRaises(RuntimeError):ns['load_source'](inp,work)
 def test_resume_notebook_rejects_wrong_training_seed(self):
  import shutil
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);ns=dict(Path=Path,json=json,zipfile=zipfile,shutil=shutil,tempfile=tempfile,WORKING=root)
   for text in [BOOTSTRAP,DRL_RESUME,TRAIN_RESUME]:
    tree=ast.parse(text);exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'notebook','exec'),ns)
   p=dict(version=drl.VERSION,arm='C-R',training_seed=552,part=1,target_seeds=list(range(930201,930206)),horizon=10000,diagnostic=False)
   put_zip(root/'resume.zip',{'run/protocol.json':json.dumps(p).encode()})
   ns['restore_drl_test'](root/'resume.zip',root/'correct','C-R',552,1,None)
   with self.assertRaises(ValueError):ns['restore_drl_test'](root/'resume.zip',root/'wrong','C-R',553,1,None)
if __name__=='__main__':unittest.main(verbosity=2)
