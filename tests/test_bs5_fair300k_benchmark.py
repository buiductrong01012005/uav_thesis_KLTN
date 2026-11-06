"""Environment/IO tests for the frozen fairness benchmark; no PyTorch needed."""
from pathlib import Path
from copy import deepcopy
import json,sys,tempfile,unittest,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.evaluation import run_fair300k_benchmark as b
from scripts.evaluation.fair300k_benchmark_metrics import rolling_unique
from scripts.analysis import report_fair300k_benchmark as report

class FixturePolicy:
 def act(self,obs,mask,epsilon,rng):return [int(rng.choice(np.flatnonzero(m))) for m in mask]

def drive(state,p,n):
 for _ in range(n):b.advance(state,FixturePolicy(),p)
 return state

def equal(a,c):
 if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,c)
 elif isinstance(a,dict):
  assert a.keys()==c.keys()
  for k in a:equal(a[k],c[k])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(c)
  for x,y in zip(a,c):equal(x,y)
 else:assert a==c

class BenchmarkTests(unittest.TestCase):
 def test_frozen_models_and_partitions(self):
  reg=b.verify_assets();self.assertEqual(reg['EMA']['selected_step'],294000);self.assertEqual(reg['OLD']['selected_step'],288000)
  fps=set()
  for label in ['EMA','OLD']:
   parts=[b.make_protocol(label,i) for i in [1,2]]
   self.assertEqual(parts[0]['target_seeds']+parts[1]['target_seeds'],list(range(930101,930111)))
   self.assertEqual([p['horizon'] for p in parts],[10000,10000]);fps.update(p['comparison_fingerprint'] for p in parts)
  self.assertEqual(len(fps),1)
 def test_both_modes_matched_physics_and_inputs(self):
  p=b.make_protocol('EMA',1,True);q=b.make_protocol('OLD',1,True)
  a=b.start_state(p,940101);c=b.start_state(q,940101)
  for _ in range(120):
   b.advance(a,FixturePolicy(),p);b.advance(c,FixturePolicy(),q)
   for key in a['arrays']:np.testing.assert_array_equal(a['arrays'][key][-1],c['arrays'][key][-1])
   self.assertEqual(a['current']['obs'].shape,(5,173));self.assertEqual(a['current']['state'].shape,(156,))
   for state in [a,c]:
    row=state['rows'][-1];self.assertAlmostEqual(row['reward_total'],row['reward_throughput']+row['reward_fairness']+row['reward_energy']+row['reward_collision'])
 def test_exact_environment_resume_48_then96(self):
  p=b.make_protocol('EMA',1,True);a=drive(b.start_state(p,940101),p,96);c=drive(b.start_state(p,940101),p,48)
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp);b.save_state(folder,p,c);c=b.load_state(folder,p);drive(c,p,48)
   for k in ['arrays','energy','collision','network','cycles','pending','rejoins','ue_trace_sha256']:equal(a[k],c[k])
   skip={'actor_ms','radio_ms'}
   equal([{k:v for k,v in r.items() if k not in skip} for r in a['rows']], [{k:v for k,v in r.items() if k not in skip} for r in c['rows']])
   wrong=deepcopy(p);wrong['label']='OLD'
   with self.assertRaisesRegex(ValueError,'integrity/protocol'):b.load_state(folder,wrong)
   meta=json.loads((folder/'progress.json').read_text());f=folder/meta['resume_file'];f.write_bytes(f.read_bytes()+b'corrupt')
   with self.assertRaises(ValueError):b.load_state(folder,p)
 def test_seed_commit_hash_and_exports(self):
  p=b.make_protocol('EMA',1,True);s=drive(b.start_state(p,940101),p,12)
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'worker';root.mkdir();b.atomic_json(root/'protocol.json',p);folder=root/'seed940101';folder.mkdir()
   b.save_state(folder,p,s);b.save_seed(folder,p,s,False);self.assertFalse(b.verify_seed_commit(folder,p)['complete'])
   b.export(root)
   with zipfile.ZipFile(root.parent/'worker_REVIEW.zip') as z:self.assertFalse(any(n.endswith('.pkl') or n.endswith('progress.json') for n in z.namelist()))
   with zipfile.ZipFile(root.parent/'worker_RESUME.zip') as z:self.assertTrue(any(n.endswith('.pkl') for n in z.namelist()))
   file=folder/'slots.csv';file.write_text(file.read_text()+'corrupt')
   with self.assertRaisesRegex(ValueError,'Corrupt'):b.verify_seed_commit(folder,p)
 def test_rolling_window_and_paired_statistic(self):
  mask=np.array([[1,0],[0,1],[1,0],[0,0]],bool)
  self.assertEqual(rolling_unique(mask,2),(2+2+1)/3);self.assertIsNone(rolling_unique(mask,5))
  rows={}
  for seed in range(10):
   for label in ['EMA','OLD']:
    rows[label,seed]=dict(throughput_mbps=10+seed+(label=='OLD'),energy_consumed_wh=1,delivered_mbit=100,policy_move_requests_total=10,active_blocked_total=1)
  aggregate,paired=report.comparison(rows,list(range(10)));result=next(r for r in paired if r['metric']=='throughput_mbps')
  self.assertEqual(result['old_minus_ema'],1);self.assertEqual(result['delta_ci95_low'],1);self.assertEqual(aggregate[0]['joules_per_mbit_pooled'],36)
 def test_partial_results_zip_and_extracted(self):
  p=b.make_protocol('EMA',1)
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);worker=root/'worker';worker.mkdir();b.atomic_json(worker/'protocol.json',p);seed=worker/'seed930101';seed.mkdir();b.atomic_json(seed/'summary.json',{'complete':False})
   with zipfile.ZipFile(root/'worker_REVIEW.zip','w') as z:
    for f in worker.rglob('*'):
     if f.is_file():z.write(f,'worker/'+str(f.relative_to(worker)))
   results,protocol,_=report.scan_inputs(root);self.assertEqual(results,{});self.assertEqual(protocol['label'],'EMA')
   out=root/'report';meta=report.report(root,out);self.assertEqual(meta['status'],'partial');self.assertFalse((out/'aggregate.csv').exists())
 def test_long_fixture_1200_each_mode(self):
  for label in ['EMA','OLD']:
   p=b.make_protocol(label,1,True);p['horizon']=1200;s=drive(b.start_state(p,940101),p,1200);row=b.summarize(s,p,True)
   self.assertTrue(row['complete']);self.assertEqual(row['slots'],1200);self.assertLess(row['energy_balance_max_abs_j'],1e-6)
   self.assertIsNotNone(row['rolling_unique_ge1_5min']);self.assertEqual(np.asarray(s['arrays']['ema_rates']).shape,(1200,90))
   self.assertAlmostEqual(sum(row[k] for k in ['loss_outside_access_mbps','loss_association_mbps','loss_access_radio_mbps','loss_bh_disconnected_mbps','loss_bh_schedule_capacity_mbps'])+row['throughput_mbps'],360)

if __name__=='__main__':unittest.main(verbosity=2)
