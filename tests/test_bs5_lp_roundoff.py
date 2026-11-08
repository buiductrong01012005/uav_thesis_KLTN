"""Numeric regression, strict rejection, and explicit old/new source compatibility."""
from pathlib import Path
from copy import deepcopy
import importlib.util,json,os,tempfile,unittest
from unittest.mock import patch
import numpy as np
from src.diagnostics import bs5_spectrum_airtime as radio
from src.envs.bs5_tcf_env import rate_metrics
from scripts.analysis.fourarm400k_lp_compat import canonical_sources,canonical_environment,manifest,verify_source
ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'tests/fixtures/lp_roundoff_negative_rate.json'
spec=importlib.util.spec_from_file_location('lp_legacy_fixture',ROOT/'tests/fixtures/lp_allocator_legacy_reference.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)

def inputs():
 d=json.loads(FIXTURE.read_text());return {int(k):v for k,v in d['offers'].items()},{int(k):v for k,v in d['paths'].items()},np.asarray(d['capacities'])
def to_rates(flows,offers):
 r=np.zeros(90);offset=0
 for j in sorted(offers):
  n=max(1,int(np.ceil(offers[j]/4)));r[offset:offset+n]=flows[j]/n;offset+=n
 return r
class RoundoffTests(unittest.TestCase):
 def test_real_highs_negative_ue_regression_is_fixed(self):
  offers,paths,C=inputs();saved=json.loads((ROOT/'tests/fixtures/lp_roundoff_solver_output.json').read_text())
  self.assertLess(min(saved['rates']),-1e-9)
  with self.assertRaises(ValueError):rate_metrics(saved['rates'])
  with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'UAV_LP_DIAGNOSTIC_DIR':tmp}):
   # Fixed raw solver output makes the regression independent of HiGHS version.
   x,violation=radio._clean_lp_solution(saved['x'],sorted(offers),offers,paths,C,np.asarray(saved['demand']),np.asarray(saved['a_ub']),np.asarray(saved['b_ub']),saved['bounds'],'saved-regression')
   rate_metrics(to_rates(dict(zip(sorted(offers),x[:5])),offers));self.assertLessEqual(violation,1e-6)
   records=[json.loads(s) for s in (Path(tmp)/'lp_numerics.jsonl').read_text().splitlines()]
   self.assertTrue(records);self.assertTrue(all(r['version']=='lp_bound_roundoff_v1' for r in records))
   # Also exercise the actual installed solver. It may or may not reproduce
   # the old negative value; either way the corrected output must be valid.
   flows,durations,minimum,violation=radio.allocate_flows(offers,paths,C)
   self.assertGreaterEqual(min(flows.values()),0);rate_metrics(to_rates(flows,offers));self.assertLessEqual(violation,1e-6)
 def test_normal_positive_optimum_is_unchanged(self):
  for offers,paths,C in [({0:4.},{0:[0]},np.array([[10.]])),({0:4.,1:8.},{0:[0],1:[1]},np.array([[100.,0],[0,100.]]))]:
   a=legacy.allocate_flows(offers,paths,C);b=radio.allocate_flows(offers,paths,C)
   self.assertEqual(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1]);self.assertEqual(a[2:],b[2:])
 def clean(self,x,C=None):
  ids=[0,1];offers={0:4.,1:4.};paths={0:[0],1:[1]};C=np.diag([10.,0.]) if C is None else C;D=np.eye(2)
  A=np.zeros((3,5));A[:2,:2]=D;A[:2,2:4]=-C;A[2,2:4]=1
  return radio._clean_lp_solution(np.asarray(x),ids,offers,paths,C,D,A,np.array([0.,0.,1.]),[(0,4),(0,4),(0,1),(0,1),(0,1)],'test')
 def test_diagnostics_accept_numpy_ids_without_interrupting_a_repair(self):
  with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'UAV_LP_DIAGNOSTIC_DIR':tmp}):
   aid=np.int64(0)
   radio._lp_diagnostic('test',[aid],{aid:np.float64(1)},{aid:[np.int64(0)]},np.array([[2.]]),np.array([-1e-9,.5,0]),np.array([0,.5,0]),'numpy-id fixture')
   record=json.loads((Path(tmp)/'lp_numerics.jsonl').read_text().splitlines()[-1]);self.assertEqual(record['ids'],[0]);self.assertEqual(record['paths'],[[0]])
 def test_zero_capacity_leaf_does_not_erase_unrelated_flow(self):
  x,v=self.clean([4.,-2e-9,.4,0.,0.]);self.assertEqual(x[0],4);self.assertEqual(x[1],0);self.assertEqual(x[-1],0);self.assertLessEqual(v,1e-6)
 def test_nan_large_negative_and_large_overoffer_still_fail(self):
  for x in [[np.nan,0,.4,0,0],[-.001,0,.4,0,0],[4.01,0,.4,0,0]]:
   with self.assertRaises(FloatingPointError):self.clean(x)
  with self.assertRaises(ValueError):rate_metrics(np.array([np.nan]))
  with self.assertRaises(ValueError):rate_metrics(np.array([4.01]))
 def test_capacity_violation_is_not_hidden(self):
  with self.assertRaises(FloatingPointError):self.clean([4,0,0,0,0])
 def test_bad_inputs_still_fail(self):
  for offers,C in [({0:np.nan},np.array([[1.]])),({0:-1.},np.array([[1.]])),({0:1.},np.array([[np.nan]]))]:
   with self.assertRaises(ValueError):radio.allocate_flows(offers,{0:[0]},C)
 def test_compatibility_only_accepts_the_two_declared_source_bytes(self):
  m=manifest();path=next(iter(m['files']));rec=m['files'][path]
  old={path:rec['legacy_sha256'],'other.py':'unchanged'};new={path:rec['guarded_sha256'],'other.py':'unchanged'}
  self.assertEqual(canonical_sources(old),canonical_sources(new));self.assertEqual(new[path],rec['guarded_sha256'])
  verify_source(path,rec['legacy_sha256']);verify_source(path,rec['guarded_sha256'])
  with self.assertRaises(ValueError):canonical_sources({path:'unknown'})
  with self.assertRaises(ValueError):verify_source(path,'unknown')
  self.assertEqual(canonical_environment({'physics_source_sha256':old}),canonical_environment({'physics_source_sha256':new}))
 def test_old_and_new_training_runs_can_be_collected_without_rewriting_protocols(self):
  from test_fourarm400k_multiseed import fixture_training,put_zip
  from scripts.analysis.fourarm400k_multiseed import collect_training,digest
  m=manifest();path=next(iter(m['files']))
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for seed,variant in [(551,'legacy_sha256'),(552,'guarded_sha256')]:
    files=fixture_training('H-only',seed);p=json.loads(files['protocol.json']);p['source_sha256'][path]=m['files'][path][variant]
    common={k:v for k,v in p.items() if k not in {'arm','policy_actions','runtime','torch_version','comparison_fingerprint'}};p['comparison_fingerprint']=digest(common);files['protocol.json']=json.dumps(p).encode();put_zip(root/f'{seed}.zip',files)
   runs,provenance=collect_training(root,True);self.assertEqual(len(runs),2)
   self.assertEqual(runs['H-only',551]['data']['protocol']['source_sha256'][path],m['files'][path]['legacy_sha256'])
   self.assertEqual({p['numerical_variant'] for p in provenance},{'legacy','guarded'})
if __name__=='__main__':unittest.main(verbosity=2)
