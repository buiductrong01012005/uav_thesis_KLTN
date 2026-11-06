from copy import deepcopy
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
import pytest
from src.baselines.bs5_delayed_h_only import DelayedHybridOnly
from src.envs.bs5_multihop_train import BS5MultihopTrainEnv
from scripts.evaluation import run_bs5_heuristic_ho as run
from scripts.analysis.report_bs5_heuristic_ho import base_identity,validate_protocols
ROOT=Path(__file__).resolve().parents[1]
def inputs():
 e=BS5MultihopTrainEnv(arm='H-only');return e,e.reset(seed=840001,assignment_seed=840001)
def test_fixed_h_and_inactive_mask():
 e,x=inputs();e.status[0]=2;h=DelayedHybridOnly();mask=e.get_action_mask();a=h.act(e.get_obs(),mask)
 assert a[0]==4 and all(mask[i,v] for i,v in enumerate(a)) and list(h.cached_roles)==[2]*5 and h.last_stats['roles_evaluated']==1
 assert np.all(e.logical_roles()[e.status==0]==2)
def test_only_own_current_state_changes_own_action():
 e,x=inputs();changed=x['obs'].copy();changed[4,130:132]=[0,0];changed[4,133]=.4
 assert DelayedHybridOnly().act(x['obs'],x['action_mask'])[:4]==DelayedHybridOnly().act(changed,x['action_mask'])[:4]
def test_true_ue_and_template_truth_are_unused():
 e,x=inputs();h=DelayedHybridOnly();g=DelayedHybridOnly();g.model.ue_positions[:]=[19,19];e.ue_positions[:]=[0,0]
 np.testing.assert_array_equal(x['obs'],e.get_obs());assert h.act(x['obs'],x['action_mask'])==g.act(e.get_obs(),x['action_mask'])
def test_unknown_and_expired_mass():
 h=DelayedHybridOnly();v,c,k,_=h.belief(np.zeros(130));assert c.sum()==0 and not k.any() and np.all(v.ue_positions==-1000) and list(v.roles)==[2]*5
 s=np.zeros(130);s[:25]=4;_,c,_,_=h.belief(s);assert c.sum()<=90+1e-9
def test_recovery_and_all_estimated_roles_are_h(monkeypatch):
 h=DelayedHybridOnly();s=np.zeros(130);s[18]=20/25;s[25+18]=1
 for a,pos in enumerate([[0,2],[0,8],[0,14],[6,14],[12,14]]):
  r=75+11*a;s[r:r+2]=np.array(pos)/19;s[r+4]=1;s[r+10]=1
 v,c,_,_=h.belief(s);plan=h.reconnect_plan(v,c);assert plan and list(plan[0])==[2]*5 and plan[1]
 def zero(view,roles):
  assert np.all(roles==2);h.last_stats['estimated_score_calls']+=1;return 0.
 monkeypatch.setattr(h,'estimate',zero);e,x=inputs();obs=x['obs'].copy();obs[:,:130]=s;h.act(obs,x['action_mask']);assert h.last_stats['recovery_plan']==1 and np.all(h.cached_roles==2)
def test_full_h_energy_even_without_useful_radio():
 e,x=inputs();e.uav_positions[:]=[[1,1],[18,1],[10,18],[0,0],[19,19]];e.ue_positions[:]=[10,10];before=e.battery.copy();e.step([4]*5)
 np.testing.assert_allclose(before-e.battery,.4,atol=1e-5);assert np.all(e.logical_roles()==2)
def test_bad_shapes_and_noncommon_broadcast():
 e,x=inputs();h=DelayedHybridOnly()
 with pytest.raises(ValueError):h.act(x['obs'],np.ones((5,10)))
 bad=x['obs'].copy();bad[4,0]+=1
 with pytest.raises(ValueError,match='broadcast'):h.act(bad,x['action_mask'])
def test_original_science_unchanged():
 c=json.loads((ROOT/'configs/bs5_heuristic_ho_extension.json').read_text())
 for rel,h in c['reference_common']['source_sha256'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
def test_provenance_bridge_and_mismatch_rejection(monkeypatch,tmp_path):
 c=json.loads((ROOT/'configs/bs5_heuristic_ho_extension.json').read_text());monkeypatch.setattr(run,'runtime_versions',lambda:c['reference_common']['runtime']);p=run.protocol(run.parser().parse_args(['--output',str(tmp_path)]))
 assert p['base_experiment_fingerprint']==c['reference_test_fingerprint'] and p['experiment_fingerprint']!=p['base_experiment_fingerprint']
 old=dict(p['base_common'],label='HO',arm='H-only',experiment_fingerprint=p['base_experiment_fingerprint']);validate_protocols({'HO':old,'HEURISTIC_HO':p})
 wrong=deepcopy(old);wrong['runtime']['numpy']='wrong';wrong['experiment_fingerprint']=run.digest({k:v for k,v in wrong.items() if k not in ['label','arm','experiment_fingerprint']})
 with pytest.raises(ValueError,match='Different base'):validate_protocols({'HO':wrong,'HEURISTIC_HO':p})
def test_production_runtime_guard(monkeypatch,tmp_path):
 monkeypatch.setattr(run,'runtime_versions',lambda:{'wrong':True})
 with pytest.raises(ValueError,match='runtime must match'):run.protocol(run.parser().parse_args(['--output',str(tmp_path)]))
def test_exact_resume_and_external_zip(tmp_path):
 def args(path,stop=0):return run.parser().parse_args(['--output',str(path),'--diagnostic','--stop-after-slots',str(stop),'--save-every','11','--log-every','48'])
 run.run(args(tmp_path/'full'));run.run(args(tmp_path/'split',37));run.run(args(tmp_path/'split'))
 p=tmp_path/'full/HEURISTIC_HO/seed840001';q=tmp_path/'split/HEURISTIC_HO/seed840001'
 def equal(a,b):
  with np.load(a/'trace.npz') as x,np.load(b/'trace.npz') as y:
   for k in x.files:np.testing.assert_array_equal(x[k],y[k])
 equal(p,q)
 run.run(args(tmp_path/'pause',19));restored=tmp_path/'restored';restored.mkdir()
 with zipfile.ZipFile(tmp_path/'HEURISTIC_HO_test_BENCH_RESUME.zip') as z:z.extractall(restored)
 run.run(args(restored));equal(p,restored/'HEURISTIC_HO/seed840001')
 run.run(args(tmp_path/'corrupt',7));f=tmp_path/'corrupt/HEURISTIC_HO/seed840001/resume.pkl';f.write_bytes(f.read_bytes()+b'x')
 with pytest.raises(ValueError,match='checksum'):run.run(args(tmp_path/'corrupt'))
def test_merge_real_review_and_diagnostic_guard(tmp_path):
 from scripts.analysis.merge_bs5_heuristic_ho import merge
 run.run(run.parser().parse_args(['--output',str(tmp_path/'run'),'--diagnostic','--stop-after-slots','3']))
 review=tmp_path/'HEURISTIC_HO_test_BENCH_REVIEW.zip';merge([review],tmp_path/'merged',allow_diagnostic=True)
 with pytest.raises(ValueError,match='Diagnostic'):merge([review],tmp_path/'bad')
 p=json.loads((tmp_path/'run/HEURISTIC_HO/protocol.json').read_text());p['base_common']['horizon']+=1
 with pytest.raises(ValueError,match='base identity'):base_identity(p)
