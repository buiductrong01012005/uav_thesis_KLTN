from pathlib import Path
from copy import deepcopy
import gzip,json,shutil
import numpy as np
import pytest
import torch
from scripts.experiments import run_bs5_crcn_return_v3 as pipeline
from scripts.experiments.bs5_crcn_common import ASSETS,TraceReplayReturnEnv,verify_assets,TraceBank,trace_spec
from scripts.training import run_bs5_cn_h1200_return as cn
from scripts.analysis.analyze_bs5_crcn_return_v3 import analyze
from scripts.analysis.analyze_bs5_cn_h1200_return import analyze as analyze_cn,load_game as load_cn_game
from src.envs.bs5_curriculum_env import condition_for

def test_protocol_matches_historical_recipe():
    cn.setup_runtime('cpu',1);p=cn.build_protocol(dict(sha256='test'));old=pipeline.read_json(ASSETS/'protocol.json')
    assert p['arm']=='C-N' and not p['allow_relay'] and p['game_schedule']==old['game_schedule']
    for k in ['total_steps','evaluation_steps','full_horizon','stage_ends','lr','gamma','epsilon_decay_steps','batch_size','target_every']:
        assert p[k]==old[k]
    spec=pipeline.read_json(pipeline.SPEC)
    prior=set(old['train_pool']+old['validation_pool']+old['fixture_pool']+old['stage_validation_pool'])
    assert not set(spec['heldout_scenarios'])&prior and spec['paired_evaluation_steps']==400000
    assert verify_assets()['checkpoint_sha256']==old_checkpoint_hash()
    torch.manual_seed(551)
    learner=cn.BS5CurriculumQMIX(cn.new_env(p,condition_for(0,'C-N',p),150,30),device='cpu')
    assert cn.weights_hash(learner)==verify_assets()['initial_weights_sha256']

def old_checkpoint_hash():return next(x['checkpoint_sha256'] for x in pipeline.read_json(ASSETS/'checkpoint_selection.json')['curve'] if x['step']==60000)

def test_locked_tape_shared_but_actions_endogenous(tmp_path):
    p=cn.build_protocol(dict(sha256='test'));c=condition_for(12000,'C-N',p);obs=[];envs=[]
    for relay in [True,False]:
        env=TraceReplayReturnEnv(c,allow_relay=relay,horizon=160,residual_battery=30,fixture=p['fixture'],trace_home=tmp_path/'traces')
        obs.append(env.reset(seed=180201,assignment_seed=180201,diagnostic=True));envs.append(env)
    rng=np.random.default_rng(42)
    for _ in range(160):
        actions=[int(rng.choice(np.flatnonzero(m))) for m in obs[0]['action_mask']]
        obs[0]=envs[0].step(actions);obs[1]=envs[1].step([8]*5)
        assert np.array_equal(envs[0].ue_positions,envs[1].ue_positions)
        assert not obs[1]['action_mask'][:,1::2].any()
    assert envs[0].trace_receipt['position_sha256']==envs[1].trace_receipt['position_sha256']
    assert not np.array_equal(envs[0].uav_positions,envs[1].uav_positions)

@pytest.fixture(scope='module')
def diagnostic_runs(tmp_path_factory):
    base=tmp_path_factory.mktemp('cn_exact');a=base/'a';b=base/'b';cn.setup_runtime('cpu',1)
    cn.prepare(a,'cpu',diagnostic=True);cn.run(a)
    cn.prepare(b,'cpu',diagnostic=True);cn.run(b,stop_after=96);cn.run(b,resume=True)
    return base,a,b

def test_exact_cn_resume(diagnostic_runs):
    base,a,b=diagnostic_runs;aa=torch.load(a/'resume.pt',map_location='cpu',weights_only=False);bb=torch.load(b/'resume.pt',map_location='cpu',weights_only=False)
    for k in aa['models']:
        for name in aa['models'][k]:assert torch.equal(aa['models'][k][name],bb['models'][k][name])
    assert aa['replay']['rng']==bb['replay']['rng'] and aa['replay']['pos']==bb['replay']['pos']
    for k in aa['replay']['arrays']:np.testing.assert_array_equal(aa['replay']['arrays'][k],bb['replay']['arrays'][k])
    for rel in aa['commit']['games']:
        _,ra,ua=load_cn_game(a,rel);_,rb,ub=load_cn_game(b,rel);assert ra==rb and ua==ub
    out=analyze_cn(a,base/'deep_cn',require_complete=True,manifest_required=False)
    assert out['training_steps']==192 and out['validation_steps']==72

@pytest.mark.parametrize('scenario',[130001,130009])
def test_historical_cr_policy_prefix(tmp_path,scenario):
    cn.setup_runtime('cpu',1);cn.TRACE_HOME=tmp_path/'traces';p=pipeline.read_json(ASSETS/'protocol.json');p.update(arm='C-R',allow_relay=True)
    c=condition_for(12000,'C-R',p);learner,step=pipeline.load_checkpoint(cn.new_env(p,c,1200,30),ASSETS/'060000.pt')
    assert step==60000 and cn.weights_hash(learner)==verify_assets()['weights_sha256']
    root=tmp_path/'control';root.mkdir();cn.atomic_json(root/'budget_ledger.json',dict(phases={'eval':dict(cap=1200,charged=0,observed=0)}))
    state=dict(steps=step,timers={k:0. for k in ['env_s','actor_s','record_s','raw_io_s']})
    config=pipeline.read_json(pipeline.SPEC)
    result,rel=cn.rollout(root,learner,None,np.random.default_rng(0),p,cn.Budget(root),state,evaluation=True,
        spec=dict(condition=c,horizon=1200,residual_battery=30.,scenario_seed=scenario,assignment_seed=scenario),
        prefix_validator=lambda rows:pipeline.compare_prefix(rows,'C-R',scenario,tmp_path,config))
    assert result['trace']['origin']=='recorded_C-R_validation'

@pytest.fixture(scope='module')
def paired_diagnostic(diagnostic_runs):
    base,a,b=diagnostic_runs;root=base/'pipeline';pipeline.prepare(root,diagnostic=True)
    shutil.copytree(a,root/'cn',dirs_exist_ok=True)
    pipeline.run(root,stop_after_games=1);assert pipeline.read_json(root/'status.json')['completed']==1
    pipeline.run(root);assert pipeline.read_json(root/'status.json')['status']=='complete'
    return root

def test_paired_resume_and_offline_export(paired_diagnostic):
    root=paired_diagnostic;result=analyze(root,root.parent/'paired_checked',deep=True,require_complete=True)
    assert result['committed_slots']=={'cn_train':192,'cn_validation':72,'paired_C-R':48,'paired_C-N':48}
    export=pipeline.export(root)
    import zipfile,subprocess,sys
    merged=root.parent/'merged';merged.mkdir()
    index=pipeline.read_json(export/'EXPORT_INDEX.json')
    for item in index['files']:
        assert pipeline.file_hash(export/item['path'])==item['sha256']
        with zipfile.ZipFile(export/item['path']) as z:z.extractall(merged)
    code="""import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','numpy','src'}:raise ImportError('blocked '+fullname)
sys.meta_path.insert(0,Block());sys.path.insert(0,sys.argv[1]);sys.argv=[sys.argv[1]+'/analyze_bs5_crcn_return_v3.py','--root',sys.argv[1],'--output',sys.argv[2],'--require-complete'];runpy.run_path(sys.argv[0],run_name='__main__')
"""
    r=subprocess.run([sys.executable,'-B','-c',code,str(merged),str(root.parent/'standalone_checked')],capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    # A completed re-run is idempotent and never repeats physical training/eval steps.
    before=pipeline.read_json(root/'paired/commit.json');pipeline.run(root);assert pipeline.read_json(root/'paired/commit.json')==before

def test_uncertain_eval_cannot_repeat(diagnostic_runs):
    base,a,b=diagnostic_runs;root=base/'uncertain';pipeline.prepare(root,diagnostic=True);shutil.copytree(a,root/'cn',dirs_exist_ok=True)
    p=pipeline.read_json(root/'protocol.json');pipeline.prepare_pair(root,p)
    path=root/'paired/C-R/budget_ledger.json';ledger=pipeline.read_json(path);ledger['phases']['eval']['charged']=1;pipeline.atomic_json(path,ledger)
    with pytest.raises(AssertionError,match='Uncommitted'):pipeline.run(root)
    assert pipeline.read_json(root/'status.json')['status']=='partial_error'

def test_prepared_export_preserves_zero_budget(tmp_path):
    root=tmp_path/'prepared';pipeline.prepare(root,diagnostic=True)
    out=pipeline.export(root)
    result=pipeline.read_json(out/'analysis/summary.json')
    assert not result['complete'] and result['updates']==0
    assert pipeline.read_json(root/'cn/budget_ledger.json')['phases']['train']['charged']==0
