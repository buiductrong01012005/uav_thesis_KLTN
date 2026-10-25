from pathlib import Path
import json,copy,zipfile,subprocess,sys
import numpy as np
import pytest
from scripts.experiments import run_bs5_return_v4_pair as run
from scripts.experiments.bs5_return_v4_common import verify_assets,ASSETS,new_env


@pytest.mark.parametrize('arm,scenario',[('C-R',130001),('C-N',190006)])
def test_original_v3_checkpoint_and_actor_inputs_reproduce_1200(arm,scenario):
    run.setup();env=new_env(arm,version='v3');model=run.actor(env,arm)
    out=env.reset(seed=scenario,assignment_seed=scenario,diagnostic=True)
    with np.load(ASSETS/f'prefix_{arm}_{scenario}.npz') as z:
        for t in range(1200):
            np.testing.assert_array_equal(out['obs'],z['obs'][t])
            np.testing.assert_array_equal(out['state'],z['state'][t])
            np.testing.assert_array_equal(out['action_mask'],z['mask'][t])
            actions=run.choose(model,out);np.testing.assert_array_equal(actions,z['actions'][t])
            out=env.step(actions)
            np.testing.assert_array_equal(env.uav_positions,z['post_positions'][t])
            np.testing.assert_array_equal(env.battery,z['post_battery'][t])
            np.testing.assert_allclose(env.last_service_snapshot['network']['ue_rate_mbps'],z['rates'][t],atol=1e-6,rtol=1e-7)


def test_frozen_protocol_weights_and_no_training_budget(tmp_path):
    root=tmp_path/'prepared';p=run.prepare(root)
    assert p['training_steps']==0 and p['planned_steps']==400000 and len(p['jobs'])==40
    assert p['checkpoints']['C-N']['step']==36000 and p['checkpoints']['C-R']['step']==60000
    assert p['jobs']==verify_assets()['jobs']
    ledger=run.read_json(root/'budget_ledger.json');ledger['phases']['C-N']['charged']=1;run.atomic_json(root/'budget_ledger.json',ledger)
    with pytest.raises(AssertionError,match='Uncommitted'):run.run(root)
    assert run.read_json(root/'status.json')['status']=='partial_error'


def test_compact_physics_audit_metrics_export_and_boundary_resume(tmp_path):
    # Short fixed-input diagnostic, not an additional full-horizon selection run.
    from scripts.analysis.bs5_return_v4_results import audit_game,analyze_run
    root=tmp_path/'tiny';p=run.prepare(root,case_only=True)
    p['horizon']=24;p['planned_steps']=24;run.atomic_json(root/'protocol.json',p)
    ledger=run.read_json(root/'budget_ledger.json');ledger['phases']['C-N']['cap']=24;run.atomic_json(root/'budget_ledger.json',ledger)
    run.run(root,stop_after_games=0);assert run.read_json(root/'budget_ledger.json')['phases']['C-N']['charged']==0
    run.run(root);assert run.read_json(root/'status.json')['status']=='complete'
    item=run.read_json(root/'commit.json')['completed'][0];idx=run.read_json(root/item['index'])
    arrays=audit_game((root/item['index']).parent,idx,deep=True)
    assert arrays['obs'].shape==(25,5,116) and arrays['rates'].dtype==np.float64
    report=analyze_run(root,tmp_path/'checked',deep=True,require_complete=True);assert report['committed_steps']==24 and report['training_steps']==0
    exported=run.export(root);merged=tmp_path/'merged';merged.mkdir()
    for part in run.read_json(exported/'EXPORT_INDEX.json')['files']:
        with zipfile.ZipFile(exported/part['path']) as z:assert z.testzip() is None;z.extractall(merged)
    code="""import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','src'}:raise ImportError('blocked '+fullname)
sys.meta_path.insert(0,Block());sys.path.insert(0,sys.argv[1]);sys.argv=[sys.argv[1]+'/bs5_return_v4_results.py','--root',sys.argv[1],'--output',sys.argv[2],'--deep-planner','--require-complete'];runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result=subprocess.run([sys.executable,'-B','-c',code,str(merged),str(tmp_path/'offline')],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    # An exported completed run may be opened again without repeating a single slot.
    before=run.read_json(root/'budget_ledger.json');run.run(root)
    assert run.read_json(root/'budget_ledger.json')==before
