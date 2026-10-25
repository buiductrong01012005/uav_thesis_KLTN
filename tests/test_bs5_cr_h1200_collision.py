"""Bounded preflight: 384 diagnostic training + 1558 control/eval env steps.

No main seed training. Reuses both diagnostic runs across tests to avoid reruns.
"""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from scripts.training import run_bs5_cr_h1200_collision as runner
from scripts.training.run_bs5_curriculum_pilot import protocol_new
from scripts.analysis.bs5_bundle import read_json, canonical_hash
from scripts.analysis.analyze_bs5_cr_h1200_collision import analyze, load_game
from src.envs.bs5_curriculum_env import condition_for


@pytest.fixture(scope="module")
def diagnostic_runs(tmp_path_factory):
    base=tmp_path_factory.mktemp("h1200_preflight")
    a,b=base/"uninterrupted",base/"resumed"
    runner.prepare(a,diagnostic=True)
    runner.run(a)
    runner.prepare(b,diagnostic=True)
    runner.run(b,stop_after=96)
    with pytest.raises(AssertionError,match="incomplete"):
        analyze(b,base/"partial_rejected",require_complete=True,manifest_required=False)
    runner.run(b,resume=True)
    (base/"bounded_compute.json").write_text(json.dumps(dict(
        main_train_steps=0,diagnostic_train_steps=384,diagnostic_eval_steps=144,
        other_control_steps=1414,total_env_steps=1942)))
    return base,a,b


def test_main_schedule_and_prefix_match():
    p=runner.build_protocol(dict(sha256="test"))
    assert p["total_steps"]==60000 and p["planned_validation_steps"]==72000
    assert p["arms"]==["C-R"] and p["training_seeds"]==[551]
    assert len(p["game_schedule"])==120
    assert [s["horizon"] for s in p["game_schedule"]]==[150]*80+[1200]*40
    assert {s["scenario_seed"] for s in p["game_schedule"][80:]}==set(range(120001,120041))
    old=protocol_new(read_json(runner.ROOT/"configs/bs5_curriculum_fixed_v1.json"),dict(sha256="test"))
    for i,spec in enumerate(p["game_schedule"][:80]):
        for key,value in old["scenario_schedule"]["551"][i].items():
            assert spec[key]==value
        assert spec["condition"]==condition_for(i*150,"C-R",old)


def test_horizon_prefix_and_full_continuity():
    p=runner.build_protocol(dict(sha256="test"))
    c=condition_for(12000,"C-R",p)
    short=runner.new_env(p,c,150,30)
    long=runner.new_env(p,c,1200,30)
    a=short.reset(seed=140001,assignment_seed=140001,diagnostic=True)
    b=long.reset(seed=140001,assignment_seed=140001,diagnostic=True)
    control_rng=np.random.default_rng(71200)
    for t in range(150):
        # Fixed action control, not a learned policy rollout.
        actions=[int(control_rng.choice(np.flatnonzero(m))) for m in a["action_mask"]]
        a=short.step(actions); b=long.step(actions)
        for key in ("obs","state","action_mask"):
            np.testing.assert_array_equal(a[key],b[key])
        assert a["reward"]==b["reward"] and not b["truncated"]
        assert bool(a["truncated"])==(t==149)
    # Continue SAME long env, preserving pin/mobility/lifecycle through all chunk markers.
    prior_battery=long.battery.copy()
    for t in range(150,1200):
        actions=[int(control_rng.choice(np.flatnonzero(m))) for m in b["action_mask"]]
        b=long.step(actions)
        assert long.t==t+1 and bool(b["truncated"])==(t==1199)
        assert not b["terminated"]
    assert short.t==150 and long.t==1200


def test_bootstrap_rule_and_numeric_td_target():
    assert not runner.learning_terminal("S4",False,True)
    assert runner.learning_terminal("S4",True,False)
    assert runner.learning_terminal("S3",False,True)
    assert not runner.learning_terminal("S4",False,False)
    p=runner.build_protocol(dict(sha256="test"),diagnostic=True)
    env=runner.new_env(p,condition_for(0,"C-R",p),6,30)
    out=env.reset(seed=140001)
    learner=runner.BS5CurriculumQMIX(env,lr=p["lr"],gamma=p["gamma"],device="cpu")
    class Constant(torch.nn.Module):
        def forward(self,q,s): return torch.full((len(q),),2.0,device=q.device)
    learner.target_mixer=Constant()
    obs=np.stack([out["obs"]]*2).astype(np.float32)
    state=np.stack([out["state"]]*2).astype(np.float32)
    acts=np.array([[int(np.flatnonzero(m)[0]) for m in out["action_mask"]]]*2)
    batch=dict(obs=obs,state=state,actions=acts,reward=np.array([.2,.2],np.float32),
        next_obs=obs,next_state=state,next_action_mask=np.stack([out["action_mask"]]*2).astype(bool),
        done=np.array([0.,1.],np.float32))
    with torch.no_grad():
        q=learner.agent(torch.tensor(obs).reshape(10,-1)).reshape(2,5,10)
        chosen=q.gather(2,torch.tensor(acts).unsqueeze(-1)).squeeze(-1)
        total=learner.mixer(chosen,torch.tensor(state))
        expected=torch.nn.functional.smooth_l1_loss(total,torch.tensor([.2+2*p["gamma"],.2]))
    actual=learner.train_batch(batch)
    assert actual["loss"]==pytest.approx(float(expected),abs=1e-6)


def test_exact_resume_models_replay_actions_batches(diagnostic_runs):
    base,a,b=diagnostic_runs
    aa=torch.load(a/"resume.pt",weights_only=False,map_location="cpu")
    bb=torch.load(b/"resume.pt",weights_only=False,map_location="cpu")
    for key in aa["models"]:
        for name in aa["models"][key]:
            assert torch.equal(aa["models"][key][name],bb["models"][key][name])
    assert aa["replay"]["pos"]==bb["replay"]["pos"]
    assert aa["replay"]["rng"]==bb["replay"]["rng"]
    for key,value in aa["replay"]["arrays"].items():
        np.testing.assert_array_equal(value,bb["replay"]["arrays"][key])
    ca,cb=aa["commit"],bb["commit"]
    for key in ("steps","updates","episodes","best_step","best_utility","events"):
        assert ca[key]==cb[key]
    for rel in ca["games"]:
        ia,ra,ua=load_game(a,rel); ib,rb,ub=load_game(b,rel)
        assert ra==rb, "Actions/current-next inputs/UE/status must match exactly"
        assert ua==ub, "Sampled indices/batch hashes and numerical updates must match exactly"
    assert ca["events"][-1]["step"]==96


def test_eval_preserves_rng_and_weights(diagnostic_runs):
    base,a,b=diagnostic_runs
    p=read_json(a/"protocol.json")
    root=base/"eval_rng_control"
    root.mkdir()
    # Exactly24 extra control eval steps; no training.
    cap=dict(phases={"eval":dict(cap=24,charged=0,observed=0)})
    runner.atomic_json(root/"budget_ledger.json",cap)
    learner=runner.BS5CurriculumQMIX(runner.new_env(p,condition_for(0,"C-R",p),6,30),device="cpu")
    before=runner.rng_state(); weights=runner.weights_hash(learner)
    commit=dict(steps=0,evaluations=[],best_utility=-1,best_step=0,timers={k:0. for k in
        ("env_s","actor_s","learner_s","record_s","raw_io_s","eval_inclusive_s")})
    runner.evaluate(root,learner,p,runner.Budget(root),commit)
    after=runner.rng_state()
    assert before["python"]==after["python"]
    assert torch.equal(before["torch"],after["torch"])
    np.testing.assert_array_equal(before["numpy"][1],after["numpy"][1])
    assert runner.weights_hash(learner)==weights


def test_offline_raw_analyzer_and_fail_closed(diagnostic_runs):
    base,a,b=diagnostic_runs
    result=analyze(a,base/"analyzed",require_complete=True,manifest_required=False)
    assert result["training_steps"]==192 and result["validation_steps"]==72
    assert result["collision_version"]=="bs5_endpoint_collision_v2"
    assert result["nonstation_collision_slots"]==0
    c=read_json(a/"commit.json")
    index=read_json(a/c["games"][-1])
    from scripts.analysis.analyze_bs5_cr_h1200_collision import validate_collision
    _,rows,_=load_game(a,c["games"][-1])
    bad=deepcopy(rows[0])
    bad["service"]["uav_positions"][0]=bad["service"]["uav_positions"][1][:]
    bad["collision"]["resolved_positions"]=deepcopy(bad["service"]["uav_positions"])
    with pytest.raises(AssertionError,match="endpoint collision"):
        validate_collision(bad,index["initial"]["charging_station_cells"])
    f=a/index["chunks"][0]["path"]
    original=f.read_bytes()
    try:
        f.write_bytes(original+b"bad")
        with pytest.raises(AssertionError,match="chunk hash"):
            analyze(a,base/"tamper_rejected",require_complete=True,manifest_required=False)
    finally: f.write_bytes(original)
    temporarily=f.with_suffix(".missing")
    f.rename(temporarily)
    try:
        with pytest.raises(FileNotFoundError):
            analyze(a,base/"missing_rejected",require_complete=True,manifest_required=False)
    finally: temporarily.rename(f)


def test_export_merge_and_checkpoint_free_audit(diagnostic_runs):
    base,a,b=diagnostic_runs
    out=runner.export(a)
    assert read_json(out/"verified_analysis/summary.json")["status"]=="complete"
    # Execute standalone stdlib-only analyzer in a fresh process with imports blocked.
    import subprocess,sys,zipfile
    merged=base/"audit_only"
    merged.mkdir()
    for f in out.glob("audit_light*.zip"):
        with zipfile.ZipFile(f) as z: z.extractall(merged)
    code="""import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','numpy','src'}: raise ImportError('blocked '+fullname)
sys.meta_path.insert(0,Block())
sys.path.insert(0,sys.argv[1])
sys.argv=[sys.argv[1]+'/analyze_bs5_cr_h1200_collision.py','--root',sys.argv[1],'--output',sys.argv[2],'--require-complete']
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    process=subprocess.run([sys.executable,"-B","-c",code,str(merged),str(base/"audit_only_analysis")],
        text=True,capture_output=True)
    assert process.returncode==0,process.stdout+process.stderr


def test_lifecycle_multicycle_slot_convention_and_relay_useful():
    from scripts.analysis.bs5_continuous_common import lifecycle
    rows=[]
    sequence=[(0,1),(1,2),(2,0),(0,0),(0,1),(1,2),(2,0),(0,0)]
    for slot,(before,after) in enumerate(sequence,1):
        rate=1.0 if slot==4 else 2.0 if slot==5 else 0.0
        rows.append(dict(slot=slot,decision=dict(status=[before,0,0,0,0]),
            post=dict(status=[after,0,0,0,0],positions=[[0,0]]*5),
            service=dict(status=[before,0,0,0,0],roles=[1,0,0,0,0],uav_positions=[[0,0]]*5,
                network=dict(routes={"1":0},association={"1":[0]},ue_rate_mbps=[rate]))))
    initial=dict(battery=[100]*5,residual_uav=4)
    cycles,_=lifecycle(rows,initial,run="fixture",scenario=0)
    assert len(cycles)==2
    assert cycles[0]["service_latency_slots"]==1
    assert cycles[0]["traffic_after_rejoin_mbit"]==15
    assert cycles[1]["useful_service_censored"] and cycles[1]["censor_reason"]=="horizon"


def test_rejects_legacy_resume_before_loading_checkpoint(tmp_path):
    runner.atomic_json(tmp_path/'protocol.json',dict(collision_version='legacy_v1'))
    with pytest.raises(AssertionError,match='cannot resume'):
        runner.run(tmp_path,resume=True)


def test_no_conflict_prefix_matches_legacy_physics():
    from src.envs.bs5_curriculum_env import BS5CurriculumEnv
    p=runner.build_protocol(dict(sha256='test'))
    c=condition_for(12000,'C-R',p)
    new=runner.new_env(p,c,1200,30)
    old=BS5CurriculumEnv(c,allow_relay=True,horizon=1200,residual_battery=30,fixture=p['fixture'])
    a=new.reset(seed=140001,assignment_seed=140001,diagnostic=True)
    b=old.reset(seed=140001,assignment_seed=140001,diagnostic=True)
    for _ in range(20):
        a=new.step([8]*5);b=old.step([8]*5)  # Stay-Access; unique initial endpoints
        for k in ('obs','state','action_mask'):
            np.testing.assert_array_equal(a[k],b[k])
        assert a['reward']==b['reward']
        assert new.last_collision_resolution['blocked_ids']==[]
        assert runner.serial(new.last_service_snapshot)==runner.serial(old.last_service_snapshot)
