"""Tests use lightweight fixtures; they do not create evaluation results."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]


def load(modname, path):
    sp=importlib.util.spec_from_file_location(modname,str(path))
    mod=importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def test_checkpoint_suffix_and_cli_are_explicit():
    src=(ROOT/'scripts'/'run_role_recovery_5k_and_curriculum.py').read_text()
    assert "best_stage3.pt" in src
    assert "curriculum_rl_seed" in src
    assert "--force" in src
    assert "run_one(ns,label,'rl',checkpoint)" in src


def test_completed_5k_check_rejects_other_checkpoint(tmp_path):
    mod=load('run_5k',ROOT/'scripts'/'run_role_recovery_5k_and_curriculum.py')
    csv=tmp_path/'trace.csv'; csv.write_text('step,utility\n1,0.4\n')
    summ=tmp_path/'sum.json'
    ck=tmp_path/'seed91_best_stage3.pt'; ck.write_bytes(b'not a real model')
    summ.write_text(json.dumps({'steps':5000,'scenario_seed':3101,'checkpoint':str(ck)}))
    assert mod.complete(summ,csv,steps=5000,scenario_seed=3101,checkpoint=ck)
    assert not mod.complete(summ,csv,steps=5001,scenario_seed=3101,checkpoint=ck)
    assert not mod.complete(summ,csv,steps=5000,scenario_seed=3102,checkpoint=ck)
    assert not mod.complete(summ,csv,steps=5000,scenario_seed=3101,checkpoint=tmp_path/'other.pt')


def test_plot_csv_requires_same_scenario_and_consecutive_steps(tmp_path):
    mod=load('plot_5k',ROOT/'scripts'/'plot_role_recovery_5k_and_curriculum.py')
    p=tmp_path/'trace.csv'; s=tmp_path/'summary.json'
    df=pd.DataFrame({'step':[1,2,3], 'utility':[.5,.6,.8], 'served_ues':[10,11,12],
                     'relay_served_ues':[2,3,4], 'relay':[1,1,2], 'decision_ms':[1,1,1]})
    df.to_csv(p,index=False); s.write_text(json.dumps({'scenario_seed':3101,'steps':3}))
    assert len(mod.check_csv(p,s,3,3101))==3
    with pytest.raises(ValueError):mod.check_csv(p,s,3,3102)
    df.loc[1,'step']=5; df.to_csv(p,index=False)
    with pytest.raises(ValueError):mod.check_csv(p,s,3,3101)
