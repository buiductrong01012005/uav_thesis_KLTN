"""Exercise real pure runner functions without pretending to execute PyTorch."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
from copy import deepcopy
import argparse,ast,csv,hashlib,json,sys
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.envs.bs5_fourarm400k import BS5FourArm400kEnv,EnergyConfig,CollisionConfig,FairConfig,OBS_DIM,STATE_DIM,VERSION,REWARD_VERSION
from src.envs.bs5_service_v2 import FULL_CONDITION,ServiceV2Config
from src.envs.bs5_tcf_env import TCFWeights,jain,rate_metrics
from src.diagnostics.bs5_episode_service_metrics import episode_service_metrics
from src.diagnostics.bs5_fourarm400k_metrics import observe_step,validation_metrics

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n')
def load_api():
    file=ROOT/'scripts/training/run_bs5_fourarm400k.py';tree=ast.parse(file.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'sha','source_lock','load_protocol','parser','make_env','evaluate','write_csv'}]
    fake_torch=SimpleNamespace(__version__='fixture-no-torch',version=SimpleNamespace(cuda=None),backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda:None)),__config__=SimpleNamespace(show=lambda:'fixture-no-torch'))
    ns=dict(Path=Path,__file__=str(file),ROOT=ROOT,CONFIG=ROOT/'configs/bs5_fourarm400k.json',asdict=asdict,argparse=argparse,np=np,scipy=scipy,torch=fake_torch,hashlib=hashlib,json=json,sys=sys,csv=csv,deepcopy=deepcopy,TCFWeights=TCFWeights,rate_metrics=rate_metrics,EnergyConfig=EnergyConfig,CollisionConfig=CollisionConfig,FairConfig=FairConfig,OBS_DIM=OBS_DIM,STATE_DIM=STATE_DIM,ENV_VERSION=VERSION,VERSION=REWARD_VERSION,digest=digest,BS5HybridTrainEnv=BS5FourArm400kEnv,FULL_CONDITION=FULL_CONDITION,ServiceV2Config=ServiceV2Config,episode_service_metrics=episode_service_metrics,jain=jain,atomic_json=atomic_json,observe_step=observe_step,validation_metrics=validation_metrics)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(file),'exec'),ns)
    return ns
class FixturePolicy:
    def act(self,obs,mask,epsilon,rng):return [int(rng.choice(np.flatnonzero(m))) for m in mask]
