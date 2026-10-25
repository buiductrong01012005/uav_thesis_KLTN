"""Bounded no-training acceptance gate for the frozen Return-v4 evaluation."""
from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.experiments import run_bs5_return_v4_pair as run
from src.envs.v4_fixed_mobile_battery_env import V4FixedMobileBatteryEnv

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 run.setup();run.verify_assets();fingerprint=dict(source=run.source_identity()['sha256'],assets=run.file_hash(run.ASSETS/'SHA256SUMS.json'),runtime=run.runtime());receipt=out/'receipt.json'
 if receipt.exists():
  old=run.read_json(receipt);assert old['fingerprint']==fingerprint and old['status']=='passed','Preserve failed/changed preflight evidence; do not silently repeat compute';print(json.dumps(old));return
 data=dict(status='started',fingerprint=fingerprint,cap=4096,charged_control_steps=0,training_steps=0,scope='Route regression controls, legacy energy/collision, exact historical1200 prefixes and tiny24-slot export/resume test')
 run.atomic_json(receipt,data);original=V4FixedMobileBatteryEnv.step
 def counted(env,actions):
  assert data['charged_control_steps']<data['cap'];data['charged_control_steps']+=1;run.atomic_json(receipt,data);return original(env,actions)
 V4FixedMobileBatteryEnv.step=counted
 try:
  import pytest
  rc=pytest.main(['-q','-p','no:cacheprovider',str(ROOT/'tests/test_bs5_return_commit.py'),str(ROOT/'tests/test_bs5_return_v4_pair.py'),str(ROOT/'tests/test_bs5_energy_return.py'),str(ROOT/'tests/test_bs5_collision_contract.py'),'--basetemp',str(out/'test_data')])
  if rc:raise RuntimeError('Preflight failed; main evaluation not started')
  data['status']='passed'
 except BaseException as exc:data.update(status='failed',error=repr(exc));raise
 finally:V4FixedMobileBatteryEnv.step=original;run.atomic_json(receipt,data)
 print(json.dumps(data))
if __name__=='__main__':main()
