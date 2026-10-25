"""Bounded acceptance tests before spending the main C-N training budget."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.experiments import run_bs5_crcn_return_v3 as pipeline
from src.envs.v4_fixed_mobile_battery_env import V4FixedMobileBatteryEnv

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    pipeline.cn.setup_runtime('cpu',1);p=pipeline.read_json(pipeline.SPEC)
    fingerprint=dict(source=pipeline.source_identity()['sha256'],assets=pipeline.verify_assets()['manifest_sha256'],runtime=pipeline.cn.runtime('cpu'))
    receipt=root/'receipt.json'
    if receipt.exists():
        old=pipeline.read_json(receipt)
        assert old['fingerprint']==fingerprint and old['status']=='passed','Existing preflight is incomplete/failed or differs in source/runtime; preserve its evidence'
        print(json.dumps(old));return
    ledger=dict(status='started',fingerprint=fingerprint,caps=p['preflight_caps'],charged={'diagnostic_train':0,'control_eval':0},main_training_steps=0,
        note='Reserve before each real env.step. Includes192+192 tiny C-N train, two C-R1200 prefixes (130001/130009) and controller/paired controls. No C-R/main C-N training.')
    pipeline.atomic_json(receipt,ledger);original=V4FixedMobileBatteryEnv.step
    def counted(env,actions):
        phase='diagnostic_train' if getattr(env,'_audit_phase','control')=='train' else 'control_eval'
        assert ledger['charged'][phase]<ledger['caps'][phase],'Preflight cap reached'
        ledger['charged'][phase]+=1;pipeline.atomic_json(receipt,ledger);return original(env,actions)
    V4FixedMobileBatteryEnv.step=counted
    try:
        import pytest
        rc=pytest.main(['-q','-p','no:cacheprovider',str(ROOT/'tests/test_bs5_crcn_return_v3.py'),str(ROOT/'tests/test_bs5_energy_return.py'),str(ROOT/'tests/test_bs5_collision_contract.py'),'--basetemp',str(root/'test_data')])
        if rc:raise RuntimeError('Preflight failed; main not started')
        ledger['status']='passed'
    except BaseException as exc:ledger.update(status='failed',error=repr(exc));raise
    finally:V4FixedMobileBatteryEnv.step=original;pipeline.atomic_json(receipt,ledger)
    print(json.dumps(ledger))
if __name__=='__main__':main()
