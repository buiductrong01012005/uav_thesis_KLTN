"""Frozen checkpoints and exact original UE tapes for controller-only evaluation."""
from pathlib import Path
from functools import lru_cache
import hashlib,json
import numpy as np
from src.envs.bs5_energy_return_env import BS5EnergyReturnEnv
from src.envs.bs5_return_commit_env import BS5CommittedReturnEnv

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'assets/bs5_return_v4_pair'

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

@lru_cache(maxsize=1)
def verify_assets():
    m=json.loads((ASSETS/'SHA256SUMS.json').read_text())
    for rel,h in m.items():assert file_hash(ASSETS/rel)==h,rel
    lock=json.loads((ASSETS/'LOCK.json').read_text())
    assert len(lock['jobs'])==40 and len(lock['traces'])==20
    assert {k:v['step'] for k,v in lock['checkpoints'].items()}=={'C-R':60000,'C-N':36000}
    for v in lock['checkpoints'].values():assert file_hash(ASSETS/v['path'])==v['sha256']
    return lock


class LockedTapeMixin:
    def __init__(self,*args,**kwargs):
        self._ue_tape=None
        super().__init__(*args,**kwargs)

    def reset(self,seed=None,*,options=None,assignment_seed=0,diagnostic=False):
        self._ue_tape=None
        out=super().reset(seed=seed,options=options,assignment_seed=assignment_seed,diagnostic=diagnostic)
        entry=verify_assets()['traces'][str(seed)]
        with np.load(ASSETS/entry['path'],allow_pickle=False) as z:tape=z['positions']
        assert hashlib.sha256(np.asarray(tape,dtype='<f8').tobytes()).hexdigest()==entry['position_sha256']
        expected=entry['initial']
        for name,value in [('battery',self.battery.tolist()),('positions',self.uav_positions.tolist()),
            ('residual_uav',int(self.residual_uav_idx)),('charging_station_cells',self.charging_stations.tolist())]:assert expected[name]==value,name
        assert np.allclose(self.ue_positions,tape[0],atol=1e-8,rtol=0)
        self._ue_tape=tape;self.trace_entry=entry
        self.ue_positions=tape[0].copy();self._refresh_request_state()
        out.update(obs=self.get_obs(),state=self.get_state(),action_mask=self.get_action_mask())
        return out

    def _advance_mobile_ues(self):
        super()._advance_mobile_ues()
        if self._ue_tape is not None:self.ue_positions=self._ue_tape[self.t+1].copy()

    def step(self,actions):
        t=self.t;out=super().step(actions)
        assert self._ue_tape is not None and np.array_equal(self.last_service_snapshot['ue_positions'],self._ue_tape[t])
        assert np.array_equal(self.ue_positions,self._ue_tape[t+1])
        if not self.cfg.allow_relay:
            assert all(a%2==0 for a in actions) and not out['action_mask'][:,1::2].any()
            assert all(x is None for x in self.last_service_snapshot['network']['routes'].values())
        return out


class FrozenTapeReturnV4(LockedTapeMixin,BS5CommittedReturnEnv):pass
class FrozenTapeReturnV3(LockedTapeMixin,BS5EnergyReturnEnv):pass

def new_env(arm,horizon=10000,version='v4'):
    lock=verify_assets();cls=FrozenTapeReturnV4 if version=='v4' else FrozenTapeReturnV3
    return cls(lock['condition'],allow_relay=arm=='C-R',horizon=horizon,
               residual_battery=30.,fixture=lock['fixture'])
