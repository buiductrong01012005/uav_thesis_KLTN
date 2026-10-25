"""Shared, immutable exogenous UE tapes; Return-v3 physics remain unchanged."""
from functools import lru_cache
from pathlib import Path
from copy import deepcopy
import hashlib,io,json,sys
import numpy as np
from scripts.analysis.bs5_bundle import atomic_json,canonical_hash,file_hash,read_json
from src.envs.bs5_energy_return_env import BS5EnergyReturnEnv

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'assets/bs5_return_v3_cr'
TRACE_VERSION='bs5_exogenous_ue_tapes_v1'

@lru_cache(maxsize=1)
def verify_assets():
    manifest=read_json(ASSETS/'ASSET_SHA256SUMS.json')
    for rel,digest in manifest.items():assert file_hash(ASSETS/rel)==digest,('asset checksum',rel)
    identity=read_json(ASSETS/'source_identity.json')
    for rel,digest in identity['files'].items():assert file_hash(ROOT/rel)==digest,('historical scientific source changed',rel)
    selected=read_json(ASSETS/'checkpoint_selection.json');best=next(e for e in selected['curve'] if e['step']==selected['best_step'])
    assert best['step']==60000 and file_hash(ASSETS/'060000.pt')==best['checkpoint_sha256']
    return dict(manifest_sha256=file_hash(ASSETS/'ASSET_SHA256SUMS.json'),historical_source_sha256=identity['sha256'],
        checkpoint_sha256=best['checkpoint_sha256'],weights_sha256=best['weights_sha256'],initial_weights_sha256=selected['curve'][0]['weights_sha256'])

@lru_cache(maxsize=1)
def historical_index():return read_json(ASSETS/'TRACE_INDEX.json')

def trace_spec(condition,horizon,seed,assignment,residual,diagnostic):
    return dict(condition=deepcopy(condition),horizon=int(horizon),scenario_seed=int(seed),assignment_seed=int(assignment),
        residual_battery=float(residual),diagnostic=bool(diagnostic))
def key(spec):return canonical_hash(spec)
def position_hash(x):return hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()
def write_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);b=io.BytesIO();np.savez_compressed(b,**arrays)
    temp=path.with_suffix('.npz.tmp');temp.write_bytes(b.getvalue());temp.replace(path)
def initial_snapshot(env):
    return dict(battery=env.battery.tolist(),residual_uav=int(env.residual_uav_idx),positions=env.uav_positions.tolist(),
        ue_positions=env.ue_positions.tolist(),charging_station_cells=env.charging_stations.tolist())

def fixture_signal_note():
    """Static direct-only reachability at starting fixtures, not a performance oracle."""
    from src.envs.bs5_launch_ab_env import BS5LaunchConfig
    cfg=BS5LaunchConfig();bs=np.delete(np.asarray(cfg.bs_xy_m),cfg.failed_bs_idx,axis=0);groups={}
    for item in historical_index().values():
        spec=item['spec'];c=spec['condition']
        if c['stage'] not in ['S1','S2']:continue
        setup=np.random.default_rng(np.random.SeedSequence([spec['assignment_seed'],0x43555252]));ids=setup.permutation(5)[:2]
        pos=np.asarray(item['initial']['positions'])[ids];ue=np.asarray(item['initial']['ue_positions'])
        xy=(pos[:,::-1]+.5)*cfg.cell_size_m
        gateway=np.linalg.norm(xy[:,None]-bs[None],axis=2).min(1)<=cfg.backhaul_radius_cells*cfg.cell_size_m+1e-9
        covered=((np.linalg.norm(ue[:,None]-pos[None],axis=2)<=cfg.access_radius_cells+1e-9)&gateway[None]).any(1)
        groups.setdefault(c['substage'],[]).append(int(covered.sum()))
    return dict(scope='matched-recipe ablation; S1/S2 fixtures teach Relay coordination, not an optimized C-N curriculum',
        diagnostic='starting-position direct one-hop coverage upper bound, ignores rates/capacity; S2 can subsequently move',
        stages={k:dict(games=len(v),zero_direct_coverage_at_start=sum(n==0 for n in v),mean_reachable_ues=float(np.mean(v))) for k,v in groups.items()},
        schedule_changed=False)

class TraceBank:
    def __init__(self,home,fixture):self.home=Path(home) if home is not None else None;self.fixture=fixture
    def get(self,spec):
        name=key(spec);original=historical_index().get(name)
        if original:
            path=ASSETS/original['path']
            with np.load(path,allow_pickle=False) as z:positions=z['positions']
            assert position_hash(positions)==original['position_sha256']
            return positions,dict(version=TRACE_VERSION,key=name,position_sha256=original['position_sha256'],
                origin=original['origin'],initial=original['initial'],spec=spec)
        assert self.home is not None,'A writable shared trace directory must be configured'
        self.home.mkdir(parents=True,exist_ok=True);path=self.home/(name+'.npz');meta=self.home/(name+'.json')
        if meta.exists():
            d=read_json(meta);assert d['spec']==spec and file_hash(path)==d['file_sha256']
            with np.load(path,allow_pickle=False) as z:positions=z['positions']
            assert position_hash(positions)==d['position_sha256'];return positions,d
        assert not path.exists(),'Uncommitted UE tape exists; preserve rather than silently regenerate'
        env=BS5EnergyReturnEnv(spec['condition'],allow_relay=True,horizon=spec['horizon'],residual_battery=spec['residual_battery'],fixture=self.fixture)
        env.reset(seed=spec['scenario_seed'],assignment_seed=spec['assignment_seed'],diagnostic=spec['diagnostic'])
        initial=initial_snapshot(env);positions=[env.ue_positions.copy()]
        # UE-only advancement: no action, networking, energy, training or env.step.
        for _ in range(spec['horizon']):
            env._advance_mobile_ues();env.t+=1;positions.append(env.ue_positions.copy())
        positions=np.asarray(positions,dtype=np.float64);origin='source_UE_generator';prefix_error=None
        scenario=spec['scenario_seed']
        if spec['horizon']>=1200 and scenario in range(130001,130011) and spec['condition']['stage']=='S4' and spec['diagnostic']:
            with np.load(ASSETS/f'prefix_{scenario}.npz') as z:
                prefix=np.concatenate((z['ue_positions'],z['post_ue_last'][None]),axis=0)
            prefix_error=float(np.max(np.abs(positions[:1201]-prefix)))
            assert prefix_error<=1e-8,('Historical UE-generator parity failed',scenario,prefix_error)
            # Exact historical prefix; only machine-precision generator drift is accepted.
            positions[:1201]=prefix;initial['ue_positions']=prefix[0].tolist();origin='source_generator_with_exact_historical_1200_prefix'
        d=dict(version=TRACE_VERSION,key=name,spec=spec,origin=origin,initial=initial,
            position_sha256=position_hash(positions),ue_only_ticks=spec['horizon'],environment_steps=0,
            historical_prefix_generator_max_error=prefix_error,python=sys.version,numpy=np.__version__)
        write_npz(path,positions=positions);d['file_sha256']=file_hash(path);atomic_json(meta,d)
        return positions,d

class TraceReplayReturnEnv(BS5EnergyReturnEnv):
    """Only exogenous UE coordinates are replayed; UAVs/batteries remain endogenous."""
    def __init__(self,*args,trace_home=None,**kwargs):
        self._ue_tape=None;self._trace_home=trace_home;self.trace_receipt=None
        super().__init__(*args,**kwargs)
    def reset(self,seed=None,*,options=None,assignment_seed=0,diagnostic=False):
        self._ue_tape=None
        out=super().reset(seed=seed,options=options,assignment_seed=assignment_seed,diagnostic=diagnostic)
        spec=trace_spec(self.condition,self.cfg.horizon,seed,assignment_seed,self.cfg.residual_initial_battery,diagnostic)
        tape,receipt=TraceBank(self._trace_home,self.fixture).get(spec)
        actual=initial_snapshot(self);expected=receipt['initial']
        for k in ['battery','residual_uav','positions','charging_station_cells']:assert actual[k]==expected[k],('initial mismatch',k)
        assert np.allclose(self.ue_positions,tape[0],atol=1e-8,rtol=0),'Initial UE-generator mismatch'
        self._ue_tape=tape;self.trace_receipt=receipt;self.ue_positions=tape[0].copy()
        self._refresh_request_state();out.update(obs=self.get_obs(),state=self.get_state(),action_mask=self.get_action_mask())
        return out
    def _advance_mobile_ues(self):
        # Preserve source internal RNG consumption and telemetry; only coordinates use the locked tape.
        super()._advance_mobile_ues()
        if self._ue_tape is not None:self.ue_positions=self._ue_tape[self.t+1].copy()
    def step(self,actions):
        assert self._ue_tape is not None and self.t<self.cfg.horizon
        before=self.t;out=super().step(actions)
        assert self.t==before+1 and np.array_equal(self.last_service_snapshot['ue_positions'],self._ue_tape[before])
        assert np.array_equal(self.ue_positions,self._ue_tape[self.t])
        if not self.cfg.allow_relay:
            assert all(a%2==0 for a in actions)
            assert not out['action_mask'][:,1::2].any()
            assert all(v is None for v in self.last_service_snapshot['network']['routes'].values())
        return out
