from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_final_fixed_map_env import ACTIVE, V4FinalFixedMapConfig, V4FinalFixedMapEnv
from src.envs.v4_a1_identity_randomization import reset_identity_randomized
from src.rl.v4_hardradius_spatial_qmix import V4HardRadiusSpatialQMIXLearner

A0_ROLES = np.asarray([ACCESS, ACCESS, ACCESS, ACCESS, RELAY], dtype=np.int64)


def learner_for(env, ckpt: Path):
    learner = V4HardRadiusSpatialQMIXLearner(
        env.cfg.num_uavs, env.obs_dim, env.state_dim, env.n_actions,
        grid_size=env.cfg.grid_size, coarse_size=env.cfg.coarse_obs_size,
        lr=3e-4, gamma=0.99, hidden_dim=64, mixing_embed_dim=32, device='cpu')
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    learner.agent.load_state_dict(payload['agent'])
    learner.mixer.load_state_dict(payload['mixer'])
    learner.update_targets()
    return learner


def fixed_role_mask(env, roles):
    base = env.get_action_mask()
    out = np.zeros_like(base)
    for i in range(env.cfg.num_uavs):
        for move in range(5):
            a = env.encode_action(move, int(roles[i]))
            out[i, a] = base[i, a]
    return out


def a0_episode(learner, layout_seed: int, frozen=False):
    env = V4FinalFixedMapEnv(V4FinalFixedMapConfig(
        seed=layout_seed, coarse_obs_size=5,
        enable_ue_mobility=False, enable_battery_cycle=False))
    out = env.reset(seed=layout_seed)
    env.roles[:] = A0_ROLES
    out['obs'] = env.get_obs(); out['state'] = env.get_state(); out['action_mask'] = fixed_role_mask(env, A0_ROLES)
    rng = np.random.default_rng(900000 + layout_seed)
    us=[]
    while True:
        if frozen:
            actions=[env.encode_action(4, int(A0_ROLES[i])) for i in range(env.cfg.num_uavs)]
        else:
            actions=learner.act(out['obs'], out['action_mask'], 0.0, rng)
        out=env.step(actions)
        env.roles[:] = A0_ROLES
        out['obs']=env.get_obs(); out['state']=env.get_state(); out['action_mask']=fixed_role_mask(env,A0_ROLES)
        us.append(float(out['reward']))
        if out['truncated']: break
    return {'mean_utility':float(np.mean(us)), 'final20':float(np.mean(us[-20:])), 'max':float(np.max(us))}


def a1_episode(learner, residual_idx: int, perm_seed: int, battery: float):
    env = V4FinalFixedMapEnv(V4FinalFixedMapConfig(
        seed=1101, coarse_obs_size=5, backhaul_radius_cells=4.0,
        enable_ue_mobility=False, enable_battery_cycle=True,
        residual_initial_battery=30.0, randomize_residual_uav=False))
    out = reset_identity_randomized(env, residual_idx, perm_seed, residual_battery=battery)
    rng=np.random.default_rng(910000 + perm_seed + residual_idx)
    us=[]; relay_counts=[]; handover_time=None; return_time=None; rejoin_time=None
    roles_first20=[]
    while True:
        actions=learner.act(out['obs'], out['action_mask'], 0.0, rng)
        nxt=env.step(actions)
        info=nxt['info']; t=int(info['t']); us.append(float(nxt['reward'])); relay_counts.append(int(info['num_relay']))
        roles=np.asarray(info['roles'],dtype=np.int64); status=np.asarray(info['status'],dtype=np.int64)
        if t <= 20: roles_first20.append([int(x) for x in roles])
        others=[i for i in range(env.cfg.num_uavs) if i != residual_idx and int(status[i])==ACTIVE and int(roles[i])==RELAY]
        if handover_time is None and int(status[residual_idx])==ACTIVE and int(roles[residual_idx])==ACCESS and others:
            handover_time=t
        if return_time is None and int(status[residual_idx]) != ACTIVE:
            return_time=t
        if return_time is not None and rejoin_time is None and int(status[residual_idx])==ACTIVE:
            rejoin_time=t
        out=nxt
        if out['truncated']: break
    return {
        'mean_utility':float(np.mean(us)), 'final20':float(np.mean(us[-20:])),
        'mean_active_relays':float(np.mean(relay_counts)),
        'handover_time':handover_time, 'early_handover_le20':handover_time is not None and handover_time <= 20,
        'return_time':return_time, 'rejoin_time':rejoin_time,
        'roles_first20': roles_first20,
    }


def stats(xs):
    a=np.asarray(xs,dtype=float)
    return {'mean':float(a.mean()), 'std':float(a.std()), 'min':float(a.min()), 'max':float(a.max()), 'n':int(len(a))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--a0-checkpoint', required=True)
    ap.add_argument('--a1-checkpoint', required=True)
    ap.add_argument('--output', default='results/final_evidence_seed81.json')
    args=ap.parse_args()
    a0_ckpt=Path(args.a0_checkpoint).resolve(); a1_ckpt=Path(args.a1_checkpoint).resolve()
    if not a0_ckpt.exists() or not a1_ckpt.exists(): raise FileNotFoundError('checkpoint missing')

    a0_env=V4FinalFixedMapEnv(V4FinalFixedMapConfig(seed=1101,coarse_obs_size=5,enable_ue_mobility=False,enable_battery_cycle=False))
    a0=learner_for(a0_env,a0_ckpt)
    heldout_seeds=list(range(1201,1221))
    a0_rows=[]
    for s in heldout_seeds:
        rl=a0_episode(a0,s,False); fr=a0_episode(a0,s,True)
        a0_rows.append({'layout_seed':s,'rl':rl,'frozen':fr,'gain_vs_frozen':rl['mean_utility']-fr['mean_utility']})

    a1_env=V4FinalFixedMapEnv(V4FinalFixedMapConfig(seed=1101,coarse_obs_size=5,backhaul_radius_cells=4.0,enable_ue_mobility=False,enable_battery_cycle=True,randomize_residual_uav=False))
    a1=learner_for(a1_env,a1_ckpt)
    # 20 matched held-out mappings: four unseen permutation seeds per residual identity.
    a1_rows=[]
    for rid in range(5):
        for k in range(4):
            ps=80000 + rid*100 + k
            low=a1_episode(a1,rid,ps,30.0); full=a1_episode(a1,rid,ps,100.0)
            a1_rows.append({'residual_uav_idx':rid,'permutation_seed':ps,'low_battery':low,'full_battery':full})

    low_early=[r['low_battery']['early_handover_le20'] for r in a1_rows]
    full_early=[r['full_battery']['early_handover_le20'] for r in a1_rows]
    same_early=[a==b for a,b in zip(low_early,full_early)]
    both_times=[]
    for r in a1_rows:
        lt=r['low_battery']['handover_time']; ft=r['full_battery']['handover_time']
        if lt is not None and ft is not None: both_times.append(abs(lt-ft))

    result={
        'scope_note':'Final-evidence audit on fixed trained seed81 checkpoints. This is held-out environment/permutation evaluation, not multi-training-seed RL.',
        'a0_static_heldout':{
            'layout_seeds':heldout_seeds,
            'rl_mean_utility':stats([r['rl']['mean_utility'] for r in a0_rows]),
            'frozen_mean_utility':stats([r['frozen']['mean_utility'] for r in a0_rows]),
            'gain_vs_frozen':stats([r['gain_vs_frozen'] for r in a0_rows]),
            'rl_win_fraction_vs_frozen':float(np.mean([r['gain_vs_frozen']>0 for r in a0_rows])),
            'rows':a0_rows,
        },
        'a1_idrand_battery_matched_heldout':{
            'num_pairs':len(a1_rows),
            'low_battery_mean_utility':stats([r['low_battery']['mean_utility'] for r in a1_rows]),
            'full_battery_mean_utility':stats([r['full_battery']['mean_utility'] for r in a1_rows]),
            'low_early_handover_fraction':float(np.mean(low_early)),
            'full_early_handover_fraction':float(np.mean(full_early)),
            'same_early_handover_indicator_fraction':float(np.mean(same_early)),
            'mean_abs_handover_time_diff_when_both_exist':float(np.mean(both_times)) if both_times else None,
            'low_return_fraction':float(np.mean([r['low_battery']['return_time'] is not None for r in a1_rows])),
            'full_return_fraction':float(np.mean([r['full_battery']['return_time'] is not None for r in a1_rows])),
            'rows':a1_rows,
        }
    }
    outp=Path(args.output); outp.parent.mkdir(parents=True,exist_ok=True); outp.write_text(json.dumps(result,indent=2))
    print(json.dumps({
        'output':str(outp.resolve()),
        'a0_rl':result['a0_static_heldout']['rl_mean_utility'],
        'a0_frozen':result['a0_static_heldout']['frozen_mean_utility'],
        'a0_gain':result['a0_static_heldout']['gain_vs_frozen'],
        'a0_win_fraction':result['a0_static_heldout']['rl_win_fraction_vs_frozen'],
        'a1_low_early':result['a1_idrand_battery_matched_heldout']['low_early_handover_fraction'],
        'a1_full_early':result['a1_idrand_battery_matched_heldout']['full_early_handover_fraction'],
        'a1_same_indicator':result['a1_idrand_battery_matched_heldout']['same_early_handover_indicator_fraction'],
        'a1_low_return':result['a1_idrand_battery_matched_heldout']['low_return_fraction'],
        'a1_full_return':result['a1_idrand_battery_matched_heldout']['full_return_fraction'],
    },indent=2))

if __name__=='__main__': main()
