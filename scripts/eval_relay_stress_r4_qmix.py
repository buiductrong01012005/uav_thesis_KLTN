"""Evaluate trained R4 Spatial-QMIX checkpoint without retraining.

Reports original utility plus operational KPIs that matter for the relay-stress
hypothesis: served UEs, fully-satisfied UEs, UEs actually routed through Relay,
UEs that geometrically require Relay, and how often Access UAVs operate beyond
all direct-BS backhaul coverage.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.thesis_cartesian_r4_relay_stress_env import (
    CartesianR4RelayStressBatteryConfig,
    CartesianR4RelayStressBatteryEnv,
)
from src.envs.v0_env import ACCESS
from src.envs.v4_fixed_mobile_battery_env import ACTIVE
from src.rl.v5_relay_request_spatial_qmix import V5RelayRequestSpatialQMIXLearner

TEST_SEEDS = tuple(range(3101, 3121))
EPS = 1e-9


class AuditR4Env(CartesianR4RelayStressBatteryEnv):
    def __init__(self, cfg):
        self.last_network = None
        super().__init__(cfg)

    def compute_network(self):
        net = super().compute_network()
        self.last_network = {
            'uav_positions': self.uav_positions.copy(),
            'ue_positions': self.ue_positions.copy(),
            'roles': self.roles.copy(),
            'status': self.status.copy(),
            'ue_rate_mbps': np.asarray(net['ue_rate_mbps'], dtype=np.float64).copy(),
            'association': {int(k): [int(u) for u in v] for k, v in net['association'].items()},
            'routes': {int(k): (None if v is None else int(v)) for k, v in net['routes'].items()},
            'backhaul_caps_mbps': {int(k): float(v) for k, v in net['backhaul_caps_mbps'].items()},
            'utility': float(net['utility']),
        }
        return net


def make_env(seed: int, horizon: int):
    return AuditR4Env(CartesianR4RelayStressBatteryConfig(
        seed=int(seed), horizon=int(horizon), residual_initial_battery=30.0,
        randomize_residual_uav=True,
    ))


def make_learner(env):
    return V5RelayRequestSpatialQMIXLearner(
        env.cfg.num_uavs, env.obs_dim, env.state_dim, env.n_actions,
        grid_size=env.cfg.grid_size, coarse_size=env.cfg.coarse_obs_size,
        lr=3e-4, gamma=0.99, hidden_dim=64, mixing_embed_dim=32, device='cpu',
    )


def direct_bh_distance_m(env, grid_positions: np.ndarray) -> np.ndarray:
    vals = []
    for p in np.asarray(grid_positions, dtype=np.float64):
        xy = env._cell_to_xy(p)
        vals.append(float(np.min(np.linalg.norm(env.surviving_bs_xy - xy, axis=1))))
    return np.asarray(vals, dtype=np.float64)


def direct_access_cells(env) -> np.ndarray:
    cells = []
    threshold_m = env.cfg.backhaul_radius_cells * env.cfg.cell_size_m
    for r in range(env.cfg.grid_size):
        for c in range(env.cfg.grid_size):
            if direct_bh_distance_m(env, np.asarray([[r, c]], dtype=float))[0] <= threshold_m + EPS:
                cells.append((r, c))
    return np.asarray(cells, dtype=np.float64)


def relay_required_ue_mask(env, ue_positions: np.ndarray, direct_cells: np.ndarray) -> np.ndarray:
    ue = np.asarray(ue_positions, dtype=np.float64)
    d2 = ((ue[:, None, :] - direct_cells[None, :, :]) ** 2).sum(axis=2)
    direct_possible = np.min(d2, axis=1) <= env.cfg.access_radius_cells ** 2 + EPS
    return ~direct_possible


def evaluate_checkpoint(checkpoint: Path, test_seeds, horizon: int):
    env0 = make_env(test_seeds[0], horizon)
    learner = make_learner(env0)
    learner.load(str(checkpoint))
    learner.agent.eval(); learner.mixer.eval()

    rows = []
    for scenario_seed in test_seeds:
        env = make_env(scenario_seed, horizon)
        direct_cells = direct_access_cells(env)
        out = env.reset(seed=scenario_seed)
        rng = np.random.default_rng(900000 + int(scenario_seed))
        sums = {k: 0.0 for k in (
            'utility', 'served_ues', 'fully_satisfied_ues', 'relay_served_ues',
            'relay_required_ues', 'relay_required_served_ues', 'active_uavs',
            'relay_uavs', 'access_uavs', 'deep_access_uavs', 'steps_any_deep_access',
            'role_switches', 'battery_drain',
        )}
        steps = 0
        while True:
            actions = learner.act(out['obs'], out['action_mask'], 0.0, rng)
            battery_before = np.asarray(env.battery, dtype=np.float64).copy()
            env.last_network = None
            out = env.step(actions)
            snap = env.last_network
            if snap is None:
                raise RuntimeError('Missing network snapshot')
            if abs(float(out['reward']) - snap['utility']) > 1e-8:
                raise AssertionError('Executed reward != captured network utility')

            rates = snap['ue_rate_mbps']
            served = rates > EPS
            full = rates >= env.cfg.ue_demand_mbps - EPS
            active = snap['status'] == ACTIVE
            access = active & (snap['roles'] == ACCESS)
            d_bs = direct_bh_distance_m(env, snap['uav_positions'])
            deep_access = access & (d_bs > env.cfg.backhaul_radius_cells * env.cfg.cell_size_m + EPS)
            req = relay_required_ue_mask(env, snap['ue_positions'], direct_cells)

            relay_served = np.zeros(env.cfg.num_ues, dtype=bool)
            for aid, users in snap['association'].items():
                rid = snap['routes'].get(aid)
                if rid is None or snap['backhaul_caps_mbps'].get(aid, 0.0) <= EPS:
                    continue
                for u in users:
                    if served[u]:
                        relay_served[u] = True

            sums['utility'] += snap['utility']
            sums['served_ues'] += int(served.sum())
            sums['fully_satisfied_ues'] += int(full.sum())
            sums['relay_served_ues'] += int(relay_served.sum())
            sums['relay_required_ues'] += int(req.sum())
            sums['relay_required_served_ues'] += int((req & served).sum())
            sums['active_uavs'] += int(active.sum())
            sums['relay_uavs'] += int(out['info'].get('num_relay', 0))
            sums['access_uavs'] += int(access.sum())
            sums['deep_access_uavs'] += int(deep_access.sum())
            sums['steps_any_deep_access'] += int(deep_access.any())
            sums['role_switches'] += int(out['info'].get('role_switches_this_slot', 0))
            sums['battery_drain'] += float(np.maximum(0.0, battery_before - env.battery).sum())
            steps += 1
            if out['truncated']:
                break

        rows.append({
            'scenario_seed': int(scenario_seed),
            'steps': steps,
            'mean_utility': sums['utility'] / steps,
            'mean_served_ues': sums['served_ues'] / steps,
            'mean_fully_satisfied_ues': sums['fully_satisfied_ues'] / steps,
            'mean_relay_served_ues': sums['relay_served_ues'] / steps,
            'mean_relay_required_ues': sums['relay_required_ues'] / steps,
            'mean_relay_required_served_ues': sums['relay_required_served_ues'] / steps,
            'relay_required_service_fraction': (sums['relay_required_served_ues'] / sums['relay_required_ues'] if sums['relay_required_ues'] else 0.0),
            'mean_active_uavs': sums['active_uavs'] / steps,
            'mean_relay_uavs': sums['relay_uavs'] / steps,
            'mean_access_uavs': sums['access_uavs'] / steps,
            'mean_deep_access_uavs': sums['deep_access_uavs'] / steps,
            'deep_access_fraction_of_access': (sums['deep_access_uavs'] / sums['access_uavs'] if sums['access_uavs'] else 0.0),
            'fraction_steps_any_deep_access': sums['steps_any_deep_access'] / steps,
            'mean_role_switches_per_step': sums['role_switches'] / steps,
            'mean_battery_drain_per_step': sums['battery_drain'] / steps,
        })

    metric_keys = [k for k in rows[0] if k.startswith('mean_') or k.startswith('fraction_') or k == 'relay_required_service_fraction']
    summary = {k: float(np.mean([r[k] for r in rows])) for k in metric_keys}
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=91, help='training seed checkpoint to evaluate')
    ap.add_argument('--checkpoint', type=Path, default=None)
    ap.add_argument('--test-seeds', type=int, nargs='+', default=list(TEST_SEEDS))
    ap.add_argument('--horizon', type=int, default=150)
    ap.add_argument('--torch-threads', type=int, default=1)
    ap.add_argument('--outdir', type=Path, default=Path('results/relay_stress_r4_qmix/eval'))
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.torch_threads))

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = ROOT / 'results/relay_stress_r4_qmix/original_reward' / f'seed{args.seed}_best_final_block.pt'
    elif not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    rows, summary = evaluate_checkpoint(checkpoint, args.test_seeds, args.horizon)
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f'seed{args.seed}_heldout_kpis.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    train_summary_path = checkpoint.parent / f'seed{args.seed}_summary.json'
    stored_utility = None
    if train_summary_path.exists() and list(args.test_seeds) == list(TEST_SEEDS) and args.horizon == 150:
        stored_utility = float(json.loads(train_summary_path.read_text(encoding='utf-8'))['heldout_test']['mean_utility'])
        if not np.isclose(summary['mean_utility'], stored_utility, rtol=0, atol=5e-7):
            raise AssertionError(f"Replay utility {summary['mean_utility']} != stored {stored_utility}")

    payload = {
        'method': 'Spatial-QMIX R4 relay-stress', 'train_seed': int(args.seed),
        'checkpoint': str(checkpoint), 'backhaul_radius_m': 200.0,
        'reward': 'original environment utility U_t', 'test_seeds': list(args.test_seeds),
        'horizon': int(args.horizon), 'stored_heldout_utility_check': stored_utility,
        'metrics': summary, 'rows': rows,
    }
    json_path = outdir / f'seed{args.seed}_heldout_kpis.json'
    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('R4_QMIX_EVAL', json.dumps(summary), flush=True)
    print('WROTE', csv_path, json_path, flush=True)


if __name__ == '__main__':
    main()
