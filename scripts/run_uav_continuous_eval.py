"""One-reset, uninterrupted, long-horizon operation of already-trained policies.

No periodic episode resets. The same BS failure, UE identities, group mobility,
UAV locations, batteries, and charging lifecycle persist for all N slots.

Requires scripts/run_uav_operational_eval.py from the earlier source patch
for its exact in-slot networking capture and policy implementations.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_uav_operational_eval import (
    CapturedCartesianEnv, LocalAllAccess, counts_from_exact_snapshot,
    episode_start_snapshot, make_learner,
)
from src.baselines.thesis_exhaustive_cartesian_role_greedy import ExhaustiveCartesianRoleGreedy
from src.envs.thesis_cartesian_r6_curriculum_env import CartesianR6BatteryConfig
from src.envs.v4_fixed_mobile_battery_env import ACTIVE, RETURNING, CHARGING
from src.envs.v0_env import ACCESS, RELAY

FIELDNAMES = [
    'step', 'utility', 'served_ues', 'fully_satisfied_ues', 'relay_served_ues',
    'effective_relay_routes', 'access', 'relay', 'returning', 'charging',
    'mean_battery', 'residual_battery', 'decision_ms', 'return_events_this_slot',
    'rejoin_events_this_slot',
]


def make_continuous_env(seed: int, min_horizon: int) -> CapturedCartesianEnv:
    # The parent simulator only uses horizon to compute its truncation flag;
    # choosing one larger than the planned evaluation avoids episode resets.
    cfg = CartesianR6BatteryConfig(
        seed=int(seed), horizon=int(min_horizon),
        enable_ue_mobility=True, enable_battery_cycle=True,
        residual_initial_battery=30.0, randomize_residual_uav=True,
    )
    return CapturedCartesianEnv(cfg)


def chosen_snapshots(steps: int) -> set[int]:
    points = {1, 20, 50, 100, 150, 250, 500, 1000}
    if steps > 0:
        points |= {steps // 4, steps // 2, (3 * steps) // 4, steps}
    return {t for t in points if t > 0 and (steps <= 0 or t <= steps)}


def json_scalar(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {type(obj).__name__} cannot be JSON encoded')


def run_one(args, label: str, policy_name: str, checkpoint: Path | None):
    env = make_continuous_env(args.scenario_seed, args.steps + 1 if args.steps else 2**31 - 1)
    out = env.reset(seed=args.scenario_seed)  # *** Only reset for this entire run. ***
    initial_state = episode_start_snapshot(env)
    learner = make_learner(env, checkpoint, args.device) if policy_name == 'rl' else None
    policy = (LocalAllAccess(env) if policy_name == 'local5a' else
              ExhaustiveCartesianRoleGreedy(env) if policy_name == 'exhaustive' else None)
    rng = np.random.default_rng(400000 + sum(ord(ch) for ch in label))

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    csvpath = outdir / f'{label}_steps.csv'
    summary_path = outdir / f'{label}_summary.json'
    snapshots: dict[str, dict] = {'0': initial_state}
    best_relay = None
    best_relay_n = -1
    first_return_step = None
    first_rejoin_step = None
    n_return_events = n_rejoin_events = 0
    utility_sum = served_sum = full_sum = relayed_sum = decision_sum = 0.0
    row_count = 0
    start = time.perf_counter()
    frames_to_keep = chosen_snapshots(args.steps)

    print(f'START {label}: scenario {args.scenario_seed}; one reset only; target steps={args.steps or "until Ctrl+C"}', flush=True)
    try:
        with csvpath.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
            writer.writeheader()
            while args.steps == 0 or row_count < args.steps:
                env._capture_network = False
                before_status = env.status.copy()
                tic = time.perf_counter()
                if learner is not None:
                    actions = learner.act(out['obs'], out['action_mask'], 0.0, rng)
                elif policy_name == 'random':
                    actions = [int(rng.choice(np.flatnonzero(mask))) for mask in out['action_mask']]
                else:
                    actions = policy.act(out['action_mask'])
                decision_ms = 1000.0 * (time.perf_counter() - tic)

                env._capture_network = True
                out = env.step(actions)
                env._capture_network = False
                if out.get('terminated') or out.get('truncated'):
                    raise RuntimeError('Unexpected termination: continuous simulator should remain active')
                snap = env.last_network_snapshot
                if snap is None or abs(float(snap['utility']) - float(out['reward'])) > 1e-8:
                    raise RuntimeError('In-slot reward and network snapshot disagree')

                row_count += 1
                served, full, via_relay, paths = counts_from_exact_snapshot(snap, env.cfg.ue_demand_mbps)
                after_status = env.status.copy()
                return_events = int(np.sum((before_status == ACTIVE) & (after_status == RETURNING)))
                rejoin_events = int(np.sum((before_status == CHARGING) & (after_status == ACTIVE)))
                n_return_events += return_events
                n_rejoin_events += rejoin_events
                if first_return_step is None and return_events:
                    first_return_step = row_count
                    if policy_name == 'rl': snapshots['first_return'] = snap
                if first_rejoin_step is None and rejoin_events:
                    first_rejoin_step = row_count
                    if policy_name == 'rl': snapshots['first_rejoin'] = snap
                row = {
                    'step': row_count,
                    'utility': float(out['reward']),
                    'served_ues': served, 'fully_satisfied_ues': full,
                    'relay_served_ues': via_relay, 'effective_relay_routes': paths,
                    'access': int(np.sum((np.asarray(snap['uav_roles']) == ACCESS) & (np.asarray(snap['uav_status']) == ACTIVE))),
                    'relay': int(np.sum((np.asarray(snap['uav_roles']) == RELAY) & (np.asarray(snap['uav_status']) == ACTIVE))),
                    'returning': int(np.sum(after_status == RETURNING)),
                    'charging': int(np.sum(after_status == CHARGING)),
                    'mean_battery': float(np.mean(out['info']['battery'])),
                    'residual_battery': float(out['info']['battery'][env.residual_uav_idx]),
                    'decision_ms': decision_ms,
                    'return_events_this_slot': return_events,
                    'rejoin_events_this_slot': rejoin_events,
                }
                writer.writerow(row)
                utility_sum += row['utility']
                served_sum += served
                full_sum += full
                relayed_sum += via_relay
                decision_sum += decision_ms

                if policy_name == 'rl':
                    snap['step_global'] = row_count
                    if row_count in frames_to_keep:
                        snapshots[str(row_count)] = snap
                    if via_relay > best_relay_n:
                        best_relay_n, best_relay = via_relay, snap

                if row_count % args.log_every == 0:
                    fh.flush()
                    print(f'{label}: {row_count} continuous steps | mean utility={utility_sum / row_count:.4f} | returns={n_return_events} rejoin={n_rejoin_events}', flush=True)
    except KeyboardInterrupt:
        print(f'Interrupted at {row_count} steps; saving partial continuous-run data.', flush=True)

    summary = {
        'method': label,
        'evaluation_type': 'one continuous simulator run without episode reset',
        'scenario_seed': args.scenario_seed,
        'number_of_environment_resets': 1,
        'steps': row_count,
        'same_ue_population_and_failed_bs_throughout': True,
        'mobility_and_battery_continuous': True,
        'policy_training_horizon': 150,
        'evaluation_horizon': env.cfg.horizon,
        'checkpoint': str(checkpoint) if checkpoint else None,
        'mean_utility': utility_sum / max(1, row_count),
        'mean_served_ues': served_sum / max(1, row_count),
        'mean_fully_satisfied_ues': full_sum / max(1, row_count),
        'mean_relay_served_ues': relayed_sum / max(1, row_count),
        'mean_decision_ms': decision_sum / max(1, row_count),
        'return_events': n_return_events,
        'rejoin_events': n_rejoin_events,
        'first_return_step': first_return_step,
        'first_rejoin_step': first_rejoin_step,
        'runtime_s': time.perf_counter() - start,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if policy_name == 'rl' and label == f'rl_seed{args.case_training_seed}':
        trace = {
            'method': label,
            'evaluation_type': summary['evaluation_type'],
            'scenario_seed': args.scenario_seed,
            'case_frames': snapshots,
            'best_observed_relay': best_relay,
            'best_relay_served_ues': max(0, best_relay_n),
            'bs_xy_m': env.bs_xy.astype(float).tolist(),
            'failed_bs_idx': int(env.cfg.failed_bs_idx),
            'charging_station_cells': env.charging_stations.astype(float).tolist(),
            'affected_mask': env.affected_mask.astype(int).tolist(),
            'cell_size_m': float(env.cfg.cell_size_m),
            'access_radius_m': float(env.cfg.access_radius_cells * env.cfg.cell_size_m),
            'backhaul_radius_m': float(env.cfg.backhaul_radius_cells * env.cfg.cell_size_m),
            'ue_demand_mbps': float(env.cfg.ue_demand_mbps),
            'bs_height_m': float(env.cfg.bs_height_m),
            'uav_altitude_m': float(env.cfg.uav_altitude_m),
        }
        (outdir / 'rl_continuous_case_study.json').write_text(json.dumps(trace, default=json_scalar), encoding='utf-8')
    print('FINISHED', json.dumps(summary), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--steps', type=int, default=10000, help='Steps in ONE run; 0 = run until Ctrl+C (only one policy supported)')
    ap.add_argument('--scenario-seed', type=int, default=3101)
    ap.add_argument('--rl-seeds', nargs='+', type=int, default=[91, 92, 93])
    ap.add_argument('--case-training-seed', type=int, default=91)
    ap.add_argument('--methods', nargs='+', choices=['rl', 'random', 'local5a', 'exhaustive'], default=['random', 'local5a', 'rl'])
    ap.add_argument('--checkpoint-root', default='results/thesis_curriculum_ablation/direct_full_task_randomized')
    ap.add_argument('--outdir', default='results/thesis_continuous_eval')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--torch-threads', type=int, default=1)
    ap.add_argument('--log-every', type=int, default=1000)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.steps < 0 or args.log_every < 1:
        ap.error('--steps must be >=0 and --log-every >=1')
    if args.steps == 0 and (len(args.methods) != 1 or args.methods[0] == 'rl' and len(args.rl_seeds) != 1):
        ap.error('--steps 0 allows only one policy at a time; specify e.g. --methods rl --rl-seeds 91')
    if 'rl' in args.methods and args.case_training_seed not in args.rl_seeds and args.steps > 0:
        ap.error('case-training-seed must be included in --rl-seeds')
    torch.set_num_threads(max(1, args.torch_threads))
    ckroot = Path(args.checkpoint_root)
    if not ckroot.is_absolute():
        ckroot = ROOT / ckroot
    checkpoints = {s: ckroot / f'seed{s}_best_final_block.pt' for s in args.rl_seeds}
    if 'rl' in args.methods:
        missing = [str(p) for p in checkpoints.values() if not p.is_file()]
        if missing:
            ap.error('Missing real trained checkpoint(s): ' + ', '.join(missing))
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        items = [(f'rl_seed{s}', 'rl', checkpoints[s]) for s in args.rl_seeds] if method == 'rl' else [(method, method, None)]
        for label, policy, checkpoint in items:
            summary_path = outdir / f'{label}_summary.json'
            csv_path = outdir / f'{label}_steps.csv'
            if args.steps and not args.force and summary_path.is_file() and csv_path.is_file():
                old = json.loads(summary_path.read_text(encoding='utf-8'))
                if (old.get('evaluation_type') == 'one continuous simulator run without episode reset'
                        and old.get('steps') == args.steps and old.get('scenario_seed') == args.scenario_seed):
                    print('SKIP existing complete run:', label, flush=True)
                    continue
            run_one(args, label, policy, checkpoint)
    print('Plot: python scripts/plot_uav_continuous_eval.py --input', outdir, flush=True)


if __name__ == '__main__':
    main()
