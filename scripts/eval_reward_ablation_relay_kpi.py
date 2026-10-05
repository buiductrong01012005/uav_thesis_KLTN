"""Read-only replay of trained A0/A1/A2/A3/A4 checkpoints with real UE/KPI usage.

Uses the same env, held-out seeds, 150-slot horizon, deterministic epsilon=0
policy and per-seed action RNG as run_thesis_no_curriculum_fulltask.py.
No retraining, no modification of reward or physics, no re-running routing.
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

from src.envs.thesis_cartesian_r6_curriculum_env import CartesianR6BatteryConfig, CartesianR6BatteryEnv
from src.rl.v5_relay_request_spatial_qmix import V5RelayRequestSpatialQMIXLearner
from src.envs.v4_fixed_mobile_battery_env import ACTIVE
from scripts.run_thesis_no_curriculum_fulltask import active_pairwise_spatial_metrics

VARIANT_PATHS = {
    "a0": "results/thesis_curriculum_ablation/direct_full_task_randomized",
    "energy": "results/reward_ablation/energy",
    "role": "results/reward_ablation/role",
    "combined": "results/reward_ablation/combined",
    "spatial": "results/reward_ablation/spatial",
}
TRAIN_SEEDS = (91, 92, 93)
HELDOUT_SEEDS = tuple(range(3101, 3121))
EPS = 1e-9


class RelayCountEnv(CartesianR6BatteryEnv):
    """Capture metric from the EXACT network calculation used by env.step()."""

    def __init__(self, cfg):
        self.in_slot = None
        super().__init__(cfg)

    def compute_network(self):
        net = super().compute_network()
        rates = np.asarray(net["ue_rate_mbps"], dtype=np.float64)
        served = int(np.count_nonzero(rates > EPS))
        full = int(np.count_nonzero(rates >= self.cfg.ue_demand_mbps - EPS))
        via_relay = 0
        active_routes = 0
        for aid, rid in net["routes"].items():
            if rid is None or net["backhaul_caps_mbps"].get(aid, 0.) <= EPS:
                continue
            n = sum(rates[int(u)] > EPS for u in net["association"].get(aid, []))
            if n:
                active_routes += 1
                via_relay += n
        if not 0 <= via_relay <= served <= self.cfg.num_ues:
            raise AssertionError("Invalid served/relayed UE accounting")
        self.in_slot = {
            "utility": float(net["utility"]),
            "served_ues": served,
            "fully_satisfied_ues": full,
            "relay_served_ues": int(via_relay),
            "effective_relay_routes": active_routes,
        }
        return net


def learner_from_checkpoint(env, path):
    learner = V5RelayRequestSpatialQMIXLearner(
        env.cfg.num_uavs, env.obs_dim, env.state_dim, env.n_actions,
        grid_size=env.cfg.grid_size, coarse_size=env.cfg.coarse_obs_size,
        hidden_dim=64, mixing_embed_dim=32, device="cpu",
    )
    learner.load(str(path))
    learner.agent.eval()
    learner.mixer.eval()
    return learner


def run_case(learner, test_seed, horizon, min_separation_m=100.0):
    env = RelayCountEnv(CartesianR6BatteryConfig(
        seed=int(test_seed), horizon=int(horizon),
        residual_initial_battery=30.0, randomize_residual_uav=True,
    ))
    out = env.reset(seed=int(test_seed))
    rng = np.random.default_rng(900000 + int(test_seed))
    sums = {name: 0.0 for name in (
        "utility", "served_ues", "fully_satisfied_ues", "relay_served_ues",
        "effective_relay_routes", "num_relays", "steps_with_relay_traffic",
        "pairwise_active_distance_m", "close_pair_fraction", "spatial_closeness_penalty",
    )}
    count = 0
    while True:
        actions = learner.act(out["obs"], out["action_mask"], 0.0, rng)
        active_before = np.asarray(env.status == ACTIVE, dtype=bool).copy()
        env.in_slot = None
        out = env.step(actions)
        slot = env.in_slot
        if slot is None or abs(slot["utility"] - float(out["reward"])) > 1e-8:
            raise AssertionError("Snapshot utility differs from executed environment reward")
        for name in ("utility", "served_ues", "fully_satisfied_ues", "relay_served_ues", "effective_relay_routes"):
            sums[name] += slot[name]
        sums["num_relays"] += float(out["info"].get("num_relay", 0))
        sums["steps_with_relay_traffic"] += int(slot["relay_served_ues"] > 0)
        spatial, mean_dist, close_frac = active_pairwise_spatial_metrics(
            env.uav_positions, active_before, env.cfg.cell_size_m, min_separation_m
        )
        sums["pairwise_active_distance_m"] += mean_dist
        sums["close_pair_fraction"] += close_frac
        sums["spatial_closeness_penalty"] += spatial
        count += 1
        if out["truncated"]:
            break
    return {"scenario_seed": int(test_seed), "steps": count,
            **{f"mean_{name}": sums[name] / count for name in sums},
            "total_relay_ue_slots": int(sums["relay_served_ues"]) }


def reference_summary(variant, train_seed, root):
    if variant == "a0":
        candidate = root / VARIANT_PATHS[variant] / f"seed{train_seed}_summary.json"
    else:
        candidate = root / VARIANT_PATHS[variant] / f"seed{train_seed}_summary.json"
    return json.loads(candidate.read_text(encoding="utf-8")) if candidate.is_file() else None


def main():
    parser = argparse.ArgumentParser(description="Evaluate served UE, actual Relay traffic and spatial KPIs from A0..A4 (NO training).")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANT_PATHS), default=list(VARIANT_PATHS))
    parser.add_argument("--train-seeds", nargs="+", type=int, default=list(TRAIN_SEEDS))
    parser.add_argument("--test-seeds", nargs="+", type=int, default=list(HELDOUT_SEEDS))
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--min-separation-m", type=float, default=100.0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--outdir", type=Path, default=Path("results/reward_ablation/relay_kpi"))
    parser.add_argument("--skip-completed", action="store_true", help="Skip exact replay file if already present")
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    per_seed = []
    for variant in args.variants:
        for train_seed in args.train_seeds:
            checkpoint = ROOT / VARIANT_PATHS[variant] / f"seed{train_seed}_best_final_block.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            results_file = outdir / f"{variant}_seed{train_seed}_episodes.json"
            if args.skip_completed and results_file.is_file():
                payload = json.loads(results_file.read_text(encoding="utf-8"))
                if (payload.get("test_seeds") == args.test_seeds
                        and payload.get("horizon") == args.horizon
                        and float(payload.get("min_separation_m", -1)) == float(args.min_separation_m)
                        and "mean_pairwise_active_distance_m" in payload.get("metrics", {})):
                    print(f"SKIP {variant} seed {train_seed} (matching completed eval)", flush=True)
                    rows = payload["rows"]
                    all_rows.extend({"variant": variant,"train_seed":train_seed,**r} for r in rows)
                    per_seed.append({"variant":variant,"train_seed":train_seed,**payload["metrics"]})
                    continue
            env = RelayCountEnv(CartesianR6BatteryConfig(seed=args.test_seeds[0], horizon=args.horizon, residual_initial_battery=30.0, randomize_residual_uav=True))
            learner = learner_from_checkpoint(env, checkpoint)
            rows = []
            for test_seed in args.test_seeds:
                row = run_case(learner, test_seed, args.horizon, args.min_separation_m)
                rows.append(row)
            metrics = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key.startswith("mean_")}
            old = reference_summary(variant, train_seed, ROOT)
            if old is not None and len(args.test_seeds) == 20 and tuple(args.test_seeds) == HELDOUT_SEEDS and args.horizon == 150:
                expected = float(old["heldout_test"]["mean_utility"])
                if not np.isclose(metrics["mean_utility"], expected, rtol=0, atol=5e-7):
                    raise AssertionError(f"{variant}/{train_seed}: replay utility {metrics['mean_utility']:.9f} != stored {expected:.9f}")
                print(f"MATCH summary utility for {variant} train seed {train_seed}: {expected:.6f}", flush=True)
            payload = {"variant":variant,"train_seed":train_seed,"checkpoint":str(checkpoint.relative_to(ROOT)),
                       "test_seeds":args.test_seeds,"horizon":args.horizon,"min_separation_m":args.min_separation_m,"metrics":metrics,"rows":rows,
                       "definition":"Number of positive-rate UEs associated with Access whose selected actual end-to-end backhaul route includes Relay", "note":"Original reward and environment unchanged; network counted within env.step()"}
            results_file.write_text(json.dumps(payload,indent=2),encoding="utf-8")
            all_rows.extend({"variant":variant,"train_seed":train_seed,**r} for r in rows)
            per_seed.append({"variant":variant,"train_seed":train_seed,**metrics})
            print(
                f"DONE {variant} seed {train_seed}: utility={metrics['mean_utility']:.5f} "
                f"served UE={metrics['mean_served_ues']:.2f} relay UE={metrics['mean_relay_served_ues']:.3f} "
                f"pair-dist={metrics['mean_pairwise_active_distance_m']:.1f}m "
                f"close-pairs={metrics['mean_close_pair_fraction']:.3f}", flush=True
            )

    def write_csv(filename, rows):
        if not rows:return
        with (outdir / filename).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    summary=[]
    for variant in args.variants:
        p=[r for r in per_seed if r['variant']==variant]
        if not p:continue
        numeric_keys=[k for k in p[0] if k.startswith('mean_')]
        summary.append({"variant":variant,"training_seeds":len(p),**{k:float(np.mean([r[k] for r in p])) for k in numeric_keys}})
    write_csv("relay_kpi_by_episode.csv",all_rows)
    write_csv("relay_kpi_by_training_seed.csv",per_seed)
    write_csv("relay_kpi_summary.csv",summary)
    print("\nFINAL SUMMARY",flush=True)
    for row in summary: print(json.dumps(row,ensure_ascii=False),flush=True)
    print(f"OUTPUT {outdir}",flush=True)


if __name__ == '__main__':main()
