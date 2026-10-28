"""Full-lifecycle fixed-action comparison, not trained policy performance."""

import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.envs.bs5_network_v3 import BS5NetworkV3Env
from src.envs.bs5_service_v2 import ServiceV2Config
from src.envs.bs5_tcf_env import TCFWeights, jain
from src.diagnostics.bs5_radio_v3 import RadioV3Config
from scripts.verification.check_bs5_service_v2_dynamic import actions_for
from scripts.training.run_bs5_network_v3 import observe_step, write_csv, atomic_json


def run(output, steps=1200):
    output.mkdir(parents=True, exist_ok=False)
    radios = [
        RadioV3Config(
            access_radius_m=200,
            uav_altitude_m=100,
            bs_tx_dbm=20,
            bs_uav_channel="legacy_urban_mean",
        )
    ]
    radios += [
        RadioV3Config(access_radius_m=r, uav_altitude_m=h, bs_tx_dbm=30)
        for h, r in itertools.product([80.0, 100.0], [200.0, 250.0, 300.0])
    ]
    all_summaries = []
    reference_positions, reference_status, reference_battery = [], [], []
    for k, radio in enumerate(radios):
        env = BS5NetworkV3Env(
            radio_config=radio,
            service_config=ServiceV2Config(beacon_radius_m=radio.access_radius_m),
            reward_weights=TCFWeights(0.45, 0.45, 0.10),
            horizon=steps,
        )
        env.reset(seed=910801, assignment_seed=910801)
        samples, rates, cycles = [], [], []
        pending = {}
        tic = time.perf_counter()
        for slot in range(steps):
            before = env.status.copy()
            actions = actions_for(env)
            mask = env.get_action_mask()
            assert all(mask[i, a] for i, a in enumerate(actions))
            out = env.step(actions)
            if k == 0:
                reference_positions.append(env.uav_positions.copy())
                reference_status.append(env.status.copy())
                reference_battery.append(env.battery.copy())
            else:
                np.testing.assert_array_equal(env.uav_positions, reference_positions[slot])
                np.testing.assert_array_equal(env.status, reference_status[slot])
                np.testing.assert_array_equal(env.battery, reference_battery[slot])
            m = observe_step(env, out["info"])
            assert 0 <= out["reward"] <= 1
            samples.append(dict(slot=slot + 1, **m))
            net = env.last_v2_network
            rates.append(net["ue_rate_mbps"].copy())
            served = {
                a: any(net["ue_rate_mbps"][u] > 1e-6 for u in users)
                for a, users in net["association"].items()
            }
            for a, r in net["routes"].items():
                if r is not None and net["backhaul_caps_mbps"].get(a, 0) > 1e-6:
                    served[r] = True
            for a in range(5):
                if before[a] == 2 and env.status[a] == 0 and a in pending:
                    pending[a]["rejoin_slot"] = env.t
                if a in pending and pending[a]["rejoin_slot"] is not None and served.get(a, False):
                    pending[a].update(served_again_slot=env.t, outcome="served_again")
                    cycles.append(pending.pop(a))
                if before[a] == 0 and env.status[a] == 1:
                    if a in pending:
                        pending[a]["outcome"] = "next_return_without_service"
                        cycles.append(pending.pop(a))
                    pending[a] = dict(agent=a, return_slot=env.t, rejoin_slot=None)
        rates = np.asarray(rates)
        summary = dict(
            config_id=k,
            **asdict(radio),
            seed=910801,
            slots=steps,
            **{
                name: float(np.mean([r[name] for r in samples]))
                for name in samples[0]
                if name != "slot"
            },
            jain_time_average=jain(rates.mean(axis=0)),
            served_again_cycles=sum(c["outcome"] == "served_again" for c in cycles),
            max_flow_violation=max(r["flow_violation_mbps"] for r in samples),
            max_airtime=max(r["schedule_airtime"] for r in samples),
            trajectory_energy_identical_to_control=True,
            wall_seconds=time.perf_counter() - tic,
        )
        all_summaries.append(summary)
        np.savez_compressed(output / f"config{k}_rates.npz", ue_rates=rates)
        write_csv(output / f"config{k}_slots.csv", samples)
        atomic_json(
            output / f"config{k}_cycles.json",
            dict(completed=cycles, pending=list(pending.values())),
        )
        write_csv(output / "summary.csv", all_summaries)
        atomic_json(output / "summary.json", all_summaries)
        print(json.dumps(summary), flush=True)
    return all_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args()
    run(args.output, args.steps)
