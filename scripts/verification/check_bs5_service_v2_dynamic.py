"""Short dynamic lifecycle diagnostic, privileged scripted navigation, not RL benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.envs.bs5_service_v2 import BS5ServiceV2Env, ServiceV2Config
from src.envs.v0_env import MOVE_DELTAS


TARGETS = np.asarray([[14, 10], [15, 13], [16, 15], [18, 17], [17, 18]])


def actions_for(env):
    # Fixed Access/Relay targets, no simulator previews and no hidden demand search.
    # Exact current self positions are used solely for deterministic fixture navigation.
    mask = env.get_action_mask()
    actions = [8] * 5
    for a in range(5):
        if env.status[a] != 0:
            continue
        role = int(a < 2)
        options = [x for x in np.flatnonzero(mask[a]) if x % 2 == role]
        actions[a] = int(
            min(
                options,
                key=lambda x: (
                    np.abs(env.uav_positions[a] + MOVE_DELTAS[x // 2] - TARGETS[a]).sum(),
                    x,
                ),
            )
        )
    return actions


def run(output, steps=1200, backhauls=(40, 100)):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Nonempty output; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for bandwidth in backhauls:
        config = ServiceV2Config(backhaul_total_mhz=bandwidth)
        env = BS5ServiceV2Env(service_config=config, horizon=steps)
        env.reset(seed=910801, assignment_seed=910801)
        cycles, pending = [], {}
        metrics = []
        tic = time.perf_counter()
        with gzip.open(output / f"BH{bandwidth}_slots.jsonl.gz", "wt", encoding="utf-8") as stream:
            for slot in range(steps):
                before = env.status.copy()
                out = env.step(actions_for(env))
                net = env.last_v2_network
                served = {
                    a: any(net["ue_rate_mbps"][u] > 1e-9 for u in users)
                    for a, users in net["association"].items()
                }
                for a, relay in net["routes"].items():
                    if relay is not None and net["backhaul_caps_mbps"].get(a, 0) > 1e-9:
                        served[relay] = True
                for a in range(5):
                    if before[a] == 2 and env.status[a] == 0 and a in pending:
                        pending[a]["rejoin_slot"] = env.t
                    if (
                        a in pending
                        and pending[a]["rejoin_slot"] is not None
                        and served.get(a, False)
                    ):
                        pending[a]["served_again_slot"] = env.t
                        pending[a]["outcome"] = "served_again"
                        cycles.append(pending.pop(a))
                    if before[a] == 0 and env.status[a] == 1:
                        if a in pending:
                            pending[a]["outcome"] = "next_return_without_service"
                            cycles.append(pending.pop(a))
                        pending[a] = dict(
                            agent=a, return_slot=env.t, rejoin_slot=None, served_again_slot=None
                        )
                row = dict(
                    slot=env.t,
                    utility=out["info"]["utility"],
                    throughput_mbps=float(net["ue_rate_mbps"].sum()),
                    qos_ues=int(np.sum(net["ue_rate_mbps"] >= 4 - 1e-9)),
                    positive_ues=int(np.sum(net["ue_rate_mbps"] > 1e-9)),
                    status=env.status.tolist(),
                    battery=env.battery.tolist(),
                    positions=env.uav_positions.tolist(),
                    report_source=env.reports.latest_delivery_source,
                    max_flow_violation=net["radio_flow_violation_mbps"],
                    schedule_time_sum=net["schedule_time_sum"],
                    radio_runtime_ms=net["radio_runtime_ms"],
                    return_stranded=env.last_return_control["stranded_ids_post"],
                )
                assert not row["return_stranded"], row
                assert row["max_flow_violation"] <= 1e-6 and row["schedule_time_sum"] <= 1 + 1e-6
                assert np.isfinite(out["obs"]).all()
                # Keep causal sensor/flow evidence; info true state is not actor observation.
                row["radio"] = env.last_v2_radio_detail
                row["service_ue_rates"] = net["ue_rate_mbps"].tolist()
                stream.write(json.dumps(row) + "\n")
                metrics.append(row)
                if env.t % 300 == 0:
                    print(
                        f"BH{bandwidth}: {env.t}/{steps}, served-again cycles={sum(x['outcome'] == 'served_again' for x in cycles)}",
                        flush=True,
                    )
        summaries.append(
            dict(
                backhaul_mhz=bandwidth,
                steps=steps,
                seed=910801,
                mean_utility=float(np.mean([x["utility"] for x in metrics])),
                mean_throughput_mbps=float(np.mean([x["throughput_mbps"] for x in metrics])),
                mean_qos_ues=float(np.mean([x["qos_ues"] for x in metrics])),
                mean_positive_ues=float(np.mean([x["positive_ues"] for x in metrics])),
                observed_served_again_cycles=sum(x["outcome"] == "served_again" for x in cycles),
                next_return_without_service=sum(
                    x["outcome"] == "next_return_without_service" for x in cycles
                ),
                unfinished_cycles=list(pending.values()),
                per_uav_served_again_cycles={
                    str(a): sum(x["agent"] == a and x["outcome"] == "served_again" for x in cycles)
                    for a in range(5)
                },
                cycles=cycles,
                wall_seconds=time.perf_counter() - tic,
                service_config=asdict(config),
                scope="scripted 2R/3A fixed-target lifecycle diagnostic, not an algorithm or learned-policy comparison",
            )
        )
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    simple = [
        {
            k: v
            for k, v in r.items()
            if k
            not in {"cycles", "unfinished_cycles", "service_config", "per_uav_served_again_cycles"}
        }
        for r in summaries
    ]
    with (output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(simple[0]))
        writer.writeheader()
        writer.writerows(simple)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/bs5_service_v2_dynamic")
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    result = run(args.output, args.steps)
    print(
        json.dumps(
            [{k: v for k, v in r.items() if k not in {"cycles", "service_config"}} for r in result],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
