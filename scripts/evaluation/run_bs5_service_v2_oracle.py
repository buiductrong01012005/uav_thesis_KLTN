"""Evaluate a privileged approximate reference on unchanged service-v2 physics."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
from src.envs.bs5_service_v2 import BS5ServiceV2Env, ServiceV2Config, FULL_CONDITION
from src.baselines.service_v2_privileged_search import PrivilegedPlacementSearch


def metrics(env, info):
    snap, net = env.last_service_snapshot, env.last_v2_network
    active = np.flatnonzero(snap["status"] == 0)
    access = [a for a in active if snap["roles"][a] == 0]
    relays = [a for a in active if snap["roles"][a] == 1]
    covered = np.zeros(env.cfg.num_ues, dtype=bool)
    for a in access:
        covered |= (
            np.linalg.norm(snap["ue_positions"] - snap["uav_positions"][a], axis=1)
            <= env.cfg.access_radius_cells + 1e-9
        )
    paths = env.last_v2_radio_detail["paths"]
    forwarding = {
        r
        for a, r in net["routes"].items()
        if r is not None and net["backhaul_caps_mbps"].get(a, 0) > 1e-9
    }
    positive = int(np.sum(net["ue_rate_mbps"] > 1e-9))
    return dict(
        utility=info["utility"],
        throughput_mbps=float(net["ue_rate_mbps"].sum()),
        qos_ues=info["qos_fraction"] * env.cfg.num_ues,
        positive_ues=positive,
        qos_fraction=info["qos_fraction"],
        ue_outage_fraction=info["ue_outage_fraction"],
        covered_ues=int(covered.sum()),
        associated_ues=sum(map(len, net["association"].values())),
        reachable_ues=sum(len(net["association"][a]) for a, path in paths.items() if path),
        active_access=len(access),
        active_relay=len(relays),
        forwarding_relay=len(forwarding),
        access_offer_mbps=float(net["access_raw_ue_rates"].sum()),
        reachable_offer_mbps=sum(net["access_offer_mbps"][a] for a, path in paths.items() if path),
        schedule_airtime=net["schedule_time_sum"],
        zero_service_slot_fraction=float(positive == 0),
        returning_uavs=int(np.sum(snap["status"] == 1)),
        charging_uavs=int(np.sum(snap["status"] == 2)),
        stranded_slot_fraction=float(bool(env.last_return_control["stranded_ids_post"])),
    )


def run(args):
    protocol = json.loads(args.protocol.read_text())
    # Match every original physics hash; extra reference files do not alter it.
    checked = {}
    for name, expected in protocol["source_sha256"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Source mismatch: " + name)
        checked[name] = actual
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        method="current-state privileged placement search with legal navigation",
        global_upper_bound=False,
        reads_future_ue_tape=False,
        period=args.period,
        budget=args.budget,
        seeds=args.seeds,
        horizon=args.slots,
        service=protocol["service"],
        allow_relay=True,
        runtime=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__),
        comparison_runtime=protocol["runtime"],
        comparison="learned checkpoint validation CSV; not rerun in this runtime",
        unchanged_source_sha256=checked,
    )
    (args.output / "protocol.json").write_text(json.dumps(receipt, indent=2) + "\n")
    summaries = []
    for seed in args.seeds:
        env = BS5ServiceV2Env(
            FULL_CONDITION,
            service_config=ServiceV2Config(**protocol["service"]),
            allow_relay=True,
            horizon=args.slots,
        )
        env.reset(seed=seed, assignment_seed=seed)
        policy = PrivilegedPlacementSearch(
            env, seed=731 + seed, period=args.period, budget=args.budget
        )
        samples, max_violation, cycles, pending = [], 0.0, [], {}
        tic = time.perf_counter()
        with gzip.open(args.output / f"slots_{seed}.jsonl.gz", "wt") as stream:
            for slot in range(args.slots):
                before = env.status.copy()
                mask = env.get_action_mask()
                actions = policy.act(mask)
                assert all(mask[i, a] for i, a in enumerate(actions))
                out = env.step(actions)
                row = metrics(env, out["info"])
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
                        pending[a].update(served_again_slot=env.t, outcome="served_again")
                        cycles.append(pending.pop(a))
                    if before[a] == 0 and env.status[a] == 1:
                        if a in pending:
                            pending[a]["outcome"] = "next_return_without_service"
                            cycles.append(pending.pop(a))
                        pending[a] = dict(agent=a, return_slot=env.t, rejoin_slot=None)
                assert net["schedule_time_sum"] <= 1 + 1e-6
                assert net["radio_flow_violation_mbps"] <= 1e-6
                assert np.isclose(row["utility"], row["throughput_mbps"] / 360, atol=1e-10)
                assert np.all(env.battery >= -1e-9)
                assert np.isfinite(out["obs"]).all()
                max_violation = max(max_violation, net["radio_flow_violation_mbps"])
                samples.append(row)
                stream.write(
                    json.dumps(
                        dict(
                            slot=env.t,
                            actions=actions,
                            **row,
                            positions=env.uav_positions.tolist(),
                            roles=env.roles.tolist(),
                            status=env.status.tolist(),
                            battery=env.battery.tolist(),
                            ue_rates=net["ue_rate_mbps"].tolist(),
                            radio=env.last_v2_radio_detail,
                        )
                    )
                    + "\n"
                )
                if env.t % 120 == 0:
                    print(
                        json.dumps(
                            dict(
                                seed=seed,
                                slot=env.t,
                                utility=float(np.mean([r["utility"] for r in samples])),
                                last120_utility=float(
                                    np.mean([r["utility"] for r in samples[-120:]])
                                ),
                                calls=policy.calls,
                                wall_seconds=time.perf_counter() - tic,
                            )
                        ),
                        flush=True,
                    )
        summary = dict(
            seed=seed,
            slots=args.slots,
            **{k: float(np.mean([r[k] for r in samples])) for k in samples[0]},
            max_flow_violation_mbps=max_violation,
            calls=policy.calls,
            served_again_cycles=sum(x["outcome"] == "served_again" for x in cycles),
            wall_seconds=time.perf_counter() - tic,
        )
        summaries.append(summary)
        (args.output / f"plans_{seed}.json").write_text(json.dumps(policy.plans, indent=2) + "\n")
        (args.output / f"cycles_{seed}.json").write_text(
            json.dumps(dict(completed=cycles, pending=list(pending.values())), indent=2) + "\n"
        )
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        with (args.output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print("COMPLETE " + json.dumps(summary), flush=True)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[620001, 620002])
    parser.add_argument("--slots", type=int, default=1200)
    parser.add_argument("--period", type=int, default=60)
    parser.add_argument("--budget", type=int, default=650)
    run(parser.parse_args())
