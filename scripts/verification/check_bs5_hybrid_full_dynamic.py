"""Two-band Hybrid energy/service probe with fixed-target navigation, no RL."""

import argparse
from copy import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.diagnostics.bs5_hybrid_lifecycle import HybridLifecycleProbe
from src.diagnostics.bs5_hybrid_fdm import score_hybrid_fdm, HybridFDMConfig
from src.diagnostics.bs5_hybrid_full import HybridFullConfig
from src.envs.bs5_tcf_env import TCFWeights, jain
from scripts.verification.check_bs5_service_v2_dynamic import actions_for


CASES = [
    ("AR", "full", [], 0.0),
    ("split_2H", "split", [0, 1], 0.0),
    ("full_2H", "full", [0, 1], 0.0),
    ("full_H0", "full", [0], 0.0),
    ("full_H1", "full", [1], 0.0),
    ("full_2H_overhead", "full", [0, 1], 0.1),
]


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        version="bs5_hybrid_full_dynamic_1",
        seeds=args.seeds,
        steps=args.steps,
        controller="Fixed-target old2R/3A controller; selected Relay actions are marked Hybrid",
        resource="Legacy Raccess200/Rbh300/h100/BS20, two existing pools; fullH20/100MHz endpoint caps",
        weights=dict(throughput=0.45, coverage=0.45, fairness=0.10),
        energy="Normalized battery points only; Hsplit=.5E_A+.5E_R; Hfull=E_A+E_R; optional+.1 per active H slot",
        hardware="Full H can emit P_A+P_R≈601.19mW; deliberately larger peak than split≈300.59mW",
        observations="Not exposed to an actor; legacy observation cannot encode H; controlled diagnostic only",
        cases=[dict(name=n, mode=m, hybrid_ids=h, overhead=e) for n, m, h, e in CASES],
    )
    write_json(args.output / "protocol.json", protocol)
    summaries, reference_ue = [], {}
    counterfactual = []
    total_started = time.perf_counter()
    for seed in args.seeds:
        reference_ue[seed] = []
        for name, mode, hybrid_ids, overhead in CASES:
            env = HybridLifecycleProbe(
                mode=mode,
                hybrid_radio_overhead=overhead,
                reward_weights=TCFWeights(0.45, 0.45, 0.10),
                horizon=args.steps,
            )
            env.reset(seed=seed, assignment_seed=seed)
            initial = float(env.battery.sum())
            rows, cycles, arrays, pending = (
                [],
                [],
                {k: [] for k in ["ue_rates", "positions", "roles", "status", "battery", "actions"]},
                {},
            )
            for slot in range(args.steps):
                before = env.status.copy()
                actions = actions_for(env)
                out = env.step(actions, hybrid_ids=hybrid_ids)  # noqa: F841 - diagnostic output is not fed to an actor
                snap = env.last_service_snapshot
                result = env.last_probe_result
                metrics = result["metrics"]
                if name == "AR":
                    reference_ue[seed].append(snap["ue_positions"].copy())
                    view = copy(env)
                    for attr, key in [
                        ("uav_positions", "uav_positions"),
                        ("ue_positions", "ue_positions"),
                        ("roles", "roles"),
                        ("status", "status"),
                        ("battery", "battery"),
                    ]:
                        setattr(view, attr, snap[key].copy())
                    logical = np.where(
                        np.isin(np.arange(5), [0, 1]) & (view.roles == 1), 2, view.roles
                    )
                    for radio_name, config in [
                        ("split_2H", HybridFDMConfig()),
                        ("full_2H", HybridFullConfig()),
                    ]:
                        m = score_hybrid_fdm(view, logical, config, env.reward_weights)["metrics"]
                        counterfactual.append(dict(seed=seed, slot=slot + 1, radio=radio_name, **m))
                else:
                    np.testing.assert_array_equal(snap["ue_positions"], reference_ue[seed][slot])
                stranded = env.last_return_control["stranded_ids_post"]
                assert (
                    not stranded
                    and metrics["flow_violation_mbps"] < 1e-6
                    and metrics["schedule_airtime"] <= 1 + 1e-6
                )
                assert np.isfinite(env.battery).all() and np.all(env.battery >= 0)
                served = {
                    a: bool(np.any(result["ue_rate_mbps"][users] > 1e-6))
                    for a, users in result["association"].items()
                }
                for a, path in result["paths"].items():
                    if len(path) == 2 and result["flows"][a] > 1e-6:
                        served[int(result["edges"][path[0]].target[1:])] = True
                for aid in range(5):
                    if before[aid] == 2 and env.status[aid] == 0 and aid in pending:
                        pending[aid]["rejoin_slot"] = env.t
                    if (
                        aid in pending
                        and pending[aid]["rejoin_slot"] is not None
                        and served.get(aid, False)
                    ):
                        pending[aid].update(served_again_slot=env.t, outcome="served_again")
                        cycles.append(pending.pop(aid))
                    if before[aid] == 0 and env.status[aid] == 1:
                        if aid in pending:
                            pending[aid]["outcome"] = "next_return_without_service"
                            cycles.append(pending.pop(aid))
                        pending[aid] = dict(agent=aid, return_slot=env.t, rejoin_slot=None)
                rows.append(
                    dict(
                        slot=slot + 1,
                        **metrics,
                        **env.last_energy_probe,
                        active_uavs=int(np.sum(snap["status"] == 0)),
                        returning_uavs=int(np.sum(snap["status"] == 1)),
                        charging_uavs=int(np.sum(snap["status"] == 2)),
                        stranded_uavs=len(stranded),
                    )
                )
                for key, value in [
                    ("ue_rates", result["ue_rate_mbps"]),
                    ("positions", snap["uav_positions"]),
                    ("roles", snap["roles"]),
                    ("status", snap["status"]),
                    ("battery", env.battery),
                    ("actions", actions),
                ]:
                    arrays[key].append(np.asarray(value).copy())
            rates = np.asarray(arrays["ue_rates"])
            used = sum(r["depleted_points"] for r in rows)
            charged = sum(r["charged_points"] for r in rows)
            residual = float(initial + charged - used - float(env.battery.sum()))
            assert abs(residual) < 1e-3
            delivered_gbit = float(rates.sum() * env.cfg.slot_s / 1000)
            summary = dict(
                case=name,
                seed=seed,
                steps=args.steps,
                **{k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "slot"},
                energy_depleted_points=used,
                energy_charged_points=charged,
                initial_battery_points=initial,
                final_battery_points=float(env.battery.sum()),
                energy_balance_residual_points=residual,
                total_delivered_gbit=delivered_gbit,
                battery_points_per_gbit=used / delivered_gbit if delivered_gbit > 0 else 0.0,
                served_again_cycles=sum(c["outcome"] == "served_again" for c in cycles),
                rejoin_events=sum(
                    "rejoin_slot" in c and c["rejoin_slot"] is not None for c in cycles
                )
                + sum(c["rejoin_slot"] is not None for c in pending.values()),
                jain_time_average=jain(rates.mean(0)),
                max_flow_violation_mbps=max(r["flow_violation_mbps"] for r in rows),
                max_airtime=max(r["schedule_airtime"] for r in rows),
            )
            summaries.append(summary)
            prefix = f"{name}_seed{seed}"
            write_csv(args.output / (prefix + "_slots.csv"), rows)
            write_json(
                args.output / (prefix + "_cycles.json"),
                dict(completed=cycles, pending=list(pending.values())),
            )
            np.savez_compressed(
                args.output / (prefix + "_trace.npz"),
                **{k: np.asarray(v) for k, v in arrays.items()},
            )
            write_csv(args.output / "summary.csv", summaries)
            print(
                f"{prefix}: T={summary['throughput_mbps']:.2f}, energy={used:.2f}, cycles={summary['served_again_cycles']}",
                flush=True,
            )
    aggregate = []
    for name, *_ in CASES:
        rows = [r for r in summaries if r["case"] == name]
        aggregate.append(
            dict(
                case=name,
                seeds=len(rows),
                **{
                    k: float(np.mean([r[k] for r in rows]))
                    for k in rows[0]
                    if k not in {"case", "seed"}
                },
            )
        )
    cf_aggregate = []
    for name in ["split_2H", "full_2H"]:
        rows = [r for r in counterfactual if r["radio"] == name]
        cf_aggregate.append(
            dict(
                radio=name,
                frames=len(rows),
                **{
                    k: float(np.mean([r[k] for r in rows]))
                    for k in rows[0]
                    if k not in {"seed", "slot", "radio"}
                },
            )
        )
    write_csv(args.output / "aggregate.csv", aggregate)
    write_csv(args.output / "matched_AR_lifecycle_radio_counterfactual.csv", counterfactual)
    write_csv(args.output / "counterfactual_aggregate.csv", cf_aggregate)
    files = [
        Path(__file__),
        ROOT / "src/diagnostics/bs5_hybrid_full.py",
        ROOT / "src/diagnostics/bs5_hybrid_lifecycle.py",
        ROOT / "src/diagnostics/bs5_hybrid_fdm.py",
    ]
    write_json(
        args.output / "receipt.json",
        dict(
            protocol=protocol,
            slots_simulated=len(summaries) * args.steps,
            matched_radio_counterfactuals=len(counterfactual),
            wall_seconds=time.perf_counter() - total_started,
            source_sha256={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
            },
            scope="Scripted navigation, two development seeds, normalized energy; not learned policies, hardware Joules or a final benchmark",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--seeds", type=int, nargs="+", default=[910801, 910802])
    args = p.parse_args()
    if args.steps < 1:
        p.error("Positive horizon required")
    run(args)
