"""Standalone BS5 bundle analyzer/validator: Python stdlib only, no model/env.

This file is also copied into each export as analyze_bundle.py. Validation
reconstructs utility from every validation slot's per-UE rates.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import tempfile
import zipfile


ARMS = ("A-R", "A-N", "B-R", "B-N")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temp.replace(path)


def canonical_hash(data):
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def rate_utility(net, demand):
    rates = net["ue_rate_mbps"]
    assert len(rates) == len(demand) and rates, "rate/demand shape"
    assert all(math.isfinite(r) and 0 <= r <= d + 1e-8 for r, d in zip(rates, demand))
    utility = statistics.fmean(min(r / d, 1) for r, d in zip(rates, demand))
    assert abs(utility - net["utility"]) < 1e-10, "rate-network utility mismatch"
    return utility


def aggregate_episode(rows, slot_s=5.0):
    """Interpret service-phase traces; lifecycle events use post-step status."""
    utility, throughput, qos, active, relays, required_rates = [], [], [], [], [], []
    role_switches = 0
    movements = [0] * 5
    roles = [0, 0]
    events = {"Return": 0, "Charge": 0, "Rejoin": 0}
    drain, distances, empty_relays, traffic_relays = [], [], [], []
    route_gain, opportunity, net_gain, oracle_gap = [], [], [], []
    required_count, required_served, first_required = [], [], None
    per_ue_sum = [0.0] * len(rows[0]["service"]["ue_positions"])
    outage_slots, continuity_breaks, previous_served = 0, 0, None
    for t, row in enumerate(rows):
        snap, post = row["service"], row["post"]
        net, demands = snap["network"], snap["ue_demand_mbps"]
        rates = net["ue_rate_mbps"]
        slot_utility = rate_utility(net, demands)
        utility.append(slot_utility)
        throughput.append(sum(rates))
        qos.append(sum(r >= d - 1e-9 for r, d in zip(rates, demands)))
        active.append(sum(s == 0 for s in snap["status"]))
        relay_ids = [
            i for i, (s, r) in enumerate(zip(snap["status"], snap["roles"])) if s == 0 and r == 1
        ]
        relays.append(len(relay_ids))
        traffic = {
            int(r)
            for a, r in net["routes"].items()
            if r is not None
            and any(
                rates[u] > 1e-12
                for u in net["association"].get(str(a), net["association"].get(a, []))
            )
        }
        traffic_relays.append(len(traffic))
        empty_relays.append(len(relay_ids) - len(traffic))
        required = row["relay_required_ue"]
        required_count.append(sum(required))
        required_served.append(sum(flag and rates[u] > 1e-12 for u, flag in enumerate(required)))
        required_rates.append(sum(rates[u] for u, flag in enumerate(required) if flag))
        if first_required is None and any(row["uav_in_relay_required_access_region"]):
            first_required = t + 1
        for i, (before, after) in enumerate(zip(row["decision"]["status"], post["status"])):
            if before == 0 and after == 1:
                events["Return"] += 1
            elif before == 1 and after == 2:
                events["Charge"] += 1
            elif before == 2 and after == 0:
                events["Rejoin"] += 1
            if before == 0:
                roles[snap["roles"][i]] += 1
                movements[row["decoded_actions"][i][0]] += 1
                role_switches += row["decision"]["roles"][i] != snap["roles"][i]
        drain.append(
            sum(max(0.0, a - b) for a, b in zip(row["decision"]["battery"], post["battery"]))
        )
        distances.extend(row["nearest_station_steps"])
        for u, rate in enumerate(rates):
            per_ue_sum[u] += rate
        served = [r > 1e-12 for r in rates]
        outage_slots += sum(not x for x in served)
        if previous_served is not None:
            continuity_breaks += sum(a and not b for a, b in zip(previous_served, served))
        previous_served = served
        if "shadow" in row:
            sh = row["shadow"]
            direct = rate_utility(sh["same_access_direct"], demands)
            all_access = rate_utility(sh["all_access"], demands)
            route_gain.append(slot_utility - direct)
            opportunity.append(all_access - direct)
            net_gain.append(slot_utility - all_access)
            if "role_enumeration" in sh:
                oracle_gap.append(max(r["utility"] for r in sh["role_enumeration"]) - slot_utility)
    first = next((i + 1 for i, u in enumerate(utility) if u > 1e-12), None)
    n = len(rows)
    total, square = sum(per_ue_sum), sum(x * x for x in per_ue_sum)
    result = dict(
        slots=n,
        utility=statistics.fmean(utility),
        reward_sum=sum(utility),
        throughput_mbps=statistics.fmean(throughput),
        service_area_mbit=sum(throughput) * slot_s,
        mean_qos_ue=statistics.fmean(qos),
        first_service_slot=first,
        first_service_censored=first is None,
        zero_reward_slots=sum(u <= 1e-12 for u in utility),
        reward_density=sum(u > 1e-12 for u in utility) / n,
        active_uav_slots=sum(active),
        mean_relay_uavs=statistics.fmean(relays),
        relay_traffic_slots=sum(traffic_relays),
        relay_empty_slots=sum(empty_relays),
        role_switches=int(role_switches),
        movement_action_counts=movements,
        access_relay_active_slots=roles,
        lifecycle_events=events,
        battery_drain=sum(drain),
        mean_nearest_station_steps=statistics.fmean(distances),
        mean_relay_required_ue=statistics.fmean(required_count),
        mean_served_relay_required_ue=statistics.fmean(required_served),
        relay_required_throughput_mbps=statistics.fmean(required_rates),
        first_relay_region_slot=first_required,
        ue_outage_fraction=outage_slots / (n * len(per_ue_sum)),
        ue_service_breaks=continuity_breaks,
        per_ue_mean_rates_mbps=[x / n for x in per_ue_sum],
        jain_mean_rate=total * total / (len(per_ue_sum) * square) if square else 0.0,
        shadow_slots=len(route_gain),
        oracle_slots=len(oracle_gap),
    )
    for key, values in [
        ("route_gain", route_gain),
        ("role_opportunity_cost", opportunity),
        ("net_role_gain", net_gain),
        ("oracle_headroom", oracle_gap),
    ]:
        result[key] = statistics.fmean(values) if values else None
    return result


def validate_rates(row):
    snap = row["service"]
    assert (
        snap["failed_bs_idx"] == 5
        and row["service_phase"] == "after_movement_before_energy_and_ue_mobility"
    )
    demand, net = snap["ue_demand_mbps"], snap["network"]
    utility = rate_utility(net, demand)
    assert abs(utility - row["reward"]) < 1e-10, "rate-reward mismatch"
    users = [u for group in net["association"].values() for u in group]
    assert len(users) == len(set(users)), "double-counted association"
    for r, allocation in net["relay_allocations_mbps"].items():
        assert sum(allocation.values()) <= net["relay_gateway_caps_mbps"][str(r)] + 1e-8
        for aid, capacity in allocation.items():
            assert capacity <= net["access_relay_caps_mbps"][f"{aid}:{r}"] + 1e-8
    if "shadow" in row:
        sh = row["shadow"]
        assert sh["actual"]["ue_rate_mbps"] == net["ue_rate_mbps"], "shadow scalar parity"
        for key in ("actual", "same_access_direct", "all_access"):
            rate_utility(sh[key], demand)
        assert all(
            a + 1e-8 >= d
            for a, d in zip(net["ue_rate_mbps"], sh["same_access_direct"]["ue_rate_mbps"])
        ), "router below direct fallback"
    for action, mask in zip(row["raw_actions"], row["action_mask"]):
        assert mask[action], "masked action executed"
    if row["arm"].endswith("N"):
        assert all(r == 0 for r, s in zip(snap["roles"], snap["status"]) if s == 0)
        assert all(r is None for r in net["routes"].values())
    return utility


def legacy_shadow_metrics(rows, corrected):
    """Reproduce only the three diagnostics affected by the historical UE-index leak.

    This verifies old stored summaries without changing their evidence. It never
    supplies analysis values: those always come from aggregate_episode(rows).
    """
    route, net, oracle = [], [], []
    for row in rows:
        if "shadow" not in row:
            continue
        shadow = row["shadow"]
        demands = row["service"]["ue_demand_mbps"]
        last_ue_index = len(demands) - 1
        route.append(last_ue_index - rate_utility(shadow["same_access_direct"], demands))
        net.append(last_ue_index - rate_utility(shadow["all_access"], demands))
        if "role_enumeration" in shadow:
            oracle.append(max(r["utility"] for r in shadow["role_enumeration"]) - last_ue_index)
    result = dict(corrected)
    for key, values in (("route_gain", route), ("net_role_gain", net), ("oracle_headroom", oracle)):
        result[key] = statistics.fmean(values) if values else None
    return result


def validate(
    root, *, require_complete=False, check_manifest=True, allow_legacy_shadow_metrics=False
):
    root = Path(root)
    if check_manifest and (root / "SHA256SUMS.json").exists():
        sums = read_json(root / "SHA256SUMS.json")
        actual = {
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and p.name != "SHA256SUMS.json"
        }
        assert actual == set(sums), "manifest file set mismatch"
        for name, digest in sums.items():
            target = (root / name).resolve()
            assert target.is_relative_to(root.resolve()), "unsafe manifest path"
            assert file_hash(target) == digest, f"hash mismatch: {name}"
    protocol, registry = read_json(root / "protocol.json"), read_json(root / "run_registry.json")
    assert registry["protocol_sha256"] == canonical_hash(protocol), "protocol drift"
    expected = {f"{arm}_seed{seed}" for seed in protocol["training_seeds"] for arm in ARMS}
    assert set(registry["runs"]) == expected, "missing/extra run registry entry"
    if require_complete:
        assert registry["status"] == "complete", "experiment incomplete"
    exogenous, initial, run_rows, val_rows = {}, {}, [], []
    residual_pairs, train_exogenous, physics_hashes = {}, {}, set()
    total_updates = total_train = total_eval = legacy_packets = 0
    for name in sorted(expected):
        entry = registry["runs"][name]
        if registry["status"] == "complete":
            assert entry["status"] == "complete", "missing/incomplete run"
        directory = root / "runs" / name
        if entry["status"] == "not_started":
            assert not (directory / "commit.json").exists()
            continue
        commit = read_json(directory / "commit.json")
        meta = read_json(directory / "meta.json")
        arm, seed = meta["arm"], meta["seed"]
        assert name == f"{arm}_seed{seed}" and meta["source_sha256"] == protocol["source_sha256"]
        assert meta["config_sha256"] == canonical_hash(meta["resolved_config"])
        assert meta["resolved_config"]["failed_bs_idx"] == 5
        assert meta["resolved_config"]["failure_mode"] == "fixed"
        physics = dict(meta["resolved_config"])
        physics.pop("allow_relay")
        physics.pop("uav_start_cells")
        physics_hashes.add(canonical_hash(physics))
        assert len(physics_hashes) == 1, "arm-specific physics changed"
        assert (meta["state_dim"], meta["obs_dim"], meta["n_actions"]) == (111, 116, 10)
        if seed in initial:
            assert initial[seed] == meta["initial_weights_sha256"], "unpaired initial weights"
        initial[seed] = meta["initial_weights_sha256"]
        assert commit["steps"] == entry["steps"]
        if entry["status"] == "complete":
            assert commit["steps"] == protocol["steps_per_run"]
        elif require_complete:
            raise AssertionError(f"incomplete run: {name}")
        eps = sorted((directory / "episodes").glob("*.json"))
        eps = [p for p in eps if int(p.stem) < commit["episodes"]]
        assert len(eps) == commit["episodes"], "missing train episode"
        update_ids, train_slots = set(), 0
        schedule = protocol["scenario_schedule"][str(seed)]
        for eid, ep_path in enumerate(eps):
            ep = read_json(ep_path)
            assert ep["episode_id"] == eid and ep["scenario_seed"] == schedule[eid]["scenario_seed"]
            assert ep["residual_battery"] == schedule[eid]["residual_battery"]
            pair_key = seed, eid
            residual_pair = ep["residual_uav_id"], ep["residual_battery"]
            assert pair_key not in residual_pairs or residual_pairs[pair_key] == residual_pair, (
                "unpaired residual UAV"
            )
            residual_pairs[pair_key] = residual_pair
            assert (
                ep["completion_status"] == "complete"
                and ep["metrics"]["slots"] == protocol["horizon"]
            )
            assert (
                ep["source_sha256"] == protocol["source_sha256"]
                and ep["config_sha256"] == meta["config_sha256"]
            )
            train_slots += ep["metrics"]["slots"]
            updates = read_json(directory / "updates" / ep_path.name)
            for update in updates:
                assert update["step"] not in update_ids, "duplicate update on resume"
                update_ids.add(update["step"])
                assert not update["nonfinite"] and all(
                    math.isfinite(update[k])
                    for k in (
                        "loss",
                        "td_abs_mean",
                        "q_total_mean",
                        "grad_norm_before_clip",
                        "grad_norm_after_clip",
                        "lr",
                    )
                )
                assert update["grad_norm_after_clip"] <= 10.0001
            trace = directory / "train_traces" / f"{eid:06d}.json.gz"
            scheduled = (
                eid % protocol["train_trace_every_episodes"] == 0 or ep["first_scenario_encounter"]
            )
            assert trace.exists() == scheduled, "train trace coverage mismatch"
            if trace.exists():
                rows = load_trace(trace)["trace"]
                assert [r["slot"] for r in rows] == list(range(1, protocol["horizon"] + 1))
                for row in rows:
                    validate_rates(row)
                    key = seed, eid, row["slot"]
                    digest = canonical_hash(
                        [row["service"]["ue_positions"], row["service"]["ue_demand_mbps"]]
                    )
                    assert key not in train_exogenous or train_exogenous[key] == digest, (
                        "unpaired training UE/demand"
                    )
                    train_exogenous[key] = digest
        assert train_slots == commit["steps"], "train step accounting"
        expected_updates = set(
            range(protocol["update_every"], commit["steps"] + 1, protocol["update_every"])
        )
        expected_updates = {
            s for s in expected_updates if s >= max(protocol["warmup"], protocol["batch_size"])
        }
        assert update_ids == expected_updates and len(update_ids) == commit["updates"], (
            "missing/extra learner update"
        )
        evaluations = []
        for step in [s for s in protocol["evaluation_steps"] if s <= commit["steps"]]:
            ev = read_json(directory / "evaluations" / f"{step:06d}" / "summary.json")
            values = []
            for scenario in protocol["validation_seeds"]:
                packet = load_trace(
                    directory / "evaluations" / f"{step:06d}" / f"{scenario}.json.gz"
                )
                rows = packet["trace"]
                residual_key = "validation", scenario
                residual = packet["initial_residual"]
                assert (
                    residual_key not in residual_pairs or residual_pairs[residual_key] == residual
                ), "unpaired validation residual UAV"
                residual_pairs[residual_key] = residual
                assert len(rows) == protocol["horizon"], "missing validation slot"
                assert [r["slot"] for r in rows] == list(range(1, protocol["horizon"] + 1)), (
                    "duplicate/missing validation slots"
                )
                for row in rows:
                    assert (
                        row["arm"] == arm
                        and row["scenario_seed"] == scenario
                        and row["checkpoint_step"] == step
                    )
                    validate_rates(row)
                    assert "shadow" in row, "missing validation shadow scores"
                    expected_oracle = (row["slot"] - 1) % protocol["oracle_every_slots"] == 0
                    assert ("role_enumeration" in row["shadow"]) == expected_oracle, (
                        "oracle coverage mismatch"
                    )
                    key = scenario, row["slot"]
                    digest = canonical_hash(
                        [row["service"]["ue_positions"], row["service"]["ue_demand_mbps"]]
                    )
                    assert key not in exogenous or exogenous[key] == digest, (
                        "UE/demand exogenous mismatch"
                    )
                    exogenous[key] = digest
                result = aggregate_episode(rows, protocol["slot_s"])
                if canonical_hash(result) != canonical_hash(packet["metrics"]):
                    assert allow_legacy_shadow_metrics and canonical_hash(
                        legacy_shadow_metrics(rows, result)
                    ) == canonical_hash(packet["metrics"]), "trace-summary mismatch"
                    legacy_packets += 1
                values.append(result["utility"])
                val_rows.append(
                    dict(run=name, arm=arm, seed=seed, step=step, scenario=scenario, **result)
                )
                total_eval += len(rows)
            mean = statistics.fmean(values)
            assert abs(mean - ev["utility"]) < 1e-12
            evaluations.append(ev)
        assert evaluations and evaluations[0]["step"] == 0
        best = max(evaluations, key=lambda row: (row["utility"], -row["step"]))
        assert best["step"] == commit["best_step"], "checkpoint selection mismatch"
        total_train += train_slots
        total_updates += len(update_ids)
        run_rows.append(
            dict(
                run=name,
                arm=arm,
                seed=seed,
                steps=train_slots,
                best_step=best["step"],
                best_utility=best["utility"],
                last_utility=evaluations[-1]["utility"],
                evaluations=evaluations,
            )
        )
    result = dict(
        status=registry["status"],
        training_steps=total_train,
        learner_updates=total_updates,
        validation_slots=total_eval,
        validated_runs=len(run_rows),
        exogenous_hashes_matched=len(exogenous),
        legacy_shadow_metric_packets=legacy_packets,
        runs=run_rows,
        validation=val_rows,
    )
    return result


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    root,
    output=None,
    *,
    require_complete=False,
    check_manifest=True,
    allow_legacy_shadow_metrics=False,
):
    root = Path(root)
    output = Path(output) if output else root / "analysis"
    result = validate(
        root,
        require_complete=require_complete,
        check_manifest=check_manifest,
        allow_legacy_shadow_metrics=allow_legacy_shadow_metrics,
    )
    output.mkdir(parents=True, exist_ok=True)
    rows = result.pop("validation")
    effects, paired = [], []
    protocol = read_json(root / "protocol.json")
    lookup = {(r["arm"], r["seed"], r["step"], r["scenario"]): r for r in rows}
    run_lookup = {(r["arm"], r["seed"]): r for r in result["runs"]}
    contrasts = {
        "launch_with_relay": {"B-R": 1, "A-R": -1},
        "launch_no_relay": {"B-N": 1, "A-N": -1},
        "relay_at_A": {"A-R": 1, "A-N": -1},
        "relay_at_B": {"B-R": 1, "B-N": -1},
        "interaction": {"B-R": 1, "B-N": -1, "A-R": -1, "A-N": 1},
    }
    for seed in protocol["training_seeds"]:
        if any((arm, seed) not in run_lookup for arm in ARMS):
            continue
        common = min(run_lookup[arm, seed]["steps"] for arm in ARMS)
        for selection in ("last_common", "best"):
            selected = {
                arm: (
                    max(s for s in protocol["evaluation_steps"] if s <= common)
                    if selection == "last_common"
                    else run_lookup[arm, seed]["best_step"]
                )
                for arm in ARMS
            }
            for contrast, weights in contrasts.items():
                for metric in (
                    "utility",
                    "throughput_mbps",
                    "mean_qos_ue",
                    "relay_required_throughput_mbps",
                ):
                    diffs = []
                    for scenario in protocol["validation_seeds"]:
                        delta = sum(
                            weight * lookup[arm, seed, selected[arm], scenario][metric]
                            for arm, weight in weights.items()
                        )
                        diffs.append(delta)
                        paired.append(
                            dict(
                                seed=seed,
                                selection=selection,
                                contrast=contrast,
                                metric=metric,
                                scenario=scenario,
                                difference=delta,
                            )
                        )
                    effects.append(
                        dict(
                            seed=seed,
                            selection=selection,
                            contrast=contrast,
                            metric=metric,
                            difference=statistics.fmean(diffs),
                            scenario_sd=statistics.stdev(diffs) if len(diffs) > 1 else 0,
                        )
                    )
    write_csv(
        output / "runs.csv",
        result["runs"],
        ["run", "steps", "best_step", "best_utility", "last_utility"],
    )
    write_csv(
        output / "effects.csv",
        effects,
        ["seed", "selection", "contrast", "metric", "difference", "scenario_sd"],
    )
    write_csv(
        output / "paired_scenarios.csv",
        paired,
        ["seed", "selection", "contrast", "metric", "scenario", "difference"],
    )
    write_csv(output / "validation_episodes.csv", rows, list(rows[0]) if rows else ["run"])
    result.update(
        effects=effects,
        caution="Validation pilot; two training seeds; no slot-level CI; no final thesis test.",
        ratio_policy="Absolute differences reported. Ratios omitted, including near-zero baselines.",
    )
    atomic_json(output / "summary.json", result)
    # Portable SVG: no matplotlib, torch or environment imports.
    colors = {"A-R": "#2563eb", "A-N": "#94a3b8", "B-R": "#dc2626", "B-N": "#f59e0b"}
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="430" viewBox="0 0 900 430">',
        '<rect width="900" height="430" fill="white"/><path d="M65 30V350H855" fill="none" stroke="black"/>',
        '<text x="65" y="20">Validation utility (0–1); training from step 0</text>',
    ]
    for y in (0, 0.25, 0.5, 0.75, 1):
        sy = 350 - y * 310
        svg.append(f'<text x="20" y="{sy + 4}">{y:g}</text><path d="M65 {sy}H855" stroke="#eee"/>')
    for run in result["runs"]:
        points = " ".join(
            f"{65 + e['step'] / protocol['steps_per_run'] * 790},{350 - e['utility'] * 310}"
            for e in run["evaluations"]
        )
        dash = ' stroke-dasharray="5 4"' if run["seed"] == protocol["training_seeds"][-1] else ""
        svg.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[run["arm"]]}" stroke-width="2"{dash}/>'
        )
    for i, arm in enumerate(ARMS):
        svg.append(
            f'<text x="{80 + i * 175}" y="390" fill="{colors[arm]}">{html.escape(arm)}</text>'
        )
    svg.append(f'<text x="650" y="420">Steps: 0 → {protocol["steps_per_run"]}</text></svg>')
    (output / "learning_curves.svg").write_text("".join(svg), encoding="utf-8")
    lines = [
        "# BS5 launch A/B validation pilot",
        "",
        f"Status: {result['status']}. Training steps: {result['training_steps']}; updates: {result['learner_updates']}.",
        "",
        "| Run | Steps | Best step | Best utility | Last utility |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["runs"]:
        lines.append(
            f"| {row['run']} | {row['steps']} | {row['best_step']} | {row['best_utility']:.6f} | {row['last_utility']:.6f} |"
        )
    lines += [
        "",
        "See effects.csv and paired_scenarios.csv for the five registered contrasts, separately by training seed.",
        "Service onset is censored when no service occurs. Geometric bounds are not observed onset times.",
        "Shadow role scores are instantaneous counterfactuals, not independent no-relay training.",
        "Two seeds support pilot observations only. No final test or statistical significance claim.",
    ]
    if result["legacy_shadow_metric_packets"]:
        lines += [
            "",
            f"Recomputed shadow diagnostics in {result['legacy_shadow_metric_packets']} historical validation packets. "
            "Stored route_gain/net_role_gain/oracle_headroom matched the historical UE-index bug exactly; "
            "all other stored metrics were checked unchanged. Original evidence was not modified.",
        ]
    if result["status"] != "complete":
        lines.append(
            "Incomplete experiment: do not compare arms at unequal budgets as final effects."
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--allow-legacy-shadow-metrics",
        action="store_true",
        help="verify historical UE-index diagnostics and recompute them from immutable traces",
    )
    args = parser.parse_args()
    if args.bundle:
        with tempfile.TemporaryDirectory(prefix="bs5_offline_") as tmp:
            with zipfile.ZipFile(args.bundle) as archive:
                for info in archive.infolist():
                    target = (Path(tmp) / info.filename).resolve()
                    if not target.is_relative_to(Path(tmp).resolve()):
                        raise ValueError("unsafe ZIP path")
                archive.extractall(tmp)
            result = analyze(
                tmp,
                args.output,
                require_complete=args.require_complete,
                allow_legacy_shadow_metrics=args.allow_legacy_shadow_metrics,
            )
    else:
        if args.root is None:
            parser.error("--root or --bundle is required")
        result = analyze(
            args.root,
            args.output,
            require_complete=args.require_complete,
            allow_legacy_shadow_metrics=args.allow_legacy_shadow_metrics,
        )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "training_steps", "learner_updates", "validation_slots")
            }
        )
    )


if __name__ == "__main__":
    main()
