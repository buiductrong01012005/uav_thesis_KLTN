"""Offline continuation audit/curves/export: stdlib only; no env, Torch or weights.

Usage: python -B analyze_bs5_curriculum_extend.py ROOT --output OUT [--require-complete]
Input is an extracted bundle/durable result directory. OUT must be outside ROOT.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import zipfile

try:
    from bs5_bundle import atomic_json, canonical_hash, file_hash, load_trace, read_json, write_csv
    from analyze_bs5_curriculum_bundle import check_rows, episode_metrics, equivalent
    from bs5_curve_plot import plot
except ImportError:
    from scripts.analysis.bs5_bundle import (
        atomic_json,
        canonical_hash,
        file_hash,
        load_trace,
        read_json,
        write_csv,
    )
    from scripts.analysis.analyze_bs5_curriculum_bundle import (
        check_rows,
        episode_metrics,
        equivalent,
    )
    from scripts.analysis.bs5_curve_plot import plot

ARMS = ("C-R", "C-N")
SCALARS = (
    "utility",
    "throughput_mbps",
    "mean_positive_service_ue",
    "mean_qos_ue",
    "jain_mean_rate",
    "mean_served_relay_required_ue",
    "relay_required_throughput_mbps",
    "net_role_gain",
    "route_gain",
    "role_opportunity_cost",
    "role_switches",
    "active_uav_slots",
    "relay_empty_slots",
    "relay_active_uav_slots",
)


def safe(root, rel):
    path = (root / rel).resolve()
    assert path.is_relative_to(root.resolve()), f"unsafe path: {rel}"
    return path


def verify_manifest(root):
    path = root / "SHA256SUMS.json"
    if path.exists():
        hashes = read_json(path)
        actual = {
            f.relative_to(root).as_posix() for f in root.rglob("*") if f.is_file() and f != path
        }
        assert actual == set(hashes), "manifest file set mismatch"
        for rel, digest in hashes.items():
            assert file_hash(safe(root, rel)) == digest, f"hash mismatch: {rel}"


def gateway_auditor(config, *, require_exclusion_flag=True):
    """Independent stdlib check of gateway capacities; BS5 is never a candidate.

    Mirrors the locked radio equations as an audit, not an environment import.
    Cache depends only on immutable config and service-phase cell geometry.
    """
    cache = {}

    def capacity(cell):
        key = tuple(cell)
        if key in cache:
            return cache[key]
        row, col = cell
        xy = ((col + 0.5) * config["cell_size_m"], (row + 0.5) * config["cell_size_m"])
        values = []
        for i, bs in enumerate(config["bs_xy_m"]):
            if i == 5:
                continue
            distance = math.hypot(xy[0] - bs[0], xy[1] - bs[1])
            if distance > config["backhaul_radius_cells"] * config["cell_size_m"] + 1e-9:
                continue
            dh = abs(config["uav_altitude_m"] - config["bs_height_m"])
            d3 = max(math.sqrt(distance * distance + dh * dh), 1)
            theta = math.degrees(math.atan2(dh, max(distance, 1e-9)))
            los = 1 / (
                1 + config["urban_a"] * math.exp(-config["urban_b"] * (theta - config["urban_a"]))
            )
            pl = 20 * math.log10(d3) + 20 * math.log10(config["backhaul_fc_hz"]) - 147.55
            pl += los * config["eta_los_db"] + (1 - los) * config["eta_nlos_db"]
            noise = (
                config["noise_psd_dbm_hz"]
                + 10 * math.log10(config["backhaul_bw_hz"])
                + config["noise_figure_db"]
            )
            values.append(
                config["backhaul_bw_hz"]
                * math.log2(1 + 10 ** ((config["backhaul_tx_dbm"] - pl - noise) / 10))
                / 1e6
            )
        cache[key] = max(values, default=0)
        return cache[key]

    def check(row):
        snap = row["service"]
        if require_exclusion_flag or "failed_bs_gateway_excluded" in row:
            assert row["failed_bs_gateway_excluded"] is True
        networks = [snap["network"]] + [
            row["shadow"][k]
            for k in ("actual", "same_access_direct", "all_access")
            if k in row.get("shadow", {})
        ]
        for net in networks:
            for key in ("direct_caps_mbps", "relay_gateway_caps_mbps"):
                for aid, value in net[key].items():
                    assert math.isclose(
                        value,
                        capacity(snap["uav_positions"][int(aid)]),
                        rel_tol=1e-10,
                        abs_tol=1e-8,
                    ), "gateway capacity mismatch / failed BS leak"
            for aid, users in net["association"].items():
                total = sum(net["ue_rate_mbps"][u] for u in users)
                assert total <= net["backhaul_caps_mbps"][str(aid)] + 1e-8, (
                    "delivered rate exceeds backhaul"
                )

    return check


def rows_at(directory, p, arm, seed, step, *, require_raw, cached=None, check_gateway=None):
    location = directory / "evaluations" / f"{step:06d}"
    expected = {f"{scenario}.json.gz" for scenario in p["validation_pool"]}
    actual = {f.name for f in location.glob("*.json.gz")}
    assert not actual or actual == expected, f"missing/duplicate eval packet: {location}"
    rows = []
    for scenario in p["validation_pool"]:
        path = location / f"{scenario}.json.gz"
        if not path.exists():
            assert not require_raw and cached is not None, f"missing eval packet: {path}"
            key = (arm, seed, step, scenario)
            assert key in cached, f"missing cached parent row: {key}"
            rows.append(cached[key])
            continue
        packet = load_trace(path)
        trace = packet["trace"]
        check_rows(trace, p, arm, scenario, evaluation=True)
        if check_gateway:
            for row in trace:
                check_gateway(row)
        assert all(r["seed"] == seed and r["checkpoint_step"] == step for r in trace)
        assert packet["condition"]["starts"] == "B" and packet["condition"]["mobility"]
        assert packet["condition"]["battery"] and packet["condition"]["active_count"] == 5
        metrics = episode_metrics(trace)
        assert equivalent(metrics, packet["metrics"]), "raw/metric mismatch"
        if arm == "C-N":
            assert abs(metrics["route_gain"]) < 1e-10 and abs(metrics["net_role_gain"]) < 1e-10
            assert metrics["relay_active_uav_slots"] == 0
        rows.append(
            dict(arm=arm, seed=seed, step=step, scenario=scenario, **metrics, raw_validated=True)
        )
    if (location / "summary.json").exists():
        summary = read_json(location / "summary.json")
        assert equivalent(statistics.fmean(r["utility"] for r in rows), summary["utility"])
    return rows


def cached_parent(root):
    path = root / "parents/analysis/validation_episodes.csv"
    result = {}
    if not path.exists():
        return result
    with path.open(encoding="utf-8", newline="") as stream:
        for original in csv.DictReader(stream):
            if original["arm"] not in ARMS:
                continue
            row = dict(original, raw_validated=False)
            for key in ("seed", "step", "scenario"):
                row[key] = int(row[key])
            for key in SCALARS + ("first_service_slot", "empty_relay_fraction"):
                if key in row:
                    row[key] = float(row[key]) if row[key] not in ("", "None", "null") else None
            row["first_service_censored"] = row.get("first_service_censored") in (
                "True",
                "true",
                "1",
            )
            k = tuple(row[k] for k in ("arm", "seed", "step", "scenario"))
            assert k not in result, "duplicate cached parent row"
            result[k] = row
    return result


def aggregate(rows):
    result = {
        k: statistics.fmean(r[k] for r in rows)
        for k in SCALARS
        if all(k in r and r[k] is not None for r in rows)
    }
    denominator = sum(r["relay_active_uav_slots"] for r in rows)
    result["empty_relay_fraction"] = (
        sum(r["relay_empty_slots"] for r in rows) / denominator if denominator else None
    )
    first = [r["first_service_slot"] for r in rows if r.get("first_service_slot") is not None]
    result["first_service_uncensored_mean_slot"] = statistics.fmean(first) if first else None
    result["first_service_censored_episodes"] = sum(
        r.get("first_service_censored", False) for r in rows
    )
    result["validation_episodes"] = len(rows)
    return result


def auc(points, start, end):
    selected = [(s, v) for s, v in points if start <= s <= end]
    if len(selected) < 2 or selected[0][0] != start or selected[-1][0] != end:
        return None
    return sum((b - a) * (u + v) / 2 for (a, u), (b, v) in zip(selected, selected[1:])) / (
        end - start
    )


def validate(root, *, require_complete=False):
    root = Path(root).resolve()
    verify_manifest(root)
    p = read_json(root / "protocol.json")
    policy = p["continuation"]
    old = read_json(root / "parents/protocol.json")
    refs = read_json(root / "parent_refs.json")
    identity = read_json(root / "source_identity.json")
    registry = read_json(root / "run_registry.json")
    assert canonical_hash(identity["files"]) == identity["sha256"] == p["source_sha256"]
    assert registry["protocol_sha256"] == canonical_hash(p), "protocol drift"
    assert canonical_hash(old) == p["parent_protocol_sha256"] == refs["parent_protocol_sha256"]
    assert old["source_sha256"] == refs["parent_source_sha256"] == policy["parent_source_sha256"]
    assert refs["current_source_sha256"] == p["source_sha256"]
    expected_runs = {f"{a}_seed{s}" for s in policy["training_seeds"] for a in ARMS}
    assert set(registry["runs"]) == expected_runs
    assert p["arms"] == list(ARMS) and p["training_seeds"] == policy["training_seeds"]
    start, end, horizon = policy["parent_steps"], policy["total_steps"], p["horizon"]
    assert p["stage_ends"] == old["stage_ends"][:-1] + [end]
    assert p["condition_boundaries"] == old["condition_boundaries"]
    assert p["common_evaluation_steps"] == list(range(0, end + 1, p["eval_every"]))
    for seed in p["training_seeds"]:
        schedule = p["scenario_schedule"][str(seed)]
        assert schedule[: start // horizon] == old["scenario_schedule"][str(seed)], (
            "frozen prefix changed"
        )
        assert len(schedule) == end // horizon
    for rel, item in refs["files"].items():
        path = safe(root / "parents", rel)
        if path.exists():
            assert file_hash(path) == item["sha256"], f"parent artifact changed: {rel}"
        else:
            assert rel.endswith(".pt"), f"missing parent evidence: {rel}"
    for identity_path, snapshot_path in (
        (root / "source_identity.json", root / "source_snapshot.zip"),
        (root / "parents/source_identity.json", root / "parents/source_snapshot.zip"),
    ):
        source = read_json(identity_path)
        assert source["sha256"] == canonical_hash(source["files"])
        with zipfile.ZipFile(snapshot_path) as z:
            import hashlib

            for rel, digest in source["files"].items():
                assert hashlib.sha256(z.read("uav_thesis/" + rel)).hexdigest() == digest
    cached = cached_parent(root)
    curves, summaries, validation, coverage = [], [], [], {}
    paired_exogenous = {}
    for name in sorted(expected_runs):
        d = root / "runs" / name
        parent = root / "parents/runs" / name
        c, meta = read_json(d / "commit.json"), read_json(d / "meta.json")
        original, pm = read_json(parent / "commit.json"), read_json(parent / "meta.json")
        check_gateway = gateway_auditor(meta["resolved_config"])
        arm, seed = meta["arm"], meta["seed"]
        assert name == f"{arm}_seed{seed}"
        assert meta["source_sha256"] == p["source_sha256"]
        assert meta["parent_meta_sha256"] == file_hash(parent / "meta.json")
        assert (
            meta["config_sha256"] == pm["config_sha256"] == canonical_hash(meta["resolved_config"])
        )
        assert c["steps"] == registry["runs"][name]["steps"] and start <= c["steps"] <= end
        if require_complete:
            assert c["steps"] == end and registry["runs"][name]["status"] == "complete"
        assert c["episodes"] == c["steps"] // horizon
        assert c["lineage"] == original["lineage"] and c["current_task"] == original["current_task"]
        index = c.get("extension_packet_index", {})
        actual = {
            f.relative_to(d).as_posix()
            for sub in ("episodes", "updates", "events", "train_traces", "evaluations")
            for f in (d / sub).rglob("*")
            if f.is_file()
        }
        assert actual == set(index), "missing/extra/uncommitted tail packet; resume before analyze"
        for rel, item in index.items():
            path = safe(d, rel)
            assert file_hash(path) == item["sha256"] and path.stat().st_size == item["bytes"]
        episode_ids = list(range(start // horizon, c["episodes"]))
        assert sorted(int(f.stem) for f in (d / "episodes").glob("*.json")) == episode_ids
        update_count = original["updates"]
        # Use actual parent replay receipt rather than deriving diagnostic clear schedules.
        replay_size = read_json(root / "preflight.json")["runs"][name]["replay_size"]
        for eid in episode_ids:
            ep = read_json(d / "episodes" / f"{eid:06d}.json")
            assert ep["episode_id"] == eid and ep["start_step"] == eid * horizon
            assert ep["end_step"] == (eid + 1) * horizon and ep["lineage"] == c["lineage"]
            assert ep["stage"] == ep["substage"] == "S4"
            assert (
                ep["source_sha256"] == p["source_sha256"]
                and ep["config_sha256"] == meta["config_sha256"]
            )
            spec = p["scenario_schedule"][str(seed)][eid]
            assert (
                ep["scenario_seed"] == spec["scenario_seed"]
                and ep["assignment_seed"] == spec["assignment_seed"]
            )
            assert ep["residual"]["energy"] == spec["residual_battery"]
            events = read_json(d / "events" / f"{eid:06d}.json")
            assert all(e.get("replay_clear") is False for e in events), "tail replay clear"
            updates = read_json(d / "updates" / f"{eid:06d}.json")
            expected = [
                s
                for s in range(eid * horizon + 1, (eid + 1) * horizon + 1)
                if s % p["update_every"] == 0
            ]
            assert [u["step"] for u in updates] == expected, "missing/duplicate update"
            for u in updates:
                update_count += 1
                assert u["update"] == update_count and u["lineage"] == c["lineage"]
                assert equivalent(u["epsilon"], p["epsilon_end"])
                assert u["target_sync"] == (u["step"] % p["target_every"] == 0)
                assert u["lr"] == p["lr"] and not u["nonfinite"]
                assert all(math.isfinite(u[k]) for k in ("loss", "q_total_mean", "target_q_mean"))
                assert u["grad_norm_after_clip"] <= 10.0001
                assert u["replay_size"] == min(
                    p["replay_capacity"], replay_size + u["step"] - ep["start_step"]
                )
                assert sum(u["replay_stage_composition"].values()) == u["replay_size"]
            replay_size = min(p["replay_capacity"], replay_size + horizon)
            trace = d / "train_traces" / f"{eid:06d}.json.gz"
            assert trace.exists() == bool(ep["capture_reason"])
            if eid == start // horizon:
                assert "first_after_resume" in ep["capture_reason"]
            if trace.exists():
                packet = load_trace(trace)
                check_rows(packet["trace"], p, arm, spec["scenario_seed"])
                for row in packet["trace"]:
                    check_gateway(row)
                assert equivalent(episode_metrics(packet["trace"]), ep["metrics"])
                for row in packet["trace"]:
                    k = (seed, eid, row["slot"])
                    assert (
                        k not in paired_exogenous or paired_exogenous[k] == row["exogenous_sha256"]
                    )
                    paired_exogenous[k] = row["exogenous_sha256"]
        assert update_count == c["updates"]
        if c["steps"] == end:
            assert c["updates"] == policy["final_updates"]
        assert c["stage_budgets"] == dict(
            original["stage_budgets"], S4=original["stage_budgets"]["S4"] + c["steps"] - start
        )
        assert c["substage_budgets"] == dict(
            original["substage_budgets"], S4=original["substage_budgets"]["S4"] + c["steps"] - start
        )
        assert c["stage_evaluations"] == original["stage_evaluations"]
        expected_steps = [s for s in p["common_evaluation_steps"] if s <= c["steps"]]
        tail_steps = [s for s in expected_steps if s > start]
        assert sorted(int(f.name) for f in (d / "evaluations").glob("*")) == tail_steps
        historical = [
            e for e in original["evaluations"] if e["step"] in old["common_evaluation_steps"]
        ]
        assert [e["step"] for e in historical] == old["common_evaluation_steps"]
        assert c["evaluations"][: len(original["evaluations"])] == original["evaluations"]
        assert [e["step"] for e in c["evaluations"][len(original["evaluations"]) :]] == tail_steps
        run_curves = []
        raw_count = 0
        for step in expected_steps:
            rows = rows_at(
                parent if step <= start else d,
                p,
                arm,
                seed,
                step,
                require_raw=step > start,
                cached=cached,
                check_gateway=check_gateway,
            )
            raw_count += sum(r["raw_validated"] for r in rows) if step <= start else 0
            summary = aggregate(rows)
            ev = next(e for e in c["evaluations"] if e["step"] == step)
            assert equivalent(summary["utility"], ev["utility"]), "commit/eval mismatch"
            assert abs(summary["throughput_mbps"] - 360 * summary["utility"]) < 1e-8
            row = dict(run=name, arm=arm, seed=seed, step=step, **summary)
            run_curves.append(row)
            validation.extend(rows)
        coverage[name] = dict(
            parent_raw_validated_episodes=raw_count,
            parent_expected_episodes=len(old["common_evaluation_steps"])
            * len(p["validation_pool"]),
            tail_raw_validated_episodes=len(tail_steps) * len(p["validation_pool"]),
        )
        best = max(run_curves, key=lambda r: (r["utility"], -r["step"]))
        after = [r for r in run_curves if r["step"] > start]
        best_after = max(after, key=lambda r: (r["utility"], -r["step"]), default=None)
        assert best["step"] == c["best_step"] and equivalent(best["utility"], c["best_utility"])
        last = run_curves[-1]
        complete = c["steps"] == end
        tail3 = run_curves[-3:] if complete else []
        points = [(r["step"], r["utility"]) for r in run_curves]
        summaries.append(
            dict(
                run=name,
                arm=arm,
                seed=seed,
                steps=c["steps"],
                updates=c["updates"],
                imported_steps=start,
                new_durable_steps=c["steps"] - start,
                last_observed=last,
                last60k=last if complete else None,
                tail_mean=statistics.fmean(r["utility"] for r in tail3) if tail3 else None,
                tail_steps=[r["step"] for r in tail3],
                best_common=best,
                best_after_parent=best_after,
                peak_to_last_drop=best["utility"] - last["utility"],
                normalized_auc_0_60k=auc(points, 0, end),
                normalized_auc_24_60k=auc(points, start, end),
                snapshots={
                    str(s): next((r for r in run_curves if r["step"] == s), None)
                    for s in (start, 36000, 48000, end)
                },
            )
        )
        curves.extend(run_curves)
    return dict(
        protocol=p,
        registry=registry,
        curves=curves,
        runs=summaries,
        coverage=coverage,
        validation=validation,
        main_training_seeds=len(p["training_seeds"]),
    )


def analyze(root, output, *, require_complete=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    assert not output.is_relative_to(root), "output must be outside immutable evidence"
    result = validate(root, require_complete=require_complete)
    output.mkdir(parents=True, exist_ok=True)
    curves = result["curves"]
    p = result["protocol"]
    fresh = p["continuation"].get("origin") == "fresh_0_to_60k_v1"
    result["origin"] = "fresh_0_to_60k_v1" if fresh else "historical_full_state_continuation"
    atomic_json(output / "curves.json", curves)
    write_csv(output / "curves.csv", curves, list(curves[0]) if curves else ["run"])
    atomic_json(output / "run_summary.json", result["runs"])
    lookup = {(r["arm"], r["seed"], r["step"]): r for r in curves}
    paired = []
    means = []
    for step in p["common_evaluation_steps"]:
        point = []
        for seed in p["training_seeds"]:
            r, n = lookup.get(("C-R", seed, step)), lookup.get(("C-N", seed, step))
            if r and n:
                point.append(
                    dict(
                        seed=seed,
                        step=step,
                        relay_utility=r["utility"],
                        no_relay_denominator=n["utility"],
                        utility_difference=r["utility"] - n["utility"],
                        throughput_gain_mbps=360 * (r["utility"] - n["utility"]),
                        utility_ratio=r["utility"] / n["utility"] if n["utility"] else None,
                    )
                )
        paired.extend(point)
        if len(point) == len(p["training_seeds"]):
            rm = statistics.fmean(r["relay_utility"] for r in point)
            nm = statistics.fmean(r["no_relay_denominator"] for r in point)
            means.append(
                dict(
                    step=step,
                    relay_mean=rm,
                    no_relay_mean_denominator=nm,
                    mean_paired_utility_difference=rm - nm,
                    mean_throughput_gain_mbps=360 * (rm - nm),
                    ratio_of_means=rm / nm if nm else None,
                    mean_paired_ratio=statistics.fmean(r["utility_ratio"] for r in point)
                    if all(r["utility_ratio"] is not None for r in point)
                    else None,
                    n_training_seeds=len(point),
                )
            )
    atomic_json(output / "paired.json", dict(per_seed=paired, means=means))
    write_csv(output / "paired.csv", paired, list(paired[0]) if paired else ["seed"])
    markers = [(s, label) for s, label in zip(p["stage_ends"][:3], ["S2", "S3", "S4"])]
    markers.append((p["continuation"]["parent_steps"], "FRESH PREFIX" if fresh else "PARENT"))
    colors = {"C-R": "#0072b2", "C-N": "#d55e00"}
    for key, label in (
        ("utility", "UTILITY"),
        ("throughput_mbps", "THROUGHPUT MBPS"),
        ("mean_positive_service_ue", "UE RATE > 1E-12"),
        ("mean_qos_ue", "UE RATE >= DEMAND - 1E-9"),
        ("empty_relay_fraction", "EMPTY RELAY / ACTIVE RELAY UAV SLOTS"),
        ("net_role_gain", "SAME GEOMETRY NET ROLE GAIN"),
    ):
        series, bands = [], []
        for arm in ARMS:
            grouped = []
            for i, seed in enumerate(p["training_seeds"]):
                points = [
                    (r["step"], r.get(key)) for r in curves if r["arm"] == arm and r["seed"] == seed
                ]
                series.append(
                    dict(
                        label=f"{arm} SEED {seed}",
                        color=colors[arm],
                        dashed=i == 1,
                        points=points,
                        width=1,
                    )
                )
            for step in p["common_evaluation_steps"]:
                vals = [lookup.get((arm, s, step), {}).get(key) for s in p["training_seeds"]]
                if all(v is not None for v in vals):
                    grouped.append(
                        (
                            step,
                            statistics.fmean(vals),
                            statistics.stdev(vals) if len(vals) > 1 else 0,
                        )
                    )
            series.append(
                dict(
                    label=f"{arm} MEAN",
                    color=colors[arm],
                    width=3,
                    points=[(s, m) for s, m, _ in grouped],
                )
            )
            bands.append(dict(color=colors[arm], points=[(s, m - d, m + d) for s, m, d in grouped]))
        plot(
            output / key,
            series,
            title=f"C-R / C-N {label}",
            ylabel=label,
            end=p["steps_per_run"],
            markers=markers,
            band=bands,
        )
    for key, label in (
        ("throughput_gain_mbps", "PAIRED RELAY GAIN MBPS"),
        ("utility_ratio", "RATIO C-R / C-N UTILITY"),
    ):
        series = [
            dict(
                label=f"PAIRED SEED {seed}",
                color=("#0072b2" if i == 0 else "#d55e00"),
                dashed=i == 1,
                points=[(r["step"], r[key]) for r in paired if r["seed"] == seed],
            )
            for i, seed in enumerate(p["training_seeds"])
        ]
        mean_key = (
            "mean_throughput_gain_mbps" if key == "throughput_gain_mbps" else "ratio_of_means"
        )
        series.append(
            dict(
                label="MEAN GAIN" if key == "throughput_gain_mbps" else "RATIO OF MEANS",
                color="#333333",
                width=3,
                points=[(r["step"], r[mean_key]) for r in means],
            )
        )
        plot(
            output / key, series, title=label, ylabel=label, end=p["steps_per_run"], markers=markers
        )
    result.pop("validation")
    result["budget_ledger"] = read_json(root / "budget_ledger.json")
    result["timing"] = [read_json(f) for f in sorted((root / "timing").glob("*.json"))]
    atomic_json(
        output / "verification.json",
        dict(
            accepted=True,
            coverage=result["coverage"],
            checkpoints_loaded=False,
            torch_or_env_required=False,
            require_complete=require_complete,
        ),
    )
    atomic_json(output / "summary.json", result)
    complete = all(r["steps"] == p["steps_per_run"] for r in result["runs"])
    lines = [
        "# BS5 fresh 0→60k — báo cáo" if fresh else "# BS5 extension — báo cáo",
        "",
        f"Trạng thái: {'đủ budget' if complete else 'chưa đủ budget'}. Không tự chạy quá 60k.",
        "Mean ±1 SD giữa hai training seeds (n=2), không phải 95% CI. Validation đã được xem; đây là pilot.",
        "Throughput = 360 × utility; hai số này không phải bằng chứng độc lập.",
        "Thí nghiệm mới từ0; không ghép kết quả/weights24k lịch sử. Gates chỉ chẩn đoán, kết quả nằm trong fresh prefix logs."
        if fresh
        else "S1 lịch sử chưa đạt role gate; không diễn giải lại là đã học vững mọi stage.",
        "",
    ]
    for r in result["runs"]:
        base = r["snapshots"][str(p["continuation"]["parent_steps"])]
        last = r["last_observed"]
        lines.append(
            f"- {r['run']}: {r['steps']} steps, {r['updates']} updates; utility {base['utility']:.6f} → {last['utility']:.6f} (Δ {last['utility'] - base['utility']:+.6f}); peak→last drop {r['peak_to_last_drop']:.6f}; tail={r['tail_mean']}."
        )
        lines.append(
            f"  Positive-service UE {base.get('mean_positive_service_ue')} → {last.get('mean_positive_service_ue')}; full-QoS UE {base.get('mean_qos_ue')} → {last.get('mean_qos_ue')}; Jain {last.get('jain_mean_rate')}; Relay rỗng {last.get('empty_relay_fraction')}."
        )
    if means:
        m = means[-1]
        lines += [
            "",
            f"Mốc chung cuối {m['step']}: C-R {m['relay_mean']:.6f}, C-N {m['no_relay_mean_denominator']:.6f}; chênh utility {m['mean_paired_utility_difference']:.6f}; ratio của means {m['ratio_of_means']}.",
        ]
    lines += [
        "",
        "Coverage raw parent/tail được công bố trong verification.json. Khi dùng cached parent tables, chưa tái audit đầy đủ raw 0–24k.",
        "QoS và positive service tách riêng trong curves; first-service có censoring, Jain tính trên per-UE episode-mean.",
        "Runtime extension nằm trong timing/ và budget_ledger.json, tách steps lịch sử. Không kết luận hội tụ với n=2.",
    ]
    budget = result["budget_ledger"]
    if fresh:
        lines.append(
            f"Fresh compute toàn lineage: {budget['whole_fresh_executed_steps']} steps thực thi mới, {budget['whole_fresh_durable_steps']} durable; historical imported=0. {budget['imported_steps']} prefix steps chỉ là chuyển state nội bộ sang tail, đã tính là compute mới."
        )
    lines += [
        f"Imported {budget['imported_steps']} steps; thực thi mới {budget['new_executed_steps']}; durable mới {budget['durable_new_steps']}; rollback {budget['rollback_steps']}; uncertainty upper bound {budget.get('uncertain_unflushed_steps_upper_bound', 0)} steps.",
        f"Tổng wall time các invocation đã flush (chưa gồm final export đang chạy): {sum(t['measured_wall_s'] for t in result['timing']):.3f}s. Closing receipt bổ sung toàn bộ export cuối.",
    ]
    (output / "REPORT_VI.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def export_bundles(root, analysis, token):
    root, analysis = Path(root), Path(analysis)
    exports = root / "exports"
    exports.mkdir(exist_ok=True)
    base = {
        f.relative_to(root).as_posix(): f
        for f in root.glob("*.json")
        if f.name != "SHA256SUMS.json"
    }
    for d in ("timing",):
        base.update({f.relative_to(root).as_posix(): f for f in (root / d).glob("*.json")})
    base.update(
        {
            "analysis/" + f.relative_to(analysis).as_posix(): f
            for f in analysis.rglob("*")
            if f.is_file()
        }
    )
    companion = []
    module_root = Path(__file__).parent
    for filename in (
        "analyze_bs5_curriculum_extend.py",
        "analyze_bs5_curriculum_bundle.py",
        "bs5_bundle.py",
        "bs5_curve_plot.py",
    ):
        base[filename] = module_root / filename
    review = dict(base)
    # Small package intentionally omits raw, weights and source snapshots.
    audit = dict(base)
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(root).as_posix()
        parts = Path(rel).parts
        if parts[0] in ("exports", "aborted_attempts") or "aborted_attempts" in parts:
            continue
        if parts[0] in ("runs", "parents") and f.suffix != ".pt":
            audit[rel] = f
    audit["source_snapshot.zip"] = root / "source_snapshot.zip"
    artifacts = []

    def write(kind, files, extra=None):
        path = exports / f"{kind}_{token[:12]}.zip"
        sums = {rel: file_hash(f) for rel, f in files.items()}
        payload = dict(extra or {})
        payload["SHA256SUMS.json"] = json.dumps(sums, indent=2).encode()
        # Include metadata entries in the manifest too.
        import hashlib

        for rel, data in list(payload.items()):
            if rel != "SHA256SUMS.json":
                sums[rel] = hashlib.sha256(data).hexdigest()
        payload["SHA256SUMS.json"] = json.dumps(sums, indent=2).encode()
        tmp = path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
            for rel, f in sorted(files.items()):
                compression = (
                    zipfile.ZIP_STORED if f.suffix in (".gz", ".zip") else zipfile.ZIP_DEFLATED
                )
                z.write(f, rel, compress_type=compression)
            for rel, data in payload.items():
                z.writestr(rel, data, compress_type=zipfile.ZIP_DEFLATED)
        with zipfile.ZipFile(tmp) as z:
            assert z.testzip() is None
        tmp.replace(path)
        item = dict(
            path=path.relative_to(root).as_posix(),
            sha256=file_hash(path),
            bytes=path.stat().st_size,
        )
        artifacts.append(item)
        return item

    write(
        "review_small",
        review,
        {
            "PACKAGE_SCOPE.json": json.dumps(
                dict(kind="review_only_not_raw_reaudit", snapshot=token)
            ).encode()
        },
    )
    companion.append(write("audit_light", audit))
    repro = dict(base)
    for f in root.rglob("*"):
        if f.is_file() and f.suffix == ".pt" and "aborted_attempts" not in f.parts:
            rel = f.relative_to(root).as_posix()
            if rel.startswith(("runs/", "parents/")):
                # Full parent + final resume; best/last selected checkpoints only.
                if "checkpoints" in f.parts:
                    c = read_json(f.parent.parent / "commit.json")
                    if int(f.stem) not in (c["steps"], c["best_step"]):
                        continue
                repro[rel] = f
    repro["source_snapshot.zip"] = root / "source_snapshot.zip"
    for rel in ("protocol.json", "source_identity.json", "source_snapshot.zip"):
        repro["parents/" + rel] = root / "parents" / rel
    if (root / "parents/run_registry.json").exists():
        repro["parents/run_registry.json"] = root / "parents/run_registry.json"
    for d in (root / "parents/runs").glob("*"):
        for rel in ("meta.json", "commit.json"):
            repro[f"parents/runs/{d.name}/{rel}"] = d / rel
    for d in (root / "runs").glob("*"):
        for rel in ("meta.json", "commit.json"):
            repro[f"runs/{d.name}/{rel}"] = d / rel
    write(
        "repro_full", repro, {"EVIDENCE_COMPANIONS.json": json.dumps(companion, indent=2).encode()}
    )
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = analyze(args.input, args.output, require_complete=args.require_complete)
    print(json.dumps(dict(runs=len(result["runs"]), coverage=result["coverage"]), indent=2))


if __name__ == "__main__":
    main()
