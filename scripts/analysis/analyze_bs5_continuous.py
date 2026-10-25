"""Standalone raw continuous diagnostic audit. No Torch, NumPy, env or weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

try:
    from bs5_bundle import (
        atomic_json,
        canonical_hash,
        file_hash,
        load_trace,
        read_json,
        validate_rates,
        write_csv,
    )
    from bs5_continuous_common import (
        MODELS,
        RETRAINED_VERSION,
        SCENARIOS,
        WINDOWS,
        busy_relays,
        geometry_classifier,
        check_contract,
        compare,
        lifecycle,
        prefix_compare,
        select_validation_bests,
    )
    from analyze_bs5_curriculum_bundle import episode_metrics
    from analyze_bs5_curriculum_extend import gateway_auditor, verify_manifest
    from bs5_curve_plot import plot
except ImportError:
    from scripts.analysis.bs5_bundle import (
        atomic_json,
        canonical_hash,
        file_hash,
        load_trace,
        read_json,
        validate_rates,
        write_csv,
    )
    from scripts.analysis.bs5_continuous_common import (
        MODELS,
        RETRAINED_VERSION,
        SCENARIOS,
        WINDOWS,
        busy_relays,
        geometry_classifier,
        check_contract,
        compare,
        lifecycle,
        prefix_compare,
        select_validation_bests,
    )
    from scripts.analysis.analyze_bs5_curriculum_bundle import episode_metrics
    from scripts.analysis.analyze_bs5_curriculum_extend import gateway_auditor, verify_manifest
    from scripts.analysis.bs5_curve_plot import plot


def expected_keys():
    return {f"{name}/{s}" for name in MODELS for s in SCENARIOS}


def safe(root, rel):
    path = (root / rel).resolve()
    assert path.is_relative_to(root.resolve()) and not path.is_symlink(), rel
    return path


class EvidenceTransport:
    """Read extracted containers explicitly; never claim their old byte hashes pass."""

    def __init__(self, root, *, allow_extracted=False, locked=None):
        self.root = Path(root).resolve()
        self.allow_extracted = allow_extracted
        self.locked = locked or {}
        self.unverified = {}

    def record(self, logical, actual, expected, kind, **extra):
        self.unverified[logical] = dict(
            logical_path=logical,
            original_sha256=expected,
            original_container_hash_verified=False,
            kind=kind,
            available_path=actual.relative_to(self.root).as_posix(),
            available_sha256=file_hash(actual) if actual.is_file() else None,
            **extra,
        )

    def checked(self, logical, expected=None):
        expected = expected or self.locked.get(logical)
        original = safe(self.root, logical)
        if original.is_file():
            assert expected is None or file_hash(original) == expected, f"corrupt file: {logical}"
            return original
        if self.allow_extracted and logical.endswith(".json.gz"):
            plain = original.with_suffix("")
            options = (
                plain,
                original / plain.name,
                plain / plain.name,
                original.parent / original.name.removesuffix(".json.gz"),
            )
            present = list(dict.fromkeys(p for p in options if p.is_file()))
            if present:
                assert all(file_hash(p) == file_hash(present[0]) for p in present), (
                    f"ambiguous extracted JSON: {logical}"
                )
                actual = present[0]
                payload = read_json(actual)
                self.record(
                    logical,
                    actual,
                    expected,
                    "extracted_json",
                    available_json_sha256=canonical_hash(payload),
                )
                return actual
        raise FileNotFoundError(f"Missing original and extracted representation: {original}")

    def trace(self, logical, expected=None):
        path = self.checked(logical, expected)
        return load_trace(path) if path.name.endswith(".gz") else read_json(path)

    def verify_locked(self):
        missing = []
        for logical, digest in self.locked.items():
            if (
                self.allow_extracted
                and logical == "references/training_source_snapshot.zip"
                and not safe(self.root, logical).is_file()
            ):
                continue  # Code identity is checked against every file below.
            try:
                self.checked(logical, digest)
            except FileNotFoundError:
                missing.append(logical)
        if missing:
            raise FileNotFoundError(
                f"Missing saved evidence (neither original nor extracted JSON): {missing}"
            )

    def source(self, logical, identity, *, fallback=None):
        original = safe(self.root, logical)
        if original.is_file():
            choices = [original]
        elif self.allow_extracted:
            choices = [original.with_suffix("") / "uav_thesis", original / "uav_thesis"]
            if fallback:
                choices.append(safe(self.root, fallback))
        else:
            choices = []
        for candidate in choices:
            try:
                if candidate.is_file():
                    with zipfile.ZipFile(candidate) as archive:
                        matches = all(
                            hashlib.sha256(archive.read("uav_thesis/" + rel)).hexdigest() == digest
                            for rel, digest in identity["files"].items()
                        )
                elif candidate.is_dir():
                    matches = all(
                        file_hash(candidate / rel) == digest
                        for rel, digest in identity["files"].items()
                    )
                else:
                    continue
            except (OSError, KeyError, zipfile.BadZipFile):
                matches = False
            if matches:
                if not original.is_file():
                    self.record(
                        logical,
                        candidate,
                        self.locked.get(logical),
                        "source_container_missing",
                        source_file_identity_verified=True,
                        frozen_source_sha256=identity["sha256"],
                    )
                return
            if candidate == original:
                raise AssertionError(f"source bytes mismatch: {logical}")
        raise FileNotFoundError(f"Missing matching source representation: {logical}")


def read_rows(root, index, *, transport=None):
    rows = []
    assert len(index) == len({i["path"] for i in index}), "duplicate chunk"
    for item in index:
        if transport is None:
            path = safe(root, item["path"])
            assert path.is_file() and file_hash(path) == item["sha256"], (
                f"corrupt/missing chunk {path}"
            )
            data = load_trace(path)
        else:
            data = transport.trace(item["path"], item["sha256"])
        assert isinstance(data, list) and len(data) == item["slots"]
        rows.extend(data)
    return rows


def validate_rows(rows, *, name, scenario, config, horizon=600, length=600):
    assert [r["slot"] for r in rows] == list(range(1, length + 1)), "missing/duplicate slot"
    assert 0 < length <= horizon
    # Continuous step_record does not include the legacy training metadata flag.
    # Surviving BS IDs are checked below; the independent capacity equations still
    # exclude BS5 for every actual/shadow network. Do not modify archived rows.
    checker = gateway_auditor(config, require_exclusion_flag=False)
    classify = geometry_classifier(
        config["grid_size"],
        config["cell_size_m"],
        config["access_radius_cells"],
        config["backhaul_radius_cells"],
        tuple(tuple(p) for p in config["bs_xy_m"]),
    )
    arm, seed = name.split("_seed")
    errors = {}
    for idx, r in enumerate(rows):
        slot = idx + 1
        assert r["arm"] == arm and r["seed"] == int(seed) and r["scenario_seed"] == scenario
        assert r["env_t_before"] == idx and r["env_t_after"] == slot
        assert not r["terminated"] and r["truncated"] == (slot == horizon)
        assert r["recurrent_reset"] == (slot == 1) and r["q_summary"]["recurrent"] is False
        assert r["ue_ids"] == list(range(90))
        assert r["surviving_bs_ids"] == [0, 1, 2, 3, 4]
        assert r["charging_station_cells"] == [[1, 1], [18, 1], [10, 18]]
        assert r["service"]["ue_demand_mbps"] == [4.0] * 90
        assert r["shadow_pure"] is True and all(
            "role_enumeration" not in v for v in r["shadow"].values()
        )
        assert r["exogenous_sha256"] == canonical_hash([r["service"]["ue_positions"], [4.0] * 90])
        assert r["service"]["ue_positions"] == r["decision"]["ue_positions"]
        assert tuple(r["relay_required_ue"]) == classify(
            tuple(tuple(p) for p in r["service"]["ue_positions"])
        ), "relay-required geometry mismatch"
        assert r["env_rng_after_step"] == r["env_rng_after_logging"]
        assert r["lifecycle_transitions"] == [
            dict(uav=a, before=b, after=s, slot=slot, phase="post")
            for a, (b, s) in enumerate(zip(r["decision"]["status"], r["post"]["status"]))
            if b != s
        ]
        validate_rates(r)
        checker(r)
        snap, net = r["service"], r["service"]["network"]
        assert len(snap["ue_positions"]) == len(net["ue_rate_mbps"]) == 90
        assert len(r["raw_actions"]) == len(r["action_mask"]) == len(r["next_action_mask"]) == 5
        for key in ("action_mask", "next_action_mask"):
            assert all(len(m) == 10 and any(m) and all(x in (0, 1) for x in m) for m in r[key])
        assert all(0 <= a < 10 for a in r["raw_actions"])
        if arm == "C-N":
            assert all(not any(m[1::2]) for m in r["action_mask"] + r["next_action_mask"])
            assert all(a % 2 == 0 for a in r["raw_actions"])
        active_access = {
            str(a) for a in range(5) if snap["status"][a] == 0 and snap["roles"][a] == 0
        }
        active_relay = {a for a in range(5) if snap["status"][a] == 0 and snap["roles"][a] == 1}
        assert set(net["association"]) == active_access
        assert {v for v in net["routes"].values() if v is not None} <= active_relay
        associated = {u for users in net["association"].values() for u in users}
        assert all(u in associated or rate <= 1e-12 for u, rate in enumerate(net["ue_rate_mbps"]))
        for a, users in net["association"].items():
            assert len(users) <= config["max_ues_per_access"]
            assert sum(net["ue_rate_mbps"][u] for u in users) <= net["backhaul_caps_mbps"][a] + 1e-8
        for phase in ("decision", "service", "post"):
            assert len(r[phase]["status"]) == 5 and all(s in (0, 1, 2) for s in r[phase]["status"])
            assert all(math.isfinite(b) and 0 <= b <= 100 for b in r[phase]["battery"])
        if idx:
            prior = rows[idx - 1]
            compare(prior["post"], r["decision"], errors, "continuity", exact=True)
            assert prior["next_action_mask"] == r["action_mask"]
            compare(prior["next_obs"], r["obs"], errors, "obs_continuity", exact=True)
            compare(prior["next_state"], r["state"], errors, "state_continuity", exact=True)
            assert prior["env_rng_after_logging"] == r["env_rng_before"]
    return errors


def interval_metrics(rows, initial, name, scenario, first, last):
    chosen = rows[first - 1 : last]
    metrics = episode_metrics(chosen)
    rates = [r["service"]["network"]["ue_rate_mbps"] for r in chosen]
    per_ue = []
    for ue in range(90):
        outage = maximum = 0
        for vector in rates:
            outage = outage + 1 if vector[ue] <= 1e-12 else 0
            maximum = max(maximum, outage)
        per_ue.append(
            dict(
                run=name,
                scenario=scenario,
                first=first,
                last=last,
                ue=ue,
                mean_rate_mbps=statistics.fmean(v[ue] for v in rates),
                zero_rate_slots=sum(v[ue] <= 1e-12 for v in rates),
                max_zero_rate_streak_slots=maximum,
                leading_outage_censored=rates[0][ue] <= 1e-12,
                trailing_outage_censored=rates[-1][ue] <= 1e-12,
            )
        )
    contexts = dict(
        charger=0, moving=0, no_ue_in_radius=0, no_gateway=0, no_active_access=0, unknown=0
    )
    for row in chosen:
        snap = row["service"]
        net = snap["network"]
        busy = busy_relays(row)
        for aid, (status, role) in enumerate(zip(snap["status"], snap["roles"])):
            if status != 0 or role != 1 or aid in busy:
                continue
            pos = snap["uav_positions"][aid]
            if pos in initial["charging_station_cells"]:
                key = "charger"
            elif pos != row["decision"]["positions"][aid]:
                key = "moving"
            elif not any(math.dist(pos, ue) <= 4.0 + 1e-9 for ue in snap["ue_positions"]):
                key = "no_ue_in_radius"
            elif net["relay_gateway_caps_mbps"].get(str(aid), 0) <= 0:
                key = "no_gateway"
            elif not net["association"]:
                key = "no_active_access"
            else:
                key = "unknown"
            contexts[key] += 1
    assert sum(contexts.values()) == metrics["relay_empty_slots"]
    unavailable = [r for r in chosen if any(s != 0 for s in r["service"]["status"])]
    metrics.update(
        run=name,
        scenario=scenario,
        first=first,
        last=last,
        empty_relay_contexts=contexts,
        mean_returning=statistics.fmean(
            sum(s == 1 for s in r["service"]["status"]) for r in chosen
        ),
        mean_charging=statistics.fmean(sum(s == 2 for s in r["service"]["status"]) for r in chosen),
        unavailable_slots=len(unavailable),
        utility_when_unavailable=statistics.fmean(r["reward"] for r in unavailable)
        if unavailable
        else None,
        throughput_when_unavailable=statistics.fmean(
            sum(r["service"]["network"]["ue_rate_mbps"]) for r in unavailable
        )
        if unavailable
        else None,
        availability_buckets={
            str(n): dict(
                slots=len(bucket),
                mean_utility=statistics.fmean(r["reward"] for r in bucket) if bucket else None,
                throughput_mbps=statistics.fmean(
                    sum(r["service"]["network"]["ue_rate_mbps"]) for r in bucket
                )
                if bucket
                else None,
            )
            for n in range(6)
            for bucket in [[r for r in chosen if sum(s != 0 for s in r["service"]["status"]) == n]]
        },
    )
    return metrics, per_ue


def write_table(output, name, rows):
    atomic_json(output / (name + ".json"), rows)
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        plain = [
            {
                k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                for k, v in r.items()
            }
            for r in rows
        ]
        write_csv(output / (name + ".csv"), plain, keys)


def validate_ledger(ledger):
    assert ledger["training_steps"] == ledger["optimizer_updates"] == 0
    totals = {"main": 0, "control": 0}
    executed = {"main": 0, "control": 0}
    for a in ledger["attempts"]:
        assert a["kind"] in totals and 0 <= a["executed_steps"] <= a["charged_steps"]
        assert a["uncertain_steps_upper_bound"] == a["charged_steps"] - a["executed_steps"]
        totals[a["kind"]] += a["charged_steps"]
        executed[a["kind"]] += a["executed_steps"]
    assert totals["main"] <= 24000 and totals["control"] <= 1200 and sum(totals.values()) <= 25200
    return totals, executed


def validate_retraining_provenance(root, p, selection, training_p):
    """New training is a separate experiment, with its own bests and export."""
    if p["version"] != RETRAINED_VERSION:
        assert selection.get("checkpoint_origin", "historical_fresh60k") == "historical_fresh60k"
        return None
    assert selection["checkpoint_origin"] == "new_authorized_fresh_0_to_60k"
    refs = root / "references"
    exp, budget, receipt = (
        read_json(refs / n)
        for n in (
            "fresh_experiment.json",
            "fresh_budget_ledger.json",
            "training_export_receipt.json",
        )
    )
    assert exp["origin"] == "fresh_0_to_60k_v1" and not exp["diagnostic"]
    assert exp["source_sha256"] == training_p["source_sha256"]
    assert exp["historical_imported_steps"] == budget["historical_imported_steps"] == 0
    assert exp["planned_new_training_steps"] == budget["planned_new_training_steps"] == 240000
    assert budget["whole_fresh_durable_steps"] == 240000
    assert budget["whole_fresh_executed_steps"] >= 240000
    assert budget["uncertain_unflushed_steps_upper_bound"] >= 0
    assert exp["policy"]["arms"] == ["C-R", "C-N"]
    assert exp["policy"]["training_seeds"] == [551, 552]
    assert exp["policy"]["parent_steps"] == 24000 and exp["policy"]["total_steps"] == 60000
    assert set(budget["runs"]) == set(MODELS)
    assert all(r["total_steps"] == 60000 for r in budget["runs"].values())
    assert receipt["snapshot"] == canonical_hash(
        dict(
            protocol=training_p,
            commits={name: read_json(refs / name / "commit.json") for name in MODELS},
        )
    )
    audit = [a for a in receipt["artifacts"] if Path(a["path"]).name.startswith("audit_light_")]
    repro = [a for a in receipt["artifacts"] if Path(a["path"]).name.startswith("repro_full_")]
    assert len(receipt["artifacts"]) == 3 and len(audit) == len(repro) == 1
    assert audit[0]["sha256"] == selection["expected_audit_sha256"]
    assert Path(repro[0]["path"]).name == selection["expected_repro"]
    curves = [r for m in selection["models"].values() for r in m["selection_curve"]]
    assert select_validation_bests(curves) == p["models"], "new bests changed after selection"
    return budget


def analyze(root, output, *, require_complete=False, allow_extracted_containers=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    assert not output.is_relative_to(root), "analysis must be outside immutable evidence"
    verify_manifest(root)
    p, registry, selection = (
        read_json(root / n)
        for n in ("protocol.json", "run_registry.json", "checkpoint_selection.json")
    )
    check_contract(p)
    assert registry["protocol_sha256"] == canonical_hash(p)
    transport = EvidenceTransport(
        root, allow_extracted=allow_extracted_containers, locked=p["evidence_files"]
    )
    transport.verify_locked()
    assert set(registry["trajectories"]) == expected_keys()
    assert p["checkpoint_selection_sha256"] == canonical_hash(selection)
    assert selection["status"] == "READY" and set(selection["models"]) == set(MODELS)
    provenance = read_json(root / "checkpoint_provenance.json")
    assert any(c["sha256"] == selection["expected_audit_sha256"] for c in provenance["companions"])
    training_p = read_json(root / "references/training_protocol.json")
    assert canonical_hash(training_p) == p["training_protocol_sha256"]
    assert training_p["continuation"]["origin"] == "fresh_0_to_60k_v1"
    retraining_budget = validate_retraining_provenance(root, p, selection, training_p)
    for name, locked in p["models"].items():
        step, utility = locked["step"], locked["validation150_utility"]
        m = selection["models"][name]
        assert m["step"] == step and math.isclose(
            m["validation150_utility"], utility, abs_tol=1e-12
        )
        assert m["file_sha256"] == m["provenance_manifest_sha256"] and m["tensor_digest"]
        assert provenance["manifest"][m["relative_path"]] == m["file_sha256"]
        assert [r["step"] for r in m["selection_curve"]] == list(range(0, 60001, 3000))
        best = max(m["selection_curve"], key=lambda r: (r["utility"], -r["step"]))
        assert best["step"] == step and best["utility"] == m["validation150_utility"]
        commit = read_json(root / "references" / name / "commit.json")
        assert commit["best_step"] == step and commit["best_utility"] == best["utility"]
        assert file_hash(root / "references" / name / "commit.json") == m["commit_sha256"]
        assert canonical_hash(m["resolved_config"]) == m["config_sha256"]
        reference_mean = statistics.fmean(
            episode_metrics(transport.trace(f"references/{name}/{s}.json.gz")["trace"])["utility"]
            for s in SCENARIOS
        )
        assert math.isclose(reference_mean, utility, rel_tol=1e-10, abs_tol=1e-10)
    identity = read_json(root / "source_identity.json")
    assert identity["sha256"] == canonical_hash(identity["files"]) == p["eval_source_sha256"]
    transport.source("source_snapshot.zip", identity)
    training_identity = read_json(root / "references/training_source_identity.json")
    assert (
        training_identity["sha256"]
        == canonical_hash(training_identity["files"])
        == training_p["source_sha256"]
    )
    transport.source(
        "references/training_source_snapshot.zip", training_identity, fallback="source_snapshot.zip"
    )
    for rel, digest in training_identity["files"].items():
        if rel.startswith("src/"):
            assert identity["files"][rel] == digest, "scientific source changed"
    ledger = read_json(root / "budget_ledger.json")
    charged, executed = validate_ledger(ledger)
    if require_complete:
        assert registry["status"] == "complete" and charged == executed == dict(
            main=24000, control=1200
        )
        assert all(
            a["status"] == "complete" and not a["uncertain_steps_upper_bound"]
            for a in ledger["attempts"]
        )
    # Controls are part of the fixed1200, never additional simulation in this analyzer.
    controls = read_json(root / "controls.json")
    if require_complete:
        assert controls["status"] == "complete"
    control_rows = {}
    for key, item in controls["runs"].items():
        rows = read_rows(root, item["chunks"], transport=transport)
        name = item["model"]
        validate_rows(
            rows,
            name=name,
            scenario=130001,
            config=selection["models"][name]["resolved_config"],
            horizon=item["horizon"],
            length=150,
        )
        control_rows[key] = rows
    control_errors = {}
    expected_controls = {f"env-{arm}-{h}" for arm in ("R", "N") for h in (150, 600)} | {
        f"policy-{n}" for n in MODELS
    }
    if require_complete:
        assert set(control_rows) == expected_controls
    if require_complete:
        assert len(ledger["attempts"]) == 48 and len({a["key"] for a in ledger["attempts"]}) == 48
        assert {a["key"] for a in ledger["attempts"] if a["kind"] == "main"} == expected_keys()
        assert {a["key"] for a in ledger["attempts"] if a["kind"] == "control"} == expected_controls
        for attempt in ledger["attempts"]:
            assert (
                attempt["executed_steps"]
                == attempt["committed_steps"]
                == (600 if attempt["kind"] == "main" else 150)
            )
            assert file_hash(safe(root, attempt["receipt"])) == attempt["receipt_sha256"]
    for arm in ("R", "N"):
        a, b = control_rows.get(f"env-{arm}-600"), control_rows.get(f"env-{arm}-150")
        if a and b:
            for x, y in zip(a, b):
                prefix_compare(x, y, control_errors)
                for field in ("obs", "state", "next_obs", "next_state"):
                    compare(x[field], y[field], control_errors, "env_parity." + field)
    windows, per_ue, cycles, uavs, trajectory, timeseries = [], [], [], [], [], []
    exogenous, reset_pairs, errors, raw_files = {}, {}, {}, set()
    for key, entry in sorted(registry["trajectories"].items()):
        if entry["status"] != "complete":
            assert not require_complete, f"incomplete trajectory {key}"
            continue
        name, scen = key.split("/")
        scenario = int(scen)
        receipt = read_json(safe(root, entry["receipt"]))
        assert file_hash(safe(root, entry["receipt"])) == entry["receipt_sha256"]
        assert (
            receipt["protocol_sha256"] == registry["protocol_sha256"]
            and receipt["reset_count"] == 1
        )
        assert receipt["model"] == name and receipt["scenario"] == scenario
        model = selection["models"][name]
        assert receipt["tensor_before"] == receipt["tensor_after"] == model["tensor_digest"]
        assert receipt["checkpoint_before"] == receipt["checkpoint_after"] == model["file_sha256"]
        assert receipt["training_steps"] == receipt["optimizer_updates"] == 0
        assert receipt["source_before"] == receipt["source_after"] == p["eval_source_sha256"]
        rows = read_rows(root, receipt["chunks"], transport=transport)
        raw_files.update(
            transport.checked(c["path"], c["sha256"]).relative_to(root).as_posix()
            for c in receipt["chunks"]
        )
        validate_rows(rows, name=name, scenario=scenario, config=model["resolved_config"])
        initial = receipt["initial"]
        compare(
            initial["positions"],
            rows[0]["decision"]["positions"],
            errors,
            "initial_positions",
            exact=True,
        )
        assert initial["residual_uav"] in range(5)
        assert initial["battery"] == [
            30.0 if i == initial["residual_uav"] else 100.0 for i in range(5)
        ]
        reset = {k: initial[k] for k in ("positions", "battery", "residual_uav", "ue_positions")}
        assert scenario not in reset_pairs or reset_pairs[scenario] == reset
        reset_pairs[scenario] = reset
        ref = transport.trace(f"references/{name}/{scenario}.json.gz")
        for idx, r in enumerate(rows):
            pair = scenario, r["slot"]
            assert pair not in exogenous or exogenous[pair] == r["exogenous_sha256"], (
                "paired exogenous mismatch"
            )
            exogenous[pair] = r["exogenous_sha256"]
            if idx < 150:
                prefix_compare(r, ref["trace"][idx], errors)
                if scenario == 130001:
                    old = control_rows[f"policy-{name}"][idx]
                    prefix_compare(r, old, errors)
                    for field in ("obs", "state", "next_obs", "next_state"):
                        compare(r[field], old[field], errors, "policy_prefix." + field)
            timeseries.append(
                dict(
                    run=name,
                    scenario=scenario,
                    slot=r["slot"],
                    utility=r["reward"],
                    throughput_mbps=sum(r["service"]["network"]["ue_rate_mbps"]),
                    active=sum(s == 0 for s in r["service"]["status"]),
                    charging=sum(s == 2 for s in r["service"]["status"]),
                    returning=sum(s == 1 for s in r["service"]["status"]),
                    post_status=r["post"]["status"],
                )
            )
        for first, last in WINDOWS:
            metric, ue = interval_metrics(rows, initial, name, scenario, first, last)
            windows.append(metric)
            per_ue.extend(ue)
            if (first, last) == (1, 600):
                trajectory.append(metric)
                compare(
                    metric, receipt["metrics"], errors, "receipt_metrics", rtol=1e-10, atol=1e-10
                )
        cs, us = lifecycle(rows, initial, run=name, scenario=scenario)
        cycles.extend(cs)
        uavs.extend(us)
    if require_complete:
        assert len(trajectory) == 40 and len(timeseries) == 24000
    # Refuse unindexed extra/duplicate raw packets, including partial ones in a complete bundle.
    actual = {
        f.relative_to(root).as_posix()
        for f in (root / "trajectories").rglob("*.json.gz")
        if f.is_file()
    }
    if allow_extracted_containers:
        actual.update(
            f.relative_to(root).as_posix()
            for f in (root / "trajectories").rglob("*")
            if f.is_file() and f.name.startswith("slots_")
        )
    assert actual == raw_files, "extra/uncommitted raw packets"
    output.mkdir(parents=True, exist_ok=True)
    paired, aggregate = [], []
    for seed in (551, 552):
        for first, last in WINDOWS:
            pairs = []
            for scenario in SCENARIOS:
                r = next(
                    (
                        w
                        for w in windows
                        if w["run"] == f"C-R_seed{seed}"
                        and (w["scenario"], w["first"], w["last"]) == (scenario, first, last)
                    ),
                    None,
                )
                n = next(
                    (
                        w
                        for w in windows
                        if w["run"] == f"C-N_seed{seed}"
                        and (w["scenario"], w["first"], w["last"]) == (scenario, first, last)
                    ),
                    None,
                )
                if r is None or n is None:
                    continue
                delta = r["utility"] - n["utility"]
                pairs.append(
                    dict(
                        seed=seed,
                        scenario=scenario,
                        first=first,
                        last=last,
                        relay=r["utility"],
                        no_relay=n["utility"],
                        delta=delta,
                        ratio=r["utility"] / n["utility"] if n["utility"] else None,
                    )
                )
            paired.extend(pairs)
            if pairs:
                rm, nm = (
                    statistics.fmean(x["relay"] for x in pairs),
                    statistics.fmean(x["no_relay"] for x in pairs),
                )
                aggregate.append(
                    dict(
                        seed=seed,
                        first=first,
                        last=last,
                        n_scenarios=len(pairs),
                        relay_mean=rm,
                        no_relay_mean=nm,
                        delta=rm - nm,
                        ratio_of_means=rm / nm if nm else None,
                        wins=sum(x["delta"] > 1e-10 for x in pairs),
                        ties=sum(abs(x["delta"]) <= 1e-10 for x in pairs),
                        losses=sum(x["delta"] < -1e-10 for x in pairs),
                    )
                )
    for name, table in (
        ("trajectories", trajectory),
        ("windows", windows),
        ("per_ue", per_ue),
        ("cycles", cycles),
        ("uavs", uavs),
        ("paired", paired),
        ("per_seed", aggregate),
        ("timeseries", timeseries),
    ):
        write_table(output, name, table)
    overall = []
    for arm in ("C-R", "C-N"):
        for first, last in WINDOWS:
            group = [
                r
                for r in windows
                if r["run"].startswith(arm) and (r["first"], r["last"]) == (first, last)
            ]
            if not group:
                continue
            row = dict(arm=arm, first=first, last=last, n_training_seeds=2)
            for field in (
                "utility",
                "throughput_mbps",
                "mean_qos_ue",
                "mean_positive_service_ue",
                "jain_mean_rate",
                "mean_served_relay_required_ue",
                "relay_required_throughput_mbps",
            ):
                means = [
                    statistics.fmean(r[field] for r in group if r["run"].endswith(str(seed)))
                    for seed in (551, 552)
                    if any(r["run"].endswith(str(seed)) for r in group)
                ]
                row["n_training_seeds"] = len(means)
                row[field] = statistics.fmean(means)
                row[field + "_sample_sd"] = statistics.stdev(means) if len(means) > 1 else None
            denominator = sum(r["relay_active_uav_slots"] for r in group)
            row["empty_relay_fraction_pooled"] = (
                sum(r["relay_empty_slots"] for r in group) / denominator if denominator else None
            )
            overall.append(row)
    write_table(output, "overall", overall)
    for entry in trajectory:
        name, scenario = entry["run"], entry["scenario"]
        points = [r for r in timeseries if r["run"] == name and r["scenario"] == scenario]
        plot(
            output / f"cycles_{name}_{scenario}",
            [
                dict(
                    label=f"UAV{a} OFFSET{a * 3}",
                    color=color,
                    points=[(r["slot"], a * 3 + r["post_status"][a]) for r in points],
                )
                for a, color in enumerate(("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00"))
            ],
            title=f"{name} SCENE{scenario} LIFECYCLE",
            ylabel="UAV OFFSET + STATUS 0A 1R 2C",
            end=600,
            markers=[],
            xlabel="EVALUATION SLOT (0 TO600)",
            footnote="POST-STEP STATUS; 0 ACTIVE 1 RETURNING 2 CHARGING; ONE CONTINUOUS GAME",
        )
    for seed in (551, 552):
        for field in ("utility", "throughput_mbps", "active", "charging", "returning"):
            series = []
            for arm, color in (("C-R", "#0072b2"), ("C-N", "#d55e00")):
                selected = [r for r in timeseries if r["run"] == f"{arm}_seed{seed}"]
                if selected:
                    series.append(
                        dict(
                            label=f"{arm} SEED{seed}",
                            color=color,
                            points=[
                                (s, statistics.fmean(r[field] for r in selected if r["slot"] == s))
                                for s in range(1, 601)
                            ],
                        )
                    )
            if series:
                plot(
                    output / f"{field}_seed{seed}",
                    series,
                    title=f"CONTINUOUS EVAL600 SEED{seed}",
                    ylabel=field.upper(),
                    end=600,
                    markers=[(s, "WINDOW") for s in (150, 300, 450)],
                    xlabel="EVALUATION SLOT (0 TO600)",
                    footnote="MEAN OF10 INSPECTED SCENARIOS WITHIN ONE TRAINING SEED; NO SMOOTHING",
                )
        for field in ("mean_qos_ue", "jain_mean_rate"):
            series = []
            for arm, color in (("C-R", "#0072b2"), ("C-N", "#d55e00")):
                points = []
                for first, last in WINDOWS[1:5]:
                    own = [
                        r[field]
                        for r in windows
                        if r["run"] == f"{arm}_seed{seed}"
                        and (r["first"], r["last"]) == (first, last)
                    ]
                    if own:
                        points.append(((first + last) / 2, statistics.fmean(own)))
                if points:
                    series.append(dict(label=arm, color=color, points=points))
            if series:
                plot(
                    output / f"{field}_seed{seed}",
                    series,
                    title=f"WINDOW METRICS SEED{seed}",
                    ylabel=field.upper(),
                    end=600,
                    markers=[],
                    xlabel="EVALUATION SLOT (WINDOW CENTERS)",
                    footnote="150-SLOT WINDOWS; JAIN ON PER-UE INTERVAL MEAN RATES; NO RESETS",
                )
        series = []
        for first, last in WINDOWS[1:5]:
            own = [
                r["delta"]
                for r in paired
                if r["seed"] == seed and (r["first"], r["last"]) == (first, last)
            ]
            if own:
                series.append(((first + last) / 2, statistics.fmean(own)))
        if series:
            plot(
                output / f"paired_delta_seed{seed}",
                [dict(label="C-R MINUS C-N", color="#0072b2", points=series)],
                title=f"PAIRED UTILITY SEED{seed}",
                ylabel="DELTA UTILITY",
                end=600,
                markers=[],
                xlabel="EVALUATION SLOT (WINDOW CENTERS)",
                footnote="PAIRED SAME SCENARIO; ONE TRAINING SEED; NOT INDEPENDENT TRAINING REPLICATIONS",
            )
    content_passed = bool(require_complete and len(trajectory) == 40)
    verification = dict(
        accepted=bool(content_passed and not transport.unverified),
        content_checks_passed=content_passed,
        original_container_hashes_verified=not transport.unverified,
        unverifiable_original_containers=list(transport.unverified.values()),
        status=("CONTENT_ONLY" if transport.unverified else "PASS")
        if require_complete
        else "INCOMPLETE",
        trajectories=len(trajectory),
        raw_steps=len(timeseries),
        charged_steps=charged,
        executed_steps=executed,
        checkpoints_loaded=False,
        training_steps=0,
        optimizer_updates=0,
        prefix_max_errors=errors,
        control_max_errors=control_errors,
        scope="integrity PASS is separate from lifecycle coverage; n=2; no unseen/convergence claim",
    )
    atomic_json(output / "verification.json", verification)
    atomic_json(
        output / "transport_integrity.json",
        dict(
            allow_extracted_containers=allow_extracted_containers,
            original_container_hashes_verified=not transport.unverified,
            unverifiable_original_containers=list(transport.unverified.values()),
        ),
    )
    report = [
        "# BS5 continuous600 — báo cáo",
        "",
        f"Trạng thái: {verification['status']}; {len(trajectory)}/40 trajectories; training0.",
        "Validation đã xem; horizon600 nằm ngoài training150; n=2 training seeds. Throughput=360×utility, không phải hai bằng chứng độc lập.",
        "Return/Charge/rejoin tự động. Useful service chỉ tính từ slot sau rejoin. Nhóm Relay rỗng là bối cảnh quan sát, chưa chứng minh nguyên nhân.",
        "",
    ]
    if transport.unverified:
        report += [
            f"CẢNH BÁO: {len(transport.unverified)} container gốc không còn; đọc bản giải nén/đối chiếu source identity. Không xác minh được hash byte ZIP/GZIP gốc; accepted=False, đây là phân tích CONTENT_ONLY, không phải integrity PASS.",
            "Kiểm tra slot, rate/capacity, budget, prefix parity và lifecycle vẫn giữ nguyên. Chi tiết từng file: transport_integrity.json.",
            "",
        ]
    if retraining_budget:
        report += [
            "Weights từ lượt train mới được cho phép; không phải checkpoint lịch sử.",
            f"Training riêng: durable={retraining_budget['whole_fresh_durable_steps']}, executed_logged={retraining_budget['whole_fresh_executed_steps']}, uncertainty_upper_bound={retraining_budget['uncertain_unflushed_steps_upper_bound']}; eval không cập nhật weights.",
            "",
        ]
    for r in aggregate:
        report.append(
            f"- Seed{r['seed']} slots{r['first']}-{r['last']}: C-R={r['relay_mean']:.6f}, C-N={r['no_relay_mean']:.6f}, delta={r['delta']:+.6f}; W/T/L={r['wins']}/{r['ties']}/{r['losses']}."
        )
    for arm in ("C-R", "C-N"):
        own = [u for u in uavs if u["run"].startswith(arm) and u["initial_group"] == "initial-full"]
        report.append(
            f"- {arm}: UAV ban đầu đầy pin đã sạc và phục vụ lại: {sum(u['useful_cycles_completed'] > 0 for u in own)}/{len(own)} quan sát UAV-trajectory; không phải số training replications."
        )
    for r in overall:
        if (r["first"], r["last"]) in ((1, 150), (151, 600), (451, 600)):
            report.append(
                f"- {r['arm']} slots{r['first']}-{r['last']}: QoS={r['mean_qos_ue']:.3f} UE; Jain={r['jain_mean_rate']:.4f}; Relay rỗng pooled={r['empty_relay_fraction_pooled']}."
            )
    report += [
        "",
        "Cycle censoring, latency và time-at-risk: bảng cycles/uavs. Outage và chuỗi đói rate: per_ue. Utility/rate theo số UAV unavailable: windows.availability_buckets. Vị trí rejoin/first useful và UAV chưa trở lại phục vụ đều được giữ.",
        "Mọi scenario thất bại giữ trong paired/timeseries. Lifecycle coverage hạn chế không đồng nghĩa lỗi integrity. Không chọn lại checkpoint theo600; không kết luận unseen generalization, curriculum>direct, steady-state hay hội tụ.",
    ]
    (output / "REPORT_VI.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--allow-extracted-containers", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                args.input,
                args.output,
                require_complete=args.require_complete,
                allow_extracted_containers=args.allow_extracted_containers,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
