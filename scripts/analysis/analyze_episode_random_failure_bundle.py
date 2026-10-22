"""Offline evidence validator/analyzer. Needs only Python + NumPy, no torch/env."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def records(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def rate_utility(row, demand):
    rate = np.asarray(row["service"]["network"]["ue_rate_mbps"])
    assert np.isfinite(rate).all() and (rate >= 0).all()
    utility = float(np.minimum(rate / demand, 1).mean())
    assert abs(utility - row["reward"]) < 1e-12
    assert abs(utility - row["service"]["network"]["utility"]) < 1e-12
    assert row["service"]["failed_bs_idx"] == row["bs"]
    return utility


def analyze(root, output):
    root, output = Path(root), Path(output)
    audit = root / "audit"
    expected = read(audit / "summary.json")
    grouped, exogenous, max_error = defaultdict(list), {}, 0.0
    direct_impossible_rates = defaultdict(list)
    demand = read(audit / "config_cartesian.json")["ue_demand_mbps"]
    for row in records(audit / "service_traces.jsonl.gz"):
        key = row["kind"], row["bs"], row["seed"]
        grouped[key].append(rate_utility(row, demand))
        max_error = max(max_error, row["parity_max_error"])
        state = json.dumps(row["service"]["ue_positions"], separators=(",", ":")).encode()
        digest = hashlib.sha256(state).hexdigest()
        exo_key = row["bs"], row["seed"], row["t"]
        if exo_key in exogenous:
            assert digest == exogenous[exo_key], "policy-dependent UE trajectory"
        exogenous[exo_key] = digest
        if row["kind"] == "cartesian":
            direct_impossible_rates[row["bs"]].append(sum(row["relay_required"]))
    assert len(grouped) == expected["paired_episodes"]
    assert sum(map(len, grouped.values())) == expected["paired_slots"]
    for row in expected["rows"]:
        values = grouped[(row["kind"], row["bs"], row["seed"])]
        assert len(values) == row["slots"]
        assert abs(float(np.mean(values)) - row["utility"]) < 1e-12
    resets = read(audit / "random_resets.json")
    reset_counts = {i: sum(r["bs"] == i for r in resets) for i in range(6)}
    pilot_rows, val_slots, updates = [], 0, 0
    pilot_protocol = read(root / "pilot" / "protocol.json")
    expected_runs = {(m, s) for m in pilot_protocol["modes"] for s in pilot_protocol["seeds"]}
    seen_runs = set()
    pilot_exogenous = {}
    for summary_path in sorted((root / "pilot").glob("*_seed*/summary.json")):
        run = summary_path.parent
        summary, protocol = read(summary_path), read(run / "protocol.json")
        seen_runs.add((summary["mode"], summary["seed"]))
        assert summary["steps"] == pilot_protocol["steps_per_run"]
        vals = defaultdict(list)
        for row in records(run / "service_traces.jsonl.gz"):
            utility = rate_utility(row, protocol["config"]["ue_demand_mbps"])
            if row["phase"] == "validation":
                vals[(row["train_step"], row["bs"])].append(utility)
                val_slots += 1
                exo_key = row["bs"], row["t"]
                digest = hashlib.sha256(
                    json.dumps(row["service"]["ue_positions"], separators=(",", ":")).encode()
                ).hexdigest()
                if exo_key in pilot_exogenous:
                    assert digest == pilot_exogenous[exo_key], "unmatched validation UE trajectory"
                pilot_exogenous[exo_key] = digest
        for evaluation in summary["evaluations"]:
            for row in evaluation["rows"]:
                values = vals[evaluation["train_step"], row["bs"]]
                assert len(values) == row["slots"]
                assert abs(float(np.mean(values)) - row["utility"]) < 1e-12
        ep = [json.loads(x) for x in (run / "episodes.jsonl").read_text().splitlines()]
        up = [json.loads(x) for x in (run / "updates.jsonl").read_text().splitlines()]
        assert sum(x["slots"] for x in ep) == summary["steps"]
        assert len(up) == summary["update_count"]
        assert len({x["step"] for x in up}) == len(up)
        assert all(np.isfinite(x["loss"]) for x in up)
        updates += len(up)
        final = summary["evaluations"][-1]
        initial = summary["evaluations"][0]
        pilot_rows.append(
            dict(
                mode=summary["mode"],
                seed=summary["seed"],
                steps=summary["steps"],
                validation_initial=initial["macro_mean_utility"],
                validation_last=final["macro_mean_utility"],
                final_per_bs=final["rows"],
                train_bs_slots=summary["train_bs_slots"],
            )
        )
    assert seen_runs == expected_runs, "missing/incomplete pilot runs"
    result = dict(
        offline_without_checkpoint=True,
        paired_episodes=len(grouped),
        paired_slots=sum(map(len, grouped.values())),
        max_parity_error=max_error,
        ue_trajectory_hashes_matched_across_action_formulations=len(exogenous),
        random_reset_counts=reset_counts,
        relay_required_ue_mean_by_bs={
            i: float(np.mean(v)) for i, v in direct_impossible_rates.items()
        },
        pilot_runs=pilot_rows,
        pilot_total_train_steps=sum(x["steps"] for x in pilot_rows),
        pilot_validation_slots=val_slots,
        finite_update_count=updates,
        caution="short pipeline pilot; not evidence of learned relay advantage/generalization",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    analyze(a.results, a.output)
