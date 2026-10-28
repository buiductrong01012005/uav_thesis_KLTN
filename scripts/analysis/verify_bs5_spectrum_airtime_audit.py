"""Verify saved fleet schedules/flow from output bytes, without solving the LP again."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path


def verify(output: Path):
    config = json.loads((output / "audit_config.json").read_text())
    receipt = json.loads((output / "AUDIT_RECEIPT.json").read_text())
    expected = 1
    for key in [
        "fixtures",
        "role_fixtures",
        "access_bandwidths_mhz",
        "backhaul_bandwidths_mhz",
        "spectrum_modes",
        "backhaul_modes",
        "rate_efficiencies",
    ]:
        expected *= len(config[key])
    count, seen, paired = 0, set(), {}
    max_violation = 0.0
    with gzip.open(output / "schedules.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            detail = json.loads(line)
            metric = detail["metric"]
            assert metric["case_id"] not in seen
            seen.add(metric["case_id"])
            before, after = detail["ue_access_rates"], detail["ue_delivered_rates"]
            assert len(before) == len(after) == config["users"]
            assert all(
                -1e-6 <= y <= x + 1e-6 and x <= config["target_ue_mbps"] + 1e-6
                for x, y in zip(before, after)
            )
            assert abs(sum(after) - metric["delivered_mbps"]) < 1e-6
            associated = [u for users in detail["association"].values() for u in users]
            assert len(associated) == len(set(associated)) == metric["associated_ues"]
            loads = [0.0] * len(detail["edges"])
            for aid, flow in detail["access_flows"].items():
                users = detail["association"][aid]
                offer = detail["access_offers"][aid]
                assert abs(sum(before[u] for u in users) - offer) < 1e-6
                assert abs(sum(after[u] for u in users) - flow) < 1e-6
                assert all(
                    abs(after[u] - (before[u] * flow / offer if offer > 0 else 0)) < 1e-6
                    for u in users
                )
                route = detail["paths"][aid]
                if not route:
                    assert abs(flow) < 1e-6
                for index in route:
                    loads[index] += flow
            airtime = sum(state["duration"] for state in detail["schedules"])
            assert airtime <= 1 + 1e-6
            capacities = [0.0] * len(loads)
            for state in detail["schedules"]:
                assert state["duration"] >= 0
                edges = [detail["edges"][i] for i in state["active_edges"]]
                if metric["backhaul_mode"] == "shared_half_duplex":
                    nodes = list(
                        itertools.chain.from_iterable((e["source"], e["target"]) for e in edges)
                    )
                    assert len(nodes) == len(set(nodes)), "Single radio/duplex conflict"
                for i, capacity in enumerate(state["capacity_mbps"]):
                    assert capacity >= 0
                    if i not in state["active_edges"]:
                        assert capacity == 0
                    assert capacity <= detail["edges"][i]["solo_mbps"] + 1e-6
                    capacities[i] += state["duration"] * capacity
            for used, available in zip(loads, capacities):
                max_violation = max(max_violation, used - available)
                assert used <= available + 1e-6, "Flow exceeds time-allocated edge capacity"
            key = tuple(
                metric[k]
                for k in [
                    "fixture",
                    "role_fixture",
                    "access_bw_label_mhz",
                    "backhaul_total_bandwidth_mhz",
                    "spectrum_mode",
                    "rate_efficiency",
                ]
            )
            paired.setdefault(key, {})[metric["backhaul_mode"]] = metric["delivered_mbps"]
            count += 1
    assert count == expected == receipt["cases"]
    for pair in paired.values():
        assert pair["shared_half_duplex"] <= pair["ideal_parallel_bound"] + 1e-6
    return dict(
        status="passed",
        checked_cases=count,
        paired_capacity_bounds=len(paired),
        max_flow_constraint_violation_mbps=max_violation,
        checked="saved bytes: UE scaling/demand, association uniqueness, shared-edge flows, endpoint duplex conflicts, airtime and ideal/audited paired bounds",
        schedules_sha256=hashlib.sha256((output / "schedules.jsonl.gz").read_bytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = verify(args.output)
    (args.output / "SCHEDULE_VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
