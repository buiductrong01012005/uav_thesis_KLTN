"""Continuous BS5 diagnostic contract and lifecycle helpers (stdlib only)."""

from copy import deepcopy
from functools import lru_cache
import math
import statistics

MODELS = {
    "C-R_seed551": (57000, 0.32567905702211114),
    "C-R_seed552": (45000, 0.34607230012056966),
    "C-N_seed551": (60000, 0.19130437384042995),
    "C-N_seed552": (51000, 0.20476644355622847),
}
SCENARIOS = list(range(130001, 130011))
HORIZON = 600
TOLERANCE = dict(rtol=1e-7, atol=1e-6, summary=1e-10)
WINDOWS = [(1, 600), (1, 150), (151, 300), (301, 450), (451, 600), (151, 600)]


RETRAINED_VERSION = "bs5_continuous_eval600_retrained_v1"


def select_validation_bests(curves):
    """Freeze new bests from the complete common validation150 curve only."""
    assert {r["run"] for r in curves} == set(MODELS), "wrong training matrix"
    chosen = {}
    for name in MODELS:
        curve = sorted((r for r in curves if r["run"] == name), key=lambda r: r["step"])
        assert [r["step"] for r in curve] == list(range(0, 60001, 3000)), "incomplete curve"
        assert all(math.isfinite(r["utility"]) and 0 <= r["utility"] <= 1 for r in curve)
        best = max(curve, key=lambda r: (r["utility"], -r["step"]))
        chosen[name] = dict(step=best["step"], validation150_utility=best["utility"])
    return chosen


def contract(*, retrained_models=None):
    p = dict(
        version="bs5_continuous_eval600_v1",
        scope="inspected-validation long-horizon diagnostic; n=2 training seeds",
        models={n: dict(step=s, validation150_utility=v) for n, (s, v) in MODELS.items()},
        scenarios=SCENARIOS,
        horizon=HORIZON,
        slot_s=5,
        main_cap=24000,
        control_cap=1200,
        total_cap=25200,
        training_steps=0,
        optimizer_updates=0,
        tolerance=TOLERANCE,
        oracle=False,
        control_allocation=dict(env_parity=600, policy150=600),
        useful_service="service slot strictly after post-step Charging->Active; delivered rate>1e-12",
        charge_semantics="automatic Return/Charge/rejoin; policy controls Active movement/role",
        horizon_audit="cfg.horizon only controls base truncation; no horizon/time input in actor obs/state or UE generator",
    )
    if retrained_models is not None:
        assert set(retrained_models) == set(MODELS), "wrong retraining matrix"
        for m in retrained_models.values():
            assert set(m) == {"step", "validation150_utility"}
            assert type(m["step"]) is int and m["step"] in range(0, 60001, 3000)
            assert (
                math.isfinite(m["validation150_utility"]) and 0 <= m["validation150_utility"] <= 1
            )
        p.update(
            version=RETRAINED_VERSION,
            models=deepcopy(retrained_models),
            checkpoint_origin="new_authorized_fresh_0_to_60k_not_historical_weights",
        )
    return p


def check_contract(p):
    expected = (
        contract(retrained_models=p["models"])
        if p.get("version") == RETRAINED_VERSION
        else contract()
    )
    for key, value in expected.items():
        assert p[key] == value, f"locked contract changed: {key}"


def compare(a, b, errors, path="", *, exact=False, rtol=1e-7, atol=1e-6):
    """Structure/exact identifiers; floating physical values have frozen tolerance."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), f"structure mismatch {path}"
        for k in a:
            compare(a[k], b[k], errors, path + "." + str(k), exact=exact, rtol=rtol, atol=atol)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"shape mismatch {path}"
        for x, y in zip(a, b):
            compare(x, y, errors, path + "[]", exact=exact, rtol=rtol, atol=atol)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert isinstance(b, (int, float)) and not isinstance(b, bool), path
        assert math.isfinite(a) and math.isfinite(b), path
        err = abs(a - b)
        errors[path] = max(errors.get(path, 0), err)
        assert (a == b) if exact else math.isclose(a, b, rel_tol=rtol, abs_tol=atol), (
            f"numeric mismatch {path}: {a} != {b}"
        )
    else:
        assert a == b, f"exact mismatch {path}: {a} != {b}"


def prefix_compare(new, reference, errors):
    for key in (
        "slot",
        "arm",
        "seed",
        "scenario_seed",
        "raw_actions",
        "action_mask",
        "next_action_mask",
        "ue_ids",
        "surviving_bs_ids",
        "terminated",
    ):
        compare(new[key], reference[key], errors, key, exact=True)
    for phase in ("decision", "service", "post"):
        for key in ("status", "roles"):
            compare(new[phase][key], reference[phase][key], errors, phase + "." + key, exact=True)
        # Historical decision snapshots did not include UE positions or obs/state.
        for key in ("positions", "uav_positions", "battery", "ue_positions", "ue_demand_mbps"):
            if key in reference[phase]:
                compare(new[phase][key], reference[phase][key], errors, phase + "." + key)
    compare(new["service"]["network"], reference["service"]["network"], errors, "network")
    compare(new["reward"], reference["reward"], errors, "reward")
    assert reference["truncated"] == (reference["slot"] == 150)
    assert not new["truncated"]


def busy_relays(row):
    net = row["service"]["network"]
    return {
        int(rid)
        for aid, rid in net["routes"].items()
        if rid is not None
        and any(net["ue_rate_mbps"][u] > 1e-12 for u in net["association"].get(str(aid), []))
    }


@lru_cache(maxsize=4)
def geometry_classifier(grid, cell_size, access_radius, backhaul_radius, bs_xy):
    """Independent grid geometry; memoized across paired policy rollouts."""
    cells = [(r, c) for r in range(grid) for c in range(grid)]
    direct = {
        p
        for p in cells
        if any(
            math.dist(((p[1] + 0.5) * cell_size, (p[0] + 0.5) * cell_size), bs)
            <= backhaul_radius * cell_size + 1e-9
            for bs in bs_xy[:5]
        )
    }
    relay = {p for p in cells if any(math.dist(p, q) <= backhaul_radius + 1e-9 for q in direct)}

    @lru_cache(maxsize=8192)
    def classify(positions):
        required = []
        for row, col in positions:
            cover = [
                (r, c)
                for r in range(
                    max(0, math.ceil(row - access_radius)),
                    min(grid - 1, math.floor(row + access_radius)) + 1,
                )
                for c in range(
                    max(0, math.ceil(col - access_radius)),
                    min(grid - 1, math.floor(col + access_radius)) + 1,
                )
                if math.hypot(row - r, col - c) <= access_radius + 1e-9
            ]
            required.append(not any(p in direct for p in cover) and any(p in relay for p in cover))
        return tuple(required)

    return classify


def traffic_for(row, agent):
    snap = row["service"]
    if snap["status"][agent] != 0:
        return 0.0
    net = snap["network"]
    if snap["roles"][agent] == 0:
        clients = [str(agent)]
    else:
        clients = [aid for aid, rid in net["routes"].items() if rid == agent]
    return sum(net["ue_rate_mbps"][u] for aid in clients for u in net["association"].get(aid, []))


def lifecycle(rows, initial, *, run, scenario):
    """All cycles from transitions; service is attributed before end-slot Return."""
    cycles, summaries = [], []
    end = rows[-1]["slot"]
    for agent in range(5):
        current = None
        count = 0

        def close(cycle, stop, reason):
            cycle["observation_end_slot"] = stop
            rejoin = cycle["rejoin_active_slot"]
            first = cycle["useful_service_return_slot"]
            cycle["complete_charge_cycle"] = rejoin is not None
            cycle["useful_service_censored"] = first is None
            cycle["censor_reason"] = reason if first is None else None
            cycle["time_at_risk_slots"] = max(0, stop - rejoin) if rejoin is not None else 0
            cycle["service_latency_slots"] = first - rejoin if first is not None else None
            arrival = cycle["charging_slot"]
            ret = cycle["return_slot"]
            cycle["return_duration_slots"] = arrival - ret if arrival is not None else None
            cycle["charging_duration_slots"] = rejoin - arrival if rejoin is not None else None
            cycle["unavailable_duration_slots"] = rejoin - ret if rejoin is not None else None
            cycle["unavailable_observed_slots"] = (rejoin if rejoin is not None else stop) - ret
            cycles.append(deepcopy(cycle))

        for row in rows:
            slot = row["slot"]
            before, after = row["decision"]["status"][agent], row["post"]["status"][agent]
            if (
                current
                and current["rejoin_active_slot"] is not None
                and slot > current["rejoin_active_slot"]
            ):
                rate = traffic_for(row, agent)
                if rate > 1e-12:
                    if current["useful_service_return_slot"] is None:
                        current["useful_service_return_slot"] = slot
                        current["first_useful_position"] = row["service"]["uav_positions"][agent]
                    current["useful_service_slots"] += 1
                    current["traffic_after_rejoin_mbit"] += rate * 5
            if before == 0 and after == 1:
                if current:
                    close(current, slot, "next_return")
                count += 1
                current = dict(
                    run=run,
                    scenario=scenario,
                    uav=agent,
                    cycle_index=count,
                    initial_battery=initial["battery"][agent],
                    initial_group="initial-residual"
                    if agent == initial["residual_uav"]
                    else "initial-full",
                    return_slot=slot,
                    charging_slot=None,
                    rejoin_active_slot=None,
                    useful_service_return_slot=None,
                    useful_service_slots=0,
                    traffic_after_rejoin_mbit=0.0,
                    return_position=row["post"]["positions"][agent],
                    role_before_return=row["service"]["roles"][agent],
                    charging_position=None,
                    rejoin_position=None,
                    first_useful_position=None,
                )
            elif before == 1 and after == 2:
                assert current and current["charging_slot"] is None
                current["charging_slot"] = slot
                current["charging_position"] = row["post"]["positions"][agent]
            elif before == 2 and after == 0:
                assert (
                    current
                    and current["charging_slot"] is not None
                    and current["rejoin_active_slot"] is None
                )
                current["rejoin_active_slot"] = slot
                current["rejoin_position"] = row["post"]["positions"][agent]
        if current:
            close(current, end, "horizon")
        own = [c for c in cycles if c["uav"] == agent]
        latencies = [
            c["service_latency_slots"] for c in own if c["service_latency_slots"] is not None
        ]
        summaries.append(
            dict(
                run=run,
                scenario=scenario,
                uav=agent,
                initial_group="initial-residual"
                if agent == initial["residual_uav"]
                else "initial-full",
                cycles_started=count,
                charge_cycles_completed=sum(c["complete_charge_cycle"] for c in own),
                useful_cycles_completed=len(latencies),
                useful_cycles_censored=len(own) - len(latencies),
                mean_uncensored_latency=statistics.fmean(latencies) if latencies else None,
                time_at_risk_slots=sum(c["time_at_risk_slots"] for c in own),
                final_status=rows[-1]["post"]["status"][agent],
                final_position=rows[-1]["post"]["positions"][agent],
            )
        )
    return cycles, summaries
