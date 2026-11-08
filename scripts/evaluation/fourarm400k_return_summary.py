"""Returning statistics from measured slots and optional planned routes."""
from types import SimpleNamespace

import numpy as np

from scripts.evaluation.fair300k_benchmark_metrics import extend_summary as base_summary


VERSION = "return_summary_v2"


def return_metrics(events, distance_m):
    """Match arrivals to departures using completed-slot boundaries."""
    distance = np.asarray(distance_m)
    starts = {}
    planned, actual, duration = [], [], []
    detours = 0
    for event in events:
        agent = event["agent"]
        if event["event"] == "return_start":
            starts[agent] = event
        elif event["event"] == "swap_start" and agent in starts:
            start = starts.pop(agent)
            lo, hi = start["slot"], event["slot"]
            if not 0 <= lo < hi <= len(distance):
                raise ValueError(f"Invalid Returning slot boundaries: {lo}, {hi}")
            meters = float(distance[lo:hi, agent].sum())
            actual.append(meters)
            duration.append(hi - lo)
            plan = start.get("route_m")
            if plan is not None:
                if not np.isfinite(plan) or plan < 0:
                    raise ValueError(f"Invalid planned Returning distance: {plan}")
                planned.append(plan)
                detours += int(meters > plan + 1e-6)

    return dict(
        return_summary_version=VERSION,
        mean_return_planned_m=float(np.mean(planned)) if planned else None,
        mean_return_actual_m=float(np.mean(actual)) if actual else None,
        mean_return_duration_slots=float(np.mean(duration)) if duration else None,
        return_completed_events=len(actual),
        return_known_plan_events=len(planned),
        return_missing_plan_events=len(actual) - len(planned),
        return_detour_events=detours,
    )


def extend_summary(state, protocol):
    # The frozen observer still calculates all non-Returning metrics. Pass an
    # empty event view to that function only, keeping the real log untouched.
    result = base_summary(dict(state, env=SimpleNamespace(energy_events=[])), protocol)
    result.update(return_metrics(
        state["env"].energy_events, [row["distance_m"] for row in state["energy"]],
    ))
    return result
