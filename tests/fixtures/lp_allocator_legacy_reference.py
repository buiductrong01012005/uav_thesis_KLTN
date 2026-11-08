"""Frozen pre-fix allocator for numeric regression tests only; never used by controllers."""
import numpy as np
from scipy.optimize import linprog

def allocate_flows(offers: dict[int, float], paths: dict[int, list[int]], capacities: np.ndarray):
    """Lexicographic LP: maximum Mbps, then max-min offer fulfillment.

    q_a uses the same proportional scaling of UE rates within each Access as
    the legacy env. Global constraints: D q <= C t, sum(t)<=1. Fixed-route
    steady-state fluid capacity only, not packet delay or empty-buffer startup.
    """
    ids = sorted(offers)
    n, edges, states = len(ids), *capacities.shape
    if np.any(~np.isfinite(capacities)) or np.any(capacities < 0):
        raise ValueError("Invalid link capacities")
    demand = np.zeros((edges, n))
    for j, aid in enumerate(ids):
        for edge in paths[aid]:
            demand[edge, j] = 1
    eligible = [j for j, aid in enumerate(ids) if paths[aid] and offers[aid] > 1e-12]
    if not eligible:
        return {aid: 0.0 for aid in ids}, np.zeros(states), 0.0, 0.0
    # Variables: per-Access Mbps, state durations, minimum satisfaction lambda.
    size = n + states + 1
    constraints, rhs = [], []
    for e in range(edges):
        row = np.zeros(size)
        row[:n] = demand[e]
        row[n:n + states] = -capacities[e]
        constraints.append(row)
        rhs.append(0.0)
    row = np.zeros(size)
    row[n:n + states] = 1
    constraints.append(row)
    rhs.append(1.0)
    for j in eligible:
        row = np.zeros(size)
        row[j] = -1
        row[-1] = offers[ids[j]]
        constraints.append(row)
        rhs.append(0.0)
    bounds = [(0, offers[a] if paths[a] else 0) for a in ids] + [(0, 1)] * states + [(0, 1)]
    a_ub, b_ub = np.asarray(constraints), np.asarray(rhs)
    options = {"dual_feasibility_tolerance": 1e-8, "primal_feasibility_tolerance": 1e-8}
    objective = np.zeros(size)
    objective[:n] = -1
    first = linprog(objective, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs", options=options)
    if not first.success:
        raise RuntimeError(first.message)
    maximum_total = float(first.x[:n].sum())
    total_constraint = np.zeros(size)
    total_constraint[:n] = -1
    a_ub = np.vstack([a_ub, total_constraint])
    b_ub = np.append(b_ub, -max(0, maximum_total - 1e-8))
    objective[:] = 0
    objective[-1] = -1
    second = linprog(objective, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs", options=options)
    if not second.success:
        raise RuntimeError(second.message)
    violation = max(0.0, float(np.max(a_ub @ second.x - b_ub)))
    if violation > 1e-6:
        raise AssertionError(f"Infeasible LP result: {violation}")
    minimum = float(second.x[-1])
    return dict(zip(ids, second.x[:n])), second.x[n:n + states], minimum, violation
