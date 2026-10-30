"""Endpoint invariants and collision regressions, without env.step or training."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.envs.bs5_collision_safe_env import (
    COLLISION_VERSION,
    assert_endpoint_occupancy,
    resolve_endpoints,
)
from src.envs.v4_fixed_mobile_battery_env import V4FixedMobileBatteryEnv

HUBS = np.array([[1, 1], [18, 1], [10, 18]])


def resolve(old, desired, status=None):
    old = np.array(old)
    desired = np.array(desired)
    status = np.array(status if status is not None else [0] * len(old))
    saved = old.copy()
    wanted = desired.copy()
    d = resolve_endpoints(desired, old, status, HUBS)
    np.testing.assert_array_equal(old, saved)
    assert d["requested_positions"] == wanted.tolist()
    assert d["version"] == COLLISION_VERSION and d["assignments_considered"] <= 32
    assert_endpoint_occupancy(desired, status, HUBS)
    return desired.tolist(), d


def test_stationary_blocker_regression_and_legacy_preserved():
    old = np.array([[5, 5], [5, 6]])
    legacy = np.array([[5, 6], [5, 6]])
    V4FixedMobileBatteryEnv._resolve_collisions(
        SimpleNamespace(charging_stations=HUBS), legacy, old, np.array([0, 0])
    )
    assert legacy[0].tolist() == legacy[1].tolist()  # historical code remains unchanged
    result, d = resolve(old, [[5, 6], [5, 6]])
    assert result == old.tolist() and d["blocked_ids"] == [0]


def test_cascading_blocking_dependency():
    old = [[5, 4], [5, 5], [5, 6]]
    result, d = resolve(old, [[5, 5], [5, 6], [5, 6]])
    assert result == old and d["blocked_ids"] == [0, 1]


def test_dependency_chain_with_free_destination_moves_together():
    desired = [[5, 5], [5, 6], [5, 7]]
    result, d = resolve([[5, 4], [5, 5], [5, 6]], desired)
    assert result == desired and d["blocked_ids"] == []


def test_simultaneous_swap_allowed():
    desired = [[5, 6], [5, 5]]
    result, _ = resolve([[5, 5], [5, 6]], desired)
    assert result == desired


def test_simultaneous_four_cycle_allowed():
    desired = [[5, 6], [6, 6], [6, 5], [5, 5]]
    result, _ = resolve([[5, 5], [5, 6], [6, 6], [6, 5]], desired)
    assert result == desired


def test_free_destination_conflict_uses_active_id_priority():
    result, d = resolve([[5, 4], [5, 6]], [[5, 5], [5, 5]])
    assert result == [[5, 5], [5, 6]] and d["blocked_ids"] == [1]


def test_returning_has_feasible_priority_over_lower_active_id():
    result, d = resolve([[5, 4], [5, 6]], [[5, 5], [5, 5]], [0, 1])
    assert result == [[5, 4], [5, 5]] and d["priority"] == [1, 0]


def test_returning_waits_for_stationary_occupant_without_teleport():
    result, d = resolve([[5, 5], [5, 6]], [[5, 6], [5, 6]], [1, 0])
    assert result == [[5, 5], [5, 6]]
    assert d["returning_blocked_ids"] == [0] and d["blocked_by"] == {"0": [1]}


def test_return_priority_cannot_break_blocked_dependency():
    old = [[5, 4], [5, 5], [5, 6]]
    result, d = resolve(old, [[5, 5], [5, 6], [5, 6]], [1, 0, 0])
    assert result == old and d["returning_blocked_ids"] == [0]


def test_charger_shared_arrival_charge_rejoin_and_departure():
    result, _ = resolve([[1, 0], [1, 2], [1, 1]], [[1, 1], [1, 1], [1, 1]], [1, 1, 2])
    assert result == [[1, 1]] * 3
    result, _ = resolve([[1, 1], [1, 1], [1, 1]], [[1, 2], [2, 1], [1, 1]], [0, 0, 2])
    assert result == [[1, 2], [2, 1], [1, 1]]
    result, _ = resolve([[1, 1], [1, 1]], [[1, 1], [1, 1]], [0, 0])
    assert result == [[1, 1], [1, 1]]


def test_departure_from_shared_hub_conflict_holds_one_at_hub():
    result, d = resolve([[1, 1], [1, 1]], [[1, 2], [1, 2]], [0, 0])
    assert result == [[1, 2], [1, 1]] and d["blocked_ids"] == [1]


def test_inactive_does_not_occupy_or_move():
    result, d = resolve([[5, 4], [5, 5]], [[5, 5], [5, 5]], [0, 3])
    assert result == [[5, 5], [5, 5]] and d["blocked_ids"] == []


def test_illegal_old_overlap_rejected_instead_of_silent_repair():
    with pytest.raises(AssertionError, match="co-location"):
        resolve([[5, 5], [5, 5]], [[5, 6], [5, 4]])


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_randomized_multistep_endpoint_invariants(n):
    rng = np.random.default_rng(71000 + n)
    motion = np.array([[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]])
    for _ in range(50):
        old = np.array([divmod(int(k), 20) for k in rng.choice(400, n, replace=False)])
        status = rng.choice([0, 1, 2, 3], size=n)
        for a in range(n):
            if status[a] == 2:
                old[a] = HUBS[a % 3]
        # Inactive overlap is allowed, but don't seed physical agents with illegal overlaps.
        try:
            assert_endpoint_occupancy(old, status, HUBS)
        except AssertionError:
            continue
        for __ in range(20):
            desired = old.copy()
            for a in range(n):
                if status[a] in (0, 1):
                    desired[a] = np.clip(old[a] + motion[rng.integers(5)], 0, 19)
            result, d = resolve(old, desired, status)
            result = np.array(result)
            assert np.all((result == old).all(1) | (result == desired).all(1))
            for a in range(n):
                if status[a] in (2, 3):
                    np.testing.assert_array_equal(result[a], old[a])
            old = result


def test_historical_scientific_files_match_original_lock():
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "configs/bs5_scientific_source_lock_v1.json").read_text())
    for rel, digest in lock["files"].items():
        if Path(rel).suffix.lower() == ".md" and not (root / rel).is_file():
            continue  # Historical documentation is kept locally, outside the source clone.
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == digest, rel
