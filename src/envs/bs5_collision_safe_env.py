"""Versioned simultaneous endpoint collision contract for the five-UAV BS5 task.

Legacy env classes and scientific locks are intentionally left byte-for-byte intact.
This is an endpoint occupancy model, not continuous-time separation/edge collision.
"""
from __future__ import annotations

import itertools
import numpy as np

from src.envs.bs5_curriculum_env import BS5CurriculumEnv

COLLISION_VERSION = "bs5_endpoint_collision_v2"
COLLISION_CONTRACT = {
    "version": COLLISION_VERSION,
    "occupancy": "unique cell outside charging stations for all participating UAVs",
    "stations": "unlimited shared hub for Returning/Charging/Active including rejoin/departure",
    "inactive": "curriculum status3 absent from routing and collision constraints",
    "choices": "requested endpoint or original endpoint only; no forced relocation",
    "priority": "Returning before Active, ascending UAV ID, among globally feasible endpoints",
    "stationary": "occupancy cannot be overwritten; dependencies propagate blocking",
    "swaps_cycles": "allowed simultaneous endpoint swaps/cycles; no swept-edge collision model",
    "return_blocked": "wait when no feasible requested move; no guaranteed progress through a stationary blocker",
    "energy": "legacy attempted-move drain preserved, including blocked Active requests",
}


def assert_endpoint_occupancy(positions, status, stations):
    hubs = {tuple(x) for x in np.asarray(stations).tolist()}
    occupied = {}
    for agent, (position, state) in enumerate(zip(positions, status)):
        if int(state) == 3:  # curriculum absence
            continue
        cell = tuple(position)
        if cell in hubs:
            continue
        if cell in occupied:
            raise AssertionError(
                f"{COLLISION_VERSION}: non-station co-location {occupied.get(cell)}/{agent} at {cell}"
            )
        occupied[cell] = agent


def resolve_endpoints(proposed, old, status_before, stations):
    """Choose first feasible lexicographic move vector, <=2**5 assignments.

    Considering joint endpoints prevents reverting an agent into a cell another
    agent has already claimed. A stationary blocker is retained; a dependency
    chain may move iff its final endpoints all satisfy occupancy.
    """
    wanted = np.asarray(proposed).copy()
    old_copy = np.asarray(old).copy()
    status = np.asarray(status_before)
    assert wanted.shape == old_copy.shape == (len(status), 2)
    assert len(status) <= 5, "This bounded resolver is versioned for the five-UAV task"
    assert np.isin(status, [0, 1, 2, 3]).all()
    assert_endpoint_occupancy(old_copy, status, stations)
    assert np.array_equal(wanted[status == 2], old_copy[status == 2]), "Charging UAV moved"
    assert np.array_equal(wanted[status == 3], old_copy[status == 3]), "Inactive UAV moved"
    assert np.all(np.abs(wanted-old_copy).sum(axis=1) <= 1), "Non-local requested movement"
    priority = sorted((i for i in range(len(status)) if status[i] in (0, 1)),
                      key=lambda i: (0 if status[i] == 1 else 1, i))
    movers = [i for i in priority if not np.array_equal(wanted[i], old_copy[i])]
    attempts = 0
    chosen = None
    # True first means the earliest feasible tuple respects priority exactly.
    for flags in itertools.product((True, False), repeat=len(movers)):
        attempts += 1
        candidate = old_copy.copy()
        for agent, move in zip(movers, flags):
            if move:
                candidate[agent] = wanted[agent]
        try:
            assert_endpoint_occupancy(candidate, status, stations)
        except AssertionError:
            continue
        chosen = candidate
        break
    assert chosen is not None, "Original legal occupancy must be a feasible fallback"
    proposed[:] = chosen
    blocked = [i for i in movers if np.array_equal(chosen[i], old_copy[i])]
    blocking = {str(i): [j for j in range(len(status)) if j != i and status[j] != 3
                         and np.array_equal(chosen[j], wanted[i])] for i in blocked}
    return dict(version=COLLISION_VERSION, requested_positions=wanted.tolist(),
                resolved_positions=chosen.tolist(), priority=priority,
                blocked_ids=blocked, blocked_by=blocking,
                returning_blocked_ids=[i for i in blocked if status[i] == 1],
                active_blocked_ids=[i for i in blocked if status[i] == 0],
                assignments_considered=attempts)


class BS5CollisionSafeCurriculumEnv(BS5CurriculumEnv):
    collision_version = COLLISION_VERSION

    def __init__(self, *args, **kwargs):
        self.last_collision_resolution = None
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        out = super().reset(*args, **kwargs)
        self.last_collision_resolution = None
        assert_endpoint_occupancy(self.uav_positions, self.status, self.charging_stations)
        return out

    def _resolve_collisions(self, proposed, old, status_before):
        self.last_collision_resolution = resolve_endpoints(
            proposed, old, status_before, self.charging_stations
        )

    def step(self, actions):
        out = super().step(actions)
        # Includes post-service Return/rejoin transitions; shared station hubs are legal.
        assert_endpoint_occupancy(self.uav_positions, self.status, self.charging_stations)
        out["info"]["collision_resolution"] = self.last_collision_resolution
        return out

    def _episode_meta(self):
        meta = super()._episode_meta()
        meta["collision_contract"] = COLLISION_CONTRACT
        return meta
