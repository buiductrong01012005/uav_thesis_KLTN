"""Regression controls for the two observed two-cell Return oscillations."""

from copy import deepcopy
from dataclasses import asdict
import json
import numpy as np
import pytest
from src.envs.bs5_energy_return_env import BS5EnergyReturnEnv
from src.envs.bs5_return_commit_env import BS5CommittedReturnEnv
from src.envs.bs5_collision_safe_env import assert_endpoint_occupancy
from scripts.analysis.bs5_return_commit_audit import CommitAuditor
from scripts.analysis.analyze_bs5_cn_h1200_return import validate_collision

CONDITION = dict(
    stage="S4",
    substage="S4",
    active_count=5,
    mobility=True,
    battery=True,
    starts="B",
    radius=None,
    jitter=0,
    stay_only=False,
)


def make(cls=BS5CommittedReturnEnv):
    e = cls(CONDITION, horizon=100)
    e.reset(seed=180401, assignment_seed=180401, diagnostic=True)
    return e


def decision(e):
    return {
        k: getattr(e, v).tolist()
        for k, v in [
            ("positions", "uav_positions"),
            ("status", "status"),
            ("battery", "battery"),
            ("roles", "roles"),
        ]
    }


def audit_step(e, actions, auditor):
    before = decision(e)
    out = e.step(actions)
    row = dict(
        decision=before,
        post=decision(e),
        raw_actions=actions,
        service={
            k: e.last_service_snapshot[k].tolist() for k in ["status", "roles", "uav_positions"]
        },
        task_condition=e.condition,
        return_control=e.last_return_control,
        collision=e.last_collision_resolution,
        charging_station_cells=e.charging_stations.tolist(),
        applied_actions=[
            dict(
                autonomous=before["status"][a] != 0
                or a in e.last_return_control["forced_return_ids"]
            )
            for a in range(5)
        ],
    )
    auditor.validate(row, asdict(e.cfg))
    validate_collision(row, e.charging_stations.tolist())
    return out, row


@pytest.mark.parametrize(
    "agent,blocker,start,blocking,initial_battery",
    [
        (4, 1, [10, 14], [10, 17], 5.49722146987915),
        (2, 0, [18, 5], [18, 2], 5.395699977874756),
    ],
)
def test_observed_oscillating_blocker_controls_arrive(
    agent, blocker, start, blocking, initial_battery, tmp_path
):
    old = make(BS5EnergyReturnEnv)
    new = make()
    auditor = CommitAuditor()
    for e in [old, new]:
        e.status[:] = 3
        e.uav_positions[:] = [[0, i] for i in range(5)]
        e.status[agent] = 1
        e.status[blocker] = 0
        e.uav_positions[agent] = start
        e.uav_positions[blocker] = blocking
        e.battery[:] = 100
        e.battery[agent] = initial_battery
        e.target_station_idx[agent] = 2 if agent == 4 else 1
    arrival = None
    rejoin = None
    history = []
    for t in range(1, 41):
        action = (4 if t % 2 else 6) if agent == 4 else (6 if t % 2 else 4)
        acts = [8] * 5
        acts[blocker] = action
        old.step(acts)
        _, row = audit_step(new, acts, auditor)
        if arrival is None and new.status[agent] == 2:
            arrival = t
        if arrival is not None and rejoin is None and new.status[agent] == 0:
            rejoin = t
        history.append(
            dict(
                slot=t,
                old_position=old.uav_positions[agent].tolist(),
                new_position=new.uav_positions[agent].tolist(),
                old_battery=float(old.battery[agent]),
                new_battery=float(new.battery[agent]),
                reason=row["return_control"]["route_decisions"].get(str(agent)),
            )
        )
    assert arrival == 6 and rejoin == 22
    assert old.status[agent] == 1 and old.battery[agent] == 0
    assert new.status[agent] == 0 and new.battery[agent] > 0
    assert new._route_cache[agent] is None and new._previous_return_cell[agent] is None
    (tmp_path / f"oscillation_{agent}.json").write_text(
        json.dumps(
            dict(
                scope="Fixed oscillating blocker control, not policy performance",
                agent=agent,
                arrival_v4=arrival,
                rejoin_v4=rejoin,
                history=history,
            ),
            indent=2,
        )
    )


def test_blocked_next_hop_replans_without_immediate_reversal():
    e = make()
    e.status[:] = 3
    e.status[:2] = [1, 0]
    e.uav_positions[:] = [[10, 14], [10, 15], [0, 2], [0, 3], [0, 4]]
    e._route_cache[0] = dict(
        station=2,
        steps=4,
        trigger_steps=4,
        path=[[10, 14], [10, 15], [10, 16], [10, 17], [10, 18]],
        energy_for_path=0.8,
    )
    e._previous_return_cell[0] = [10, 13]
    e._return_planning = True
    p = e._plan(e.uav_positions[0], 0)
    assert p["path"][1] == [9, 14] and e._route_decisions["0"]["reason"] == "replan_blocked"
    assert e._route_decisions["0"]["previous_cell_excluded"]


def test_previous_cell_exclusion_has_necessary_escape_fallback():
    e = make()
    e.status[:] = 3
    e.status[:2] = [1, 0]
    e.uav_positions[:] = [[0, 0], [1, 0], [0, 2], [0, 3], [0, 4]]
    e._previous_return_cell[0] = [0, 1]
    e._return_planning = True
    p = e._plan(e.uav_positions[0], 0)
    assert p["path"][1] == [0, 1]
    assert e._route_decisions["0"]["reason"] == "necessary_backtrack"


def test_trigger_queries_are_pure_and_reset_clears_all_memory():
    e = make()
    e.status[:] = 3
    e.status[0] = 1
    e.uav_positions[0] = [10, 14]
    e.battery[0] = 10
    e.target_station_idx[0] = 2
    audit_step(e, [8] * 5, CommitAuditor())
    memory = deepcopy(e._route_cache)
    previous = deepcopy(e._previous_return_cell)
    e._nearest_station(e.uav_positions[0])
    e._plan(e.uav_positions[0], 0)
    assert e._route_cache == memory and e._previous_return_cell == previous
    e.reset(seed=180402, assignment_seed=180402)
    assert e._route_cache == e._previous_return_cell == [None] * 5


@pytest.mark.parametrize("battery", [0.0, 0.1, 0.19])
def test_unfunded_return_stays_and_never_transmits(battery):
    e = make()
    e.status[:] = 3
    e.status[0] = 1
    e.uav_positions[0] = [10, 16]
    e.battery[0] = battery
    e.target_station_idx[0] = 2
    out, _ = audit_step(e, [8] * 5, CommitAuditor())
    assert e.uav_positions[0].tolist() == [10, 16] and e.battery[0] == 0
    assert out["reward"] == 0 and e.last_return_control["stranded_ids_post"] == [0]


def test_random_controllers_energy_and_route_memory_match_independent_auditor():
    e = make()
    e.battery[:] = [7, 8, 9, 10, 11]
    auditor = CommitAuditor()
    rng = np.random.default_rng(180403)
    for _ in range(100):
        actions = [int(rng.choice(np.flatnonzero(m))) for m in e.get_action_mask()]
        audit_step(e, actions, auditor)
        assert_endpoint_occupancy(e.uav_positions, e.status, e.charging_stations)
    assert auditor.rows == 100


def test_battery_free_curriculum_actor_inputs_and_reward_unchanged():
    c = dict(
        CONDITION,
        stage="S1",
        substage="S1",
        battery=False,
        mobility=False,
        active_count=2,
        starts="fixture",
        stay_only=True,
    )
    a = BS5EnergyReturnEnv(c)
    b = BS5CommittedReturnEnv(c)
    x = a.reset(seed=180404, assignment_seed=180404)
    y = b.reset(seed=180404, assignment_seed=180404)
    for _ in range(5):
        actions = [int(np.flatnonzero(m)[0]) for m in x["action_mask"]]
        x = a.step(actions)
        y = b.step(actions)
        assert x["reward"] == y["reward"]
        for key in ["obs", "state", "action_mask"]:
            np.testing.assert_array_equal(x[key], y[key])
