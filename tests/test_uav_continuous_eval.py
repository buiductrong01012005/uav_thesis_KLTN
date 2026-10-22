"""Small checks that prove a timestep boundary never resets state automatically."""

import numpy as np
from scripts.evaluation.run_uav_continuous_eval import (
    make_continuous_env,
    chosen_snapshots,
)


def test_continuity_past_training_horizon_and_return_rejoin():
    e = make_continuous_env(seed=3101, min_horizon=1200)
    out = e.reset(seed=3101)
    ue_initial = e.ue_positions.copy()
    battery_initial = e.battery.copy()
    seen_returns = 0
    seen_rejoins = 0
    for t in range(800):
        prev = e.status.copy()
        actions = [
            int(e.encode_action(4, 0))
            if m[int(e.encode_action(4, 0))]
            else int(np.flatnonzero(m)[0])
            for m in out["action_mask"]
        ]
        out = e.step(actions)
        assert out["info"]["t"] == t + 1
        assert not out["truncated"]
        seen_returns += int(np.sum((prev == 0) & (e.status == 1)))
        seen_rejoins += int(np.sum((prev == 2) & (e.status == 0)))
    assert e.t == 800
    assert not np.array_equal(e.ue_positions, ue_initial)
    assert not np.array_equal(e.battery, battery_initial)
    assert seen_returns >= 1
    assert seen_rejoins >= 1


def test_case_frame_times_are_bounded():
    assert chosen_snapshots(150) == {1, 20, 37, 50, 75, 100, 112, 150}
    assert 10000 in chosen_snapshots(10000)
