import numpy as np

from scripts.training.run_v4_a1_r4_idrand_scratch_100k import make_env, reset_env
from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_a1_identity_randomization import RELAY_ANCHOR_CELL


def test_idrand_runner_preserves_positions_and_pins_residual():
    env = make_env()
    expected = sorted(map(tuple, env.start_positions.tolist()))
    for rid in range(5):
        reset_env(env, rid, 200 + rid)
        assert tuple(env.uav_positions[rid]) == RELAY_ANCHOR_CELL
        assert sorted(map(tuple, env.uav_positions.tolist())) == expected
        assert env.roles[rid] == RELAY
        assert sum(int(x == RELAY) for x in env.roles) == 1
        assert all(env.roles[i] == ACCESS for i in range(5) if i != rid)


def test_idrand_runner_changes_nonresidual_mapping_across_seeds():
    env = make_env()
    reset_env(env, 2, 300)
    a = env.uav_positions.copy()
    reset_env(env, 2, 301)
    b = env.uav_positions.copy()
    assert tuple(a[2]) == tuple(b[2]) == RELAY_ANCHOR_CELL
    assert np.any(np.delete(a, 2, axis=0) != np.delete(b, 2, axis=0))
