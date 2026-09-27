import numpy as np

from scripts.audit_v4_a1_r4_identity_randomization import (
    RELAY_ANCHOR,
    randomized_identity_layout,
    reset_identity_randomized,
)
from scripts.run_v4_a1_r4_100k import make_env
from src.envs.v0_env import ACCESS, RELAY


def test_randomized_identity_layout_preserves_geometry_and_pins_residual():
    env = make_env()
    original = sorted(map(tuple, env.start_positions.tolist()))
    for rid in range(env.cfg.num_uavs):
        layout = randomized_identity_layout(rid, 1234 + rid)
        assert tuple(layout[rid]) == RELAY_ANCHOR
        assert sorted(map(tuple, layout.tolist())) == original
        assert len({tuple(x) for x in layout.tolist()}) == env.cfg.num_uavs


def test_reset_identity_randomized_sets_only_residual_relay_and_battery():
    env = make_env()
    for rid in range(env.cfg.num_uavs):
        reset_identity_randomized(env, rid, 900 + rid, 30.0)
        assert tuple(env.uav_positions[rid]) == RELAY_ANCHOR
        assert int(env.roles[rid]) == RELAY
        assert all(int(env.roles[i]) == ACCESS for i in range(env.cfg.num_uavs) if i != rid)
        assert float(env.battery[rid]) == 30.0
        assert all(float(env.battery[i]) == env.cfg.max_battery for i in range(env.cfg.num_uavs) if i != rid)
