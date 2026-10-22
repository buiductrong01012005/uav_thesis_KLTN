from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.run_v4_a1_r4_idrand_scratch_100k import make_env, reset_env
from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_a1_identity_randomization import RELAY_ANCHOR_CELL


def main():
    env = make_env()
    assert env.cfg.backhaul_radius_cells == 4.0
    assert env.cfg.enable_ue_mobility is False
    assert env.cfg.enable_battery_cycle is True
    original_positions = sorted(map(tuple, env.start_positions.tolist()))

    ue_ref = None
    layouts = []
    for rid in range(env.cfg.num_uavs):
        out = reset_env(env, rid, 12345 + rid)
        assert tuple(env.uav_positions[rid]) == RELAY_ANCHOR_CELL
        assert sorted(map(tuple, env.uav_positions.tolist())) == original_positions
        assert int(env.roles[rid]) == RELAY
        assert all(int(env.roles[i]) == ACCESS for i in range(env.cfg.num_uavs) if i != rid)
        assert float(env.battery[rid]) == env.cfg.residual_initial_battery
        assert all(
            float(env.battery[i]) == env.cfg.max_battery
            for i in range(env.cfg.num_uavs)
            if i != rid
        )
        assert out["obs"].shape == (env.cfg.num_uavs, env.obs_dim)
        assert out["state"].shape == (env.state_dim,)
        if ue_ref is None:
            ue_ref = env.ue_positions.copy()
        else:
            np.testing.assert_allclose(env.ue_positions, ue_ref)
        layouts.append(env.uav_positions.copy())

    # Same residual ID but another permutation seed should alter at least one
    # non-residual identity position while preserving the anchor and geometry.
    reset_env(env, 0, 999)
    a = env.uav_positions.copy()
    reset_env(env, 0, 1000)
    b = env.uav_positions.copy()
    assert tuple(a[0]) == tuple(b[0]) == RELAY_ANCHOR_CELL
    assert np.any(a[1:] != b[1:])

    print("A1_R4_IDRAND_SETUP_OK")
    print(f"BACKHAUL_RADIUS_CELLS={env.cfg.backhaul_radius_cells}")
    print(f"RELAY_ANCHOR_CELL={RELAY_ANCHOR_CELL}")
    print("INITIALIZATION_DEFAULT=SCRATCH")
    print("STATIC_UE_LAYOUT=True")
    print("IDENTITY_POSITION_RANDOMIZATION=True")


if __name__ == "__main__":
    main()
