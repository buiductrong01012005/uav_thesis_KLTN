from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.v0_env import ACCESS, RELAY
from scripts.training.run_v4_a1_relay_dropout_100k import make_env as make_r5_env
from scripts.training.run_v4_a1_r4_100k import (
    make_env,
    reset_env,
    discover_a0_checkpoint,
)


def main():
    env = make_env()
    r5_env = make_r5_env()

    # This audit is intentionally one-factor-at-a-time: only backhaul radius changes.
    r4_cfg = asdict(env.cfg)
    r5_cfg = asdict(r5_env.cfg)
    diffs = {k: (r5_cfg[k], r4_cfg[k]) for k in r5_cfg if r5_cfg[k] != r4_cfg[k]}
    assert diffs == {"backhaul_radius_cells": (5.0, 4.0)}, diffs
    assert np.isclose(env.cfg.access_radius_cells, 4.0)
    assert np.isclose(env.cfg.backhaul_radius_cells, 4.0)

    ue_snapshots = []
    for residual in range(env.cfg.num_uavs):
        out = reset_env(env, residual)
        ue_snapshots.append(env.ue_positions.copy())
        assert env.cfg.enable_ue_mobility is False
        assert env.cfg.enable_battery_cycle is True
        assert int(env.residual_uav_idx) == residual
        assert np.isclose(env.battery[residual], 30.0)

        others = [i for i in range(env.cfg.num_uavs) if i != residual]
        assert np.allclose(env.battery[others], 100.0)
        assert int(env.roles[residual]) == RELAY
        assert np.all(env.roles[others] == ACCESS)
        assert int(np.sum(env.roles == RELAY)) == 1

        # Full movement x role action space remains available to Active UAVs.
        mask = out["action_mask"]
        for i in range(env.cfg.num_uavs):
            allowed_roles = {env.decode_action(int(a))[1] for a in np.flatnonzero(mask[i])}
            assert ACCESS in allowed_roles
            assert RELAY in allowed_roles

    for x in ue_snapshots[1:]:
        assert np.array_equal(x, ue_snapshots[0])

    ckpt = discover_a0_checkpoint(None)
    assert ckpt.exists()
    print("A1_R4_SETUP_OK")
    print(f"DIMS obs={env.obs_dim} state={env.state_dim} actions={env.n_actions}")
    print("STATIC_UE=True BATTERY=True DYNAMIC_ROLES=True")
    print("INITIAL_ROLE_RULE=residual UAV is sole Relay; others are Access")
    print("ACCESS_RADIUS_CELLS=4.0")
    print("BACKHAUL_RADIUS_CELLS=4.0")
    print("ONLY_CONFIG_DIFF_FROM_A1_R5=backhaul_radius_cells:5.0->4.0")
    print(f"A0_CHECKPOINT={ckpt}")


if __name__ == "__main__":
    main()
