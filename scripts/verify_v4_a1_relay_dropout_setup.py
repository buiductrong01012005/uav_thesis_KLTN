from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.v0_env import ACCESS, RELAY
from scripts.run_v4_a1_relay_dropout_100k import (
    make_env,
    reset_env,
    discover_a0_checkpoint,
)


def main():
    env = make_env()
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

        # Full dynamic role action space remains available to every Active UAV.
        mask = out["action_mask"]
        for i in range(env.cfg.num_uavs):
            allowed_roles = set()
            for a in np.flatnonzero(mask[i]):
                _, role = env.decode_action(int(a))
                allowed_roles.add(int(role))
            assert ACCESS in allowed_roles
            assert RELAY in allowed_roles

    for x in ue_snapshots[1:]:
        assert np.array_equal(x, ue_snapshots[0])

    ckpt = discover_a0_checkpoint(None)
    assert ckpt.exists()
    print("A1_RELAY_DROPOUT_SETUP_OK")
    print(f"DIMS obs={env.obs_dim} state={env.state_dim} actions={env.n_actions}")
    print("STATIC_UE=True BATTERY=True DYNAMIC_ROLES=True")
    print("INITIAL_ROLE_RULE=residual UAV is sole Relay; others are Access")
    print(f"A0_CHECKPOINT={ckpt}")


if __name__ == "__main__":
    main()
