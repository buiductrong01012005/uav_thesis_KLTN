from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.run_v4_a1_local_100k import (
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
        # Full dynamic role action space: each active UAV can choose both roles.
        mask = out["action_mask"]
        for i in range(env.cfg.num_uavs):
            assert mask[i].sum() >= 2
    for x in ue_snapshots[1:]:
        assert np.array_equal(x, ue_snapshots[0])

    ckpt = discover_a0_checkpoint(None)
    assert ckpt.exists()
    print("A1_SETUP_OK")
    print(f"DIMS obs={env.obs_dim} state={env.state_dim} actions={env.n_actions}")
    print("STATIC_UE=True BATTERY=True DYNAMIC_ROLES=True")
    print(f"A0_CHECKPOINT={ckpt}")


if __name__ == "__main__":
    main()
