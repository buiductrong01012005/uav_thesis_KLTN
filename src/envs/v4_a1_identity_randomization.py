from __future__ import annotations

import numpy as np

from src.envs.v0_env import ACCESS, RELAY

RELAY_ANCHOR_CELL = (11, 6)


def randomized_identity_layout(env, residual_uav_idx: int, permutation_seed: int) -> np.ndarray:
    """Permute UAV identities over the original five start cells.

    The residual/low-battery UAV is always placed at the known useful relay
    anchor. The remaining four UAV IDs are randomly assigned to the remaining
    four original start cells. Geometry is preserved exactly; only ID<->position
    mapping changes.
    """
    base = np.asarray(env.start_positions, dtype=np.int64)
    anchor = np.asarray(RELAY_ANCHOR_CELL, dtype=np.int64)
    matches = np.where(np.all(base == anchor[None, :], axis=1))[0]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one relay anchor {RELAY_ANCHOR_CELL}, found {len(matches)}"
        )
    anchor_slot = int(matches[0])

    rid = int(residual_uav_idx)
    if not 0 <= rid < len(base):
        raise ValueError(f"residual_uav_idx out of range: {rid}")

    remaining_positions = np.delete(base, anchor_slot, axis=0)
    other_ids = [i for i in range(len(base)) if i != rid]
    rng = np.random.default_rng(int(permutation_seed))
    shuffled_ids = [other_ids[int(i)] for i in rng.permutation(len(other_ids))]

    layout = np.empty_like(base)
    layout[rid] = anchor
    for pos, uid in zip(remaining_positions, shuffled_ids):
        layout[uid] = pos
    return layout


def reset_identity_randomized(
    env,
    residual_uav_idx: int,
    permutation_seed: int,
    residual_battery: float | None = None,
):
    """Reset the fixed UE layout and apply randomized ID<->position mapping."""
    out = env.reset(seed=env.cfg.seed)
    rid = int(residual_uav_idx)
    env.residual_uav_idx = rid
    env.uav_positions[:] = randomized_identity_layout(env, rid, permutation_seed)
    env.roles[:] = ACCESS
    env.roles[rid] = RELAY
    env.battery[:] = env.cfg.max_battery
    if residual_battery is None:
        residual_battery = env.cfg.residual_initial_battery
    env.battery[rid] = float(residual_battery)

    out["obs"] = env.get_obs()
    out["state"] = env.get_state()
    out["action_mask"] = env.get_action_mask()
    out["episode_meta"] = env._episode_meta()
    return out
