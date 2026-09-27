from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v4_a0_kaggle_long import ROLES, make_env, reset_env, step_env

# Collision-free paths from the fixed A0 start deployment to the known
# coordinate-ascent placement. Move indices: N=0,S=1,E=2,W=3,Stay=4.
PATHS = [
    [0, 0, 0, 3, 3],             # UAV0 -> (5,0)
    [0, 0, 0, 0],                # UAV1 -> (6,3)
    [1, 1],                      # UAV2 -> (14,4)
    [2],                         # UAV3 -> (9,6)
    [2, 2, 0, 0, 0, 0, 0, 0],  # UAV4 relay -> (5,8), avoids UAV3
]
TARGET_POSITIONS = np.asarray(
    [[5, 0], [6, 3], [14, 4], [9, 6], [5, 8]], dtype=np.int64
)


def run_reference() -> dict:
    env = make_env()
    out = reset_env(env)
    utilities: list[float] = []
    arrival_step = None

    for t in range(env.cfg.horizon):
        actions = []
        for i, path in enumerate(PATHS):
            move = path[t] if t < len(path) else 4
            actions.append(env.encode_action(move, int(ROLES[i])))
        out = step_env(env, actions)
        utilities.append(float(out["reward"]))
        if arrival_step is None and np.array_equal(env.uav_positions, TARGET_POSITIONS):
            arrival_step = t + 1

    result = {
        "arrival_step": int(arrival_step or -1),
        "final_positions": env.uav_positions.tolist(),
        "mean_episode_utility": float(np.mean(utilities)),
        "final_utility": float(utilities[-1]),
        "last100_mean_utility": float(np.mean(utilities[-100:])),
    }
    return result


def main():
    result = run_reference()
    assert result["arrival_step"] == 8, result
    assert result["final_positions"] == TARGET_POSITIONS.tolist(), result
    assert result["final_utility"] > 0.72, result
    assert result["mean_episode_utility"] > 0.70, result
    print("A0_REFERENCE_OK", json.dumps(result))


if __name__ == "__main__":
    main()
