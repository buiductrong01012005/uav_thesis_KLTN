from __future__ import annotations

from itertools import product
import numpy as np

from src.envs.stage_a_scout_env import MOVE_DELTAS


class GreedyCoveragePolicy:
    """Online Scout heuristic using only the current known-mask and positions.

    It evaluates all valid joint one-step movements (at most 5^3=125) and
    chooses the joint action that maximizes the number of currently unknown
    cells that would be revealed after the move. Tie-breakers discourage
    redundant footprint overlap and unnecessary movement.

    It never reads the hidden UE map.

    Footprints and per-agent candidate positions are cached/vectorized for
    speed.  The scoring rule is identical to the original implementation.
    """

    def __init__(self, env):
        self.env = env
        g = env.cfg.grid_size
        self._footprints = {
            (r, c): frozenset(env._scout_footprint(np.asarray([r, c], dtype=np.int64)))
            for r in range(g)
            for c in range(g)
        }

    def act(self, action_mask: np.ndarray) -> list[int]:
        n = self.env.cfg.num_uavs
        valid_per_agent = [np.flatnonzero(action_mask[i]).tolist() for i in range(n)]

        # Precompute the result of every valid single-agent move once.  The
        # previous code recomputed the same footprint inside every joint tuple.
        next_pos: list[dict[int, np.ndarray]] = []
        footprints: list[dict[int, frozenset[tuple[int, int]]]] = []
        for i in range(n):
            p_i, f_i = {}, {}
            for a in valid_per_agent[i]:
                p = self.env.uav_positions[i] + MOVE_DELTAS[a]
                p_i[a] = p
                f_i[a] = self._footprints[(int(p[0]), int(p[1]))]
            next_pos.append(p_i)
            footprints.append(f_i)

        unknown_cells = np.argwhere(~self.env.known_mask)
        best_score = None
        best_actions = None

        for actions in product(*valid_per_agent):
            positions = np.stack([next_pos[i][a] for i, a in enumerate(actions)], axis=0)
            fps = [footprints[i][a] for i, a in enumerate(actions)]
            union = set().union(*fps)
            new_cells = sum(not self.env.known_mask[cell] for cell in union)
            overlap = sum(len(fp) for fp in fps) - len(union)
            moves = sum(a != 4 for a in actions)

            # Exact same tie-breaker as before, but evaluate all unknown cells
            # and all next UAV positions with NumPy rather than nested loops.
            if len(unknown_cells):
                manhattan = np.abs(unknown_cells[:, None, :] - positions[None, :, :]).sum(axis=2)
                nearest = manhattan.min(axis=1)
                frontier_distance = int(np.maximum(nearest - 1, 0).sum())
            else:
                frontier_distance = 0

            score = (new_cells, -frontier_distance, -overlap, -moves)
            if best_score is None or score > best_score:
                best_score = score
                best_actions = list(actions)

        return best_actions
