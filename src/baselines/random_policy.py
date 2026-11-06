from __future__ import annotations

import numpy as np


class RandomPolicy:
    """Samples uniformly from valid per-agent actions."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, action_mask: np.ndarray) -> list[int]:
        actions: list[int] = []
        for mask in action_mask:
            valid = np.flatnonzero(mask)
            if len(valid) == 0:
                raise RuntimeError("Agent has no valid action.")
            actions.append(int(self.rng.choice(valid)))
        return actions
