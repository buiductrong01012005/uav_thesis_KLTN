"""Collection-time masks and task lineage, without modifying historical replay."""

from dataclasses import dataclass

import numpy as np

from src.rl.qmix import ReplayBuffer, Transition


@dataclass
class CurriculumTransition(Transition):
    current_action_mask: np.ndarray
    stage_id: str
    substage_id: str
    collection_step: int
    lineage: int


class CurriculumReplay(ReplayBuffer):
    def __init__(self, capacity, seed=0):
        super().__init__(capacity, seed)
        self.stage_counts = {}

    def add(self, transition):
        assert all(transition.current_action_mask[i, a] for i, a in enumerate(transition.actions))
        assert transition.next_action_mask.any(axis=1).all()
        if len(self.data) == self.capacity:
            old = self.data[self.pos].substage_id
            self.stage_counts[old] -= 1
            if not self.stage_counts[old]:
                del self.stage_counts[old]
        self.stage_counts[transition.substage_id] = (
            self.stage_counts.get(transition.substage_id, 0) + 1
        )
        super().add(transition)

    def composition(self):
        return dict(self.stage_counts)
