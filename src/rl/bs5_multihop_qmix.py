"""Same spatial body and QMIX; H-only has a 5-move head, no role decision."""

from copy import deepcopy
import torch
from torch import nn
from src.rl.bs5_hybrid_qmix import HybridQMIX, make_learner as legacy_learner


class HOnlyQMIX(HybridQMIX):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = 5
        self.agent.head[-1] = nn.Linear(128, 5).to(self.device)
        self.target_agent = deepcopy(self.agent)
        self.optimizer = torch.optim.Adam(
            list(self.agent.parameters()) + list(self.mixer.parameters()), lr=kwargs.get("lr", 3e-4)
        )


def make_learner(arm, **kwargs):
    return HOnlyQMIX(**kwargs) if arm == "H-only" else legacy_learner(arm, **kwargs)
