"""5A + Recovery: fixed all-Access composition, shared greedy + recovery logic.

The ONE admissible role vector assigns Access to every available UAV.
Movement search is inherited unmodified from ExhaustiveCartesianRoleGreedy;
zero-service recovery is inherited unmodified from _ServiceRecoveryMixin.
This baseline uses privileged exact UE positions/network what-if evaluation,
just like the other search-based references, and is not an information-matched
or complexity-matched comparator for QMIX.
"""

from __future__ import annotations

import numpy as np

from src.baselines.recovery_aware_role_greedy import _ServiceRecoveryMixin
from src.baselines.thesis_exhaustive_cartesian_role_greedy import (
    ExhaustiveCartesianRoleGreedy,
)
from src.envs.v0_env import ACCESS
from src.envs.v4_fixed_mobile_battery_env import ACTIVE


class RecoveryAwareAllAccessGreedy(_ServiceRecoveryMixin, ExhaustiveCartesianRoleGreedy):
    """Search one all-Access role vector with the same movement/recovery policy."""

    def _best_role_vector(self):
        roles = np.full(int(self.env.cfg.num_uavs), ACCESS, dtype=np.int64)
        score = float(self._score(roles))
        return roles, score, 1, {0: 1}

    def act(self, action_mask):
        actions = super().act(action_mask)
        n_active = int(np.sum(self.env.status == ACTIVE))
        self.last_stats.update(
            {
                "requested_composition": "5A",
                "effective_composition": f"{n_active}A0R",
                "selected_relay_ids": [],
                "privileged_current_ue_network_search": True,
                "future_rollout": False,
            }
        )
        return actions
