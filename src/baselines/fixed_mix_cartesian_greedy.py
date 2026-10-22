"""Current-state, fixed-composition role search for Cartesian R6 UAV recovery.

Requested patterns with 5 ACTIVE UAVs:
  4A1R: choose the best of C(5,1)=5 Relay identities.
  3A2R: choose the best of C(5,2)=10 Relay pairs.

When a UAV Returns/Charges, protect the minimum Access count (3 whenever
possible) and reduce the Relay budget. Thus both policies produce 3A1R at
four active UAVs; 3A1R is not a separate five-UAV configuration because the
simulator has only Access and Relay roles, no standby/idle role.

Search uses simulator's *current* exact network model (privileged what-if).
After choosing the role vector, inherit the one-pass, five-direction coordinate
movement search used by the exhaustive-role reference. No future lookahead,
no joint 5**N movement enumeration, and no trainable parameters.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from src.baselines.thesis_exhaustive_cartesian_role_greedy import (
    ExhaustiveCartesianRoleGreedy,
)
from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_fixed_mobile_battery_env import ACTIVE


class FixedMixCartesianGreedy(ExhaustiveCartesianRoleGreedy):
    """Best role identities for fixed 4A1R or 3A2R, local move optimization."""

    MIXES = {"4A1R": 1, "3A2R": 2}

    def __init__(self, env, *, composition: str):
        super().__init__(env)
        if composition not in self.MIXES:
            raise ValueError(f"Unsupported composition {composition!r}; choose 4A1R or 3A2R")
        if env.cfg.num_uavs != 5:
            raise ValueError("Fixed mix is defined for the five-UAV thesis scenario")
        self.composition = composition
        self._last_role_ids = ()
        self._last_effective_role_count = 0

    def effective_relay_target(self, n_active: int) -> int:
        # Prioritize >=3 Access when possible. 5 active: requested full pattern.
        # 4 active: 3A1R (with one UAV in Returning/Charging).
        # 3 or fewer active: all remaining Access.
        return max(0, min(self.MIXES[self.composition], n_active - 3))

    def _best_role_vector(self):
        e = self.env
        active_ids = tuple(int(i) for i in np.flatnonzero(e.status == ACTIVE))
        n_relay = self.effective_relay_target(len(active_ids))
        role_vec = np.full(e.cfg.num_uavs, ACCESS, dtype=np.int64)
        best_roles = role_vec.copy()
        best_value = -float("inf")
        chosen = ()
        evaluated = 0

        for relays in combinations(active_ids, n_relay):
            candidate = role_vec.copy()
            if relays:
                candidate[list(relays)] = RELAY
            value = float(self._score(candidate))
            evaluated += 1
            if value > best_value + 1e-12:
                best_value, best_roles, chosen = value, candidate, tuple(relays)
            # lexicographic combinations are enumerated in deterministic order;
            # tie keeps the first one, which is reproducible.

        self._last_role_ids = chosen
        self._last_effective_role_count = n_relay
        return best_roles, best_value, evaluated, {n_relay: evaluated}

    def act(self, action_mask):
        actions = super().act(action_mask)
        active_ids = np.flatnonzero(self.env.status == ACTIVE)
        n_active = len(active_ids)
        n_relays = int(self._last_effective_role_count)
        self.last_stats.update(
            {
                "requested_composition": self.composition,
                "effective_composition": f"{n_active - n_relays}A{n_relays}R",
                "selected_relay_ids": list(self._last_role_ids),
                "privileged_current_ue_network_search": True,
                "future_rollout": False,
            }
        )
        return actions
