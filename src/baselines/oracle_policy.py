from __future__ import annotations

from itertools import permutations, product
import numpy as np

from src.envs.v0_env import ACCESS, RELAY


class StaticOraclePolicy:
    """
    Privileged V0 static oracle.

    The oracle knows the complete UE map and exhaustively searches ALL static
    configurations of UAV positions and Access/Relay roles. For the default
    5x5, 3-UAV V0 this is only:

        (25 cells * 2 roles)^3 = 125,000 configurations.

    Objective (lexicographic):
      1) maximize number of served UEs;
      2) minimize total Manhattan travel from the current deployment;
      3) minimize a small communication-geometry cost for deterministic ties.

    It then moves each UAV one cell per slot toward that globally best static
    configuration. Therefore it is an exact oracle for the best *static*
    steady-state served-user ratio under the V0 model. It is NOT claimed to be
    an exact finite-horizon trajectory oracle, because it does not optimize the
    cumulative transient reward along the travel path.
    """

    def __init__(self, env):
        self.env = env
        self.targets: np.ndarray | None = None
        self.target_roles: np.ndarray | None = None
        self.oracle_served_count: int | None = None
        self.oracle_served_ratio: float | None = None
        self.target_travel_distance: int | None = None

    def act(self, action_mask: np.ndarray) -> list[int]:
        if self.targets is None:
            self._plan_targets()

        actions: list[int] = []
        for i in range(self.env.cfg.num_uavs):
            move_idx = self._move_toward(
                self.env.uav_positions[i], self.targets[i], action_mask[i]
            )
            role = int(self.target_roles[i])
            actions.append(self.env.encode_action(move_idx, role))
        return actions

    def _plan_targets(self):
        size = self.env.cfg.grid_size
        cells = [
            np.asarray([r, c], dtype=np.int64)
            for r in range(size)
            for c in range(size)
        ]

        best_score = None
        best_positions = None
        best_roles = None
        best_travel = None

        # Enumerate ordered UAV states directly. UAV identities are allowed to
        # occupy the same cell because collision avoidance is outside V0.
        for positions_tuple in product(cells, repeat=self.env.cfg.num_uavs):
            positions = np.stack(positions_tuple, axis=0)

            for roles_tuple in product((ACCESS, RELAY), repeat=self.env.cfg.num_uavs):
                roles = np.asarray(roles_tuple, dtype=np.int64)
                served = self._served_count_static(positions, roles)

                travel = int(
                    np.abs(self.env.uav_positions - positions).sum()
                )
                geom = self._geometry_cost(positions, roles)

                # Served count dominates. Among equal service configurations,
                # prefer less travel from the actual starting deployment, then
                # deterministic smaller geometry cost.
                score = (served, -travel, -geom)
                if best_score is None or score > best_score:
                    best_score = score
                    best_positions = positions.copy()
                    best_roles = roles.copy()
                    best_travel = travel

        if best_positions is None or best_roles is None:
            raise RuntimeError("Oracle failed to find a static V0 configuration.")

        self.targets = best_positions
        self.target_roles = best_roles
        self.oracle_served_count = int(best_score[0])
        self.oracle_served_ratio = self.oracle_served_count / len(self.env.ue_positions)
        self.target_travel_distance = int(best_travel)

    def _served_count_static(self, positions: np.ndarray, roles: np.ndarray) -> int:
        R_a = self.env.cfg.access_range
        R_b = self.env.cfg.backhaul_range
        bs = self.env.bs_position

        access_ids = np.flatnonzero(roles == ACCESS)
        relay_ids = np.flatnonzero(roles == RELAY)

        valid_access = np.zeros(self.env.cfg.num_uavs, dtype=bool)

        for access_id in access_ids:
            a = positions[access_id]

            if self.env._distance(a, bs) <= R_b:
                valid_access[access_id] = True
                continue

            for relay_id in relay_ids:
                relay = positions[relay_id]
                if (
                    self.env._distance(a, relay) <= R_b
                    and self.env._distance(relay, bs) <= R_b
                ):
                    valid_access[access_id] = True
                    break

        served = 0
        for ue in self.env.ue_positions:
            for access_id in np.flatnonzero(valid_access):
                if self.env._distance(ue, positions[access_id]) <= R_a:
                    served += 1
                    break

        return served

    def _geometry_cost(self, positions: np.ndarray, roles: np.ndarray) -> float:
        """Only a deterministic tertiary tie-breaker; never changes service."""
        bs = self.env.bs_position
        cost = 0.0

        access_ids = np.flatnonzero(roles == ACCESS)
        relay_ids = np.flatnonzero(roles == RELAY)

        for relay_id in relay_ids:
            cost += self.env._distance(positions[relay_id], bs)

        for access_id in access_ids:
            a = positions[access_id]
            direct = self.env._distance(a, bs)
            if len(relay_ids) == 0:
                cost += direct
            else:
                via = min(
                    self.env._distance(a, positions[r])
                    + self.env._distance(positions[r], bs)
                    for r in relay_ids
                )
                cost += min(direct, via)

        return float(cost)

    def _move_toward(self, current, target, mask):
        if np.array_equal(current, target):
            return 4  # Stay

        dr = int(target[0] - current[0])
        dc = int(target[1] - current[1])

        candidates = []
        if dr < 0:
            candidates.append(0)  # North
        elif dr > 0:
            candidates.append(1)  # South

        if dc > 0:
            candidates.append(2)  # East
        elif dc < 0:
            candidates.append(3)  # West

        candidates.append(4)

        # Movement validity does not depend on role in V0.
        for move_idx in candidates:
            action = self.env.encode_action(move_idx, ACCESS)
            if mask[action]:
                return move_idx

        return 4
