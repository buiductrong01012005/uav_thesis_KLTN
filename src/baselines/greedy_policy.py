from __future__ import annotations

from itertools import permutations, product
import numpy as np

from src.envs.v0_env import ACCESS, RELAY


class GreedyScriptedPolicy:
    """
    Transparent V0 baseline with privileged knowledge of the fixed UE map.

    It plans a static 2-Access + 1-Relay target configuration by exhaustive
    search over the small 5x5 grid, then moves each UAV one grid cell per slot
    toward its assigned target. This is intentionally simple and is NOT the
    proposed learning method.

    Purpose: establish that the environment contains a high-service solution
    and provide a strong sanity reference for the later QMIX experiment.
    """

    def __init__(self, env):
        self.env = env
        self.targets = None
        self.target_roles = None
        self.assignment_signature = None

    def act(self, action_mask: np.ndarray) -> list[int]:
        if self.targets is None:
            self._plan_targets()

        actions = []
        for i in range(self.env.cfg.num_uavs):
            move_idx = self._move_toward(self.env.uav_positions[i], self.targets[i], action_mask[i])
            role = int(self.target_roles[i])
            actions.append(self.env.encode_action(move_idx, role))
        return actions

    def _plan_targets(self):
        size = self.env.cfg.grid_size
        cells = [np.asarray([r, c], dtype=np.int64) for r in range(size) for c in range(size)]

        best_score = None
        best_config = None

        # For V0 we deliberately benchmark the intuitive A,A,R structure.
        # Search all static target cells. Repeated target cells are allowed;
        # collision avoidance is outside V0 scope.
        for a1, a2, relay in product(cells, repeat=3):
            served = self._served_count_static(a1, a2, relay)
            if served == 0:
                continue

            # Prefer more served UEs, then shorter total Access-to-UE and
            # relay-to-BS geometry as deterministic tie breakers.
            geom = self._geometry_cost(a1, a2, relay)
            score = (served, -geom)

            if best_score is None or score > best_score:
                best_score = score
                best_config = (a1.copy(), a2.copy(), relay.copy())

        if best_config is None:
            raise RuntimeError("No valid scripted A,A,R target configuration found.")

        target_positions = list(best_config)
        target_roles = [ACCESS, ACCESS, RELAY]

        # Assign current UAV identities to the three targets with minimum
        # total Manhattan travel distance.
        best_perm = None
        best_dist = None
        for perm in permutations(range(3)):
            dist = 0
            for uav_id, target_id in enumerate(perm):
                dist += int(
                    np.abs(self.env.uav_positions[uav_id] - target_positions[target_id]).sum()
                )
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_perm = perm

        self.targets = np.stack([target_positions[j] for j in best_perm], axis=0)
        self.target_roles = np.asarray([target_roles[j] for j in best_perm], dtype=np.int64)

    def _served_count_static(self, a1, a2, relay) -> int:
        R_a = self.env.cfg.access_range
        R_b = self.env.cfg.backhaul_range
        bs = self.env.bs_position

        relay_valid = self.env._distance(relay, bs) <= R_b

        access_valid = []
        for a in (a1, a2):
            direct = self.env._distance(a, bs) <= R_b
            via_relay = relay_valid and self.env._distance(a, relay) <= R_b
            access_valid.append(direct or via_relay)

        served = 0
        for ue in self.env.ue_positions:
            ok = False
            for a, valid in zip((a1, a2), access_valid):
                if valid and self.env._distance(ue, a) <= R_a:
                    ok = True
                    break
            served += int(ok)

        return served

    def _geometry_cost(self, a1, a2, relay) -> float:
        bs = self.env.bs_position
        # Small deterministic tie-breaker; service count always dominates.
        return (
            self.env._distance(relay, bs)
            + self.env._distance(a1, relay)
            + self.env._distance(a2, relay)
        )

    def _move_toward(self, current, target, mask):
        if np.array_equal(current, target):
            return 4  # Stay

        # Candidate Manhattan-reducing moves; choose fixed order for
        # determinism. We verify validity against the environment action mask.
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

        candidates.append(4)  # Stay fallback

        role = ACCESS  # either role shares the same movement validity
        for move_idx in candidates:
            action = self.env.encode_action(move_idx, role)
            if mask[action]:
                return move_idx

        return 4
