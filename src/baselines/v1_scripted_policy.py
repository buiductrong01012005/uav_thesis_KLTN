from __future__ import annotations

from itertools import product
import numpy as np

from src.envs.v1_env import SCOUT, ACCESS, RELAY


class V1ScriptedPolicy:
    """Transparent online heuristic for V1 using *revealed information only*.

    Behaviour:
    - before any UE is discovered: all UAVs Scout nearest distinct unknown cells;
    - after some (but not all) UEs are discovered: one UAV keeps Scouting while
      one Access + one Relay serve the currently known demand;
    - once all M UEs are discovered: switch to two Access + one Relay and move
      toward the best static configuration for the discovered map.

    The policy knows M (the scenario parameter) but never reads hidden UE
    positions or ``true_ue_map`` when choosing actions.
    """

    def __init__(self, env):
        self.env = env
        self._ar_cache = {}
        self._aar_cache = {}

    def act(self, action_mask: np.ndarray) -> list[int]:
        discovered = int(self.env.discovered_ue_map.sum())
        total = len(self.env.ue_positions)  # M is assumed known in V1.

        if discovered == 0:
            return self._all_scout(action_mask)

        if discovered < total:
            return self._one_scout_one_access_one_relay(action_mask)

        return self._two_access_one_relay(action_mask)

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def _all_scout(self, action_mask: np.ndarray) -> list[int]:
        unknown = self._unknown_cells()
        if not unknown:
            return [self.env.encode_action(4, SCOUT) for _ in range(self.env.cfg.num_uavs)]

        targets = self._assign_distinct_nearest_targets(self.env.uav_positions, unknown)
        return [
            self.env.encode_action(
                self._move_toward(self.env.uav_positions[i], targets[i], action_mask[i], SCOUT),
                SCOUT,
            )
            for i in range(self.env.cfg.num_uavs)
        ]

    def _one_scout_one_access_one_relay(self, action_mask: np.ndarray) -> list[int]:
        access_target, relay_target = self._best_ar_targets()
        unknown = self._unknown_cells()

        best = None
        # Choose which identity Scouts and assign the remaining two identities
        # to Access / Relay by minimum travel distance.
        for scout_id in range(3):
            remaining = [i for i in range(3) if i != scout_id]
            scout_target = self._nearest_cell(self.env.uav_positions[scout_id], unknown)
            if scout_target is None:
                scout_target = self.env.uav_positions[scout_id].copy()

            for access_id, relay_id in (remaining, remaining[::-1]):
                cost = (
                    self._manhattan(self.env.uav_positions[scout_id], scout_target)
                    + self._manhattan(self.env.uav_positions[access_id], access_target)
                    + self._manhattan(self.env.uav_positions[relay_id], relay_target)
                )
                candidate = (cost, scout_id, access_id, relay_id, scout_target)
                if best is None or candidate[0] < best[0]:
                    best = candidate

        _, scout_id, access_id, relay_id, scout_target = best
        roles = [None] * 3
        targets = [None] * 3
        roles[scout_id], targets[scout_id] = SCOUT, scout_target
        roles[access_id], targets[access_id] = ACCESS, access_target
        roles[relay_id], targets[relay_id] = RELAY, relay_target

        return [
            self.env.encode_action(
                self._move_toward(self.env.uav_positions[i], targets[i], action_mask[i], roles[i]),
                roles[i],
            )
            for i in range(3)
        ]

    def _two_access_one_relay(self, action_mask: np.ndarray) -> list[int]:
        a1, a2, r = self._best_aar_targets()
        target_positions = [a1, a2, r]
        target_roles = [ACCESS, ACCESS, RELAY]

        # Exact identity assignment across only 3! permutations.
        from itertools import permutations

        best_perm = None
        best_cost = None
        for perm in permutations(range(3)):
            cost = sum(
                self._manhattan(self.env.uav_positions[i], target_positions[perm[i]])
                for i in range(3)
            )
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_perm = perm

        actions = []
        for i in range(3):
            target_idx = best_perm[i]
            role = target_roles[target_idx]
            target = target_positions[target_idx]
            move = self._move_toward(self.env.uav_positions[i], target, action_mask[i], role)
            actions.append(self.env.encode_action(move, role))
        return actions

    # ------------------------------------------------------------------
    # Planning from revealed map only
    # ------------------------------------------------------------------

    def _known_ue_positions(self) -> list[np.ndarray]:
        positions: list[np.ndarray] = []
        for r in range(self.env.cfg.grid_size):
            for c in range(self.env.cfg.grid_size):
                count = int(self.env.discovered_ue_map[r, c])
                positions.extend([np.asarray([r, c], dtype=np.int64)] * count)
        return positions

    def _knowledge_key(self):
        return tuple(self.env.discovered_ue_map.ravel().tolist())

    def _best_ar_targets(self):
        key = self._knowledge_key()
        if key in self._ar_cache:
            a, r = self._ar_cache[key]
            return a.copy(), r.copy()
        cells = self._all_cells()
        ues = self._known_ue_positions()
        best_score = None
        best = None

        for access, relay in product(cells, repeat=2):
            served = self._served_by_static_accesses([access], relay, ues)
            geom = self.env._distance(access, relay) + self.env._distance(
                relay, self.env.bs_position
            )
            score = (served, -geom)
            if best_score is None or score > best_score:
                best_score = score
                best = (access.copy(), relay.copy())
        self._ar_cache[key] = tuple(x.copy() for x in best)
        return tuple(x.copy() for x in best)

    def _best_aar_targets(self):
        key = self._knowledge_key()
        if key in self._aar_cache:
            a1, a2, r = self._aar_cache[key]
            return a1.copy(), a2.copy(), r.copy()
        cells = self._all_cells()
        ues = self._known_ue_positions()
        best_score = None
        best = None

        for a1, a2, relay in product(cells, repeat=3):
            served = self._served_by_static_accesses([a1, a2], relay, ues)
            geom = (
                self.env._distance(a1, relay)
                + self.env._distance(a2, relay)
                + self.env._distance(relay, self.env.bs_position)
            )
            score = (served, -geom)
            if best_score is None or score > best_score:
                best_score = score
                best = (a1.copy(), a2.copy(), relay.copy())
        self._aar_cache[key] = tuple(x.copy() for x in best)
        return tuple(x.copy() for x in best)

    def _served_by_static_accesses(self, accesses, relay, ues) -> int:
        rb = self.env.cfg.backhaul_range
        ra = self.env.cfg.access_range
        relay_valid = self.env._distance(relay, self.env.bs_position) <= rb

        valid = []
        for a in accesses:
            direct = self.env._distance(a, self.env.bs_position) <= rb
            via = relay_valid and self.env._distance(a, relay) <= rb
            valid.append(direct or via)

        served = 0
        for ue in ues:
            if any(ok and self.env._distance(ue, a) <= ra for a, ok in zip(accesses, valid)):
                served += 1
        return served

    # ------------------------------------------------------------------
    # Geometry / movement helpers
    # ------------------------------------------------------------------

    def _unknown_cells(self) -> list[np.ndarray]:
        return [
            np.asarray([r, c], dtype=np.int64)
            for r in range(self.env.cfg.grid_size)
            for c in range(self.env.cfg.grid_size)
            if not self.env.known_mask[r, c]
        ]

    def _all_cells(self) -> list[np.ndarray]:
        return [
            np.asarray([r, c], dtype=np.int64)
            for r in range(self.env.cfg.grid_size)
            for c in range(self.env.cfg.grid_size)
        ]

    def _assign_distinct_nearest_targets(self, positions, candidates):
        remaining = [c.copy() for c in candidates]
        targets = []
        # Assign the most constrained/closest next cell one agent at a time.
        for pos in positions:
            if not remaining:
                targets.append(pos.copy())
                continue
            dists = [self._manhattan(pos, c) for c in remaining]
            idx = int(np.argmin(dists))
            targets.append(remaining.pop(idx))
        return targets

    def _nearest_cell(self, position, candidates):
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda c: (self._manhattan(position, c), int(c[0]), int(c[1])),
        ).copy()

    @staticmethod
    def _manhattan(a, b) -> int:
        return int(np.abs(np.asarray(a) - np.asarray(b)).sum())

    def _move_toward(self, current, target, mask, role):
        if np.array_equal(current, target):
            return 4

        dr = int(target[0] - current[0])
        dc = int(target[1] - current[1])
        candidates = []
        if dr < 0:
            candidates.append(0)
        elif dr > 0:
            candidates.append(1)
        if dc > 0:
            candidates.append(2)
        elif dc < 0:
            candidates.append(3)
        candidates.append(4)

        for move_idx in candidates:
            action = self.env.encode_action(move_idx, role)
            if mask[action]:
                return move_idx
        return 4
