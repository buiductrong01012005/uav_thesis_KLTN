"""Strong online role/target exhaustive greedy for the target-aware R6 battery env.

For five UAVs the policy explicitly considers all 2^N Access/Relay role vectors.
For every realizable role vector it enumerates every action-valid Relay->Access
target assignment.  It does NOT enumerate the 5^N joint movement space.
After choosing the best role/target configuration at current geometry, it uses a
one-pass coordinate greedy for Access movement; Relay movement is the same
geometry-based assisted one-step controller used by the environment.

This is a privileged small-fleet benchmark: scoring calls the exact simulator
network model and therefore is not presented as a scalable deployed policy.
"""

from __future__ import annotations

from itertools import product
import time
import numpy as np

from src.envs.v0_env import ACCESS, RELAY, MOVE_DELTAS
from src.envs.v4_fixed_mobile_battery_env import ACTIVE
from src.envs.v7_r6_macro_relay_stable_env import RELAY_BASE, STAY


class ExhaustiveRoleTargetGreedy:
    def __init__(self, env):
        self.env = env
        self.last_stats = {}

    def _snapshot(self):
        e = self.env
        return (
            e.uav_positions.copy(),
            e.roles.copy(),
            e.relay_target.copy(),
            e.last_assisted_move.copy(),
        )

    def _restore(self, snap):
        e = self.env
        e.uav_positions[:] = snap[0]
        e.roles[:] = snap[1]
        e.relay_target[:] = snap[2]
        e.last_assisted_move[:] = snap[3]

    def _apply_configuration(self, role_vec, target_map, access_moves=None):
        e = self.env
        base_pos = self._base_positions.copy()
        planned = base_pos.copy()
        if access_moves is not None:
            for i, mv in access_moves.items():
                cand = base_pos[i] + MOVE_DELTAS[int(mv)]
                if e._inside_grid(cand):
                    planned[i] = cand

        e.roles[:] = role_vec
        e.relay_target[:] = -1
        e.last_assisted_move[:] = STAY
        # Place Access actors at the planned cells first. Relay assisted movement
        # is then computed against the target's planned Access position.
        e.uav_positions[:] = planned
        for actor, target in target_map.items():
            move, _ = e._best_relay_plan(int(actor), planned[int(target)])
            cand = base_pos[int(actor)] + MOVE_DELTAS[int(move)]
            if e._inside_grid(cand):
                e.uav_positions[int(actor)] = cand
            else:
                e.uav_positions[int(actor)] = base_pos[int(actor)]
            e.relay_target[int(actor)] = int(target)
            e.last_assisted_move[int(actor)] = int(move)

    def _score(self, role_vec, target_map, access_moves=None):
        snap = self._snapshot()
        try:
            self._apply_configuration(role_vec, target_map, access_moves)
            return float(self.env.compute_network()["utility"])
        finally:
            self._restore(snap)

    def _enumerate_configs(self, mask):
        e = self.env
        active_ids = [i for i in range(e.cfg.num_uavs) if e.status[i] == ACTIVE]
        inactive_ids = [i for i in range(e.cfg.num_uavs) if e.status[i] != ACTIVE]
        role_vectors = 0
        target_configs = 0
        candidates = []

        # Explicitly enumerate every Access/Relay role vector for the active fleet.
        for bits in product((ACCESS, RELAY), repeat=len(active_ids)):
            role_vectors += 1
            roles = np.full(e.cfg.num_uavs, ACCESS, dtype=np.int64)
            for i, r in zip(active_ids, bits):
                roles[i] = int(r)
            for i in inactive_ids:
                roles[i] = ACCESS

            access = [i for i in active_ids if roles[i] == ACCESS]
            relays = [i for i in active_ids if roles[i] == RELAY]
            if not access:
                # 5R (or all-active-relay) is deliberately considered; its
                # immediate service is zero because there is no Access UAV.
                candidates.append((0.0, roles.copy(), {}))
                target_configs += 1
                continue

            if not relays:
                score = self._score(roles, {})
                candidates.append((score, roles.copy(), {}))
                target_configs += 1
                continue

            target_lists = []
            feasible = True
            for actor in relays:
                valid_targets = [
                    target
                    for target in access
                    if RELAY_BASE + int(target) < mask.shape[1]
                    and bool(mask[int(actor), RELAY_BASE + int(target)])
                ]
                if not valid_targets:
                    feasible = False
                    break
                target_lists.append(valid_targets)
            if not feasible:
                # Role vector was still enumerated, but cannot be realized under
                # the current target-aware action mask/request state.
                continue

            for targets in product(*target_lists):
                target_map = {int(a): int(t) for a, t in zip(relays, targets)}
                score = self._score(roles, target_map)
                candidates.append((score, roles.copy(), target_map))
                target_configs += 1

        if not candidates:
            roles = np.full(e.cfg.num_uavs, ACCESS, dtype=np.int64)
            return roles, {}, role_vectors, target_configs
        # Prefer higher utility, then fewer Relays on exact ties.
        candidates.sort(key=lambda z: (z[0], -int(np.sum(z[1] == RELAY))), reverse=True)
        _, roles, target_map = candidates[0]
        return roles, target_map, role_vectors, target_configs

    def act(self, action_mask):
        t0 = time.perf_counter()
        e = self.env
        mask = np.asarray(action_mask)
        self._base_positions = e.uav_positions.copy()
        roles, target_map, n_roles, n_targets = self._enumerate_configs(mask)

        # One-pass coordinate greedy ONLY for Access movement. This avoids 5^N
        # joint movement enumeration while still allowing geometry adaptation.
        access_moves = {}
        for i in range(e.cfg.num_uavs):
            if e.status[i] != ACTIVE or roles[i] != ACCESS:
                continue
            best_move = STAY
            best_score = -np.inf
            for mv in range(5):
                # Cartesian move index and target-aware Access action share 0..4.
                if not bool(mask[i, mv]):
                    continue
                trial = dict(access_moves)
                trial[i] = int(mv)
                score = self._score(roles, target_map, trial)
                if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and mv == STAY):
                    best_score = score
                    best_move = int(mv)
            access_moves[i] = best_move

        actions = [STAY] * e.cfg.num_uavs
        for i in range(e.cfg.num_uavs):
            valid = np.flatnonzero(mask[i])
            if e.status[i] != ACTIVE:
                actions[i] = int(valid[0]) if len(valid) else STAY
            elif roles[i] == RELAY and i in target_map:
                actions[i] = int(RELAY_BASE + target_map[i])
            else:
                actions[i] = int(access_moves.get(i, STAY))

        self.last_stats = {
            "role_vectors_evaluated": int(n_roles),
            "target_configs_evaluated": int(n_targets),
            "selected_relay_count": int(np.sum((roles == RELAY) & (e.status == ACTIVE))),
            "decision_ms": float((time.perf_counter() - t0) * 1000.0),
        }
        return actions
