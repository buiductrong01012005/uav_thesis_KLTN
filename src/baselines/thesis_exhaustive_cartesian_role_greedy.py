"""Role-exhaustive fast dynamic baseline for the thesis R6 environment.

The policy explicitly evaluates every Access/Relay role vector of the currently
Active fleet (2^N for N active UAVs), including all-Access and all-Relay.  It
uses the exact simulator network utility at the *current* geometry to choose a
role vector.  It then performs one deterministic coordinate-greedy pass over
movement for that selected vector only.

This intentionally avoids exponential joint movement enumeration (5^N) and any
future rollout.  For the five-UAV thesis setting it is a strong, transparent
online heuristic, but the 2^N role enumeration is not claimed to scale to large
fleets.
"""

from __future__ import annotations

from itertools import product
import time

import numpy as np

from src.envs.v0_env import ACCESS, RELAY, MOVE_DELTAS
from src.envs.v4_fixed_mobile_battery_env import ACTIVE

STAY = 4


class ExhaustiveCartesianRoleGreedy:
    def __init__(self, env):
        self.env = env
        self.last_stats: dict = {}

    def _snapshot(self):
        e = self.env
        return e.uav_positions.copy(), e.roles.copy()

    def _restore(self, snap):
        e = self.env
        e.uav_positions[:] = snap[0]
        e.roles[:] = snap[1]

    def _score(self, role_vec: np.ndarray, moves: dict[int, int] | None = None) -> float:
        """Immediate utility after applying a role vector and optional one-step moves.

        Collision resolution uses the same environment helper as the actual step,
        so the scored geometry matches the executable Cartesian actions.
        """
        e = self.env
        snap = self._snapshot()
        try:
            old = snap[0]
            proposed = old.copy()
            if moves:
                for i, mv in moves.items():
                    cand = old[int(i)] + MOVE_DELTAS[int(mv)]
                    if e._inside_grid(cand):
                        proposed[int(i)] = cand
            e._resolve_collisions(proposed, old, e.status.copy())
            e.uav_positions[:] = proposed
            e.roles[:] = role_vec
            return float(e.compute_network()["utility"])
        finally:
            self._restore(snap)

    def _best_role_vector(self):
        e = self.env
        active_ids = [int(i) for i in range(e.cfg.num_uavs) if e.status[i] == ACTIVE]
        inactive_ids = [int(i) for i in range(e.cfg.num_uavs) if e.status[i] != ACTIVE]

        best_roles = np.full(e.cfg.num_uavs, ACCESS, dtype=np.int64)
        best_score = -np.inf
        evaluated = 0
        family_counts = {r: 0 for r in range(len(active_ids) + 1)}

        for bits in product((ACCESS, RELAY), repeat=len(active_ids)):
            roles = np.full(e.cfg.num_uavs, ACCESS, dtype=np.int64)
            for i, role in zip(active_ids, bits):
                roles[i] = int(role)
            for i in inactive_ids:
                roles[i] = ACCESS  # role ignored while unavailable

            n_relay = int(sum(int(r == RELAY) for r in bits))
            family_counts[n_relay] += 1
            score = self._score(roles)
            evaluated += 1

            # Deterministic tie-break: prefer fewer Relays, then lexicographic
            # role vector.  The preference only applies to exact utility ties.
            current_relays = int(np.sum(best_roles[active_ids] == RELAY)) if active_ids else 0
            if score > best_score + 1e-12:
                best_score = score
                best_roles = roles.copy()
            elif abs(score - best_score) <= 1e-12:
                if n_relay < current_relays:
                    best_roles = roles.copy()
                elif n_relay == current_relays and tuple(roles.tolist()) < tuple(
                    best_roles.tolist()
                ):
                    best_roles = roles.copy()

        return best_roles, float(best_score), int(evaluated), family_counts

    def act(self, action_mask):
        t0 = time.perf_counter()
        e = self.env
        mask = np.asarray(action_mask)

        roles, role_score, n_roles, family_counts = self._best_role_vector()

        # One-pass coordinate greedy movement for the selected role vector only.
        # Earlier chosen moves remain fixed while later UAVs are optimized.
        chosen_moves: dict[int, int] = {}
        movement_evals = 0
        for i in range(e.cfg.num_uavs):
            if e.status[i] != ACTIVE:
                continue
            best_move = STAY
            best_score = -np.inf
            for mv in range(5):
                action = e.encode_action(mv, int(roles[i]))
                if not bool(mask[i, action]):
                    continue
                trial = dict(chosen_moves)
                trial[i] = int(mv)
                score = self._score(roles, trial)
                movement_evals += 1
                if score > best_score + 1e-12:
                    best_score = score
                    best_move = int(mv)
                elif abs(score - best_score) <= 1e-12 and mv == STAY:
                    best_move = STAY
            chosen_moves[i] = best_move

        actions = []
        noop = e.encode_action(STAY, ACCESS)
        for i in range(e.cfg.num_uavs):
            if e.status[i] != ACTIVE:
                actions.append(int(noop))
                continue
            action = int(e.encode_action(chosen_moves.get(i, STAY), int(roles[i])))
            if not bool(mask[i, action]):
                raise RuntimeError(f"constructed invalid action {action} for UAV {i}")
            actions.append(action)

        active_ids = np.flatnonzero(e.status == ACTIVE)
        n_active = int(len(active_ids))
        n_relay = int(np.sum(roles[active_ids] == RELAY)) if n_active else 0
        self.last_stats = {
            "role_vectors_evaluated": n_roles,
            "expected_role_vectors": int(2**n_active),
            "role_family_vector_counts": {str(k): int(v) for k, v in family_counts.items()},
            "selected_relay_count": n_relay,
            "selected_access_count": int(n_active - n_relay),
            "selected_role_vector": [int(x) for x in roles.tolist()],
            "role_only_utility": role_score,
            "movement_candidates_evaluated": int(movement_evals),
            "decision_ms": float((time.perf_counter() - t0) * 1000.0),
        }
        return actions
