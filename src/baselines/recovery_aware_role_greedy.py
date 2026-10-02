"""Diagnostic navigation repair for the thesis current-state role-search policies.

This is NOT an RL method or a new online baseline claim: it uses privileged
simulator UE positions, as the existing search references already do.
It preserves the existing role-selection rule and adds a multi-cell navigation
fallback ONLY after actual service has been zero for a configurable number of
consecutive steps. No future environment rollout and no teleporting.

Purpose: separate one-step greedy-navigation failure from role-composition
constraints. Report original and repaired variants separately.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.baselines.thesis_exhaustive_cartesian_role_greedy import ExhaustiveCartesianRoleGreedy
from src.baselines.fixed_mix_cartesian_greedy import FixedMixCartesianGreedy
from src.envs.v0_env import ACCESS, RELAY, MOVE_DELTAS
from src.envs.v4_fixed_mobile_battery_env import ACTIVE


class _ServiceRecoveryMixin:
    """One-step movement toward promising DIRECTLY backhauled Access goals."""

    def __init__(self, *args, zero_service_patience: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        if zero_service_patience < 1:
            raise ValueError("zero_service_patience must be >= 1")
        self.zero_service_patience = int(zero_service_patience)
        self.zero_service_run = 0
        self.recovery_steps = 0
        self.recovery_active = False
        self._last_goal_positions: dict[int, tuple[int, int]] = {}

    def observe_service(self, actual_reward: float) -> None:
        """Call AFTER the actual env.step; never use hypothetical what-if reward."""
        if float(actual_reward) <= 1e-10:
            self.zero_service_run += 1
        else:
            self.zero_service_run = 0

    def _access_goal_candidates(self) -> tuple[np.ndarray, np.ndarray]:
        """All grid cells with actual UE coverage and a feasible direct BS path.

        We use direct-backhaul goals to avoid dependence on the relay's current
        position while the system is entirely disconnected. The method is
        privileged, so do NOT call it information-equivalent to QMIX.
        """
        e = self.env
        grid = np.indices((e.cfg.grid_size, e.cfg.grid_size)).reshape(2, -1).T
        # Simulator cells are (row, col); physical positions are (x=col, y=row).
        goal_xy = (grid[:, ::-1].astype(float) + 0.5) * float(e.cfg.cell_size_m)
        gateways = np.asarray(e.surviving_bs_xy, dtype=float)
        max_bh = float(e.cfg.backhaul_radius_cells) * float(e.cfg.cell_size_m)
        can_backhaul = np.min(np.linalg.norm(goal_xy[:, None, :] - gateways[None, :, :], axis=2), axis=1) <= max_bh + 1e-9
        counts = np.sum(
            np.linalg.norm(grid[:, None, :].astype(float) - np.asarray(e.ue_positions)[None, :, :], axis=2)
            <= float(e.cfg.access_radius_cells) + 1e-9,
            axis=1,
        )
        mask = can_backhaul & (counts > 0)
        return grid[mask], np.minimum(counts[mask], e.cfg.max_ues_per_access).astype(float)

    def _recovery_actions(self, base_actions: list[int], mask: np.ndarray) -> list[int]:
        e = self.env
        roles = [e.decode_action(int(a))[1] for a in base_actions]
        access_ids = [i for i in range(e.cfg.num_uavs)
                      if e.status[i] == ACTIVE and roles[i] == ACCESS]
        goals, coverage = self._access_goal_candidates()
        if not len(goals) or not access_ids:
            self._last_goal_positions = {}
            return base_actions

        # Assign nearby promising, spatially distinct goals to current Access
        # UAVs; no forecast or hypothetical future state is queried.
        occupied_goals: list[np.ndarray] = []
        targets: dict[int, np.ndarray] = {}
        for uid in access_ids:
            dist = np.sum(np.abs(goals - e.uav_positions[uid][None, :]), axis=1)
            # Weight covered users but do not blindly send everyone to one
            # hotspot; nearest viable goal is preferred when coverage ties.
            value = (1.5 * coverage - .8 * dist).astype(float)
            for assigned in occupied_goals:
                separation = np.linalg.norm(goals - assigned[None, :], axis=1)
                value -= 12.0 * np.exp(-(separation / 3.0)**2)
            goal = goals[int(np.argmax(value))]
            targets[uid] = goal.copy()
            occupied_goals.append(goal.copy())

        actions = list(base_actions)
        reserved: set[tuple[int, int]] = set()
        for uid in access_ids:
            target = targets[uid]
            old = np.asarray(e.uav_positions[uid])
            candidates = []
            for mv in range(5):
                a = int(e.encode_action(mv, ACCESS))
                if not bool(mask[uid, a]):
                    continue
                nxt = old + MOVE_DELTAS[mv]
                if not e._inside_grid(nxt):
                    continue
                distance = float(np.sum(np.abs(nxt - target)))
                occupied = (int(nxt[0]), int(nxt[1])) in reserved
                # deterministic tie-break: don't prefer Stay on zero-reward
                # plateaus unless it is actually the closest to the target.
                candidates.append((distance, occupied, int(mv == 4), mv, a, nxt))
            if not candidates:
                continue
            _, _, _, _, action, nxt = min(candidates, key=lambda q: (q[1], q[0], q[2], q[3]))
            actions[uid] = action
            reserved.add((int(nxt[0]), int(nxt[1])))

        self._last_goal_positions = {int(k): tuple(map(int, v)) for k,v in targets.items()}
        return actions

    def act(self, action_mask):
        actions = list(super().act(action_mask))
        self.recovery_active = self.zero_service_run >= self.zero_service_patience
        if self.recovery_active:
            actions = self._recovery_actions(actions, np.asarray(action_mask))
            self.recovery_steps += 1
        self.last_stats.update({
            "zero_service_run_before_decision": self.zero_service_run,
            "recovery_active": self.recovery_active,
            "recovery_steps": self.recovery_steps,
            "recovery_access_goals": self._last_goal_positions.copy(),
        })
        return actions


class RecoveryAwareFixedMixGreedy(_ServiceRecoveryMixin, FixedMixCartesianGreedy):
    pass


class RecoveryAwareExhaustiveGreedy(_ServiceRecoveryMixin, ExhaustiveCartesianRoleGreedy):
    pass
