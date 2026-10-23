"""Versioned training conditions; target BS5 physics and reward stay unchanged."""

from copy import deepcopy

import numpy as np

from src.envs.bs5_launch_ab_env import BS5LaunchConfig, BS5LaunchEnv, LAUNCHES
from src.envs.v0_env import MOVE_DELTAS

INACTIVE = 3  # Explicit curriculum absence, not Returning/Charging or dead battery.
VERSION = "bs5_fixed_stage_curriculum_v1"


def condition_for(step, arm, protocol):
    """All boundaries are episode boundaries; gates never control the schedule."""
    ends = protocol["stage_ends"]
    stage = next((f"S{i + 1}" for i, end in enumerate(ends) if step < end), "S4")
    if arm.startswith("D"):
        stage = "D"
    condition = dict(
        stage=stage,
        substage=stage,
        active_count=5,
        mobility=True,
        battery=True,
        starts="B",
        radius=None,
        jitter=0,
        stay_only=False,
    )
    if stage == "S1":
        condition.update(
            active_count=2, mobility=False, battery=False, starts="fixture", stay_only=True
        )
    elif stage == "S2":
        second = step >= (ends[0] + ends[1]) // 2
        condition.update(
            substage="S2b" if second else "S2a",
            active_count=2,
            mobility=False,
            battery=False,
            starts="fixture",
            radius=3 if second else 1,
            jitter=2 if second else 1,
        )
    elif stage == "S3":
        part = min(3, (step - ends[1]) // ((ends[2] - ends[1]) // 4))
        condition.update(
            substage=f"S3{chr(97 + part)}",
            active_count=2 if part == 0 else 5,
            mobility=part >= 2,
            battery=part >= 3,
            starts="intermediate" if part == 0 else "B",
        )
    return condition


def task_signature(condition):
    # S3d and S4 are the same physical task: a label change does not clear replay.
    return tuple(
        (key, value) for key, value in sorted(condition.items()) if key not in {"stage", "substage"}
    )


class BS5CurriculumEnv(BS5LaunchEnv):
    def __init__(
        self,
        condition,
        *,
        allow_relay=True,
        horizon=150,
        residual_battery=30.0,
        fixture=((19, 16), (16, 11)),
    ):
        self.condition = deepcopy(condition)
        self.fixture = np.asarray(fixture, dtype=np.int64)
        self.active_ids = np.arange(5)
        self.anchors = None
        self.ready = False
        super().__init__(
            BS5LaunchConfig(
                uav_start_cells=LAUNCHES["B"],
                allow_relay=allow_relay,
                horizon=horizon,
                residual_initial_battery=residual_battery,
            )
        )

    def reset(self, seed=None, *, options=None, assignment_seed=0, diagnostic=False):
        self.ready = False
        out = super().reset(seed=seed, options=options)
        setup = np.random.default_rng(np.random.SeedSequence([int(assignment_seed), 0x43555252]))
        c = self.condition
        self.cfg.enable_ue_mobility = c["mobility"]
        self.active_ids = setup.permutation(5)[:2] if c["active_count"] == 2 else np.arange(5)
        self.anchors = self.uav_positions.copy()
        if c["active_count"] == 2:
            self.status[:] = INACTIVE
            self.status[self.active_ids] = 0
            # Inactive positions never enter collisions or routing; no fake charge cycle.
            self.uav_positions[:] = [[0, i] for i in range(5)]
            anchors = self.fixture if c["starts"] == "fixture" else np.array([[17, 12], [15, 9]])
            self.uav_positions[self.active_ids] = anchors
            self.anchors[self.active_ids] = anchors
            if c["jitter"]:
                if diagnostic:
                    offsets = np.array([[0, 1], [-1, -1]]) * c["jitter"]
                else:
                    axial = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
                    offsets = axial[setup.integers(0, len(axial), 2)] * c["jitter"]
                self.uav_positions[self.active_ids] = np.clip(anchors + offsets, 0, 19)
        self.ready = True
        self._refresh_request_state()
        out.update(obs=self.get_obs(), state=self.get_state(), action_mask=self.get_action_mask())
        out["episode_meta"] = self._episode_meta()
        return out

    def get_action_mask(self):
        mask = super().get_action_mask()
        if not self.ready:
            return mask
        for agent in self.active_ids:
            for action in range(10):
                move = action // 2
                candidate = self.uav_positions[agent] + MOVE_DELTAS[move]
                if self.condition["stay_only"] and move != 4:
                    mask[agent, action] = 0
                radius = self.condition["radius"]
                if radius is not None and np.max(np.abs(candidate - self.anchors[agent])) > radius:
                    mask[agent, action] = 0
        assert mask.any(axis=1).all(), "Every slot, including inactive agents, needs a legal noop"
        return mask

    def _resolve_collisions(self, proposed, old, status_before):
        if not np.any(status_before == INACTIVE):
            return super()._resolve_collisions(proposed, old, status_before)
        occupied = set()
        for agent in np.flatnonzero(status_before != INACTIVE):
            if tuple(proposed[agent]) in occupied:
                proposed[agent] = old[agent]
            occupied.add(tuple(proposed[agent]))

    def step(self, actions):
        frozen = not self.condition["battery"]
        fields = (
            "hover_drain",
            "move_extra_drain",
            "access_radio_drain",
            "relay_radio_drain",
            "returning_step_drain",
            "return_safety_margin",
            "charge_rate_per_slot",
        )
        saved = {name: getattr(self.cfg, name) for name in fields}
        if frozen:
            for name in fields:
                setattr(self.cfg, name, 0.0)
        try:
            result = super().step(actions)
        finally:
            for name, value in saved.items():
                setattr(self.cfg, name, value)
        inactive = np.setdiff1d(np.arange(5), self.active_ids)
        assert np.all(self.status[inactive] == INACTIVE), "Inactive agent reactivated"
        return result

    def _status_name(self, status):
        return "CurriculumInactive" if status == INACTIVE else super()._status_name(status)

    def _episode_meta(self):
        meta = super()._episode_meta()
        meta.update(
            curriculum_version=VERSION,
            condition=self.condition,
            inactive_status_code=INACTIVE,
            active_ids=self.active_ids.tolist(),
            battery_freeze="zero energy/return threshold, preserve actual reset battery",
        )
        return meta
