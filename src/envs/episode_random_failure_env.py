"""Versioned episode-level single-BS failure; legacy environments are unchanged.

The six BS coordinates stay fixed. Only one failed ID changes at reset, not
mid-episode. The added one-hot is sufficient to identify this topology change;
it does not make the pre-existing coarse UE observation fully Markov.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from src.envs.final_movement_only_env import FinalMovementOnlyBatteryFocusEnv
from src.envs.thesis_cartesian_r6_curriculum_env import (
    CartesianR6BatteryConfig,
    CartesianR6BatteryEnv,
)
from src.envs.v9_r6_macro_relay_battery_focus_env import (
    V9R6MacroRelayBatteryFocusConfig,
    V9R6MacroRelayBatteryFocusEnv,
)


@dataclass
class EpisodeFailureOptions:
    failure_mode: str = "fixed"
    failure_candidates: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    expose_failure_context: bool = True
    record_service_snapshot: bool = False


@dataclass
class EpisodeFailureCartesianConfig(CartesianR6BatteryConfig, EpisodeFailureOptions):
    pass


@dataclass
class EpisodeFailureV9Config(V9R6MacroRelayBatteryFocusConfig, EpisodeFailureOptions):
    pass


class _EpisodeFailureMixin:
    def __init__(self, config):
        # Runtime failed_bs_idx is episode-specific; never mutate the caller's
        # shared dataclass or silently change a second environment's config.
        config = deepcopy(config)
        n = len(config.bs_xy_m)
        if config.failure_mode not in ("fixed", "random"):
            raise ValueError("failure_mode must be 'fixed' or 'random'")
        self._check_id(config.failed_bs_idx, n)
        ids = tuple(config.failure_candidates)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("failure_candidates must be nonempty and unique")
        for idx in ids:
            self._check_id(idx, n)
        config.failure_candidates = ids
        self.configured_failed_bs_idx = int(config.failed_bs_idx)
        self.failure_source = "configured"
        self.failure_rng = self._new_failure_rng(config.seed)
        self.last_reset_seed = int(config.seed)
        self._in_step = False
        self.last_service_snapshot = None
        super().__init__(config)

    @staticmethod
    def _check_id(idx, n):
        if isinstance(idx, bool) or not isinstance(idx, Integral) or not 0 <= idx < n:
            raise ValueError(f"failed BS ID must be an integer in [0,{n - 1}]")

    @staticmethod
    def _new_failure_rng(seed):
        # A separate stream: drawing a failure never consumes UE/battery RNG.
        return np.random.default_rng(np.random.SeedSequence([int(seed), 0x425346]))

    def _set_failure_topology(self, idx):
        self._check_id(idx, len(self.bs_xy))
        self.cfg.failed_bs_idx = int(idx)
        self.failed_bs_xy = self.bs_xy[idx].copy()
        self.failed_bs_cell = self.bs_cells[idx].copy()
        self.surviving_bs_xy = np.delete(self.bs_xy, idx, axis=0)
        self.surviving_bs_cells = np.delete(self.bs_cells, idx, axis=0)
        self.affected_mask = self._build_failed_bs_voronoi_mask()
        self.affected_cells = np.argwhere(self.affected_mask)
        if len(self.affected_cells) < self.cfg.num_hotspots:
            raise ValueError("selected failure region has too few hotspot cells")

    def reset(self, seed=None, *, options=None):
        options = {} if options is None else dict(options)
        if set(options) - {"failed_bs_idx"}:
            raise ValueError("unknown reset option")
        forced = options.get("failed_bs_idx")
        if "failed_bs_idx" in options:
            self._check_id(forced, len(self.bs_xy))
        if seed is not None:
            self.failure_rng = self._new_failure_rng(seed)
            self.last_reset_seed = int(seed)
        if "failed_bs_idx" in options:
            idx, self.failure_source = forced, "explicit_reset_override"
        elif self.cfg.failure_mode == "random":
            idx = int(self.failure_rng.choice(self.cfg.failure_candidates))
            self.failure_source = "random"
        else:
            idx, self.failure_source = self.configured_failed_bs_idx, "configured"
        # Rebuild BEFORE UE initialization, mobility validity tests, request
        # telemetry, and returned obs/state. A config-only mutation is unsafe.
        self._set_failure_topology(idx)
        self.last_service_snapshot = None
        out = super().reset(seed=seed)
        out["episode_meta"] = self._episode_meta()
        return out

    def get_state(self):
        base = super().get_state()
        if not self.cfg.expose_failure_context:
            return base
        failed = np.zeros(len(self.bs_xy), dtype=np.float32)
        failed[self.cfg.failed_bs_idx] = 1.0
        return np.concatenate((base, failed))

    def _episode_meta(self):
        meta = super()._episode_meta()
        meta.update(
            failure_version="episode_single_bs_v1",
            failure_mode=self.cfg.failure_mode,
            failure_source=self.failure_source,
            failure_candidates=list(self.cfg.failure_candidates),
            configured_failed_bs_idx=self.configured_failed_bs_idx,
            failed_bs_xy_m=self.failed_bs_xy.tolist(),
            surviving_bs_ids=[i for i in range(len(self.bs_xy)) if i != self.cfg.failed_bs_idx],
            bs_xy_m=self.bs_xy.tolist(),
            affected_cell_count=int(self.affected_mask.sum()),
            failure_context="onehot_after_legacy_state_before_agent_id"
            if self.cfg.expose_failure_context
            else "omitted_diagnostic_ablation",
            last_explicit_reset_seed=self.last_reset_seed,
            failure_changes_only_at_reset=True,
            ue_mobility=bool(self.cfg.enable_ue_mobility),
        )
        return meta

    def compute_network(self):
        net = super().compute_network()
        if self.cfg.record_service_snapshot and self._in_step:
            self.last_service_snapshot = dict(
                service_t=int(self.t),
                failed_bs_idx=int(self.cfg.failed_bs_idx),
                uav_positions=self.uav_positions.copy(),
                roles=self.roles.copy(),
                status=self.status.copy(),
                battery=self.battery.copy(),
                ue_positions=self.ue_positions.copy(),
                network=deepcopy(net),
            )
        return net

    def step(self, actions):
        self._in_step = True
        try:
            out = super().step(actions)
        finally:
            self._in_step = False
        out["info"]["failed_bs_idx"] = int(self.cfg.failed_bs_idx)
        return out


class EpisodeFailureCartesianEnv(_EpisodeFailureMixin, CartesianR6BatteryEnv):
    def __init__(self, config=None):
        super().__init__(config or EpisodeFailureCartesianConfig())


class EpisodeFailureV9Env(_EpisodeFailureMixin, V9R6MacroRelayBatteryFocusEnv):
    def __init__(self, config=None):
        super().__init__(config or EpisodeFailureV9Config())


class EpisodeFailureMovementEnv(_EpisodeFailureMixin, FinalMovementOnlyBatteryFocusEnv):
    def __init__(self, config=None):
        super().__init__(config or EpisodeFailureV9Config())
