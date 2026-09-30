"""Cartesian movement × role environments for the thesis curriculum experiment.

The comparison is intentionally designed so that Cartesian and target-aware RL
receive the same rich spatial/request observation and use the same Spatial-QMIX
network architecture.  The main control difference is therefore the action/
relay-association formulation:

Cartesian:
    action = move_index * 2 + role
    move_index in {N,S,E,W,Stay}, role in {Access,Relay}
    Relay routing is opportunistic in the deterministic network layer.

Target-aware (V7/V8/V9 source):
    0..4 = Access movement
    5+j  = RelayTo(UAV j)
    Relay routing is target-exclusive.

The request telemetry is exposed to both methods; Cartesian does not use an
explicit relay target, so the target-onehot block in its observation is zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.envs.v0_env import ACCESS, RELAY, MOVE_DELTAS
from src.envs.v4_fixed_mobile_battery_env import ACTIVE
from src.envs.v4_final_fixed_map_env import V4FinalFixedMapEnv
from src.envs.v7_r6_macro_relay_stable_env import (
    STAY,
    V7R6MacroRelayStableConfig,
    V7R6MacroRelayStableEnv,
)


@dataclass
class CartesianR6Config(V7R6MacroRelayStableConfig):
    horizon: int = 150
    backhaul_radius_cells: float = 6.0
    enable_ue_mobility: bool = True
    enable_battery_cycle: bool = False

    # Match target-aware battery parameters exactly when battery is enabled.
    max_battery: float = 100.0
    hover_drain: float = 0.1
    move_extra_drain: float = 0.1
    relay_radio_drain: float = 0.1
    access_radio_drain: float = 0.2
    returning_step_drain: float = 0.2
    return_safety_margin: float = 5.0
    charge_rate_per_slot: float = 5.0
    rejoin_threshold: float = 80.0
    residual_initial_battery: float = 30.0
    randomize_residual_uav: bool = True


@dataclass
class CartesianR6BatteryConfig(CartesianR6Config):
    enable_battery_cycle: bool = True


@dataclass
class CartesianR6BatteryFocusConfig(CartesianR6BatteryConfig):
    randomize_residual_battery: bool = False
    residual_battery_min: float = 24.0
    residual_battery_max: float = 36.0


class CartesianR6Env(V7R6MacroRelayStableEnv):
    """Rich-observation R6 environment with Cartesian movement×role actions."""

    n_actions = 10

    def __init__(self, config: CartesianR6Config | None = None):
        super().__init__(config or CartesianR6Config())
        self.cfg: CartesianR6Config

    def reset(self, seed: int | None = None):
        out = super().reset(seed=seed)
        # Cartesian control has no designated relay target.
        self.relay_target[:] = -1
        self.last_assisted_move[:] = STAY
        self._refresh_request_state()
        out = self._refresh_output(out)
        out["episode_meta"].update({
            "profile": "thesis_cartesian_r6",
            "action_design": "5 movement directions x {Access, Relay}",
            "routing": "opportunistic deterministic one-hop Relay routing",
        })
        return out

    def compute_network(self) -> dict:
        # Explicitly use the original Cartesian/opportunistic V4 routing rather
        # than V7 target-exclusive routing. This preserves the old formulation.
        return V4FinalFixedMapEnv.compute_network(self)

    def get_action_mask(self) -> np.ndarray:
        masks = np.ones((self.cfg.num_uavs, self.n_actions), dtype=np.int8)
        noop = self.encode_action(STAY, ACCESS)
        for i in range(self.cfg.num_uavs):
            if self.status[i] != ACTIVE:
                masks[i, :] = 0
                masks[i, noop] = 1
                continue
            for a in range(self.n_actions):
                move_idx, _ = self.decode_action(a)
                cand = self.uav_positions[i] + MOVE_DELTAS[move_idx]
                if not self._inside_grid(cand):
                    masks[i, a] = 0
        return masks

    def step(self, actions: Sequence[int]):
        if len(actions) != self.cfg.num_uavs:
            raise ValueError(f"Expected {self.cfg.num_uavs} actions")
        mask = self.get_action_mask()
        raw = [int(a) for a in actions]
        for i, a in enumerate(raw):
            if not 0 <= a < self.n_actions:
                raise ValueError("Cartesian action must be in [0,9]")
            if self.status[i] == ACTIVE and not mask[i, a]:
                raise ValueError(f"invalid Cartesian action {a} for UAV {i}")

        self.relay_target[:] = -1
        self.last_assisted_move[:] = STAY
        # Bypass V7 macro step and execute the Cartesian actions directly.
        out = V4FinalFixedMapEnv.step(self, raw)
        self.relay_target[:] = -1
        self.last_assisted_move[:] = STAY

        # Request telemetry is refreshed for the next decision and is available
        # to BOTH action formulations in the controlled comparison.
        self._refresh_request_state()
        out = self._refresh_output(out)
        out["info"].update({
            "relay_needed": self.relay_needed.astype(int).tolist(),
            "request_open": self.request_open.astype(int).tolist(),
            "request_load_mbps": self.request_load_mbps.astype(float).tolist(),
            "request_direct_bh_mbps": self.request_direct_bh_mbps.astype(float).tolist(),
            "request_deficit": self.request_deficit.astype(float).tolist(),
            "relay_target": [-1] * self.cfg.num_uavs,
            "relay_claim_conflicts_this_slot": 0,
        })
        return out

    def _episode_meta(self) -> dict:
        meta = V7R6MacroRelayStableEnv._episode_meta(self)
        meta.update({
            "control_version": "thesis_cartesian_r6",
            "action_design": "movement x role; 10 actions",
            "relay_assignment": "no explicit target; network layer opportunistically chooses one-hop routes",
        })
        return meta


class CartesianR6BatteryEnv(CartesianR6Env):
    """Stage 2: same Cartesian semantics with automatic battery lifecycle."""

    def __init__(self, config: CartesianR6BatteryConfig | None = None):
        super().__init__(config or CartesianR6BatteryConfig())
        self.cfg: CartesianR6BatteryConfig
        if not self.cfg.enable_battery_cycle:
            raise ValueError("Cartesian Stage 2 requires battery cycle ON")


class CartesianR6BatteryFocusEnv(CartesianR6BatteryEnv):
    """Stage 3: Stage-2 semantics + training-only residual-battery randomization."""

    def __init__(self, config: CartesianR6BatteryFocusConfig | None = None):
        super().__init__(config or CartesianR6BatteryFocusConfig())
        self.cfg: CartesianR6BatteryFocusConfig
        if self.cfg.residual_battery_min <= 0:
            raise ValueError("residual_battery_min must be positive")
        if self.cfg.residual_battery_max < self.cfg.residual_battery_min:
            raise ValueError("invalid residual battery range")
        self.sampled_residual_initial_battery = float(self.cfg.residual_initial_battery)

    def reset(self, seed: int | None = None):
        out = super().reset(seed=seed)
        if self.cfg.randomize_residual_battery:
            sampled = float(self.rng.uniform(
                self.cfg.residual_battery_min,
                self.cfg.residual_battery_max,
            ))
            self.battery[int(self.residual_uav_idx)] = sampled
            self.sampled_residual_initial_battery = sampled
            out["obs"] = self.get_obs()
            out["state"] = self.get_state()
            out["action_mask"] = self.get_action_mask()
        else:
            self.sampled_residual_initial_battery = float(self.cfg.residual_initial_battery)
        return out

    def return_energy_margin(self, agent_id: int) -> float:
        i = int(agent_id)
        if self.status[i] != ACTIVE:
            return float("nan")
        _, dist_steps = self._nearest_station(self.uav_positions[i])
        threshold = (
            float(dist_steps) * float(self.cfg.returning_step_drain)
            + float(self.cfg.return_safety_margin)
        )
        return float(self.battery[i] - threshold)

    def min_active_return_margin(self) -> float:
        vals = [
            self.return_energy_margin(i)
            for i in range(self.cfg.num_uavs)
            if self.status[i] == ACTIVE
        ]
        return float(np.min(vals)) if vals else float("nan")
