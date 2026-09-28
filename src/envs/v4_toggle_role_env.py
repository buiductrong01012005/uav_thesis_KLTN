"""Six-action V4 environment for cleaner movement/role credit assignment.

Policy action space (per Active UAV):
    0 North
    1 South
    2 East
    3 West
    4 Stay
    5 ToggleRole  (Access <-> Relay, no movement this slot)

The radio model, reward, battery lifecycle and state/observation are unchanged
from V4FinalFixedMapEnv.  The only control change is that movement actions keep
the current role, while role switching is represented by one explicit action.

This module deliberately preserves the original 10-action environment instead
of mutating it, so old checkpoints/results remain reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from src.envs.v0_env import ACCESS, RELAY, MOVE_DELTAS
from src.envs.v4_fixed_mobile_battery_env import (
    ACTIVE,
    V4FixedMobileBatteryEnv,
)
from src.envs.v4_final_fixed_map_env import (
    V4FinalFixedMapConfig,
    V4FinalFixedMapEnv,
)

MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_EAST = 2
MOVE_WEST = 3
STAY = 4
TOGGLE_ROLE = 5

ACTION_NAMES = {
    MOVE_NORTH: "North",
    MOVE_SOUTH: "South",
    MOVE_EAST: "East",
    MOVE_WEST: "West",
    STAY: "Stay",
    TOGGLE_ROLE: "ToggleRole",
}


class V4ToggleRoleEnv(V4FinalFixedMapEnv):
    """V4 environment with five movement actions + one explicit role toggle."""

    n_actions = 6

    def step(self, actions: Sequence[int]):
        if len(actions) != self.cfg.num_uavs:
            raise ValueError(f"Expected {self.cfg.num_uavs} actions")

        # Translate the six policy actions into the original environment's
        # movement+role primitive.  This reuses exactly the same network,
        # collision, battery and reward logic; only the action encoding changes.
        translated: list[int] = []
        for i, raw in enumerate(actions):
            a = int(raw)
            if not 0 <= a < self.n_actions:
                raise ValueError("policy action must be in [0, 5]")

            if self.status[i] != ACTIVE:
                # Action is ignored by the base environment for unavailable UAVs.
                translated.append(V4FixedMobileBatteryEnv.encode_action(STAY, ACCESS))
                continue

            current_role = int(self.roles[i])
            if a == TOGGLE_ROLE:
                next_role = RELAY if current_role == ACCESS else ACCESS
                translated.append(V4FixedMobileBatteryEnv.encode_action(STAY, next_role))
            else:
                # Movement and Stay preserve the current role.
                translated.append(V4FixedMobileBatteryEnv.encode_action(a, current_role))

        return super().step(translated)

    def get_action_mask(self) -> np.ndarray:
        masks = np.ones((self.cfg.num_uavs, self.n_actions), dtype=np.int8)
        for i in range(self.cfg.num_uavs):
            if self.status[i] != ACTIVE:
                masks[i, :] = 0
                masks[i, STAY] = 1
                continue

            # Boundary masking only affects physical movement actions.
            for move_idx in range(5):
                cand = self.uav_positions[i] + MOVE_DELTAS[move_idx]
                if not self._inside_grid(cand):
                    masks[i, move_idx] = 0
            # ToggleRole is always valid for an Active UAV.  We intentionally do
            # not add a role-specific reward or a hand-coded "Relay needed" mask.
            masks[i, TOGGLE_ROLE] = 1
        return masks

    @staticmethod
    def policy_action_name(action: int) -> str:
        if int(action) not in ACTION_NAMES:
            raise ValueError("policy action must be in [0, 5]")
        return ACTION_NAMES[int(action)]


@dataclass
class V4ToggleRoleA1Config(V4FinalFixedMapConfig):
    """Locked A1 stress setup with only the action representation changed."""

    enable_ue_mobility: bool = False
    enable_battery_cycle: bool = True
    backhaul_radius_cells: float = 4.0
    residual_initial_battery: float = 30.0
    randomize_residual_uav: bool = False  # scenario logic below owns this RNG
    relay_anchor_cell: tuple[int, int] = (11, 6)
    scenario_seed: int = 81


class V4ToggleRoleA1Env(V4ToggleRoleEnv):
    """A1 identity-randomized Relay-dropout environment for the 6-action audit.

    Every episode keeps the same static UE layout, but:
      * one UAV identity is sampled as the residual-battery UAV;
      * that UAV is placed at the Relay anchor and starts as the sole Relay;
      * the remaining identities are randomly permuted over the other four
        original starting cells;
      * only the residual UAV starts at battery=30, all others start full.

    This matches the clean A1 protocol used before the 6-action redesign.
    """

    def __init__(self, config: V4ToggleRoleA1Config | None = None):
        cfg = config or V4ToggleRoleA1Config()
        super().__init__(cfg)
        self.cfg: V4ToggleRoleA1Config
        self._scenario_rng = np.random.default_rng(self.cfg.scenario_seed)
        self._base_start_cells = np.asarray(self.cfg.uav_start_cells, dtype=np.int64)
        self._relay_anchor = np.asarray(self.cfg.relay_anchor_cell, dtype=np.int64)
        if not self._inside_grid(self._relay_anchor):
            raise ValueError("relay_anchor_cell must be inside the grid")

        # Remove exactly one occurrence of the anchor if it is one of the
        # original cells; otherwise reserve the first four original positions.
        cells = [tuple(map(int, x)) for x in self._base_start_cells]
        anchor_t = tuple(map(int, self._relay_anchor))
        if anchor_t in cells:
            cells.remove(anchor_t)
        self._other_start_cells = np.asarray(cells[: self.cfg.num_uavs - 1], dtype=np.int64)
        if len(self._other_start_cells) != self.cfg.num_uavs - 1:
            raise ValueError("Need four non-anchor starting cells")

    def _apply_scenario(self, residual_uav_idx: int, permutation: np.ndarray):
        residual = int(residual_uav_idx)
        if not 0 <= residual < self.cfg.num_uavs:
            raise ValueError("residual_uav_idx out of range")
        others = [i for i in range(self.cfg.num_uavs) if i != residual]
        if sorted(map(int, permutation.tolist())) != list(range(self.cfg.num_uavs - 1)):
            raise ValueError("permutation must contain 0..3 exactly once")

        self.residual_uav_idx = residual
        self.uav_positions[residual] = self._relay_anchor
        for j, uid in enumerate(others):
            self.uav_positions[uid] = self._other_start_cells[int(permutation[j])]

        self.roles[:] = ACCESS
        self.roles[residual] = RELAY
        self.status[:] = ACTIVE
        self.battery[:] = self.cfg.max_battery
        self.battery[residual] = self.cfg.residual_initial_battery
        self.target_station_idx[:] = -1
        self.return_start_time[:] = -1
        self.station_arrival_time[:] = -1
        self.rejoin_time[:] = -1

    def _refresh_reset_output(self, out: dict) -> dict:
        out["obs"] = self.get_obs()
        out["state"] = self.get_state()
        out["action_mask"] = self.get_action_mask()
        out["episode_meta"] = self._episode_meta()
        out["episode_meta"].update({
            "action_design": "6 actions: N/S/E/W/Stay preserve role; ToggleRole switches Access<->Relay",
            "residual_uav_idx": int(self.residual_uav_idx),
            "relay_anchor_cell": self._relay_anchor.astype(int).tolist(),
        })
        return out

    def reset(self, seed: int | None = None):
        # Keep UE geometry deterministic through the normal environment seed;
        # use a separate RNG for identity/position randomization across episodes.
        out = super().reset(seed=seed)
        residual = int(self._scenario_rng.integers(0, self.cfg.num_uavs))
        perm = self._scenario_rng.permutation(self.cfg.num_uavs - 1)
        self._apply_scenario(residual, perm)
        return self._refresh_reset_output(out)

    def reset_with_scenario(
        self,
        residual_uav_idx: int,
        permutation_seed: int,
        *,
        layout_seed: int = 1101,
        residual_battery: float | None = None,
    ) -> dict:
        """Deterministic reset used for matched evaluation/audits."""
        out = V4FinalFixedMapEnv.reset(self, seed=layout_seed)
        perm_rng = np.random.default_rng(int(permutation_seed))
        perm = perm_rng.permutation(self.cfg.num_uavs - 1)
        self._apply_scenario(int(residual_uav_idx), perm)
        if residual_battery is not None:
            self.battery[int(residual_uav_idx)] = float(residual_battery)
        return self._refresh_reset_output(out)
