"""Final BS5 environment: reset state, then execute one causal slot at a time.

The four mixins contain disjoint functionality, with no super() dispatch chain.
Read reset() and step() here first, then follow their explicitly named phases.
"""

from src.configuration import environment_defaults, environment_spec

from copy import deepcopy
from dataclasses import dataclass, field, asdict
from collections.abc import Sequence
import math
import numpy as np
from src.envs.config import SimulationConfig, ServiceV2Config, ARM_ACTIONS, MOVE_DELTAS
from src.envs.curriculum import FULL_CONDITION
from src.envs.reward import TCFWeights, FairConfig, rate_metrics, jain, update_ema
from src.envs.energy import EnergyMixin, EnergyConfig, linear_learning_rate as linear_learning_rate
from src.envs.movement import (
    MovementMixin,
    assert_endpoint_occupancy,
    COLLISION_CONTRACT,
    RETURN_CONTRACT,
)
from src.envs.radio.multihop import VERSION as ROUTING_VERSION
from src.envs.observation import (
    ObservationMixin,
    BeaconReports,
    DelayedHistoryMap,
    OBS_DIM as OBS_DIM,
    STATE_DIM as STATE_DIM,
)
from src.envs.radio.channel import RadioMixin

VERSION = "bs5_fourarm400k_v1"
REWARD_VERSION = "throughput_instant_fair_energy_blocked_fourarm_v1"


@dataclass(frozen=True)
@environment_defaults("collision")
class CollisionConfig:
    collision_weight: float

    def __post_init__(self):
        if not math.isfinite(self.collision_weight) or self.collision_weight < 0:
            raise ValueError("Collision weight must be finite and nonnegative")


@dataclass
class SlotState:
    """Values shared by the ordered movement, radio and energy phases of one slot."""

    actions: Sequence[int]
    raw: np.ndarray = field(init=False)
    e: EnergyConfig = field(init=False)
    enabled: bool = field(init=False)
    before_e: np.ndarray = field(init=False)
    original_status: np.ndarray = field(init=False)
    old: np.ndarray = field(init=False)
    old_roles: np.ndarray = field(init=False)
    active_max_j: float = field(init=False)
    floor: float = field(init=False)
    effective: np.ndarray = field(init=False)
    proposed: np.ndarray = field(init=False)
    roles: np.ndarray = field(init=False)
    net: dict = field(init=False)
    metrics: dict = field(init=False)
    original_qos: tuple = field(init=False)
    reward: float = field(init=False)


class BS5FourArm400kEnv(MovementMixin, RadioMixin, EnergyMixin, ObservationMixin):
    return_version = "energy_swap_committed_bfs_reserve_v1"

    def __init__(
        self,
        condition=None,
        *,
        arm="C-H",
        service_config=None,
        reward_weights=None,
        coverage_floor_mbps=None,
        horizon=None,
        residual_battery=None,
        fixture=None,
        energy_config=None,
        collision_config=None,
        fair_config=None,
        simulation_config=None,
    ):
        self.fair_config = fair_config or FairConfig(mode="instant")
        if self.fair_config.mode != "instant":
            raise ValueError("Four-arm400k uses frozen legacy instantaneous fairness")
        if arm not in {"C-N", "C-R", "C-H", "H-only"}:
            raise ValueError("Unknown four-arm experiment")
        self.reward_weights = reward_weights or TCFWeights(
            **environment_spec()["reward"]["weights"]
        )
        self.coverage_floor_mbps = float(
            environment_spec()["reward"]["coverage_floor_mbps"]
            if coverage_floor_mbps is None
            else coverage_floor_mbps
        )
        rate_metrics(
            [0.0], coverage_floor_mbps=self.coverage_floor_mbps, weights=self.reward_weights
        )
        expected = (self.fair_config.throughput_weight, 0.0, self.fair_config.fairness_weight)
        if not np.allclose(
            [
                self.reward_weights.throughput,
                self.reward_weights.coverage,
                self.reward_weights.fairness,
            ],
            expected,
            atol=1e-12,
        ):
            raise ValueError("Use matched service weights (0.4,0,0.6)")
        self.energy_config = energy_config or EnergyConfig()
        self.collision_config = collision_config or CollisionConfig()
        self.service_config = service_config or ServiceV2Config()
        self.multihop_arm = arm
        self.training_arm = "C-H" if arm == "H-only" else arm
        self.policy_actions = ARM_ACTIONS[arm]
        self._proxy_api = False
        self.probe_mode = "full"
        from src.envs.radio.full import HybridFullConfig

        self.hybrid_config = HybridFullConfig()
        self.hybrid_radio_overhead = 0.0
        self.condition = deepcopy(condition or FULL_CONDITION)
        self.fixture = np.asarray(
            environment_spec()["fixture"] if fixture is None else fixture, dtype=np.int64
        )
        self.active_ids = np.arange(5)
        self.anchors = None
        self.ready = False
        self.cfg = (
            deepcopy(simulation_config) if simulation_config is not None else SimulationConfig()
        )
        self.allow_relay = arm != "C-N"
        if horizon is not None:
            self.cfg.horizon = horizon
        if residual_battery is not None:
            self.cfg.residual_initial_battery = residual_battery
        self.configured_failed_bs_idx = int(self.cfg.failed_bs_idx)
        self.failure_source = "configured"
        self.last_reset_seed = int(self.cfg.seed)
        self._in_step = False
        self.last_service_snapshot = None
        self.rng = np.random.default_rng(self.cfg.seed)
        self.bs_xy = np.asarray(self.cfg.bs_xy_m, dtype=np.float64)
        self.failed_bs_xy = self.bs_xy[self.cfg.failed_bs_idx].copy()
        self.surviving_bs_xy = np.delete(self.bs_xy, self.cfg.failed_bs_idx, axis=0)
        self.bs_cells = np.asarray([self._xy_to_cell(xy) for xy in self.bs_xy], dtype=np.int64)
        self.failed_bs_cell = self.bs_cells[self.cfg.failed_bs_idx]
        self.surviving_bs_cells = np.delete(self.bs_cells, self.cfg.failed_bs_idx, axis=0)
        self.affected_mask = self._build_failed_bs_voronoi_mask()
        self.affected_cells = np.argwhere(self.affected_mask)
        self.start_positions = np.asarray(self.cfg.uav_start_cells, dtype=np.int64)
        self.charging_stations = np.asarray(self.cfg.charging_station_cells, dtype=np.int64)
        self._validate_cells(self.start_positions)
        self._validate_cells(self.charging_stations)
        self.t = 0
        self.uav_positions = self.start_positions.copy()
        self.roles = np.zeros(5, dtype=np.int64)
        self.status = np.zeros(5, dtype=np.int64)
        self.battery = np.full(5, self.cfg.max_battery, dtype=np.float32)
        for name in [
            "target_station_idx",
            "return_start_time",
            "station_arrival_time",
            "rejoin_time",
        ]:
            setattr(self, name, np.full(5, -1, dtype=np.int64))
        self.hotspot_centers = np.zeros((self.cfg.num_hotspots, 2), dtype=np.float64)
        self.hotspot_vel = np.zeros_like(self.hotspot_centers)
        self.ue_hotspot = np.arange(self.cfg.num_ues, dtype=np.int64) % self.cfg.num_hotspots
        self.ue_offsets = np.zeros((self.cfg.num_ues, 2), dtype=np.float64)
        self.ue_positions = np.zeros((self.cfg.num_ues, 2), dtype=np.float64)
        self.residual_uav_idx = 0
        self.ue_local_velocity_cells_per_slot = np.zeros((self.cfg.num_ues, 2), dtype=np.float64)
        self._route_cache = [None] * 5
        self._previous_return_cell = [None] * 5
        self._return_planning = False
        self._route_decisions = {}
        self._hybrid_ids = set()
        self.hybrid_extra_debit = np.zeros(5)
        self.last_probe_result = self.last_energy_probe = None
        self.last_collision_resolution = self.last_return_control = None
        self.reports = self.last_v2_network = self.last_v2_radio_detail = None
        self._temporal_ready = False
        self.ema_rates = np.zeros(90)
        self.history_reports = None
        self.energy_events, self.collision_events = [], []
        self.cfg.access_bw_hz = (
            self.service_config.access_channel_mhz
            * self.service_config.access_occupied_fraction
            * 1e6
        )
        self.cfg.backhaul_bw_hz = (
            self.service_config.backhaul_total_mhz
            * self.service_config.backhaul_occupied_fraction
            * 1e6
        )
        slots = self.energy_config.swap_seconds / self.cfg.slot_s
        if not math.isclose(slots, round(slots)):
            raise ValueError("Swap duration must be a whole number of slots")
        self.swap_slots = int(round(slots))

    @property
    def n_actions(self):
        return self.policy_actions

    def logical_roles(self):
        roles = self.roles.copy()
        for aid in self._hybrid_ids:
            if roles[aid] == 1:
                roles[aid] = 2
        if self.multihop_arm == "H-only":
            roles[self.status == 0] = 2
        return roles

    def encode_action(self, move_idx, role=None):
        if self.multihop_arm == "H-only":
            if int(move_idx) != move_idx or not 0 <= move_idx < 5 or role not in (None, 2):
                raise ValueError("H-only action is move 0..4")
            return int(move_idx)
        roles = self.policy_actions // 5
        if (
            int(move_idx) != move_idx
            or int(role) != role
            or not 0 <= move_idx < 5
            or not 0 <= role < roles
        ):
            raise ValueError("Invalid move/role")
        return int(move_idx * roles + role)

    def decode_action(self, action):
        if int(action) != action or not 0 <= action < self.n_actions:
            raise ValueError("Invalid policy action")
        if self.multihop_arm == "H-only":
            return int(action), 2
        roles = self.policy_actions // 5
        return int(action) // roles, int(action) % roles

    def get_action_mask(self):
        mask = np.ones((5, self.n_actions), dtype=np.int8)
        for aid in range(5):
            if self.status[aid] != 0:
                mask[aid] = 0
                mask[aid, self.encode_action(4, 2 if self.multihop_arm == "H-only" else 0)] = 1
                continue
            for action in range(self.n_actions):
                move, role = self.decode_action(action)
                candidate = self.uav_positions[aid] + MOVE_DELTAS[move]
                legal = self._inside_grid(candidate) and (self.allow_relay or role != 1)
                if self.ready:
                    radius = self.condition["radius"]
                    legal = legal and (not self.condition["stay_only"] or move == 4)
                    legal = legal and (
                        radius is None or np.max(np.abs(candidate - self.anchors[aid])) <= radius
                    )
                mask[aid, action] = int(legal)
        assert mask.any(axis=1).all(), "Every slot needs a legal noop"
        return mask

    def reset(self, seed=None, *, options=None, assignment_seed=0, diagnostic=False):
        """Reset topology/RNG, sample UEs and residual UAV, apply curriculum, seed reports."""
        self._temporal_ready = False
        self.ema_rates = np.zeros(90, float)
        self.history_reports = None
        self._hybrid_ids = set()
        self.hybrid_extra_debit = np.zeros(5)
        self.last_probe_result = None
        self.reports = self.last_v2_network = self.last_v2_radio_detail = None
        self._route_cache = [None] * 5
        self._previous_return_cell = [None] * 5
        self._return_planning = False
        self._route_decisions = {}
        self.ready = False
        options = {} if options is None else dict(options)
        if set(options) - {"failed_bs_idx"}:
            raise ValueError("unknown reset option")
        if options.get("failed_bs_idx", 5) != 5:
            raise ValueError("BS5 is locked for this experiment")
        if "failed_bs_idx" in options:
            self._check_id(options["failed_bs_idx"], len(self.bs_xy))
        if seed is not None:
            self.last_reset_seed = int(seed)
        self.failure_source = (
            "explicit_reset_override" if "failed_bs_idx" in options else "configured"
        )
        self._set_failure_topology(options.get("failed_bs_idx", self.configured_failed_bs_idx))
        self.last_service_snapshot = None
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.uav_positions = self.start_positions.copy()
        self.roles[:] = self.status[:] = 0
        for name in [
            "target_station_idx",
            "return_start_time",
            "station_arrival_time",
            "rejoin_time",
        ]:
            getattr(self, name).fill(-1)
        # Preserve all reset-time random draws, including the initial clustering pass.
        self._init_mobile_ues()
        self.residual_uav_idx = (
            int(self.rng.integers(0, 5)) if self.cfg.randomize_residual_uav else 0
        )
        self.battery[:] = self.cfg.max_battery
        if self.cfg.initialize_residual_battery:
            self.battery[self.residual_uav_idx] = self.cfg.residual_initial_battery
        setup = np.random.default_rng(np.random.SeedSequence([int(assignment_seed), 0x43555252]))
        c = self.condition
        self.active_ids = setup.permutation(5)[:2] if c["active_count"] == 2 else np.arange(5)
        self.anchors = self.uav_positions.copy()
        if c["active_count"] == 2:
            self.status[:] = 3
            self.status[self.active_ids] = 0
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
        self.last_collision_resolution = None
        assert_endpoint_occupancy(self.uav_positions, self.status, self.charging_stations)
        self.last_return_control = None
        self.reports = BeaconReports(self.service_config)
        self.reports.capture(self, None)
        self.reports.deliver(self.t)
        self.energy_j = (
            self.battery.astype(float) / self.cfg.max_battery * self.energy_config.capacity_j
        )
        self.swap_remaining = np.zeros(5, dtype=int)
        self.energy_events = []
        self.total_consumed_j = np.zeros(5)
        self.last_energy_audit = {}
        self.collision_events = []
        self.last_collision_audit = {}
        self.history_reports = DelayedHistoryMap(self.service_config)
        self.history_reports.capture(self)
        self.history_reports.deliver(self.t)
        self._temporal_ready = True
        return dict(
            obs=self.get_obs(),
            state=self.get_state(),
            action_mask=self.get_action_mask(),
            episode_meta=self._episode_meta(),
        )

    def step(self, actions):
        """Execute actions at t, serve UEs, debit energy, advance mobility and report t+1."""
        slot = SlotState(actions)
        self._prepare_slot(slot)  # Validate actions and reserve Return energy.
        self._move_slot(slot)  # Resolve endpoints, advance committed Return routes.
        self._in_step = True
        try:
            slot.net = self.compute_network()  # Service uses pre-mobility UE positions.
        finally:
            self._in_step = False
        self._account_energy(slot)  # Actual RF airtime, flight, swapping, next-slot reserve.
        out = self._finish_slot(slot)  # Advance UEs/time, deliver delayed beacon reports.
        out = self._apply_collision_cost(actions, out)
        return self._finish_reward(out)  # Update EMA/history, final reward and tensors.

    def _finish_slot(self, slot):
        if self.condition["mobility"]:
            self._advance_mobile_ues()
        self.t += 1
        self.reports.capture(self, slot.net)
        self.reports.deliver(self.t)
        info = dict(
            t=self.t,
            **slot.metrics,
            **self.last_energy_probe,
            utility=float(slot.net["utility"]),
            radio_runtime_ms=slot.net["radio_runtime_ms"],
            qos_ues_original=slot.original_qos[0],
            qos_fraction_original=slot.original_qos[1],
            status=self.status.tolist(),
            status_names=[self._status_name(s) for s in self.status],
            battery=self.battery.tolist(),
            roles=self.logical_roles().tolist(),
            uav_positions=self.uav_positions.tolist(),
            routes=slot.net["routes"],
            access_loads=slot.net["access_loads"],
            backhaul_caps_mbps=slot.net["backhaul_caps_mbps"],
            role_switches_this_slot=int(
                np.sum((slot.old_roles != slot.roles) & (slot.effective == 0))
            ),
            training_env_version="bs5_multihop_energy_swap_v1",
            reward_version="weighted_service_minus_energy_v1",
            original_status=slot.original_status.tolist(),
        )
        return dict(
            obs=self.get_obs(),
            state=self.get_state(),
            action_mask=self.get_action_mask(),
            reward=slot.reward,
            terminated=False,
            truncated=self.t >= self.cfg.horizon,
            info=info,
        )

    def _apply_collision_cost(self, actions, out):
        detail = self.last_collision_resolution
        effective = np.asarray(self.last_return_control["effective_status"])
        active_blocked = np.zeros(5, dtype=np.int8)
        return_blocked = np.zeros(5, dtype=np.int8)
        active_blocked[detail["active_blocked_ids"]] = 1
        return_blocked[detail["returning_blocked_ids"]] = 1
        requested = np.asarray(detail["requested_positions"], dtype=int)
        resolved = np.asarray(detail["resolved_positions"], dtype=int)
        if np.any(active_blocked & (effective != 0)):
            raise AssertionError("Autonomous motion must not incur actor penalty")
        cost = float(active_blocked.sum() / 5.0)
        penalty = self.collision_config.collision_weight * cost
        out["reward"] = float(out["reward"] - penalty)
        metrics = dict(
            active_blocked_uavs=float(active_blocked.sum()),
            returning_blocked_uavs=float(return_blocked.sum()),
            collision_slot_fraction=float(bool(active_blocked.any())),
            blocked_move_cost_norm=cost,
            reward_collision=-penalty,
            policy_move_requests=float(
                sum(
                    (
                        effective[i] == 0 and self.decode_action(int(actions[i]))[0] != 4
                        for i in range(5)
                    )
                )
            ),
            reward_total=out["reward"],
        )
        self.last_collision_audit = dict(
            slot=self.t,
            effective_status=effective.tolist(),
            requested_positions=requested.tolist(),
            resolved_positions=resolved.tolist(),
            active_blocked=active_blocked.tolist(),
            returning_blocked=return_blocked.tolist(),
        )
        if active_blocked.any() or return_blocked.any():
            self.collision_events.append(
                dict(
                    **self.last_collision_audit,
                    blocked_by=detail["blocked_by"],
                    reward_collision=-penalty,
                )
            )
        self.last_energy_probe.update(metrics)
        out["info"].update(
            metrics,
            reward_version="weighted_service_minus_energy_and_blocked_move_v2",
            training_env_version="bs5_energy_collision_v2",
        )
        return out

    def _finish_reward(self, out):
        self.ema_rates = update_ema(
            self.ema_rates, self.last_v2_network["ue_rate_mbps"], self.fair_config.rho
        )
        self.history_reports.capture(self)
        self.history_reports.deliver(self.t)
        info = out["info"]
        ema_fair = jain(self.ema_rates)
        instant_fair = float(info["jain_service_qualified"])
        used_fair = ema_fair if self.fair_config.mode == "ema" else instant_fair
        rt = self.fair_config.throughput_weight * info["throughput_norm"]
        rf = self.fair_config.fairness_weight * used_fair
        service = rt + rf
        reward = service + info["reward_energy"] + info["reward_collision"]
        metrics = dict(
            service_score=service,
            reward_throughput=rt,
            reward_coverage=0.0,
            reward_fairness=rf,
            fairness_used=used_fair,
            jain_ema=ema_fair,
            jain_instant_qualified=instant_fair,
            ema_mean_ue_rate_mbps=float(self.ema_rates.mean()),
            ema_p05_ue_rate_mbps=float(np.quantile(self.ema_rates, 0.05)),
            history_delivery_source=float(self.history_reports.latest_source),
            reward_total=reward,
        )
        self.last_energy_probe.update(metrics)
        info.update(
            metrics,
            reward=reward,
            training_env_version="bs5_fourarm400k_v1",
            reward_version="throughput_instant_fair_energy_blocked_fourarm_v1",
        )
        self.last_temporal_audit = dict(
            ema_rates=self.ema_rates.copy(), history_map=self.history_reports.vector(self.t).copy()
        )
        out.update(obs=self.get_obs(), state=self.get_state(), reward=float(reward))
        assert out["obs"].shape == (5, 173) and out["state"].shape == (156,)
        return out

    def _episode_meta(self):
        """Describe the final model and the effective episode, without legacy switches."""
        fleet_size = len(self.uav_positions)
        energy = self.energy_config
        return {
            "metadata_version": "uav_qmix_final_episode_v2",
            "training_env_version": VERSION,
            "reward_version": REWARD_VERSION,
            "arm": self.multihop_arm,
            "n_actions": int(self.n_actions),
            "obs_dim": OBS_DIM,
            "state_dim": STATE_DIM,
            "num_uavs": fleet_size,
            "num_ues": int(self.cfg.num_ues),
            "num_hotspots": int(self.cfg.num_hotspots),
            "horizon": int(self.cfg.horizon),
            "slot_s": float(self.cfg.slot_s),
            "action_design": (
                "move0..4; fixed Hybrid while Active"
                if self.multihop_arm == "H-only"
                else "move*3+role0/1/2 (Access/Relay/Hybrid)"
                if self.training_arm == "C-H"
                else "move*2+role0/1 (Access/Relay); Relay masked in C-N"
            ),
            "allow_relay": self.allow_relay,
            "routing_version": ROUTING_VERSION,
            "routing_contract": {
                "topology": "Acyclic BS-rooted minimum isolated-airtime forest",
                "forwarding_roles": ["Relay", "Hybrid"] if self.allow_relay else [],
                "relay_depth_cap": None,
                "allocation": "Fixed-route global flow/schedule LP; half-duplex backhaul Tx/Rx and endpoint resource constraints",
                "traffic_model": "Steady-state fluid flow, not packet latency",
            },
            "failed_bs_idx": int(self.cfg.failed_bs_idx),
            "configured_failed_bs_idx": self.configured_failed_bs_idx,
            "failure_source": self.failure_source,
            "failure_changes_only_at_reset": True,
            "last_explicit_reset_seed": self.last_reset_seed,
            "failed_bs_cell": self.failed_bs_cell.astype(int).tolist(),
            "failed_bs_xy_m": self.failed_bs_xy.tolist(),
            "surviving_bs_ids": [i for i in range(len(self.bs_xy)) if i != self.cfg.failed_bs_idx],
            "bs_xy_m": self.bs_xy.tolist(),
            "affected_cell_count": int(self.affected_mask.sum()),
            "charging_stations": self.charging_stations.astype(int).tolist(),
            "ue_demand_mbps": float(self.cfg.ue_demand_mbps),
            "access_radius_cells": float(self.cfg.access_radius_cells),
            "backhaul_radius_cells": float(self.cfg.backhaul_radius_cells),
            "coarse_obs_shape": list(self.reports.counts.shape),
            "mobility_model": {
                "type": "correlated_moving_groups_with_individual_local_motion",
                "enabled": bool(self.condition["mobility"]),
                "num_groups": int(self.cfg.num_hotspots),
                "group_center_speed_mps": float(self.cfg.group_center_speed_mps),
                "group_turn_noise_std_rad": float(self.cfg.group_turn_noise_std_rad),
                "ue_local_speed_mps": [
                    float(self.cfg.ue_local_speed_min_mps),
                    float(self.cfg.ue_local_speed_max_mps),
                ],
                "ue_local_turn_noise_std_rad": float(self.cfg.ue_local_turn_noise_std_rad),
                "group_spread_cells": float(self.cfg.group_spread_cells),
                "max_ue_offset_cells": float(self.cfg.max_ue_offset_cells),
                "confined_to_failed_bs_region": True,
            },
            "condition": deepcopy(self.condition),
            "active_ids": self.active_ids.tolist(),
            "status_codes": {"Active": 0, "Returning": 1, "Swapping": 2, "CurriculumInactive": 3},
            "role_observation": {"Access": 0.0, "Relay": 1.0, "Hybrid": 0.5},
            "service_config": asdict(self.service_config),
            "actor_contract": {
                "base": "130 delayed common report + 7 exact self + 6 failure one-hot + 5 agent ID",
                "history": "25 delayed mean-EMA-rate cells from detectable UE beacons",
                "sensing": "Active Access/Hybrid beacon footprint; no future/global UE census",
                "report_period_slots": self.service_config.report_period_slots,
                "report_delay_slots": self.service_config.report_delay_slots,
                "report_max_age_slots": self.service_config.report_max_age_slots,
            },
            "state_contract": "66 centralized features + 90 normalized per-UE EMA rates; training only",
            "radio_contract": "Urban mean path loss/Shannon; colored Access reuse; separate Access/backhaul pools; full Hybrid endpoint budgets",
            "reward": {
                "formula": "w_T*T + w_F*J - lambda_E*E_norm - lambda_C*blocked_norm",
                "weights": asdict(self.reward_weights),
                "throughput_normalizer_mbps": float(self.cfg.num_ues * self.cfg.ue_demand_mbps),
                "fairness": "Instantaneous Jain across all UEs; rates below coverage floor zeroed",
                "coverage_floor_mbps": self.coverage_floor_mbps,
                "coverage_term": "Zero weight in the final environment; coverage remains a diagnostic",
                "energy_weight": energy.energy_weight,
                "energy_normalizer_j": fleet_size * energy.reference_w * self.cfg.slot_s,
                "collision_weight": self.collision_config.collision_weight,
                "blocked_normalizer": fleet_size,
                "blocked_scope": "Rejected Active moves only; autonomous Returning excluded",
            },
            "fair_config": asdict(self.fair_config),
            "history_contract": "EMA starts at zero; every UE rate including zero updates EMA for observations/state and diagnostics; reward uses instantaneous qualified Jain",
            "battery_model": {
                "type": "physical_joules_with_grounded_pack_replacement",
                "enabled": bool(self.condition["battery"]),
                "capacity_j": energy.capacity_j,
                "soc_representation": "battery/max_battery = energy_j/capacity_j",
                "max_battery": self.cfg.max_battery,
                "initialize_residual_battery": bool(self.cfg.initialize_residual_battery),
                "residual_uav_idx": int(self.residual_uav_idx),
                "residual_initial_battery": float(self.cfg.residual_initial_battery),
                "swap_slots": self.swap_slots,
                "arrival_floor_j": energy.arrival_soc * energy.capacity_j,
                "curriculum_freeze": "No consumption, reserve transitions or swapping when battery is disabled; preserve reset SOC",
            },
            "energy_config": asdict(energy),
            "energy_contract": "Optional flight baseline from resolved cruise/hover displacement + Active circuit + actual Access/backhaul Tx airtime / PA efficiency; Returning payload radio off",
            "return_contract": deepcopy(RETURN_CONTRACT),
            "collision_contract": deepcopy(COLLISION_CONTRACT),
            "collision_config": asdict(self.collision_config),
            "excluded_energy": "Vertical maneuvers, turns, battery aging, ground charging and spare inventory not modeled; empirical flight baseline includes original avionics",
        }
