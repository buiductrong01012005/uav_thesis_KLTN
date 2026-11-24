"""Physical battery reserve, radio/flight debits, pack replacement and conservation."""

from src.configuration import environment_defaults

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import numpy as np
from src.envs.config import MOVE_DELTAS
from src.envs.reward import rate_metrics


@dataclass(frozen=True)
@environment_defaults("energy")
class EnergyConfig:
    capacity_wh: float
    hover_w: float
    cruise_w: float
    active_circuit_w: float
    pa_efficiency: float
    arrival_soc: float
    swap_seconds: float
    energy_weight: float
    include_flight_energy: bool
    # Fixed reference, independent of arm, active count, seed and ablation.
    reference_w: float

    def __post_init__(self):
        if any(
            not math.isfinite(v) or v < 0
            for k, v in asdict(self).items()
            if k != "include_flight_energy"
        ):
            raise ValueError("Nonnegative finite energy parameters required")
        if not (
            self.capacity_wh > 0
            and self.reference_w > 0
            and 0 < self.pa_efficiency <= 1
            and 0 <= self.arrival_soc < 1
            and self.swap_seconds > 0
        ):
            raise ValueError("Invalid energy configuration")

    @property
    def capacity_j(self):
        return self.capacity_wh * 3600


def linear_learning_rate(step, total_steps, start=3e-4, end=3e-5):
    if total_steps <= 0 or not 0 < end <= start:
        raise ValueError("Invalid LR schedule")
    progress = np.clip(float(step) / total_steps, 0.0, 1.0)
    return float(start + (end - start) * progress)


class EnergyMixin:
    def _prepare_slot(self, slot):
        slot.raw = np.asarray(slot.actions)
        mask = self.get_action_mask()
        if slot.raw.shape != (5,) or not np.equal(slot.raw, slot.raw.astype(int)).all():
            raise ValueError("Five integer actions required")
        slot.raw = slot.raw.astype(int)
        if any((a < 0 or a >= self.n_actions or (not mask[i, a]) for i, a in enumerate(slot.raw))):
            raise ValueError("Invalid or masked action")
        slot.e = self.energy_config
        if not np.allclose(
            self.battery / self.cfg.max_battery * slot.e.capacity_j, self.energy_j, atol=1e-07
        ):
            raise ValueError("Use set_battery_soc(), not direct battery mutation")
        slot.enabled = bool(self.condition["battery"])
        slot.before_e = self.energy_j.copy()
        slot.original_status = self.status.copy()
        slot.old = self.uav_positions.copy()
        slot.old_roles = self.logical_roles().copy()
        self._slot_actions = slot.raw.tolist()
        slot.active_max_j = (
            (max(slot.e.hover_w, slot.e.cruise_w) if slot.e.include_flight_energy else 0)
            + slot.e.active_circuit_w
            + (
                10 ** ((self.cfg.access_tx_dbm - 30) / 10)
                + 10 ** ((self.cfg.backhaul_tx_dbm - 30) / 10)
            )
            / slot.e.pa_efficiency
        ) * self.cfg.slot_s
        slot.floor = slot.e.arrival_soc * slot.e.capacity_j
        if slot.enabled:
            for aid in np.flatnonzero(self.status == 0):
                move, _ = self.decode_action(slot.raw[aid])
                candidate = slot.old[aid] + MOVE_DELTAS[move]
                _, budget = self._route_budget(candidate, int(aid))
                if self.energy_j[aid] < slot.floor + budget + slot.active_max_j:
                    self._start_return(int(aid), "preflight_reserve")

    def _account_energy(self, slot):
        tx, rx = (np.zeros(5), np.zeros(5))
        for schedule in self.last_v2_radio_detail["schedules"]:
            for idx in schedule["active_edges"]:
                edge = self.last_v2_radio_detail["edges"][idx]
                for endpoint, fractions in [("source", tx), ("target", rx)]:
                    name = edge[endpoint]
                    if name.startswith("u"):
                        fractions[int(name[1:])] += schedule["duration"]
        if np.any(tx + rx > 1 + 1e-06):
            raise AssertionError("Half-duplex schedule violated")
        distance = np.linalg.norm(slot.proposed - slot.old, axis=1) * self.cfg.cell_size_m
        airborne = np.isin(slot.effective, [0, 1])
        service_active = self.last_service_snapshot["status"] == 0
        access_duty = np.array(
            [
                float(
                    service_active[i]
                    and slot.roles[i] in (0, 2)
                    and (len(slot.net["association"].get(i, [])) > 0)
                )
                for i in range(5)
            ]
        )
        flight = np.where(airborne, np.where(distance > 0, slot.e.cruise_w, slot.e.hover_w), 0.0)
        if not slot.e.include_flight_energy:
            flight[:] = 0
        components = dict(
            flight_j=flight * self.cfg.slot_s,
            circuit_j=service_active.astype(float) * slot.e.active_circuit_w * self.cfg.slot_s,
            access_pa_j=access_duty
            * 10 ** ((self.cfg.access_tx_dbm - 30) / 10)
            / slot.e.pa_efficiency
            * self.cfg.slot_s,
            backhaul_pa_j=tx
            * 10 ** ((self.cfg.backhaul_tx_dbm - 30) / 10)
            / slot.e.pa_efficiency
            * self.cfg.slot_s,
        )
        if not slot.enabled:
            components = {k: np.zeros(5) for k in components}
        requested = sum(components.values())
        scale = np.minimum(
            1.0, np.divide(slot.before_e, requested, out=np.ones(5), where=requested > 0)
        )
        components = {k: v * scale for k, v in components.items()}
        used = sum(components.values())
        self.energy_j -= used
        self.total_consumed_j += used
        inserted, removed = (np.zeros(5), np.zeros(5))
        swapped = []
        if slot.enabled:
            for aid in np.flatnonzero(slot.effective == 2):
                self.swap_remaining[aid] -= 1
                if self.swap_remaining[aid] == 0:
                    removed[aid] = self.energy_j[aid]
                    inserted[aid] = slot.e.capacity_j
                    self.energy_j[aid] = slot.e.capacity_j
                    self.status[aid], self.roles[aid] = (0, 0)
                    self.target_station_idx[aid] = -1
                    self.rejoin_time[aid] = self.t + 1
                    swapped.append(int(aid))
                    self.energy_events.append(
                        dict(
                            event="swap_complete",
                            slot=self.t + 1,
                            agent=int(aid),
                            removed_j=float(removed[aid]),
                            inserted_j=slot.e.capacity_j,
                        )
                    )
        self._sync_soc()
        if slot.enabled:
            for aid in np.flatnonzero((self.status == 0) & (slot.effective == 0)):
                _, budget = self._route_budget(self.uav_positions[aid], int(aid))
                if self.energy_j[aid] < slot.floor + budget + slot.active_max_j:
                    self._start_return(int(aid), "next_slot_reserve")
        stranded = np.flatnonzero((self.status == 1) & (self.energy_j <= 1e-07)).tolist()
        for aid in np.flatnonzero(self.status != 1):
            self._route_cache[aid] = None
            self._previous_return_cell[aid] = None
        residual = slot.before_e + inserted - removed - used - self.energy_j
        if np.max(np.abs(residual)) > 1e-07:
            raise AssertionError("Battery conservation failed")
        cost_norm = float(used.sum() / (5 * slot.e.reference_w * self.cfg.slot_s))
        slot.metrics = rate_metrics(
            slot.net["ue_rate_mbps"],
            self.cfg.ue_demand_mbps,
            self.coverage_floor_mbps,
            self.reward_weights,
        )
        slot.original_qos = (slot.metrics["qos_ues"], slot.metrics["qos_fraction"])
        slot.metrics["qos_ues"] = int(
            np.sum(slot.net["ue_rate_mbps"] >= self.cfg.ue_demand_mbps - 1e-06)
        )
        slot.metrics["qos_fraction"] = slot.metrics["qos_ues"] / self.cfg.num_ues
        penalty = slot.e.energy_weight * cost_norm
        slot.reward = slot.metrics["tcf_score"] - penalty
        self.last_energy_audit = dict(
            slot=self.t + 1,
            energy_before_j=slot.before_e.tolist(),
            energy_after_j=self.energy_j.tolist(),
            effective_status=slot.effective.tolist(),
            post_status=self.status.tolist(),
            roles=slot.roles.tolist(),
            distance_m=distance.tolist(),
            bh_tx_fraction=tx.tolist(),
            bh_rx_fraction=rx.tolist(),
            access_tx_fraction=access_duty.tolist(),
            **{k: v.tolist() for k, v in components.items()},
            consumed_j=used.tolist(),
            inserted_j=inserted.tolist(),
            removed_j=removed.tolist(),
            shortfall_j=(requested - used).tolist(),
            swap_remaining=self.swap_remaining.tolist(),
            balance_residual_j=residual.tolist(),
        )
        self.last_energy_probe = dict(
            depleted_points=float(used.sum() / slot.e.capacity_j * 100),
            charged_points=float((inserted - removed).sum() / slot.e.capacity_j * 100),
            replacement_net_j=float((inserted - removed).sum()),
            inserted_pack_j=float(inserted.sum()),
            removed_pack_j=float(removed.sum()),
            extra_hybrid_debit_points=0.0,
            consumed_j=float(used.sum()),
            consumed_wh=float(used.sum() / 3600),
            **{k: float(v.sum()) for k, v in components.items()},
            distance_m=float(distance.sum()),
            swaps_completed=float(len(swapped)),
            swapping_uavs=float(np.sum(slot.effective == 2)),
            energy_cost_norm=cost_norm,
            reward_energy=-penalty,
            reward_total=slot.reward,
            energy_shortfall_j=float((requested - used).sum()),
        )
        self.last_return_control = dict(
            effective_status=slot.effective.tolist(),
            stranded_ids_post=stranded,
            route_decisions=deepcopy(self._route_decisions),
        )

    def set_battery_soc(self, percent):
        """Explicit fixture API; do not mutate battery directly in this version."""
        soc = np.broadcast_to(np.asarray(percent, dtype=float), (5,))
        if not np.isfinite(soc).all() or np.any((soc < 0) | (soc > 100)):
            raise ValueError("SOC must be in [0,100] percent")
        self.energy_j[:] = soc / 100 * self.energy_config.capacity_j
        self._sync_soc()

    def _sync_soc(self):
        self.battery[:] = self.energy_j / self.energy_config.capacity_j * self.cfg.max_battery

    def _route_budget(self, cell, agent):
        plan = self._plan(cell, agent)
        flight = max(self.energy_config.hover_w, self.energy_config.cruise_w)
        cost = flight * self.cfg.slot_s if self.energy_config.include_flight_energy else 0.0
        return (plan, plan["trigger_steps"] * cost)

    def _start_return(self, aid, reason):
        plan, budget = self._route_budget(self.uav_positions[aid], aid)
        self.status[aid] = 1
        self.target_station_idx[aid] = plan["station"]
        event_slot = self.t + int(reason == "next_slot_reserve")
        self.return_start_time[aid] = event_slot
        self._route_cache[aid] = None
        self._previous_return_cell[aid] = None
        self.energy_events.append(
            dict(
                event="return_start",
                slot=event_slot,
                agent=int(aid),
                reason=reason,
                position=self.uav_positions[aid].tolist(),
                station=int(plan["station"]),
                route_steps=plan["steps"],
                route_m=None if plan["steps"] is None else plan["steps"] * self.cfg.cell_size_m,
                soc_percent=float(self.battery[aid]),
                route_budget_j=budget,
            )
        )

    def _status_name(self, status):
        return {0: "Active", 1: "Returning", 2: "Swapping", 3: "CurriculumInactive"}[int(status)]
