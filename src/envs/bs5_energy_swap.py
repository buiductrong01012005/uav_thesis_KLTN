"""Physical-unit, explicitly approximate battery-swap experiment. Legacy untouched."""
from dataclasses import asdict, dataclass
from copy import deepcopy
import math
import numpy as np
from src.envs.bs5_multihop_train import BS5MultihopTrainEnv
from src.envs.bs5_collision_safe_env import resolve_endpoints
from src.envs.v0_env import MOVE_DELTAS
from src.envs.bs5_tcf_env import rate_metrics

VERSION = 'bs5_multihop_energy_swap_v1'
REWARD_VERSION = 'weighted_service_minus_energy_v1'


@dataclass(frozen=True)
class EnergyConfig:
    capacity_wh: float = 129.96
    hover_w: float = 474.76260465841614
    cruise_w: float = 434.4545004414246
    active_circuit_w: float = 31.5
    pa_efficiency: float = .25
    arrival_soc: float = .20
    swap_seconds: float = 60.
    energy_weight: float = 0.
    include_flight_energy: bool = True
    # Fixed reference, independent of arm, active count, seed and ablation.
    reference_w: float = 508.6673535929253

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for k, v in asdict(self).items()
               if k != 'include_flight_energy'):
            raise ValueError('Nonnegative finite energy parameters required')
        if not (self.capacity_wh > 0 and self.reference_w > 0 and
                0 < self.pa_efficiency <= 1 and 0 <= self.arrival_soc < 1 and
                self.swap_seconds > 0):
            raise ValueError('Invalid energy configuration')

    @property
    def capacity_j(self):
        return self.capacity_wh * 3600


def linear_learning_rate(step, total_steps, start=3e-4, end=3e-5):
    if total_steps <= 0 or not 0 < end <= start:
        raise ValueError('Invalid LR schedule')
    progress = np.clip(float(step) / total_steps, 0., 1.)
    return float(start + (end-start)*progress)


class BS5EnergySwapEnv(BS5MultihopTrainEnv):
    return_version = "energy_swap_committed_bfs_reserve_v1"
    def __init__(self, *args, energy_config=None, **kwargs):
        self.energy_config = energy_config or EnergyConfig()
        self.energy_events = []
        super().__init__(*args, **kwargs)
        slots = self.energy_config.swap_seconds / self.cfg.slot_s
        if not math.isclose(slots, round(slots)):
            raise ValueError('Swap duration must be a whole number of slots')
        self.swap_slots = int(round(slots))

    def reset(self, *args, **kwargs):
        out = super().reset(*args, **kwargs)
        self.energy_j = self.battery.astype(float) / self.cfg.max_battery * self.energy_config.capacity_j
        self.swap_remaining = np.zeros(5, dtype=int)
        self.energy_events = []
        self.total_consumed_j = np.zeros(5)
        self.last_energy_audit = {}
        out['episode_meta'] = self._episode_meta()
        return out

    def set_battery_soc(self, percent):
        """Explicit fixture API; do not mutate battery directly in this version."""
        soc = np.broadcast_to(np.asarray(percent, dtype=float), (5,))
        if not np.isfinite(soc).all() or np.any((soc < 0) | (soc > 100)):
            raise ValueError('SOC must be in [0,100] percent')
        self.energy_j[:] = soc/100*self.energy_config.capacity_j
        self._sync_soc()

    def _sync_soc(self):
        self.battery[:] = self.energy_j/self.energy_config.capacity_j*self.cfg.max_battery

    def _status_name(self, status):
        return 'Swapping' if int(status) == 2 else super()._status_name(status)

    def _route_budget(self, cell, agent):
        # A conservative per-slot bound covers flight and waiting, but not
        # indefinite blocking. No finite reserve can cover arbitrary blockers.
        plan = self._plan(cell, agent)
        flight = max(self.energy_config.hover_w, self.energy_config.cruise_w)
        cost = flight*self.cfg.slot_s if self.energy_config.include_flight_energy else 0.
        return plan, plan['trigger_steps']*cost

    def _start_return(self, aid, reason):
        plan, budget = self._route_budget(self.uav_positions[aid], aid)
        self.status[aid] = 1
        self.target_station_idx[aid] = plan['station']
        event_slot = self.t+int(reason == 'next_slot_reserve')
        self.return_start_time[aid] = event_slot
        self._route_cache[aid] = None
        self._previous_return_cell[aid] = None
        self.energy_events.append(dict(event='return_start', slot=event_slot, agent=int(aid),
            reason=reason, position=self.uav_positions[aid].tolist(),
            station=int(plan['station']), route_steps=plan['steps'],
            route_m=None if plan['steps'] is None else plan['steps']*self.cfg.cell_size_m,
            soc_percent=float(self.battery[aid]), route_budget_j=budget))

    def step(self, actions):
        raw = np.asarray(actions)
        mask = self.get_action_mask()
        if raw.shape != (5,) or not np.equal(raw, raw.astype(int)).all():
            raise ValueError('Five integer actions required')
        raw = raw.astype(int)
        if any(a < 0 or a >= self.n_actions or not mask[i, a] for i, a in enumerate(raw)):
            raise ValueError('Invalid or masked action')
        e = self.energy_config
        if not np.allclose(self.battery/self.cfg.max_battery*e.capacity_j, self.energy_j, atol=1e-7):
            raise ValueError('Use set_battery_soc(), not direct battery mutation')
        enabled = bool(self.condition['battery'])
        before_e = self.energy_j.copy()
        original_status = self.status.copy()
        old = self.uav_positions.copy()
        old_roles = self.logical_roles().copy()
        self._slot_actions = raw.tolist()
        # Preflight uses requested endpoint: even blocked moves can cost hover.
        active_max_j = ((max(e.hover_w, e.cruise_w) if e.include_flight_energy else 0)
            + e.active_circuit_w + (10**((self.cfg.access_tx_dbm-30)/10)
            + 10**((self.cfg.backhaul_tx_dbm-30)/10))/e.pa_efficiency)*self.cfg.slot_s
        floor = e.arrival_soc*e.capacity_j
        if enabled:
            for aid in np.flatnonzero(self.status == 0):
                move, _ = self.decode_action(raw[aid])
                candidate = old[aid] + MOVE_DELTAS[move]
                _, budget = self._route_budget(candidate, int(aid))
                if self.energy_j[aid] < floor + budget + active_max_j:
                    self._start_return(int(aid), 'preflight_reserve')
        effective = self.status.copy()
        proposed = old.copy()
        roles = self.logical_roles().copy()
        self._return_planning = True
        self._route_decisions = {}
        try:
            for aid in range(5):
                if effective[aid] == 0:
                    move, role = self.decode_action(raw[aid])
                    proposed[aid] = old[aid] + MOVE_DELTAS[move]
                    roles[aid] = role
                elif effective[aid] == 1:
                    plan = self._plan(old[aid], aid)
                    self.target_station_idx[aid] = plan['station']
                    return_move_j = e.cruise_w*self.cfg.slot_s if e.include_flight_energy else 0.
                    if plan['path'] is not None and len(plan['path']) > 1 and (not enabled or self.energy_j[aid] >= return_move_j):
                        proposed[aid] = plan['path'][1]
        finally:
            self._return_planning = False
        self.last_collision_resolution = resolve_endpoints(proposed, old, effective, self.charging_stations)
        self.uav_positions = proposed
        self._hybrid_ids = set(np.flatnonzero((roles == 2) & (effective == 0)).tolist())
        self.roles[:] = np.where(roles == 2, 1, roles)
        for aid in np.flatnonzero(effective == 1):
            plan = self._route_cache[aid]
            if not np.array_equal(old[aid], proposed[aid]) and plan is not None:
                self._previous_return_cell[aid] = old[aid].tolist()
                plan['path'] = plan['path'][1:]
                plan['steps'] = plan['trigger_steps'] = len(plan['path'])-1
            station = self.charging_stations[self.target_station_idx[aid]]
            if np.array_equal(proposed[aid], station):
                self.status[aid] = 2
                self.swap_remaining[aid] = self.swap_slots
                self.station_arrival_time[aid] = self.t+1
                self.energy_events.append(dict(event='swap_start',slot=self.t+1,agent=int(aid),
                    station=int(self.target_station_idx[aid])))
        self._in_step = True
        try:
            net = self.compute_network()
        finally:
            self._in_step = False
        # BH downlink edges carry source -> target; account for Tx and Rx separately.
        tx, rx = np.zeros(5), np.zeros(5)
        for schedule in self.last_v2_radio_detail['schedules']:
            for idx in schedule['active_edges']:
                edge = self.last_v2_radio_detail['edges'][idx]
                for endpoint, fractions in [('source', tx), ('target', rx)]:
                    name = edge[endpoint]
                    if name.startswith('u'):
                        fractions[int(name[1:])] += schedule['duration']
        if np.any(tx+rx > 1+1e-6):
            raise AssertionError('Half-duplex schedule violated')
        distance = np.linalg.norm(proposed-old, axis=1)*self.cfg.cell_size_m
        airborne = np.isin(effective, [0,1])
        service_active = self.last_service_snapshot['status'] == 0
        access_duty = np.array([float(service_active[i] and roles[i] in (0,2)
            and len(net['association'].get(i, [])) > 0) for i in range(5)])
        flight = np.where(airborne, np.where(distance > 0, e.cruise_w, e.hover_w), 0.)
        if not e.include_flight_energy:
            flight[:] = 0
        components = dict(flight_j=flight*self.cfg.slot_s,
            circuit_j=service_active.astype(float)*e.active_circuit_w*self.cfg.slot_s,
            access_pa_j=access_duty*10**((self.cfg.access_tx_dbm-30)/10)/e.pa_efficiency*self.cfg.slot_s,
            backhaul_pa_j=tx*10**((self.cfg.backhaul_tx_dbm-30)/10)/e.pa_efficiency*self.cfg.slot_s)
        if not enabled:
            components = {k: np.zeros(5) for k in components}
        requested = sum(components.values())
        # Preserve component conservation even under exhaustion caused by blockers.
        scale = np.minimum(1., np.divide(before_e, requested, out=np.ones(5), where=requested > 0))
        components = {k: v*scale for k,v in components.items()}
        used = sum(components.values())
        self.energy_j -= used
        self.total_consumed_j += used
        inserted, removed = np.zeros(5), np.zeros(5)
        swapped = []
        if enabled:
            for aid in np.flatnonzero(effective == 2):
                self.swap_remaining[aid] -= 1
                if self.swap_remaining[aid] == 0:
                    removed[aid] = self.energy_j[aid]
                    inserted[aid] = e.capacity_j
                    self.energy_j[aid] = e.capacity_j
                    self.status[aid], self.roles[aid] = 0, 0
                    self.target_station_idx[aid] = -1
                    self.rejoin_time[aid] = self.t+1
                    swapped.append(int(aid))
                    self.energy_events.append(dict(event='swap_complete',slot=self.t+1,
                        agent=int(aid), removed_j=float(removed[aid]), inserted_j=e.capacity_j))
        self._sync_soc()
        if enabled:
            for aid in np.flatnonzero((self.status == 0) & (effective == 0)):
                _, budget = self._route_budget(self.uav_positions[aid], int(aid))
                if self.energy_j[aid] < floor+budget+active_max_j:
                    self._start_return(int(aid), 'next_slot_reserve')
        stranded = np.flatnonzero((self.status == 1) & (self.energy_j <= 1e-7)).tolist()
        for aid in np.flatnonzero(self.status != 1):
            self._route_cache[aid] = None
            self._previous_return_cell[aid] = None
        residual = before_e + inserted - removed - used - self.energy_j
        if np.max(np.abs(residual)) > 1e-7:
            raise AssertionError('Battery conservation failed')
        cost_norm = float(used.sum()/(5*e.reference_w*self.cfg.slot_s))
        metrics = rate_metrics(net['ue_rate_mbps'], self.cfg.ue_demand_mbps,
            self.coverage_floor_mbps, self.reward_weights)
        original_qos = metrics['qos_ues'], metrics['qos_fraction']
        metrics['qos_ues'] = int(np.sum(net['ue_rate_mbps'] >= self.cfg.ue_demand_mbps-1e-6))
        metrics['qos_fraction'] = metrics['qos_ues']/self.cfg.num_ues
        penalty = e.energy_weight*cost_norm
        reward = metrics['tcf_score']-penalty
        self.last_energy_audit = dict(slot=self.t+1, energy_before_j=before_e.tolist(),
            energy_after_j=self.energy_j.tolist(), effective_status=effective.tolist(),
            post_status=self.status.tolist(), roles=roles.tolist(), distance_m=distance.tolist(),
            bh_tx_fraction=tx.tolist(), bh_rx_fraction=rx.tolist(),access_tx_fraction=access_duty.tolist(),
            **{k:v.tolist() for k,v in components.items()}, consumed_j=used.tolist(),
            inserted_j=inserted.tolist(), removed_j=removed.tolist(),
            shortfall_j=(requested-used).tolist(), swap_remaining=self.swap_remaining.tolist(),
            balance_residual_j=residual.tolist())
        self.last_energy_probe = dict(depleted_points=float(used.sum()/e.capacity_j*100),
            charged_points=float((inserted-removed).sum()/e.capacity_j*100),
            replacement_net_j=float((inserted-removed).sum()),
            inserted_pack_j=float(inserted.sum()), removed_pack_j=float(removed.sum()),
            extra_hybrid_debit_points=0., consumed_j=float(used.sum()),
            consumed_wh=float(used.sum()/3600), **{k:float(v.sum()) for k,v in components.items()},
            distance_m=float(distance.sum()), swaps_completed=float(len(swapped)),
            swapping_uavs=float(np.sum(effective == 2)), energy_cost_norm=cost_norm,
            reward_energy=-penalty, reward_total=reward,
            energy_shortfall_j=float((requested-used).sum()))
        self.last_return_control = dict(effective_status=effective.tolist(),stranded_ids_post=stranded,
            route_decisions=deepcopy(self._route_decisions))
        if self.cfg.enable_ue_mobility:
            self._advance_mobile_ues()
        self.t += 1
        self.reports.capture(self, net)
        self.reports.deliver(self.t)
        info = dict(t=self.t, **metrics, **self.last_energy_probe,
            utility=float(net['utility']), radio_runtime_ms=net['radio_runtime_ms'],
            qos_ues_original=original_qos[0],qos_fraction_original=original_qos[1],
            status=self.status.tolist(),status_names=[self._status_name(s) for s in self.status],
            battery=self.battery.tolist(),roles=self.logical_roles().tolist(),
            uav_positions=self.uav_positions.tolist(),routes=net['routes'],access_loads=net['access_loads'],
            backhaul_caps_mbps=net['backhaul_caps_mbps'],
            role_switches_this_slot=int(np.sum((old_roles != roles) & (effective == 0))),
            training_env_version=VERSION,reward_version=REWARD_VERSION,
            original_status=original_status.tolist())
        return dict(obs=self.get_obs(),state=self.get_state(),action_mask=self.get_action_mask(),
            reward=reward,terminated=False,truncated=self.t >= self.cfg.horizon,info=info)

    def _episode_meta(self):
        meta = super()._episode_meta()
        meta.update(training_env_version=VERSION,reward_version=REWARD_VERSION,
            routing='Unchanged multihop BS-rooted forest and half-duplex flow/schedule LP',
            legacy_normalized_energy_parameters_used=False,
            energy_config=asdict(self.energy_config), energy_contract='Joules; empirical two-state battery electrical flight baseline + active common circuit + actual RF Tx airtime / PA efficiency',
            lifecycle_contract='Return pays flight, payload radio off; 12 complete grounded slots after arrival; replace pack to 100%; unlimited prepared spare packs assumed; station hubs shared',
            collision_contract='Inherited endpoint occupancy/priority; actual resolved displacement determines cruise vs hover energy',
            return_contract='Committed BFS; conservative route reserve + 20% arrival floor + next active slot; indefinite obstruction may exhaust reserve',
            excluded_energy='Vertical maneuvers, turns, battery aging, ground charging and spare inventory not modeled; empirical baseline includes original avionics')
        return meta
