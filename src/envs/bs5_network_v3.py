"""Versioned network audit branch; old service-v2 and TCF files are untouched."""

from copy import copy, deepcopy
from dataclasses import asdict
import time
import numpy as np
from src.envs.bs5_tcf_env import BS5TCFEnv
from src.envs.bs5_launch_ab_env import score_network
from src.diagnostics.bs5_spectrum_airtime import access_rates, allocate_flows
from src.diagnostics.bs5_radio_v3 import RadioV3Config, build_routes_v3, schedule_matrix_v3

VERSION = "bs5_network_v3_1"


class BS5NetworkV3Env(BS5TCFEnv):
    def __init__(self, *args, radio_config=None, **kwargs):
        self.radio_config = radio_config or RadioV3Config()
        super().__init__(*args, **kwargs)
        self.cfg.access_radius_cells = self.radio_config.access_radius_m / self.cfg.cell_size_m
        self.cfg.backhaul_radius_cells = self.radio_config.backhaul_radius_m / self.cfg.cell_size_m
        self.cfg.uav_altitude_m = self.radio_config.uav_altitude_m
        self.cfg.backhaul_tx_dbm = self.radio_config.uav_bh_tx_dbm

    def compute_network(self):
        if self._v2_initializing:
            return score_network(self, self.cfg.allow_relay)
        tic = time.perf_counter()
        association = score_network(self, self.cfg.allow_relay)["association"]
        view = copy(self)
        view.roles = self.roles.copy()
        view.roles[self.status != 0] = -1
        if not self.cfg.allow_relay:
            view.roles[view.roles == 1] = -1
        c = self.service_config
        before, _, budget = access_rates(
            view,
            association,
            c.access_channel_mhz * c.access_occupied_fraction,
            c.spectrum_mode,
            c.rate_efficiency,
        )
        offers = {a: float(before[users].sum()) for a, users in association.items()}
        bw = c.backhaul_total_mhz * c.backhaul_occupied_fraction
        edges, paths = build_routes_v3(view, association, bw, c.rate_efficiency)
        matrix, states = schedule_matrix_v3(view, edges, bw, c.rate_efficiency)
        flows, durations, minimum, violation = allocate_flows(offers, paths, matrix)
        final = np.zeros(self.cfg.num_ues)
        routes = {}
        for a, users in association.items():
            if offers[a] > 0:
                final[users] = before[users] * flows[a] / offers[a]
            routes[a] = int(edges[paths[a][0]].target[1:]) if len(paths[a]) == 2 else None
        net = self._network_result(final, association, routes, flows)
        net.update(
            access_offer_mbps=offers,
            access_raw_ue_rates=before,
            radio_runtime_ms=(time.perf_counter() - tic) * 1000,
            radio_flow_violation_mbps=violation,
            schedule_time_sum=float(durations.sum()),
            min_access_offer_fulfillment=minimum,
            network_version=VERSION,
            radio_config=asdict(self.radio_config),
            access_nominal_channel_mhz=c.access_channel_mhz,
            access_nominal_reserved_fleet_mhz=budget.get("access_reserved_fleet_mhz", 0)
            / c.access_occupied_fraction,
            backhaul_total_mhz=c.backhaul_total_mhz,
            backhaul_occupied_mhz=bw,
        )
        detail = dict(
            edges=[asdict(e) for e in edges],
            paths=paths,
            flows=flows,
            schedules=[
                dict(
                    active_edges=list(state),
                    duration=float(durations[j]),
                    capacity_mbps=matrix[:, j].tolist(),
                )
                for j, state in enumerate(states)
                if durations[j] > 1e-10
            ],
        )
        if getattr(self, "_in_step", False):
            self.last_v2_network, self.last_v2_radio_detail = net, detail
            self.last_service_snapshot = dict(
                service_t=int(self.t),
                failed_bs_idx=5,
                uav_positions=self.uav_positions.copy(),
                roles=self.roles.copy(),
                status=self.status.copy(),
                battery=self.battery.copy(),
                ue_positions=self.ue_positions.copy(),
                network=deepcopy(net),
            )
        return net

    def _episode_meta(self):
        meta = super()._episode_meta()
        meta.update(
            network_version=VERSION,
            radio_config=asdict(self.radio_config),
            energy_scope="legacy normalized role/movement drain; fixed altitude has no ascent energy; BS power not drawn from UAV battery",
        )
        return meta
