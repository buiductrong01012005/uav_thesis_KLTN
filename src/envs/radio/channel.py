"""Radio interface used by the live simulator and independent baseline beliefs."""

from copy import deepcopy
from dataclasses import asdict
from typing import Iterable
import math
import time
import numpy as np
from src.envs.config import SimulationConfig, ServiceV2Config
from src.envs.radio.multihop import score_multihop


class RadioMixin:
    def compute_network(self):
        tic = time.perf_counter()
        result = score_multihop(self, self.logical_roles(), self.reward_weights)
        self.last_probe_result = result
        rates, association = (result["ue_rate_mbps"], result["association"])
        edges, paths = (result["edges"], result["paths"])
        routes = {
            a: int(edges[path[-1]].source[1:]) if len(path) > 1 else None
            for a, path in paths.items()
        }
        net = self._network_result(rates, association, routes, result["flows"])
        net.update(
            access_offer_mbps={
                a: float(result["access_raw_ue_rates"][users].sum())
                for a, users in association.items()
            },
            access_raw_ue_rates=result["access_raw_ue_rates"],
            radio_runtime_ms=(time.perf_counter() - tic) * 1000,
            radio_flow_violation_mbps=result["metrics"]["flow_violation_mbps"],
            schedule_time_sum=result["metrics"]["schedule_airtime"],
            routing_version="bs5_multihop_forest_1",
            forwarded_mbps_by_uav=result["forwarded_mbps_by_uav"],
            path_wireless_hops=result["path_wireless_hops"],
        )
        self.last_v2_radio_detail = dict(
            edges=[asdict(e) for e in edges],
            paths=paths,
            schedules=[
                dict(active_edges=list(state), duration=float(result["durations"][j]))
                for j, state in enumerate(result["states"])
                if result["durations"][j] > 1e-10
            ],
        )
        if getattr(self, "_in_step", False):
            self.last_v2_network = net
            self.last_service_snapshot = dict(
                service_t=int(self.t),
                failed_bs_idx=5,
                uav_positions=self.uav_positions.copy(),
                ue_positions=self.ue_positions.copy(),
                roles=result["roles"].copy(),
                status=self.status.copy(),
                battery=self.battery.copy(),
                network=deepcopy(net),
            )
        return net

    def _network_result(self, ue_rate, assoc, routes, bh_cap):
        utility = float(np.mean(np.minimum(ue_rate / self.cfg.ue_demand_mbps, 1.0)))
        return {
            "utility": utility,
            "ue_rate_mbps": np.asarray(ue_rate, dtype=np.float64),
            "association": assoc,
            "routes": {int(k): None if v is None else int(v) for k, v in routes.items()},
            "access_loads": {int(k): len(v) for k, v in assoc.items()},
            "backhaul_caps_mbps": {int(k): float(v) for k, v in bh_cap.items()},
        }

    def _a2g_path_loss_db(self, p1_xy, h1, p2_xy, h2, fc_hz) -> float:
        p1 = np.asarray(p1_xy, dtype=np.float64)
        p2 = np.asarray(p2_xy, dtype=np.float64)
        horizontal = float(np.linalg.norm(p1 - p2))
        dh = abs(float(h2 - h1))
        d3 = max(math.sqrt(horizontal * horizontal + dh * dh), 1.0)
        theta = math.degrees(math.atan2(dh, max(horizontal, 1e-09)))
        a, b = (self.cfg.urban_a, self.cfg.urban_b)
        p_los = 1.0 / (1.0 + a * math.exp(-b * (theta - a)))
        fspl = 20.0 * math.log10(d3) + 20.0 * math.log10(fc_hz) - 147.55
        extra = p_los * self.cfg.eta_los_db + (1.0 - p_los) * self.cfg.eta_nlos_db
        return fspl + extra

    def _a2a_path_loss_db(self, p1_xy, p2_xy, fc_hz) -> float:
        p1 = np.asarray(p1_xy, dtype=np.float64)
        p2 = np.asarray(p2_xy, dtype=np.float64)
        d = max(float(np.linalg.norm(p1 - p2)), self.cfg.cell_size_m)
        fspl = 20.0 * math.log10(d) + 20.0 * math.log10(fc_hz) - 147.55
        return fspl + self.cfg.a2a_extra_loss_db

    def _shannon_rate_mbps(self, rx_dbm: float, bw_hz: float) -> float:
        noise_dbm = self.cfg.noise_psd_dbm_hz + 10.0 * math.log10(bw_hz) + self.cfg.noise_figure_db
        snr_linear = 10.0 ** ((rx_dbm - noise_dbm) / 10.0)
        return float(bw_hz * math.log2(1.0 + snr_linear) / 1000000.0)

    def _bh_a2g_rate(self, p1_xy, h1, p2_xy, h2) -> float:
        pl = self._a2g_path_loss_db(p1_xy, h1, p2_xy, h2, self.cfg.backhaul_fc_hz)
        return self._shannon_rate_mbps(self.cfg.backhaul_tx_dbm - pl, self.cfg.backhaul_bw_hz)

    def _bh_a2a_rate(self, p1_xy, p2_xy) -> float:
        pl = self._a2a_path_loss_db(p1_xy, p2_xy, self.cfg.backhaul_fc_hz)
        return self._shannon_rate_mbps(self.cfg.backhaul_tx_dbm - pl, self.cfg.backhaul_bw_hz)

    def _xy_to_cell(self, xy: Iterable[float]) -> np.ndarray:
        x, y = map(float, xy)
        c = int(np.clip(x // self.cfg.cell_size_m, 0, self.cfg.grid_size - 1))
        r = int(np.clip(y // self.cfg.cell_size_m, 0, self.cfg.grid_size - 1))
        return np.asarray([r, c], dtype=np.int64)

    def _cell_to_xy(self, cell: Iterable[float]) -> np.ndarray:
        r, c = map(float, cell)
        return np.asarray(
            [(c + 0.5) * self.cfg.cell_size_m, (r + 0.5) * self.cfg.cell_size_m], dtype=np.float64
        )

    def _within_radius_cells(self, p1_grid, p2_grid, radius_cells: float) -> bool:
        p1 = np.asarray(p1_grid, dtype=np.float64)
        p2 = np.asarray(p2_grid, dtype=np.float64)
        return bool(np.linalg.norm(p1 - p2) <= float(radius_cells) + 1e-09)

    def _xy_within_radius_m(self, p1_xy, p2_xy, radius_cells: float) -> bool:
        return bool(
            np.linalg.norm(
                np.asarray(p1_xy, dtype=np.float64) - np.asarray(p2_xy, dtype=np.float64)
            )
            <= radius_cells * self.cfg.cell_size_m + 1e-09
        )


class RadioGeometry(RadioMixin):
    """Independent public radio template; baseline estimates replace all UE/fleet data."""

    def __init__(self):
        self.service_config = ServiceV2Config()
        self.cfg = SimulationConfig()
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
        self.bs_xy = np.asarray(self.cfg.bs_xy_m, dtype=np.float64)
        self.surviving_bs_xy = np.delete(self.bs_xy, self.cfg.failed_bs_idx, axis=0)
