"""Final BS5 geometry and resource defaults; no historical environment inheritance."""

from src.configuration import environment_defaults

from dataclasses import dataclass
import numpy as np

ACCESS, RELAY = 0, 1
ACTIVE, RETURNING, SWAPPING, INACTIVE = 0, 1, 2, 3
MOVE_DELTAS = np.asarray([[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]], dtype=np.int64)
STAY = 4


@dataclass
@environment_defaults("simulation")
class SimulationConfig:
    seed: int
    grid_size: int
    cell_size_m: float
    horizon: int
    slot_s: float
    bs_xy_m: tuple[tuple[float, float], ...]
    failed_bs_idx: int
    uav_start_cells: tuple[tuple[int, int], ...]
    charging_station_cells: tuple[tuple[int, int], ...]
    uav_altitude_m: float
    bs_height_m: float
    ue_height_m: float
    access_fc_hz: float
    backhaul_fc_hz: float
    access_tx_dbm: float
    backhaul_tx_dbm: float
    noise_psd_dbm_hz: float
    noise_figure_db: float
    urban_a: float
    urban_b: float
    eta_los_db: float
    eta_nlos_db: float
    a2a_extra_loss_db: float
    num_ues: int
    num_hotspots: int
    ue_demand_mbps: float
    max_ues_per_access: int
    max_ue_offset_cells: float
    max_battery: float
    access_radius_cells: float
    backhaul_radius_cells: float
    residual_initial_battery: float
    randomize_residual_uav: bool
    group_center_speed_mps: float
    group_turn_noise_std_rad: float
    ue_local_speed_min_mps: float
    ue_local_speed_max_mps: float
    ue_local_turn_noise_std_rad: float
    group_spread_cells: float
    initialize_residual_battery: bool


@dataclass(frozen=True)
@environment_defaults("service")
class ServiceV2Config:
    access_channel_mhz: float
    access_occupied_fraction: float
    backhaul_total_mhz: float
    backhaul_occupied_fraction: float
    spectrum_mode: str
    rate_efficiency: float
    report_period_slots: int
    report_delay_slots: int
    report_max_age_slots: int
    beacon_radius_m: float

    def __post_init__(self):
        if min(self.access_channel_mhz, self.backhaul_total_mhz, self.beacon_radius_m) <= 0:
            raise ValueError("Resource budgets must be positive")
        if not all(
            0 < x <= 1
            for x in [
                self.access_occupied_fraction,
                self.backhaul_occupied_fraction,
                self.rate_efficiency,
            ]
        ):
            raise ValueError("Fractions outside (0,1]")
        if self.spectrum_mode not in {
            "colored_reuse",
            "orthogonal_pool",
            "reuse1",
            "orthogonal_per_uav",
        }:
            raise ValueError("Unknown spectrum mode")
        if (
            not isinstance(self.report_period_slots, int)
            or self.report_period_slots < 1
            or not isinstance(self.report_delay_slots, int)
            or self.report_delay_slots < 0
            or self.report_max_age_slots < self.report_period_slots + self.report_delay_slots + 1
        ):
            raise ValueError("Invalid reporting timing")


ARM_ACTIONS = {"C-N": 10, "C-R": 10, "C-H": 15, "H-only": 5}
