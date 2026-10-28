"""Controlled lifecycle probe, NOT a Hybrid actor environment or RL interface.

Old 10-action navigation is retained; selected Relay actions are externally
marked Hybrid. Parent lifecycle uses a Relay proxy; actual radio uses H and
extra active energy is debited before the parent's reserve/return update.
Observations/state are stripped from outputs: old telemetry cannot encode H.
Never train an actor from this diagnostic harness.
"""

from copy import deepcopy
from dataclasses import asdict
import time
import numpy as np

from src.diagnostics.bs5_hybrid_fdm import HybridFDMConfig, score_hybrid_fdm
from src.diagnostics.bs5_hybrid_full import HybridFullConfig
from src.envs.bs5_tcf_env import BS5TCFEnv

VERSION = "bs5_hybrid_lifecycle_probe_1"


class HybridLifecycleProbe(BS5TCFEnv):
    def __init__(self, *, mode="full", hybrid_radio_overhead=0.0, **kwargs):
        if (
            mode not in {"full", "split"}
            or not np.isfinite(hybrid_radio_overhead)
            or hybrid_radio_overhead < 0
        ):
            raise ValueError("Invalid probe mode/energy overhead")
        self.probe_mode = mode
        self.hybrid_config = HybridFullConfig() if mode == "full" else HybridFDMConfig()
        self.hybrid_radio_overhead = float(hybrid_radio_overhead)
        self._hybrid_ids = set()
        self.hybrid_extra_debit = np.zeros(5)
        self.last_probe_result = None
        self.last_energy_probe = None
        super().__init__(**kwargs)

    @staticmethod
    def _diagnostic_output(out):
        for name in ["obs", "state"]:
            out.pop(name, None)
        out["diagnostic_only"] = True
        return out

    def reset(self, *args, **kwargs):
        self._hybrid_ids = set()
        self.hybrid_extra_debit = np.zeros(5)
        self.last_probe_result = None
        return self._diagnostic_output(super().reset(*args, **kwargs))

    def logical_roles(self):
        roles = self.roles.copy()
        for aid in self._hybrid_ids:
            if roles[aid] == 1:
                roles[aid] = 2
        return roles

    def _resolve_collisions(self, proposed, old, status_before):
        self.hybrid_extra_debit = np.zeros(5)
        forced = []
        before = self.battery.copy()
        if self.condition["battery"]:
            alpha = self.hybrid_config.access_fraction
            beta = self.hybrid_config.backhaul_fraction
            hybrid_drain = (
                alpha * self.cfg.access_radio_drain
                + beta * self.cfg.relay_radio_drain
                + self.hybrid_radio_overhead
            )
            extra = hybrid_drain - self.cfg.relay_radio_drain
            if extra < 0:
                raise ValueError("Probe does not support Hybrid radio cheaper than Relay proxy")
            for aid in sorted(self._hybrid_ids):
                if status_before[aid] != 0:
                    continue
                move = float(not np.array_equal(proposed[aid], old[aid]))
                full_cost = self.cfg.hover_drain + move * self.cfg.move_extra_drain + hybrid_drain
                if float(self.battery[aid]) < full_cost:
                    status_before[aid] = 1
                    self.status[aid] = 1
                    if self.return_start_time[aid] < 0:
                        self.return_start_time[aid] = self.t + 1
                    forced.append(aid)
                else:
                    self.battery[aid] = max(0.0, float(self.battery[aid]) - extra)
                    self.hybrid_extra_debit[aid] = extra
        super()._resolve_collisions(proposed, old, status_before)
        effective = self.last_return_control["effective_status"]
        for aid in self._hybrid_ids:
            if effective[aid] != 0 and self.hybrid_extra_debit[aid] > 0:
                # A numerical-boundary override must not pay both H and Return.
                self.battery[aid] = before[aid]
                self.hybrid_extra_debit[aid] = 0
        self.last_return_control["forced_return_ids"] = sorted(
            set(self.last_return_control["forced_return_ids"]) | set(forced)
        )

    def compute_network(self):
        if getattr(self, "_v2_initializing", True):
            return super().compute_network()
        tic = time.perf_counter()
        result = score_hybrid_fdm(
            self, self.logical_roles(), self.hybrid_config, self.reward_weights
        )
        self.last_probe_result = result
        rates, association = result["ue_rate_mbps"], result["association"]
        edges, paths = result["edges"], result["paths"]
        routes = {
            a: int(edges[path[0]].target[1:]) if len(path) == 2 else None
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
            probe_version=VERSION,
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

    def step(self, actions, *, hybrid_ids=()):
        requested = set(map(int, hybrid_ids))
        if not requested.issubset(set(range(5))):
            raise ValueError("Invalid Hybrid agent ID")
        self._hybrid_ids = {
            a for a in requested if self.status[a] == 0 and int(actions[a]) % 2 == 1
        }
        before = self.battery.astype(float).copy()
        out = super().step(actions)
        after = self.battery.astype(float)
        self.last_energy_probe = dict(
            depleted_points=float(np.maximum(before - after, 0).sum()),
            charged_points=float(np.maximum(after - before, 0).sum()),
            extra_hybrid_debit_points=float(self.hybrid_extra_debit.sum()),
        )
        out["info"]["hybrid_probe"] = dict(
            version=VERSION,
            mode=self.probe_mode,
            **self.last_energy_probe,
            metrics=self.last_probe_result["metrics"],
        )
        return self._diagnostic_output(out)
