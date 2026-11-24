"""Causal beacon broadcasts, EMA history and actor/mixer tensors."""

from collections import deque
import numpy as np

OBS_DIM, STATE_DIM = (173, 156)
ROLE_SCALARS = {0: 0.0, 1: 1.0, 2: 0.5}


def own_features(env, agent):
    roles = env.logical_roles()
    role = ROLE_SCALARS[int(roles[agent])] if env.status[agent] == 0 else 0.0
    return [
        *(env.uav_positions[agent] / 19).tolist(),
        role,
        float(env.battery[agent] / env.cfg.max_battery),
        *(float(env.status[agent] == s) for s in [0, 1, 2]),
    ]


class BeaconReports:
    """Ideal local request beacon reception + delayed common control broadcast.

    A UE is detected only within beacon radius of an Active Access UAV. Reports
    deduplicate UE IDs. Counts are *detected* demand, not a full census. Cells
    keep the latest sampled detected count and footprint fraction, with age;
    old data expires. No assumed satellite/global user-location feed.
    """

    def __init__(self, config):
        self.config = config
        self.queue = deque()
        self.counts = np.zeros((5, 5), dtype=np.float32)
        self.coverage = np.zeros((5, 5), dtype=np.float32)
        self.timestamps = np.full((5, 5), -1, dtype=int)
        self.fleet = np.zeros((5, 9), dtype=np.float32)
        self.fleet_timestamp = -1
        self.last_capture = -1
        self.latest_delivery_source = -1

    def deliver(self, now):
        while self.queue and self.queue[0]["arrival_t"] <= now:
            packet = self.queue.popleft()
            sampled = packet["sampled"]
            self.counts[sampled] = packet["counts"][sampled]
            self.coverage[sampled] = packet["coverage"][sampled]
            self.timestamps[sampled] = packet["source_t"]
            self.fleet[:] = packet["fleet"]
            self.fleet_timestamp = packet["service_t"]
            self.latest_delivery_source = packet["source_t"]

    def vector(self, now):
        max_age = self.config.report_max_age_slots
        age = now - self.timestamps
        valid = (self.timestamps >= 0) & (age <= max_age)
        density = np.where(valid, np.clip(self.counts / 25, 0, 4), 0)
        visibility = np.where(valid, self.coverage, 0)
        age_map = np.where(valid, np.clip(age / max_age, 0, 1), 1)
        result = [*density.ravel(), *visibility.ravel(), *age_map.ravel()]
        fleet_age = now - self.fleet_timestamp
        fleet_valid = self.latest_delivery_source >= 0 and fleet_age <= max_age
        for row in self.fleet:
            result.extend(row.tolist() if fleet_valid else [0.0] * 9)
            result.extend(
                [min(max(fleet_age, 0) / max_age, 1) if fleet_valid else 1.0, float(fleet_valid)]
            )
        return np.asarray(result, dtype=np.float32)

    def capture(self, env, net):
        cfg, t = (self.config, int(env.t))
        if t % cfg.report_period_slots or self.last_capture == t:
            return
        self.last_capture = t
        roles = env.logical_roles()
        active = np.flatnonzero((env.status == 0) & np.isin(roles, [0, 2]))
        probes = np.asarray([[r, c] for r in range(20) for c in range(20)], dtype=float)
        seen = np.zeros(len(env.ue_positions), dtype=bool)
        scanned = np.zeros(400, dtype=bool)
        for aid in active:
            seen |= (
                np.linalg.norm(env.ue_positions - env.uav_positions[aid], axis=1)
                * env.cfg.cell_size_m
                <= cfg.beacon_radius_m + 1e-09
            )
            scanned |= (
                np.linalg.norm(probes - env.uav_positions[aid], axis=1) * env.cfg.cell_size_m
                <= cfg.beacon_radius_m + 1e-09
            )
        counts = np.zeros((5, 5), dtype=np.float32)
        for position in env.ue_positions[seen]:
            cell = np.clip(np.floor(position / 4).astype(int), 0, 4)
            counts[tuple(cell)] += 1
        coverage = scanned.reshape(5, 4, 5, 4).mean(axis=(1, 3)).astype(np.float32)
        sampled = (coverage > 0) | (counts > 0)
        features = []
        for aid in range(5):
            load = 0 if net is None else net.get("access_offer_mbps", {}).get(aid, 0)
            delivered = (
                0
                if net is None
                else sum((float(net["ue_rate_mbps"][u]) for u in net["association"].get(aid, [])))
            )
            if net is not None and roles[aid] in (1, 2):
                forwarded = net["forwarded_mbps_by_uav"].get(aid, 0)
                if roles[aid] == 1:
                    load = delivered = forwarded
                else:
                    load += forwarded
                    delivered += forwarded
            deficit = 0 if load <= 1e-12 else max(0, 1 - delivered / load)
            features.append(own_features(env, aid) + [min(load / 100, 1), deficit])
        self.queue.append(
            dict(
                source_t=t,
                arrival_t=t + cfg.report_delay_slots,
                counts=counts,
                coverage=coverage,
                sampled=sampled,
                fleet=np.asarray(features, dtype=np.float32),
                service_t=max(-1, t - 1),
            )
        )


class DelayedHistoryMap:
    """Receivable UE beacons may report their measured historical rate.

    Only UE IDs detected within an active Access beacon footprint contribute.
    Captures and broadcasts follow the existing period/delay/expiry contract.
    Unknown cells are distinguished using the existing count/visibility/age maps.
    """

    def __init__(self, config):
        self.config = config
        self.queue = deque()
        self.values = np.zeros((5, 5), np.float32)
        self.timestamps = np.full((5, 5), -1, int)
        self.last_capture = -1
        self.latest_source = -1

    def capture(self, env):
        cfg, now = (self.config, int(env.t))
        if now % cfg.report_period_slots or now == self.last_capture:
            return
        self.last_capture = now
        active = np.flatnonzero((env.status == 0) & np.isin(env.logical_roles(), [0, 2]))
        probes = np.asarray([[r, c] for r in range(20) for c in range(20)], float)
        seen = np.zeros(90, bool)
        scanned = np.zeros(400, bool)
        for aid in active:
            seen |= (
                np.linalg.norm(env.ue_positions - env.uav_positions[aid], axis=1)
                * env.cfg.cell_size_m
                <= cfg.beacon_radius_m + 1e-09
            )
            scanned |= (
                np.linalg.norm(probes - env.uav_positions[aid], axis=1) * env.cfg.cell_size_m
                <= cfg.beacon_radius_m + 1e-09
            )
        count, total = (np.zeros((5, 5)), np.zeros((5, 5)))
        for uid in np.flatnonzero(seen):
            cell = tuple(np.clip(np.floor(env.ue_positions[uid] / 4).astype(int), 0, 4))
            count[cell] += 1
            total[cell] += env.ema_rates[uid] / env.cfg.ue_demand_mbps
        sampled = scanned.reshape(5, 4, 5, 4).mean(axis=(1, 3)) > 0
        sampled |= count > 0
        values = np.divide(total, count, out=np.zeros_like(total), where=count > 0).astype(
            np.float32
        )
        self.queue.append(
            dict(source=now, arrival=now + cfg.report_delay_slots, sampled=sampled, values=values)
        )

    def deliver(self, now):
        while self.queue and self.queue[0]["arrival"] <= now:
            packet = self.queue.popleft()
            mask = packet["sampled"]
            self.values[mask] = packet["values"][mask]
            self.timestamps[mask] = packet["source"]
            self.latest_source = packet["source"]

    def vector(self, now):
        valid = (self.timestamps >= 0) & (now - self.timestamps <= self.config.report_max_age_slots)
        return np.where(valid, self.values, 0).astype(np.float32).ravel()


class ObservationMixin:
    def _base_obs(self):
        shared = (
            np.zeros(130, dtype=np.float32) if self.reports is None else self.reports.vector(self.t)
        )
        return np.asarray(
            [
                np.concatenate([shared, own_features(self, a), np.eye(6)[5], np.eye(5)[a]])
                for a in range(5)
            ],
            dtype=np.float32,
        )

    def _base_state(self):
        if not hasattr(self, "ue_positions"):
            return np.zeros(66, dtype=np.float32)
        density = np.zeros((5, 5), dtype=np.float32)
        for position in self.ue_positions:
            cell = tuple(np.clip(np.floor(position / 4).astype(int), 0, 4))
            density[cell] += 1 / 25
        return np.asarray(
            [
                *density.ravel(),
                *(x for a in range(5) for x in own_features(self, a)),
                *np.eye(6, dtype=np.float32)[5],
            ],
            dtype=np.float32,
        )

    def get_obs(self):
        original = self._base_obs()
        if not self._temporal_ready:
            return original
        history = self.history_reports.vector(self.t)
        return np.concatenate([original, np.tile(history, (5, 1))], axis=1).astype(np.float32)

    def get_state(self):
        original = self._base_state()
        if not self._temporal_ready:
            return original
        return np.r_[original, self.ema_rates / 4].astype(np.float32)
