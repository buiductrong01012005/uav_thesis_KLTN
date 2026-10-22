"""V5 relay-request environment for hierarchical-but-decentralized role learning.

V5 keeps the V4 radio model, battery lifecycle, reward, one-hop routing and
physical movement.  The control redesign only makes the Relay decision more
semantically grounded:

* Access UAVs automatically emit ``relay_needed`` when their direct backhaul
  is persistently weak relative to the traffic they can collect.
* An Access policy may accept one open request and becomes a persistent
  ``Relay(target=j)``.  The policy still controls every movement step.
* A Relay policy may explicitly release its assignment and return to Access.
* The Relay target is a primary navigation/control target only.  The existing
  deterministic routing layer may still route other Access UAVs through that
  Relay when geometry/capacity make it useful.

Per-agent policy actions (10 total):
    0..4 : North/South/East/West/Stay, preserving the current role
    5..8 : AcceptRelay(slot), where the four slots map to the four *other* UAV
           IDs in ascending order for the acting agent
    9    : ReleaseRelay (Relay -> Access, no movement)

The relative target-slot mapping avoids an otherwise redundant self-target
entry while keeping the action count equal to the original 10-action V4 core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np

from src.envs.v0_env import ACCESS, RELAY, MOVE_DELTAS
from src.envs.v4_fixed_mobile_battery_env import ACTIVE
from src.envs.v4_final_fixed_map_env import V4FinalFixedMapConfig, V4FinalFixedMapEnv

MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_EAST = 2
MOVE_WEST = 3
STAY = 4
ACCEPT_BASE = 5
N_ACCEPT_SLOTS = 4
RELEASE_RELAY = 9


@dataclass
class V5RelayRequestConfig(V4FinalFixedMapConfig):
    """Control parameters for the relay-request abstraction."""

    # Direct-backhaul / offered-load thresholds.  Hysteresis prevents a request
    # from flickering around one threshold.
    relay_request_on_ratio: float = 0.75
    relay_request_off_ratio: float = 0.90
    relay_request_min_load_mbps: float = 1.0

    # A newly accepted Relay should be allowed to move for a couple of slots
    # before ReleaseRelay becomes valid.  This is control hysteresis, not reward.
    relay_min_dwell_slots: int = 2


class V5RelayRequestEnv(V4FinalFixedMapEnv):
    """V4 network with automatic Access requests and persistent Relay targets."""

    n_actions = 10
    # x,y,relay,battery,active,returning,charging,
    # relay_needed,request_open,load_norm,backhaul_deficit,target_onehot(5)
    UAV_FEAT = 16

    def __init__(self, config: V5RelayRequestConfig | None = None):
        super().__init__(config or V5RelayRequestConfig())
        self.cfg: V5RelayRequestConfig
        if self.cfg.num_uavs != 5:
            raise ValueError("V5 relay-request action mapping is defined for 5 UAVs")
        if not 0.0 < self.cfg.relay_request_on_ratio < self.cfg.relay_request_off_ratio:
            raise ValueError("Require 0 < request_on_ratio < request_off_ratio")
        if self.cfg.relay_min_dwell_slots < 0:
            raise ValueError("relay_min_dwell_slots must be non-negative")

        n = self.cfg.num_uavs
        self.relay_needed = np.zeros(n, dtype=bool)
        self.request_open = np.zeros(n, dtype=bool)
        self.request_load_mbps = np.zeros(n, dtype=np.float64)
        self.request_direct_bh_mbps = np.zeros(n, dtype=np.float64)
        self.request_deficit = np.zeros(n, dtype=np.float64)
        self.relay_target = np.full(n, -1, dtype=np.int64)
        self.relay_assignment_age = np.zeros(n, dtype=np.int64)

    # -------------------------- request/assignment logic ---------------------

    def target_for_accept_action(self, agent_id: int, action: int) -> int:
        """Map an agent-relative AcceptRelay slot to an absolute UAV id."""
        a = int(action)
        if not ACCEPT_BASE <= a < ACCEPT_BASE + N_ACCEPT_SLOTS:
            raise ValueError("action is not an AcceptRelay action")
        others = [j for j in range(self.cfg.num_uavs) if j != int(agent_id)]
        return int(others[a - ACCEPT_BASE])

    def accept_action_for_target(self, agent_id: int, target_id: int) -> int:
        """Inverse mapping used by tests/diagnostics."""
        others = [j for j in range(self.cfg.num_uavs) if j != int(agent_id)]
        if int(target_id) not in others:
            raise ValueError("target must be a different UAV")
        return ACCEPT_BASE + others.index(int(target_id))

    def _compute_request_metrics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return per-UAV (Access offer, direct BH capacity, deficit).

        The association is exactly the one used by the current deterministic
        network layer.  ``offer`` is the access-side traffic that could be
        delivered absent a backhaul bottleneck.  ``direct`` intentionally
        ignores Relays: it asks whether this Access intrinsically needs relay
        assistance in the current geometry/load.
        """
        n = self.cfg.num_uavs
        offer = np.zeros(n, dtype=np.float64)
        direct = np.zeros(n, dtype=np.float64)
        deficit = np.zeros(n, dtype=np.float64)

        # Association is independent of which backhaul route is selected, so
        # reuse the canonical network computation rather than inventing a new
        # UE scheduler for the request signal.
        net = V4FinalFixedMapEnv.compute_network(self)
        assoc = net.get("association", {})

        for aid_raw, users in assoc.items():
            aid = int(aid_raw)
            if self.status[aid] != ACTIVE or self.roles[aid] != ACCESS:
                continue

            # Access-side offered traffic under the same equal sharing model.
            users = list(users)
            if users:
                n_u = len(users)
                bw = self.cfg.access_bw_hz / n_u
                tx_dbm_per_ue = self.cfg.access_tx_dbm - 10.0 * math.log10(n_u)
                axy = self._cell_to_xy(self.uav_positions[aid])
                total = 0.0
                for u in users:
                    uxy = self._cell_to_xy(self.ue_positions[int(u)])
                    pl = self._a2g_path_loss_db(
                        uxy,
                        self.cfg.ue_height_m,
                        axy,
                        self.cfg.uav_altitude_m,
                        self.cfg.access_fc_hz,
                    )
                    r = self._shannon_rate_mbps(tx_dbm_per_ue - pl, bw)
                    total += min(r, self.cfg.ue_demand_mbps)
                offer[aid] = float(total)

            # Best direct Access -> surviving-BS link under the locked R4/R5
            # feasibility model.
            axy = self._cell_to_xy(self.uav_positions[aid])
            caps = []
            for bsxy in self.surviving_bs_xy:
                if self._xy_within_radius_m(axy, bsxy, self.cfg.backhaul_radius_cells):
                    caps.append(
                        self._bh_a2g_rate(
                            axy,
                            self.cfg.uav_altitude_m,
                            bsxy,
                            self.cfg.bs_height_m,
                        )
                    )
            direct[aid] = max(caps) if caps else 0.0
            if offer[aid] > 1e-9:
                deficit[aid] = float(np.clip(1.0 - direct[aid] / offer[aid], 0.0, 1.0))

        return offer, direct, deficit

    def _active_primary_relay_for(self, target: int) -> int | None:
        for rid in range(self.cfg.num_uavs):
            if (
                self.status[rid] == ACTIVE
                and self.roles[rid] == RELAY
                and int(self.relay_target[rid]) == int(target)
            ):
                return int(rid)
        return None

    def _refresh_request_state(self) -> None:
        offer, direct, deficit = self._compute_request_metrics()
        prev = self.relay_needed.copy()
        needed = np.zeros(self.cfg.num_uavs, dtype=bool)

        for i in range(self.cfg.num_uavs):
            if self.status[i] != ACTIVE or self.roles[i] != ACCESS:
                continue
            load = float(offer[i])
            if load < self.cfg.relay_request_min_load_mbps:
                continue
            ratio = float(direct[i] / max(load, 1e-9))
            threshold = (
                self.cfg.relay_request_off_ratio if prev[i] else self.cfg.relay_request_on_ratio
            )
            needed[i] = ratio < threshold

        self.relay_needed[:] = needed
        self.request_load_mbps[:] = offer
        self.request_direct_bh_mbps[:] = direct
        self.request_deficit[:] = deficit

        # A request is open to a *new* Relay only while it has no active primary
        # assignment.  The raw relay_needed flag remains visible to the current
        # Relay, so successful service does not make it immediately release and
        # create request/release oscillation.
        for i in range(self.cfg.num_uavs):
            self.request_open[i] = bool(
                self.relay_needed[i] and self._active_primary_relay_for(i) is None
            )

    def _cleanup_assignments_after_step(self) -> None:
        """Enforce the V5 persistent-assignment invariant.

        An *active* Relay must always have exactly one valid active Access
        primary target.  Non-active UAVs keep no assignment, and Access UAVs
        may not retain stale targets.  The parent battery lifecycle already
        restores a charging UAV as Access on rejoin; this check makes that
        invariant explicit and robust to any future lifecycle/control change.
        """
        n = self.cfg.num_uavs
        for rid in range(n):
            target = int(self.relay_target[rid])

            if self.status[rid] != ACTIVE:
                self.relay_target[rid] = -1
                self.relay_assignment_age[rid] = 0
                continue

            if self.roles[rid] != RELAY:
                self.relay_target[rid] = -1
                self.relay_assignment_age[rid] = 0
                continue

            valid_target = (
                0 <= target < n
                and target != rid
                and self.status[target] == ACTIVE
                and self.roles[target] == ACCESS
            )
            if not valid_target:
                # In V5, Relay is not a free-standing role: without a valid
                # primary Access request it deterministically falls back to
                # Access.
                self.relay_target[rid] = -1
                self.relay_assignment_age[rid] = 0
                self.roles[rid] = ACCESS

    def _initialize_primary_relay_target(self, relay_id: int) -> int | None:
        """Attach the legacy initial Relay to the strongest current request."""
        self._refresh_request_state()
        candidates = np.flatnonzero(self.relay_needed)
        candidates = [int(x) for x in candidates if int(x) != int(relay_id)]
        if not candidates:
            self.relay_target[int(relay_id)] = -1
            self.relay_assignment_age[int(relay_id)] = 0
            if self.status[int(relay_id)] == ACTIVE:
                self.roles[int(relay_id)] = ACCESS
            self._refresh_request_state()
            return None
        # Prefer a severe and heavily-loaded request, not simply the lowest id.
        max_load = self.cfg.max_ues_per_access * self.cfg.ue_demand_mbps
        scores = {
            t: float(self.request_deficit[t])
            * min(1.0, float(self.request_load_mbps[t]) / max_load)
            for t in candidates
        }
        target = max(candidates, key=lambda t: (scores[t], -t))
        self.relay_target[int(relay_id)] = int(target)
        self.relay_assignment_age[int(relay_id)] = self.cfg.relay_min_dwell_slots
        self._refresh_request_state()
        return int(target)

    # ------------------------------ RL interface -----------------------------

    def reset(self, seed: int | None = None):
        self.relay_needed[:] = False
        self.request_open[:] = False
        self.request_load_mbps[:] = 0.0
        self.request_direct_bh_mbps[:] = 0.0
        self.request_deficit[:] = 0.0
        self.relay_target[:] = -1
        self.relay_assignment_age[:] = 0
        out = super().reset(seed=seed)
        self._refresh_request_state()
        return self._refresh_output(out)

    def step(self, actions: Sequence[int]):
        if len(actions) != self.cfg.num_uavs:
            raise ValueError(f"Expected {self.cfg.num_uavs} actions")

        # Build the mask once from the pre-action joint state, exactly as a
        # decentralized simultaneous policy sees it.  AcceptRelay intents are
        # then resolved jointly so two agents cannot both become the primary
        # Relay for the same request in one slot.
        mask = self.get_action_mask()
        raw_actions = [int(a) for a in actions]
        for i, a in enumerate(raw_actions):
            if not 0 <= a < self.n_actions:
                raise ValueError("policy action must be in [0, 9]")
            if self.status[i] == ACTIVE and not mask[i, a]:
                if ACCEPT_BASE <= a < ACCEPT_BASE + N_ACCEPT_SLOTS:
                    raise ValueError(f"invalid AcceptRelay action {a} for UAV {i}")
                if a == RELEASE_RELAY:
                    raise ValueError(f"invalid ReleaseRelay action for UAV {i}")

        # Collect simultaneous volunteer intents.  If several Access UAVs ask
        # to serve the same target, one deterministic winner is admitted and
        # all other volunteers remain Access/Stay for this slot.  Identity is
        # shuffled over start positions in the dynamic benchmark, so this
        # tie-break is not a fixed spatial shortcut.  We log every conflict so
        # a learned policy can only be called coordinated when conflicts are
        # rare.
        accept_intents: dict[int, list[int]] = {}
        for i, a in enumerate(raw_actions):
            if (
                self.status[i] == ACTIVE
                and self.roles[i] == ACCESS
                and ACCEPT_BASE <= a < ACCEPT_BASE + N_ACCEPT_SLOTS
                and mask[i, a]
            ):
                target = self.target_for_accept_action(i, a)
                accept_intents.setdefault(int(target), []).append(int(i))

        # Resolve all AcceptRelay proposals as a deterministic directed
        # matching.  A UAV may participate in at most one accepted pair: it
        # cannot become Relay for one request while simultaneously remaining
        # the Access target of another Relay.  Sorting by actor id is only a
        # contention tie-break; start-position identities are shuffled in the
        # dynamic benchmark, and every rejected proposal is logged.
        proposals = sorted(
            (int(actor), int(target))
            for target, claimants in accept_intents.items()
            for actor in claimants
        )
        accepted_actor_to_target: dict[int, int] = {}
        used_actors: set[int] = set()
        used_targets: set[int] = set()
        rejected_claims: list[tuple[int, int]] = []
        for actor, target in proposals:
            structural_conflict = (
                actor in used_targets or target in used_targets or target in used_actors
            )
            if structural_conflict:
                rejected_claims.append((actor, target))
                continue
            accepted_actor_to_target[actor] = target
            used_actors.add(actor)
            used_targets.add(target)
        claim_conflicts = len(rejected_claims)

        translated: list[int] = []
        accepted: list[tuple[int, int]] = []
        released: list[int] = []

        for i, a in enumerate(raw_actions):
            if self.status[i] != ACTIVE:
                translated.append(self.encode_action(STAY, ACCESS))
                continue

            current_role = int(self.roles[i])
            if 0 <= a <= STAY:
                # Physical movement always preserves the persistent role.
                translated.append(self.encode_action(a, current_role))
            elif ACCEPT_BASE <= a < ACCEPT_BASE + N_ACCEPT_SLOTS:
                target = self.target_for_accept_action(i, a)
                if accepted_actor_to_target.get(int(i)) == int(target):
                    self.relay_target[i] = int(target)
                    self.relay_assignment_age[i] = 0
                    translated.append(self.encode_action(STAY, RELAY))
                    accepted.append((int(i), int(target)))
                else:
                    # Lost/invalidated simultaneous claim: no movement and no
                    # role change.  This preserves exactly one primary Relay
                    # per request without silently turning extra Access UAVs
                    # into Relays.
                    translated.append(self.encode_action(STAY, ACCESS))
            elif a == RELEASE_RELAY:
                self.relay_target[i] = -1
                self.relay_assignment_age[i] = 0
                translated.append(self.encode_action(STAY, ACCESS))
                released.append(int(i))
            else:  # pragma: no cover - protected by range checks above
                raise AssertionError("unreachable action branch")

        out = V4FinalFixedMapEnv.step(self, translated)

        # Age only persistent, active Relay assignments.  Clean up Return/Rejoin
        # and target-role changes before exposing the next action mask.
        for rid in range(self.cfg.num_uavs):
            if (
                self.status[rid] == ACTIVE
                and self.roles[rid] == RELAY
                and self.relay_target[rid] >= 0
            ):
                self.relay_assignment_age[rid] += 1
        self._cleanup_assignments_after_step()
        self._refresh_request_state()

        out = self._refresh_output(out)
        out["info"].update(
            {
                "relay_needed": self.relay_needed.astype(int).tolist(),
                "request_open": self.request_open.astype(int).tolist(),
                "request_load_mbps": self.request_load_mbps.astype(float).tolist(),
                "request_direct_bh_mbps": self.request_direct_bh_mbps.astype(float).tolist(),
                "request_deficit": self.request_deficit.astype(float).tolist(),
                "relay_target": self.relay_target.astype(int).tolist(),
                "relay_assignment_age": self.relay_assignment_age.astype(int).tolist(),
                "accepted_relay_targets_this_slot": [[a, t] for a, t in accepted],
                "rejected_relay_claims_this_slot": [[a, t] for a, t in rejected_claims],
                "relay_claim_conflicts_this_slot": int(claim_conflicts),
                "released_relay_ids_this_slot": released,
            }
        )
        return out

    def get_action_mask(self) -> np.ndarray:
        masks = np.zeros((self.cfg.num_uavs, self.n_actions), dtype=np.int8)
        for i in range(self.cfg.num_uavs):
            if self.status[i] != ACTIVE:
                masks[i, STAY] = 1
                continue

            # Movement/Stay actions preserve the current role.
            for move_idx in range(5):
                cand = self.uav_positions[i] + MOVE_DELTAS[move_idx]
                if self._inside_grid(cand):
                    masks[i, move_idx] = 1

            if self.roles[i] == ACCESS:
                # Only unclaimed active Access requests are valid targets.
                for a in range(ACCEPT_BASE, ACCEPT_BASE + N_ACCEPT_SLOTS):
                    target = self.target_for_accept_action(i, a)
                    if self.request_open[target]:
                        masks[i, a] = 1
            else:
                # A Relay keeps moving with actions 0..4.  Release is explicit
                # and delayed very slightly to avoid one-slot role chattering.
                if self.relay_assignment_age[i] >= self.cfg.relay_min_dwell_slots:
                    masks[i, RELEASE_RELAY] = 1
        return masks

    # -------------------------- compact V5 observation -----------------------

    def get_state(self) -> np.ndarray:
        gden = max(self.cfg.grid_size - 1, 1)
        feats: list[float] = self._coarse_demand_map().ravel().astype(np.float32).tolist()
        max_load = max(1.0, self.cfg.max_ues_per_access * self.cfg.ue_demand_mbps)

        for i in range(self.cfg.num_uavs):
            feats.extend((self.uav_positions[i] / gden).astype(np.float32).tolist())
            feats.append(float(self.status[i] == ACTIVE and self.roles[i] == RELAY))
            feats.append(float(self.battery[i] / self.cfg.max_battery))
            feats.extend(
                [
                    float(self.status[i] == ACTIVE),
                    float(self.status[i] == 1),  # RETURNING
                    float(self.status[i] == 2),  # CHARGING
                ]
            )
            feats.append(float(self.relay_needed[i]))
            feats.append(float(self.request_open[i]))
            feats.append(float(np.clip(self.request_load_mbps[i] / max_load, 0.0, 1.0)))
            feats.append(float(np.clip(self.request_deficit[i], 0.0, 1.0)))
            target_oh = np.zeros(self.cfg.num_uavs, dtype=np.float32)
            if self.status[i] == ACTIVE and self.roles[i] == RELAY and self.relay_target[i] >= 0:
                target_oh[int(self.relay_target[i])] = 1.0
            feats.extend(target_oh.tolist())

        return np.asarray(feats, dtype=np.float32)

    def get_obs(self) -> np.ndarray:
        state = self.get_state()
        rows = []
        for i in range(self.cfg.num_uavs):
            aid = np.zeros(self.cfg.num_uavs, dtype=np.float32)
            aid[i] = 1.0
            rows.append(np.concatenate([state, aid]))
        return np.stack(rows, axis=0)

    def _refresh_output(self, out: dict) -> dict:
        out["obs"] = self.get_obs()
        out["state"] = self.get_state()
        out["action_mask"] = self.get_action_mask()
        if "episode_meta" in out:
            out["episode_meta"] = self._episode_meta()
        return out

    @property
    def state_dim(self) -> int:
        return int(self.get_state().shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.get_obs().shape[1])

    def _episode_meta(self) -> dict:
        meta = super()._episode_meta()
        meta.update(
            {
                "control_version": "V5_relay_request_persistent_assignment",
                "n_actions": int(self.n_actions),
                "action_design": (
                    "0-4 movement preserve role; 5-8 AcceptRelay(other UAV target); 9 ReleaseRelay"
                ),
                "relay_request_on_ratio": float(self.cfg.relay_request_on_ratio),
                "relay_request_off_ratio": float(self.cfg.relay_request_off_ratio),
                "relay_request_min_load_mbps": float(self.cfg.relay_request_min_load_mbps),
                "relay_min_dwell_slots": int(self.cfg.relay_min_dwell_slots),
                "relay_claim_conflict_rule": "deterministic joint matching of simultaneous AcceptRelay intents; no duplicate primary target and no Relay-to-Relay target chain",
                "reward": "mean user satisfaction only; unchanged",
            }
        )
        return meta


@dataclass
class V5RelayRequestA1Config(V5RelayRequestConfig):
    """Static-UE + battery stress stage for the first V5 learnability test."""

    enable_ue_mobility: bool = False
    enable_battery_cycle: bool = True
    backhaul_radius_cells: float = 4.0
    residual_initial_battery: float = 30.0
    randomize_residual_uav: bool = False
    relay_anchor_cell: tuple[int, int] = (11, 6)
    scenario_seed: int = 81


class V5RelayRequestA1Env(V5RelayRequestEnv):
    """Identity-randomized A1 setup with an initially assigned low-battery Relay."""

    def __init__(self, config: V5RelayRequestA1Config | None = None):
        cfg = config or V5RelayRequestA1Config()
        super().__init__(cfg)
        self.cfg: V5RelayRequestA1Config
        self._scenario_rng = np.random.default_rng(self.cfg.scenario_seed)
        self._base_start_cells = np.asarray(self.cfg.uav_start_cells, dtype=np.int64)
        self._relay_anchor = np.asarray(self.cfg.relay_anchor_cell, dtype=np.int64)
        if not self._inside_grid(self._relay_anchor):
            raise ValueError("relay_anchor_cell must be inside the grid")

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
        self.relay_target[:] = -1
        self.relay_assignment_age[:] = 0
        self.relay_needed[:] = False
        self.request_open[:] = False
        self._initialize_primary_relay_target(residual)

    def _finalize_scenario_reset(self, out: dict) -> dict:
        out = self._refresh_output(out)
        out["episode_meta"].update(
            {
                "residual_uav_idx": int(self.residual_uav_idx),
                "relay_anchor_cell": self._relay_anchor.astype(int).tolist(),
                "initial_relay_target": int(self.relay_target[self.residual_uav_idx]),
                "identity_position_randomization": True,
            }
        )
        return out

    def reset(self, seed: int | None = None):
        # Use the fixed UE-layout seed through the parent reset.  Scenario RNG is
        # separate so UAV identity-position randomization changes across episodes.
        out = V5RelayRequestEnv.reset(self, seed=seed)
        residual = int(self._scenario_rng.integers(0, self.cfg.num_uavs))
        perm = self._scenario_rng.permutation(self.cfg.num_uavs - 1)
        self._apply_scenario(residual, perm)
        return self._finalize_scenario_reset(out)

    def reset_with_scenario(
        self,
        residual_uav_idx: int,
        permutation_seed: int,
        *,
        layout_seed: int = 1101,
        residual_battery: float | None = None,
    ) -> dict:
        out = V5RelayRequestEnv.reset(self, seed=layout_seed)
        perm = np.random.default_rng(int(permutation_seed)).permutation(self.cfg.num_uavs - 1)
        self._apply_scenario(int(residual_uav_idx), perm)
        if residual_battery is not None:
            self.battery[int(residual_uav_idx)] = float(residual_battery)
        self._refresh_request_state()
        return self._finalize_scenario_reset(out)
