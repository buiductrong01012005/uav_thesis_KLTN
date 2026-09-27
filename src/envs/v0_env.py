"""
uav_recovery_v0.py

Minimal V0 environment for a multi-UAV network-recovery thesis.

V0 intentionally contains ONLY:
- 5x5 grid
- 3 homogeneous UAVs
- known UE map
- movement + {Access, Relay}
- direct backhaul or one-relay-hop backhaul
- team reward = served-user ratio

It intentionally excludes:
- Scout / unknown UE
- battery / charging
- power, bandwidth, SINR optimization
- continuous trajectories
- multi-hop relay routing

The goal of V0 is only to verify that a cooperative policy can learn
spatial Access/Relay coordination before more difficult features are added.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np


ACCESS = 0
RELAY = 1

ROLE_NAMES = {
    ACCESS: "Access",
    RELAY: "Relay",
}

MOVE_NAMES = ["North", "South", "East", "West", "Stay"]

MOVE_DELTAS = np.asarray(
    [
        [-1, 0],  # North
        [1, 0],   # South
        [0, 1],   # East
        [0, -1],  # West
        [0, 0],   # Stay
    ],
    dtype=np.int64,
)


@dataclass
class V0Config:
    grid_size: int = 5
    num_uavs: int = 3
    horizon: int = 50

    # Distances are Euclidean distances between grid-cell centers.
    access_range: float = 1.5
    backhaul_range: float = 2.5

    # Fixed V0 map.
    surviving_bs: tuple[int, int] = (0, 4)

    ue_positions: tuple[tuple[int, int], ...] = (
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
    )

    # Fixed initial deployment for the first overfitting/sanity experiment.
    uav_start_positions: tuple[tuple[int, int], ...] = (
        (2, 0),
        (2, 2),
        (1, 3),
    )


class UAVRecoveryEnvV0:
    """
    Lightweight environment independent of Gym/PettingZoo.

    Action encoding for each UAV:
        action = move_index * 2 + role

    where
        move_index in {0,1,2,3,4}
        role       in {ACCESS=0, RELAY=1}

    Therefore each UAV has 10 discrete actions.
    """

    def __init__(self, config: V0Config | None = None):
        self.cfg = config or V0Config()

        if self.cfg.num_uavs != 3:
            raise ValueError("The first V0 specification is intentionally fixed at 3 UAVs.")

        self.ue_positions = np.asarray(self.cfg.ue_positions, dtype=np.int64)
        self.bs_position = np.asarray(self.cfg.surviving_bs, dtype=np.int64)
        self.start_positions = np.asarray(
            self.cfg.uav_start_positions, dtype=np.int64
        )

        self.t = 0
        self.uav_positions = self.start_positions.copy()
        self.roles = np.full(self.cfg.num_uavs, ACCESS, dtype=np.int64)

    # ------------------------------------------------------------------
    # Core RL interface
    # ------------------------------------------------------------------

    def reset(self):
        self.t = 0
        self.uav_positions = self.start_positions.copy()
        self.roles = np.full(self.cfg.num_uavs, ACCESS, dtype=np.int64)

        return {
            "obs": self.get_obs(),
            "state": self.get_state(),
            "action_mask": self.get_action_mask(),
        }

    def step(self, actions: Sequence[int]):
        """
        All UAV actions are applied simultaneously.

        Sequence within one slot:
        1. Decode movement + new role.
        2. Apply valid movements.
        3. Apply role changes.
        4. Compute backhaul and served UEs.
        5. Compute common team reward.
        6. Advance time.
        """
        if len(actions) != self.cfg.num_uavs:
            raise ValueError(f"Expected {self.cfg.num_uavs} actions.")

        new_positions = self.uav_positions.copy()
        new_roles = self.roles.copy()

        for i, action in enumerate(actions):
            move_idx, role = self.decode_action(int(action))
            candidate = self.uav_positions[i] + MOVE_DELTAS[move_idx]

            # Invalid boundary movement becomes "stay".
            # The action mask should normally prevent this during learning.
            if self._inside_grid(candidate):
                new_positions[i] = candidate

            new_roles[i] = role

        self.uav_positions = new_positions
        self.roles = new_roles

        served_mask, valid_access, relay_chains = self.compute_served_users()

        served_count = int(served_mask.sum())
        served_ratio = served_count / len(self.ue_positions)
        reward = float(served_ratio)

        self.t += 1
        terminated = False
        truncated = self.t >= self.cfg.horizon

        info = {
            "t": self.t,
            "served_count": served_count,
            "served_ratio": served_ratio,
            "valid_access_count": int(valid_access.sum()),
            "relay_chains": relay_chains,
            "num_access": int((self.roles == ACCESS).sum()),
            "num_relay": int((self.roles == RELAY).sum()),
        }

        return {
            "obs": self.get_obs(),
            "state": self.get_state(),
            "action_mask": self.get_action_mask(),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }

    # ------------------------------------------------------------------
    # Communication/service model
    # ------------------------------------------------------------------

    def compute_backhaul(self):
        """
        An Access UAV has valid backhaul if either:

        Direct:
            Access -> surviving BS

        or one relay hop:
            Access -> Relay -> surviving BS

        Relay capacity is assumed unlimited in V0, so one relay may support
        multiple Access UAVs. This is a deliberate V0 simplification.
        """
        valid_access = np.zeros(self.cfg.num_uavs, dtype=bool)
        relay_chains: list[tuple[int, int | None]] = []

        access_ids = np.flatnonzero(self.roles == ACCESS)
        relay_ids = np.flatnonzero(self.roles == RELAY)

        for access_id in access_ids:
            access_pos = self.uav_positions[access_id]

            # Direct backhaul.
            if self._distance(access_pos, self.bs_position) <= self.cfg.backhaul_range:
                valid_access[access_id] = True
                relay_chains.append((int(access_id), None))
                continue

            # One-relay-hop backhaul.
            for relay_id in relay_ids:
                relay_pos = self.uav_positions[relay_id]

                access_to_relay = (
                    self._distance(access_pos, relay_pos)
                    <= self.cfg.backhaul_range
                )
                relay_to_bs = (
                    self._distance(relay_pos, self.bs_position)
                    <= self.cfg.backhaul_range
                )

                if access_to_relay and relay_to_bs:
                    valid_access[access_id] = True
                    relay_chains.append((int(access_id), int(relay_id)))
                    break

        return valid_access, relay_chains

    def compute_served_users(self):
        """
        UE u is served iff:
        - at least one Access UAV is within access_range, AND
        - that Access UAV has valid direct or one-hop relay backhaul.
        """
        valid_access, relay_chains = self.compute_backhaul()
        served = np.zeros(len(self.ue_positions), dtype=bool)

        valid_access_ids = np.flatnonzero(valid_access)

        for ue_id, ue_pos in enumerate(self.ue_positions):
            for access_id in valid_access_ids:
                if (
                    self._distance(ue_pos, self.uav_positions[access_id])
                    <= self.cfg.access_range
                ):
                    served[ue_id] = True
                    break

        return served, valid_access, relay_chains

    # ------------------------------------------------------------------
    # State / observation
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """
        Centralized V0 state.

        Contents:
        - each UAV: normalized (row, col) + role one-hot
        - normalized UE occupancy map
        - BS mask

        UE map is intentionally visible in V0.
        """
        features: list[float] = []
        denom = max(self.cfg.grid_size - 1, 1)

        for i in range(self.cfg.num_uavs):
            features.extend((self.uav_positions[i] / denom).tolist())

            if self.roles[i] == ACCESS:
                features.extend([1.0, 0.0])
            else:
                features.extend([0.0, 1.0])

        ue_map = np.zeros(
            (self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32
        )
        for pos in self.ue_positions:
            ue_map[tuple(pos)] += 1.0

        if ue_map.max() > 0:
            ue_map /= ue_map.max()

        bs_map = np.zeros_like(ue_map)
        bs_map[tuple(self.bs_position)] = 1.0

        features.extend(ue_map.ravel().tolist())
        features.extend(bs_map.ravel().tolist())

        return np.asarray(features, dtype=np.float32)

    def get_obs(self) -> np.ndarray:
        """
        In V0 every UAV sees the full relevant state.

        A one-hot agent ID is appended because a parameter-shared agent
        network otherwise receives identical observations for all UAVs and
        may produce identical actions purely due to symmetry.

        Shape:
            [num_uavs, state_dim + num_uavs]
        """
        state = self.get_state()
        obs = []

        for i in range(self.cfg.num_uavs):
            agent_id = np.zeros(self.cfg.num_uavs, dtype=np.float32)
            agent_id[i] = 1.0
            obs.append(np.concatenate([state, agent_id]))

        return np.stack(obs, axis=0)

    def get_action_mask(self) -> np.ndarray:
        """
        Masks actions whose movement would leave the grid.

        Shape:
            [num_uavs, 10]
        """
        masks = np.ones((self.cfg.num_uavs, 10), dtype=np.int8)

        for i, pos in enumerate(self.uav_positions):
            for action in range(10):
                move_idx, _ = self.decode_action(action)
                candidate = pos + MOVE_DELTAS[move_idx]

                if not self._inside_grid(candidate):
                    masks[i, action] = 0

        return masks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def decode_action(action: int) -> tuple[int, int]:
        if not 0 <= action < 10:
            raise ValueError("V0 action must be in [0, 9].")

        move_idx = action // 2
        role = action % 2
        return move_idx, role

    @staticmethod
    def encode_action(move_idx: int, role: int) -> int:
        if move_idx not in range(5):
            raise ValueError("move_idx must be in [0, 4].")
        if role not in (ACCESS, RELAY):
            raise ValueError("role must be ACCESS or RELAY.")
        return move_idx * 2 + role

    def _inside_grid(self, pos: np.ndarray) -> bool:
        return bool(
            np.all(pos >= 0)
            and np.all(pos < self.cfg.grid_size)
        )

    @staticmethod
    def _distance(a: Iterable[int], b: Iterable[int]) -> float:
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def run_manual_sanity_checks():
    """
    Hand-designed configuration used to verify that Relay is actually useful.

    Positions:
        Access candidates near the UE hotspot:
            UAV0 = (3,1)
            UAV1 = (4,2)

        Bridge candidate:
            UAV2 = (2,3)

        Surviving BS:
            (0,4)

    Expected ordering:
        A,A,R > A,R,R > A,A,A > R,R,R
    """
    env = UAVRecoveryEnvV0()
    env.reset()

    env.uav_positions = np.asarray(
        [(3, 1), (4, 2), (2, 3)], dtype=np.int64
    )

    cases = {
        "AAA": [ACCESS, ACCESS, ACCESS],
        "RRR": [RELAY, RELAY, RELAY],
        "AAR": [ACCESS, ACCESS, RELAY],
        "ARR": [ACCESS, RELAY, RELAY],
    }

    results = {}

    for name, roles in cases.items():
        env.roles = np.asarray(roles, dtype=np.int64)

        served, valid_access, chains = env.compute_served_users()

        results[name] = {
            "served": int(served.sum()),
            "total_ue": len(env.ue_positions),
            "served_ratio": float(served.mean()),
            "valid_access": valid_access.tolist(),
            "chains": chains,
        }

    return results


if __name__ == "__main__":
    results = run_manual_sanity_checks()

    for name, result in results.items():
        print(
            f"{name}: "
            f"{result['served']}/{result['total_ue']} served, "
            f"ratio={result['served_ratio']:.3f}, "
            f"chains={result['chains']}"
        )
