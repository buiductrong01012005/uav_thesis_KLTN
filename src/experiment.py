"""Construct the same physical environment for train, checkpoint eval and baselines."""

from copy import deepcopy
from src.configuration import environment_spec
from src.envs.config import SimulationConfig, ServiceV2Config
from src.envs.curriculum import FULL_CONDITION
from src.envs.energy import EnergyConfig
from src.envs.environment import BS5FourArm400kEnv, CollisionConfig
from src.envs.reward import TCFWeights, FairConfig


def environment_protocol(env):
    """The flat metric/protocol fields retained by existing report writers."""
    weights = env["reward"]["weights"]
    return dict(
        service=deepcopy(env["service"]),
        reward_weights=deepcopy(weights),
        coverage_floor_mbps=env["reward"]["coverage_floor_mbps"],
        energy_config=deepcopy(env["energy"]),
        collision_config=deepcopy(env["collision"]),
        fair_config=dict(
            env["fairness"],
            throughput_weight=weights["throughput"],
            fairness_weight=weights["fairness"],
        ),
    )


def make_environment(spec=None, *, arm="C-N", condition=None, horizon=None, residual_battery=None):
    spec = deepcopy(spec if spec is not None else environment_spec())
    p = environment_protocol(spec)
    return BS5FourArm400kEnv(
        condition or FULL_CONDITION,
        arm=arm,
        simulation_config=SimulationConfig(**spec["simulation"]),
        service_config=ServiceV2Config(**p["service"]),
        reward_weights=TCFWeights(**p["reward_weights"]),
        coverage_floor_mbps=p["coverage_floor_mbps"],
        energy_config=EnergyConfig(**p["energy_config"]),
        collision_config=CollisionConfig(**p["collision_config"]),
        fair_config=FairConfig(**p["fair_config"]),
        horizon=horizon,
        residual_battery=residual_battery,
        fixture=spec["fixture"],
    )
