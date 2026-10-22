"""Curriculum profiles for the V5 dynamic role + mobility problem.

The final task is unchanged.  Curriculum changes only the amount of temporal
complexity exposed during training:

Stage 1: mobile traffic + dynamic Access/Relay, battery disabled.
Stage 2: same task with battery lifecycle enabled on a medium horizon.
Stage 3: full long-horizon mobile + battery task.

All stages use the same fixed BS topology, V5 action semantics, request logic,
network model, observation/state layout, and mean-user-satisfaction reward.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.envs.v5_dynamic_tradeoff_env import (
    V5DynamicTradeoffConfig,
    V5DynamicTradeoffEnv,
)


@dataclass(frozen=True)
class V5CurriculumStage:
    index: int
    name: str
    horizon: int
    battery_cycle: bool
    initial_battery_min: float
    initial_battery_max: float
    randomize_train_phase: bool
    train_transitions: int
    eval_every: int
    eps_start: float
    eps_end: float
    eps_decay_steps: int


DEFAULT_CURRICULUM: tuple[V5CurriculumStage, ...] = (
    V5CurriculumStage(
        index=1,
        name="mobility_role_no_battery",
        horizon=200,
        battery_cycle=False,
        initial_battery_min=100.0,
        initial_battery_max=100.0,
        # Start from the direct-friendly end and traverse toward the
        # relay-stressed end.  There is no time feature in the policy state, so
        # the actor still has to infer the regime from traffic/network state.
        randomize_train_phase=False,
        train_transitions=30_000,
        eval_every=5_000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=25_000,
    ),
    V5CurriculumStage(
        index=2,
        name="battery_intro",
        horizon=300,
        battery_cycle=True,
        initial_battery_min=70.0,
        initial_battery_max=100.0,
        randomize_train_phase=True,
        train_transitions=30_000,
        eval_every=5_000,
        eps_start=0.40,
        eps_end=0.05,
        eps_decay_steps=25_000,
    ),
    V5CurriculumStage(
        index=3,
        name="full_long_horizon",
        horizon=600,
        battery_cycle=True,
        initial_battery_min=55.0,
        initial_battery_max=100.0,
        randomize_train_phase=True,
        train_transitions=60_000,
        eval_every=10_000,
        eps_start=0.25,
        eps_end=0.05,
        eps_decay_steps=50_000,
    ),
)


def make_curriculum_env(
    stage: V5CurriculumStage,
    seed: int,
    *,
    training: bool,
) -> V5DynamicTradeoffEnv:
    """Build one stage while preserving the final V5 formulation."""
    return V5DynamicTradeoffEnv(
        V5DynamicTradeoffConfig(
            seed=int(seed),
            horizon=int(stage.horizon),
            enable_ue_mobility=True,
            enable_battery_cycle=bool(stage.battery_cycle),
            backhaul_radius_cells=4.0,
            initial_battery_min=float(stage.initial_battery_min),
            initial_battery_max=float(stage.initial_battery_max),
            randomize_start_identity=True,
            # Validation is phase-locked for repeatable cross-checking.  Stage
            # 2/3 training randomizes phase so the learned policy cannot key on
            # a fixed episode clock.
            randomize_tradeoff_phase=(bool(stage.randomize_train_phase) if training else False),
            tradeoff_segment_slots=160,
            relay_request_on_ratio=0.75,
            relay_request_off_ratio=0.90,
            relay_min_dwell_slots=2,
        )
    )


def curriculum_stage_dict(stage: V5CurriculumStage) -> dict:
    return {
        "index": int(stage.index),
        "name": stage.name,
        "horizon": int(stage.horizon),
        "battery_cycle": bool(stage.battery_cycle),
        "initial_battery_min": float(stage.initial_battery_min),
        "initial_battery_max": float(stage.initial_battery_max),
        "randomize_train_phase": bool(stage.randomize_train_phase),
        "train_transitions": int(stage.train_transitions),
        "eval_every": int(stage.eval_every),
        "eps_start": float(stage.eps_start),
        "eps_end": float(stage.eps_end),
        "eps_decay_steps": int(stage.eps_decay_steps),
    }
