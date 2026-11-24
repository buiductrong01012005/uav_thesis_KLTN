"""Load functional JSON configs and resolve a run before any simulation starts."""

from copy import deepcopy
from dataclasses import field
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def environment_spec():
    return read_json(ROOT / "configs/environment.json")


def architecture_spec():
    return read_json(ROOT / "configs/training.json")["architecture"]


def environment_defaults(section):
    """Give typed dataclasses their defaults from the single environment declaration."""

    def decorate(cls):
        env = environment_spec()
        if section == "fairness":
            defaults = dict(
                env[section],
                throughput_weight=env["reward"]["weights"]["throughput"],
                fairness_weight=env["reward"]["weights"]["fairness"],
            )
        else:
            defaults = env[section]
        for name in cls.__annotations__:
            value = defaults[name]
            if isinstance(value, (list, dict)):
                setattr(cls, name, field(default_factory=lambda value=value: deepcopy(value)))
            else:
                setattr(cls, name, value)
        return cls

    return decorate


def merge(base, changes):
    result = deepcopy(base)
    for key, value in changes.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) else deepcopy(value)
    return result


def set_value(config, assignment):
    """A dotted CLI override must name an existing parameter; values use JSON syntax."""
    path, separator, raw = assignment.partition("=")
    if not separator:
        raise ValueError("Use --set section.parameter=JSON_VALUE")
    target = config
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    if parts[-1] not in target:
        raise ValueError("Unknown config parameter: " + path)
    target[parts[-1]] = json.loads(raw)


def resolve_training(path, overrides=None, assignments=()):
    path = Path(path).resolve()
    training = read_json(path)
    env_path = (path.parent / training.pop("environment")).resolve()
    eval_path = (path.parent / training.pop("evaluation")).resolve()
    config = dict(
        environment=read_json(env_path), training=training, evaluation=read_json(eval_path)
    )
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    for assignment in assignments:
        if assignment.startswith("training.diagnostic="):
            raise ValueError("Use --diagnostic/--no-diagnostic to select a profile")
    diagnostic = overrides.get("diagnostic", training["diagnostic"])
    profile = training.pop("diagnostic_profile")
    if diagnostic:
        config = merge(config, profile)
    for assignment in assignments:
        set_value(config, assignment)
    config["training"].update(overrides)
    t = config["training"]
    t["diagnostic"] = diagnostic
    t.setdefault("training_seed", t["training_seeds"][0])
    t["stage_ends"] = [*t["stage_ends"], t["total_steps"]]
    if t["total_steps"] <= 0 or sorted(set(t["stage_ends"])) != t["stage_ends"]:
        raise ValueError(
            "Budget must exceed the three curriculum boundaries; use --diagnostic for a short run"
        )
    t["evaluation_steps"] = sorted(
        {
            0,
            t["total_steps"],
            *range(t["stage_ends"][2], t["total_steps"] + 1, t["evaluation_every"]),
        }
    )
    # Validate the complete episode schedule before creating output/checkpoints.
    from src.envs.curriculum import condition_for

    step = 0
    horizon = config["environment"]["simulation"]["horizon"]
    while step < t["total_steps"]:
        condition = condition_for(step, t["arm"], t)
        length = (
            horizon
            if condition["stage"] == "S4"
            else (
                t["battery_horizon"]
                if condition["substage"] == "S3d" and not diagnostic
                else t["early_horizon"]
            )
        )
        if length < 1 or step + length > t["total_steps"]:
            raise ValueError(
                "Budget must contain whole episodes for the configured curriculum/horizons"
            )
        step += length
    return config


def experiment_plan():
    """Full W00 replication is only for bundle benchmark/catalog commands."""
    t = read_json(ROOT / "configs/training.json")
    e = read_json(ROOT / "configs/evaluation.json")
    from src.envs.config import ARM_ACTIONS

    return dict(
        e["replication"],
        training_seeds=t["training_seeds"],
        additional_training_seeds=t["training_seeds"][1:],
        arms=list(ARM_ACTIONS),
        heuristics=[label for label in e["labels"] if label != "RANDOM"],
        total_training_steps=t["total_steps"],
        validation_seeds=t["validation_seeds"],
        test_seeds=e["test_seeds"],
        test_horizon=e["horizon"],
    )
