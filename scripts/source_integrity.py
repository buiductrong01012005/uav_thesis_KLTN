"""Exact source integrity for new runs; historical exceptions live in the old source ZIP."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_source(path, recorded_sha, root=ROOT):
    rel = Path(path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Invalid locked source path: " + str(path))
    current = (root / rel).resolve()
    if not current.is_relative_to(root.resolve()) or not current.is_file():
        raise ValueError(
            "Missing locked source; use the matching historical source for old checkpoints: "
            + str(path)
        )
    if hashlib.sha256(current.read_bytes()).hexdigest() != recorded_sha:
        raise ValueError("Locked source changed: " + str(path))


def canonical_training(protocol):
    return deepcopy(protocol)


def canonical_environment(contract):
    return deepcopy(contract)


def describe_source(protocol):
    path = "src/envs/radio/airtime.py"
    recorded = protocol["source_sha256"][path]
    verify_source(path, recorded)
    return dict(actual_allocator_sha256=recorded, numerical_variant="guarded_exact_source")
