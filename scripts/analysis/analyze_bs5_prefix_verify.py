"""Offline prefix comparison: stdlib only; never imports Torch or loads weights."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for attempt in range(6):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * 2**attempt)


def lines(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def first_difference(a, b, keys):
    for key in keys:
        if a.get(key) != b.get(key):
            return dict(field=key, a=a.get(key), b=b.get(key))
    return None


def compare(a, b, *, limit=None):
    """Separate strict protocol, RNG, behavior, and exact/tolerant numerical results."""
    a, b = Path(a), Path(b)
    ia, ib = read(a / "initial.json"), read(b / "initial.json")
    initial = first_difference(ia, ib, ["state", "resolved_config", "runtime", "schedule"])
    mismatches = []
    if initial:
        mismatches.append(dict(step=0, layer="initial", **initial))
    step_keys = {
        "protocol": [
            "episode",
            "slot",
            "epsilon",
            "obs",
            "state",
            "mask",
            "next_obs",
            "next_state",
            "next_mask",
            "done",
            "exogenous",
            "decision",
            "post",
            "replay_size",
            "lineage",
            "target_sync",
        ],
        "rng": ["rng_before", "rng_after", "replay_rng"],
        "behavior": ["actions", "reward", "service"],
    }
    counts = dict(steps=0, updates=0, evaluations=0)
    exact, tolerant = True, True
    max_errors = {}
    for kind, filename in (
        ("step", "steps.jsonl.gz"),
        ("update", "updates.jsonl.gz"),
        ("eval", "evals.jsonl.gz"),
    ):
        left, right = lines(a / filename), lines(b / filename)
        if limit is not None and kind == "step":
            left, right = itertools.islice(left, limit), itertools.islice(right, limit)
        elif limit is not None:
            endpoint = limit if kind == "update" else 0
            left = itertools.takewhile(lambda r: r["step"] <= endpoint, left)
            right = itertools.takewhile(lambda r: r["step"] <= endpoint, right)
        for x, y in itertools.zip_longest(left, right):
            if x is None or y is None:
                mismatches.append(dict(step=(x or y).get("step", 0), layer="length", stream=kind))
                break
            counts[{"step": "steps", "update": "updates", "eval": "evaluations"}[kind]] += 1
            if kind == "step":
                for layer, keys in step_keys.items():
                    diff = first_difference(x, y, keys)
                    if diff and not any(m["layer"] == layer for m in mismatches):
                        mismatches.append(dict(step=x["step"], layer=layer, **diff))
            elif kind == "update":
                diff = first_difference(
                    x,
                    y,
                    [
                        "step",
                        "update",
                        "batch",
                        "batch_fields",
                        "ids",
                        "current_masks",
                        "rng_before",
                        "rng_after",
                        "target_sync",
                    ],
                )
                if diff and not any(m["layer"] == "batch" for m in mismatches):
                    mismatches.append(dict(step=x["step"], layer="batch", **diff))
            else:
                diff = first_difference(x, y, ["step", "rows", "rng_before", "rng_after"])
                if diff and not any(m["layer"] == "eval" for m in mismatches):
                    mismatches.append(dict(step=x["step"], layer="eval", **diff))
            if kind in ("step", "update"):
                if x["numeric_hashes"] != y["numeric_hashes"]:
                    exact = False
                    if not any(m["layer"] == "numerics" for m in mismatches):
                        mismatches.append(
                            dict(
                                step=x["step"],
                                layer="numerics",
                                stream=kind,
                                exact=False,
                                errors=y.get("errors", {}),
                            )
                        )
                for key, error in y.get("errors", {}).items():
                    dest = max_errors.setdefault(key, dict(max_abs=0.0, max_rel=0.0))
                    for metric in ("max_abs", "max_rel"):
                        dest[metric] = max(dest[metric], error[metric])
                    tolerant = tolerant and error["within_tolerance"]
    ea, eb = read(a / "endpoint.json"), read(b / "endpoint.json")
    if limit is None:
        diff = first_difference(
            ea,
            eb,
            ["steps", "updates", "replay_size", "lineage", "reset_applied", "target_sync_steps"],
        )
        if diff:
            mismatches.append(dict(step=min(ea["steps"], eb["steps"]), layer="endpoint", **diff))
        if ea["state"] != eb["state"]:
            exact = False
            mismatches.append(
                dict(step=min(ea["steps"], eb["steps"]), layer="final_state", field="state")
            )
        expected = dict(
            steps=3000,
            updates=501,
            replay_size=3000,
            lineage=0,
            reset_applied=0,
            target_sync_steps=[1000, 2000, 3000],
        )
        for name, end in (("A", ea), ("B", eb)):
            if (
                any(end.get(k) != v for k, v in expected.items())
                or end["status"] != "complete_stop_before_clear"
            ):
                mismatches.append(
                    dict(step=end["steps"], layer="endpoint", field=name + "_stop_contract")
                )
        if counts != dict(steps=3000, updates=501, evaluations=2):
            mismatches.append(
                dict(step=0, layer="length", field="expected_prefix_counts", counts=counts)
            )
    for name in sorted(
        {p.name for p in a.glob("state_*.json")} & {p.name for p in b.glob("state_*.json")}
    ):
        step = int(name.removeprefix("state_").removesuffix(".json"))
        if limit is not None and step > limit:
            continue
        if read(a / name) != read(b / name):
            exact = False
            mismatches.append(dict(step=step, layer="final_state", field=name))
    layers = {m["layer"] for m in mismatches}
    behavior = "behavior" not in layers and "eval" not in layers
    protocol = not layers.intersection(
        {"initial", "protocol", "rng", "batch", "length", "endpoint"}
    )
    # Strict pass: exact numerical agreement as well as protocol and behavior.
    status = (
        "MATCH_SAME_RUNTIME"
        if protocol and behavior and exact
        else (
            "NUMERICALLY_CLOSE_BEHAVIOR_DIFFERS"
            if protocol and tolerant and not behavior
            else "BLOCKED_OR_UNRESOLVED"
        )
    )
    order = {
        v: i
        for i, v in enumerate(
            (
                "initial",
                "protocol",
                "rng",
                "batch",
                "numerics",
                "final_state",
                "behavior",
                "eval",
                "endpoint",
                "length",
            )
        )
    }
    mismatches.sort(key=lambda m: (m["step"], order[m["layer"]]))
    return dict(
        status=status,
        initial_equal=initial is None,
        protocol_equal=protocol,
        behavior_equal=behavior,
        numerics_exact=exact,
        numerics_within_frozen_tolerance=tolerant,
        counts=counts,
        max_errors=max_errors,
        first_divergence=mismatches[0] if mismatches else None,
        first_by_layer=mismatches,
        comparison_limit=limit,
    )


def verify_manifest(root):
    root = Path(root).resolve()
    manifest = read(root / "REVIEW_SHA256SUMS.json")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert actual == set(manifest) | {"REVIEW_SHA256SUMS.json"}, "review file set mismatch"
    for name, expected in manifest.items():
        path = (root / name).resolve()
        assert path.is_relative_to(root)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--verify-review", action="store_true")
    args = parser.parse_args()
    if args.verify_review:
        verify_manifest(args.root)
    value = compare(args.root / "trials/A/diagnostic", args.root / "trials/B/diagnostic")
    recorded = read(args.root / "prefix_comparison.json")
    assert value == recorded, "recorded comparison differs from offline reconstruction"
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
