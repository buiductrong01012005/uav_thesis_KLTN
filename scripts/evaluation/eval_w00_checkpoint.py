"""Evaluate one W00 best.pt using its adjacent training protocol, without a bundle."""
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import signal

from scripts.w00 import ROOT, local_output
from scripts.evaluation import run_fourarm400k_drl_benchmark as drl
from scripts.evaluation import run_fourarm400k_heuristics as core
from scripts.evaluation.fourarm400k_return_summary import extend_summary
from scripts.analysis.fourarm400k_lp_compat import verify_source

VERSION = "w00_single_checkpoint_eval_v1"
ENV_KEYS = ("service", "reward_weights", "coverage_floor_mbps", "energy_config",
            "collision_config", "fair_config", "training_env_version", "obs_dim", "state_dim")


def load_checkpoint(path):
    """Read exactly path; never search for or substitute a frozen model."""
    import torch

    path = Path(path).resolve(strict=True)
    protocol_path = path.with_name("protocol.json")
    tp = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (tp.get("encoder", "cnn") != "cnn" or tp.get("supplemental")
            or tp.get("training_env_version") != "bs5_fourarm400k_v1"
            or tp.get("obs_dim") != 173 or tp.get("state_dim") != 156):
        raise ValueError("This entry point requires a main W00 CNN checkpoint and protocol")
    expected_actions = {"C-N": 10, "C-R": 10, "C-H": 15, "H-only": 5}
    if tp.get("arm") not in expected_actions or tp["policy_actions"] != expected_actions[tp["arm"]]:
        raise ValueError("Training arm/action contract mismatch")
    for rel, expected in tp["source_sha256"].items():
        relative = Path(rel)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid locked source path")
        verify_source(rel, expected, ROOT)
    payload = path.read_bytes()
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if saved.get("protocol_sha256") != core.digest(tp):
        raise ValueError("Checkpoint does not belong to the adjacent protocol.json")
    if saved.get("selection_metric") != "reward_total":
        raise ValueError("Expected best.pt selected by validation reward_total")
    if not 0 <= saved["selected_step"] <= tp["total_steps"]:
        raise ValueError("Selected checkpoint step is outside the training budget")
    model = dict(selected_step=saved["selected_step"], validation_score=saved["validation_score"],
                 checkpoint_sha256=hashlib.sha256(payload).hexdigest())
    status_path = path.with_name("status.json")
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        selected = status["best_by_metric"]["reward_total"]
        if (selected["step"] != model["selected_step"]
                or selected["score"] != model["validation_score"]
                or status["committed_training_steps"] < model["selected_step"]):
            raise ValueError("Checkpoint differs from the committed validation selection")
        stable = path.parent / selected["file"]
        if not stable.resolve().is_relative_to(path.parent):
            raise ValueError("Selection path escapes the training directory")
        if stable.exists() and stable.read_bytes() != payload:
            raise ValueError("Checkpoint differs from the immutable selected model")
    actor = drl.FrozenActor(payload, tp, model)
    return path, tp, model, actor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-seed", type=int, required=True)
    parser.add_argument("--output", type=local_output)
    parser.add_argument("--slots", type=int, default=10000)
    parser.add_argument("--diagnostic", action="store_true", help="Use reserved diagnostic test seed")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--stop-after-slots", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-seconds", type=float, default=36000)
    args = parser.parse_args(argv)
    if args.slots < 1 or args.save_every < 1 or args.log_every < 1 or args.stop_after_slots < 0:
        parser.error("Positive slots/save/log intervals and nonnegative stop-after-slots required")
    path, tp, model, actor = load_checkpoint(args.checkpoint)
    cfg = json.loads(core.CONFIG.read_text())
    allowed = cfg["diagnostic_seeds"] if args.diagnostic else tp["reserved_test_seeds"]
    if args.test_seed not in allowed:
        parser.error("Use a reserved test seed, or --diagnostic with a diagnostic seed")
    args.output = args.output or local_output(path.parent / f"eval/seed-{args.test_seed}")
    if args.output == path.parent or path.parent.is_relative_to(args.output):
        parser.error("Evaluation output must not contain the training directory")
    args.label = f"DRL_{tp['arm']}_S{tp['training_seed']}"
    args.part = 1 if args.diagnostic or args.test_seed in allowed[:5] else 2
    args.seeds = [args.test_seed]

    def protocol(label, part, diagnostic=False, seeds=None):
        chosen = cfg["diagnostic_seeds"] if diagnostic else [args.test_seed]
        horizon = min(args.slots, cfg["diagnostic_horizon"]) if diagnostic else args.slots
        environment = {k: deepcopy(tp[k]) for k in ENV_KEYS}
        source = {p.relative_to(ROOT).as_posix(): core.sha(p) for p in (ROOT / "src").rglob("*.py")}
        physics = {k: v for k, v in source.items() if k.startswith(("src/envs/", "src/diagnostics/"))}
        contract = dict(environment=environment, horizon=horizon, test_seeds=chosen,
                        initial_residual_battery=30., assignment_seed="scenario seed",
                        physics_source_sha256=physics)
        p = dict(version=VERSION, config=dict(cfg, horizon=horizon), training_protocol=environment,
                 source_sha256=source, environment_contract=contract,
                 environment_fingerprint=core.digest(contract), label=label, arm=tp["arm"],
                 training_seed=tp["training_seed"], part=part, diagnostic=diagnostic,
                 training_diagnostic=tp["diagnostic"], target_seeds=chosen, horizon=horizon,
                 selection=dict(model, metric="reward_total"), model_protocol_sha256=core.digest(tp),
                 checkpoint_path=path.as_posix(), training_protocol_path=path.with_name("protocol.json").as_posix(),
                 evaluation_adapter_sha256=core.sha(__file__),
                 actor_loader_sha256=core.sha(Path(drl.__file__)),
                 launcher_sha256=core.sha(ROOT / "scripts/w00.py"),
                 return_summary_sha256=core.sha(ROOT / "scripts/evaluation/fourarm400k_return_summary.py"),
                 rollout_runner_sha256=core.sha(Path(core.__file__)))
        p["comparison_fingerprint"] = core.digest(p)
        return p

    def starter(p, seed):
        return core.start_state(p, seed, controller=actor)

    print(json.dumps(dict(checkpoint=path.as_posix(), **model,
                          training_protocol_sha256=core.digest(tp)), indent=2), flush=True)
    signal.signal(signal.SIGTERM, core.stop_handler)
    signal.signal(signal.SIGINT, core.stop_handler)
    original = core.extend_summary
    core.extend_summary = extend_summary
    try:
        return core.run(args, protocol_factory=protocol, state_factory=starter, runtime_factory=drl.runtime)
    finally:
        core.extend_summary = original
