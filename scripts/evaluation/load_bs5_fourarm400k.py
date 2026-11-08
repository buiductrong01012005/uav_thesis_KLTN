"""Load an imported four-arm checkpoint by training seed, without running rollouts."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "configs/bs5_fourarm400k_trained_models.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_path(record):
    path = (ROOT / record["path"]).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Asset path leaves the repository")
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"Asset checksum mismatch: {record['path']}")
    return path


def load_trained_model(arm, training_seed=551, *, registry=DEFAULT_REGISTRY, device="cpu"):
    """Return (learner, protocol, entry); callers must use the matching arm's masks."""
    import torch
    from src.rl.bs5_fourarm400k_qmix import make_learner

    index = json.loads(Path(registry).read_text(encoding="utf-8"))
    seed = index["seeds"][str(training_seed)]
    entry = seed["models"][arm]
    protocol = json.loads(checked_path(entry["protocol"]).read_text(encoding="utf-8"))
    status = json.loads(checked_path(entry["status"]).read_text(encoding="utf-8"))
    checked_path(entry["validation"])
    protocol_digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (protocol["training_seed"] != training_seed or protocol["arm"] != arm
            or protocol_digest != entry["protocol_digest"]
            or protocol["total_steps"] != seed["training_steps"]
            or status["status"] != "complete"
            or status["committed_training_steps"] != protocol["total_steps"]):
        raise ValueError("Training lineage mismatch")
    for name, expected in protocol["source_sha256"].items():
        from scripts.analysis.fourarm400k_lp_compat import verify_source
        if not (ROOT / name).resolve().is_relative_to(ROOT):
            raise ValueError("Source path leaves the repository")
        verify_source(name, expected, ROOT)
    best = status["best_by_metric"]["reward_total"]
    selected = (checked_path(entry["status"]).parent / best["file"]).resolve()
    if not selected.is_relative_to(ROOT) or file_sha256(selected) != entry["checkpoint"]["sha256"]:
        raise ValueError("Public checkpoint differs from its saved selection")
    payload = torch.load(checked_path(entry["checkpoint"]), map_location="cpu", weights_only=True)
    expected = {
        "arm": arm, "selected_step": entry["selected_step"],
        "selection_metric": "reward_total", "protocol_sha256": protocol_digest,
        "training_env_version": protocol["training_env_version"],
        "reward_version": status["reward_version"], "preset": protocol["preset"],
        "fair_config": protocol["fair_config"], "obs_dim": entry["obs_dim"],
        "state_dim": entry["state_dim"], "n_actions": entry["policy_actions"],
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint metadata mismatch")
    if (best["step"] != entry["selected_step"]
            or not math.isclose(best["score"], entry["validation_score"], rel_tol=1e-10, abs_tol=1e-12)
            or not math.isclose(payload["validation_score"], best["score"], rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError("Checkpoint selection mismatch")
    # Model construction consumes RNG; preserve the caller's CPU random stream.
    with torch.random.fork_rng(devices=[]):
        learner = make_learner(arm, device="cpu")
    learner.agent.load_state_dict(payload["agent"], strict=True)
    learner.mixer.load_state_dict(payload["mixer"], strict=True)
    learner.agent.to(device).eval()
    learner.mixer.to(device).eval()
    learner.device = torch.device(device)
    # Inference only: optimizer/replay/target states are absent from REVIEWs.
    return learner, protocol, entry


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seed", type=int, default=551)
    parser.add_argument("--arm", choices=["C-N", "C-R", "C-H", "H-only", "all"], default="all")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    arms = ["C-N", "C-R", "C-H", "H-only"] if args.arm == "all" else [args.arm]
    for arm in arms:
        learner, _, entry = load_trained_model(arm, args.training_seed, registry=args.registry)
        with torch.inference_mode():
            values = learner.agent(torch.zeros(5, entry["obs_dim"]))
        if values.shape != (5, entry["policy_actions"]) or not torch.isfinite(values).all():
            raise ValueError("Invalid actor output")
        print(json.dumps({"training_seed": args.training_seed, "arm": arm,
                          "selected_step": entry["selected_step"], "shape": list(values.shape),
                          "checkpoint": entry["checkpoint"]["path"], "status": "verified"}))


if __name__ == "__main__":
    main()
