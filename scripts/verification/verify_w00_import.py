"""Verify the W00 delivery, all frozen models, and six ablation checkpoint pairs."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if sys.platform == "win32":
    sys.path.insert(0, str(ROOT / "scripts/compat/windows"))

from scripts.verification.check_final400k import check_tables, check_models


def verify():
    tables, _ = check_tables(ROOT / "results/w00_main/benchmark")
    models = check_models(ROOT / "checkpoints/w00_main/FOURARM400K_FROZEN_MODELS.zip")
    receipt = json.loads((ROOT / "results/w00_ablation/checkpoint_verification.json").read_text(encoding="utf-8"))
    expected = {(v, s) for v in ("mlp", "no_curriculum") for s in (551, 552, 553)}
    if len(receipt) != 6 or {(r["variant"], r["training_seed"]) for r in receipt} != expected:
        raise ValueError("Expected exactly six unique ablation checkpoint pairs")
    pairs = []
    for record in receipt:
        variant, seed = record["variant"], record["training_seed"]
        folder = ROOT / "checkpoints/w00_ablation" / f"W00_CR_{variant}_S{seed}"
        with zipfile.ZipFile(ROOT / "results/w00_ablation/evidence/raw" / f"W00_CR_{variant}_S{seed}_EVAL.zip") as z:
            if z.testzip() is not None:
                raise ValueError("Corrupt ablation EVAL ZIP")
            ep = json.loads(z.read("eval_protocol.json"))
        tp = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        checkpoint = hashlib.sha256((folder / "best.pt").read_bytes()).hexdigest()
        protocol_hash = hashlib.sha256(json.dumps(tp, sort_keys=True).encode()).hexdigest()
        if (checkpoint != ep["checkpoint_sha256"] or checkpoint != record["expected_sha256"]
                or protocol_hash != ep["training_protocol_sha256"]
                or (ep["variant"], ep["training_seed"], ep["arm"]) != (variant, seed, "C-R")
                or tp["training_seed"] != seed or tp["total_steps"] != 399600
                or status["status"] != "complete" or status["committed_training_steps"] != 399600
                or status["best_by_metric"]["reward_total"]["step"] != ep["selection"]["selected_step"]):
            raise ValueError(f"Ablation checkpoint/protocol mismatch: {variant} seed {seed}")
        pairs.append(dict(variant=variant, training_seed=seed, checkpoint_sha256=checkpoint, status="matched"))
    return dict(status="passed", tables=tables, frozen_models=len(models), ablation_pairs=pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("OK: 150 W00 runs, 12 loaded frozen models, 6 matched ablation checkpoints/protocols")


if __name__ == "__main__":
    main()
