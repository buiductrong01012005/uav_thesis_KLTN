from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLORS = {
    "target_aware": "#2563EB",   # blue
    "cartesian": "#F97316",      # orange
}
LABELS = {
    "target_aware": "Target-aware Spatial-QMIX",
    "cartesian": "Cartesian movement × role Spatial-QMIX",
}


def load_method(root: Path, method: str, seeds):
    common_curves = []
    native_curves = []
    x_ref = None
    for seed in seeds:
        p = root / method / f"seed{seed}_history.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        rows = d["history"]
        x = np.asarray([int(r["global_steps"]) for r in rows])
        common = np.asarray([float(r["validation_mean_utility"]) for r in rows])
        native = np.asarray([
            np.nan if r.get("stage_validation_mean_utility") is None
            else float(r["stage_validation_mean_utility"])
            for r in rows
        ])
        if x_ref is None:
            x_ref = x
        elif not np.array_equal(x_ref, x):
            raise RuntimeError(f"Different eval steps in {method} seed {seed}")
        common_curves.append(common)
        native_curves.append(native)
    return x_ref, np.vstack(common_curves), np.vstack(native_curves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/thesis_full_curriculum")
    ap.add_argument("--outdir", default="results/thesis_curriculum_figures")
    ap.add_argument("--seeds", nargs="+", type=int, default=[91,92,93])
    args = ap.parse_args()
    root = Path(args.indir)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    loaded = {}
    for method in ("cartesian", "target_aware"):
        loaded[method] = load_method(root, method, args.seeds)

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    # Curriculum stage bands are neutral; algorithms remain distinctly colored.
    ax.axvspan(0, 30000, color="#E5E7EB", alpha=0.32)
    ax.axvspan(30000, 60000, color="#D1FAE5", alpha=0.20)
    ax.axvspan(60000, 90000, color="#EDE9FE", alpha=0.25)
    ax.axvline(30000, color="#6B7280", linestyle="--", linewidth=1.2)
    ax.axvline(60000, color="#6B7280", linestyle="--", linewidth=1.2)

    records = []
    for method in ("cartesian", "target_aware"):
        x, a, _ = loaded[method]
        mean = a.mean(axis=0); sd = a.std(axis=0)
        color = COLORS[method]
        ax.plot(x, mean, color=color, linewidth=2.6, label=LABELS[method])
        ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=0.16)
        ax.scatter(x, mean, color=color, s=11, alpha=0.55)
        for i, step in enumerate(x):
            records.append({
                "method": method, "global_steps": int(step),
                "mean": float(mean[i]), "std": float(sd[i]),
            })

    ymax = ax.get_ylim()[1]
    ax.text(15000, ymax, "Stage 1\nBattery OFF", ha="center", va="top", fontsize=9)
    ax.text(45000, ymax, "Stage 2\nBattery ON", ha="center", va="top", fontsize=9)
    ax.text(75000, ymax, "Stage 3\nBattery ON + transition-focused curriculum",
            ha="center", va="top", fontsize=9)
    ax.set_xlim(0, 90000)
    ax.set_xlabel("Cumulative training transitions")
    ax.set_ylabel("Validation mean user-satisfaction utility")
    ax.set_title("From-scratch curriculum learning: action-formulation comparison")
    ax.grid(alpha=0.20)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "learning_curve_cartesian_vs_target_aware.png", dpi=300)
    plt.close(fig)

    # Stage-1 learnability uses a Battery-OFF validation profile matching Stage 1.
    # Step 0 is omitted here because its stage-native value is intentionally not
    # duplicated in the common-validation row; the first measured point is 1k.
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for method in ("cartesian", "target_aware"):
        x, _, native = loaded[method]
        keep = (x <= 30000) & (x > 0)
        stage = native[:, keep]
        mean = np.nanmean(stage, axis=0); sd = np.nanstd(stage, axis=0)
        color = COLORS[method]
        ax.plot(x[keep], mean, color=color, linewidth=2.6, label=LABELS[method])
        ax.fill_between(x[keep], mean-sd, mean+sd, color=color, alpha=0.16)
        ax.scatter(x[keep], mean, color=color, s=13, alpha=0.60)
    ax.set_xlim(0,30000)
    ax.set_xlabel("Stage-1 training transitions")
    ax.set_ylabel("Battery-OFF validation mean utility")
    ax.set_title("Stage 1 learnability from random initialization: action formulation")
    ax.grid(alpha=0.20); ax.legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curve_stage1_from_scratch.png", dpi=300)
    plt.close(fig)

    pd.DataFrame(records).to_csv(out / "learning_curve_aggregate.csv", index=False)
    print(out / "learning_curve_cartesian_vs_target_aware.png")
    print(out / "learning_curve_stage1_from_scratch.png")
    print(out / "learning_curve_aggregate.csv")


if __name__ == "__main__":
    main()
