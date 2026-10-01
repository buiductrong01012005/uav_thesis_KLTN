from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIRECT_COLOR = "#F59E0B"
CURRICULUM_COLOR = "#2563EB"


def load_history(path: Path, value_key: str):
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d["history"]
    x = np.asarray([int(r["global_steps"]) for r in rows], dtype=int)
    y = np.asarray([float(r[value_key]) for r in rows], dtype=float)
    return x, y


def load_curriculum(path: Path):
    # Common Battery-ON validation is already stored as validation_mean_utility.
    return load_history(path, "validation_mean_utility")


def load_direct(path: Path):
    return load_history(path, "validation_mean_utility")


def aggregate(paths, loader):
    xs = None
    ys = []
    for p in paths:
        x, y = loader(p)
        if xs is None:
            xs = x
        elif not np.array_equal(xs, x):
            raise RuntimeError(f"Evaluation steps differ: {p}")
        ys.append(y)
    arr = np.vstack(ys)
    return xs, arr.mean(axis=0), arr.std(axis=0), arr


def load_summary(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--curriculum-dir",
        default="results/thesis_full_curriculum/cartesian",
    )
    ap.add_argument(
        "--direct-dir",
        default="results/thesis_curriculum_ablation/direct_full_task",
    )
    ap.add_argument(
        "--outdir",
        default="results/thesis_curriculum_ablation/figures",
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[91, 92, 93])
    args = ap.parse_args()

    curriculum_dir = Path(args.curriculum_dir)
    direct_dir = Path(args.direct_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    curriculum_hist = [
        curriculum_dir / f"seed{s}_history.json"
        for s in args.seeds
    ]
    direct_hist = [
        direct_dir / f"seed{s}_history.json"
        for s in args.seeds
    ]

    xc, mc, sc, ac = aggregate(curriculum_hist, load_curriculum)
    xd, md, sd, ad = aggregate(direct_hist, load_direct)
    if not np.array_equal(xc, xd):
        raise RuntimeError("Direct and curriculum runs use different evaluation steps")

    # Main learning-curve figure.
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.plot(
        xd, md,
        color=DIRECT_COLOR,
        linewidth=2.4,
        label="Spatial-QMIX",
    )
    ax.fill_between(
        xd, md - sd, md + sd,
        color=DIRECT_COLOR,
        alpha=0.17,
    )
    ax.plot(
        xc, mc,
        color=CURRICULUM_COLOR,
        linewidth=2.4,
        label="Spatial-QMIX + Curriculum",
    )
    ax.fill_between(
        xc, mc - sc, mc + sc,
        color=CURRICULUM_COLOR,
        alpha=0.17,
    )
    ax.axvline(30000, linestyle="--", linewidth=1.1, color="#6B7280")
    ax.axvline(60000, linestyle="--", linewidth=1.1, color="#6B7280")

    ymax = max(float(np.max(md + sd)), float(np.max(mc + sc)))
    ax.text(
        15000, ymax * 0.98,
        "Curriculum:\nBattery OFF",
        ha="center", va="top", fontsize=9,
    )
    ax.text(
        45000, ymax * 0.98,
        "Curriculum:\nBattery ON",
        ha="center", va="top", fontsize=9,
    )
    ax.text(
        75000, ymax * 0.98,
        "Curriculum:\nBattery ON +\ntransition focus",
        ha="center", va="top", fontsize=9,
    )
    ax.text(
        45000, ymax * 0.08,
        "Spatial-QMIX baseline: Battery ON throughout 0–90k",
        ha="center", va="bottom", fontsize=8, color=DIRECT_COLOR,
    )

    ax.set_xlim(0, 90000)
    ax.set_xlabel("Cumulative training transitions")
    ax.set_ylabel("Validation mean utility")
    ax.set_title("Curriculum ablation for Spatial-QMIX")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "curriculum_ablation_learning_curve.png", dpi=300)
    plt.close(fig)

    # Held-out comparison.
    rows = []
    for label, directory in [
        ("Spatial-QMIX", direct_dir),
        ("Spatial-QMIX + Curriculum", curriculum_dir),
    ]:
        for s in args.seeds:
            d = load_summary(directory / f"seed{s}_summary.json")
            rows.append({
                "method": label,
                "seed": s,
                "heldout_mean_utility": float(d["heldout_test"]["mean_utility"]),
                "heldout_mean_relays": float(d["heldout_test"]["mean_relays"]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "curriculum_ablation_multiseed.csv", index=False)

    agg = (
        df.groupby("method")
        .agg(
            heldout_mean=("heldout_mean_utility", "mean"),
            heldout_seed_std=("heldout_mean_utility", "std"),
            mean_relays=("heldout_mean_relays", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(outdir / "curriculum_ablation_aggregate.csv", index=False)

    order = ["Spatial-QMIX", "Spatial-QMIX + Curriculum"]
    values = [
        float(agg.loc[agg.method == m, "heldout_mean"].iloc[0])
        for m in order
    ]
    errors = [
        float(agg.loc[agg.method == m, "heldout_seed_std"].iloc[0])
        for m in order
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(
        order,
        values,
        yerr=errors,
        capsize=5,
        color=[DIRECT_COLOR, CURRICULUM_COLOR],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_ylabel("Held-out mean utility")
    ax.set_title("Held-out effect of curriculum training")
    ax.grid(axis="y", alpha=0.22)
    ax.set_ylim(0, max(np.asarray(values) + np.asarray(errors)) * 1.18)
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width()/2,
            v + 0.012,
            f"{v:.3f}",
            ha="center",
        )
    fig.tight_layout()
    fig.savefig(outdir / "curriculum_ablation_heldout.png", dpi=300)
    plt.close(fig)

    # Paired test-scenario comparison across all train seeds.
    paired_rows = []
    for s in args.seeds:
        dd = load_summary(direct_dir / f"seed{s}_summary.json")
        dc = load_summary(curriculum_dir / f"seed{s}_summary.json")
        dmap = {
            int(r["seed"]): float(r["mean_utility"])
            for r in dd["heldout_test"]["rows"]
        }
        cmap = {
            int(r["seed"]): float(r["mean_utility"])
            for r in dc["heldout_test"]["rows"]
        }
        for test_seed in sorted(set(dmap) & set(cmap)):
            paired_rows.append({
                "training_seed": s,
                "test_seed": test_seed,
                "spatial_qmix": dmap[test_seed],
                "spatial_qmix_curriculum": cmap[test_seed],
                "curriculum_minus_direct": cmap[test_seed] - dmap[test_seed],
            })

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(outdir / "curriculum_ablation_paired_heldout.csv", index=False)

    print(agg.to_string(index=False))
    if len(paired):
        gain = float(paired["curriculum_minus_direct"].mean() * 100.0)
        wins = int((paired["curriculum_minus_direct"] > 0).sum())
        print(
            f"\nCurriculum - direct paired mean gain: {gain:.3f} pp"
        )
        print(
            f"Curriculum paired wins: {wins}/{len(paired)} "
            f"({wins/len(paired):.3f})"
        )
    print("\nCreated figures/tables in:", outdir)


if __name__ == "__main__":
    main()
