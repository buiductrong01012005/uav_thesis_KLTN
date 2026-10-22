from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED_COLORS = {
    91: "#60A5FA",
    92: "#F59E0B",
    93: "#10B981",
}


def load_curve(indir: Path, seed: int):
    history_path = indir / f"seed{seed}_history.json"
    summary_path = indir / f"seed{seed}_summary.json"

    history = json.loads(history_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # True step-0 point = the same-seed battery-aware checkpoint before
    # transition-focused fine-tuning begins.
    initial = float(summary["initial_validation_v8"]["dynamic"]["mean_utility"])

    x = [0]
    y = [initial]

    for row in history["history"]:
        x.append(int(row["steps"]))
        y.append(float(row["validation_dynamic_mean_utility"]))

    return np.asarray(x, dtype=int), np.asarray(y, dtype=float)


def centered_moving_average(y: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y.copy()
    out = np.empty_like(y, dtype=float)
    half = window // 2
    for i in range(len(y)):
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def save_raw(xs, curves, seeds, outdir):
    arr = np.vstack(curves)
    mean = arr.mean(axis=0)
    sd = arr.std(axis=0)

    plt.figure(figsize=(8.0, 5.0))
    for seed, y in zip(seeds, curves):
        plt.plot(
            xs,
            y,
            marker="o",
            markersize=3.2,
            linewidth=1.0,
            alpha=0.45,
            color=SEED_COLORS.get(seed),
            label=f"Seed {seed}",
        )
    plt.plot(
        xs,
        mean,
        linewidth=2.4,
        marker="o",
        markersize=4.2,
        color="#1F2937",
        label="Mean",
    )
    plt.fill_between(xs, mean - sd, mean + sd, color="#94A3B8", alpha=0.25, label="±1 SD")

    plt.xlabel("Training transitions")
    plt.ylabel("Validation mean utility")
    plt.title("Learning curve of the proposed Spatial-QMIX method")
    plt.xlim(0, xs.max())
    plt.grid(alpha=0.22)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "learning_curve_dense_raw.png", dpi=300)
    plt.close()


def save_smoothed(xs, curves, seeds, outdir, window):
    arr = np.vstack(curves)
    mean = arr.mean(axis=0)
    sd = arr.std(axis=0)
    smooth = centered_moving_average(mean, window)

    plt.figure(figsize=(8.0, 5.0))

    # Raw measured points remain visible: no fabricated measurements.
    plt.scatter(xs, mean, s=20, color="#64748B", alpha=0.65, label="Measured 3-seed mean")
    plt.plot(
        xs,
        smooth,
        linewidth=2.6,
        color="#2563EB",
        label=f"{window}-point moving average",
    )
    plt.fill_between(
        xs,
        mean - sd,
        mean + sd,
        color="#93C5FD",
        alpha=0.25,
        label="±1 SD of measured values",
    )

    plt.xlabel("Training transitions")
    plt.ylabel("Validation mean utility")
    plt.title("Learning trend of the proposed Spatial-QMIX method")
    plt.xlim(0, xs.max())
    plt.grid(alpha=0.22)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "learning_curve_dense_smoothed.png", dpi=300)
    plt.close()


def save_zoom(xs, curves, seeds, outdir):
    arr = np.vstack(curves)
    mean = arr.mean(axis=0)
    sd = arr.std(axis=0)
    keep = xs <= 10000

    plt.figure(figsize=(8.0, 5.0))
    for seed, y in zip(seeds, curves):
        plt.plot(
            xs[keep],
            y[keep],
            marker="o",
            markersize=3.2,
            linewidth=1.0,
            alpha=0.45,
            color=SEED_COLORS.get(seed),
            label=f"Seed {seed}",
        )
    plt.plot(xs[keep], mean[keep], linewidth=2.4, marker="o", color="#1F2937", label="Mean")
    plt.fill_between(
        xs[keep],
        (mean - sd)[keep],
        (mean + sd)[keep],
        color="#94A3B8",
        alpha=0.25,
        label="±1 SD",
    )
    plt.xlabel("Training transitions")
    plt.ylabel("Validation mean utility")
    plt.title("Early learning behavior (0–10k transitions)")
    plt.xlim(0, 10000)
    plt.grid(alpha=0.22)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "learning_curve_dense_0_10k.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="results/thesis_dense_learning_curve")
    parser.add_argument("--outdir", default="results/thesis_figures")
    parser.add_argument("--seeds", type=int, nargs="+", default=[91, 92, 93])
    parser.add_argument("--smooth-window", type=int, default=3)
    args = parser.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    curves = []
    xs = None
    for seed in args.seeds:
        x, y = load_curve(indir, seed)
        if xs is None:
            xs = x
        elif not np.array_equal(xs, x):
            raise RuntimeError("Dense histories do not have identical evaluation steps.")
        curves.append(y)

    save_raw(xs, curves, args.seeds, outdir)
    save_smoothed(xs, curves, args.seeds, outdir, args.smooth_window)
    save_zoom(xs, curves, args.seeds, outdir)

    arr = np.vstack(curves)
    df = pd.DataFrame({"transitions": xs})
    for seed, y in zip(args.seeds, curves):
        df[f"seed{seed}"] = y
    df["mean"] = arr.mean(axis=0)
    df["std"] = arr.std(axis=0)
    df.to_csv(outdir / "learning_curve_dense_data.csv", index=False)

    print("Created:")
    print(outdir / "learning_curve_dense_raw.png")
    print(outdir / "learning_curve_dense_smoothed.png")
    print(outdir / "learning_curve_dense_0_10k.png")
    print(outdir / "learning_curve_dense_data.csv")


if __name__ == "__main__":
    main()
