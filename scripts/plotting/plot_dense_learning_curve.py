from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


def load(seed, d):
    h = json.loads((d / f"seed{seed}_history.json").read_text())
    s = json.loads((d / f"seed{seed}_summary.json").read_text())
    x = [0] + [int(r["steps"]) for r in h["history"]]
    y0 = float(s["initial_validation_v8"]["dynamic"]["mean_utility"])
    y = [y0] + [float(r["validation_dynamic_mean_utility"]) for r in h["history"]]
    return np.asarray(x), np.asarray(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/thesis_dense_learning_curve")
    ap.add_argument("--outdir", default="results/thesis_figures")
    args = ap.parse_args()
    d = Path(args.indir)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [91, 92, 93]
    ys = []
    xs = None
    for s in seeds:
        x, y = load(s, d)
        if xs is None:
            xs = x
        if not np.array_equal(xs, x):
            raise RuntimeError("different evaluation steps")
        ys.append(y)
    a = np.vstack(ys)
    mean = a.mean(0)
    sd = a.std(0)

    plt.figure(figsize=(7.2, 4.6))
    for s, y in zip(seeds, a):
        plt.plot(xs, y, marker=".", alpha=0.35, label=f"Seed {s}")
    plt.plot(xs, mean, marker="o", linewidth=2.2, label="Mean")
    plt.fill_between(xs, mean - sd, mean + sd, alpha=0.18, label="±1 SD")
    plt.xlabel("Training transitions")
    plt.ylabel("Validation mean utility")
    plt.title("Learning curve of the proposed Spatial-QMIX method")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "learning_curve_full.png", dpi=300)
    plt.close()

    keep = xs <= 10000
    plt.figure(figsize=(7.2, 4.6))
    for s, y in zip(seeds, a):
        plt.plot(xs[keep], y[keep], marker=".", alpha=0.35, label=f"Seed {s}")
    plt.plot(xs[keep], mean[keep], marker="o", linewidth=2.2, label="Mean")
    plt.fill_between(xs[keep], (mean - sd)[keep], (mean + sd)[keep], alpha=0.18, label="±1 SD")
    plt.xlabel("Training transitions")
    plt.ylabel("Validation mean utility")
    plt.title("Early learning behavior (0–10k transitions)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "learning_curve_0_10k.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
