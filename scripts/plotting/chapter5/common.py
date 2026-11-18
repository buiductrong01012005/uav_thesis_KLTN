"""Shared labels, audited CSV loading and explicit plotting receipts."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, StrMethodFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ARMS = ["C-N", "C-R", "C-H", "H-only", "DYNAMIC_AR", "HEURISTIC_HO", "RANDOM"]
NAMES = ["QMIX-A", "QMIX-AR", "QMIX-ARH", "QMIX-H", "Heuristic-AR", "Heuristic-H", "Random"]
COLORS = ["#6c6f78", "#1f5f99", "#138a72", "#c4561c", "#6f4e9a", "#8d6b2f", "#46586b"]
STYLES = ["--", "-", "-.", ":", "--", "-.", ":"]
LINE_COLORS = ["#606878", "#176b9b", "#3a7958", "#a85716", "#754e87", "#317b78", "#888176"]
Y = [0, 1, 2, 3, 4.3, 5.4, 6.5]


def options(description, argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=Path, default=ROOT / "results/w00_main/benchmark")
    parser.add_argument("--training", type=Path, default=ROOT / "results/w00_main/training")
    parser.add_argument("--output", type=Path, default=ROOT / "figures/w00_main/chapter5")
    parser.add_argument("--overwrite", action="store_true", help="Replace only these plot products")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    return parser.parse_args(argv)


def read(path, columns):
    frame = pd.read_csv(path)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    for key in columns:
        if key not in {"configuration", "arm", "metric"}:
            if not np.isfinite(pd.to_numeric(frame[key], errors="coerce")).all():
                raise ValueError(f"{path}: nonfinite {key}")
    return frame


def run_values(folder, metrics):
    frame = read(folder / "per_run.csv", ["configuration", "seed", *metrics])
    if len(frame) != 150 or frame.duplicated(["configuration", "training_seed", "seed"]).any():
        raise ValueError("Expected the 150 unique published W00 runs")
    for arm in ARMS:
        rows = frame[frame.configuration == arm]
        expected = {(s, t) for s in (551, 552, 553) for t in range(930201, 930211)}
        if arm in ARMS[:4]:
            actual = set(zip(rows.training_seed, rows.seed))
            if actual != expected or len(rows) != 30:
                raise ValueError("Incomplete training/test design: " + arm)
        elif (len(rows) != 10 or rows.training_seed.notna().any()
              or set(rows.seed) != set(range(930201, 930211))):
            raise ValueError("Baseline must have exactly ten unreplicated runs: " + arm)
    if set(frame.configuration) != set(ARMS):
        raise ValueError("Unexpected W00 configuration")
    return frame


def per_test_seed(folder, metrics):
    # Three training replicates per scenario; a baseline stays one run per scenario.
    return run_values(folder, metrics).groupby(["configuration", "seed"], as_index=False)[metrics].mean()


def number(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",")


def log_axis(ax, ticks):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def seed_panel(ax, values, metric, title, *, scale=1, log=False, xmax=90, ticks=None,
               label_at_point=False):
    for y, arm, color in zip(Y, ARMS, COLORS):
        x = values.loc[values.configuration == arm, metric].to_numpy() * scale
        ax.hlines(y, x.min(), x.max(), color=color, lw=1.3, alpha=.8)
        ax.scatter(x, np.full(len(x), y), s=7, color=color, alpha=.45, lw=0, zorder=2)
        ax.scatter([x.mean()], [y], s=30, color=color, zorder=3)
        if label_at_point:
            near_edge = x.mean() > xmax * .8
            ax.annotate(number(x.mean()), xy=(x.mean(), y),
                        xytext=(-6 if near_edge else 6, 0), textcoords="offset points",
                        ha="right" if near_edge else "left", va="center", color=color,
                        fontsize=8, fontweight="bold")
        else:
            ax.annotate(number(x.mean()), xy=(1., y), xycoords=("axes fraction", "data"),
                        ha="right", va="center", color=color, fontsize=8, fontweight="bold")
    if log:
        log_axis(ax, [10, 30, 100, 300])
        ax.set_xlim(9, 2500)
    else:
        ax.set_xlim(0, xmax)
        if ticks is not None:
            ax.set_xticks(ticks)
    ax.axhline(3.65, color="#c9ced6", lw=.8)
    ax.set_xlabel(title)
    ax.grid(axis="x", alpha=.25)


def save(fig, args, relative, values, method, inputs, script):
    stem = args.output / relative
    products = [stem.with_suffix(ext) for ext in (".pdf", ".png", ".csv", ".json")]
    if not args.overwrite and any(p.exists() for p in products):
        raise FileExistsError(f"Plot exists: {stem}; choose a new output or --overwrite")
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(products[0], metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(products[1], dpi=200)
    plt.close(fig)
    values.to_csv(products[2], index=False)
    receipt = dict(method=method, inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                   labels=dict(zip(ARMS, NAMES)), script=Path(script).relative_to(ROOT).as_posix(),
                   script_sha256=hashlib.sha256(Path(script).read_bytes()).hexdigest(),
                   common_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    products[3].write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(products[0].relative_to(ROOT) if products[0].is_relative_to(ROOT) else products[0])
