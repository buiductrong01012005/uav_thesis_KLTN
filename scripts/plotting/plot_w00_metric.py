"""Plot measured CSV values by run, seed, or time block, with explicit units."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--by", choices=("run", "seed", "time"), default="seed")
    parser.add_argument("--x", help="Time/block column; required for --by time")
    parser.add_argument("--group", default="configuration")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    required = [args.metric, args.group]
    x = "seed" if args.by == "seed" else args.x if args.by == "time" else "run_index"
    if args.by == "time" and not args.x:
        parser.error("--by time requires --x naming an existing time/block column")
    if args.by == "run":
        frame = frame.copy()
        frame[x] = frame.groupby(args.group, dropna=False).cumcount() + 1
    required.append(x)
    if any(k not in frame for k in required):
        parser.error(f"Missing required columns {required}; available: {list(frame)}")
    if not np.isfinite(pd.to_numeric(frame[args.metric], errors="coerce")).all():
        raise ValueError("Metric must contain finite measured values")
    # Average already computed per-run values; never reconstruct a pooled ratio.
    data = frame.groupby([args.group, x], dropna=False)[args.metric].agg(["mean", "count"]).reset_index()
    args.output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for name, values in data.groupby(args.group, dropna=False):
        values = values.sort_values(x)
        ax.plot(values[x], values["mean"], marker=".", label=str(name))
    ax.set(xlabel=x, ylabel=f"{args.metric} ({args.unit})")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"{args.metric}_by_{args.by}.{suffix}", dpi=160)
    plt.close(fig)
    data.to_csv(args.output / "plotted_values.csv", index=False)
    (args.output / "method.json").write_text(json.dumps(dict(
        input=str(args.input), metric=args.metric, unit=args.unit, by=args.by, x=x,
        method="Arithmetic mean of input CSV metric values within each group/x; no pooled-ratio recomputation"
    ), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
