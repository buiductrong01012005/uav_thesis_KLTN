import sys
import statistics as st
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter

data, out = Path(sys.argv[1]), Path(sys.argv[2])
import pandas as pd

_r = pd.read_csv(data / "per_run.csv")
_t = _r.groupby(["configuration", "seed"], as_index=False)[
    ["throughput_mbps", "qos_fraction", "blocked_per_policy_move", "joules_per_mbit"]
].mean()
rows = [
    {**{k: str(v) for k, v in rr.items()}, "variant": rr["configuration"]}
    for rr in _t.to_dict("records")
]
variants = ["C-N", "C-R", "C-H", "H-only", "DYNAMIC_AR", "HEURISTIC_HO", "RANDOM"]
names = [
    "C-N",
    "C-R",
    "C-H",
    "H-only",
    "Đối chứng\nchọn vai trò",
    "Đối chứng\nvai trò kết hợp",
    "Random",
]
colors = ["#6c6f78", "#1f5f99", "#138a72", "#c4561c", "#6f4e9a", "#8d6b2f", "#46586b"]
ypos = [0, 1, 2, 3, 4.3, 5.4, 6.5]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)


def vals(v, key, scale=1.0):
    return [float(r[key]) * scale for r in rows if r["variant"] == v]


def panel(ax, key, title, scale=1.0, log=False, fmt="{:.2f}", xmax=None, ticks=None):
    for y, v, c in zip(ypos, variants, colors):
        x = vals(v, key, scale)
        m = st.mean(x)
        ax.hlines(y, min(x), max(x), color=c, lw=1.3, alpha=0.8)
        ax.scatter(x, [y] * len(x), s=7, color=c, alpha=0.45, lw=0, zorder=2)
        ax.scatter([m], [y], s=30, color=c, zorder=3)
        ax.annotate(
            fmt.format(m).replace(".", ","),
            xy=(1.0, y),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="center",
            color=c,
            fontsize=8,
            fontweight="bold",
        )
    if log:
        ax.set_xscale("log")
        ax.set_xlim(9, 2500)
        ax.set_xticks([10, 30, 100, 300])
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_xlim(0, xmax)
        if ticks:
            ax.set_xticks(ticks)
    ax.axhline(3.65, color="#c9ced6", lw=0.8)
    ax.set_xlabel(title)
    ax.grid(axis="x", alpha=0.25)


def save(fig, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)


for name, specs in [
    (
        "02_benchmark_phuc_vu",
        [
            (
                "throughput_mbps",
                "Thông lượng (Mbps)",
                1,
                False,
                "{:.2f}",
                235,
                [0, 50, 100, 150, 200],
            ),
            ("qos_fraction", "UE đạt 4 Mbps (%)", 100, False, "{:.2f}", 50, [0, 10, 20, 30, 40]),
        ],
    ),
    (
        "03_benchmark_van_hanh",
        [
            (
                "blocked_per_policy_move",
                "Yêu cầu di chuyển bị chặn (%)",
                100,
                False,
                "{:.2f}",
                17.5,
                [0, 4, 8, 12],
            ),
            (
                "joules_per_mbit",
                "Điện năng trên dữ liệu (J/Mbit, log)",
                1,
                True,
                "{:.2f}",
                None,
                None,
            ),
        ],
    ),
]:
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.45), sharey=True, layout="constrained")
    for ax, (k, t, s, lg, f, xm, tk), lab in zip(axes, specs, ["(a)", "(b)"]):
        panel(ax, k, t, s, lg, f, xm, tk)
        ax.set_title(lab, loc="left", fontsize=9, fontweight="bold")
    axes[0].set_yticks(ypos, names)
    axes[0].set_ylim(ypos[-1] + 0.5, -0.5)
    save(fig, name)
print("ok")
