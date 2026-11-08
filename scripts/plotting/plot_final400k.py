"""CSV figures imported from KLTN_400K_script_va_hinh; layout and statistics preserved."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, StrMethodFormatter
import numpy as np
import pandas as pd


import argparse
parser = argparse.ArgumentParser(description="Draw the three thesis figures from audited CSVs; no rollout.")
parser.add_argument("--input", type=Path, default=Path("results/final400k/benchmark"))
parser.add_argument("--training", type=Path, default=Path("results/final400k/training"))
parser.add_argument("--output", type=Path, default=Path("results/final400k/figures"))
args = parser.parse_args()
root = args.input
output = args.output
output.mkdir(parents=True, exist_ok=True)

arms = ["C-N", "C-R", "C-H", "H-only", "DYNAMIC_AR", "HEURISTIC_HO", "RANDOM"]
names = [
    "C-N", "C-R", "C-H", "H-only",
    "Phương pháp dựa trên quy tắc\nchọn hướng di chuyển và vai trò\n(truy nhập hoặc chuyển tiếp)",
    "Phương pháp di chuyển\ndựa trên quy tắc\nvới vai trò kết hợp",
    "Random",
]
rows = [0, 1, 2, 3, 4.3, 6, 7.3]
colors = ["#606878", "#176b9b", "#3a7958", "#a85716", "#754e87", "#317b78", "#888176"]
styles = ["--", "-", "-.", ":", "--", "-.", ":"]
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight", pad_inches=.05)
    fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight", pad_inches=.05)
    plt.close(fig)


data = pd.read_csv(args.training / "learning_mean_sd.csv")
fig, ax = plt.subplots(figsize=(6.3, 2.95), layout="constrained")
for arm, color, style in zip(arms[:4], colors, styles):
    frame = data[(data.arm == arm) & (data.metric == "reward_total")].sort_values("step")
    assert len(frame) == 67 and (frame.training_seeds == 3).all() and frame.step.iloc[-1] == 399600
    x = frame.step.to_numpy() / 1000
    y = frame["mean"].to_numpy()
    sd = frame.sample_sd.to_numpy()
    ax.plot(x, y, ls=style, color=color, label=arm, lw=1.75)
    ax.fill_between(x, y - sd, y + sd, color=color, alpha=.13, lw=0)
ax.axvline(12, color="#3b3b3b", lw=.8, ls=(0, (3, 3)))
ax.text(16, .96, "Nhiệm vụ đầy đủ", transform=ax.get_xaxis_transform(), va="top", fontsize=8)
ax.legend(loc="lower right", frameon=False, ncol=4, fontsize=8.5, columnspacing=1.2)
ax.set(xlabel="Số bước môi trường (nghìn)", ylabel="Phần thưởng kiểm định",
       xticks=range(0, 401, 50), xlim=(-3, 403))
ax.grid(axis="y", alpha=.24)
save(fig, "learning")

a = pd.read_csv(root / "aggregate.csv").set_index("configuration").loc[arms]
fig, axes = plt.subplots(2, 2, figsize=(6.55, 6.35), sharey="row", layout="constrained")
metrics=[
    ('throughput_mbps','(a) Thông lượng (Mbps)',False),
    ('qos_ues','(b) UE đạt 4 Mbps mỗi slot',False),
    ('joules_per_mbit_pooled','(c) Điện năng (J/Mbit, log)',True),
    ('actor_ms','(d) Thời gian quyết định (ms, log)',True)]
for ax,(key,title,log) in zip(axes.flat,metrics):
    v,lo,hi=(a[col].to_numpy() for col in [key,key+'_ci95_low',key+'_ci95_high'])
    for pos,mean,low,high,color in zip(rows,v,lo,hi,colors):
        ax.errorbar(mean,pos,xerr=[[max(0,mean-low)],[max(0,high-mean)]],fmt='o',
                    ms=4.4,color=color,capsize=2,lw=1.15)
    if log:
        ax.set_xscale('log')
        if key == 'joules_per_mbit_pooled':
            ax.set_xlim(10, 440)
            ax.set_xticks([10, 20, 40, 100, 200, 400])
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_xlim(left=0)
    ax.set_title(title,loc='left',fontsize=8.5,pad=5)
    ax.grid(axis='x',alpha=.22)
    ax.grid(axis='y',alpha=.13)
for ax in axes[:, 0]:
    ax.set_yticks(rows, names, fontsize=8.5)
    ax.set_ylim(rows[-1] + .4, -.4)
save(fig,'benchmark')

block = pd.read_csv(root / "temporal_1000slot.csv")
fig, (trend, trend_key, split, split_key) = plt.subplots(
    4, 1, figsize=(6.55, 7.7), layout="constrained",
    gridspec_kw={"height_ratios": [2.2, 1.05, 2.75, .53]},
)
for arm, name, color, style in zip(arms, names, colors, styles):
    mean = block[block.configuration == arm].groupby("block").throughput_mbps.mean()
    trend.plot(mean.index, mean.values, marker="o", ms=2.5, color=color, ls=style,
               label=name, lw=1.45)
trend.set(xlabel="Khối liên tiếp (1.000 slot mỗi khối)", ylabel="Mbps",
          xticks=range(1, 11), ylim=(0, 190))
trend.set_title("(a) Diễn biến thông lượng", loc="left", fontsize=9, pad=5)
trend.grid(axis="y", alpha=.23)
trend_key.set_axis_off()
lines, labels = trend.get_legend_handles_labels()
legend_order = [0, 4, 1, 5, 2, 3, 6]
trend_key.legend([lines[i] for i in legend_order], [labels[i] for i in legend_order],
                 ncol=4, loc="center", frameon=False, fontsize=8, columnspacing=1.15)

parts = [
    ("throughput_mbps", "Đã truyền", "#296978"),
    ("loss_bh_schedule_capacity_mbps", "Thiếu tài nguyên truyền dẫn", "#df9d4b"),
    ("loss_bh_disconnected_mbps", "Không có tuyến về BS", "#b85a70"),
    ("loss_access_radio_mbps", "Giới hạn vô tuyến truy nhập", "#7a98a8"),
    ("loss_association_mbps", "Chưa ghép nối", "#aab9c8"),
    ("loss_outside_access_mbps", "Ngoài vùng truy nhập", "#dce2e7"),
]
left = np.zeros(len(arms))
for key, label, color in parts:
    values = a[key].to_numpy()
    split.barh(rows, values, left=left, color=color, label=label, height=.63)
    left += values
assert np.allclose(left, 360, atol=1e-5)
split.set_yticks(rows, names, fontsize=8.5)
split.set(ylim=(rows[-1] + .4, -.4), xlim=(0, 360), xticks=range(0, 361, 60), xlabel="Mbps")
split.set_title("(b) Phân rã nhu cầu 360 Mbps", loc="left", fontsize=9, pad=5)
split.grid(axis="x", alpha=.22)
split.set_axisbelow(True)
split_key.set_axis_off()
bars, labels = split.get_legend_handles_labels()
split_key.legend(bars, labels, ncol=3, loc="center", frameon=False,
                 fontsize=7.2, columnspacing=1.15)
save(fig, "waterfall")
