"""Figure 5.3: arithmetic means in ten successive 1,000-slot blocks."""
from . import common as c


def main(argv=None):
    args = c.options(__doc__, argv)
    source = args.input / "temporal_1000slot.csv"
    frame = c.read(source, ["configuration", "block", "throughput_mbps"])
    if len(frame) != 1500 or frame.duplicated(["configuration", "training_seed", "scenario_seed", "block"]).any():
        raise ValueError("Expected 150 runs times ten unique blocks")
    if set(frame.configuration) != set(c.ARMS):
        raise ValueError("Expected seven configurations")
    for arm in c.ARMS:
        counts = frame[frame.configuration == arm].groupby("block").size()
        if set(counts.index) != set(range(1, 11)) or not (counts == (30 if arm in c.ARMS[:4] else 10)).all():
            raise ValueError("Incomplete block design: " + arm)
    values = frame.groupby(["configuration", "block"], as_index=False).throughput_mbps.mean()
    fig, (ax, legend) = c.plt.subplots(2, 1, figsize=(6.3, 3.4), layout="constrained",
                                     gridspec_kw={"height_ratios": [4, 1]})
    for arm, name, color, style in zip(c.ARMS, c.NAMES, c.LINE_COLORS, c.STYLES):
        rows = values[values.configuration == arm].sort_values("block")
        ax.plot(rows.block, rows.throughput_mbps, marker="o", ms=2.5, color=color, ls=style, label=name, lw=1.45)
    ax.set(xlabel="Khối liên tiếp (1.000 slot mỗi khối)", ylabel="Thông lượng (Mbps)",
           xticks=range(1, 11), ylim=(0, 190))
    ax.grid(axis="y", alpha=.23)
    legend.set_axis_off()
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 4, 1, 5, 2, 6, 3]
    legend.legend([handles[i] for i in order], [labels[i] for i in order], ncol=4, loc="center", frameon=False, fontsize=8)
    c.save(fig, args, "W00/02a_thong_luong_theo_khoi", values,
           "Equal arithmetic mean of run block throughput: 30 runs per learned configuration, ten per baseline", [source], __file__)


if __name__ == "__main__":
    main()
