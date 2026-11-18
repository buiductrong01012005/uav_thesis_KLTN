"""Figure 5.6: throughput versus per-run J/Mbit and whole-fleet actor time."""
from . import common as c


def main(argv=None):
    args = c.options(__doc__, argv)
    metrics = ["throughput_mbps", "joules_per_mbit", "actor_ms"]
    values = c.run_values(args.input, metrics).groupby("configuration")[metrics].mean().loc[c.ARMS].reset_index()
    fig, (axes, legend) = c.plt.subplots(2, 1, figsize=(6.3, 3.4), layout="constrained",
                                       gridspec_kw={"height_ratios": [4, 1]})
    # Replace the upper placeholder with two independent cost axes.
    grid = axes.get_subplotspec().subgridspec(1, 2, wspace=.15)
    axes.remove()
    left, right = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    for ax, metric in zip([left, right], ["joules_per_mbit", "actor_ms"]):
        for arm, name, color in zip(c.ARMS, c.NAMES, c.COLORS):
            row = values[values.configuration == arm].iloc[0]
            ax.scatter(row[metric], row.throughput_mbps, color=color, s=30,
                       marker="o" if arm in c.ARMS[:4] else "s",
                       label=name + ("*" if arm == "DYNAMIC_AR" else ""))
        ax.set_ylim(0, 190)
        ax.grid(alpha=.23)
    c.log_axis(left, [10, 30, 100, 300])
    c.log_axis(right, [.1, 1, 10, 100, 1000])
    left.set(xlabel="Điện năng trên dữ liệu (J/Mbit, log)", ylabel="Thông lượng (Mbps)", xlim=(9, 450))
    right.set(xlabel="Thời gian quyết định cho cả đội (ms, log)", xlim=(.08, 1600))
    right.tick_params(labelleft=False)
    left.set_title("(a)", loc="left", fontsize=9, fontweight="bold")
    right.set_title("(b)", loc="left", fontsize=9, fontweight="bold")
    legend.set_axis_off()
    handles, labels = left.get_legend_handles_labels()
    order = [0, 4, 1, 5, 2, 6, 3]
    legend.legend([handles[i] for i in order], [labels[i] for i in order], ncol=4, loc="center", frameon=False, fontsize=8)
    c.save(fig, args, "W00/05_thong_luong_chi_phi", values,
           "Per-configuration arithmetic mean of per-run throughput, J/Mbit and whole-fleet actor_ms; 30 learned runs or ten baseline runs. Heuristic-AR* timings mix five Intel Xeon 2.20 GHz and five AMD EPYC 7B12 scenarios; other configurations use Intel Xeon", [args.input / "per_run.csv"], __file__)


if __name__ == "__main__":
    main()
