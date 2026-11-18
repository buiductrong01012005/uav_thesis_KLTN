"""Figure 5.2: the delivered throughput and five losses sum to 360 Mbps."""
from . import common as c

PARTS = [("throughput_mbps", "Đã truyền", "#296978"),
         ("loss_bh_schedule_capacity_mbps", "Thiếu tài nguyên truyền dẫn", "#df9d4b"),
         ("loss_bh_disconnected_mbps", "Không có tuyến về BS", "#b85a70"),
         ("loss_access_radio_mbps", "Giới hạn vô tuyến truy nhập", "#7a98a8"),
         ("loss_association_mbps", "Chưa ghép nối", "#aab9c8"),
         ("loss_outside_access_mbps", "Ngoài vùng truy nhập", "#dce2e7")]


def main(argv=None):
    args = c.options(__doc__, argv)
    source = args.input / "aggregate.csv"
    columns = [p[0] for p in PARTS]
    values = c.read(source, ["configuration", *columns]).set_index("configuration").loc[c.ARMS, columns].copy()
    if len(values) != 7 or not c.np.allclose(values[columns].sum(axis=1), 360, atol=1e-5):
        raise ValueError("Demand accounting must total 360 Mbps for each configuration")
    fig, (ax, legend) = c.plt.subplots(2, 1, figsize=(6.3, 4.1), layout="constrained",
                                     gridspec_kw={"height_ratios": [4, 1]})
    left = c.np.zeros(7)
    for key, label, color in PARTS:
        x = values[key].to_numpy()
        ax.barh(c.Y, x, left=left, color=color, label=label, height=.63)
        for y, start, width in zip(c.Y, left, x):
            if width >= 24:
                ax.text(start+width/2, y, c.number(width, 1), ha="center", va="center", fontsize=7.5,
                        color="white" if key == "throughput_mbps" else "#222222")
        left += x
    ax.set_yticks(c.Y, c.NAMES)
    ax.set(ylim=(c.Y[-1]+.5, -.5), xlim=(0, 360), xticks=range(0, 361, 60), xlabel="Mbps")
    ax.grid(axis="x", alpha=.22)
    ax.set_axisbelow(True)
    legend.set_axis_off()
    legend.legend(*ax.get_legend_handles_labels(), loc="center", ncol=3, frameon=False, fontsize=7)
    c.save(fig, args, "W00/02b_phan_ra_nhu_cau", values.reset_index(),
           "Published per-configuration slot-average demand accounting; no recomputation of losses", [source], __file__)


if __name__ == "__main__":
    main()
