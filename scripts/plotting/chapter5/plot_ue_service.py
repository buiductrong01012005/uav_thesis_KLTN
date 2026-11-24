"""Figure 5.4: UE service per slot and unique service in a five-minute window."""

from . import common as c


def main(argv=None):
    args = c.options(__doc__, argv)
    metrics = ["service_coverage_ues", "rolling_unique_ge1_5min"]
    values = c.per_test_seed(args.input, metrics)
    fig, axes = c.plt.subplots(1, 2, figsize=(6.3, 3.3), sharey=True, layout="constrained")
    for ax, metric, title in zip(axes, metrics, ["(a) Trong một slot", "(b) Trong cửa sổ 5 phút"]):
        c.seed_panel(
            ax,
            values,
            metric,
            "Số UE đạt ít nhất 1 Mbps",
            ticks=range(0, 91, 15),
            label_at_point=True,
        )
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
    axes[0].set_yticks(c.Y, c.NAMES)
    axes[0].set_ylim(c.Y[-1] + 0.5, -0.5)
    c.save(
        fig,
        args,
        "W00/04_ue_1mbps_slot_va_5phut",
        values,
        "Mean across training replicates within each test seed; dots are ten test seeds, large dot is their mean, line is min/max. Five-minute metric counts unique UEs served at least once, not continuously",
        [args.input / "per_run.csv"],
        __file__,
    )


if __name__ == "__main__":
    main()
