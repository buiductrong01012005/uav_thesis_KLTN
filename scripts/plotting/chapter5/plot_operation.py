"""Figure 5.5: blocked policy moves and per-run energy per delivered Mbit."""

from . import common as c


def main(argv=None):
    args = c.options(__doc__, argv)
    metrics = ["blocked_per_policy_move", "joules_per_mbit"]
    values = c.per_test_seed(args.input, metrics)
    fig, axes = c.plt.subplots(1, 2, figsize=(6.3, 3.45), sharey=True, layout="constrained")
    c.seed_panel(
        axes[0],
        values,
        metrics[0],
        "Yêu cầu di chuyển bị chặn (%)",
        scale=100,
        xmax=17.5,
        ticks=[0, 4, 8, 12],
    )
    c.seed_panel(axes[1], values, metrics[1], "Điện năng trên dữ liệu (J/Mbit, log)", log=True)
    for ax, title in zip(axes, ["(a)", "(b)"]):
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
    axes[0].set_yticks(c.Y, c.NAMES)
    axes[0].set_ylim(c.Y[-1] + 0.5, -0.5)
    c.save(
        fig,
        args,
        "W00/03_benchmark_van_hanh",
        values,
        "Arithmetic mean of per-run ratios across training seeds within each test seed; ten seed dots, mean and min/max. Not the pooled J/Mbit ratio",
        [args.input / "per_run.csv"],
        __file__,
    )


if __name__ == "__main__":
    main()
