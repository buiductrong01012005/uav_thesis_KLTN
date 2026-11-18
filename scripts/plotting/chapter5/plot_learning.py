"""Figure 5.1: validation reward, with sample SD across three training seeds."""
from . import common as c


def main(argv=None):
    args = c.options(__doc__, argv)
    source = args.training / "learning_mean_sd.csv"
    # Other reported metrics can have missing SD; only reward_total is plotted.
    frame = c.pd.read_csv(source)
    frame = frame[(frame.metric == "reward_total") & frame.arm.isin(c.ARMS[:4])].copy()
    fig, ax = c.plt.subplots(figsize=(6.3, 2.95), layout="constrained")
    for arm, name, color, style in zip(c.ARMS[:4], c.NAMES, c.LINE_COLORS, c.STYLES):
        rows = frame[frame.arm == arm].sort_values("step")
        if (len(rows) != 67 or rows.step.duplicated().any() or rows.step.iloc[-1] != 399600
                or not (rows.training_seeds == 3).all()
                or not c.np.isfinite(rows[["step", "mean", "sample_sd"]].to_numpy()).all()):
            raise ValueError("Expected 67 complete three-seed reward validations per arm")
        x, y, sd = rows.step.to_numpy()/1000, rows["mean"].to_numpy(), rows.sample_sd.to_numpy()
        ax.plot(x, y, color=color, ls=style, lw=1.75, label=name)
        ax.fill_between(x, y-sd, y+sd, color=color, alpha=.13, lw=0)
    ax.axvline(12, color="#3b3b3b", lw=.8, ls=(0, (3, 3)))
    ax.text(16, .96, "Nhiệm vụ đầy đủ", transform=ax.get_xaxis_transform(), va="top", fontsize=8)
    ax.legend(loc="lower right", frameon=False, ncol=2, fontsize=8, columnspacing=1.1)
    ax.set(xlabel="Số bước môi trường (nghìn)", ylabel="Phần thưởng kiểm định",
           xticks=range(0, 401, 50), xlim=(-3, 403))
    ax.grid(axis="y", alpha=.24)
    c.save(fig, args, "learning", frame, "Validation reward mean +/- sample SD across three training seeds; each seed averages two validation scenarios", [source], __file__)


if __name__ == "__main__":
    main()
