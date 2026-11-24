"""Learning figures use only existing train/validation logs, never new rollouts."""

from pathlib import Path
import numpy as np

FIELDS = {
    "learning_curve.png": [
        ("reward_total", "Reward tổng sau phạt", "reward", 1),
        ("throughput_mbps", "Thông lượng", "Mbps", 1),
        ("service_coverage_ues", "UE đạt ít nhất 1 Mbps", "UE/slot", 1),
        ("qos_ues", "UE đạt đủ 4 Mbps", "UE/slot", 1),
        ("jain_instant_qualified", "Jain tức thời dùng trong reward", "Jain", 1),
        ("jain_time_average", "Jain tốc độ trung bình cả lượt", "Jain", 1),
    ],
    "fairness_continuity.png": [
        ("p05_ue_mean_rate_mbps", "Phân vị 5% tốc độ trung bình UE", "Mbps", 1),
        ("min_ue_mean_rate_mbps", "Tốc độ trung bình UE thấp nhất", "Mbps", 1),
        ("rolling_unique_ge1_1min", "UE khác nhau đạt 1 Mbps trong 1 phút", "UE", 1),
        ("rolling_unique_ge1_5min", "UE khác nhau đạt 1 Mbps trong 5 phút", "UE", 1),
        ("longest_zero_gap_slots_max", "Gián đoạn UE không nhận dữ liệu dài nhất", "phút", 5 / 60),
        ("longest_below1_gap_slots_max", "Gián đoạn UE dưới 1 Mbps dài nhất", "phút", 5 / 60),
    ],
    "energy_operation.png": [
        ("consumed_wh", "Năng lượng đội mỗi slot", "Wh/slot", 1),
        ("joules_per_mbit", "Năng lượng trên dữ liệu thực nhận", "J/Mbit", 1),
        ("flight_j", "Bay và giữ vị trí", "Wh/slot", 1 / 3600),
        ("radio_j", "Mạch vô tuyến và PA", "Wh/slot", 1 / 3600),
        ("effective_return_fraction", "Returning", "% UAV-slot", 100),
        ("effective_swap_fraction", "Đang thay pin", "% UAV-slot", 100),
    ],
    "role_collision.png": [
        ("active_access", "Access active", "UAV/slot", 1),
        ("active_relay", "Relay active", "UAV/slot", 1),
        ("active_hybrid", "Vai trò kết hợp active", "UAV/slot", 1),
        ("forwarding_total", "UAV có chuyển tiếp dữ liệu", "UAV/slot", 1),
        ("blocked_per_policy_move", "Yêu cầu di chuyển active bị chặn", "%", 100),
        ("longest_blocked_run_slots", "Di chuyển bị chặn liên tiếp dài nhất", "slot", 1),
        ("zero_service_slot_fraction", "Toàn mạng không có dữ liệu", "% slot", 100),
        ("role_switches", "Thay đổi vai trò khi active", "UAV/slot", 1),
    ],
    "training_diagnostics.png": [
        ("mean_loss", "TD loss", "Smooth L1 loss", 1),
        ("learning_rate", "Learning rate", "LR", 1),
        ("epsilon", "Xác suất khám phá", "epsilon", 1),
    ],
}


def value(row, key):
    if key == "radio_j":
        return sum(row.get(k, 0) for k in ["circuit_j", "access_pa_j", "backhaul_pa_j"])
    v = row.get(key)
    return float(v) if v is not None else np.nan


def val_series(rows, key):
    steps = sorted({r["step"] for r in rows})
    result = []
    low = []
    high = []
    for step in steps:
        vals = np.asarray([value(r, key) for r in rows if r["step"] == step])
        vals = vals[np.isfinite(vals)]
        result.append(float(vals.mean()) if len(vals) else np.nan)
        low.append(float(vals.min()) if len(vals) else np.nan)
        high.append(float(vals.max()) if len(vals) else np.nan)
    return np.asarray(steps), np.asarray(result), np.asarray(low), np.asarray(high)


def plot_run(output, saved):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    p = saved["protocol"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train, validation = saved["training"], saved["validation"]
    for filename, fields in FIELDS.items():
        shape = (1, 3) if len(fields) <= 3 else ((2, 4) if len(fields) > 6 else (2, 3))
        fig, axs = plt.subplots(
            *shape,
            figsize=(16 if shape[1] == 4 else 14, 4 if shape[0] == 1 else 8),
            layout="constrained",
        )
        for ax, (key, title, unit, scale) in zip(np.asarray(axs).flat, fields):
            x = np.asarray([r["step"] for r in train])
            y = np.asarray([value(r, key) for r in train]) * scale
            if np.isfinite(y).any():
                ax.plot(x / 1000, y, alpha=0.3, lw=0.8, label="Training episode")
                if key == "mean_loss":
                    smooth = np.asarray(
                        [
                            np.nanmean(y[max(0, i - 4) : i + 1])
                            if np.isfinite(y[max(0, i - 4) : i + 1]).any()
                            else np.nan
                            for i in range(len(y))
                        ]
                    )
                    ax.plot(
                        x / 1000,
                        smooth,
                        color="#2474a8",
                        lw=1.5,
                        label="Training, trung bình 5 episode",
                    )
            steps, mean, lo, hi = val_series(validation, key)
            if np.isfinite(mean).any():
                ax.plot(
                    steps / 1000,
                    mean * scale,
                    color="#dd7720",
                    marker="o",
                    ms=2,
                    lw=1.4,
                    label="Validation, trung bình",
                )
                ax.fill_between(steps / 1000, lo * scale, hi * scale, alpha=0.12, color="#dd7720")
                best = saved.get("best_by_metric", {}).get("reward_total", {}).get("step")
                if best in steps:
                    idx = int(np.flatnonzero(steps == best)[0])
                    if np.isfinite(mean[idx]):
                        ax.scatter(
                            best / 1000,
                            mean[idx] * scale,
                            marker="*",
                            color="#8b3a00",
                            s=65,
                            zorder=5,
                        )
            if not np.isfinite(y).any() and not np.isfinite(mean).any():
                ax.text(
                    0.5, 0.5, "Chưa có dữ liệu", ha="center", va="center", transform=ax.transAxes
                )
            for boundary in p["stage_ends"][:-1]:
                ax.axvline(boundary / 1000, color="grey", alpha=0.3, ls="--", lw=0.7)
            ax.set(title=title, xlabel="Bước môi trường (nghìn)", ylabel=unit)
            ax.grid(alpha=0.2)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7, loc="best")
        fig.suptitle(
            f"{p['arm']}: {saved['step']:,}/{p['total_steps']:,} bước, Fair tức thời, {saved['status']}\nDải màu: min–max các seed validation; dấu sao: best theo reward tổng.",
            fontsize=12,
        )
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    for key, title in [
        ("reward_throughput", "Thưởng thông lượng"),
        ("reward_fairness", "Thưởng Jain"),
        ("reward_energy", "Phạt năng lượng"),
        ("reward_collision", "Phạt bị chặn"),
    ]:
        steps, mean, _, _ = val_series(validation, key)
        ax.plot(steps / 1000, mean, marker="o", ms=2, label=title)
    ax.axhline(0, color="grey", lw=0.7)
    ax.set(
        xlabel="Bước môi trường (nghìn)",
        ylabel="Giá trị sau nhân trọng số",
        title="Các thành phần của reward trên validation",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.savefig(output / "reward_components.png", dpi=150)
    plt.close(fig)
