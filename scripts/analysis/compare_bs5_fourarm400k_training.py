"""Overlay four training/validation outputs. No test or model reselection."""

from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import zipfile
import numpy as np

VERSION = "bs5_fourarm400k_v1"
ARMS = ["C-N", "C-R", "C-H", "H-only"]


def digest(p):
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()


def parse_csv(data):
    rows = []
    for raw in csv.DictReader(io.StringIO(data.decode())):
        row = {}
        for k, v in raw.items():
            if v in ("", None):
                row[k] = None
            else:
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
        rows.append(row)
    return rows


def collect(input_dir):
    from scripts.analysis.fourarm400k_multiseed import normalized, PLAN

    total_steps = PLAN["total_training_steps"]
    runs = {}
    provenance = []

    def consume(read, prefix, origin):
        name = (prefix + "/" if prefix else "") + "protocol.json"
        p = json.loads(read(name))
        if p.get("version") != VERSION or p.get("diagnostic"):
            return
        if p["arm"] not in ARMS or p["total_steps"] != total_steps:
            raise ValueError("Unknown training arm/budget")
        common = {
            k: v
            for k, v in p.items()
            if k
            not in {"arm", "policy_actions", "runtime", "torch_version", "comparison_fingerprint"}
        }
        if digest(common) != p["comparison_fingerprint"]:
            raise ValueError("Wrong comparison fingerprint")
        prefix = prefix + "/" if prefix else ""
        try:
            status = json.loads(read(prefix + "status.json"))
            train = parse_csv(read(prefix + "training.csv"))
            validation = parse_csv(read(prefix + "validation.csv"))
        except (KeyError, FileNotFoundError):
            return
        if (
            status.get("training_env_version") != VERSION
            or status.get("arm") != p["arm"]
            or status.get("target_steps") != total_steps
            or not 0 <= status["committed_training_steps"] <= total_steps
        ):
            raise ValueError("Wrong output status")
        seen = set()
        for row in validation:
            key = (row["step"], row["seed"])
            if (
                key in seen
                or row["seed"] not in p["validation_seeds"]
                or row["step"] not in p["evaluation_steps"]
                or row["step"] > status["committed_training_steps"]
            ):
                raise ValueError("Invalid validation rows")
            seen.add(key)
        if status["status"] == "complete":
            expected = {(s, seed) for s in p["evaluation_steps"] for seed in p["validation_seeds"]}
            if status["committed_training_steps"] != total_steps or seen != expected:
                raise ValueError("Complete status requires every scheduled validation row")
        best = status.get("best_by_metric", {}).get("reward_total")
        if best:
            selected = [r for r in validation if r["step"] == best["step"]]
            if len(selected) != len(p["validation_seeds"]) or not np.isclose(
                np.mean([r["reward_total"] for r in selected]),
                best["score"],
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError("Best reward does not match validation log")
        data = dict(protocol=p, status=status, training=train, validation=validation)
        if p["arm"] in runs and runs[p["arm"]] != data:
            raise ValueError(
                "Multiple different outputs for " + p["arm"] + "; attach one version per arm"
            )
        runs[p["arm"]] = data
        provenance.append(dict(origin=origin, arm=p["arm"]))

    for path in sorted(Path(input_dir).rglob("*.zip")):
        with zipfile.ZipFile(path) as z:
            relevant = []
            for name in z.namelist():
                if name == "protocol.json" or name.endswith("/protocol.json"):
                    p = json.loads(z.read(name))
                    if p.get("version") == VERSION and not p.get("diagnostic"):
                        relevant.append(name)
            if not relevant:
                continue
            if z.testzip() is not None:
                raise ValueError("Corrupt ZIP: " + str(path))
            for name in relevant:
                consume(z.read, name.rsplit("/", 1)[0] if "/" in name else "", str(path))
    for path in sorted(Path(input_dir).rglob("protocol.json")):
        p = json.loads(path.read_text())
        if p.get("version") != VERSION or p.get("diagnostic"):
            continue

        def read(name):
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe result path")
            return (path.parent / name).read_bytes()

        consume(read, "", str(path.parent))
    if not runs:
        raise ValueError("Attach four-arm400k REVIEW ZIPs or their extracted directories")
    if (
        len({digest(normalized(r["protocol"])) for r in runs.values()}) != 1
        or len({r["protocol"]["training_seed"] for r in runs.values()}) != 1
    ):
        raise ValueError("Training protocols,weights,seeds,source or budgets differ")
    return runs, provenance


def compare(input_dir, output):
    from scripts.analysis.plot_bs5_fourarm400k import val_series
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs, provenance = collect(input_dir)
    colors = {"C-N": "#555555", "C-R": "#2365a0", "C-H": "#cc6b28", "H-only": "#449879"}
    groups = {
        "validation_overlays.png": [
            ("reward_total", "Reward tổng", 1),
            ("throughput_mbps", "Thông lượng (Mbps)", 1),
            ("qos_ues", "UE đạt 4 Mbps mỗi slot", 1),
            ("service_coverage_ues", "UE đạt 1 Mbps mỗi slot", 1),
            ("joules_per_mbit", "Năng lượng trên dữ liệu (J/Mbit)", 1),
            ("blocked_per_policy_move", "Yêu cầu di chuyển bị chặn (%)", 100),
        ],
        "fairness_overlays.png": [
            ("jain_time_average", "Jain trên tốc độ trung bình UE", 1),
            ("p05_ue_mean_rate_mbps", "Phân vị 5% tốc độ UE (Mbps)", 1),
            ("rolling_unique_ge1_5min", "UE khác nhau trong 5 phút", 1),
            ("longest_zero_gap_slots_max", "Gián đoạn UE dài nhất (phút)", 5 / 60),
            ("effective_unavailable_fraction", "Returning và thay pin (% UAV-slot)", 100),
            ("zero_service_slot_fraction", "Toàn mạng không phục vụ (% slot)", 100),
        ],
    }
    plt.rcParams["font.family"] = "DejaVu Sans"
    for filename, fields in groups.items():
        fig, axs = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
        for ax, (key, title, scale) in zip(axs.flat, fields):
            for arm in ARMS:
                if arm not in runs:
                    continue
                r = runs[arm]
                steps, mean, lo, hi = val_series(r["validation"], key)
                if np.isfinite(mean).any():
                    ax.plot(steps / 1000, mean * scale, label=arm, color=colors[arm])
                    ax.fill_between(
                        steps / 1000, lo * scale, hi * scale, color=colors[arm], alpha=0.08
                    )
            ax.set(title=title, xlabel="Bước môi trường (nghìn)")
            ax.grid(alpha=0.2)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8)
        fig.suptitle(
            "So sánh validation giữa bốn cấu hình, cùng seed train 551\nDải màu: min–max seed validation. Đây chưa phải kết quả test.",
            fontsize=12,
        )
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    rows = []
    for arm in ARMS:
        if arm not in runs:
            continue
        run = runs[arm]
        status = run["status"]
        best = status["best_by_metric"].get("reward_total")
        row = dict(
            arm=arm,
            committed_steps=status["committed_training_steps"],
            target_steps=run["protocol"]["total_steps"],
            status=status["status"],
        )
        if best:
            selected = [r for r in run["validation"] if r["step"] == best["step"]]
            row.update(selected_step=best["step"], selected_reward=best["score"])
            for k in [
                "throughput_mbps",
                "qos_ues",
                "service_coverage_ues",
                "jain_time_average",
                "p05_ue_mean_rate_mbps",
                "rolling_unique_ge1_5min",
                "longest_zero_gap_slots_max",
                "joules_per_mbit",
                "blocked_per_policy_move",
            ]:
                vals = [r[k] for r in selected if r.get(k) is not None]
                row[k] = float(np.mean(vals)) if vals else None
        rows.append(row)
    with (output / "validation_best_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    complete = len(runs) == 4 and all(
        r["status"]["status"] == "complete"
        and r["status"]["committed_training_steps"] == r["protocol"]["total_steps"]
        for r in runs.values()
    )
    meta = dict(
        status="all_training_complete" if complete else "partial_training",
        present=list(runs),
        missing=[a for a in ARMS if a not in runs],
        provenance=provenance,
        scope="Validation comparison only; checkpoint selection remains the predeclared reward criterion; not test ranking",
    )
    (output / "completeness.json").write_text(json.dumps(meta, indent=2) + "\n")
    text = (
        "# So sánh đường học bốn cấu hình\n\n"
        + (
            "Đã đủ bốn lượt train theo ngân sách cấu hình."
            if complete
            else "Chưa đủ bốn lượt train hoàn tất; không xếp hạng cuối cùng."
        )
        + "\n\nBảng validation_best_summary.csv dùng đúng best theo reward của từng cấu hình. Để kết luận hiệu năng, cần benchmark các checkpoint cố định trên cùng tập test. Không chọn lại model bằng kết quả test. Dải min–max hai seed validation không phải khoảng tin cậy.\n"
    )
    (output / "REPORT_VI.md").write_text(text)
    print(json.dumps(meta, indent=2), flush=True)
    return meta


if __name__ == "__main__":
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    compare(a.input, a.output)
