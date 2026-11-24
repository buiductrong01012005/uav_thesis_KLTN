"""Verify raw physical/UE evidence and merge three heuristic test arms."""

from pathlib import Path
from itertools import combinations
import argparse
import csv
import hashlib
import io
import json
import zipfile
import numpy as np
from scipy.stats import t

VERSION = "fourarm400k_heuristic_benchmark_v1"
LABELS = ("RANDOM", "DYNAMIC_AR", "HEURISTIC_HO")
ARMS = {"RANDOM": "C-R", "DYNAMIC_AR": "C-R", "HEURISTIC_HO": "H-only"}
METRICS = [
    "throughput_mbps",
    "qos_ues",
    "service_coverage_ues",
    "positive_ues",
    "jain_ema",
    "jain_instant_qualified",
    "jain_time_average",
    "p05_ue_mean_rate_mbps",
    "min_ue_mean_rate_mbps",
    "rolling_unique_ge1_1min",
    "rolling_unique_ge1_5min",
    "longest_zero_gap_slots_max",
    "longest_below1_gap_slots_max",
    "zero_service_slot_fraction",
    "energy_consumed_wh",
    "joules_per_mbit",
    "blocked_per_policy_move",
    "longest_consecutive_blocked_slots",
    "effective_unavailable_uav_slot_fraction",
    "effective_return_uav_slot_fraction",
    "effective_swap_uav_slot_fraction",
    "saturated_backhaul_slot_fraction",
    "unique_ge1_ues",
    "unique_qos_ues",
    "arrival_below20_events",
    "min_arrival_soc_percent",
    "actor_ms",
    "actor_ms_p95",
    "actor_ms_max",
    "actor_over_5s_slot_fraction",
    "energy_shortfall_total_j",
    "reward_total",
    "loss_outside_access_mbps",
    "loss_association_mbps",
    "loss_access_radio_mbps",
    "loss_bh_disconnected_mbps",
    "loss_bh_schedule_capacity_mbps",
    "active_access",
    "active_relay",
    "active_hybrid",
    "forwarding_total",
    "mean_return_planned_m",
    "mean_return_actual_m",
    "mean_return_duration_slots",
]


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def validate_raw(read, prefix, row, p):
    h = p["horizon"]
    commit = json.loads(read(prefix + "/COMMIT.json"))
    required = {
        "slots.csv",
        "ue_summary.csv",
        "trace.npz",
        "energy.npz",
        "energy_events.json",
        "collision.npz",
        "collision_events.json",
        "cycles.json",
        "network_samples.jsonl",
        "summary.json",
    }
    if set(commit["files_sha256"]) != required:
        raise ValueError("Incomplete scientific seed commit")
    for name, value in commit["files_sha256"].items():
        if (
            Path(name).name != name
            or hashlib.sha256(read(prefix + "/" + name)).hexdigest() != value
        ):
            raise ValueError("Corrupt seed file: " + name)
    with np.load(io.BytesIO(read(prefix + "/trace.npz")), allow_pickle=False) as z:
        raw = z["ue_rates"]
        pos = z["ue_positions"]
        ema = z["ema_rates"]
        if (
            raw.shape != (h, 90)
            or not np.isfinite(raw).all()
            or raw.min() < -1e-9
            or raw.max() > 4 + 1e-6
        ):
            raise ValueError("Invalid UE rate trace")
        r = np.maximum(raw, 0.0)
        checks = {
            "throughput_mbps": r.sum(1).mean(),
            "qos_ues": (r >= 4 - 1e-6).sum(1).mean(),
            "service_coverage_ues": (r >= 1 - 1e-9).sum(1).mean(),
            "jain_time_average": float(r.mean(0).sum() ** 2 / (90 * np.dot(r.mean(0), r.mean(0))))
            if r.any()
            else 0.0,
        }
        previous = np.vstack([np.zeros((1, 90)), ema[:-1]])
        if not np.allclose(previous * (59 / 60) + r / 60, ema, rtol=1e-11, atol=1e-11):
            raise ValueError("EMA recurrence mismatch")

        def jains(values):
            maxima = values.max(1)
            scaled = np.divide(
                values, maxima[:, None], out=np.zeros_like(values), where=maxima[:, None] > 1e-9
            )
            den = 90 * np.square(scaled).sum(1)
            return np.divide(scaled.sum(1) ** 2, den, out=np.zeros(len(values)), where=den > 0)

        jf = jains(ema)
        ji = jains(np.where(r >= 1 - 1e-9, r, 0.0))
        chain = "00" * 32
        for values in pos:
            chain = hashlib.sha256(
                bytes.fromhex(chain) + np.asarray(values, dtype="<f8").tobytes(order="C")
            ).hexdigest()
        if chain != row["ue_trace_sha256"]:
            raise ValueError("UE trace hash mismatch")
    with np.load(io.BytesIO(read(prefix + "/energy.npz")), allow_pickle=False) as z:
        con = z["consumed_j"]
        balance = (
            z["energy_before_j"] + z["inserted_j"] - z["removed_j"] - con - z["energy_after_j"]
        )
        if con.shape != (h, 5) or not np.isfinite(con).all() or np.abs(balance).max() > 1e-5:
            raise ValueError("Energy balance failed")
        checks["energy_consumed_wh"] = con.sum() / 3600
        checks["energy_shortfall_total_j"] = z["shortfall_j"].sum()
        consumed = con.sum(1)
        slot_records = csv.DictReader(io.StringIO(read(prefix + "/slots.csv").decode()))
        fields = [
            "slot",
            "throughput_mbps",
            "qos_ues",
            "service_coverage_ues",
            "jain_ema",
            "jain_instant_qualified",
            "reward_total",
            "active_blocked_uavs",
        ]
        logged = np.asarray([[float(rec[k]) for k in fields] for rec in slot_records])
        if logged.shape != (h, len(fields)) or not np.isfinite(logged).all():
            raise ValueError("Invalid per-slot CSV")
        np.testing.assert_array_equal(logged[:, 0], np.arange(1, h + 1))
        np.testing.assert_allclose(logged[:, 1], r.sum(1), atol=1e-7, rtol=1e-9)
        np.testing.assert_allclose(logged[:, 2], (r >= 4 - 1e-6).sum(1), atol=0, rtol=0)
        np.testing.assert_allclose(logged[:, 3], (r >= 1 - 1e-9).sum(1), atol=0, rtol=0)
        np.testing.assert_allclose(logged[:, 4], jf, atol=1e-10, rtol=1e-9)
        np.testing.assert_allclose(logged[:, 5], ji, atol=1e-10, rtol=1e-9)
        tp = p["training_protocol"]
        ec = tp["energy_config"]
        penalty = tp["collision_config"]["collision_weight"] * logged[:, 7] / 5
        energy_penalty = ec["energy_weight"] * consumed / (5 * ec["reference_w"] * 5)
        shared_base = tp["reward_weights"]["throughput"] * r.sum(1) / 360 - energy_penalty - penalty
        reward_ema = shared_base + tp["reward_weights"]["fairness"] * jf
        reward_instant = shared_base + tp["reward_weights"]["fairness"] * ji
        np.testing.assert_allclose(
            logged[:, 6],
            reward_ema if row["label"] == "EMA" else reward_instant,
            atol=1e-9,
            rtol=1e-9,
        )
        checks.update(
            jain_ema=jf.mean(),
            jain_instant_qualified=ji.mean(),
            reward_under_ema=reward_ema.mean(),
            reward_under_instant=reward_instant.mean(),
        )
    for k, value in checks.items():
        if not np.isclose(row[k], value, rtol=1e-9, atol=1e-7):
            raise ValueError("Summary/raw mismatch: " + k)


def scan_inputs(input_dir, allow_diagnostic=False):
    results = {}
    protocols = {}
    provenance = []

    def consume(p, read, prefix, origin):
        if p.get("version") != VERSION:
            return
        if p.get("diagnostic") and not allow_diagnostic:
            raise ValueError("Diagnostic results cannot be merged as test")
        if p["label"] not in LABELS or p["arm"] != ARMS[p["label"]] or p["part"] not in (1, 2):
            raise ValueError("Unknown worker")
        keys = [
            "version",
            "config",
            "training_protocol",
            "source_sha256",
            "environment_contract",
            "environment_fingerprint",
        ]
        common = {k: p[k] for k in keys}
        if (
            digest(common) != p["comparison_fingerprint"]
            or digest(p["environment_contract"]) != p["environment_fingerprint"]
        ):
            raise ValueError("Invalid comparison/environment fingerprint")
        if p["config"]["test_seeds"] != list(range(930201, 930211)):
            raise ValueError("Wrong test seeds")
        expected = (
            p["config"]["diagnostic_seeds"]
            if p["diagnostic"]
            else p["config"]["test_seeds"][(p["part"] - 1) * 5 : p["part"] * 5]
        )
        if (
            p["horizon"] != (p["config"]["diagnostic_horizon"] if p["diagnostic"] else 10000)
            or not p["target_seeds"]
            or len(set(p["target_seeds"])) != len(p["target_seeds"])
            or not set(p["target_seeds"]) <= set(expected)
        ):
            raise ValueError("Wrong worker partition or horizon")
        identity = (p["label"], p["part"], tuple(p["target_seeds"]))
        if identity in protocols and protocols[identity] != p:
            raise ValueError("Conflicting worker protocols")
        protocols[identity] = p
        for seed in p["target_seeds"]:
            base = (prefix + "/" if prefix else "") + f"seed{seed}"
            try:
                row = json.loads(read(base + "/summary.json"))
            except (KeyError, FileNotFoundError):
                continue
            if not row["complete"]:
                continue
            if (
                row["diagnostic"] != p["diagnostic"]
                or row["label"] != p["label"]
                or row["arm"] != p["arm"]
                or row["seed"] != seed
                or row["slots"] != p["horizon"]
                or row["target_slots"] != p["horizon"]
                or row["comparison_fingerprint"] != p["comparison_fingerprint"]
                or row["environment_fingerprint"] != p["environment_fingerprint"]
            ):
                raise ValueError("Summary/worker mismatch")
            commit = json.loads(read(base + "/COMMIT.json"))
            if commit["protocol_sha256"] != digest(p):
                raise ValueError("Seed commit/worker mismatch")
            validate_raw(read, base, row, p)
            key = (p["label"], seed)
            if key in results and results[key] != row:
                raise ValueError("Conflicting complete seed: " + str(key))
            results[key] = row
        provenance.append(dict(origin=origin, worker=[p["label"], p["part"], p["target_seeds"]]))

    for file in sorted(Path(input_dir).rglob("*.zip")):
        with zipfile.ZipFile(file) as z:
            relevant = []
            for name in z.namelist():
                if name == "protocol.json" or name.endswith("/protocol.json"):
                    p = json.loads(z.read(name))
                    if p.get("version") == VERSION:
                        relevant.append((name, p))
            if not relevant:
                continue
            if z.testzip() is not None:
                raise ValueError("Corrupt ZIP: " + str(file))
            for name, p in relevant:
                consume(
                    p,
                    z.read,
                    name.rsplit("/", 1)[0] if "/" in name else "",
                    dict(file=str(file), sha256=sha(file), kind="zip"),
                )
    for file in sorted(Path(input_dir).rglob("protocol.json")):
        p = json.loads(file.read_text())
        if p.get("version") != VERSION:
            continue
        root = file.parent

        def read(relative):
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("Unsafe result path")
            return (root / relative).read_bytes()

        consume(p, read, "", dict(directory=str(root), kind="extracted"))
    if not protocols:
        raise ValueError("Attach the new heuristic REVIEW ZIPs or extracted folders")
    if (
        len({p["comparison_fingerprint"] for p in protocols.values()}) != 1
        or len({p["diagnostic"] for p in protocols.values()}) != 1
    ):
        raise ValueError("Different scientific protocols or diagnostic/test mixture")
    p = next(iter(protocols.values()))
    for seed in p["config"]["diagnostic_seeds"] if p["diagnostic"] else p["config"]["test_seeds"]:
        hashes = {
            results[label, seed]["ue_trace_sha256"] for label in LABELS if (label, seed) in results
        }
        if len(hashes) > 1:
            raise ValueError("Unpaired UE traces: " + str(seed))
    return results, p, provenance


def comparison(results, seeds):
    aggregate = []
    paired = []
    for label in LABELS:
        rows = [results[label, s] for s in seeds]
        rec = dict(label=label, seeds=len(rows))
        for key in METRICS:
            values = [r[key] for r in rows if r.get(key) is not None]
            rec[key] = float(np.mean(values)) if values else None
        energy = sum(r["energy_consumed_wh"] * 3600 for r in rows)
        data = sum(r["delivered_mbit"] for r in rows)
        requests = sum(r["policy_move_requests_total"] for r in rows)
        rec["joules_per_mbit_pooled"] = energy / data if data else None
        rec["blocked_fraction_pooled"] = (
            sum(r["active_blocked_total"] for r in rows) / requests if requests else None
        )
        aggregate.append(rec)
    for a, b in combinations(LABELS, 2):
        for key in METRICS:
            pairs = [
                (results[a, s][key], results[b, s][key])
                for s in seeds
                if results[a, s].get(key) is not None and results[b, s].get(key) is not None
            ]
            if not pairs:
                continue
            aa, bb = np.asarray(pairs).T
            delta = bb - aa
            half = (
                float(t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta)))
                if len(delta) > 1
                else None
            )
            paired.append(
                dict(
                    metric=key,
                    reference=a,
                    comparison=b,
                    paired_seeds=len(delta),
                    reference_mean=float(aa.mean()),
                    comparison_mean=float(bb.mean()),
                    difference=float(delta.mean()),
                    ci95_low=float(delta.mean() - half) if half is not None else None,
                    ci95_high=float(delta.mean() + half) if half is not None else None,
                    relative_change_percent=float((bb.mean() / aa.mean() - 1) * 100)
                    if aa.mean() != 0
                    else None,
                    ci_scope="Paired t interval across scenario seeds; descriptive secondary metrics, no multiple-comparison adjustment",
                )
            )
    return aggregate, paired


def report(input_dir, output, allow_diagnostic=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results, p, provenance = scan_inputs(input_dir, allow_diagnostic)
    seeds = p["config"]["diagnostic_seeds"] if p["diagnostic"] else p["config"]["test_seeds"]
    paired = [s for s in seeds if all((label, s) in results for label in LABELS)]
    missing = {label: [s for s in seeds if (label, s) not in results] for label in LABELS}
    csv_write(output / "per_seed.csv", [results[k] for k in sorted(results)])
    complete = len(paired) == len(seeds)
    meta = dict(
        status=("diagnostic_only" if p["diagnostic"] else "complete") if complete else "partial",
        paired_seeds=paired,
        missing=missing,
        comparison_fingerprint=p["comparison_fingerprint"],
        environment_fingerprint=p["environment_fingerprint"],
        provenance=provenance,
        scope=p["config"]["scope"],
    )
    (output / "completeness.json").write_text(json.dumps(meta, indent=2) + "\n")
    generated = [
        "aggregate.csv",
        "paired_comparison.csv",
        "paired_metrics.png",
        "service_temporal.png",
    ]
    if not complete:
        for name in generated:
            (output / name).unlink(missing_ok=True)
        text = f"# Benchmark heuristic chưa hoàn tất\n\nCó {len(paired)}/{len(seeds)} seed ghép cặp đủ ba cấu hình. Thiếu: {missing}. Chưa tổng hợp kết luận cuối cùng.\n"
    else:
        aggregate, stats = comparison(results, seeds)
        csv_write(output / "aggregate.csv", aggregate)
        csv_write(output / "paired_comparison.csv", stats)
        fields = [
            ("throughput_mbps", "Thông lượng (Mbps)"),
            ("qos_ues", "UE đạt 4 Mbps mỗi slot"),
            ("rolling_unique_ge1_5min", "UE khác nhau trong 5 phút"),
            ("jain_time_average", "Jain trên tốc độ trung bình UE"),
            ("energy_consumed_wh", "Năng lượng (Wh)"),
            ("joules_per_mbit_pooled", "Năng lượng trên dữ liệu (J/Mbit)"),
            ("blocked_fraction_pooled", "Tỷ lệ yêu cầu di chuyển bị chặn"),
            ("actor_ms", "Thời gian quyết định trung bình (ms)"),
            ("actor_ms_p95", "Thời gian quyết định p95 (ms)"),
            ("actor_over_5s_slot_fraction", "Tỷ lệ quyết định vượt slot 5 giây"),
        ]
        lines = [
            "# Benchmark ba baseline trong môi trường FourArm400k",
            "",
            "Đây là kết quả heuristic; chưa có DRL. Không dùng kết quả test để đổi tiêu chí chọn checkpoint hoặc điều chỉnh controller.",
            "",
            "| Chỉ số | Random | Dynamic A/R | Heuristic vai trò kết hợp |",
            "|---|---:|---:|---:|",
        ]
        for key, name in fields:
            values = [
                f"{r[key]:.6f}" if r.get(key) is not None else "không xác định" for r in aggregate
            ]
            lines.append("| " + name + " | " + " | ".join(values) + " |")
        lines += [
            "",
            "Các heuristic giữ nguyên hàm chấm điểm dự đoán T/C/F cũ với trọng số bằng nhau. Hàm thưởng đánh giá thực tế dùng 0,4 thông lượng + 0,6 Fair tức thời − phạt năng lượng − phạt bị chặn, giống DRL. Chưa gọi đây là heuristic tối ưu năng lượng.",
            "",
            "Thời gian quyết định nhỏ hơn slot 5 giây không đủ để kết luận heuristic không chạy thời gian thực. So sánh thời gian với DRL sau này cần cùng điều kiện CPU. Khoảng tin cậy chỉ phản ánh seed kịch bản, các phép so sánh phụ chưa điều chỉnh cho nhiều chỉ số.",
        ]
        if p["diagnostic"]:
            lines.insert(1, "**CHỈ LÀ DIAGNOSTIC, KHÔNG PHẢI KẾT QUẢ TEST.**")
        text = "\n".join(lines) + "\n"
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        groups = {
            "paired_metrics.png": [
                ("throughput_mbps", "Thông lượng (Mbps)", 1),
                ("joules_per_mbit", "J/Mbit", 1),
                ("blocked_per_policy_move", "Yêu cầu di chuyển bị chặn (%)", 100),
            ],
            "service_temporal.png": [
                ("qos_ues", "UE đạt 4 Mbps mỗi slot", 1),
                ("rolling_unique_ge1_5min", "UE khác nhau trong 5 phút", 1),
                ("jain_time_average", "Jain trên tốc độ trung bình UE", 1),
                ("p05_ue_mean_rate_mbps", "Phân vị 5% tốc độ UE (Mbps)", 1),
                ("longest_zero_gap_slots_max", "Gián đoạn UE dài nhất (phút)", 5 / 60),
                ("effective_unavailable_uav_slot_fraction", "Returning và thay pin (%)", 100),
            ],
        }
        for filename, fields in groups.items():
            shape = (1, 3) if len(fields) == 3 else (2, 3)
            fig, axes = plt.subplots(
                *shape, figsize=(14, 4 if shape[0] == 1 else 8), layout="constrained"
            )
            for ax, (key, title, scale) in zip(np.asarray(axes).flat, fields):
                for seed in seeds:
                    vals = [results[label, seed].get(key) for label in LABELS]
                    ax.plot(
                        range(3),
                        [float(v) * scale if v is not None else np.nan for v in vals],
                        color="grey",
                        alpha=0.3,
                        lw=0.8,
                    )
                for idx, label in enumerate(LABELS):
                    vals = [results[label, seed].get(key) for seed in seeds]
                    ax.scatter(
                        [idx] * len(vals),
                        [float(v) * scale if v is not None else np.nan for v in vals],
                        s=14,
                    )
                ax.set_xticks(range(3), ["Random", "Dynamic A/R", "Heu kết hợp"])
                ax.set_title(title, fontsize=11)
                ax.grid(axis="y", alpha=0.2)
            fig.suptitle(
                "DIAGNOSTIC ONLY"
                if p["diagnostic"]
                else "10 seed × 10.000 slot; heuristic trong môi trường mới",
                fontsize=12,
            )
            fig.savefig(output / filename, dpi=160)
            plt.close(fig)
    (output / "REPORT_VI.md").write_text(text)
    print(json.dumps(meta, indent=2), flush=True)
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-diagnostic", action="store_true")
    a = p.parse_args()
    report(a.input, a.output, a.allow_diagnostic)
