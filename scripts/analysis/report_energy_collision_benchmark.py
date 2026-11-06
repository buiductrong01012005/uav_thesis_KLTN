"""Merge four review archives; compare complete paired seeds, never choose models."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.stats import t


METRICS = ["throughput_mbps", "energy_consumed_wh", "joules_per_mbit", "service_coverage_ues",
           "qos_ues", "jain_time_average", "blocked_per_policy_move", "unique_ge1_ues",
           "unique_qos_ues", "longest_zero_gap_slots_max", "longest_consecutive_blocked_slots",
           "zero_service_slot_fraction", "unavailable_uav_slot_fraction", "saturated_backhaul_slot_fraction",
           "arrival_below20_events", "min_arrival_soc_percent", "actor_ms", "actor_ms_p95",
           "energy_shortfall_total_j"]


def csv_write(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def scan_archives(paths):
    results, protocols, provenance = {}, {}, []
    for path in sorted(paths):
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:
                raise ValueError("Corrupt archive: " + str(path))
            protocol_names = [n for n in z.namelist() if n.endswith("/protocol.json")]
            # Ignore dataset/training/previous-analysis ZIPs attached to a compare notebook.
            if not protocol_names:
                continue
            for name in protocol_names:
                p = json.loads(z.read(name))
                if p.get("version") != "energy_collision_frozen120k_benchmark_v1":
                    continue
                if p["diagnostic"]:
                    continue
                identity = (p["label"], p["part"])
                if identity in protocols and protocols[identity] != p:
                    raise ValueError("Conflicting protocols for " + str(identity))
                protocols[identity] = p
                prefix = name.rsplit("/", 1)[0]
                for member in z.namelist():
                    if member.startswith(prefix+"/seed") and member.endswith("/summary.json"):
                        row = json.loads(z.read(member))
                        if row["label"] != p["label"] or row["seed"] not in p["target_seeds"]:
                            raise ValueError("Seed/worker identity mismatch")
                        if row["comparison_fingerprint"] != p["comparison_fingerprint"] or row["target_slots"] != p["horizon"]:
                            raise ValueError("Summary/protocol mismatch")
                        if not row["complete"] or row["slots"] != p["horizon"]:
                            continue
                        key = (row["label"], row["seed"])
                        if key in results and results[key] != row:
                            raise ValueError("Conflicting duplicate complete seed: " + str(key))
                        results[key] = row
                provenance.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), worker=list(identity)))
    if not protocols:
        raise ValueError("No non-diagnostic benchmark review/resume archive found")
    fingerprints = {p["comparison_fingerprint"] for p in protocols.values()}
    if len(fingerprints) != 1:
        raise ValueError("Different scientific protocols/checkpoint registry")
    p = next(iter(protocols.values()))
    if p["horizon"] != 10000 or len(p["config"]["test_seeds"]) != 10:
        raise ValueError("Unexpected final evaluation budget")
    for label, seed in results:
        if ("P03", seed) in results and ("P05", seed) in results:
            if results["P03", seed]["ue_trace_sha256"] != results["P05", seed]["ue_trace_sha256"]:
                raise ValueError("UE trace mismatch for paired seed " + str(seed))
    return results, p, provenance


def comparison(results, seeds):
    aggregate, paired = [], []
    for label in ("P03", "P05"):
        rows = [results[label, seed] for seed in seeds]
        record = dict(label=label, seeds=len(rows))
        for key in METRICS:
            values = [r[key] for r in rows if r.get(key) is not None]
            record[key] = float(np.mean(values)) if values else None
        energy = sum(r["energy_consumed_wh"]*3600 for r in rows)
        mbit = sum(r["delivered_mbit"] for r in rows)
        requests = sum(r["policy_move_requests_total"] for r in rows)
        record["joules_per_mbit_pooled"] = energy/mbit if mbit else None
        record["blocked_fraction_pooled"] = sum(r["active_blocked_total"] for r in rows)/requests if requests else None
        aggregate.append(record)
    for key in METRICS:
        pairs = [(results["P03", s][key], results["P05", s][key]) for s in seeds
                 if results["P03", s].get(key) is not None and results["P05", s].get(key) is not None]
        if not pairs:
            continue
        a, b = np.asarray(pairs).T
        delta = b-a
        half = float(t.ppf(.975, len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None
        paired.append(dict(metric=key, paired_seeds=len(delta), p03_mean=float(a.mean()), p05_mean=float(b.mean()),
            p05_minus_p03=float(delta.mean()), delta_ci95_low=float(delta.mean()-half) if half is not None else None,
            delta_ci95_high=float(delta.mean()+half) if half is not None else None,
            p05_relative_change_percent=float((b.mean()/a.mean()-1)*100) if a.mean()!=0 else None,
            ci_scope="paired t interval over scenario seeds conditional on these two frozen policies; no multiple-comparison adjustment"))
    return aggregate, paired


def report(paths, output):
    output.mkdir(parents=True, exist_ok=True)
    results, p, provenance = scan_archives(paths)
    target = p["config"]["test_seeds"]
    paired_seeds = [s for s in target if ("P03", s) in results and ("P05", s) in results]
    missing = {label: [s for s in target if (label, s) not in results] for label in ("P03", "P05")}
    csv_write(output / "per_seed.csv", [results[k] for k in sorted(results)])
    meta = dict(status="complete" if len(paired_seeds)==10 else "partial", paired_seeds=paired_seeds,
                missing=missing, comparison_fingerprint=p["comparison_fingerprint"], input_archives=provenance,
                scope=p["config"]["scope"])
    (output / "completeness.json").write_text(json.dumps(meta, indent=2)+"\n")
    if len(paired_seeds) != 10:
        text = f"# Chưa đủ bộ benchmark\n\nHiện có {len(paired_seeds)}/10 seed ghép cặp hoàn tất.\n\nThiếu: `{missing}`.\n\nĐã xuất per_seed.csv và completeness.json. Chưa tổng hợp kết luận hoặc khoảng tin cậy cuối cùng.\n"
        (output / "REPORT_VI.md").write_text(text)
        print(text)
        return meta
    aggregate, paired = comparison(results, paired_seeds)
    csv_write(output / "aggregate.csv", aggregate)
    csv_write(output / "paired_comparison.csv", paired)
    (output / "paired_comparison.json").write_text(json.dumps(paired, indent=2)+"\n")
    lines = ["# Benchmark P03/P05: 10 seed × 10.000 slot", "",
             "Hai checkpoint cố định: P03 @102k, P05 @96k. Không chọn lại model hoặc trọng số theo kết quả test.", "",
             "| Chỉ số | P03 | P05 |", "|---|---:|---:|"]
    names = {"throughput_mbps":"Thông lượng (Mbps)", "energy_consumed_wh":"Năng lượng đội (Wh)",
             "joules_per_mbit_pooled":"J/Mbit (gộp dữ liệu)", "service_coverage_ues":"UE ≥1 Mbps/slot",
             "qos_ues":"UE ≥4 Mbps/slot", "jain_time_average":"Jain theo thời gian", "blocked_fraction_pooled":"Tỷ lệ yêu cầu di chuyển bị chặn"}
    for key, label in names.items():
        format_value = lambda value: f"{value:.4f}" if value is not None else "không xác định"
        lines.append(f"| {label} | {format_value(aggregate[0][key])} | {format_value(aggregate[1][key])} |")
    lines += ["", "Khoảng tin cậy trong paired_comparison.csv áp dụng cho chênh lệch P05 − P03 qua seed kịch bản. Chúng không đo độ biến động giữa các seed huấn luyện và chưa điều chỉnh cho việc xét nhiều metric. Thông lượng là chỉ số so sánh chính.", "",
              "Reward tổng của hai cấu hình dùng trọng số phạt khác nhau nên không được coi là một thước đo chung để xếp hạng. Tỷ lệ bị chặn là yêu cầu di chuyển bị từ chối, không phải tai nạn vật lý.", "",
              "Không gộp với kết quả dùng mô hình pin cũ. Tệp per_seed.csv chứa các chỉ số gián đoạn, SOC, xung đột và thời gian quyết định để kiểm tra các đánh đổi."]
    (output / "REPORT_VI.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, key, title in zip(axes, ["throughput_mbps", "joules_per_mbit", "blocked_per_policy_move"],
                             ["Thông lượng (Mbps)", "Năng lượng trên dữ liệu (J/Mbit)", "Yêu cầu di chuyển bị chặn (%)"]):
        scale = 100 if key == "blocked_per_policy_move" else 1
        a = np.asarray([results["P03", s][key]*scale if results["P03", s][key] is not None else np.nan for s in target])
        b = np.asarray([results["P05", s][key]*scale if results["P05", s][key] is not None else np.nan for s in target])
        for aa, bb in zip(a, b):
            ax.plot([0, 1], [aa, bb], color="grey", alpha=.4, lw=1)
        ax.scatter(np.zeros(10), a, color="#2474a8", label="P03")
        ax.scatter(np.ones(10), b, color="#dd7720", label="P05")
        ax.set_xticks([0, 1], ["P03 @102k", "P05 @96k"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "paired_metrics.png", dpi=180)
    fig.savefig(output / "paired_metrics.pdf")
    plt.close(fig)
    print("Completed paired report:", output)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.input.rglob("*.zip"), args.output)
