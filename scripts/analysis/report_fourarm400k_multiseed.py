"""Crossed train/scenario uncertainty and raw audits, with optional Random baseline."""

from pathlib import Path
from itertools import combinations
import argparse
import io
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.fourarm400k_multiseed import PLAN, ARMS, SEEDS, sources, digest
from scripts.analysis.compare_fourarm400k_multiseed import write_csv
from scripts.source_integrity import canonical_environment, verify_source, describe_source
from scripts.analysis.report_fourarm400k_heuristics import validate_raw, METRICS
from scripts.evaluation import run_fourarm400k_heuristics as core
from scripts.evaluation.fourarm400k_return_summary import return_metrics
from scripts.evaluation.run_fourarm400k_drl_benchmark import VERSION as DRL_VERSION, SCIENCE_KEYS

HEURISTICS = PLAN["heuristics"]
LABELS = ARMS + HEURISTICS
RANDOM = "RANDOM"
DISPLAY = {
    "C-N": "C-N",
    "C-R": "C-R",
    "C-H": "C-H",
    "H-only": "H-only",
    "DYNAMIC_AR": "Heu A/R",
    "HEURISTIC_HO": "Heu kết hợp",
    RANDOM: "Random A/R",
}
COLORS = dict(
    zip(
        LABELS + [RANDOM],
        ["#777777", "#3073a5", "#44916f", "#d77c28", "#8c5a9b", "#537882", "#ae5365"],
    )
)


def scan(input_dir, allow_diagnostic=False, include_random=False):
    heuristics = HEURISTICS + [RANDOM] if include_random else HEURISTICS
    results = {}
    protocols = []
    origins = []
    expected_environment = core.make_protocol("DYNAMIC_AR", 1)["environment_contract"]
    for p, read, origin in sources(input_dir):
        version = p.get("version")
        if version not in [core.VERSION, DRL_VERSION]:
            continue
        diag = bool(p.get("diagnostic"))
        if diag and not allow_diagnostic:
            raise ValueError("Diagnostic output cannot enter the held-out test report")
        if p["part"] not in (1, 2):
            raise ValueError("Unknown test partition")
        target = (
            p["config"]["diagnostic_seeds"]
            if diag
            else PLAN["test_seeds"][(p["part"] - 1) * 5 : p["part"] * 5]
        )
        if (
            not p["target_seeds"]
            or len(set(p["target_seeds"])) != len(p["target_seeds"])
            or not set(p["target_seeds"]) <= set(target)
        ):
            raise ValueError("Wrong scenario seed partition")
        if p["horizon"] != (p["config"]["diagnostic_horizon"] if diag else PLAN["test_horizon"]):
            raise ValueError("Wrong test horizon")
        if digest(p["environment_contract"]) != p[
            "environment_fingerprint"
        ] or canonical_environment(p["environment_contract"]) != canonical_environment(
            expected_environment
        ):
            raise ValueError("Benchmark physical environment changed")
        if version == core.VERSION:
            if p["label"] not in heuristics:
                continue
            if p["arm"] != core.LABELS[p["label"]]:
                raise ValueError("Heuristic arm mismatch")
            keys = [
                "version",
                "config",
                "training_protocol",
                "source_sha256",
                "environment_contract",
                "environment_fingerprint",
            ]
            configuration = p["label"]
            training_seed = None
        else:
            keys = SCIENCE_KEYS
            configuration = p["arm"]
            training_seed = p["training_seed"]
            if (
                configuration not in ARMS
                or training_seed not in SEEDS
                or p["selection"]["metric"] != "reward_total"
            ):
                raise ValueError("Unexpected DRL training replicate/selection")
        if digest({k: p[k] for k in keys}) != p["comparison_fingerprint"]:
            raise ValueError("Benchmark protocol fingerprint invalid")
        for rel, sha in p["source_sha256"].items():
            verify_source(rel, sha, ROOT)
        protocols.append(p)
        for seed in p["target_seeds"]:
            prefix = f"seed{seed}"
            try:
                row = json.loads(read(prefix + "/summary.json"))
            except (KeyError, FileNotFoundError):
                continue
            if not row.get("complete"):
                continue
            if (
                row["label"] != p["label"]
                or row["arm"] != p["arm"]
                or row["seed"] != seed
                or row["slots"] != p["horizon"]
                or row["target_slots"] != p["horizon"]
                or row["diagnostic"] != diag
                or row["environment_fingerprint"] != p["environment_fingerprint"]
                or row["comparison_fingerprint"] != p["comparison_fingerprint"]
                or row["selected_step"] != p["selection"]["selected_step"]
            ):
                raise ValueError("Seed summary/protocol mismatch")
            commit = json.loads(read(prefix + "/COMMIT.json"))
            if commit["protocol_sha256"] != digest(p):
                raise ValueError("Commit belongs to a different worker")
            validate_raw(read, prefix, row, p)
            rec = dict(row, configuration=configuration, training_seed=training_seed)
            with np.load(io.BytesIO(read(prefix + "/energy.npz")), allow_pickle=False) as energy:
                rec.update(
                    return_metrics(
                        json.loads(read(prefix + "/energy_events.json")), energy["distance_m"]
                    )
                )
            key = (configuration, training_seed, seed)
            if key in results and results[key] != rec:
                raise ValueError("Conflicting completed run: " + str(key))
            results[key] = rec
        origins.append(
            dict(
                origin=origin,
                label=configuration,
                training_seed=training_seed,
                part=p["part"],
                scenarios=p["target_seeds"],
                **describe_source(p),
            )
        )
    if not protocols:
        raise ValueError("Attach new heuristic/DRL benchmark REVIEW files")
    if len({p["diagnostic"] for p in protocols}) != 1:
        raise ValueError("Diagnostic/test mixture")
    if len({digest(canonical_environment(p["environment_contract"])) for p in protocols}) != 1:
        raise ValueError("Environment mismatch")
    if len({p["model_bundle_sha256"] for p in protocols if p["version"] == DRL_VERSION}) > 1:
        raise ValueError("Different frozen model bundles")
    for scenario in {r["seed"] for r in results.values()}:
        if len({r["ue_trace_sha256"] for r in results.values() if r["seed"] == scenario}) > 1:
            raise ValueError("UE trajectories are not paired: " + str(scenario))
    # A model hash and selection must remain fixed across every test partition.
    for arm in ARMS:
        for seed in SEEDS:
            if (
                len(
                    {
                        digest(p["selection"])
                        for p in protocols
                        if p["version"] == DRL_VERSION
                        and p["arm"] == arm
                        and p["training_seed"] == seed
                    }
                )
                > 1
            ):
                raise ValueError("Model changed between scenario partitions")
    return results, protocols, origins


def expected_keys(labels, scenarios):
    return {
        (label, ts, sc)
        for label in labels
        for ts in (SEEDS if label in ARMS else [None])
        for sc in scenarios
    }


def matrix(results, label, scenarios, metric):
    return np.asarray(
        [
            [
                np.nan if results[label, t, s].get(metric) is None else results[label, t, s][metric]
                for s in scenarios
            ]
            for t in (SEEDS if label in ARMS else [None])
        ],
        dtype=float,
    )


def distribution(results, label, scenarios, metric, ti, si):
    """Resample crossed axes; each heuristic stays a single vector of scenario runs."""
    train = ti if label in ARMS else np.zeros((len(si), 1), dtype=int)

    def sampled(key):
        a = matrix(results, label, scenarios, key)
        return a, a[train[:, :, None], si[:, None, :]]

    if metric == "joules_per_mbit_pooled":
        a, x = sampled("energy_consumed_wh")
        b, y = sampled("delivered_mbit")
        a = a * 3600
        x = x * 3600
    elif metric == "blocked_fraction_pooled":
        a, x = sampled("active_blocked_total")
        b, y = sampled("policy_move_requests_total")
    else:
        a, x = sampled(metric)
        if not np.isfinite(a).all():
            return None, None
        return float(a.mean()), x.mean(axis=(1, 2))
    if not np.isfinite(a).all() or not np.isfinite(b).all() or b.sum() <= 0:
        return None, None
    den = y.sum(axis=(1, 2))
    boot = np.divide(x.sum(axis=(1, 2)), den, out=np.full(len(den), np.nan), where=den > 0)
    return float(a.sum() / b.sum()), boot


def aggregate(results, labels, scenarios, replicates=4000):
    rng = np.random.default_rng(20261106)
    ti = rng.integers(0, 3, size=(replicates, 3))
    si = rng.integers(0, len(scenarios), size=(replicates, len(scenarios)))
    metrics = list(dict.fromkeys(METRICS + ["joules_per_mbit_pooled", "blocked_fraction_pooled"]))
    rows = []
    dists = {}
    comparisons = []
    for label in labels:
        row = dict(
            configuration=label,
            training_replicates=3 if label in ARMS else 0,
            scenario_seeds=len(scenarios),
            actual_evaluation_runs=len(scenarios) * (3 if label in ARMS else 1),
        )
        for metric in metrics:
            mean, boot = distribution(results, label, scenarios, metric, ti, si)
            row[metric] = mean
            valid = None if boot is None else boot[np.isfinite(boot)]
            row[metric + "_ci95_low"] = (
                float(np.quantile(valid, 0.025))
                if valid is not None and len(valid) == replicates
                else None
            )
            row[metric + "_ci95_high"] = (
                float(np.quantile(valid, 0.975))
                if valid is not None and len(valid) == replicates
                else None
            )
            dists[label, metric] = (mean, boot)
        rows.append(row)
    for a, b in combinations(labels, 2):
        for metric in metrics:
            aa, ba = dists[a, metric]
            bb, bootb = dists[b, metric]
            if aa is None or bb is None or ba is None or bootb is None:
                continue
            delta = bootb - ba
            valid = delta[np.isfinite(delta)]
            comparisons.append(
                dict(
                    reference=a,
                    comparison=b,
                    metric=metric,
                    reference_mean=aa,
                    comparison_mean=bb,
                    difference=bb - aa,
                    relative_change_percent=100 * (bb / aa - 1) if aa != 0 else None,
                    ci95_low=float(np.quantile(valid, 0.025)) if len(valid) == replicates else None,
                    ci95_high=float(np.quantile(valid, 0.975))
                    if len(valid) == replicates
                    else None,
                    scope="Exploratory paired crossed bootstrap: shared training-seed index for DRL and shared scenario index; heuristics only scenario-resampled; three training seeds; no multiple-comparison adjustment",
                )
            )
    return rows, comparisons


def report(input_dir, output, heuristics_only=False, allow_diagnostic=False, include_random=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results, protocols, origins = scan(input_dir, allow_diagnostic, include_random)
    labels = list(HEURISTICS if heuristics_only else LABELS)
    if include_random:
        labels.append(RANDOM)
    diag = protocols[0]["diagnostic"]
    scenarios = protocols[0]["config"]["diagnostic_seeds"] if diag else PLAN["test_seeds"]
    needed = expected_keys(labels, scenarios)
    missing = sorted(needed - set(results), key=str)
    complete = not missing
    selected = {k: v for k, v in results.items() if k in needed}
    write_csv(output / "per_run.csv", [selected[k] for k in sorted(selected, key=str)])
    meta = dict(
        status=("diagnostic_only" if diag else "complete") if complete else "partial",
        configurations=labels,
        expected_runs=len(needed),
        completed_runs=len(selected),
        missing=[list(k) for k in missing],
        environment_fingerprint=digest(canonical_environment(protocols[0]["environment_contract"])),
        actual_environment_fingerprints=sorted({p["environment_fingerprint"] for p in protocols}),
        provenance=origins,
        scope=PLAN["statistics"],
    )
    (output / "completeness.json").write_text(json.dumps(meta, indent=2) + "\n")
    products = [
        "aggregate.csv",
        "paired_comparison.csv",
        "scenario_summary.csv",
        "training_seed_summary.csv",
        "main_metrics.png",
        "service_metrics.png",
        "training_seed_sensitivity.png",
    ]
    if not complete:
        for name in products:
            (output / name).unlink(missing_ok=True)
        (output / "REPORT_VI.md").write_text(
            f"# Benchmark chưa hoàn tất\n\nĐã có {len(selected)}/{len(needed)} lượt hoàn chỉnh. Xem completeness.json; chưa tạo bảng xếp hạng cuối cùng.\n"
        )
        return meta
    aggregates, comparisons = aggregate(selected, labels, scenarios)
    write_csv(output / "aggregate.csv", aggregates)
    write_csv(output / "paired_comparison.csv", comparisons)
    scenario_rows = []
    train_rows = []
    summary_metrics = [
        "throughput_mbps",
        "qos_ues",
        "service_coverage_ues",
        "jain_time_average",
        "p05_ue_mean_rate_mbps",
        "energy_consumed_wh",
        "actor_ms",
    ]
    for label in labels:
        for sc in scenarios:
            rr = [selected[label, ts, sc] for ts in (SEEDS if label in ARMS else [None])]
            scenario_rows.append(
                dict(
                    configuration=label,
                    scenario_seed=sc,
                    **{k: float(np.mean([r[k] for r in rr])) for k in summary_metrics},
                )
            )
        if label in ARMS:
            for ts in SEEDS:
                rr = [selected[label, ts, sc] for sc in scenarios]
                train_rows.append(
                    dict(
                        configuration=label,
                        training_seed=ts,
                        scenario_seeds=len(scenarios),
                        **{k: float(np.mean([r[k] for r in rr])) for k in summary_metrics},
                        joules_per_mbit_pooled=sum(r["energy_consumed_wh"] * 3600 for r in rr)
                        / sum(r["delivered_mbit"] for r in rr),
                    )
                )
    write_csv(output / "scenario_summary.csv", scenario_rows)
    write_csv(output / "training_seed_summary.csv", train_rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = [COLORS[label] for label in labels]
    groups = {
        "main_metrics.png": [
            ("throughput_mbps", "Thông lượng (Mbps)", 1),
            ("qos_ues", "UE đạt 4 Mbps mỗi slot", 1),
            ("joules_per_mbit_pooled", "J/Mbit", 1),
            ("energy_consumed_wh", "Điện cả đội (kWh)", 0.001),
            ("blocked_fraction_pooled", "Yêu cầu di chuyển bị chặn (%)", 100),
            ("actor_ms", "Thời gian quyết định (ms)", 1),
        ],
        "service_metrics.png": [
            ("jain_time_average", "Jain trên tốc độ trung bình UE", 1),
            ("p05_ue_mean_rate_mbps", "Phân vị 5% tốc độ UE (Mbps)", 1),
            ("rolling_unique_ge1_5min", "UE khác nhau đạt 1 Mbps / 5 phút", 1),
            ("longest_zero_gap_slots_max", "Gián đoạn UE dài nhất (phút)", 5 / 60),
            ("effective_unavailable_uav_slot_fraction", "Returning và thay pin (%)", 100),
            ("actor_over_5s_slot_fraction", "Quyết định vượt slot 5 giây (%)", 100),
        ],
    }
    for filename, fields in groups.items():
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        for ax, (metric, title, scale) in zip(axes.flat, fields):
            for i, (label, color, row) in enumerate(zip(labels, colors, aggregates)):
                mean = row.get(metric)
                lo = row.get(metric + "_ci95_low")
                hi = row.get(metric + "_ci95_high")
                if mean is None:
                    continue
                ax.bar(i, mean * scale, color=color, alpha=0.85)
                if lo is not None and hi is not None:
                    # Percentile intervals need not contain the point estimate: draw endpoints directly.
                    ax.plot([i, i], [lo * scale, hi * scale], color="black", lw=1)
                    ax.plot([i - 0.07, i + 0.07], [lo * scale, lo * scale], color="black")
                    ax.plot([i - 0.07, i + 0.07], [hi * scale, hi * scale], color="black")
            ax.set_xticks(
                range(len(labels)), [DISPLAY[label] for label in labels], rotation=25, ha="right"
            )
            ax.set_title(title, fontsize=11)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
        title = (
            "Các phương pháp đối chứng trên cùng seed kịch bản"
            if heuristics_only
            else "So sánh các cấu hình trên cùng seed kịch bản"
        )
        fig.suptitle(("DIAGNOSTIC ONLY — " if diag else "") + title, fontsize=13)
        fig.supxlabel(
            "Thanh sai số: bootstrap 95%; DRL lấy lại mẫu cả seed train và seed kịch bản. Chỉ ba seed train nên diễn giải thận trọng.",
            fontsize=9,
        )
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    if train_rows:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for ax, (metric, title) in zip(
            axes,
            [
                ("throughput_mbps", "Thông lượng (Mbps)"),
                ("qos_ues", "UE đạt 4 Mbps mỗi slot"),
                ("joules_per_mbit_pooled", "J/Mbit"),
            ],
        ):
            for j, ts in enumerate(SEEDS):
                vals = [
                    next(
                        r[metric]
                        for r in train_rows
                        if r["configuration"] == a and r["training_seed"] == ts
                    )
                    for a in ARMS
                ]
                ax.scatter(np.arange(4) + (j - 1) * 0.12, vals, label=f"Seed train {ts}")
            ax.set_xticks(range(4), ARMS)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle("Mỗi chấm: một model, trung bình cùng 10 seed test")
        fig.savefig(output / "training_seed_sensitivity.png", dpi=160)
        plt.close(fig)
    lines = [
        "# Benchmark nhiều seed huấn luyện",
        "",
        "**CHỈ DIAGNOSTIC.**" if diag else "Đánh giá trên 10 seed test × 10.000 slot.",
        "",
        f"Có {len(selected)} lượt đánh giá thực tế. Mỗi cấu hình DRL dùng ba model best đã chọn riêng theo validation; mỗi heuristic chạy một lần cho mỗi seed kịch bản. Không nhân bản heuristic thành ba thí nghiệm độc lập.",
        "",
        "| Cấu hình | Thông lượng (Mbps) | UE đạt 4 Mbps/slot | J/Mbit gộp |",
        "|---|---:|---:|---:|",
    ]
    for r in aggregates:
        lines.append(
            "| "
            + DISPLAY[r["configuration"]]
            + " | "
            + f"{r['throughput_mbps']:.3f} | {r['qos_ues']:.3f} | {r['joules_per_mbit_pooled']:.3f} |"
        )
    lines += [
        "",
        "Thống kê Returning được tính lại từ dữ liệu từng slot cho mọi lượt. Các chuyến thiếu tuyến dự kiến vẫn đóng góp quãng đường thực tế và thời lượng; số chuyến thiếu tuyến nằm trong per_run.csv. Hình chính có một nhãn cho mỗi cấu hình. Các model theo seed train nằm trong bảng training_seed_summary.csv và hình riêng. Learning curve được tổng hợp ở notebook train, không đưa heuristic vào đồ thị học.",
        "",
        "Khoảng bootstrap giữ ghép cặp theo kịch bản. Với DRL, chỉ số seed train được lấy lại mẫu độc lập với chỉ số kịch bản, rồi áp dụng chung các chỉ số đó giữa các cấu hình. Đây là thiết kế giao nhau, không coi 30 lượt của một cấu hình là 30 lần train độc lập. Ba seed train là kiểm tra độ nhạy ban đầu; khoảng bất định còn hạn chế và các phép so sánh phụ chưa hiệu chỉnh đa kiểm định.",
        "",
        "Tổng điện dùng Wh; J/Mbit gộp dùng tổng joule chia tổng Mbit. Tỷ lệ bị chặn gộp dùng tổng bị chặn chia tổng yêu cầu di chuyển. Hai heuristic tìm kiếm vẫn dùng điểm T/C/F cũ. Random chọn đều trong các hành động hợp lệ. Tất cả dùng cùng reward đánh giá. Không điều chỉnh controller, trọng số hoặc checkpoint từ test.",
        "",
        "Thời gian quyết định gồm toàn bộ lời gọi controller; báo thêm p95, cực đại và tỷ lệ vượt slot 5 giây trong aggregate.csv. Thời gian các CPU khác nhau chưa phải đối chứng tốc độ tuyệt đối trên cùng phần cứng.",
    ]
    (output / "REPORT_VI.md").write_text("\n".join(lines) + "\n")
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--heuristics-only", action="store_true")
    p.add_argument("--allow-diagnostic", action="store_true")
    p.add_argument("--include-random", action="store_true")
    a = p.parse_args()
    print(
        json.dumps(
            report(a.input, a.output, a.heuristics_only, a.allow_diagnostic, a.include_random),
            indent=2,
        )
    )
