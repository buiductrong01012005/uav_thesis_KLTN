"""Fleet spectrum/SINR/half-duplex audit; fixed static fixtures, no training."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(key, "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy

from src.diagnostics.bs5_spectrum_airtime import audit_case, fixture_env


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summaries(rows):
    keys = ["access_bw_label_mhz", "backhaul_total_bandwidth_mhz", "spectrum_mode", "backhaul_mode", "rate_efficiency"]
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for key, group in groups.items():
        item = dict(zip(keys, key))
        item["synthetic_fixture_count"] = len(group)
        for field in ["delivered_mbps", "access_offer_mbps", "utility", "qos_ues", "associated_ues",
                      "access_reserved_fleet_mhz", "access_distinct_used_mhz", "backhaul_loss_mbps"]:
            values = [g[field] for g in group]
            item[f"mean_{field}"] = float(np.mean(values))
            if field == "delivered_mbps":
                item[f"min_{field}"], item[f"max_{field}"] = float(min(values)), float(max(values))
        result.append(item)
    return result


def save_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(rows, aggregate, references, receipt):
    def select(access, bh):
        return next(r for r in aggregate if r["access_bw_label_mhz"] == access
                    and r["backhaul_total_bandwidth_mhz"] == bh and r["rate_efficiency"] == 0.75
                    and r["backhaul_mode"] == "shared_half_duplex" and r["spectrum_mode"] == "colored_reuse")
    matched = [select(20, bh) for bh in [20, 40, 100]]
    lines = [
        "# Audit phổ, nhiễu và airtime — BS5 v2",
        "", f"Hoàn tất {len(rows):,} ca tĩnh tổng hợp; 0 slot rollout, 0 update train, 0 checkpoint evaluation.",
        "Mỗi ca dùng toàn đội 5 UAV, 90 UE, 3 geometry × 3 role fixture. "
        "Đây không phải so sánh thuật toán C-R/C-N và không phải kết quả trên seed test.",
        "", "## Phát hiện chính trên các fixture đã kiểm tra",
        "", f"Với colored_reuse, nhãn Access20MHz và hiệu suất minh họa0.75: "
        f"tăng BH20→40→100MHz cho throughput trung bình "
        f"{matched[0]['mean_delivered_mbps']:.2f}→{matched[1]['mean_delivered_mbps']:.2f}→{matched[2]['mean_delivered_mbps']:.2f}Mbps.",
        f"Giữ BH40MHz, tăng nhãn Access10→15→20MHz chỉ cho "
        f"{select(10,40)['mean_delivered_mbps']:.2f}→{select(15,40)['mean_delivered_mbps']:.2f}→"
        f"{select(20,40)['mean_delivered_mbps']:.2f}Mbps. "
        "Trong nhánh này backhaul/lịch là điểm nghẽn mạnh; không suy ra tăng Access luôn vô ích.",
        "Các số trên là mô tả fixture, không phải kết quả policy hay mức bảo đảm 90UE đạt QoS. "
        "Graph coloring cần nhiều kênh, phải khai báo cả budget phổ đội thay vì chỉ báo MHz/UAV.",
        "", "## Căn cứ tham số",
        "", "- Access20MHz: Xu et al., *Throughput Maximization of UAV Networks*, "
        "IEEE/ACM ToN 30(2), 2022, Section III-C, p.885, DOI 10.1109/TNET.2021.3125982. "
        "Bài dùng 20MHz/UAV và phân kênh/tái sử dụng theo graph coloring; không cấp miễn phí cùng kênh không nhiễu cho mọi UAV.",
        "- BH20MHz và 100MHz tổng qua 5 carrier × 20MHz: *Performance Evaluation of Direct-Link Backhaul "
        "for UAV-Aided Emergency Networks*, Sensors 2019, Table 2, §4.2/5.2, DOI 10.3390/s19153342. "
        "Bài ở 3.5GHz/công suất khác; audit giữ 5GHz/20dBm để cô lập ảnh hưởng tài nguyên.",
        "- Access10/15MHz là giá trị ablation có trong ETSI TS136101 Table5.6-1; "
        "BH40MHz là điểm sensitivity do ta chọn, không gán cho bài báo.",
        "- Half-duplex/lịch liên kết: Tafintsev et al., IEEE OJCOMS 2024, DOI 10.1109/OJCOMS.2024.3449234. "
        "Chỉ tham khảo nguyên tắc scheduling; không chép mô hình mmWave thành sub-6GHz.",
        "Chi tiết URL, vị trí bảng/mục và phạm vi sử dụng trong references.json. "
        "Đây là mô hình Shannon hiệu dụng, chưa mô phỏng LTE resource blocks. "
        "Hiệu suất 0.75 chỉ là độ nhạy minh họa overhead, không phải số đo hoặc thông số lấy từ các bài.",
        "", "## Ngân sách phổ Access",
        "", "| Chế độ | Ý nghĩa nhãn B MHz | Tổng phổ dự phòng cho đội | Nhiễu |",
        "|---|---|---|---|",
        "| orthogonal_pool | Tổng đội chia đều B | B | Không đồng kênh |",
        "| reuse1 | Mỗi Access dùng chung kênh B | B | Có, SINR với flat PSD |",
        "| orthogonal_per_uav | Mỗi UAV có kênh riêng B | 5B | Không đồng kênh |",
        "| colored_reuse | Kênh B, tô màu graph khoảng cách ≤2Raccess | Tối đa 5B; ghi riêng số kênh thực dùng | Có giữa các UAV cùng màu ở xa |",
        "Không diễn giải orthogonal_per_uav/colored_reuse thắng pool/reuse1 là thuật toán tốt hơn: "
        "ngân sách phổ khả dụng khác nhau. Pool/reuse1 là cặp cùng tổng phổ, "
        "orthogonal_per_uav/colored_reuse là cặp cùng trần phổ dự phòng 5B.",
        "", "## Kết quả tổng hợp tại Access nhãn 20MHz, hiệu suất minh họa 0.75",
        "", "Trung bình mô tả trên 9 fixture, không có CI hoặc giả định các ca là mẫu độc lập của thế giới thực.",
        "", "| Access scheme | BH tổng MHz | Dự phòng Access MHz | Access offer Mbps | Đến UE Mbps | UE đạt 4Mbps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        if row["access_bw_label_mhz"] == 20 and row["rate_efficiency"] == 0.75 and row["backhaul_mode"] == "shared_half_duplex":
            lines.append(f"| {row['spectrum_mode']} | {row['backhaul_total_bandwidth_mhz']} | "
                         f"{row['mean_access_reserved_fleet_mhz']:.0f} | {row['mean_access_offer_mbps']:.2f} | "
                         f"{row['mean_delivered_mbps']:.2f} | {row['mean_qos_ues']:.2f} |")
    lines += [
        "", "## So 10/15/20MHz tại BH40MHz, hiệu suất0.75",
        "", "Đổi nhãn băng thông cũng đổi ngân sách phổ như cột dưới; đây là sensitivity tài nguyên, không phải so thuật toán cùng budget.",
        "", "| Access scheme | Nhãn MHz | Phổ dự phòng đội MHz | Đến UE Mbps | UE đạt 4Mbps |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        if row["backhaul_total_bandwidth_mhz"] == 40 and row["rate_efficiency"] == 0.75 and row["backhaul_mode"] == "shared_half_duplex" and row["access_bw_label_mhz"] in [10, 15, 20]:
            lines.append(f"| {row['spectrum_mode']} | {row['access_bw_label_mhz']} | "
                         f"{row['mean_access_reserved_fleet_mhz']:.0f} | {row['mean_delivered_mbps']:.2f} | {row['mean_qos_ues']:.2f} |")
    lines += [
        "", "## Airtime thực sự đã ràng buộc những gì?",
        "", "Mỗi BS/UAV có một radio backhaul single-beam ở 5GHz; cùng node không thể tham gia hai link "
        "trong một trạng thái (kể cả Relay nhận và chuyển tiếp). Link không chung node có thể đồng thời nhưng phải chịu "
        "nhiễu BH đồng kênh. Tổng thời gian tất cả trạng thái ≤1; tổng Mbps qua mỗi edge không vượt capacity "
        "đã nhân airtime. Flow chung BS→Relay chứa tổng dữ liệu mọi Access dùng Relay đó.",
        "Access 2GHz dùng radio riêng và được phép đồng thời BH 5GHz. Không có nhiễu nền từ terrestrial Access "
        "hoặc hệ thống khác, antenna gains/beamforming, fading ngẫu nhiên hoặc giới hạn core-network.",
        "LP tối đa tổng Mbps, rồi max-min phần offer được thỏa mãn giữa Access reachable. "
        "Các UE trong một Access cùng tỷ lệ scale như env cũ. Đây là scheduler offline có geometry chính xác, "
        "không phải policy online. Route cố định chọn bottleneck riêng lẻ tốt nhất, chưa joint route-schedule optimization.",
        "ideal_parallel_bound cho mọi edge hoạt động độc lập toàn thời gian/no-interference, "
        "chỉ là cận lạc quan. Nó giữ flow conservation nhưng bỏ budget radio, không được dùng làm throughput triển khai.",
        "Lịch là fluid steady-state, có thể cần buffer; chưa mô phỏng thứ tự packet, delay, startup hay latency nhận–phát. "
        "0.5 airtime không có nghĩa CPU chờ 2.5s hoặc đo được độ trễ packet.",
        "", "## Quyết định sau audit",
        "", "20MHz Access có nguồn bài báo trực tiếp để làm cấu hình tham chiếu; 15MHz vẫn là đối chứng cần giữ. "
        "Chưa chứng minh 20MHz là tối ưu hoặc đủ cho 90UE đạt 4Mbps trong mọi geometry.",
        "BH phải chọn bằng throughput sau lịch/flow, không chỉ bằng raw link capacity hoặc tăng cùng tỷ lệ với Access. "
        "100MHz trong audit là effective aggregate bandwidth theo ý tưởng carrier aggregation, "
        "chưa phải LTE CA triển khai; giữ công suất20dBm nên không sao chép kết quả công suất43dBm của bài Sensors.",
        "Các layout là ca stress tổng hợp, không phải mẫu từ phân bố UE triển khai; chưa dùng chúng để chốt policy, "
        "checkpoint hoặc weights. Không train trên seed260001; v2 không đổi env/reward/observation cũ.",
        "Bước tiếp theo là chọn giả định ngân sách phổ khả dụng, sau đó đưa scheduler/radio đã chốt vào env mới "
        "và kiểm tra động qua chu kỳ sạc. Hybrid vẫn là nhánh thử riêng, chưa có rollout toàn đội trong v2.",
        "", "## Đầu ra và tính tái lập",
        "", "cases.csv: từng ca; aggregate.csv: nhóm cùng config; schedules.jsonl.gz: UE rates, "
        "flow/route và lịch cụ thể của mọi ca; fixtures.json: tọa độ; references.json: nguồn tham số; "
        "AUDIT_RECEIPT.json: số ca, LP residual, thời gian, phiên bản và hash nguồn.",
        "Không trộn số trong audit với benchmark best120k cũ: mô hình radio và scheduler khác.",
        "", f"Thời gian đo phần audit: {receipt['wall_seconds']:.2f}s wall / {receipt['cpu_seconds']:.2f}s CPU. "
        f"LP constraint violation lớn nhất: {receipt['max_lp_constraint_violation_mbps']:.3g} Mbps.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bs5_spectrum_airtime_audit_v2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/bs5_spectrum_airtime_audit_v2")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output is nonempty; choose a new --output to preserve prior results")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    references = json.loads((ROOT / "configs/bs5_spectrum_references_v2.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "status.json", {"status": "running"})
    start, cpu = time.perf_counter(), time.process_time()
    rows, fixtures = [], {}
    try:
        with gzip.open(args.output / "schedules.jsonl.gz", "wt", encoding="utf-8") as stream:
            for layout, roles in itertools.product(config["fixtures"], config["role_fixtures"]):
                env = fixture_env(layout, roles, config)
                fixtures[layout + ":" + roles] = dict(
                    uav_cells=env.uav_positions.tolist(), ue_cells=env.ue_positions.tolist(),
                    roles=env.roles.tolist(), gateway_xy_m=env.surviving_bs_xy.tolist(),
                )
                grid = itertools.product(config["access_bandwidths_mhz"], config["backhaul_bandwidths_mhz"],
                                         config["spectrum_modes"], config["backhaul_modes"], config["rate_efficiencies"])
                for access, bh, spectrum, scheduler, eta in grid:
                    metric, detail = audit_case(env, access_bw=access, backhaul_bw=bh,
                                               spectrum_mode=spectrum, backhaul_mode=scheduler, efficiency=eta)
                    row = dict(case_id=len(rows), fixture=layout, role_fixture=roles, **metric)
                    rows.append(row)
                    stream.write(json.dumps(dict(metric=row, **detail), allow_nan=False) + "\n")
                print(f"completed {layout}/{roles}: {len(rows)} cases", flush=True)
        aggregate = summaries(rows)
        sources = ["src/diagnostics/bs5_spectrum_airtime.py", "src/diagnostics/bs5_resource_audit.py",
                   "scripts/analysis/run_bs5_spectrum_airtime_audit.py", "src/envs/bs5_launch_ab_env.py",
                   "src/envs/v4_fixed_mobile_battery_env.py", "src/envs/v4_final_fixed_map_env.py"]
        receipt = dict(
            version=config["version"], environment_step_calls=0, training_updates=0, checkpoint_evaluations=0,
            cases=len(rows), fixtures=len(fixtures), diagnostic_seed=config["diagnostic_seed"],
            wall_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu,
            max_lp_constraint_violation_mbps=max(r["lp_constraint_violation_mbps"] for r in rows),
            max_schedule_time_sum=max(r["schedule_time_sum"] for r in rows),
            numpy_version=np.__version__, scipy_version=scipy.__version__, python=platform.python_version(),
            config_sha256=file_hash(args.config), source_sha256={p: file_hash(ROOT / p) for p in sources},
            scope=config["scope"], promotion_status=config["promotion_status"],
        )
        save_csv(args.output / "cases.csv", rows)
        save_csv(args.output / "aggregate.csv", aggregate)
        for name, data in [("AUDIT_RECEIPT", receipt), ("fixtures", fixtures),
                           ("audit_config", config), ("references", references)]:
            write_json(args.output / f"{name}.json", data)
        (args.output / "REPORT_VI.md").write_text(report(rows, aggregate, references, receipt), encoding="utf-8")
        write_json(args.output / "status.json", {"status": "complete", "cases": len(rows)})
        print(json.dumps(receipt, indent=2))
    except BaseException:
        write_json(args.output / "status.json", {"status": "failed_or_interrupted", "completed_cases": len(rows)})
        raise


if __name__ == "__main__":
    main()
