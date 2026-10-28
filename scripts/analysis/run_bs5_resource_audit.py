"""Fast static resource audit. No training, weights or 10k rollout required."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.diagnostics.bs5_resource_audit import run_audit


def report_text(result: dict) -> str:
    designs = result["design_targets"]
    rows = result["static_scenes"]
    lines = [
        "# Kiểm tra tài nguyên BS5 — ca tĩnh, không train",
        "",
        "Đơn vị: Mbps = megabit/giây; 4 Mbps = 0,5 MB/s. Mục tiêu cũ vẫn là 4 Mbps/UE.",
        "Các ca UE đồng vị trí là tổng hợp để cô lập điểm nghẽn, không phải kết quả benchmark di động.",
        "",
        "| Profile | Access MHz | BH MHz | BH dBm | UE đủ QoS/25, Access 200m, BH 100m | Throughput Mbps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if (
            row["users"] == 25
            and row["access_distance_m"] == 200
            and row["gateway_distance_m"] == 100
        ):
            lines.append(
                f"| {row['profile']} | {row['access_bandwidth_mhz']:g} | "
                f"{row['gateway_bandwidth_mhz']:g} | {row['gateway_tx_dbm']:g} | "
                f"{row['qos_ues']} | {row['delivered_mbps']:.4f} |"
            )
    lines += [
        "",
        "## Thiết kế theo tải tham chiếu 25 UE × 4 Mbps = 100 Mbps",
        "",
        "| Profile | Access tối thiểu tại 200m, MHz | Khoảng cách BH tối đa để đạt 100 Mbps, m | BH tại 300m, Mbps |",
        "|---|---:|---:|---:|",
    ]
    for row in designs:
        radius = row["max_gateway_distance_for_target_m"]
        radius_text = "không đạt ngay tại 0m" if radius is None else f"{radius:.2f}"
        bandwidth = row["min_access_bw_at_edge_mhz"]
        bandwidth_text = "không có nghiệm hữu hạn" if bandwidth is None else f"{bandwidth:.4f}"
        lines.append(
            f"| {row['profile']} | {bandwidth_text} | "
            f"{radius_text} | {row['gateway_rate_at_radius_mbps']:.4f} |"
        )
    lines += [
        "",
        "Access 20 MHz là ứng viên kiểm tra tiếp, chưa phải cấu hình cuối. "
        "10 MHz vẫn thiếu một chút tại rìa 200 m khi chia cho 25 UE.",
        "Tăng riêng băng thông BH không đảm bảo đủ 100 Mbps tại 300 m với công suất 20 dBm.",
        "Ngưỡng khoảng cách thiết kế chưa có fading/interference margin; bán kính 200/300 m cũ là hard cutoff.",
        "",
        "## Relay dùng chung — router thật",
        "",
        "shared_relay.csv thêm 20 ca hai Access (mỗi chiếc 25 UE) dùng chung một Relay. "
        "Kiểm tổng backhaul cấp cho các client không vượt capacity Relay–BS và phân rã loss bảo toàn flow. "
        "Tại một số geometry Access còn có direct fallback; cột relay_clients cho biết số client thực sự dùng Relay.",
        "",
        "## Hybrid thử riêng",
        "",
        "| Profile | Duplex | Chế độ | Tổng Mbps lớn nhất trong lịch thử | Mbps UE từ xa tại lịch đó | Năng lượng hub/slot |",
        "|---|---|---|---:|---:|---:|",
    ]
    for profile in [r["profile"] for r in designs]:
        for duplex in ["ideal_parallel", "HD"]:
            for mode in ["Relay", "Hybrid"]:
                group = [
                    r
                    for r in result["hybrid_airtime"]
                    if r["profile"] == profile and r["duplex"] == duplex and r["mode"] == mode
                ]
                best = max(
                    group, key=lambda r: (r["total_delivered_mbps"], r["remote_delivered_mbps"])
                )
                energy = best["hub_energy_units"]
                energy_text = "chưa hiệu chỉnh" if energy is None else f"{energy:.2f}"
                lines.append(
                    f"| {profile} | {duplex} | {mode} | {best['total_delivered_mbps']:.3f} | "
                    f"{best['remote_delivered_mbps']:.3f} | {energy_text} |"
                )
    lines += [
        "",
        "Hybrid chỉ là abstraction 2 UAV với cùng băng thông/công suất từng radio, hub bật thêm radio Access; "
        "năng lượng radio cộng Access+Relay, hover tính một lần. HD chia thời gian nhận BH và chuyển tiếp; "
        "Access 2 GHz được giả định chạy song song BH 5 GHz. Ideal-parallel là cận lạc quan.",
        "Mỗi profile có cùng hai UAV; Relay thuần không phục vụ UE tại hub. "
        "Scheduler ưu tiên UE local có thể làm UE từ xa thiệt; cần đọc cả hai cột.",
        "Không suy ra Hybrid tốt hơn trên horizon dài: chưa chạy pin, mobility, policy hay Return. "
        "Năng lượng là đơn vị chuẩn hóa, chưa đo phần cứng; tăng công suất BH cần mô hình năng lượng mới.",
        "",
        "## Observation / reward / benchmark tiếp theo",
        "",
        "Observation cũ dùng bản đồ toàn cục tính trực tiếp từ UE thật; "
        "request_load/deficit cũng bắt nguồn từ geometry/capacity chính xác. Không chỉ làm trễ demand map rồi giữ telemetry oracle.",
        "Thiết kế tiếp: map ước lượng + observed mask + tuổi thông tin; telemetry cục bộ/toàn đội có timestamp. "
        "UE chưa kết nối cần discovery/prior riêng; không được tự coi simulator position là báo cáo UE.",
        "Reward cũ vẫn giữ nguyên. Chỉ so U/QoS/outage/fairness sau khi chốt mục tiêu; "
        "không chọn hệ số bằng seed test 260001. Khoảng cách an toàn là constraint, "
        "radio quality dùng capacity/SNR thay vì phạt mọi khoảng cách.",
        "Bốn phương pháp dự kiến: C-R, C-N, Dynamic Role (Dynamic-Recovery), Random (vẫn được dùng Relay). "
        "Hybrid là nhánh thử thêm; không thay Access/Relay mặc định.",
        "Chưa train lại hoặc đổi cấu hình production. Checkpoint cũ chỉ là đối chiếu, "
        "không gọi là policy đã học với tài nguyên/obs/reward mới.",
        "",
        "## File và bằng chứng",
        "",
        "static_scenes.csv: accounting demand = geometry loss + association-cap loss + Access loss + BH loss + delivered. "
        "Đây là phân rã theo allocation hiện tại, không phải oracle tối ưu.",
        "link_budgets.csv: raw capacity và feasibility tách riêng; ngoài hard radius thì rate thực tế là 0 dù Shannon raw còn dương.",
        "design_targets.csv: inverse link budget; hybrid_airtime.csv: toàn bộ lịch airtime; "
        "AUDIT_RECEIPT.json: số ca, thời gian và hash nguồn/cấu hình.",
        "",
        "Nguồn tham khảo cho bước thiết kế, không chứng minh trực tiếp các bán kính đã đặt:",
        "- Al-Hourani et al. (2014), Optimal LAP Altitude for Maximum Coverage, DOI:10.1109/LWC.2014.2342736.",
        "- Fouda et al. (2018), UAV-Based in-band Integrated Access and Backhaul for 5G Communications, arXiv:1807.07230.",
        "- Placement Optimization for UAV-Enabled Wireless Networks with Multi-Hop Backhaul, arXiv:1809.04963.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bs5_resource_audit_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/bs5_resource_audit_v1")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output is nonempty; choose another --output to preserve previous diagnostics")
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_audit(config)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in [
        "static_scenes",
        "shared_relay",
        "link_budgets",
        "design_targets",
        "hybrid_airtime",
    ]:
        rows = result[name]
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (args.output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    sources = [
        "src/diagnostics/bs5_resource_audit.py",
        "scripts/analysis/run_bs5_resource_audit.py",
        "src/envs/bs5_launch_ab_env.py",
        "src/envs/v4_fixed_mobile_battery_env.py",
        "src/envs/v4_final_fixed_map_env.py",
        "src/envs/thesis_cartesian_r6_curriculum_env.py",
        "src/envs/v7_r6_macro_relay_stable_env.py",
        "src/envs/v6_r6_macro_relay_mobility_env.py",
        "src/envs/episode_random_failure_env.py",
    ]
    receipt = dict(
        **result["receipt"],
        wall_seconds=time.perf_counter() - start_wall,
        cpu_seconds=time.process_time() - start_cpu,
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
    )
    for name, value in [
        ("AUDIT_RECEIPT", receipt),
        ("resolved_config", result["baseline_config"]),
        ("audit_config", config),
    ]:
        (args.output / f"{name}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (args.output / "REPORT_VI.md").write_text(report_text(result), encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), **receipt), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
