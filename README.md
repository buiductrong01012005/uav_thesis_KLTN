# UAV Thesis

Mã nguồn mô phỏng mạng UAV, huấn luyện học tăng cường đa tác tử và đánh giá các chính sách phục vụ UE. Repo chứa các phiên bản thí nghiệm BS5, Hybrid, multi-hop, năng lượng/thay pin và Fair300k.

## Cấu trúc repo

| Đường dẫn | Nội dung |
| --- | --- |
| `src/envs/` | Các môi trường mô phỏng và biến thể thí nghiệm |
| `src/rl/` | QMIX, DQN, MAPPO và các kiến trúc theo môi trường |
| `src/baselines/`, `src/diagnostics/` | Baseline, chỉ số và công cụ chẩn đoán |
| `scripts/training/`, `scripts/experiments/` | Huấn luyện và điều phối thí nghiệm |
| `scripts/evaluation/`, `scripts/analysis/` | Đánh giá checkpoint, kiểm tra dữ liệu và tổng hợp |
| `scripts/verification/` | Preflight, xác minh và xuất gói kết quả |
| `scripts/launchers/`, `scripts/plotting/`, `scripts/validation/` | Chạy nhiều lượt, vẽ biểu đồ và kiểm tra thiết lập |
| `configs/`, `tests/` | Cấu hình, kiểm thử và fixture cần thiết |
| `results/benchmark_assets/` | Checkpoint và metadata benchmark cục bộ |
| `results/`, `checkpoints/` | Kết quả thí nghiệm và checkpoint cục bộ khác |
| `docs/`, `reports/`, `archive/` | Tài liệu, báo cáo, bản sao lưu và ZIP cục bộ |

Git bỏ qua dữ liệu trong `results/`, `checkpoints/`, `docs/`, `reports/`, `archive/`, các notebook và tài liệu cục bộ theo `.gitignore`. Bản clone mới cần được bổ sung đúng bộ checkpoint trước khi đánh giá. Các file đã được Git theo dõi vẫn giữ trạng thái theo dõi dù khớp quy tắc ignore.

## Cài đặt

Dùng Python 3.11 hoặc 3.12 và chạy lệnh từ thư mục gốc repo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Trên Linux/Kaggle, kích hoạt môi trường bằng `source .venv/bin/activate`. Với Kaggle offline, dùng các thư viện đã cài sẵn của Dataset/notebook thí nghiệm. `requirements.txt` là danh sách phụ thuộc chung; các bộ phiên bản lịch sử được ghi chú trong file này. Tái hiện kết quả phải dùng đúng runtime trong protocol đã lưu.

Các file requirements riêng của gói bàn giao được lưu cục bộ trong `archive/dependencies/fair300k_2026-11-06/`. ZIP nhận ban đầu nằm trong `archive/packages/incoming/`.

## Các entry point chính

| Thí nghiệm | Runner | Cấu hình |
| --- | --- | --- |
| Energy/swap 120k | [run_bs5_energy_swap_train.py](scripts/training/run_bs5_energy_swap_train.py) | [bs5_energy_swap_train120k.json](configs/bs5_energy_swap_train120k.json) |
| Energy/collision 120k | [run_bs5_energy_collision_train.py](scripts/training/run_bs5_energy_collision_train.py) | [bs5_energy_collision_train120k.json](configs/bs5_energy_collision_train120k.json) |
| Fair300k | [run_bs5_fair300k.py](scripts/training/run_bs5_fair300k.py) | [bs5_fair300k.json](configs/bs5_fair300k.json) |
| Benchmark Fair300k | [run_fair300k_benchmark.py](scripts/evaluation/run_fair300k_benchmark.py) | [bs5_fair300k_benchmark.json](configs/bs5_fair300k_benchmark.json) |
| Tổng hợp Fair300k | [report_fair300k_benchmark.py](scripts/analysis/report_fair300k_benchmark.py) | Đọc các REVIEW cùng protocol |

Xem tham số bằng `--help`, ví dụ:

```powershell
python -m scripts.training.run_bs5_fair300k --help
python -m scripts.evaluation.run_fair300k_benchmark --help
python -m scripts.analysis.report_fair300k_benchmark --help
```

## Checkpoint benchmark

Bộ Fair300k cục bộ đặt tại:

```text
results/benchmark_assets/fair300k/
  selection_manifest.json
  EMA/
    best.pt
    training_protocol.json
    training_status.json
    validation.csv
  OLD/
    best.pt
    training_protocol.json
    training_status.json
    validation.csv
```

Runner ưu tiên `results/benchmark_assets/<bộ>/` và hỗ trợ vị trí `benchmark_assets/<bộ>/` của Dataset gốc. Các đường dẫn gốc bên trong manifest được ánh xạ khi đọc; nội dung checkpoint, protocol và hash được giữ nguyên. Benchmark bảy phương pháp cũ dùng checkpoint trong `checkpoints/bs5_v4_best120k/`.

Mỗi bộ checkpoint yêu cầu đúng snapshot nguồn đã khóa của nó. Bộ bảy phương pháp lịch sử có thể từ chối bản nguồn mới; dùng snapshot tương ứng khi tái hiện, giữ nguyên các kiểm tra hash.

Fair EMA được khóa ở bước 294.000, Fair OLD ở bước 288.000, cùng seed train 551. Seed test ghép cặp là 930101–930110, mỗi seed 10.000 slot; part 1 dùng năm seed đầu, part 2 dùng năm seed cuối. Không chọn lại checkpoint dựa trên test.

Kiểm tra hash, metadata và khả năng nạp mô hình, không chạy rollout test:

```powershell
python -m scripts.evaluation.run_fair300k_benchmark --label EMA --part 1 --output results/fair300k_check_ema --check-only
python -m scripts.evaluation.run_fair300k_benchmark --label OLD --part 1 --output results/fair300k_check_old --check-only
```

Rollout benchmark dùng khóa tiến trình `fcntl`, cần Linux/Kaggle. Trước lượt test mới, chạy preflight trên đúng runtime đó:

```bash
python -m scripts.verification.verify_fair300k_benchmark --label EMA --output results/preflight_fair300k_ema
python -m scripts.verification.verify_fair300k_benchmark --label OLD --output results/preflight_fair300k_old
```

Preflight dùng diagnostic và validation để kiểm tra; đây không phải kết quả benchmark khoa học. Resume kiểm tra cả nguồn, checkpoint, protocol và runtime. Dùng đúng bản nguồn đã tạo một lượt đang dở; thay đổi runner sẽ làm hash nguồn thay đổi và có thể khiến resume từ bản cũ bị từ chối.

## Kiểm tra mã nguồn

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m pytest -q -p no:cacheprovider tests/test_bs5_temporal_fair.py tests/test_bs5_energy_swap.py tests/test_bs5_energy_collision.py
python -m ruff check --no-cache scripts/evaluation/benchmark_asset_paths.py tests/test_benchmark_asset_paths.py
```

Một số kiểm thử cần checkpoint cục bộ hoặc Linux. Bộ Energy/collision benchmark P03/P05 cần bộ asset riêng; gói Fair300k chỉ chứa EMA/OLD. Chỉ chạy bộ test tương ứng với dữ liệu đã cài. Windows có thể hạn chế thao tác thay thế file hoặc tạo thư mục tạm; không coi các kiểm tra bị chặn bởi môi trường là đã đạt.

Các source receipt và protocol xác minh byte nguồn. `.gitattributes` giữ nguyên LF/CRLF của các file được khóa, tránh Git tự đổi dòng và làm sai checksum. Không sửa hash để bỏ qua thay đổi công thức, reward, seed, tham số, checkpoint hoặc giao thức đánh giá.
