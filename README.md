# UAV QMIX — bản phát triển chính

Mô phỏng năm UAV phục vụ 90 UE. Bản phát triển chính nằm ngay trong repo `uav_thesis/` này.

## Mở môi trường Windows PowerShell

Mở PowerShell tại thư mục chứa README này:

```powershell
conda activate uav_sim_py311
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
```

Dùng Python 3.11 và các phiên bản trong [requirements.txt](requirements.txt). Mọi lệnh dưới đây chạy từ thư mục gốc này, trừ đoạn có `Push-Location`.

## Train → eval → resume

Ví dụ train QMIX-AR, seed 551, ngân sách 399.600 bước trên CPU:

```powershell
python run.py train qmix-ar --seed 551 --steps 399600 --output runs/qmix-ar-s551
python run.py eval --checkpoint runs/qmix-ar-s551/best.pt --test-seed 930201 --slots 10000 --output runs/qmix-ar-s551-eval
```

Tên arm: `qmix-a` (C-N), `qmix-ar` (C-R), `qmix-arh` (C-H), `qmix-h` (H-only). Chọn thư mục train mới cho mỗi lượt. `best.pt` được chọn theo reward validation; giữ toàn bộ thư mục train, gồm `resolved_config.json`, `protocol.json`, `status.json`, `resume.pt` và `selected/` khi di chuyển hoặc tiếp tục model.

Nhấn Ctrl+C một lần để dừng ở cuối episode, hoặc thêm `--pause-at-step 6000` ngay từ đầu. Tiếp tục ngân sách ban đầu:

```powershell
python run.py train qmix-ar --resume --output runs/qmix-ar-s551
```

Eval nạp đúng đường dẫn checkpoint và kiểm hash source/config/protocol/selection. Để dừng eval có lưu trạng thái, thêm `--stop-after-slots 500`; tiếp tục bằng lệnh eval cũ, cùng output và bỏ cờ dừng. Benchmark có cùng cách dừng/tiếp tục.

Cấu hình nằm ở [environment.json](configs/environment.json), [training.json](configs/training.json), [evaluation.json](configs/evaluation.json). `scripts/train.py` và `scripts/evaluate.py` dùng chung logic với các bí danh `run.py`. Checkpoint mới dùng source của bản phát triển này; không trộn source của lượt train khác.

## Kiểm tra nhanh

```powershell
python -B -m scripts.verification.smoke_compact --output runs/smoke-01
```

Dùng output chưa tồn tại. Lệnh chạy train diagnostic 96 bước, eval đúng checkpoint vừa tạo 40 slot, ba baseline mỗi loại 8 slot và xuất sáu hình từ CSV đi kèm. Log và `smoke_report.json` nằm dưới output. Diagnostic chỉ kiểm đường chạy, không thay model W00 trong so sánh khoa học.

## Benchmark đối chứng

Ví dụ ngắn trên seed 930201:

```powershell
python -m scripts.w00 benchmark baseline --label RANDOM --part 1 --seeds 930201 --slots 8 --run-id local-demo --output runs/baseline-random
python -m scripts.w00 benchmark baseline --label DYNAMIC_AR --part 1 --seeds 930201 --slots 8 --run-id local-demo --output runs/baseline-ar
python -m scripts.w00 benchmark baseline --label HEURISTIC_HO --part 1 --seeds 930201 --slots 8 --run-id local-demo --output runs/baseline-h
```

`DYNAMIC_AR` là Heuristic-AR, `HEURISTIC_HO` là Heuristic-H. Đổi `--slots 8` thành `--slots 10000` khi chạy đầy đủ. Partition 1 dùng seed 930201–930205, partition 2 dùng 930206–930210; bỏ `--seeds` để chạy cả partition, dùng output riêng.

## Vẽ sáu hình Chương 5

Bốn CSV W00 đi kèm ở `results/w00_main/`, đã kiểm SHA-256 với repo gốc. Các lệnh chỉ đọc CSV, không cần checkpoint hoặc raw:

```powershell
python -m scripts.plotting.chapter5.plot_learning --output runs/chapter5
python -m scripts.plotting.chapter5.plot_demand_breakdown --output runs/chapter5
python -m scripts.plotting.chapter5.plot_throughput_blocks --output runs/chapter5
python -m scripts.plotting.chapter5.plot_ue_service --output runs/chapter5
python -m scripts.plotting.chapter5.plot_operation --output runs/chapter5
python -m scripts.plotting.chapter5.plot_throughput_cost --output runs/chapter5
```

Mỗi hình có PNG, PDF, CSV và JSON ghi nguồn. Chọn output mới hoặc thêm `--overwrite` để vẽ lại. Output train/eval mới không tự thay bốn CSV khóa luận. Không nhúng ảnh sinh ra trong README để tránh liên kết phụ thuộc vào một lượt chạy cục bộ.

## Checkpoint W00 lịch sử

Bố cục kho trên máy hiện tại, tính từ thư mục gốc repo này:

| Nguồn | Đường dẫn |
|---|---|
| Source trước khi áp dụng bản gọn | `archive/before_compact_apply_2026-11-24/` |
| Bản nộp | `../kltn2026-uav-qmix-restoration/` |
| Source W00 nguyên vẹn | `../drive_upload/source_w00/` |
| 12 model W00, protocol/status và manifest | `../drive_upload/checkpoints_w00/` |

Checkpoint W00 lịch sử phải dùng `source_w00/` và metadata tương ứng. Không bỏ kiểm hash để nạp bằng source mới. Ví dụ từ thư mục phát triển:

```powershell
Push-Location ../drive_upload/source_w00
$env:PYTHONPATH = "$PWD/scripts/compat/windows;$PWD"
python run.py eval --checkpoint ../checkpoints_w00/models/C_R_S551/best.pt --test-seed 930201 --slots 10000 --output runs/eval-cr-s551
python -m scripts.w00 benchmark drl --arm C-R --training-seed 551 --models ../checkpoints_w00 --part 1 --seeds 930201 --slots 10000 --run-id history-cr-s551 --output runs/benchmark-cr-s551
Remove-Item Env:PYTHONPATH
Pop-Location
```

Lệnh lịch sử ghi output riêng dưới `source_w00/runs/`. Giữ raw/evidence/checkpoint ở kho ngoài bản phát triển. Khi giải nén ZIP ở nơi khác, điều chỉnh các đường dẫn kho lịch sử tương đối trong mục này; train mới và vẽ sáu hình vẫn tự chạy bằng nội dung ZIP.

## Đóng gói và Git Windows

```powershell
python -B -m scripts.packaging.export_source --with-w00-csv
```

ZIP làm việc ở `archive/packages/uav_thesis_work_<ngày>.zip` (hậu tố số nếu trùng tên), có thư mục gốc `uav_thesis/`. Gói gồm source, config, test, README, `.gitattributes` và đúng bốn CSV; không có checkpoint, raw, evidence hoặc output kiểm thử. Receipt cạnh ZIP ghi CRC và SHA-256 từng file, cùng kết quả import từ bản giải nén. Bỏ `--with-w00-csv` nếu chỉ cần ZIP mã nguồn.

Sau khi giải nén, mở PowerShell tại thư mục chứa `run.py`, kích hoạt Conda và chạy lệnh smoke ở trên. Đã kiểm các lệnh diagnostic, eval, benchmark ngắn và sáu hình bằng bản giải nén; ví dụ 399.600 bước/10.000 slot dành cho lượt thí nghiệm đầy đủ.

Giữ nguyên [.gitattributes](.gitattributes): `-text` ngăn Git chuyển LF/CRLF, kể cả khi `core.autocrlf=true`, để bảo toàn hash source/protocol/CSV. `.gitignore` chỉ cho phép đúng bốn CSV W00 trong results; các output và tài liệu cục bộ tiếp tục bị bỏ qua.
