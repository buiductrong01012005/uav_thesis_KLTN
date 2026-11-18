# UAV Thesis — Huấn luyện và đánh giá QMIX

## Chạy nhanh trên Windows

```powershell
conda activate uav_sim_py311
Set-Location 'F:\Study\cloud\uavThesis\uav_thesis'
$env:PYTHONDONTWRITEBYTECODE = '1'

python run.py train qmix-a --seed 551 --steps 120000 --output runs/qmix-a/seed-551
python run.py eval --checkpoint runs/qmix-a/seed-551/best.pt --test-seed 930201 --output runs/qmix-a/seed-551/eval/seed-930201
```

Dùng môi trường Conda `uav_sim_py311` đã cài [requirements.txt](requirements.txt) và chạy các lệnh từ thư mục repo.
Hai lệnh đầu huấn luyện QMIX-A rồi đánh giá trực tiếp `runs/qmix-a/seed-551/best.pt` của lượt vừa chạy.
Train/eval qua `run.py` và benchmark qua `scripts.w00` tự bật adapter Windows.

## Vẽ lại sáu hình Chương 5

Các script riêng trong `scripts/plotting/chapter5/` đọc CSV W00 đã lưu và có thể chỉnh trực tiếp trong code. Hình 5.1 dùng dữ liệu kiểm định từ ba seed huấn luyện đến bước 399.600. Các hình dùng `per_run.csv` cần đủ 150 lượt đánh giá. Lượt train/eval mới không tự cập nhật CSV khóa luận. Chạy các lệnh sau để lưu hình vào `runs/local/chapter5-redraw/`.

```powershell
python -m scripts.plotting.chapter5.plot_learning --training results/w00_main/training --output runs/local/chapter5-redraw
python -m scripts.plotting.chapter5.plot_demand_breakdown --input results/w00_main/benchmark --output runs/local/chapter5-redraw
python -m scripts.plotting.chapter5.plot_throughput_blocks --input results/w00_main/benchmark --output runs/local/chapter5-redraw
python -m scripts.plotting.chapter5.plot_ue_service --input results/w00_main/benchmark --output runs/local/chapter5-redraw
python -m scripts.plotting.chapter5.plot_operation --input results/w00_main/benchmark --output runs/local/chapter5-redraw
python -m scripts.plotting.chapter5.plot_throughput_cost --input results/w00_main/benchmark --output runs/local/chapter5-redraw
```

| Hình | Dữ liệu đầu vào | File PDF trong thư mục xuất |
| --- | --- | --- |
| 5.1 – Phần thưởng kiểm định | `results/w00_main/training/learning_mean_sd.csv` | `learning.pdf` |
| 5.2 – Phân rã nhu cầu 360 Mbps | `results/w00_main/benchmark/aggregate.csv` | `W00/02b_phan_ra_nhu_cau.pdf` |
| 5.3 – Thông lượng theo khối 1.000 slot | `results/w00_main/benchmark/temporal_1000slot.csv` | `W00/02a_thong_luong_theo_khoi.pdf` |
| 5.4 – Số UE đạt ít nhất 1 Mbps theo slot và trong cửa sổ 5 phút | `results/w00_main/benchmark/per_run.csv` | `W00/04_ue_1mbps_slot_va_5phut.pdf` |
| 5.5 – Yêu cầu di chuyển bị chặn và điện năng trên mỗi Mbit | `results/w00_main/benchmark/per_run.csv` | `W00/03_benchmark_van_hanh.pdf` |
| 5.6 – Thông lượng theo điện năng và thời gian ra quyết định | `results/w00_main/benchmark/per_run.csv` | `W00/05_thong_luong_chi_phi.pdf` |

Mỗi lệnh xuất PDF, PNG, CSV số liệu và JSON ghi thông tin về dữ liệu đầu vào. Nếu bỏ `--output`, hình ghi vào `figures/w00_main/chapter5/`. Dùng `--overwrite` để cập nhật hình đã có.

## Tham số train và eval

| Tên dùng trong lệnh train | Mã trong dữ liệu | Tên trong khóa luận | Vai trò của UAV |
| --- | --- | --- | --- |
| `qmix-a` | C-N | QMIX-A | Chỉ truy nhập |
| `qmix-ar` | C-R | QMIX-AR | Chọn truy nhập hoặc chuyển tiếp |
| `qmix-arh` | C-H | QMIX-ARH | Chọn truy nhập, chuyển tiếp hoặc kết hợp |
| `qmix-h` | H-only | QMIX-H | Luôn dùng vai trò kết hợp |

Train gọi `scripts/training/run_bs5_fourarm400k.py` qua `scripts.w00`, dùng cấu hình `configs/bs5_fourarm400k.json`. Runner mặc định dùng CPU, preset `service_fair`, seed 551 và 399.600 bước (mốc gọi tắt là 400K). `--steps N` truyền xuống runner thành `--total-steps N`; ngoài diagnostic, `N` phải từ 24.000 trở lên và thỏa `(N - 12000) % 1200 == 0`.

Kết quả nằm ở `runs/qmix-a/seed-551/`. Mỗi lượt train mới cần thư mục chưa tồn tại trong `runs/`. File `best.pt` lưu mô hình được chọn theo phần thưởng kiểm định trung bình.

Để tiếp tục một lượt đang dở, giữ nguyên lệnh và thêm `--resume`. Cần giữ cả thư mục huấn luyện, gồm `resume.pt` và `selected/`.

```powershell
python run.py train qmix-a --seed 551 --steps 120000 --output runs/qmix-a/seed-551 --resume
```

Eval gọi `scripts/evaluation/eval_w00_checkpoint.py`, nạp đúng tệp `.pt` được chỉ định và đối chiếu hash/protocol/selection; không nhận ZIP. Giữ `protocol.json` cùng thư mục với `best.pt` khi sao chép checkpoint sang máy khác. Lệnh này dùng cho bốn cấu hình QMIX có CNN trong thí nghiệm W00 chính. Seed kiểm thử phải nằm trong `reserved_test_seeds` của `protocol.json`, thường là 930201–930210.

Mặc định, lượt đánh giá chạy 10.000 slot; đổi bằng `--slots`. Nếu bỏ `--output`, kết quả ghi vào `eval/seed-<test-seed>/` trong thư mục train. Với ví dụ đầu README, thư mục `runs/qmix-a/seed-551/eval/seed-930201/seed930201/` chứa `summary.json`, `slots.csv` và `trace.npz`. Output eval phải nằm dưới `runs/`.

## Chạy benchmark W00

Lệnh benchmark QMIX đọc trực tiếp ZIP ở `--models`, mặc định là `checkpoints/w00_main/FOURARM400K_FROZEN_MODELS.zip`. Khi đánh giá một file `best.pt` riêng, dùng lệnh eval ở mục trước.

Ví dụ dưới đây chạy QMIX-AR (mã `C-R`, seed huấn luyện 551) và Random trên cùng seed kiểm thử 930201. Mỗi lượt chạy 100 slot để kiểm tra.

```powershell
python -m scripts.w00 benchmark drl --arm C-R --training-seed 551 --models checkpoints/w00_main/FOURARM400K_FROZEN_MODELS.zip --part 1 --seeds 930201 --slots 100 --run-id local_demo --output runs/local/benchmark-demo/qmix-ar
python -m scripts.w00 benchmark baseline --label RANDOM --part 1 --seeds 930201 --slots 100 --run-id local_demo --output runs/local/benchmark-demo/random
```

Bỏ `--slots` để chạy đủ 10.000 slot. Bỏ `--seeds` để chạy cả nhóm của `--part`. Nhóm 1 gồm 930201–930205, nhóm 2 gồm 930206–930210. Lệnh chạy đối chứng nhận `RANDOM` cho Random, `DYNAMIC_AR` cho Heuristic-AR và `HEURISTIC_HO` cho Heuristic-H. Cấu hình đánh giá nằm trong `configs/bs5_fourarm400k_heuristic_benchmark.json`.

Hai lượt chạy thử ghi kết quả riêng trong `runs/local/benchmark-demo/`.

Eval và benchmark lưu thêm các gói `*_REVIEW.zip`, `*_RESUME.zip` và file `*_INDEX.json` cạnh thư mục kết quả. Để dừng có chủ đích, dùng `--stop-after-slots`. Khi tiếp tục, bỏ cờ này, giữ nguyên thư mục kết quả và các tham số của lượt chạy. Nếu chuyển sang máy khác bằng gói `*_RESUME.zip`, giải nén để khôi phục thư mục kết quả trước khi chạy lại lệnh.

## Tổng hợp lại bảng W00 từ dữ liệu gốc

Hai lệnh này đọc kết quả đã lưu và ghi bảng mới vào `runs/local/`, không train lại. Các script tổng hợp chưa tự bật adapter Windows; đặt `PYTHONPATH` như dưới đây để dùng adapter `fcntl` có sẵn:

```powershell
$env:PYTHONPATH = "$PWD/scripts/compat/windows;$PWD"
python -m scripts.analysis.report_final400k --input results/w00_main/raw --output runs/local/w00-raw-rebuilt
python -m scripts.analysis.compare_fourarm400k_multiseed --input results/w00_main/training_runs --output runs/local/w00-training-rebuilt --freeze-models
Remove-Item Env:PYTHONPATH
```

## Các thư mục chính

| Đường dẫn | Nội dung |
| --- | --- |
| [run.py](run.py), [scripts/w00.py](scripts/w00.py) | Lệnh huấn luyện, đánh giá checkpoint và chạy benchmark trên Windows |
| [src/envs/](src/envs/), [src/rl/](src/rl/) | Môi trường mô phỏng UAV và mã QMIX |
| [scripts/](scripts/README.md), [configs/](configs/) | Script chạy thí nghiệm, vẽ hình, kiểm tra kết quả và các tệp cấu hình |
| `checkpoints/w00_main/FOURARM400K_FROZEN_MODELS.zip` | 12 checkpoint của QMIX-A, QMIX-AR, QMIX-ARH và QMIX-H, mỗi cấu hình có ba seed huấn luyện |
| `results/w00_main/training/` | Bảng tổng hợp đường học và kiểm định của 12 lượt huấn luyện W00 |
| `results/w00_main/benchmark/` | CSV tổng hợp 150 lượt kiểm thử của bốn cấu hình QMIX và ba đối chứng |
| `results/w00_main/training_runs/` | Dữ liệu gốc của 12 lượt huấn luyện W00: log, checkpoint, protocol/status |
| `results/w00_main/raw/` | Kết quả kiểm thử gốc theo từng slot của 150 lượt W00 |
| `checkpoints/w00_ablation/W00_CR_mlp_S551/` | QMIX-AR thay CNN bằng MLP: `best.pt`, protocol/status/log. Seed 552/553 ở thư mục cùng mẫu tên |
| `checkpoints/w00_ablation/W00_CR_no_curriculum_S551/` | QMIX-AR bỏ huấn luyện tăng dần độ khó: `best.pt`, protocol/status/log. Seed 552/553 ở thư mục cùng mẫu tên |
| `results/w00_ablation/` | Kết quả đánh giá hai biến thể với QMIX-AR đầy đủ: `evidence/summary.csv`, `evidence/paired_by_run.csv` và setup/audit |
| `results/w00_ablation/checkpoint_verification.json`, `results/w00_ablation/evidence/raw/` | Receipt và sáu `*_EVAL.zip` mà verifier đọc trực tiếp |
| `results/appendix_w05/`, `checkpoints/appendix_w05/` | Dữ liệu và model W05 |
| `results/appendix_reward_sweep/`, `checkpoints/appendix_reward_sweep/` | Dữ liệu và model khảo sát trọng số phần thưởng W00–W08 |
| `figures/w00_main/thesis_v12/` | Hình benchmark phục vụ của bản v12, vẽ bằng `scripts/plotting/plot_w00_7cfg.py` |
| `docs/thesis/` | Bản LaTeX (9) và PDF (35) của khóa luận |
| `archive/frozen_sources/w00_ablation/extracted/uav_thesis/` | Nguồn cố định cho train lại ablation qua `scripts.w00 train --variant mlp` hoặc `--variant no_curriculum` |
| `runs/` | Kết quả các lượt chạy mới. Các ví dụ kiểm tra và vẽ lại trong README dùng `runs/local/` |

Danh mục đường dẫn: [repository_paths.json](configs/repository_paths.json),
[w00_assets.json](configs/w00_assets.json).

## Thử nhanh với 96 bước huấn luyện và 40 slot

Hai lệnh dưới đây huấn luyện 96 bước rồi nạp `best.pt` vừa tạo để đánh giá 40 slot. Dùng thư mục mới mỗi lần thử. Chế độ `--diagnostic` có thể chọn checkpoint ở bước 0 vì mục đích của lượt này là kiểm tra việc lưu và nạp mô hình.

```powershell
python run.py train qmix-a --seed 551 --steps 96 --diagnostic --output runs/local/readme-smoke/train
python run.py eval --checkpoint runs/local/readme-smoke/train/best.pt --test-seed 930201 --slots 40 --output runs/local/readme-smoke/eval
```

## Kiểm tra checkpoint và kết quả W00

Lệnh sau kiểm tra bảng tổng hợp của 150 lượt W00, nạp 12 checkpoint chính và đối chiếu sáu checkpoint QMIX-AR ablation với sáu gói đánh giá gốc.

```powershell
python -m scripts.verification.verify_w00_import --output runs/local/w00-check/verification.json
```

Để đối chiếu cả kết quả gốc theo từng slot, chạy thêm lệnh sau.

```powershell
python -m scripts.verification.check_final400k --input results/w00_main/benchmark --raw results/w00_main/raw --output runs/local/w00-check/raw.json
```

## Đóng gói mã nguồn

```powershell
python -B -m scripts.packaging.export_source
```

Lệnh tạo ZIP mã nguồn mới và receipt `.verification.json` trong `archive/packages/`, kiểm tra CRC/hash, đủ `src/envs/`, rồi import train/eval từ bản giải nén riêng.

Trên máy khác, giải nén ZIP mã nguồn rồi chạy từ thư mục chứa `run.py`. ZIP không chứa checkpoint, kết quả, tài liệu hoặc nguồn ablation trong `archive/`; cần chép thêm các đầu vào tương ứng ở mục tra cứu để benchmark, vẽ hình hoặc train ablation.
