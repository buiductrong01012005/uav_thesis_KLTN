# UAV Thesis — Four-arm 400K

## W00 đang dùng trên Windows — tích hợp 09/11/2026

Đường chạy mặc định: `python -m scripts.w00`. Mã khoa học đang chạy trong `src/` trùng byte với phần tương ứng của snapshot W00 chính `archive/frozen_sources/w00_main_returnfix/uav_thesis_source_2026-11-07_2.zip`. Giữ bản vá đường dẫn Windows và receipt đã có trong repo; entrypoint native có mã `w00_native_windows_v1`. Bản ablation có cây nguồn riêng, không nhập mô hình MLP vào `src/` của W00 chính. Các mục 400K phía dưới được giữ nguyên làm tài liệu lịch sử, bao gồm nội dung người dùng đã sửa trước phiên này.

| Đường dẫn | Vai trò |
| --- | --- |
| `checkpoints/w00_main/FOURARM400K_FROZEN_MODELS.zip` | 12 checkpoint chính, manifest và protocol nguyên bản |
| `checkpoints/w00_ablation/W00_CR_{variant}_S{551,552,553}/` | Sáu bộ `best.pt`, protocol/status và log train đã khớp EVAL |
| `results/w00_main/{benchmark,training}/` | 150 lượt đã công bố, bảng tổng hợp và learning |
| `results/w00_ablation/` | 60 lượt ablation, 30 lượt C-R tham chiếu, bảng ghép cặp và chứng cứ |
| `results/w00_ablation/checkpoint_verification.json` | SHA-256 đầy đủ của sáu checkpoint, protocol và ZIP FINAL/EVAL; **6/6 khớp, không thiếu checkpoint** |
| `archive/frozen_sources/{w00_main_returnfix,w00_main_training,w00_ablation,appendix_w05}/` | Snapshot/Dataset nguyên byte, tách lịch sử train, benchmark, ablation và phụ lục |
| `results/{appendix_w05,appendix_reward_sweep}/`, `checkpoints/{appendix_w05,appendix_reward_sweep}/` | W05 và khảo sát trọng số, giữ riêng với W00 chính |
| `notebooks/kaggle/{w00_main,w00_ablation,appendix_w05}/` | Notebook gốc và hướng dẫn |
| `figures/w00_main/thesis_v12/` | Hình/script v12 bàn giao; script chạy từ repo tại `scripts/plotting/plot_w00_7cfg.py` |
| `archive/historical_artifacts/w00_legacy_plot_bundle_2026-11-09/` | Tám tệp hình/script cũ đã chuyển, kiểm tra hash trước/sau |
| `archive/packages/incoming/w00_windows_2026-11-09/` | ZIP chính và 12 ZIP FINAL/EVAL đầu vào nguyên bản |
| `runs/local/<run_id>/` | Mọi lượt chạy mới, tách số liệu đã công bố |
| `configs/w00_assets.json` | Danh mục nguồn, hash snapshot và cách tổng hợp hình |
| `archive/imports/w00_windows_2026-11-09/` | So sánh trước nhập, backup README, manifest chuyển tệp và log kiểm thử thực tế |

Các bản `results/final400k/`, `results/benchmark_assets/`, `results/drl400k_training/` và gói nhập trước vẫn được cấu hình, test, notebook hoặc tài liệu tham chiếu nên giữ nguyên. Cả 98 đường dẫn trong source lock được giữ, kể cả mô-đun mang tên phiên bản cũ. Xem `retained_source_references.json` trong audit. Sáu EVAL nhận riêng đã được so hash và gộp với đúng bản chứng cứ cùng byte. Không xóa dữ liệu hoặc bản chưa rõ vai trò.

### Môi trường thực tế

Chạy PowerShell tại repo root. Phiên này đã tạo `.venv` từ Conda `uav_sim_py311` bằng `--system-site-packages`, dùng thư viện đã cài để kiểm thử offline. Venv này phụ thuộc môi trường Conda gốc; máy mới cần cài theo `requirements.txt` và chọn runtime lịch sử khi tái hiện số liệu đã công bố.

```powershell
$Py = Join-Path $PWD '.venv\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:MPLBACKEND = 'Agg'
$env:PYTHONPATH = "$PWD/scripts/compat/windows;$PWD"
& $Py -m pip check
$Run = 'runs/local/' + (Get-Date -Format 'yyyyMMdd_HHmmss')
```

Đã dùng Windows, Python **3.11.15**, PyTorch **2.13.0+cpu**, NumPy **2.4.6**, SciPy **1.17.1**, pandas **3.0.5**, Matplotlib **3.11.1**, filelock **3.29.7**. Đây là runtime kiểm tra tương thích; protocol lịch sử ghi runtime khác và không được sửa. Chưa kiểm thử WSL2 trong phiên này. Snapshot Linux/Kaggle nguyên bản vẫn nằm trong ZIP, hash trong `configs/w00_assets.json`; phải giải nén vào cây riêng và dùng runtime gốc cho tái lập lịch sử. Không dùng adapter native để resume lượt lịch sử khác fingerprint.

### Train từ đầu N bước và resume

Cấu hình chính: `configs/bs5_fourarm400k.json`; `--arm` chọn C-N/C-R/C-H/H-only, `--preset service_fair`, CPU. N phải ≥24.000 và `(N - 12000) % 1200 == 0`; runner từ chối N sai, không làm tròn. Mốc công bố là 399.600. Model mới của budget khác không thuộc manifest benchmark 400K.

```powershell
$N = 24000
& $Py -m scripts.w00 train --variant main --arm C-R --training-seed 551 --total-steps $N --output "$Run/train"
# Với C-R, có thể chọn --variant mlp hoặc --variant no_curriculum.
# Hai biến thể dùng snapshot ablation riêng; protocol native ghi hash adapter.

# Smoke riêng 96 bước; dừng đúng biên episode 48 rồi tiếp tục cùng thư mục:
& $Py -m scripts.w00 train --arm C-R --training-seed 551 --diagnostic --pause-at-step 48 --output "$Run/train_smoke"
& $Py -m scripts.w00 train --arm C-R --training-seed 551 --diagnostic --resume --output "$Run/train_smoke"
```

### Benchmark N slot trên mỗi seed

`--slots` tạo horizon tùy biến hoàn chỉnh, ghi horizon, run ID, hash adapter và fingerprint vào protocol; vẫn xác minh checkpoint qua manifest. Số slot không đổi bước train checkpoint. `--stop-after-slots` chỉ tạm dừng rồi resume bằng cùng lệnh bỏ cờ dừng. Phải giữ `--run-id`, horizon, seed, model, nguồn và runtime khi resume. Dùng cùng run ID cho một nhóm cấu hình muốn so sánh; dùng output riêng từng cấu hình/seed train/part. Hash UE chỉ so giữa các lượt cùng seed và cùng horizon; không so trace 40 slot với hash lịch sử 10.000 slot. Metric cửa sổ dài giữ định nghĩa gốc, có thể null nếu dữ liệu ngắn.

```powershell
$N = 100
& $Py -m scripts.w00 benchmark drl --arm C-R --training-seed 551 --part 1 --seeds 930201 --slots $N --run-id local_custom --output "$Run/benchmark"
& $Py -m scripts.w00 benchmark baseline --label RANDOM --part 1 --seeds 930201 --slots $N --run-id local_custom --output "$Run/random"
# Bỏ --slots để giữ horizon chuẩn 10.000; part 1/2 là hai nhóm 5 seed.
# Chẩn đoán ngắn đã chạy: --diagnostic --seeds 940201 --slots 40.
```

Đường runner gốc vẫn giữ: `python -m scripts.evaluation.run_fourarm400k_returnfix drl` hoặc `baseline`, với `--models checkpoints/w00_main/FOURARM400K_FROZEN_MODELS.zip` cho DRL. Runner gốc giữ giao thức 10.000 slot và chẩn đoán 96 slot; không nhận `--slots`. Phiên này không chạy lại 150 lượt benchmark hoặc train đủ 399.600 bước.

### Hình và metric từ CSV

```powershell
New-Item -ItemType Directory -Force -Path "$Run/figures_v12" | Out-Null
& $Py scripts/plotting/plot_w00_7cfg.py results/w00_main/benchmark "$Run/figures_v12"
& $Py -m scripts.plotting.plot_final400k --input results/w00_main/benchmark --training results/w00_main/training --output "$Run/figures_legacy"
& $Py -m scripts.plotting.plot_w00_metric --input results/w00_main/benchmark/per_run.csv --metric joules_per_mbit --unit J/Mbit --by seed --output "$Run/metric_seed"
& $Py -m scripts.plotting.plot_w00_metric --input results/w00_main/benchmark/temporal_1000slot.csv --metric throughput_mbps --unit Mbps --by time --x block --output "$Run/metric_time"
# Theo lượt: dùng per_run.csv với --by run. Script yêu cầu cột tương ứng.
```

V12 giữ trung bình số đo J/Mbit theo lượt (gộp các seed train trong mỗi seed test); hình lịch sử giữ tỷ số gộp và các hình learning/waterfall. Hai cách tổng hợp không thay thế nhau. Script metric ghi `plotted_values.csv` và `method.json`, không suy ngược dữ liệu theo slot từ bảng đã tổng hợp.

### Kiểm thử và bằng chứng thực tế

Chạy lại toàn bộ pipeline ngắn bằng một lệnh dưới đây. Script chạy train 96 bước, so sánh resume từ bước 48, benchmark 40 slot với dừng/resume và chạy lặp, kiểm tra trace/metric rồi vẽ hình. `--with-ablation` cần snapshot ablation đã khôi phục; `--with-tests` chạy thêm các test hồi quy hiện có. Dùng output mới; lệnh ghi từng argument, mã thoát và log vào `smoke_report.json` cùng thư mục `logs/`.

```powershell
& $Py -m scripts.verification.smoke_w00_pipeline --output "$Run/pipeline_smoke" --with-ablation --with-tests
```

Đã chạy lại trên Windows ngày **10/11/2026** tại `runs/local/w00_pipeline_20261110/`: train chính, MLP và no-curriculum hoàn tất 96 bước; resume train từ bước 48 khớp bitwise; benchmark dừng 17 rồi hoàn tất 40 slot, chạy lặp khớp dữ liệu khoa học/trace và baseline khớp hash UE cùng horizon; vẽ hình thành công; **43 test hồi quy đạt**. Chi tiết từng lệnh và mã thoát nằm trong `smoke_report.json`. Benchmark ngắn vẫn dùng checkpoint đóng băng trong manifest; đây là kiểm tra pipeline, không thay kết quả đã công bố.

```powershell
& $Py -m scripts.verification.verify_w00_import --output "$Run/import_verified.json"
& $Py -m scripts.verification.verify_bs5_fourarm400k --arm C-R --output "$Run/train_verify"
& $Py -m scripts.verification.verify_fourarm400k_drl_benchmark --arm C-R --training-seed 551 --models checkpoints/w00_main/FOURARM400K_FROZEN_MODELS.zip --output "$Run/drl_verify"
& $Py -m pytest -q -p no:cacheprovider tests/test_bs5_lp_roundoff.py tests/test_bs5_lpfix_io.py tests/test_fourarm400k_return_summary.py tests/test_final400k_delivery.py tests/test_fourarm400k_multiseed.py tests/test_fourarm400k_heuristic_io.py tests/test_bs5_fourarm400k_io.py
```

Lệnh, mã thoát và log thực tế nằm trong `validation_commands.json` và `validation_train_commands.json` của audit. Lỗi đầu tiên do ACL `tempfile` trong sandbox (`WinError 5`), đã chạy lại ngoài sandbox với thư mục tạm riêng. Xem báo cáo bàn giao `provenance/W00_WINDOWS_HANDOFF.md` để biết kết quả cuối, các kiểm tra số liệu và gói mã nguồn. Không có checkpoint ablation còn thiếu; raw REVIEW train/resume lịch sử đầy đủ vẫn là hiện vật riêng, không suy ra từ FINAL.

---


Repo tập trung vào bốn cấu hình **C-N, C-R, C-H và H-only**, mỗi cấu hình có ba seed huấn luyện **551/552/553**. Mốc 400K của protocol là **399.600 bước môi trường**. Đánh giá cuối giữ cả ba baseline RANDOM, DYNAMIC_AR và HEURISTIC_HO để so sánh khoa học: **150 lượt, bảy cấu hình, 10 seed kịch bản 930201–930210**.

| Đường dẫn | Vai trò hiện tại |
| --- | --- |
| `src/envs/bs5_fourarm400k.py`, `src/rl/bs5_fourarm400k_qmix.py` | Môi trường và mô hình Four-arm |
| `scripts/training/`, `scripts/evaluation/` | Huấn luyện, nạp checkpoint và benchmark |
| `scripts/analysis/`, `scripts/plotting/` | Tổng hợp log, kiểm tra số liệu và vẽ hình |
| `scripts/verification/`, `tests/`, `configs/` | Kiểm tra nguồn, checkpoint, LPFIX/RETURNFIX và cấu hình |
| `results/benchmark_assets/fourarm400k/` | 12 checkpoint đã chốt và metadata |
| `results/final400k/{benchmark,raw,training}/` | CSV/audit được giao, 150 raw benchmark và bảng learning |
| `results/final400k/{raw_rebuilt,training_rebuilt,figures_rebuilt}/` | Kết quả tái tổng hợp và hình đã kiểm tra local |
| `results/drl400k_training/` | REVIEW train seed 551 đầy đủ và log/model rút gọn đa seed |
| `archive/packages/incoming/final400k/` | ZIP gốc đã kiểm tra checksum |
| `archive/legacy_pre400k_2026-11-08/repo/` | Thí nghiệm cũ đã chuyển khỏi cây làm việc chính, giữ đường dẫn tương đối |

Git bỏ qua `results/`, `checkpoints/`, `docs/`, `reports/`, `archive/`, notebook và tài liệu cục bộ theo `.gitignore`. Bản clone/ZIP mã nguồn cần thêm các gói dữ liệu được liệt kê bên dưới. Các bản README trước khi cập nhật được giữ trong audit local để bảo toàn nội dung đã chỉnh sửa.

## Cài đặt

Dùng Python 3.11 hoặc 3.12 và chạy lệnh từ thư mục gốc repo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Trên Linux/Kaggle, kích hoạt môi trường bằng `source .venv/bin/activate`. Với Kaggle offline, dùng các thư viện đã cài sẵn của Dataset/notebook thí nghiệm. `requirements.txt` là danh sách phụ thuộc chung; các bộ phiên bản lịch sử được ghi chú trong file này. Tái hiện kết quả phải dùng đúng runtime trong protocol đã lưu.

Các file requirements riêng của gói bàn giao được lưu cục bộ trong `archive/dependencies/fair300k_2026-11-06/`. ZIP nhận ban đầu nằm trong `archive/packages/incoming/`.

### Conda trên Windows và chạy ngắn

Với môi trường đã có `uav_sim_py311`, kích hoạt từ PowerShell tại gốc repo:

```powershell
conda activate uav_sim_py311
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "$PWD/scripts/compat/windows;$PWD"
python -m pip install -r requirements.txt
python -m pip check
```

Chỉ cần cài requirements khi thiếu phụ thuộc hoặc tạo môi trường mới. Tạo môi trường mới bằng `conda create -n uav_sim_py311 python=3.11 pip` nếu tên này chưa tồn tại. Các thư viện chạy chính là NumPy, SciPy, PyTorch, pandas, matplotlib và filelock; pytest/Ruff phục vụ kiểm tra. Môi trường Conda không thay thế cơ chế tương thích hệ điều hành: `fcntl` là module Unix, không cài bằng pip trên Windows. Dòng `PYTHONPATH` bật adapter `scripts/compat/windows/fcntl.py` cho các runner đóng băng. Riêng `check_final400k.py` tự bật adapter trên Windows, kể cả khi chưa đặt `PYTHONPATH`.

Đã chạy thực tế trên Conda `uav_sim_py311` (Python 3.11.15): `pip check` đạt, nạp đủ 12 model khi không đặt `PYTHONPATH`, train diagnostic hoàn tất 96 bước/13 update, benchmark DRL dừng và lưu đúng 100 slot, sáu test CSV đạt. Log tại `archive/imports/conda_short_2026-11-08/`. Đây là kiểm tra tương thích local; tái hiện kết quả lịch sử vẫn dùng runtime ghi trong protocol.

Runner train giữ giao thức curriculum và yêu cầu ngân sách mới ít nhất 24.000 bước, sau 12.000 phải gồm trọn các episode 1.200 bước. Do đó `--total-steps 10000` bị từ chối; `--diagnostic` là bài kiểm tra 96 bước, không phải 10.000 bước. Để chạy khoảng 10.000 bước của lịch gốc rồi lưu đầy đủ optimizer/replay/RNG:

```powershell
python -m scripts.training.run_bs5_fourarm400k --arm C-H --training-seed 552 --pause-at-step 10000 --output results/train_short/CH552_prefix10k
```

Lệnh này dừng ở **10.050 bước**, tại biên episode đầu tiên đạt mốc 10.000, với trạng thái `paused_at_episode_boundary`; tổng mục tiêu trong protocol vẫn 399.600. Đây mới là phần đầu curriculum, chưa phải huấn luyện xong full task. Lịch validation gốc mới chạy mốc 0 trước 12.000 bước, vì vậy `best.pt` lúc này vẫn là checkpoint được chọn ở bước 0; trọng số hiện tại, optimizer và replay được lưu trong `resume.pt`. Muốn tiếp tục, dùng cùng output và `--resume`, bỏ `--pause-at-step`.

Benchmark đúng 100 slot của một seed, dùng checkpoint 400K đã chốt:

```powershell
python -m scripts.evaluation.run_fourarm400k_returnfix drl --arm C-H --training-seed 552 --models results/benchmark_assets/fourarm400k --part 1 --seeds 930201 --stop-after-slots 100 --output results/benchmark_short/CH552_100slots
```

Kết quả ghi `slots=100`, `target_slots=10000`, `complete=false`; đây là snapshot benchmark chạy dở, không phải kết quả hoàn tất 100 slot hay bộ 150 lượt cuối. Chạy lại cùng lệnh và bỏ `--stop-after-slots` để tiếp tục đến 10.000 slot; resume yêu cầu cùng nguồn/runtime/model và Windows có hạn chế thay thế ZIP đã ghi trong phần kiểm tra local. Runner frozen chỉ nhận model train đủ 399.600 bước trong catalog, không nhận model train ngắn mới. Để kiểm tra thật ngắn bằng seed diagnostic riêng, dùng `--diagnostic` với output mới: giao thức này hoàn tất **96 slot**. Train/benchmark chính xác ngân sách 10.000/100 với trạng thái hoàn tất cần một giao thức thí nghiệm ngắn riêng; không sửa cấu hình hoặc hash của bộ 400K đã chốt.

## Four-arm 400k: mã nguồn và kết quả theo seed

Mã nguồn mới đã tích hợp trực tiếp vào `src/envs/bs5_fourarm400k.py`, `src/rl/bs5_fourarm400k_qmix.py` và `src/diagnostics/bs5_fourarm400k_metrics.py`. Runner, công cụ xuất/kiểm tra và tổng hợp nằm trong `scripts/`. Các file khoa học được đối chiếu với 98 hash nguồn trong protocol gốc; giữ nguyên reward, tham số, lịch train và tiêu chí chọn checkpoint. Bootstrap notebook có kiểm tra đường dẫn ZIP tương thích Windows/Linux; bản nhận ban đầu được giữ trong `archive/imports/drl400k_2026-11-06/`.

Bản nhập đầu giữ toàn bộ REVIEW **seed train 551**, đủ C-N, C-R, C-H và H-only, theo cấu trúc dưới đây. Bản nhập cuối đã bổ sung **đủ 12 model của seed train 551/552/553**, mỗi lượt hoàn tất **399.600 bước môi trường**. Bộ rút gọn đa seed nằm riêng tại `results/drl400k_training/multiseed/`, giữ nguyên REVIEW đầy đủ của seed 551.

```text
results/drl400k_training/
  seed_551/
    IMPORT_MANIFEST.json
    DRL_CN_S551_400K/
    DRL_CR_S551_400K/
    DRL_CH_S551_400K/
    DRL_H_ONLY_S551_400K/
    comparison/
```

Mỗi thư mục run giữ toàn bộ nội dung REVIEW: checkpoint công khai `best*.pt`, lịch sử `selected/`, protocol, status, log, dữ liệu validation thô và hình. `IMPORT_MANIFEST.json` ghi hash từng file và ZIP gốc. Bốn ZIP kết quả gốc nằm tại `archive/packages/incoming/drl400k/seed_551/`; ZIP nguồn nằm trong thư mục `source/` bên cạnh.

[bs5_fourarm400k_trained_models.json](configs/bs5_fourarm400k_trained_models.json) ánh xạ **seed → arm → checkpoint**, kèm hash và bước chọn theo reward validation. Kết quả và trọng số được giữ dưới `results/`; mã nạp và cấu hình tham chiếu nằm trong repo để tái sử dụng. Bản clone hoặc ZIP mã nguồn cần được bổ sung thư mục kết quả tương ứng.

Kiểm tra hash nguồn, protocol, checkpoint và nạp bốn model bằng PyTorch, không chạy rollout:

```powershell
python -m scripts.evaluation.load_bs5_fourarm400k --training-seed 551 --arm all
```

Trong Python, dùng `from scripts.evaluation.load_bs5_fourarm400k import load_trained_model`, rồi gọi `learner, protocol, entry = load_trained_model("C-H", training_seed=551)`. Đây là model phục vụ inference; dùng observation 173 chiều và action mask của đúng arm trong môi trường Four-arm, trạng thái QMIX 156 chiều. Các REVIEW không chứa `resume.pt`, optimizer, replay hoặc RNG để tiếp tục train chính xác; cần gói RESUME riêng cho mục đích đó.

Tổng hợp đường học riêng seed 551:

```powershell
$env:PYTHONUTF8 = "1"
python -m scripts.analysis.compare_bs5_fourarm400k_training --input results/drl400k_training/seed_551 --output results/drl400k_training/seed_551/comparison
```

Không đưa nhiều seed train vào cùng lệnh tổng hợp trên: lệnh này chỉ tổng hợp validation 720001/720002 riêng seed 551. Pipeline cuối dưới đây tổng hợp ba seed train và đã có benchmark test dành riêng 930201–930210; không chọn lại checkpoint bằng test. Năm notebook và hướng dẫn của bản nhập đầu được lưu tại `archive/legacy_pre400k_2026-11-08/repo/scripts/notebooks/fourarm400k/`; bộ sinh hiện hành là `scripts/notebooks/build_fourarm400k.py`.

## Pipeline 400K cuối: file và cách chạy

Bản mã tham chiếu là `03_SOURCE/uav_thesis_source_2026-11-07_2.zip` trong `archive/packages/incoming/final400k/UAV_BENCHMARK_400K_KAGGLE_CPU_RETURNFIX (2).zip`. Đã nhập phần benchmark đa seed, tổng hợp raw, LPFIX và RETURNFIX còn thiếu. Toàn bộ `src/` trùng byte với bản tham chiếu, gồm LPFIX có sẵn trong gói. Các thay đổi runner chỉ chuẩn hóa dấu `/` khi ghi đường dẫn trên Windows; [receipt tích hợp](configs/bs5_final400k_integration.json) lưu hash trước/sau. Khi tiếp tục snapshot lịch sử, giải nén đúng nguồn tham chiếu và dùng đúng runtime gốc; không dùng runner đã đổi hash để resume snapshot cũ.

| Bước | File chính | Đầu vào | Cách chạy từ gốc repo | Đầu ra |
| --- | --- | --- | --- | --- |
| Train | `scripts/training/run_bs5_fourarm400k.py`; `src/envs/bs5_fourarm400k.py`; `src/rl/bs5_fourarm400k_qmix.py` | `configs/bs5_fourarm400k.json`; seed 551/552/553; arm C-N/C-R/C-H/H-only | `python -m scripts.training.run_bs5_fourarm400k --arm C-H --training-seed 552 --output results/train_new/CH552` | Training/validation CSV, protocol/status, selected checkpoint, REVIEW/RESUME |
| Tổng hợp train | `scripts/analysis/compare_fourarm400k_multiseed.py` | 12 run trong `results/drl400k_training/multiseed/` | `python -m scripts.analysis.compare_fourarm400k_multiseed --input results/drl400k_training/multiseed --output results/final400k/training_rebuilt --freeze-models` | Đường học ba seed, bảng validation, manifest và ZIP 12 model |
| Benchmark | `scripts/evaluation/run_fourarm400k_returnfix.py` → `run_fourarm400k_drl_benchmark.py` / `run_fourarm400k_heuristics.py` | Frozen model; `configs/bs5_multiseed400k_plan.json`; `configs/bs5_fourarm400k_heuristic_benchmark.json` | Lệnh DRL/baseline bên dưới | Mỗi part 5 seed × 10.000 slot, raw trace, COMMIT, REVIEW/RESUME |
| Tổng hợp raw | `scripts/analysis/report_final400k.py` → `report_fourarm400k_multiseed.py`; `fourarm400k_return_summary.py` | REVIEW benchmark hoặc `results/final400k/raw/` | `python -m scripts.analysis.report_final400k --input results/final400k/raw --output results/final400k/raw_rebuilt` | Per-run/completeness; khi đủ 150 có aggregate, audit, bảng theo seed và temporal để vẽ |
| Vẽ hình | `scripts/plotting/plot_final400k.py` | CSV cuối ở `results/final400k/benchmark/`; learning CSV ở `results/final400k/training/` | `python -m scripts.plotting.plot_final400k` | `results/final400k/figures/{learning,benchmark,waterfall}.{pdf,png}` |
| Kiểm tra | `scripts/verification/check_final400k.py` | CSV/audit; tùy chọn model và raw | `python -m scripts.verification.check_final400k --models results/benchmark_assets/fourarm400k --raw results/final400k/raw` | `results/final400k/verification.json` |

### 1. Vẽ hình từ CSV — chạy được ngay

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUTF8 = "1"
python -m scripts.verification.check_final400k
python -m scripts.plotting.plot_final400k --input results/final400k/benchmark --training results/final400k/training --output results/final400k/figures
```

`SEVEN_CONFIG_400K_FINAL_BENCHMARK (1).zip` đã được nhập nguyên CSV và `audit.json` vào `results/final400k/benchmark/`: **150 lượt = 4 arm × 3 seed train × 10 seed kịch bản + 3 baseline × 10 seed kịch bản**, mỗi lượt 10.000 slot, tổng 1.500.000 slot. Baseline không có seed train và không được nhân bản thành ba thí nghiệm. `per_test_seed.csv` có 70 dòng, `per_training_seed.csv` có 12 dòng, `temporal_1000slot.csv` có 1.500 khối. Audit nhận từ gói giữ nguyên; báo cáo xác minh local ghi riêng.

Hình dùng bảng đã chốt, khoảng bootstrap ghép cặp và mẫu số J/Mbit gộp của gói. Ba seed train chỉ cung cấp kiểm tra độ nhạy ban đầu. Thời gian actor trên CPU khác nhau không phải đối chứng tốc độ trên cùng phần cứng. Script vẽ nhập từ `KLTN_400K_script_va_hinh.zip`, chỉ thêm tham số đường dẫn và PNG xem trước. Bảng learning gốc giữ tại `results/final400k/training/`.

### 2. Benchmark từ checkpoint — đủ model, cần runtime phù hợp

Trên Windows, để kiểm tra/nạp model và chạy test mà giữ nguyên cơ chế khóa của runner:

```powershell
$env:PYTHONPATH = "$PWD/scripts/compat/windows;$PWD"
python -m scripts.verification.check_final400k --models results/benchmark_assets/fourarm400k
```

Adapter `scripts/compat/windows/fcntl.py` dùng khóa byte Windows; Linux dùng `fcntl` gốc. Benchmark khoa học và resume lịch sử dùng Linux/Kaggle CPU cùng phiên bản NumPy/SciPy/PyTorch và phần cứng ghi trong protocol. Đủ file cho lượt benchmark mới; kiểm tra/nạp model không thực hiện rollout 10.000 slot. Chạy preflight trước lượt mới:

```bash
python -m scripts.verification.verify_fourarm400k_drl_benchmark --arm C-H --training-seed 552 --models results/benchmark_assets/fourarm400k --output results/preflight400k/CH552
python -m scripts.evaluation.run_fourarm400k_returnfix drl --arm C-H --training-seed 552 --models results/benchmark_assets/fourarm400k --part 1 --output results/benchmark_new/CH552_P1 --check-only
# Bỏ --check-only để rollout; dùng thư mục mới cho mỗi arm/seed/part.
python -m scripts.evaluation.run_fourarm400k_returnfix baseline --label RANDOM --part 1 --output results/benchmark_new/RANDOM_P1
```

Lặp DRL cho bốn arm, ba seed train và part 1/2; baseline cho RANDOM, DYNAMIC_AR, HEURISTIC_HO và part 1/2. Không ghi đè raw đã nhận. RETURNFIX sửa thống kê chuyến Returning thiếu `route_m`, giữ nguyên reward và mô phỏng. LPFIX và hai hash allocator nằm trong `configs/bs5_lp_roundoff_compat.json`; không hạ ngưỡng kiểm tra hay sửa protocol/COMMIT cũ.

**12 checkpoint:** bản lưu do chủ repo chọn là `archive/packages/incoming/final400k/rs (2).zip`, chứa `rs/{CN,CR,CH,HO}-{551,552,553}/best.pt`, protocol/status, training và validation CSV. Cả 12 trọng số trùng SHA-256 với frozen manifest. Không có file `.zip` lồng trong `rs (2).zip`; 12 `.pt` dùng định dạng ZIP nội bộ của PyTorch đã kiểm tra CRC và nạp model. Gói RETURNFIX chứa `01_KAGGLE_DATASET_UPLOAD/FOURARM400K_FROZEN_MODELS.zip`; bản tiện dùng ở `archive/packages/incoming/final400k/FOURARM400K_FROZEN_MODELS.zip`, giải nén thành `results/benchmark_assets/fourarm400k/` cho runner.

[Danh mục asset](configs/bs5_final400k_assets.json) ghi đường dẫn và SHA-256 mọi ZIP/CSV. [Manifest 12 model](configs/bs5_fourarm400k_frozen_models.json) ghi checkpoint, protocol, bước chọn và checksum từng file. **Chưa có URL tải từ xa**; đây là đường dẫn lưu local. Khi clone sang máy khác, lấy gói từ bản lưu của chủ repo, xác minh hash trong danh mục rồi giải nén về các đường dẫn trên. Máy hiện tại không cần tải thêm checkpoint.

| Arm / seed train | File trong `results/benchmark_assets/fourarm400k/` | SHA-256 checkpoint |
| --- | --- | --- |
| C-H / 551 | `models/C_H_S551/best.pt` | `8fb6a58a7e9c22327f8356c8b01eda3fdd5604255d81b4d25903ea36a4b4931f` |
| C-H / 552 | `models/C_H_S552/best.pt` | `c80f259044f887a337a25c819b5ace84d3ba259c502d94d433b6e6a3dfa9ddc9` |
| C-H / 553 | `models/C_H_S553/best.pt` | `192160ff2a6931272e1353f4e8b01889f989ca9f8d31c99a98de32afb4c15328` |
| C-N / 551 | `models/C_N_S551/best.pt` | `89da2133cb2b875df22fa79d82508d2021778d6b4a252aaf33e5a9d1ffd536ba` |
| C-N / 552 | `models/C_N_S552/best.pt` | `9c05cbd41de7d9a7f4a73a82502519308666e16dc7814be90c4502fb8c34f8a1` |
| C-N / 553 | `models/C_N_S553/best.pt` | `16f04df8b51cfec0c841bd145c13f04d961589d1cd4e69275bb8892d87fae448` |
| C-R / 551 | `models/C_R_S551/best.pt` | `67cf386b28ae0497eabd3594a13f6596fd1e73335af4012724d6f4e1c23e6f23` |
| C-R / 552 | `models/C_R_S552/best.pt` | `bad989ff1d3d0390ba5c27192c7d11c0aa3d0645a0bae68473e0c54e4d9ea45e` |
| C-R / 553 | `models/C_R_S553/best.pt` | `6f969b5a1841daa280ef37b7a669c25c01961fcdbe7c63a4eb855408a07e0e87` |
| H-only / 551 | `models/H_only_S551/best.pt` | `10bdb36f49aa8c19b6252ecee5f508fa034b6abb965f2dfa8afd655615793899` |
| H-only / 552 | `models/H_only_S552/best.pt` | `0129c419a14b315355ddd59443b3ceb404e6529073a4c55e00206d16b32fa35c` |
| H-only / 553 | `models/H_only_S553/best.pt` | `36d6764e71dab691caa174492fac81af825a3303d88f13fffbed3f372ada99c1` |

### 3. Tổng hợp lại từ REVIEW thô — đủ 150/150 lượt, chạy được ngay

```powershell
$env:PYTHONPATH = "$PWD/scripts/compat/windows;$PWD" # Chỉ trên Windows
python -m scripts.analysis.report_final400k --input results/final400k/raw --output results/final400k/raw_rebuilt
```

Đã kiểm kê status, protocol và `COMMIT.json`, thay vì suy số lượt từ tên ZIP:

| Gói gốc | Lượt hoàn tất dùng được |
| --- | ---: |
| `MULTI400K_CN_S551-552-553_P1-2_RESUME.zip` | 20: seed train 551/552, đủ part 1/2; snapshot train 553 chưa COMMIT không tính |
| `MULTI400K_CN_S553_P1-2_REVIEW.zip` | 10: seed train 553, đủ part 1/2 |
| `MULTI400K_CR_S551-552-553_P1-2_REVIEW.zip` | 30 |
| `MULTI400K_CH_S551-552-553_P1-2_REVIEW.zip` | 30 |
| `MULTI400K_H_ONLY_S551-552-553_P1-2_REVIEW.zip` | 30 |
| `MULTI400K_DYNAMIC_AR_PART{1,2}_REVIEW.zip` | 10 |
| `MULTI400K_HEURISTIC_HO_PART{1,2}_REVIEW.zip` | 10 |
| `MULTI400K_RANDOM_10SEEDS_REVIEW.zip` | 10: đủ seed 930201–930210, part 1/2 |

Các gói này không có manifest tổng riêng; bằng chứng toàn vẹn là hash ZIP, protocol, status và manifest từng lượt `COMMIT.json`. RESUME C-N đã đủ 20 lượt, không cần tải lại REVIEW riêng của seed 551/552. Các ZIP MULTI400K là bản lưu gốc; protocol bên trong xác nhận đây là log **benchmark test**, còn log **huấn luyện** rút gọn nằm trong `rs (2).zip`. Rs không chứa optimizer/replay/RNG để tiếp tục train chính xác.

Hash vỏ ZIP C-N seed 553 hiện có khác hash gói cùng tên được ghi trong audit lịch sử; bốn ZIP heuristic khớp chính xác. Đã kiểm tra nội dung 150 lượt theo COMMIT, trace và CSV cuối, giữ nguyên audit nhận ban đầu. Chi tiết nằm tại `archive/imports/final400k_2026-11-08/audit_archive_comparison.json`; không sửa audit để làm cho hash vỏ ZIP trùng nhau.

Gói RANDOM bổ sung đã hoàn tất 5/5 lượt ở mỗi part, đủ 10 COMMIT và không phải diagnostic. Đã chạy lại tổng hợp toàn bộ raw: `completeness.json` báo **complete, 150/150, missing=[]**. Các bảng dựng lại khớp bảng đã giao ở mọi cột số chung (rtol=1e-9, atol=1e-7): per-run 150 dòng, aggregate 7 dòng, per-test-seed 70 dòng, per-training-seed 12 dòng và temporal 1.500 dòng. Bao gồm cả khoảng tin cậy bootstrap trong aggregate.

Script tham chiếu tạo `scenario_summary.csv`/`training_seed_summary.csv`; adapter `report_final400k.py` dùng nguyên raw auditor và crossed bootstrap đó, bổ sung `per_test_seed.csv`, `per_training_seed.csv`, `temporal_1000slot.csv` và audit local để nối trực tiếp sang script vẽ. Audit local được lưu riêng với audit lịch sử nhiều giai đoạn. Vẽ từ dữ liệu vừa dựng lại:

```powershell
python -m scripts.plotting.plot_final400k --input results/final400k/raw_rebuilt --training results/final400k/training_rebuilt --output results/final400k/figures_rebuilt
```

### Kiểm tra local

Đã xác minh thiết kế ghép cặp 150 lượt và 1.500.000 slot; nạp đủ 12 actor và QMIX mixer; đối chiếu rs/frozen và tái tạo learning từ 12 log train. Raw auditor kiểm tra COMMIT, trace UE, reward, cân bằng năng lượng và Returning của đủ 150 lượt. Các bảng dựng lại khớp CSV cuối; ba hình PDF/PNG được vẽ lại từ bảng dựng lại. `--check-only` của runner DRL đạt; chưa chạy lại benchmark 150 lượt.

Sau khi chuyển thí nghiệm cũ, cả bốn protocol train vẫn nạp đủ 98 hash nguồn, 399.600 bước; 12 model vẫn nạp được và pytest thu thập đủ 71 test mà không lỗi import. **45 test chạy lại đạt**: 27 test CSV/LPFIX/RETURNFIX/catalog và 18 test notebook IO/protocol. Ruff cho mã tích hợp mới và `git diff --check` đạt. Cả ba PNG vẽ từ raw dựng lại trùng từng pixel với hình vẽ từ CSV đã giao. Kết quả chi tiết lưu trong `archive/imports/random400k_2026-11-08/`; các log kiểm tra trước đó ở `archive/imports/final400k_2026-11-08/`.

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "$PWD/scripts/compat/windows;$PWD" # Chỉ trên Windows
python -m pytest -q -p no:cacheprovider tests/test_bs5_lp_roundoff.py tests/test_bs5_lpfix_io.py tests/test_fourarm400k_return_summary.py tests/test_final400k_delivery.py tests/test_fourarm400k_multiseed.py
```

Ở các lượt kiểm tra trước, 56 test khác nhau đạt. Hai test xuất ZIP sau resume vẫn lỗi `PermissionError [WinError 5]` khi thay thế ZIP đã tồn tại trên Windows: `test_main_run_partial_export_resume_and_scanner` và `test_real_two_heuristics_and_drl_adapter_partial_resume_and_strict_merge`. Đã thử riêng và thử retry; giữ thao tác thay thế nguyên tử và giữ nguyên test để chạy lại trên Linux/runtime phù hợp. Ca `test_actual_evaluate_io_1200slots_each_arm` chưa hoàn tất trong lượt nhanh vì cần rollout dài. Runtime local là Python 3.13.5, NumPy 2.3.4, SciPy 1.15.3, PyTorch 2.12.1 CPU; runtime tái hiện lịch sử phải theo protocol được lưu.

### Dọn repo và bản lưu

Theo phạm vi ưu tiên bốn cấu hình 400K, **413 file mã/cấu hình/test/hướng dẫn cũ** đã chuyển khỏi cây hoạt động vào `archive/legacy_pre400k_2026-11-08/repo/`. Cùng checkpoint/kết quả cũ (`checkpoints/`, `results/test10_final/`, `results/benchmark_assets/fair300k/`), tổng cộng **493 file, 833.796.809 byte**, tất cả đã kiểm tra SHA-256 sau khi chuyển. Manifest từng file nằm tại `archive/imports/random400k_2026-11-08/legacy_manifest.json`. Git hiển thị xóa các file cũ đã theo dõi; bản lưu khoa học còn ở archive local.

**Giữ toàn bộ 93 file Python trong `src/`** vì checkpoint và protocol benchmark khóa hash cả cây nguồn, kể cả một số phiên bản cũ. Giữ `configs/bs5_fair300k.json`, `configs/bs5_service_v2_pilot.json`, helper train service-v2 và `fair300k_benchmark_metrics.py` vì pipeline Four-arm còn phụ thuộc chúng. Không xóa hay thay hash để bỏ qua khóa nguồn. Các file cũ trong `src/`, `docs/`, `reports/` và archive chưa rõ vai trò được giữ để xem trước khi xóa; danh sách lịch sử tại `archive/imports/final400k_2026-11-08/REVIEW_BEFORE_DELETE.md` cần đọc cùng manifest chuyển mới.

ZIP gốc nằm ở `archive/packages/incoming/final400k/`, ngoài phần Git theo dõi. Bốn ZIP heuristic được thêm lại ở gốc repo đã đối chiếu SHA-256 rồi chỉ bỏ bản trùng. RANDOM có SHA-256 `c2f61ecf3e478e921de7cd5bdd56d43d7a6c1670accc42fbaa891db1d0a4e9d6`. Không còn thiếu raw hay checkpoint cho pipeline cuối trên máy này. Để tiếp tục train chính xác từ lượt cũ vẫn cần RESUME train đầy đủ optimizer/replay/RNG; rs là bản model/log rút gọn.

`.gitattributes` giữ byte LF/CRLF của nguồn được khóa. Tài liệu, báo cáo, dữ liệu và archive local vẫn tuân theo `.gitignore`.
