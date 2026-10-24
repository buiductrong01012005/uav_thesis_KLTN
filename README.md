# UAV Thesis

D-R-reset (hai seed 551/552, full task + sáu replay resets):
[spec/config](configs/README_bs5_direct_replay_reset.md),
[notebook Kaggle](scripts/notebooks/bs5_direct_replay_reset_kaggle.ipynb),
[bốn cell để copy và hướng dẫn resume](scripts/notebooks/BS5_DR_RESET_KAGGLE.md).

Pilot curriculum BS5, lịch cố định 3k/3k/6k/12k: xem
[spec và lệnh chạy](scripts/training/README_bs5_curriculum.md),
[notebook Kaggle](scripts/notebooks/bs5_curriculum_kaggle.ipynb).

Pilot A/B launch, BS5 cố định: xem [hướng dẫn chạy](scripts/training/README_bs5_launch_ab.md).
Chạy trên Kaggle: dùng [notebook 4 cell](scripts/notebooks/bs5_launch_ab_kaggle.ipynb)
hoặc [sao chép các cell](scripts/notebooks/README.md). Dataset tự giải nén ZIP;
kết quả được ghi vào `/kaggle/working`.

Mã nguồn mô phỏng, huấn luyện và đánh giá cho luận văn UAV. Các phiên bản môi trường, thuật toán, cấu hình và giao thức thí nghiệm được giữ nguyên khi sắp xếp repo.

## Cấu trúc

```text
src/
  envs/             Môi trường mô phỏng và các phiên bản nghiên cứu
  rl/               Các thuật toán học tăng cường
  baselines/        Các chính sách baseline
scripts/
  training/         Huấn luyện từng mô hình hoặc giai đoạn
  evaluation/       Đánh giá checkpoint và chạy baseline
  launchers/        Chạy nhiều seed, chuỗi giai đoạn và pilot launcher
  analysis/         Audit, chẩn đoán và tổng hợp kết quả
  plotting/         Vẽ biểu đồ và bản đồ
  validation/       Kiểm tra thiết lập thí nghiệm
configs/            Cấu hình thí nghiệm
tests/              Bộ kiểm thử
docs/
  audits/           Hướng dẫn audit
  notes/            Ghi chú nghiên cứu
  archive/          Hướng dẫn và manifest của gói nguồn cũ
  thesis/           Tài liệu luận văn cục bộ
reports/
  tables/           Bảng tổng hợp đã xuất
  figures/          Các bộ hình đã xuất
archive/
  source/           Bản sao mã nguồn cũ (.bak)
  packages/         Các gói ZIP cục bộ
results/            Kết quả do các runner tạo
checkpoints/        Checkpoint mô hình
```

Một số thư mục chứa dữ liệu cục bộ được Git bỏ qua và chỉ có trên máy đã chạy thí nghiệm.

## Cài đặt và kiểm tra

Dùng Python 3.11 hoặc 3.12. Chạy từ thư mục gốc của repo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Trên Linux/macOS, kích hoạt môi trường bằng `source .venv/bin/activate`. Nếu chỉ chạy thí nghiệm, cài `requirements.txt`.

## Chạy script

Các entry point được chia theo chức năng; tên file, tham số CLI và mặc định thí nghiệm được giữ nguyên. Ví dụ xem trợ giúp và kiểm tra kế hoạch chạy:

```powershell
python -m scripts.training.run_relay_stress_r4_qmix --help
python -m scripts.evaluation.eval_relay_stress_r4_qmix --help
python -m scripts.launchers.run_relay_stress_r4_multiseed --dry-run
python -m scripts.analysis.audit_r4_pareto_roles --dry-run
python -m scripts.plotting.render_uav_relay_map --help
```

Cũng có thể chạy trực tiếp: `python scripts/training/run_relay_stress_r4_qmix.py --help`. Các đường dẫn tương đối nên được truyền từ thư mục gốc repo như trước.

Xem [danh mục script](scripts/README.md). Các lệnh cũ dạng `scripts/<tên>.py` cần thêm tên nhóm tương ứng trong danh mục này.

Tài liệu nghiên cứu nằm trong `docs/`; bảng và hình đã xuất nằm trong `reports/`. Các file mới hoặc vừa di chuyển vào hai thư mục này được giữ cục bộ theo `.gitignore`, nên không có trong bản clone mới. Các tài liệu đã được Git theo dõi trước đó vẫn được giữ nguyên.

Hướng dẫn `README_RUN.md` cũ và manifest đóng gói được lưu cục bộ trong `docs/archive/`; một số entry point trong đó không có trong repo hiện tại. Chọn giao thức thí nghiệm theo tài liệu nghiên cứu tương ứng.

Hướng dẫn nối BS5 curriculum từ full state24k tới tổng60k (chỉ C-R/C-N, seeds551/552):
[Windows/Kaggle](scripts/notebooks/BS5_CURRICULUM_EXTEND_60K_KAGGLE.md) và
[notebook Kaggle](scripts/notebooks/bs5_curriculum_extend_60k_kaggle.ipynb).
Runner kiểm đủ bốn full resume states trước rollout; audit_light hoặc weights-only không đủ để continuation.

Khi không có full states24k, dùng [BS5 fresh0→60k](scripts/notebooks/BS5_CURRICULUM_FRESH_60K_KAGGLE.md): train mới4 runs, không yêu cầu parent và không ghép kết quả lịch sử.

## BS5 continuous evaluation600

Frozen-policy diagnostic, not training. See [Kaggle guide](scripts/notebooks/BS5_CONTINUOUS_EVAL600_KAGGLE.md). Requires the four exact best fresh60k checkpoints and full audit/repro provenance. Main24000 + controls1200; no budget increase.

Khi thiếu weights và đã cho phép train mới, dùng [một Dataset + notebook train→eval600](scripts/notebooks/BS5_RETRAIN_EVAL600_KAGGLE.md): chỉ upload ZIP nguồn, chạy mới bốn runs60k rồi tự khóa best và eval600. ZIP kết quả gồm cả audit và repro có weights; đây là lượt chạy mới, không thay checkpoint lịch sử.
