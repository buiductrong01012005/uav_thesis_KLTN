# Một Dataset, 2 hoặc 4 notebook train độc lập

Upload `BS5_RETURN_V4_PARALLEL_DATASET.zip` thành một Kaggle Dataset duy nhất.
Import các notebook trong `BS5_RETURN_V4_PARALLEL_NOTEBOOKS.zip`, rồi Add Input
CÙNG Dataset này vào mỗi notebook. Đây là phiên bản source mới cho cách chia job;
không trộn với checkpoint/resume từ Dataset cũ.

## Chọn một cách chia

| Cách | Notebook cần chạy | Compute mỗi notebook |
|---|---|---|
| 4 notebook | `CR_551`, `CR_552`, `CN_551`, `CN_552` | 60k train +66k validation |
| 2 notebook | `CR_551_552`, `CN_551_552` | 120k train +132k validation |

Tên đầy đủ: `bs5_return_v4_train_<JOB>_kaggle.ipynb`.
Chạy song song khi tài khoản được cấp đủ phiên GPU. Nếu chỉ có2 phiên GPU,
có thể dùng hai notebook theo nhánh, hoặc chạy4 notebook đơn thành2 đợt.
Không chạy cả cách2 và cách4 vì sẽ trùng arm/seed.

Mỗi notebook mặc định yêu cầu CUDA, sử dụng một GPU. Nếu hiện `+cpu` hoặc
`cuda:false`, nó báo lỗi ngay trước preflight/main. Bật Accelerator GPU và
dùng môi trường Kaggle có PyTorch CUDA. Không tự cài đè Torch trong một run
đang resume. Muốn CPU có chủ đích: DEVICE='cpu', REQUIRE_GPU=False trước prepare.

Chạy song song giảm thời gian chờ nếu có đủ tài nguyên độc lập; tổng compute và
GPU-hour không tự giảm. Không suy ra số phiên GPU được phép từ số notebook có
thể mở. Các giới hạn/quota thực tế do tài khoản Kaggle quyết định.

## Những gì được giữ nguyên

Return-v4, router/collision, reward, pin/trạm sạc, mạng QMIX và phép cập nhật đều
giữ nguyên. Mỗi run60k, curriculum S1–S3 game150 và S4 game1200. Validation
0/6k/.../60k trên5 trace cố định ×1200; chọn best bằng utility, ties earliest.

Lịch exogenous phụ thuộc training seed, không phụ thuộc việc job chạy trước/sau
hay cùng máy. C-R/C-N cùng seed có cùng initial weights, schedules và UE tapes.
Tapes được tạo deterministic từ source và được kiểm lại HASH THỰC TẾ khi ghép.
Model chỉ nhận observation hiện tại; không nhận trace tương lai.

Mọi output tách theo JOB_LABEL. Không có output thư mục dùng chung để hai
notebook cùng ghi. Shared Dataset ở `/kaggle/input` chỉ đọc; state/replay/raw
ở `/kaggle/working` của từng notebook.

## Lưu và resume

`review_small.zip` có report/learning curve/bảng. Giữ output notebook và:

- `training_resume__<JOB>.zip`: đủ model/target/Adam/replay/RNG và raw evidence.
- `training_for_benchmark__<JOB>.zip`: gói nhỏ checkpoint best/last, lựa chọn
  validation, CSV/PNG/PDF curve, protocol và UE tape receipts; chỉ xuất khi job
  được gán hoàn tất. Không dùng gói này để resume training.

Phiên sau Add Input source Dataset + đúng resume của notebook đó. Khi chỉ có
một candidate, notebook tự nhận; nhiều candidate thì điền RESUME_INPUT_DIR.
Giữ đúng runtime/device/loại GPU trong cùng job khi resume. Các job độc lập
có thể dùng GPU model khác nhau nhưng stack Python/Torch/NumPy/device phải
khớp để ghép; mọi runtime ban đầu được ghi đầy đủ.

Dừng mềm trước hạn, không reset giữa game. Ngân sách tổng10,5h và dự trữ30 phút
cho export. Không cam kết mỗi job hoàn tất trong12h. Nếu bị kill đột ngột giữa
game, giữ output/ledger; không sửa counter để ép lặp compute.

Preflight train mới chỉ kiểm forward/backward + exact resume bằng seed955xxx,
shared tapes và no-Relay masks. Không chạy benchmark5 heuristic trong notebook
train. CUDA smoke thực sự dùng GPU đang được chọn; chỉ kiểm tra nhỏ, không
train seed551/552 trước main.

## Ghép đủ job và benchmark

Import `bs5_return_v4_parallel_benchmark_kaggle.ipynb`, dùng CPU, Add Input:

1. Cùng source Dataset mới.
2. Bốn ZIP handoff đơn hoặc hai ZIP handoff theo nhánh, bao phủ đủ C-R/C-N×551/552.

Notebook tự tìm handoff, hoặc điền TRAINING_INPUTS khi có nhiều phiên bản.
Nó từ chối thiếu/trùng job, recipe/source/runtime stack khác nhau, initial weights
khác nhau, actual UE hash khác nhau hoặc checkpoint SHA/selection không khớp.
Không âm thầm lấy checkpoint mới nhất hay lấy mô hình tốt nhất trên test.

Merge chỉ tạo bundle inference: không train thêm, không nối replay, không trộn
weights. Giữ protocol/hash/runtime riêng từng job trong TRAINING_PROVENANCE.json.
Dựng learning curve chung từ CSV thực đo trước khi benchmark.

Benchmark có CPU controls/profile56 slot bằng scenario284901 và actor random
initialization để đo chi phí, KHÔNG cập nhật mạng và KHÔNG chạy final test trong
profile. Sau đó chạy cùng10 test seeds260001–260010 ×10k, một reset/game:
4 learned configurations +5 nonlearning =90 game/900k slot. Random chỉ một cấu
hình có Relay. Các baseline còn lại:5A/4A1R/3A2R/Dynamic +Recovery.

Checkpoint được khóa theo validation. Greedy privileged exact UE/network không
phải đối chứng cùng thông tin/chi phí với QMIX. Số preview và latency được báo
riêng. Main summary chỉ dùng scenario đã đủ mọi configuration; giữ toàn bộ raw
per-seed/per-game và cycle censoring. Có benchmark_resume.zip qua nhiều phiên.

## Kiểm chứng và giới hạn

Test source kiểm kết quả 4 shard so với chạy tuần tự (weights, optimizer, replay,
RNG), pause/resume, ghép4 và ghép2, chặn thiếu/trùng/tape mismatch, thông tin
provenance và compatibility benchmark. Không suy ra thời gian trainGPU từ testCPU.

CLI:

```
python -B scripts/training/run_bs5_v4_fresh_train.py prepare --output CR551 --device cuda --arms C-R --seeds 551
python -B scripts/training/run_bs5_v4_fresh_train.py run --output CR551
python -B scripts/training/run_bs5_v4_fresh_train.py export --output CR551
python -B scripts/experiments/merge_bs5_v4_training.py --output MERGED --inputs CR551.zip CR552.zip CN551.zip CN552.zip
```
