# C-R 60k, S4 horizon 1.200 — Kaggle

## Chạy

1. Tạo Kaggle Dataset bằng ZIP source bàn giao; giữ nguyên cấu trúc ZIP/thư mục `uav_thesis`.
2. Import `bs5_cr_h1200_60k_kaggle.ipynb` làm notebook, Add Input Dataset đó.
3. Mặc định CPU, một seed551. Runtime phải có NumPy, Torch, pytest; notebook không tự đổi Torch/CUDA. Nếu thiếu pytest, cài trước rồi khởi động lại runtime.
4. Run All. Không cần dữ liệu/checkpoint lịch sử. Source được tìm tự động từ input ZIP hoặc thư mục đã giải nén; nhiều candidate thì phải chỉ rõ SOURCE_PATH.
5. Download các file trong `bs5_cr_h1200_seed551_export...` khi hoàn tất hoặc dừng tại game boundary.

Giới hạn notebook mặc định7 giờ **của lệnh run**, dừng ở boundary; có thể thay `MAX_SESSION_SECONDS` trước chạy. Đó không phải giới hạn được bảo đảm của Kaggle. Validation đang chạy sẽ hoàn tất trước pause. Không bấm Stop/kill giữa game nếu muốn exact boundary resume. Không giả định working tự tồn tại qua phiên mới.

## Phạm vi

- Chỉ C-R, training seed551, fresh60k.
- S1 0–3k, S2 3–6k, S3 6–12k: horizon150, conditions và schedule prefix cũ.
- S4 12–60k:48k bước =40 game1200, phủ một lần pool training120001–120040.
- Validation0/12/24/36/48/60k,10 seed130001–130010×1200:72k bước riêng.
- Preflight:8 tests,384 diagnostic-training steps;1.518 control/eval steps. Nếu dùng CUDA có thêm24 train+24 eval compatibility smoke. Main budget không bị trộn với preflight.
- Một map, BS5 hỏng, launch B; scenario seed không phải map khác. Không C-N/direct/seed552/120k/10k benchmark.
- Giữ network/reward/physics; đổi learning protocol ở S4 gồm horizon, time-limit bootstrap và clear replay một lần tại12k. Đây không phải ablation chỉ thay horizon.

## Resume

Full state gồm online/target networks, Adam, replay packed arrays/RNG/cursor, exploration/global RNG, counters và frozen schedule/protocol/source/runtime. Checkpoint commit sau mỗi game. Validation/weights không được thay learning state.

Để chuyển phiên: giữ toàn bộ `audit_light*`, `repro_full*`, `EXPORT_INDEX.json`; tạo Dataset từ export đó, Add Input, đặt RESUME_INPUT_DIR đúng thư mục, dùng cùng source/runtime/device. Notebook giải nén/ghép và kiểm hash. Chỉ giữ best weights không đủ resume.

Kill giữa game có thể mất in-flight state. Ledger tính cả bước đã thực hiện/reserved và runner chặn tự retrain phần mất vượt compute cap; báo partial thay vì giấu rollback. Với source/runtime khác không gọi đó là exact resume.

## Audit offline

Giải nén tất cả audit_light ZIP vào một thư mục. Không cần Torch, NumPy, checkpoint hoặc env:

```bash
python -B analyze_bs5_cr_h1200.py --root MERGED --output ANALYSIS --require-complete
```

Nếu kèm repro files thì kiểm thêm hash của chúng; thiếu weights không ngăn tính lại metric. Partial export không pass require-complete. Không sửa/xóa protocol, budget hoặc raw để vượt validation.

Metrics có utility, QoS/fairness, delivered relay-required traffic và Relay rỗng/UAV-slot, tám cửa sổ150slot, lifecycle nhiều chu kỳ, rejoin→service/censoring, đi vòng/blocked move, availability. Time-limit S4 bootstrap từ final next state của cùng game trước reset. Chunk150slot không chặn bootstrap.

Best chọn theo mean utility toàn1200 trên10 validation game, tie lấy mốc sớm; vẫn giữ last full state để nối tiếp. Một training seed chưa cho kết luận hội tụ, hiệu ứng nhân quả horizon hoặc ưu thế curriculum/direct/C-R/C-N. Không đếm throughput/utility là hai bằng chứng độc lập.

## CLI

```bash
python -B scripts/training/run_bs5_cr_h1200.py prepare --output OUTPUT --device cpu
python -B scripts/training/run_bs5_cr_h1200.py run --output OUTPUT --max-seconds 25200
python -B scripts/training/run_bs5_cr_h1200.py run --output OUTPUT --resume --max-seconds 25200
python -B scripts/training/run_bs5_cr_h1200.py export --output OUTPUT
```

Export cuối phiên, không rebuild ZIP/analyze tại mọi mốc. ZIP parts độc lập, không password/multipart nhị phân. Nếu resume.pt nén vẫn lớn hơn80MiB sẽ có file riêng và restore_to trong index. Giữ tất cả companion cho resume. Review_small + audit_light + EXPORT_INDEX đủ để gửi phân tích, không cần chạy lại policy.
