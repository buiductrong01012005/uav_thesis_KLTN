# BS5 continuous evaluation600 trên Kaggle

Đây là **evaluation-only**, không phải train tiếp. Bốn policy best đã chọn từ validation150 của fresh0–60k được giữ cố định. Mỗi scenario chạy một trajectory600 slot liên tục, không reset tại150/300/450. Không tăng horizon hoặc budget để tìm kết quả tốt hơn.

Nếu thiếu weights và đã cho phép **train mới**, dùng [một Dataset + notebook train→eval600](BS5_RETRAIN_EVAL600_KAGGLE.md). Luồng đó chọn best của lượt mới và ghi provenance riêng, không thay checkpoint lịch sử trong giao thức dưới đây.

## Dataset cần có

1. ZIP nguồn mới có `scripts/evaluation/eval_bs5_continuous.py`.
2. `audit_light_6d851d3f2b66.zip` của fresh60k, hoặc hai phần seed đã ghép vào cùng root. Cần full raw audit, không phải review nhỏ.
3. **`repro_full_6d851d3f2b66.zip` của cùng lượt fresh60k**, có manifest và companion hash. Trong đó cần:

```text
runs/C-R_seed551/checkpoints/057000.pt
runs/C-R_seed552/checkpoints/045000.pt
runs/C-N_seed551/checkpoints/060000.pt
runs/C-N_seed552/checkpoints/051000.pt
```

Audit không có weights. Không dùng `snapshot_f4a1ca622efd_repro_full.zip`: đó là thí nghiệm khác. Không thay best bằng last/resume ở mốc khác và không retrain để tạo weights thay thế.

Kaggle tự giải nén Dataset. Dùng notebook **`bs5_continuous_eval600_kaggle.ipynb`**, bật GPU rồi chạy4 cell theo thứ tự. Cell2 audit dữ liệu150 và khóa checkpoint; thiếu weights thì ghi BLOCKED và dừng trước env stepping. Để bổ sung sau một preflight BLOCKED, giữ output cũ và dùng tên output mới; output BLOCKED tiêu0 steps. Với output đã chạy dở, không prepare lại hoặc đổi tên để đặt lại budget.

Runtime gốc: Torch2.11.0+cu128, NumPy2.1.3, TeslaT4, một Torch thread; các precision/determinism flags lấy từ protocol. Ưu tiên cùng runtime. Notebook không tự cài lại package lớn hoặc đổi flags. Control parity trong runtime hiện tại và tái tạo raw150 lịch sử được ghi riêng; lệch action hoặc vượt tolerance thì fail-closed.

## Budget và lỗi giữa trajectory

- Main:4 models ×10 scenarios130001..130010 ×600 =24.000 env steps.
- Control: hai capabilities × hai horizons ×150 =600; bốn policy150 =600. Tổng control1.200, tổng trần25.200; training/optimizer updates0.
- Các test offline/synthetic không gọi env.step thật. Prefix parity dùng physics và policy thật chỉ chạy sau khi đủ weights, trong1.200 control steps nói trên.
- Trước mỗi chunk10 bước, ledger reserve budget; raw chunk được flush cùng hash và receipt. Nếu crash, phần chưa xác nhận vẫn tính trong uncertainty upper bound. Không đặt lại ledger.
- Chạy lại cell3 chỉ tiếp tục khi các trajectory/control đã có receipt hoàn chỉnh. Nếu ngắt giữa game, runner báo INCOMPLETE và từ chối reset/chạy lại game đó. Không có exact in-flight state resume trong phiên bản này.
- Các cửa sổ150 chỉ dùng cho analysis. Rejoin xảy ra cuối slot; useful service chỉ tính từ slot tiếp theo. Mọi cycle dựng từ trạng thái từng slot, giữ cycle censored/incomplete và không lấy traffic của cycle sau để hoàn tất cycle trước.

## Lệnh CLI

Từ repo, với audit/repro đã giải nén:

```sh
python -B -m scripts.evaluation.eval_bs5_continuous prepare --result-root /path/audit --checkpoint-root /path/repro --output /kaggle/working/bs5_continuous600 --device cuda --horizon 600
python -B -m scripts.evaluation.eval_bs5_continuous run --output /kaggle/working/bs5_continuous600
```

Nếu checkpoint lưu dưới tên `best.pt`, có thể truyền `--checkpoint-map map.json` vào prepare. Map là `{run: relative_path}` bên trong repro đã kiểm manifest; tensor và step vẫn phải khớp checkpoint đã khóa. `run --checkpoint-root PATH` chỉ thay vị trí tìm lại cùng file hash, không thay weights/protocol.

Khi xong, runner analyze và build final bundle một lần, tự mở bundle để kiểm CRC rồi chạy standalone analyzer từ clean extraction. Cell4 dùng bundle có sẵn để tải về. File export receipt ghi hash và timer đóng sau export. Giữ cả output directory nếu cần kiểm chứng một lần chạy bị gián đoạn.

```sh
python -B analyze_bs5_continuous.py extracted_audit --output rebuilt_analysis --require-complete
```

Analyzer dùng stdlib, không import Torch/NumPy/env, không load weights. `--require-complete` đòi đủ40 trajectories,24.000 main+1.200 control, raw/hash/continuity/prefix/paired exogenous/masks/capacity và no-update receipts. Thiếu hoặc hỏng dữ liệu không accepted. Báo cáo600 tách riêng benchmark150; n=2 seeds, validation đã xem, không gọi unseen test, steady-state hay hội tụ.
