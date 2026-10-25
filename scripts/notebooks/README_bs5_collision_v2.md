# Kaggle — C-R60k, collision-v2, horizon1.200

## Chạy fresh

1. Upload ZIP source collision-v2 thành một Kaggle Dataset mới.
2. Import `bs5_cr_h1200_collision_v2_kaggle.ipynb`, Add Input Dataset mới, giữ `DEVICE="cpu"`, Run All.
3. Không cần input checkpoint hoặc kết quả train cũ. Main là một run fresh C-R/seed551/60k. Chưa chạy seed552, C-N,120k hoặc benchmark10k.

Mặc định: S1 0–3k, S2 3–6k, S3 6–12k với horizon150; S4 12–60k gồm40 game×1200. Giữ điều kiện/substage, reward, radio/routing, battery parameters, observation, QMIX, gamma và epsilon từ pilot trước. Validation0/12/24/36/48/60k,10 game1200 mỗi mốc:72k validation steps riêng.

Preflight chạy collision invariant suite (không env stepping) +10 integration tests, gồm384 diagnostic train steps và1.558 control/eval steps. CUDA tùy chọn thêm24 train+24 eval cho compatibility; CUDA exact-resume chưa được xác nhận trên máy bàn giao. Notebook dùng runtime có sẵn, không tự nâng Torch/CUDA. Thiếu pytest thì cài trước rồi khởi động lại.

## Quy tắc collision-v2

- Mọi UAV tham gia phải có endpoint khác nhau ngoài charging stations, kể cả Returning; status3 curriculum-inactive không chiếm ô.
- Trạm sạc là hub chung cho Returning/Charging/Active, kể cả slot rejoin và chờ rời trạm. Không ép UAV teleport khỏi trạm.
- Mỗi UAV được giữ requested endpoint hoặc đứng ở ô cũ, không tự thêm hướng né hay đổi role.
- Stationary occupancy được bảo toàn. Blocking dependency được xét đồng thời, tránh lỗi “trả về ô cũ đã bị chiếm”.
- Cho phép swap/cycle endpoint đồng thời. Đây là grid endpoint model, chưa kiểm khoảng cách an toàn liên tục hoặc crossing edges trong một slot.
- Returning ưu tiên trước Active, rồi ID tăng dần, trong các phương án endpoint khả thi. Nếu một UAV đứng yên chặn đường thì Returning có thể phải chờ. Không cam kết tiến lên vô điều kiện bằng cách cho UAV chồng ô.
- Giữ phí pin cho attempted Active movement như core cũ, kể cả khi move bị chặn; không đổi thêm reward/energy model ở lượt này.

Raw từng slot có requested/resolved endpoints, priority, blocked IDs/by, Returning waits. Live env và offline analyzer đều kiểm không trùng ô. Analyzer kiểm thêm assignment đúng priority, lựa chọn chỉ requested/old, current/next continuity và version. Metrics có số slot Returning chờ và Returning pin0; cần xem các chỉ số này sau run để biết giới hạn vận hành khi đường về bị chặn.

Core/runner cũ không bị sửa. Class mới `BS5CollisionSafeCurriculumEnv` và runner/analyzer/config mới có version `bs5_endpoint_collision_v2`. Những số trước sửa vẫn là pilot trên env cũ; không gọi chênh lệch lần chạy mới là chỉ do horizon.

## Output và resume

Output mới: `/kaggle/working/bs5_cr_h1200_collision_v2_seed551`. Notebook dừng tại game boundary sau tối đa7 giờ của lệnh run (có thể chỉnh `MAX_SESSION_SECONDS` trước chạy); không giả định đây là giới hạn Kaggle. Validation đang chạy hoàn tất trước pause.

- `review_small.zip`: báo cáo, curve, protocol và metrics.
- Toàn bộ `audit_light*.zip` + `EXPORT_INDEX.json`: raw đầy đủ để kiểm offline.
- `repro_full*` và audit companions: full training state để resume, gồm replay/RNG/optimizer/targets; không chỉ best weights.

Để đổi session, nén toàn bộ thư mục export vào một ZIP ngoài rồi tạo Dataset từ ZIP ngoài đó, giữ nguyên ZIP companion bên trong. Đặt `RESUME_INPUT_DIR` tới thư mục export. Chỉ resume cùng source/runtime/device và collision-v2. Không resume checkpoint/env-v1 bằng runner-v2. Kill giữa game có thể mất in-flight state; ledger giữ compute và chặn tự retrain vượt cap. Ưu tiên pause tại boundary, download output trước khi kết thúc phiên.

Không tự nối120k. Best chọn max mean utility toàn1200 trên10 validation game, tie lấy mốc sớm; last full state phục vụ resume. Ghi raw từ đầu, chỉ full analyze/export cuối phiên để giảm overhead.

## CLI

```bash
python -B scripts/training/run_bs5_cr_h1200_collision.py prepare --output OUTPUT --device cpu
python -B scripts/training/run_bs5_cr_h1200_collision.py run --output OUTPUT --max-seconds 25200
python -B scripts/training/run_bs5_cr_h1200_collision.py run --output OUTPUT --resume --max-seconds 25200
python -B scripts/training/run_bs5_cr_h1200_collision.py export --output OUTPUT
```

Giải nén audit_light cùng root rồi chạy, không cần Torch/NumPy/checkpoint/env:

```bash
python -B analyze_bs5_cr_h1200_collision.py --root MERGED --output ANALYSIS --require-complete
```

Sửa collision không bảo đảm utility tăng hoặc tự giải quyết việc đứt cầu Relay. Cần phân tích raw sau run; một training seed chưa chứng minh robustness/hội tụ/C-R hơn C-N.
