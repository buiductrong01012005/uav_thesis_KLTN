# Chạy BS5 C-R Return-v3 trên Kaggle

Tạo Dataset mới từ `BS5_CR_H1200_RETURN_V3_DATASET.zip`, import notebook
`bs5_cr_h1200_return_v3_kaggle.ipynb`, Add Input, chọn CPU và Run All.
Notebook nhận ZIP hoặc Dataset được giải nén, yêu cầu chọn rõ SOURCE_PATH nếu có nhiều bản.

## Giao thức

- Fresh C-R, seed551, tổng60k train; không import v2 checkpoint/replay/optimizer.
- S1 0–3k, S2 3–6k, S3 6–12k horizon150; S4 12–60k horizon1200.
- Validation tại0/12/24/36/48/60k,10 scenario UE mỗi mốc,1200 slot/game; tổng72k.
- Cùng map BS5 hỏng, launch B, ba trạm `(1,1)`, `(18,1)`, `(10,18)`.
- Trạm shared không giới hạn số UAV, mỗi UAV Charging nhận5pin/slot, ngưỡng rejoin80.
- Giữ reward, radio/router, actor/state dimensions, QMIX, hyperparameters và curriculum.
- S4 bootstrap qua time limit bằng final next state, clear replay khi vào S4 như v2.

## Thay đổi Return

BFS bốn hướng dựa trên vị trí đội UAV tại thời điểm hiện tại, chặn ô ngoài trạm có UAV khác.
Lập lại đường mỗi slot, chọn trạm có đường ngắn nhất, hòa chọn station ID rồi đường N/S/E/W.
Không quan sát tương lai/UE để chọn đường sạc, không sửa role Access/Relay của policy.
Ngưỡng Return dùng chiều dài đường đang khả thi ×0,2 +5 dự trữ như cũ.
Nếu hiện tại bị vây kín, dùng Manhattan lower bound và log no-route: không cam kết đủ pin để chờ vô hạn.
UAV Returning chỉ di chuyển nếu đủ chi phí nguyên slot0,2; chờ vẫn hao0,2 như cũ.
Thiếu năng lượng: đứng tại chỗ, không radio, giữ Returning; log stranded. Không teleport/cứu/reset UAV.
Nếu action Active không đủ pin trả chi phí thì chuyển sang Return trước movement/service; log override.
Game vẫn chạy tới horizon; stranded có thể tồn tại tới hết game và được ghi như thất bại năng lượng.
Không thêm status mới, reward mới hay queue tại trạm. Collision-v2 vẫn kiểm endpoint, cho phép swap và shared hub.

Đây là controller tự động có thông tin vị trí đội UAV như cơ chế Returning hiện tại.
Lợi ích tìm đường là thay đổi môi trường/controller, không phải novelty học được của QMIX.

## Raw và đối chiếu

Raw từ bước đầu gồm current/next inputs và masks, UE/rates/router, position/status/pin, RNG,
sample indices/hashes, return paths/target trước-sau, forced overrides, waits, no-route, stranded.
Analyzer stdlib độc lập dựng reverse distance fields để kiểm planner BFS, không nhập env, NumPy hoặc Torch.
Theo dõi utility, QoS, Relay rỗng theo UAV-slot, lưu lượng UE cần Relay, lifecycle censored và stranded riêng.
Throughput bị cap demand bằng360×utility, không phải hai bằng chứng độc lập.
Một seed và validation đã quan sát; không kết luận hội tụ, C-R>C-N hoặc curriculum>direct.

## Tiếp tục cùng run Return-v3

Notebook dừng an toàn ở game boundary nếu đạt MAX_SESSION_SECONDS. Khi output tồn tại,
Run All sẽ resume cùng source/runtime/device; cap không được reset. Sau kill giữa game không tự rollback.
Để đổi session: giữ toàn bộ thư mục export, mọi audit_light/repro_full companion và EXPORT_INDEX.json.
Nén cả thư mục export thành một ZIP ngoài để các ZIP companion không bị Dataset tự giải nén;
khai báo RESUME_INPUT_DIR rồi chạy cùng source và runtime. Không chỉ upload best.pt.

Preflight cap512 diagnostic-train +4096 control/eval, ghi ledger trước từng env step.
Main tự dừng60k; không tự chạy120k. CPU là mặc định đã kiểm; CUDA chỉ có compatibility smoke khi yêu cầu.

Sau run gửi `review_small.zip`, mọi `audit_light*.zip` và `EXPORT_INDEX.json` để audit.
Giữ mọi `repro_full*` để tái lập/resume hoặc frozen-policy evaluation sau này.
