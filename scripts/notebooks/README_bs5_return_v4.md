# Return-v4: sửa vòng lặp và eval lại hai checkpoint đã khóa

Tạo Kaggle Dataset từ BS5_RETURN_V4_PAIR_DATASET.zip, import bs5_return_v4_pair_kaggle.ipynb, Add Input Dataset đó, chọn CPU và bật Internet nếu cần cài runtime rồi Run All. Không cần các ZIP audit cũ hoặc Dataset train. Dataset đã có đúng C-R60k/C-N36k và20 UE tape10k được dùng trong run v3 hoàn tất.

Chỉ đổi controller: giữ route suffix khi next hop còn trống, replan khi next hop bị chặn, loại ô vừa rời khỏi BFS nếu vẫn có route khác; cho phép quay lại nếu đó là đường thoát duy nhất. Memory chỉ thuộc controller Returning, không vào actor. Trigger vẫn dùng BFS v3 không memory. Giữ nguyên drains/margin/charge/hysteresis, reward, router, collision-v2,111 state/116 obs/10 actions. Các file khoa học v3 nguyên vẹn; lớp mới/version bs5_energy_return_v4_commit riêng biệt. Không bảo đảm tiến độ trước chướng ngại tùy ý.

Mặc định chạy đúng40 game =2 policy×20 scenario×10.000 slot=400.000 env steps; train=0. Checkpoint được chọn trước, không chọn lại sau sửa. Mỗi game đúng một reset; epsilon0; weights frozen. Hai arm dùng cùng exact tape/initial pin/assignment/trạm sạc từ run cũ, UAV/association/pin/charging tự diễn biến. UE tape chỉ được đưa vào quan sát từng slot hiện tại, controller không đọc tương lai. So v3/v4 cùng trace trong before_after.csv; tách seen và previously-heldout. Các trace đã được xem/chẩn đoán, nên đây là kiểm độ nhạy với controller mới, không phải một heldout test mới chưa xem. Một training seed551.

Preflight có ledger cap4096 control steps, train=0. Gồm hai ca blocker dao động, route-next-hop bị chặn, ngõ cụt cần backtrack, reset/exit-memory, thiếu pin, random controls, invariant collision/energy, hai prefix1.200 C-R130001/C-N190006 chạy bằng envv3 để xác nhận runtime/checkpoint tái lập raw cũ, tiny24slot và export/resume/analyzer không Torch. Không yêu cầu v4 khớp prefix v3 vì controller đã đổi.

Soft session limit7h là cấu hình của lệnh eval, không phải thông tin về giới hạn nền tảng Kaggle. Dừng ở game boundary; chừa thời gian export. Có thể hạ MAX_SESSION_SECONDS. Exact resume yêu cầu cùng source/assets/runtime Python/Torch2.11.0+cpu/NumPy2.1.3. Bộ chọn hành động luôn deterministicCPU,1 thread. Không có optimizer/replay hoặc train call.

Mỗi slot giữ float64 UE rates, exact float32 obs/state/battery, masks/actions/association/routes/phase geometry và toàn bộ telemetry Return/collision. Actor inputs lưu NPZ một lần (h+1 frame), UE coordinates có sẵn ở Dataset được tham chiếu bằng hash; tránh lặp hàng triệu số UE/obs trong JSON. Vì vậy export ít part hơn bộ audit trước. Mọi game có hashNPZ/chunks, checkpoint/weights hash trước–sau, input trace hash và config. Dữ liệu đủ kiểm độc lập physics/controller; không chỉ có các bảng tổng hợp.

Tải review_small.zip, EXPORT_INDEX.json và mọi evidence_part*.zip. Để tiếp tục ở phiên sau, giữ tất cả evidence parts cùng index trong một thư mục (có thể ZIP ngoài để upload Dataset), Add Input Dataset resume và source, điền RESUME_INPUT_DIR rồi Run All. Không tự chép đè OUTPUT_ROOT hiện có; cùng phiên thì RESUME_INPUT_DIR=None. Ledger chặn lặp game đang dở sau hard interruption/partial_error; giữ output để kiểm, không xóa/sửa counter. Không thay baseline hay lọc ca190006 khỏi bảng chính.

Analyzer offline từ thư mục MERGED sau khi giải nén mọi evidence part:

```bash
python -B MERGED/bs5_return_v4_results.py --root MERGED --output ANALYSIS --deep-planner --require-complete
```

Chỉ cần NumPy và stdlib; không Torch/env và không tạo thêm env.step. Bỏ --require-complete cho run pause. Mặc định đã kiểm hash, exact inputs/phase continuity, mask, rate bounds, energy và exact collision; --deep-planner dựng thêm reverse BFS/route memory/trigger/lifecycle cho từng snapshot. Raw UE tape nằm trong Dataset gốc; muốn kiểm service theo geometry/radio hoặc replay observation cần dùng đúng Dataset này và source snapshot, không tự tạo UE trace mới.

Báo cáo giữ QoS4Mbps, fairness tức thời/cả game, service-positive, outage dài nhất và cửa sổ1.000, Return stranded/wait/reversals, Relay-empty và mọi cycle với censoring. Utility và throughput là cùng bằng chứng đổi đơn vị. Hết stranded ở ca known-failure không đủ để khẳng định mọi trace đều an toàn.
