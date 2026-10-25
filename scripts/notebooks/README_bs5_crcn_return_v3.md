# Kaggle: đối chứng Relay/không Relay trên Return-v3

## Chạy

1. Tạo Kaggle Dataset mới từ `BS5_CRCN_RETURN_V3_DATASET.zip`. Có thể để Kaggle giải nén hoặc giữ ZIP nguyên; notebook hỗ trợ cả hai.
2. Import `bs5_crcn_return_v3_kaggle.ipynb`, Add Input Dataset đó, chọn CPU. Bật Internet nếu runtime cần cài thư viện.
3. Run All, giữ nguyên các tham số mặc định. Không cần Dataset/kết quả/checkpoint cũ.

Notebook dùng tiến trình Python mới và runtime riêng (Torch 2.11.0+cpu, NumPy 2.1.3, pytest 8.4.2), không thay thư viện đang được kernel import. Nếu có nhiều bản Dataset cùng tên marker, điền SOURCE_PATH rõ ràng. Dữ liệu source/assets được kiểm SHA256 trước khi dùng.

## Giao thức đã khóa

- C-R: checkpoint Return-v3 seed551 best/last60k có sẵn; không học thêm. Kèm metadata, source identity lịch sử và prefix validation đầy đủ.
- C-N: fresh seed551, khởi tạo đúng cùng weights gốc, cùng curriculum/hyperparameters/60k train/72k validation. Giữ head10 actions; mask Relay trong actor và target, router không dùng Relay. Không warm-start từ C-R60k.
- Toàn bộ120 UE tape của lịch train và10 tape validation1.200 của C-R được đính kèm để C-N gặp đúng dữ liệu cũ. Không đổi reward, trạm sạc, controller, replay/task reset hoặc time-limit semantics.
- C-N chọn best theo mean utility validation1.200 ở0/12/24/36/48/60k; hòa chọn checkpoint sớm hơn. C-R cố định60k. Không dùng kết quả10k để chọn checkpoint.
- Paired10k: seen130001–130010 và heldout190001–190010, hai policy, tổng40 game/400k env steps. Mỗi game reset đúng một lần, epsilon0, weights cố định. C-R chạy trước C-N cho mỗi scenario theo thứ tự đã khóa.
- UE tape10k tạo bởi generator gốc, không env.step/không nhìn policy; đầu1.200 của nhóm seen khớp tọa độ lịch sử. Cả hai arm replay cùng tape/initial assignment/pin/trạm sạc. Chỉ UE là ngoại sinh; chuyển động UAV/role/association/pin/Return tự diễn biến. Actor chỉ nhận quan sát hiện tại, không được đọc tape tương lai.
- Hai arm seen kiểm prefix1.200 với validation đã chọn trước khi đi tiếp tới10k. Action/observation/state/geometry/pin phải khớp; rate dùng atol1e-6, rtol1e-7.

Đây là đối chứng cùng recipe ở một training seed. Các fixtureS1/S2 gốc hướng tới phối hợp Relay; giữ nguyên để tái lập recipe và ghi thống kê coverage tĩnh trong baseline_note.json. Không diễn giải C-N là upper bound của mọi cách huấn luyện không Relay. Không tự train120k, thay curriculum hay benchmark các baseline cũ.

## Preflight và ngân sách

Trước main, notebook chạy các kiểm tra có giới hạn: tiny C-N192 bước liên tục so với96+resume96 (384 diagnostic-train tổng), hai prefix C-R1.200 ở130001/130009, trace chung với hành động khác nhau, kiểm physics/collision, masked targets và export/offline analyzer. Ledger giới hạn512 diagnostic-train +4096 control/eval; main-training=0. Các seed preflight tách khỏi heldout190001–190010. Receipt PASS được tái dùng khi source/assets/runtime không đổi.

Ngân sách main mới:60.000 C-N train +72.000 validation +400.000 paired eval =532.000 env steps; không tính preflight. Mỗi phase có ledger dự trữ trước env.step. Tạo UE tape chỉ tiến generator UE và được ghi riêng, không phải rollout policy. Thời gian còn phụ thuộc CPU/disk; không bảo đảm hoàn tất trong một phiên. Soft limit7 giờ chỉ áp dụng lệnh main, dừng ở game boundary (và sau validation mốc liên quan); cần thêm thời gian export. Có thể hạ MAX_SESSION_SECONDS trước lần chạy đầu. Không đổi thông số giao thức để chạy nhanh hơn.

## Tiếp tục

Khi `paused_at_game_boundary`, tải toàn bộ file trong thư mục export. Tạo Dataset resume bằng cách nén nguyên thư mục export vào một ZIP ngoài, giữ các companion ZIP bên trong. Phiên sau Add Input Dataset source và resume, đặt RESUME_INPUT_DIR đến thư mục chứa đúng một EXPORT_INDEX.json. Run All để kiểm và tiếp tục. Nếu OUTPUT_ROOT đang tồn tại trong cùng phiên, giữ RESUME_INPUT_DIR=None và Run All.

Resume đòi hỏi cùng source/assets và chính xác runtime Python/Torch/NumPy; lưu protocol/runtime cũ. Checkpoint resume giữ optimizer, replay, RNG/exploration, lineage, batch audit và commit. Eval tiếp tục ở game đã commit, không lặp game. Không tự resume in-flight game hoặc xóa ledger khi hard interruption/partial_error; bảo toàn output và kiểm tra compute chưa commit trước. Ledger cố ý chặn việc lặp âm thầm.

## Kết quả cần giữ/gửi lại

- `review_small.zip`: báo cáo nhanh, CSV/JSON tổng hợp, trạng thái và export index.
- Toàn bộ `audit_light_part*.zip` + `EXPORT_INDEX.json`: raw train/validation/eval, UE tapes, nguồn snapshot, protocol, ledgers, prefix checks, analyzer độc lập và hash.
- Toàn bộ `repro_full_part*.zip`: resume/checkpoints. Giữ cùng audit companions để tiếp tục hoặc tái lập policy.

Mỗi part là ZIP độc lập, không phải ZIP split-volume. Gộp nội dung mọi companion vào cùng thư mục, không chồng nhiều phiên export. Analyzer dùng stdlib:

```bash
python -B MERGED/analyze_bs5_crcn_return_v3.py --root MERGED --output ANALYSIS --require-complete
```

Bỏ `--require-complete` cho run tạm dừng; thêm `--deep-planner` để dựng BFS độc lập từng snapshot. Mặc định đã kiểm hash, continuity, UE digest, masks, rates, energy và resolver collision chính xác. Bộ tiny preflight có deep audit; default export không dựng lại mọi batch update/BFS toàn bộ main, nhưng giữ đủ raw để kiểm sâu sau.

Báo cáo utility/throughput/QoS/Jain từng slot và cả game, zero-throughput streak/window1.000, stranded/wait/reversal Return, pin khi đến trạm, các chu kỳ Return–charge–rejoin–useful với censoring. `aggregate.csv` tách seen/heldout, `paired.csv` so chênh lệch từng trace; C-N Relay-empty=N/A. Không gộp Jain cả game với Jain tức thời, không xem throughput và utility là hai bằng chứng độc lập, không coi đã useful một lần là phục vụ liên tục.
