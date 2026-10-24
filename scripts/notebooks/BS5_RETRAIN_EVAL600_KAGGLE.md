# BS5: một Dataset, train mới rồi eval600

Upload **ZIP nguồn mới** thành **một Kaggle Dataset**, import notebook
`bs5_retrain_eval600_kaggle.ipynb`, Add Input Dataset đó, bật GPU và **Run All**.
Kaggle tự giải nén ZIP; không cần upload audit cũ, weights cũ hoặc cài lại Torch.

Bốn cell lần lượt tìm nguồn, chuẩn bị lượt chạy, chạy toàn bộ, và hiện link tải về.
Notebook dùng runtime Torch/NumPy hiện có và khóa runtime/source cho lượt chạy này.
Các thư viện khoa học cần thiết đã có trong Kaggle; không yêu cầu Internet.

## Công việc và giới hạn

- Train **mới** C-R/C-N × seed551/552, 60k/run, tổng **240k training steps** theo
  runner fresh60k hiện có: prefix24k → tail36k. Không nhập checkpoint/metrics cũ.
- Giữ nguyên env, reward, router, learner, seed và lịch curriculum. 62 file khoa
  học trong `src/` được kiểm hash trước khi chạy.
- Chọn best riêng từng run trên full-task validation150 tại 0,3k,…60k; hòa chính
  xác chọn mốc sớm nhất. Best có thể nằm trong prefix24k và vẫn được giữ đúng.
- Khóa weights đó rồi eval600 liên tục: 4 × 10 × 600 =24k main steps; controls
  1.200; tổng eval tối đa25.200. Không update optimizer trong eval, không tăng horizon.
- Đây là **thí nghiệm mới**, không tái tạo hoặc thay thế checkpoint đã mất của
  lượt cũ. Best/utility mới không bị buộc khớp các mốc57k/45k/60k/51k lịch sử.
- Dựng mọi cycle từ trạng thái từng slot; phục vụ trở lại chỉ tính sau slot
  rejoin; giữ censoring, starvation và mọi scenario thất bại.

## File tải về

Khi hoàn tất, cell4 hiện **`bs5_retrain_eval600_results.zip`**, chứa:

- Training `audit_light`, `repro_full` **có đủ bốn best weights**, `review_small`,
  export receipt, fresh experiment và budget ledger.
- Eval600 audit bundle, export receipt, checkpoint selection và hash manifest.

Giữ ZIP kết quả đầy đủ để phân tích hoặc chạy đánh giá lại; không chỉ tải audit.
Đây là ZIP bàn giao nhiều artifact; các ZIP bên trong giữ nguyên hash/provenance.

## Nếu Kaggle bị ngắt

Output nằm tại `/kaggle/working/bs5_retrain_eval600`. Giữ toàn bộ output khi cần
chuyển phiên; thư mục working không tự tồn tại ở phiên mới. Không prepare đè.
Chạy lại cell3 dùng trạng thái training đã commit, bỏ qua run đã hoàn tất; training
rollback nếu có được ghi riêng trong ledger, không giấu vào budget durable240k.
Nếu đổi source/runtime giữa lượt, runner dừng để bảo toàn provenance.

Eval không hỗ trợ resume giữa trajectory: nếu một game bị ngắt, giữ bằng chứng
INCOMPLETE; runner không tự reset game, không đặt lại ledger hoặc chạy thêm budget.
Không tự train thêm quá60k/run để tìm best tốt hơn. Kết quả eval150/600 cùng tập
validation đã xem, n=2 training seeds, không gọi unseen test hoặc hội tụ.

Notebook không chạy huấn luyện trên máy lúc đóng gói. Các kiểm tra bàn giao dùng
dữ liệu synthetic/offline; kết quả train/eval thật chỉ có sau khi bạn chạy Kaggle.
