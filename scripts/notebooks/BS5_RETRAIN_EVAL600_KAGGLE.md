# BS5: một Dataset, train mới rồi eval600

## Chạy lại từ đầu sau lỗi export

Dùng **`bs5_retrain_eval600_fresh_kaggle.ipynb`** với ZIP source mới bàn giao cùng
notebook. Upload ZIP thành Dataset, chỉ attach Dataset source mới, bật GPU và
Run All. Không cần kết quả/checkpoint cũ. Notebook kiểm hash hai bộ đọc đã sửa
lỗi `failed_bs_gateway_excluded` trước khi train và dùng output riêng
`/kaggle/working/bs5_retrain_eval600_fresh`, không ghi đè lượt cũ.

Budget và giao thức giữ nguyên: C-R/C-N × seed551/552 ×60k =240k train, rồi
eval600 tối đa25.200 bước. Máy cục bộ chỉ kiểm thử và đóng gói; train thật chạy
trên Kaggle khi người dùng Run All.

Cell cuối kiểm hash ZIP đầy đủ, hiện link tải và tạo bản cùng byte `.zip.bin`
kèm `_transport.json`. **Tải ZIP về ngay khi hoàn tất**. Khi lưu output thành
Dataset, giữ thêm `.zip.bin` và JSON để có bản container gốc nếu Kaggle giải nén
ZIP. Muốn phục hồi, copy `.zip.bin` thành tên `.zip` ghi trong JSON rồi đối chiếu
SHA256; không giải nén/đóng gói lại để tạo bản gốc. Bản `.bin` dùng hardlink nếu
hỗ trợ, nếu không sẽ copy và tốn thêm dung lượng bằng ZIP.

## Notebook thông thường

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

## Khôi phục lỗi export `failed_bs_gateway_excluded`

Nếu đủ 40 trajectory `complete` nhưng export báo KeyError trên, giữ phiên Kaggle
và thư mục working. Mở [BS5_RECOVER_EVAL600_CELL.py](BS5_RECOVER_EVAL600_CELL.py),
copy toàn bộ vào **một cell mới** và chạy riêng cell đó. Không chạy lại cell3
hoặc Run All. Không cần sửa/upload Dataset, cài thư viện hay bật Internet.

Cell lấy các bộ đọc từ source snapshot đã lưu, xác minh hash và vá bản sao trong
working. Nó kiểm tra receipts/budget, phân tích raw slots, kiểm tra lại ZIP rồi
xuất `bs5_retrain_eval600_recovered_results.zip` cùng `recovery_receipt.json`
trong `/kaggle/working/bs5_retrain_eval600_recovery` (có thể có hậu tố số).
ZIP giữ training artifacts và đúng bốn best weights đã chọn; raw, checkpoint,
source snapshot và budget gốc không đổi. Không train/eval thêm bước nào.
Registry lỗi gốc được giữ trong `recovery_original`; chỉ bản dẫn xuất được xác
nhận hoàn tất sau khi đủ receipts và audit offline chấp nhận. Thiếu/hỏng dữ liệu
sẽ dừng; đây không phải cách tiếp tục một trajectory bị ngắt.

Nếu đã lưu kết quả thành Dataset và bắt đầu phiên mới, dùng notebook riêng
[bs5_recover_dataset_kaggle.ipynb](bs5_recover_dataset_kaggle.ipynb): attach Dataset
kết quả rồi Run All. Hướng dẫn: [BS5_RECOVER_DATASET_KAGGLE.md](BS5_RECOVER_DATASET_KAGGLE.md).
