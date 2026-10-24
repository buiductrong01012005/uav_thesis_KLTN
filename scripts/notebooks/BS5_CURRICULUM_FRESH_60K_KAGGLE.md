# Chạy mới BS5 curriculum0→60k trên Kaggle

Lỗi `parents: []` ở notebook continuation nghĩa là Dataset không có đủ bốn full `resume.pt`24k và source snapshot. Audit ZIP có logs nhưng không thay thế Adam/replay/RNG. Người dùng đã cho phép chạy từ đầu, nên dùng notebook **`bs5_curriculum_fresh_60k_kaggle.ipynb`**.

Chỉ cần upload **ZIP nguồn mới** làm Dataset. Kaggle tự giải nén; notebook tìm trực tiếp thư mục `uav_thesis/`. Không cần upload parent24k hay audit cũ. Bật GPU trong Notebook Settings. Chạy4 cells theo thứ tự; cell3 chạy train, cell4 xuất file tải về.

Đây là **thí nghiệm mới**, C-R/C-N × seeds551/552, tổng60k/run = **240k training steps mới**. Không nối curves/weights cũ; không dùng kết quả24k cũ làm prefix. Tổng compute nhiều hơn144k của phương án chỉ nối thêm36k/run. Không tự chạy main trên máy khi chuẩn bị bộ Kaggle.

Giữ env/reward/router/learner và lịch curriculum3k→3k→6k→48k. Source `src/` được kiểm bằng manifest khoa học đã audit. Không thêm seed/arm. C-N mask Relay cả act/bootstrap. Replay chỉ clear khi physical task thay đổi trước10.5k; không clear12k/24k, giữ epsilon global về.05 trong18k và optimizer/targets/RNG. Dừng tại60k; không tự kéo90k/100k.

Không bắt buộc Torch2.11/NumPy2.1.3 để khởi tạo **fresh**: dùng runtime hiện tại, ghi rõ phiên bản/GPU/flags và khóa runtime khi resume cùng lineage. Điều này không tạo equivalence bitwise với kết quả GPU lịch sử. Nếu đổi source/runtime giữa một lượt đang chạy, guard sẽ dừng; khôi phục source/runtime đã frozen để tiếp tục.

Runner chia nội bộ thành `prefix24k/` và `tail36k/`: prefix được train mới từ0 bằng đúng curriculum runner, sau đó tail restore full state vừa tạo. Prefix160 episodes dùng generator cũ với budget24k, suffix240 episodes dùng streamv1 độc lập như spec60k, nên không làm đổi fixture prefix khi tăng count. Việc chuyển state24k không phải nhập compute lịch sử; ledger ghi tổng fresh compute240k, imported lịch sử0. Có49.5k full-target exposure trong toàn lineage; S4 riêng48k.

Không full analyze/export trong từng episode/eval. Stage gates/retention chỉ ở phần curriculum trước/đến12k; không rerun S1 tại24k hay60k. Full-task eval chung0,3k,…60k (21 mốc), mốc4.5/7.5/10.5 chỉ chẩn đoán. Raw logs/shadows vẫn đầy đủ, oracle chỉ sampled1/76. Phân tích/export khi hoàn tất từng run60k, snapshot có receipt đóng timer sau export. Prefix dùng timer runner gốc và một outer phase gộp train/eval/persistence; tail dùng các phase tối ưu riêng.

## Lệnh tương đương

Từ thư mục repo:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B scripts/training/run_bs5_curriculum_fresh_60k.py prepare-fresh --output results/bs5_curriculum_fresh_60k --device cuda
python -B scripts/training/run_bs5_curriculum_fresh_60k.py run --output results/bs5_curriculum_fresh_60k --resume
python -B scripts/training/run_bs5_curriculum_fresh_60k.py export --output results/bs5_curriculum_fresh_60k
```

Chạy lại cell3/lệnh `run --resume` để tiếp tục output đang có. Đừng prepare lại vào output cũ. Mỗi eval là durable full-state checkpoint. Nếu bị ngắt giữa hai eval, rollback phần chưa commit và ledger ghi attempts; không tự export tích lũy khi interrupt. Giữ toàn bộ output directory nếu muốn resume ở một session Kaggle mới, copy nó vào `/kaggle/working/bs5_curriculum_fresh_60k` trước cell2 và giữ runtime/source đúng snapshot. Cell1 không tìm parent, nên `parents: []` không còn áp dụng cho chế độ này.

Trước khi đủ bốn fresh prefixes24k, chưa có tail bundle; mọi full state/log đã commit vẫn nằm trong `prefix24k/`. Sau khi có tail, cell4 xuất snapshot durable. Sau một interrupt, chạy cell3 để reconcile packet trước khi export. Nếu đã xong60k, gọi lại cell3/4 chỉ kiểm và dùng snapshot, không train vượt budget.

Offline analyzer không cần Torch/NumPy/env/checkpoints, dùng **audit_light** đã giải nén và output ở ngoài evidence:

```powershell
python -B scripts/analysis/analyze_bs5_curriculum_extend.py 'path/to/audit_extracted' --output reports/fresh60k_rebuilt --require-complete
```

Tải `review_small_*.zip`, `audit_light_*.zip`, `repro_full_*.zip`, `receipt_*.json` và `fresh_budget_ledger.json`. Review nhỏ để upload kiểm chứng; audit chứa raw evidence; repro chứa source/parent nội bộ/best/last/full resume và hash companion audit. Giữ cả audit + repro. Mean±1 sample SD là giữa2 training seeds, không phải95%CI; throughput360×utility không là bằng chứng độc lập.
