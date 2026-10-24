# Nối C-R/C-N từ 24k tới tổng 60k

**Nếu gặp `parents: []` và muốn chạy lại từ đầu:** dùng [notebook fresh0→60k](bs5_curriculum_fresh_60k_kaggle.ipynb) và [hướng dẫn fresh](BS5_CURRICULUM_FRESH_60K_KAGGLE.md). Chế độ fresh đã được người dùng cho phép; không yêu cầu checkpoint24k và không ghép kết quả cũ.

Dùng `bs5_curriculum_extend_60k_kaggle.ipynb` cùng thư mục. Kaggle Dataset tự giải nén ZIP: notebook tìm trực tiếp thư mục nguồn và thư mục parent, không yêu cầu ZIP còn nguyên.

## Inputs bắt buộc

- Source ZIP mới (thư mục `uav_thesis/` sau khi Kaggle giải nén).
- **Full parent 24k**: `protocol.json`, `source_identity.json`, `source_snapshot.zip` và `runs/{C-R_seed551,C-R_seed552,C-N_seed551,C-N_seed552}/{meta.json,commit.json,resume.pt}`. Nên kèm `checkpoints/<best_step>.pt` của từng run để giữ best lịch sử.
- Audit_light curriculum 24k đã có, để nối raw common eval 0–24k. Có thể nằm ở Dataset khác; notebook tìm theo protocol. Nếu chỉ có cached tables, báo rõ coverage, không nhận là tái audit raw đầy đủ.

**Audit ZIP không thay thế full resume.** Không dùng prefix CPU seed551, D-R-reset, best12k hay weights-only làm parent. Thiếu một trong bốn state sẽ dừng trước rollout.

Main ưu tiên runtime của parent: **PyTorch 2.11 CUDA, NumPy 2.1.3**. Notebook kiểm và dừng khi khác; không tự thay flags hay chuyển sang CPU. Nếu runtime Kaggle khác, cài đúng bộ đã dùng ở parent trong cell riêng trước các cell chạy, rồi restart kernel. Ví dụ ` %pip install torch==2.11.0 numpy==2.1.3 `; chọn đúng CUDA wheel/build của parent và kiểm lại GPU/driver. Không cam kết bitwise giữa runtime/GPU khác. Nếu parent không ghi numerical flags, preflight ghi rõ thiếu thông tin đó.

## Windows / CLI

Từ thư mục repo, PowerShell (đổi đường dẫn parent/evidence sang thư mục đã giải nén thật):

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$parent='F:\path\curriculum24k_full'
$evidence='F:\path\curriculum24k_audit'
$out='results/bs5_curriculum_extend_60k'
python -B scripts/training/run_bs5_curriculum_extend.py preflight --parent "$parent" --output reports/bs5_extend_preflight
python -B scripts/training/run_bs5_curriculum_extend.py prepare-extension --parent "$parent" --evidence "$evidence" --output "$out"
python -B scripts/training/run_bs5_curriculum_extend.py run --output "$out" --resume
```

Chạy lại lệnh `run ... --resume` nếu bị ngắt. Không chạy lại `prepare-extension` vào output cũ. Nếu prepare bị ngắt, dùng output mới và giữ nguyên output dở để kiểm tra. Lệnh run kiểm source hiện tại đúng snapshot extension; đổi code thì cần một authorization/output mới, không bypass guard.

Phân tích offline sau khi giải nén audit_light (Python stdlib, không cần env/Torch/NumPy/checkpoint):

```powershell
python -B scripts/analysis/analyze_bs5_curriculum_extend.py 'F:\path\audit_extracted' --output 'reports/bs5_extend_rebuilt' --require-complete
```

Export thủ công từ durable commit:

```powershell
python -B scripts/training/run_bs5_curriculum_extend.py export --output "$out"
```

Sau interrupt, chạy `run --resume` để archive chunk chưa commit trước khi analyze/export. Run không tự export khi bị ngắt. `--resume` tiếp tục từ eval commit gần nhất; ledger tách steps thử rồi rollback và steps đã durable. Process bị kill giữa slot có thể thiếu số đếm phần chưa flush; trường uncertainty phải được giữ khi báo budget.

## Budget và output

Chỉ 4 lineages; mỗi run thêm 36k và 9.000 updates, tổng 60k/13.257 updates. Stage 0–12k và prefix 160 episodes giữ nguyên; suffix stream v1 độc lập train/fixture/battery/assignment, cùng lịch theo seed cho hai capability. Epsilon giữ lịch global, bằng khoảng .05 ở tail; không clear replay.

12 mốc eval mới/run, 27k→60k, mỗi mốc 10 scenarios ×150 slots; tổng 72.000 eval slots mới. Có live JSON một dòng/mốc eval. Full analyzer/export chỉ chạy khi xong từng run; snapshot đã đóng gói không rebuild khi gọi lại. Không có gate/retention rollout mới trong S4. Parent và source cũ không bị sửa.

`exports/` có `review_small_<snapshot>.zip`, `audit_light_<snapshot>.zip`, `repro_full_<snapshot>.zip`, `receipt_<snapshot>.json`. Receipt được ghi sau khi export kết thúc, chứa hash/size và closing timer; tải kèm ZIP để không bỏ sót chi phí export cuối. Gói repro dùng hash audit làm companion, không nhân bản toàn bộ raw evidence. Gói lớn có thể tải bằng output Kaggle. Review không chứa weights/raw và không phải bundle raw để tái audit.

Hình CSV/JSON/SVG/PNG giữ raw dips, trục total steps, đường seed và mean±1 sample SD n=2; throughput=360×utility. Positive service/QoS tách riêng. Tail gồm54/57/60k. Dừng cứng ở60k, không tự kéo90/100k.
