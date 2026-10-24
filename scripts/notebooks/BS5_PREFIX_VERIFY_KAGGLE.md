# Kiểm tra prefix D-R / D-R-reset, seed 551

Chạy hai implementation gốc trong hai tiến trình sạch, cùng runtime deterministic.
Mỗi nhánh có 3.000 environment steps, 20 episode, 501 updates; eval full-task tại
0/3k. Dừng sau update/target sync tại 3k, trước clear và trước collect bước 3001.
Không chạy ma trận hoặc 24k. Chỉ dùng source mới; không cần dataset kết quả cũ.

Tolerance đóng trong `configs/bs5_prefix_verify_v1.json`: `atol=1e-7`, `rtol=1e-6`;
relative error dùng mẫu số tối thiểu `1e-12`. Giữ cả exact hashes và tolerant
errors. Action/mask/scenario/replay IDs/RNG phải khớp chính xác. Trạng thái
`MATCH_SAME_RUNTIME` còn yêu cầu tensors và endpoint states khớp chính xác.

## Windows / terminal tại repo

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B scripts/training/run_bs5_prefix_verify.py --output results/bs5_prefix_verify_20261023 --device auto
```

`auto` chọn CUDA nếu có, nếu không dùng CPU. Có thể ghi `--device cpu` để chốt
CPU. Không chạy cả hai device chỉ để tìm một kết quả pass. Script đặt môi trường
cho các child process trước import/CUDA init; bật deterministic algorithms strict,
cuDNN deterministic, tắt benchmark/TF32, precision highest, cuBLAS `:4096:8`,
threads=1. Unsupported sẽ báo unresolved; không tự hạ sang warn-only.

## Kaggle

Upload ZIP source mới thành dataset và gắn vào notebook; Kaggle tự giải nén ZIP.
Giữ nguyên Torch/NumPy của runtime, không tự nâng phiên bản giữa hai nhánh.

Cell 1 — tìm source đã giải nén:

```python
from pathlib import Path
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
matches = sorted(Path("/kaggle/input").rglob("scripts/training/run_bs5_prefix_verify.py"))
assert len(matches) == 1, f"Cần đúng một source mới; tìm được: {matches}"
REPO = matches[0].parents[2]
OUT = Path("/kaggle/working/bs5_prefix_verify_20261023")
print("REPO:", REPO)
print("OUT:", OUT)
```

Cell 2 — chuyển vào source:

```python
os.chdir(REPO)
print(Path.cwd())
```

Cell 3 — chạy đúng cặp prefix:

```python
import subprocess
subprocess.run([
    sys.executable, "-B", "scripts/training/run_bs5_prefix_verify.py",
    "--output", str(OUT), "--device", "auto",
], check=True)
```

Cell 4 — xem kết luận và tải ZIP review đã được script đóng cuối tác vụ:

```python
import json
from IPython.display import FileLink, display

print(json.loads((OUT / "verification.json").read_text()))
os.chdir("/kaggle/working")
for package in sorted((OUT / "exports").glob("*.zip")):
    display(FileLink(str(package.relative_to(Path.cwd()))))
```

Gửi ZIP `bs5_prefix_verify_review_*.zip` để phân tích. Gói có report, ledger,
runtime/config/source identity, comparison, per-step/update diagnostics, raw trace
đủ 20 episode mỗi nhánh và eval 0/3k; không chứa optimizer tensors. Các NPZ tensor
đầy đủ để debug chỉ giữ tại `OUT/trials/*/diagnostic/numeric/`, không đưa vào review.
Không nén lại toàn bộ `/kaggle/working` vì sẽ kéo theo các tensor cache debug lớn.

## Budget và chạy tiếp khi gián đoạn

Mặc định tổng 6.000 train steps. Nếu khớp thì dừng. Nếu khác, script cho phép đúng
một self-repeat A, tới first mismatch + 150 bước, làm tròn theo episode, tối đa
3k; tổng tự động tối đa 9k. Trần ledger là 15k để giữ giới hạn toàn tác vụ, không
phải yêu cầu dùng hết; script không tự sửa khoa học hoặc chạy cặp thứ ba.

`budget_ledger.json` reserve toàn prefix trước khi launch, rồi charge trước mỗi
environment step. Reservation bị gián đoạn vẫn giữ; không reset budget khi resume.
Chỉ dùng `--resume` cùng output/device/source để dùng lại prefix đã hoàn tất và
chạy phần chưa bắt đầu. Nếu một prefix đã reserve nhưng dở dang, script dừng với
unresolved để kiểm tra, không tự chạy lại âm thầm hoặc áp dụng clear khi resume.
Nếu process bị kill cứng, `active.lock` còn lại: xác nhận process đã dừng trước
khi xử lý lock; đừng tạo output mới để né ledger.

`--prepare-only` chỉ đóng identity/config/ledger, không chạy train. Để chạy output
đã prepare, thêm `--resume`. Output không được ghi đè.

## Đọc review offline

Giải nén review ra thư mục riêng rồi chạy, không cần Torch/NumPy/checkpoint:

```text
python -B analyze_bs5_prefix_verify.py . --verify-review
```

Lệnh xác minh danh sách/hash các file review và tái dựng comparison từ log đã
ghi. Max errors weights/grad được đo trong lúc hai tiến trình chạy từ tensors;
review giữ hash/error, không đủ tensors để tính lại từng error của weights.
Trạng thái chưa pass phải được bàn giao nguyên trạng. Chỉ xác nhận tương đương
trong runtime diagnostic; không hồi tố quy +10,2% lịch sử cho replay reset.
