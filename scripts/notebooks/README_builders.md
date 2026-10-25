# Bộ dựng notebook BS5

Năm bộ dựng H1200/collision/Return dùng chung `_notebook.py` để tạo cell, metadata và JSON UTF-8 với newline LF. Nội dung notebook được giữ nguyên từng byte so với bản đã bàn giao.

| Bộ dựng | Notebook |
|---|---|
| `build_bs5_cr_h1200_notebook.py` | `bs5_cr_h1200_60k_kaggle.ipynb` |
| `build_bs5_cr_h1200_collision_notebook.py` | `bs5_cr_h1200_collision_v2_kaggle.ipynb` |
| `build_bs5_cr_h1200_return_notebook.py` | `bs5_cr_h1200_return_v3_kaggle.ipynb` |
| `build_bs5_crcn_return_v3_notebook.py` | `bs5_crcn_return_v3_kaggle.ipynb` |
| `build_bs5_return_v4_notebook.py` | `bs5_return_v4_pair_kaggle.ipynb` |

Chạy trực tiếp vẫn ghi notebook cạnh script:

```powershell
python -B scripts/notebooks/build_bs5_return_v4_notebook.py
```

Import module không ghi file. Có thể dựng vào thư mục riêng để kiểm tra:

```python
from pathlib import Path
from scripts.notebooks.build_bs5_return_v4_notebook import build

build(Path("archive/notebook_preview"))
```

Các bộ dựng chỉ tạo file notebook, không thực thi cell, cài thư viện, train hoặc eval. Return-v3 dùng notebook collision-v2 đi kèm làm đầu vào. Khi chuyển script ra ngoài repo, giữ `_notebook.py` cạnh script và giữ notebook đầu vào này nếu dùng bộ dựng Return-v3.

`tests/test_bs5_notebook_builders.py` kiểm tra nội dung đầu ra, cú pháp cell, import không ghi file và chạy script ngoài repo.

Run đã chuẩn bị hoặc đang resume phải dùng đúng source snapshot/commit và runtime đã khóa cho run đó. Giữ bản source đã dùng khi chạy; không thay bằng source refactor giữa phiên hoặc sửa manifest để vượt kiểm tra hash. Bản trước refactor nằm ở commit `1c96674` và ZIP nguồn đã bàn giao ngày 2026-10-26.
