# BS5 paired experiments

Thư mục này điều phối các bước train/eval theo giao thức đã khóa. Các runner huấn luyện riêng nằm trong `scripts/training/`; các bộ đọc audit nằm trong `scripts/analysis/`.

| Giao thức | Entry point | Tài sản cần có ở repo root |
|---|---|---|
| C-R/C-N Return-v3 | `run_bs5_crcn_return_v3.py` | `assets/bs5_return_v3_cr/` |
| Đánh giá Return-v4 với hai checkpoint đã chọn | `run_bs5_return_v4_pair.py` | `assets/bs5_return_v4_pair/` |

`bs5_crcn_common.py` và `bs5_return_v4_common.py` xác minh manifest, checkpoint và UE tape của từng Dataset. Bản ZIP chỉ chứa mã nguồn không cung cấp các tài sản này. Giữ đúng runtime và dữ liệu được khóa; không thay checkpoint hoặc tạo tape mới để vượt kiểm tra.

Chạy lệnh từ repo root; xem CLI bằng:

```powershell
python -B -m scripts.experiments.run_bs5_crcn_return_v3 --help
python -B -m scripts.experiments.run_bs5_return_v4_pair --help
```

Hướng dẫn chạy và tiếp tục phiên Kaggle:

- [C-R/C-N Return-v3](../notebooks/README_bs5_crcn_return_v3.md)
- [Return-v4](../notebooks/README_bs5_return_v4.md)

Return-v3 gồm train C-N theo recipe đã khóa và paired evaluation. Return-v4 chỉ đổi controller Returning và đánh giá lại checkpoint cố định, không huấn luyện. Hai phiên bản được giữ riêng để bảo toàn khả năng tái lập.
