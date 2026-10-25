# BS5 preflight

Các entry point này chạy bộ kiểm tra có ngân sách trước thí nghiệm. Chúng có thể tạo diagnostic rollout và ghi ledger; chỉ chạy theo hướng dẫn của đúng giao thức.

| Entry point | Hướng dẫn |
|---|---|
| `preflight_bs5_return.py` | [Return-v3](../notebooks/README_bs5_return_v3.md) |
| `preflight_bs5_crcn_return_v3.py` | [C-R/C-N Return-v3](../notebooks/README_bs5_crcn_return_v3.md) |
| `preflight_bs5_return_v4.py` | [Return-v4](../notebooks/README_bs5_return_v4.md) |

C-R/C-N và Return-v4 cần Dataset tài sản riêng như mô tả trong [experiments](../experiments/README.md). ZIP mã nguồn không chứa các checkpoint/tape đó. Không hạ kiểm tra hash, runtime hay giới hạn compute khi thiếu tài sản.

Giữ nguyên đường dẫn `scripts/verification/` vì các notebook mới gọi trực tiếp các entry point này. Các script kiểm tra thiết lập cũ vẫn ở `scripts/validation/`.
