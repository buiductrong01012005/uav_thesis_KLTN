# UAV Thesis

Mã nguồn mô phỏng, huấn luyện và đánh giá UAV.

- `src/`: môi trường, thuật toán và baseline.
- `scripts/`: runner train, đánh giá, audit và kiểm tra.
- `configs/`: cấu hình và provenance thí nghiệm.
- `tests/`: kiểm thử.

Dùng Python 3.11 hoặc 3.12, chạy từ gốc repo:

```sh
python -m pip install -r requirements.txt
python -m pytest -q -p no:cacheprovider
```

Dùng `python -m scripts.<nhóm>.<runner> --help` để xem tham số.
Dữ liệu, checkpoint, báo cáo và notebook được giữ cục bộ.
