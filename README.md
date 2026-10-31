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

Baseline heuristic Hybrid (H-only), chạy kiểm tra 96 slot:

```sh
python -B -m scripts.evaluation.run_bs5_heuristic_ho --diagnostic --output results/heuristic_ho_smoke
```

Chạy test chính thức yêu cầu đúng phiên bản runtime được khóa trong
`configs/bs5_heuristic_ho_extension.json`; dùng `--help` để chọn phase và seed.
