# D-R-reset — Kaggle, đúng hai main runs

Upload ZIP source mới làm Dataset rồi **Add Input**. Kaggle tự giải nén; cell 1 tìm source đã giải nén. Bật **GPU** và **Internet** trước khi chạy. Dùng notebook `bs5_direct_replay_reset_kaggle.ipynb` hoặc copy bốn cell dưới đây.

Main chỉ có `D-R-reset_seed551/552`, mỗi run 24k steps, scratch. Smoke dùng seed riêng 955551/955552, mỗi run 384 steps, không nhập vào main. Không chạy runner curriculum cũ. Dataset đã chứa compact baseline ~2.3 MB; không cần upload ZIP baseline ~700 MB.

### Cell 1 — tìm source Dataset

```python
from pathlib import Path
import os
import sys
import json
import subprocess

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
SOURCE_FILTER = ""  # Điền tên Dataset nếu có nhiều bản source.
candidates = sorted({
    p.parents[2] for p in Path("/kaggle/input").rglob("run_bs5_direct_replay_reset.py")
    if p.parent.name == "training" and p.parents[1].name == "scripts"
    and SOURCE_FILTER in str(p)
    and (p.parents[2] / "configs/bs5_direct_replay_reset_baseline_v2.json").is_file()
    and (p.parents[2] / "AGENTS.md").is_file()
})
if len(candidates) != 1:
    raise RuntimeError(f"Cần đúng một source; tìm thấy {candidates}. Chỉnh SOURCE_FILTER.")
ROOT = candidates[0].resolve()
print("Source:", ROOT)
```

### Cell 2 — cd, runtime và output

```python
import importlib.metadata

os.chdir(ROOT)
# Cài trước khi import torch trong kernel. Runner luôn chạy bằng subprocess mới.
if importlib.metadata.version("torch").split("+")[0] != "2.11.0":
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--no-cache-dir",
        "torch==2.11.0", "--index-url", "https://download.pytorch.org/whl/cu128"
    ], check=True)
if importlib.metadata.version("numpy") != "2.1.3":
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "numpy==2.1.3"], check=True)
subprocess.run([sys.executable, "-B", "-c",
    "import torch,numpy; print('torch',torch.__version__,'numpy',numpy.__version__); "
    "assert torch.cuda.is_available(), 'Bật GPU Accelerator'; print(torch.cuda.get_device_name(0))"
], check=True)
WORK = Path("/kaggle/working")
RUN_NAME = "bs5_dr_reset_20261023"  # Đổi tên khi mở thí nghiệm độc lập.
if not RUN_NAME or RUN_NAME in {".", ".."} or any(c in RUN_NAME for c in "/\\"):
    raise ValueError("RUN_NAME phải là một tên thư mục")
OUT = WORK / RUN_NAME
SMOKE = WORK / (RUN_NAME + "_smoke")
BASE = [sys.executable, "-B", "-u", "-m", "scripts.training.run_bs5_direct_replay_reset"]
def run_reset(*args):
    subprocess.run(BASE + list(args), cwd=ROOT, check=True, env=os.environ.copy())
print("CWD:", Path.cwd(), "\nMain output:", OUT)
```

### Cell 3 — smoke một lần, prepare rồi train/resume

```python
if not (SMOKE / "protocol.json").is_file():
    run_reset("--prepare", "--smoke", "--device", "cuda", "--outdir", str(SMOKE))
smoke_registry = json.loads((SMOKE / "run_registry.json").read_text())
if smoke_registry["status"] != "complete":
    run_reset("--run", "--resume", "--export-policy", "final", "--outdir", str(SMOKE))
elif not (SMOKE / "analysis/verification.json").is_file():
    run_reset("--export", "--outdir", str(SMOKE))
assert json.loads((SMOKE / "analysis/verification.json").read_text())["complete"]
timings = [json.loads(p.read_text()) for p in (SMOKE / "timing").glob("*.json")]
outer = {k: sum(t["outer"].get(k, 0) for t in timings)
         for k in ("setup", "train_loop", "eval", "persistence", "analysis", "export", "other")}
# Chỉ là ước lượng: smoke horizon6, main150; lifecycle/oracle slot76 và kích thước
# replay/evidence lớn hơn khiến phép ngoại suy có sai số đáng kể.
estimate_h = (outer["train_loop"] / 768 * 48000 + outer["eval"] / 216 * 27000) * 1.5 / 3600
print("Smoke outer seconds:", outer)
print("Ước lượng train+eval ×1.5 (giờ), chưa gồm export lớn:", round(estimate_h, 2))
if not (OUT / "protocol.json").is_file():
    run_reset("--prepare", "--outdir", str(OUT))
# Prepare kiểm initial weights từng seed so C-R/D-R cũ; sai hash thì dừng trước rollout.
run_reset("--run", "--resume", "--outdir", str(OUT))
```

### Cell 4 — export nếu cần, link tải ZIP

```python
from IPython.display import FileLink, display
import hashlib

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
# Chỉ chạy sau khi cell 3 đã dừng. Reconcile không train, không clear replay mới.
run_reset("--reconcile", "--outdir", str(OUT))
protocol = json.loads((OUT / "protocol.json").read_text())
registry = json.loads((OUT / "run_registry.json").read_text())
commits = {name: json.loads((OUT / "runs" / name / "commit.json").read_text())
           for name, entry in registry["runs"].items() if entry["status"] != "not_started"}
snapshot = canonical_hash(dict(protocol=protocol, commits=commits))
latest_path = OUT / "exports/latest.json"
latest = json.loads(latest_path.read_text()) if latest_path.is_file() else {}
if latest.get("snapshot_sha256") != snapshot:
    run_reset("--export", "--outdir", str(OUT))
    latest = json.loads(latest_path.read_text())
print("Status:", registry["status"], "| Tải review ZIP trước để xem kết quả.")
for name, expected in sorted(latest["bundles"].items(), key=lambda item: "_review.zip" not in item[0]):
    path = OUT / "exports" / name
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == expected, f"ZIP hash mismatch: {name}"
    print(name, f"{path.stat().st_size / 1024**2:.2f} MiB")
    display(FileLink(str(path)))
display(FileLink(str(latest_path)))  # Receipt có timer gồm export/review hoàn tất.
```

Giữ **audit_light + repro_full + review + latest.json** hoặc Save Version với outputs trước khi kết thúc session. `/kaggle/working` có thể mất khi đổi session. Review vài MB dùng đọc nhanh; audit có toàn bộ raw mới; repro có source/weights/resume và hash companion audit. Không có checkpoint lịch sử trong các gói mới.

Nếu audit vượt 480 MiB, exporter tạo thêm shard theo seed (đánh dấu partial). Có thể tải tất cả shard thay audit lớn rồi giải nén chung vào một thư mục để khớp đầy đủ `SHA256SUMS.json`. Một shard riêng chưa phải complete audit; xem `seed_shards` trong receipt/companion để kiểm hash. Review ZIP vẫn là gói ưu tiên gửi để xem kết quả.

### Khôi phục sang session khác

Thực hiện trước cell 3. Giải nén audit vào `OUT`, rồi giải nén repro **cùng OUT**: `runs/*/resume.pt` và `runs/*/checkpoints` sẽ ghép vào evidence. Repro chứa `source_snapshot.zip`; giải nén source đó vào thư mục riêng và dùng `ROOT = <thư mục>/uav_thesis`, `os.chdir(ROOT)`. Giữ nguyên tên/path tương đối `source_snapshot.zip` ở OUT. Không sửa source/config/device sau prepare. Có thể kiểm companion hash audit trong `companion.json`. Sau đó cell 3 dùng `--resume`; smoke cũ có thể phục hồi riêng hoặc chạy smoke mới, không trộn main data.

Crash chỉ giữ đến durable boundary gần nhất: step0, các mốc eval hoặc reset. Tối đa 3.000 train steps chưa commit có thể phải chạy lại, kể cả khi đã collect đến mốc nhưng chưa lưu xong checkpoint/eval. State `resume.pt` chứa marker before/after clear; log sau commit được đưa vào `aborted_attempts/`. Không overwrite evidence cũ.

### Lệnh riêng (Windows hoặc Kaggle terminal)

```text
python -B -m scripts.training.run_bs5_direct_replay_reset --prepare --outdir results/bs5_dr_reset_main
python -B -m scripts.training.run_bs5_direct_replay_reset --run --resume --outdir results/bs5_dr_reset_main
python -B -m scripts.training.run_bs5_direct_replay_reset --reconcile --outdir results/bs5_dr_reset_main
python -B -m scripts.training.run_bs5_direct_replay_reset --export --outdir results/bs5_dr_reset_main
python -B -m scripts.analysis.analyze_bs5_direct_replay_reset_bundle <audit_light.zip> --output <report-dir> --require-complete
```

CPU chỉ dùng smoke trên runtime local khác phiên bản: thêm `--smoke --device cpu` khi prepare và chọn output riêng. Main dùng đúng runtime PyTorch2.11.0 và cần khớp initial hash; không nới guard để chạy cho qua. `--export-policy final` chỉ export cuối ma trận; `manual` chỉ ghi log/checkpoint, export bằng lệnh riêng. Mặc định `per-run`, đúng hai lần export khi hai run mới hoàn tất; không analyze/export theo từng eval.
