"""Build the portable Kaggle notebook using stdlib only. No execution in this builder."""
import json
from pathlib import Path


def markdown(s): return dict(cell_type="markdown",metadata={},source=s.splitlines(True))
def code(s): return dict(cell_type="code",metadata={},source=s.splitlines(True),outputs=[],execution_count=None)


cells=[markdown("""# C-R60k — collision-v2, game S4 dài1.200 slot

1. Tạo Kaggle Dataset từ **ZIP source được bàn giao**, rồi Add Input vào notebook này.
2. Mặc định dùng CPU. Không cần Internet nếu runtime đã có NumPy, Torch và pytest.
3. Run All. Preflight chạy ngắn; main chỉ C-R/seed551. Không train C-N hoặc tự tăng120k.
4. Kết quả ở `/kaggle/working/bs5_cr_h1200_collision_v2_seed551` và thư mục `_export...` bên cạnh.

**Train fresh với env collision-v2. Không dùng checkpoint/resume của env cũ.** Giữ reward, QMIX, epsilon và curriculum/horizon. Quy tắc mới: ô ngoài charger không trùng; swap endpoint được phép; hub sạc cho phép mọi status cùng ô; Returning chỉ ưu tiên trong phương án khả thi và có thể chờ. Không ép UAV tự né/teleport. Raw ghi blocked moves, Returning waits và pin0 để audit.

Ngân sách chính: **60k train +72k validation**, chưa gồm preflight tối đa512 diagnostic-train và2.000 control/eval bước. S1–S3:150 slot; S4:40 game×1.200. Mốc validation:0/12/24/36/48/60k, mỗi mốc10 game dài.

Notebook mặc định dừng an toàn ở game boundary sau khoảng7 giờ của lệnh train; đây là lựa chọn của notebook, không phải cam kết giới hạn Kaggle. Có thể đổi `MAX_SESSION_SECONDS` trước chạy. Download/export trước khi kết thúc session; Kaggle working không được giả định tự tồn tại qua session.
"""),code('''from pathlib import Path
SOURCE_PATH = None  # None: tìm tự động; hoặc đường dẫn ZIP/thư mục source trong /kaggle/input
RESUME_INPUT_DIR = None  # thư mục chứa export ZIPs + EXPORT_INDEX.json, hoặc merged result root
OUTPUT_ROOT = Path("/kaggle/working/bs5_cr_h1200_collision_v2_seed551")
DEVICE = "cpu"  # chỉ đổi thành "cuda" TRƯỚC prepare run mới
RUN_PREFLIGHT = True
RUN_MAIN = True
MAX_SESSION_SECONDS = 7 * 3600  # None: không đặt giới hạn phiên bằng notebook
'''),code('''import os, sys, zipfile, hashlib, json, shutil, subprocess, time
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
MARKER = "configs/bs5_cr_h1200_60k_collision_v2.json"

def safe_extract(z, target):
    target=Path(target).resolve()
    for member in z.infolist():
        dest=(target/member.filename).resolve()
        if not dest.is_relative_to(target): raise ValueError("Unsafe ZIP member: "+member.filename)
        if (member.external_attr >> 16) & 0o170000 == 0o120000: raise ValueError("ZIP symlink rejected")
    z.extractall(target)

if SOURCE_PATH is None:
    directories=[p.parent.parent for p in INPUT.rglob("bs5_cr_h1200_60k_collision_v2.json")
                 if p.parent.name=="configs" and (p.parent.parent/"scripts/training/run_bs5_cr_h1200_collision.py").exists()]
    archives=[]
    for f in INPUT.rglob("*.zip"):
        try:
            with zipfile.ZipFile(f) as z:
                if "uav_thesis/"+MARKER in z.namelist(): archives.append(f)
        except zipfile.BadZipFile: pass
    candidates=directories+archives
    if len(candidates)!=1:
        raise RuntimeError("Chọn SOURCE_PATH chính xác từ candidates: "+str(candidates))
    SOURCE_PATH=candidates[0]
SOURCE_PATH=Path(SOURCE_PATH)
SOURCE_ROOT=WORK/"bs5_collision_v2_source"/"uav_thesis"
if SOURCE_PATH.is_dir():
    shutil.copytree(SOURCE_PATH,SOURCE_ROOT,dirs_exist_ok=True)
else:
    with zipfile.ZipFile(SOURCE_PATH) as z:
        if z.testzip() is not None: raise RuntimeError("Source ZIP CRC failed")
        safe_extract(z,SOURCE_ROOT.parent)
assert (SOURCE_ROOT/MARKER).exists()
sys.path.insert(0,str(SOURCE_ROOT))
os.chdir(SOURCE_ROOT)
for namespace in ("src", "scripts"):
    loaded=sys.modules.get(namespace)
    if loaded is not None:
        origins=list(getattr(loaded,"__path__",[]))
        if getattr(loaded,"__file__",None): origins.append(loaded.__file__)
        if any(not Path(origin).resolve().is_relative_to(SOURCE_ROOT.resolve()) for origin in origins):
            raise RuntimeError("Kernel đang giữ source cũ. Restart Session rồi Run All notebook collision-v2.")
import numpy as np, torch
try:
    import pytest
except ImportError as e:
    raise RuntimeError("Runtime thiếu pytest. Cài pytest rồi khởi động lại; notebook không tự đổi Torch/CUDA.") from e
print({"source":str(SOURCE_ROOT),"python":sys.version,"numpy":np.__version__,"torch":torch.__version__,"device":DEVICE})
'''),code('''# Preflight bounded; cached by exact source/runtime identity, never silently repeated.
from scripts.training import run_bs5_cr_h1200_collision as runner
runner.setup_runtime(DEVICE,1)
identity=runner.source_identity()
fingerprint={"source":identity["sha256"],"runtime":runner.runtime(DEVICE)}
PREFLIGHT=WORK/"bs5_collision_v2_preflight"
PREFLIGHT.mkdir(exist_ok=True)
receipt=PREFLIGHT/"receipt.json"
if receipt.exists():
    previous=json.loads(receipt.read_text())
    if previous.get("fingerprint")!=fingerprint or previous.get("status")!="passed":
        raise RuntimeError("Preflight cũ khác nguồn/runtime hoặc đã lỗi. Giữ log; không tự lặp compute.")
    print("Preflight đã PASS cùng source/runtime, dùng lại receipt.")
elif RUN_PREFLIGHT:
    runner.atomic_json(receipt,{"status":"started","fingerprint":fingerprint,
        "diagnostic_train_cap":512,"control_eval_cap":2000})
    cmd=[sys.executable,"-B","-m","pytest","-q","-p","no:cacheprovider",
         "tests/test_bs5_cr_h1200_collision.py","tests/test_bs5_collision_contract.py","--basetemp",str(PREFLIGHT/"test_data")]
    result=subprocess.run(cmd,cwd=SOURCE_ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (PREFLIGHT/"pytest.log").write_text(result.stdout)
    print(result.stdout)
    if result.returncode:
        runner.atomic_json(receipt,{"status":"failed","fingerprint":fingerprint,"exit_code":result.returncode})
        raise RuntimeError("Preflight failed. Không chạy main; giữ pytest.log để audit.")
    extra_train=extra_eval=0
    if DEVICE=="cuda":
        # Compatibility smoke only; exact resume parity suite above is on CPU.
        gpu=PREFLIGHT/"cuda_smoke"
        runner.prepare(gpu,"cuda",diagnostic=True)
        runner.run(gpu,stop_after=24)
        extra_train=24; extra_eval=24
    runner.atomic_json(receipt,{"status":"passed","fingerprint":fingerprint,
        "diagnostic_train_steps":384+extra_train,"control_eval_steps":1558+extra_eval,
        "note":"collision contract suite +10 integration tests; CPU exact resume; optional CUDA compatibility smoke"})
else:
    raise RuntimeError("Chưa có preflight PASS. Bật RUN_PREFLIGHT=True trước main.")
'''),code('''# Optional restore. Do not copy only weights/best.pt and call that a resume.
if RESUME_INPUT_DIR is not None:
    src=Path(RESUME_INPUT_DIR)
    if OUTPUT_ROOT.exists(): raise RuntimeError("Output đã tồn tại; không chép đè khi restore.")
    OUTPUT_ROOT.mkdir(parents=True)
    if (src/"protocol.json").exists() and (src/"resume.pt").exists():
        shutil.copytree(src,OUTPUT_ROOT,dirs_exist_ok=True)
    else:
        indices=list(src.rglob("EXPORT_INDEX.json"))
        if len(indices)!=1: raise RuntimeError("Cần đúng một EXPORT_INDEX.json; chọn RESUME_INPUT_DIR cụ thể.")
        index=json.loads(indices[0].read_text())
        for item in index["files"]:
            found=list(src.rglob(item["path"]))
            if len(found)!=1: raise RuntimeError("Thiếu/không rõ companion: "+item["path"])
            f=found[0]
            if hashlib.sha256(f.read_bytes()).hexdigest()!=item["sha256"]: raise RuntimeError("Companion hash failed")
            if item["kind"]=="independent_zip":
                with zipfile.ZipFile(f) as z: safe_extract(z,OUTPUT_ROOT)
            else:
                dest=(OUTPUT_ROOT/item["restore_to"]).resolve()
                if not dest.is_relative_to(OUTPUT_ROOT.resolve()): raise ValueError("unsafe restore_to")
                dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,dest)
    assert (OUTPUT_ROOT/"resume.pt").exists(), "Thiếu full training resume state"

if RUN_MAIN:
    if not OUTPUT_ROOT.exists(): runner.prepare(OUTPUT_ROOT,DEVICE)
    p=runner.read_json(OUTPUT_ROOT/"protocol.json")
    assert p.get("collision_version")=="bs5_endpoint_collision_v2", "Không resume env cũ bằng runner collision-v2"
    assert not p["diagnostic"] and p["total_steps"]==60000 and p["training_seed"]==551
    assert p["device"]==DEVICE and p["source_sha256"]==identity["sha256"]
    # Copy the receipt only when creating this main lineage, not on a restored hashed export.
    if not (OUTPUT_ROOT/"preflight_receipt.json").exists():
        shutil.copy2(receipt,OUTPUT_ROOT/"preflight_receipt.json")
    runner.run(OUTPUT_ROOT,resume=(OUTPUT_ROOT/"resume.pt").exists(),max_seconds=MAX_SESSION_SECONDS)
    print(runner.read_json(OUTPUT_ROOT/"status.json"))
else:
    print("RUN_MAIN=False: main chưa chạy.")
'''),code('''# Export once after complete or graceful boundary pause. No policy re-evaluation.
if RUN_MAIN and (OUTPUT_ROOT/"commit.json").exists():
    EXPORT=runner.export(OUTPUT_ROOT)
    print("Export:",EXPORT)
    from IPython.display import display, FileLink
    for f in sorted(EXPORT.iterdir()):
        if f.is_file(): display(FileLink(str(f)))
    print("Gửi lại review_small.zip, toàn bộ audit_light ZIPs và EXPORT_INDEX.json để phân tích offline.")
    print("Giữ thêm mọi repro_full file + raw companions để resume; không chỉ giữ best weights.")
'''),markdown("""## Đọc kết quả và chạy tiếp

- `review_small.zip`: bảng, curve, protocol và báo cáo nhanh.
- `audit_light*.zip`: đầy đủ raw để audit; giải nén cùng một thư mục. Analyzer không cần Torch/checkpoint.
- `repro_full*` cùng audit files, `EXPORT_INDEX.json`: restore full state và history. Nếu có file `.pt` riêng, đặt vào `restore_to` theo index.
- `paused_at_game_boundary` là dừng an toàn. Tải các output, nén toàn bộ thư mục export thành một ZIP ngoài rồi upload Dataset (giữ nguyên các ZIP companion bên trong), khai báo `RESUME_INPUT_DIR` ở phiên sau, dùng cùng source/runtime/device.
- Lỗi/kill giữa game có thể có compute chưa commit. Runner giữ ledger và chặn tự rollback/rerun vượt cap; gửi status/ledger để kiểm tra, không sửa/xóa chúng để chạy tiếp.
- Dừng tự động tại60k. Nối120k cần quyết định sau khi xem bằng chứng.

Đây là **một seed**, validation đã dùng trước đây. Không suy ra hội tụ hoặc ưu thế curriculum/direct/C-R/C-N mới. Best được chọn bằng mean utility toàn1.200 slot; last full state phục vụ resume.
""")]

notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name="Python 3",language="python",name="python3"),
    language_info=dict(name="python",version="3"),kaggle=dict(accelerator="none",isInternetEnabled=False,
    language="python",sourceType="notebook")),nbformat=4,nbformat_minor=5)
for i,cell in enumerate(cells): cell["id"]=f"bs5-h1200-{i:02d}"
path=Path(__file__).with_name("bs5_cr_h1200_collision_v2_kaggle.ipynb")
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
print(path)
