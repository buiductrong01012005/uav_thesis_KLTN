"""Build the self-contained offline Kaggle recovery notebook and refresh its cell."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import re
import textwrap
import zlib

ROOT = Path(__file__).resolve().parents[2]


def build():
    folder = ROOT / "scripts/notebooks"
    helper = (ROOT / "scripts/analysis/recover_bs5_continuous_export.py").read_bytes()
    payload = base64.b64encode(zlib.compress(helper, 9)).decode("ascii")
    packed = (
        "PAYLOAD = (\n"
        + "\n".join('    "' + line + '"' for line in textwrap.wrap(payload, 100))
        + "\n)\n"
    )
    digest = hashlib.sha256(helper).hexdigest()
    legacy_path = folder / "BS5_RECOVER_EVAL600_CELL.py"
    legacy = legacy_path.read_text(encoding="utf-8")
    legacy = re.sub(r"PAYLOAD = \([\s\S]*?\n\)\n", packed, legacy, count=1)
    legacy = re.sub(
        r'assert hashlib.sha256\(helper\).hexdigest\(\) == "[0-9a-f]+"',
        f'assert hashlib.sha256(helper).hexdigest() == "{digest}"',
        legacy,
    )
    legacy_path.write_text(legacy, encoding="utf-8", newline="\n")
    # The frozen-source patch is shared with the existing-session recovery cell.
    patch = legacy[legacy.index("IDENTITY =") : legacy.index("# Compressed stdlib-only")]
    patch = patch.replace(
        'zipfile.ZipFile(EVIDENCE / "source_snapshot.zip")', "zipfile.ZipFile(SOURCE_SNAPSHOT)"
    )
    bootstrap = (
        patch
        + packed
        + f'''
helper = zlib.decompress(base64.b64decode(PAYLOAD))
assert hashlib.sha256(helper).hexdigest() == "{digest}"
TOOL = TOOLS / "recover_bs5_continuous_export.py"
compile(helper, str(TOOL), "exec")
TOOL.write_bytes(helper)
print("Tools ready: frozen-source hashes checked; only reader copies patched.")
'''
    )
    reader = (ROOT / "scripts/analysis/analyze_bs5_continuous.py").read_bytes()
    reader_payload = base64.b64encode(zlib.compress(reader, 9)).decode("ascii")
    bootstrap += (
        "\nANALYZER_PAYLOAD = (\n"
        + "\n".join('    "' + line + '"' for line in textwrap.wrap(reader_payload, 100))
        + ")\n"
    )
    bootstrap += f'''
reader = zlib.decompress(base64.b64decode(ANALYZER_PAYLOAD))
assert hashlib.sha256(reader).hexdigest() == "{hashlib.sha256(reader).hexdigest()}"
compile(reader, "analyze_bs5_continuous.py", "exec")
(TOOLS / "analyze_bs5_continuous.py").write_bytes(reader)
print("Transport-aware reader ready; missing original container hashes will be reported explicitly.")
'''
    locate = """from pathlib import Path
# Some imports are shared with the following notebook cells.
import base64  # noqa: F401
import hashlib  # noqa: F401
import json
import os
import subprocess  # noqa: F401
import sys
import zipfile  # noqa: F401
import zlib  # noqa: F401
from IPython.display import FileLink, Markdown, display  # noqa: F401

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)

# Leave empty for automatic discovery. Set an exact continuous600 folder only
# when multiple result Datasets are attached. If source bytes are missing,
# attach the ORIGINAL source Dataset used for the saved run.
EVIDENCE_OVERRIDE = ""
required = ("checkpoint_selection.json", "protocol.json", "run_registry.json",
            "budget_ledger.json", "controls.json", "source_identity.json", "source_snapshot.zip")
if EVIDENCE_OVERRIDE:
    candidates = [Path(EVIDENCE_OVERRIDE).absolute()]
else:
    # Kaggle can expose Dataset aliases through directory links. Path.rglob on
    # Python 3.13 skips those links by default. Walk them, with cycle protection.
    fast = set()
    for pattern in ("continuous600", "*/continuous600", "*/*/continuous600",
                    "datasets/*/*/continuous600", "datasets/*/*/*/continuous600"):
        fast.update(p for p in INPUT.glob(pattern) if p.is_dir())
    found = {p.resolve(): p for p in sorted(fast)}
    seen, walk_errors = set(), []
    scan = os.walk(INPUT, followlinks=True, onerror=lambda exc: walk_errors.append(str(exc))) if not found else ()
    for base, directories, filenames in scan:
        folder = Path(base)
        real = folder.resolve()
        if real in seen:
            directories[:] = []
            continue
        seen.add(real)
        if folder.name == "continuous600" or "checkpoint_selection.json" in filenames:
            found.setdefault(real, folder)
        directories[:] = [n for n in directories if n not in
                          ("src", "scripts", "tests", "configs", "runs", "parents", "exports")
                          and not n.startswith("repro_")]
    candidates = sorted(found.values())
    if walk_errors:
        print("Input traversal warnings:", walk_errors)
diagnostic = []
for folder in candidates:
    missing = [name for name in required if not (folder / name).is_file()]
    diagnostic.append(dict(folder=str(folder), resolved=str(folder.resolve()), missing=missing))
    print("Found evaluation folder:", folder)
    print("Missing required files:", missing or "NONE")
if not candidates:
    mounted = [str(p) for p in INPUT.iterdir()] if INPUT.is_dir() else []
    print("Input entries:", mounted)
    print("No continuous600 folder or checkpoint_selection.json found. Check Add Input.")
    diagnostic.append(dict(input_entries=mounted))
diagnostic_path = WORK / "bs5_input_diagnostic.json"
diagnostic_path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
assert len(candidates) == 1, (
    "Attach ONE saved result Dataset containing continuous600, or set "
    f"EVIDENCE_OVERRIDE to its exact folder. See diagnostics above: {candidates}"
)
EVIDENCE = candidates[0]
assert EVIDENCE.is_relative_to(INPUT.absolute()), "Select the saved Dataset under /kaggle/input"
assert EVIDENCE.name == "continuous600", "Keep the continuous600 folder structure"
missing = [name for name in required if not (EVIDENCE / name).is_file()]
assert not [name for name in missing if name != "source_snapshot.zip"], (
    f"Missing saved metadata in {EVIDENCE}: {missing}. Do not train again to diagnose this."
)
protocol = json.loads((EVIDENCE / "protocol.json").read_text())
missing_references = [name for name in protocol.get("evidence_files", {})
                      if not (EVIDENCE / name).is_file()]
if missing_references:
    print(f"Original reference containers absent: {len(missing_references)}. Reader will check extracted JSON/source.")
    print("Original compressed hashes remain UNVERIFIED if only extracted content survives.")
    json_refs = [name for name in missing_references if name.endswith(".json.gz")]
    unavailable = []
    for name in json_refs:
        original = EVIDENCE / name
        plain = original.with_suffix("")
        options = (plain, original / plain.name, plain / plain.name,
                   original.parent / original.name.removesuffix(".json.gz"))
        if not any(path.is_file() for path in options):
            unavailable.append(name)
    print(f"Extracted reference JSON found: {len(json_refs) - len(unavailable)}/{len(json_refs)}")
    if unavailable:
        print("Missing JSON examples:", unavailable[:5])

# FROZEN_SOURCE_RESTORER
SOURCE_SNAPSHOT = EVIDENCE / "source_snapshot.zip"
if not SOURCE_SNAPSHOT.is_file():
    SOURCE_SNAPSHOT = restore_eval_source(EVIDENCE, INPUT, WORK)
FLOW = EVIDENCE.parent
registry = json.loads((EVIDENCE / "run_registry.json").read_text())
complete = sum(item["status"] == "complete" for item in registry["trajectories"].values())
print("Input:", FLOW)
print(f"Recorded trajectories complete: {complete}/40. Original status: {registry['status']}")
print("OFFLINE RECOVERY: CPU only; no training, model loading, rollouts or Internet.")
"""
    restorer = (ROOT / "scripts/analysis/restore_bs5_source_snapshot.py").read_text(
        encoding="utf-8"
    )
    locate = locate.replace(
        "# FROZEN_SOURCE_RESTORER", restorer[restorer.index("def restore_eval_source") :]
    )
    (folder / "BS5_RECOVER_DATASET_FIRST_CELL.py").write_text(
        locate, encoding="utf-8", newline="\n"
    )
    run = """OUT = WORK / "bs5_dataset_recovery"
index = 1
while OUT.exists() and not (OUT / "recovery_receipt.json").is_file():
    OUT = WORK / f"bs5_dataset_recovery_{index}"
    index += 1
print("Output:", OUT, flush=True)
# auto preserves an audited evaluation if training exports are missing, and
# includes only selected weights whose bytes match the recorded SHA256.
command = [
    sys.executable, "-B", "-u", str(TOOL), "--delivery-mode", "auto",
    "--allow-extracted-containers",
    "--workflow", str(FLOW),
]
if not (EVIDENCE / "source_snapshot.zip").is_file():
    command += ["--eval-source-snapshot", str(SOURCE_SNAPSHOT)]
command += ["--output", str(OUT)]
subprocess.run(command, check=True, cwd=WORK,
               env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
receipt = json.loads((OUT / "recovery_receipt.json").read_text())
assert receipt["content_checks_passed"] and receipt["added_training_steps"] == receipt["added_environment_steps"] == 0
print("Analysis status:", receipt["integrity_status"], "; original-container integrity accepted:", receipt["accepted"])
print("Original evidence unchanged:", receipt["original_evidence_unchanged"])
print("Best weights included:", receipt["selected_weights_count"], "/ 4")
"""
    download = """availability = json.loads((OUT / "training_availability.json").read_text())
print("Download the results ZIP and receipt below; keep the ZIP before ending the session.")
for path in (Path(receipt["path"]), OUT / "recovery_receipt.json", OUT / "training_availability.json"):
    print(path, f"({path.stat().st_size / 1024**2:.2f} MiB)")
    display(FileLink(path.relative_to(WORK).as_posix()))
display(Markdown((OUT / "analysis/REPORT_VI.md").read_text(encoding="utf-8")))
if not availability["complete_training_export"]:
    print("Original training export incomplete; evaluation remains independently audited.")
    print("Missing training file(s):", availability["missing_training_files"])
if not availability["includes_selected_weights"]:
    print("No checkpoint substitution or training was performed. Exact missing weights:")
    for item in availability["missing_selected_weights"]:
        print(item["model"], item["relative_path"], "SHA256:", item["expected_sha256"])
if not receipt["accepted"]:
    print("CONTENT_ONLY: extracted-data checks passed; original ZIP/GZIP hashes cannot be verified.")
    display(FileLink((OUT / "analysis/transport_integrity.json").relative_to(WORK).as_posix()))
"""
    cells = []
    entries = [
        (
            "markdown",
            "intro",
            """# BS5: khôi phục kết quả từ Dataset đã lưu
Attach **Dataset kết quả** có thư mục `continuous600`, rồi **Run All**.
CPU là đủ; tắt GPU và Internet. Nếu source snapshot/source giải nén không còn,
attach thêm **Dataset source gốc đã dùng lúc chạy**, không dùng bản source mới.

Notebook kiểm tra receipts/budget và toàn bộ raw slots, sửa bộ đọc trên bản sao,
kiểm tra lại ZIP rồi xuất báo cáo. **Không train/eval thêm** và không sửa Dataset.
Weights được giữ đúng theo SHA256 đã ghi. Thiếu training export vẫn xuất được
phân tích đã audit; thiếu weights sẽ báo đúng file, không thay checkpoint khác.
Nếu trajectory thiếu/hỏng hoặc chưa hoàn tất, notebook dừng và giữ bằng chứng.
Kaggle có thể giải nén ZIP/GZIP: khi chỉ còn JSON, báo cáo giữ nguyên các kiểm tra
nội dung nhưng ghi `CONTENT_ONLY`, `accepted=False` vì hash container gốc không
còn kiểm chứng được. Không coi đó là integrity PASS và không sửa protocol.
""",
        ),
        ("code", "locate", locate),
        ("code", "bootstrap", bootstrap),
        ("code", "recover", run),
        ("code", "download", download),
    ]
    for kind, name, source in entries:
        if kind == "code":
            compile(source, name, "exec")
        cell = dict(cell_type=kind, id=name, metadata={}, source=source.splitlines(keepends=True))
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = dict(
        cells=cells,
        nbformat=4,
        nbformat_minor=5,
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python", version="3.11"),
            kaggle=dict(
                accelerator="none",
                isInternetEnabled=False,
                isGpuEnabled=False,
                language="python",
                sourceType="notebook",
            ),
        ),
    )
    path = folder / "bs5_recover_dataset_kaggle.ipynb"
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    (folder / "bs5_recover_dataset_kaggle_v2.ipynb").write_bytes(path.read_bytes())
    (folder / "bs5_recover_dataset_kaggle_v3.ipynb").write_bytes(path.read_bytes())
    return path


if __name__ == "__main__":
    print(build())
