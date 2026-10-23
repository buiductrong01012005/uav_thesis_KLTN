# BS5 curriculum: b?n cell Kaggle

Upload ZIP m? ngu?n m?i th?nh Dataset r?i Add Input; Kaggle t? gi?i n?n. B?t GPU tr??c cell 2 n?u d?ng. M? notebook `bs5_curriculum_kaggle.ipynb` ho?c ch?p l?n l??t b?n cell d??i ??y.

Cell 3 ki?m tra preflight + smoke tr??c r?i m?i ch?y 8 run ? 24k. ??i `RUN_NAME` khi t?o th? nghi?m m?i. Ch?y cell 4 sau khi ti?n tr?nh train k?t th?c ho?c d?ng ?? t?i `audit_light` v? `repro_full`. Resume qua phi?n kh?c c?n Dataset `repro_full`, c?ng source/device v? `RESTORE_FROM`; `audit_light` kh?ng c? checkpoint.

## Cell 1

```python
from pathlib import Path
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
INPUT = Path("/kaggle/input")
SOURCE_FILTER = ""  # Optional dataset name if multiple source versions exist.
candidates = sorted(
    {
        p.parents[2]
        for p in INPUT.rglob("run_bs5_curriculum_pilot.py")
        if p.parent.name == "training"
        and p.parents[1].name == "scripts"
        and SOURCE_FILTER in str(p)
        and all(
            (p.parents[2] / name).is_file()
            for name in (
                "src/envs/bs5_curriculum_env.py",
                "configs/bs5_curriculum_fixed_v1.json",
                "AGENTS.md",
            )
        )
    }
)
if len(candidates) != 1:
    raise RuntimeError(
        f"Expected one source root, found {len(candidates)}: {candidates}. "
        "Add the source Dataset or set SOURCE_FILTER."
    )
ROOT = candidates[0].resolve()
print("Source:", ROOT)
```

## Cell 2

```python
import json
import torch

os.chdir(ROOT)
WORK = Path("/kaggle/working")
RUN_NAME = "bs5_curriculum_fixed_20261023"  # Change for an independent experiment.
if not RUN_NAME or RUN_NAME in {".", ".."} or any(c in RUN_NAME for c in "/\\"):
    raise ValueError("RUN_NAME must be a single folder name")
OUT = WORK / RUN_NAME
CFG = WORK / f"{RUN_NAME}_config.json"
PROTOCOL = OUT / "protocol.json"
WORK.mkdir(parents=True, exist_ok=True)
# Optional: path to an already extracted repro_full Dataset from a prior session.
RESTORE_FROM = ""
if RESTORE_FROM and not OUT.exists():
    import shutil
    previous = Path(RESTORE_FROM)
    if not (previous / "protocol.json").is_file():
        raise RuntimeError("RESTORE_FROM must contain protocol.json")
    shutil.copytree(previous, OUT)
    source_zip = OUT / "source/uav_thesis_source.zip"
    if not source_zip.is_file():
        raise RuntimeError("Full resume needs source/uav_thesis_source.zip; audit_light cannot resume")
    shutil.copy2(source_zip, OUT / "source_snapshot.zip")
if PROTOCOL.is_file():
    config = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    print("Reusing frozen protocol; device:", config["device"])
else:
    if OUT.exists():
        raise RuntimeError("Incomplete preparation: preserve this folder and choose a new RUN_NAME")
    config = json.loads((ROOT / "configs/bs5_curriculum_fixed_v1.json").read_text(encoding="utf-8"))
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    CFG.write_text(json.dumps(config, indent=2), encoding="utf-8")
if config["device"].startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("This protocol needs CUDA. Enable a GPU accelerator before resuming.")
print("CWD:", Path.cwd())
print("Output:", OUT)
print("Device:", config["device"], "| torch:", torch.__version__)
```

## Cell 3

```python
import subprocess

BASE = [sys.executable, "-B", "-u", "-m", "scripts.training.run_bs5_curriculum_pilot"]


def run_bs5(*args):
    subprocess.run(BASE + list(args), cwd=ROOT, check=True, env=os.environ.copy())


if not PROTOCOL.is_file():
    run_bs5("--prepare", "--config", str(CFG), "--outdir", str(OUT))
protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
print("Frozen steps/run:", protocol["steps_per_run"], flush=True)
print("Smoke acceptance:", json.loads((OUT / "preflight/smoke_acceptance.json").read_text()), flush=True)
run_bs5("--run", "--resume", "--outdir", str(OUT))
```

## Cell 4

```python
import shutil
import zipfile
from IPython.display import FileLink, display

# Run only after cell 3 has finished or stopped.
subprocess.run(
    [
        sys.executable,
        "-B",
        "-u",
        "-m",
        "scripts.training.run_bs5_curriculum_pilot",
        "--export",
        "--outdir",
        str(OUT),
    ],
    cwd=ROOT,
    check=True,
    env=os.environ.copy(),
)
registry = json.loads((OUT / "run_registry.json").read_text(encoding="utf-8"))
print("Experiment status:", registry["status"])
os.chdir(WORK)
for kind in ("audit_light", "repro_full"):
    source = OUT / "exports" / f"{kind}.zip"
    with zipfile.ZipFile(source) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")
    destination = WORK / f"{RUN_NAME}_{kind}.zip"
    shutil.copy2(source, destination)
    print(destination, f"({destination.stat().st_size / 1024**2:.1f} MiB)")
    display(FileLink(destination.name))
```
