# BS5 launch A/B: Kaggle

[Import this notebook](bs5_launch_ab_kaggle.ipynb), or copy the four cells below.

Upload the source ZIP as a Kaggle Dataset, then **Add Input**. Kaggle extracts the ZIP automatically. Source is read directly from `/kaggle/input`; every experiment output goes to `/kaggle/working`.

Optionally enable a GPU before cell 2. This notebook uses Kaggle's installed NumPy and PyTorch and requires no installation cell. Run the four code cells in order. Cell 3 runs preparation and then all eight main runs; running it again resumes committed evaluations. Do not edit Dataset source after preparation.

Use a new `RUN_NAME` for an independent experiment. Resume works while the same output folder survives; a new Kaggle session may require restoring the previous full experiment folder from `repro_full.zip` (including `resume.pt`). `/kaggle/working` is not guaranteed to persist across sessions. Keep the same source version and frozen device when resuming. When restoring a full export, also copy `source/uav_thesis_source.zip` to `source_snapshot.zip` at the restored output root.

Cell 4 can also save an interrupted experiment after the training process stops; check the printed status. Download both bundles or save the notebook version with outputs before ending the session.

## 1. Find the already extracted Dataset source

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
        for p in INPUT.rglob("run_bs5_launch_ab.py")
        if p.parent.name == "training"
        and p.parents[1].name == "scripts"
        and SOURCE_FILTER in str(p)
        and all(
            (p.parents[2] / name).is_file()
            for name in (
                "src/envs/bs5_launch_ab_env.py",
                "configs/bs5_launch_ab_v1.json",
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

## 2. Change directory and choose writable output/device

```python
import json
import torch

os.chdir(ROOT)
WORK = Path("/kaggle/working")
RUN_NAME = "bs5_launch_ab_20261022"  # Change for an independent experiment.
if not RUN_NAME or RUN_NAME in {".", ".."} or any(c in RUN_NAME for c in "/\\"):
    raise ValueError("RUN_NAME must be a single folder name")
OUT = WORK / RUN_NAME
CFG = WORK / f"{RUN_NAME}_config.json"
PROTOCOL = OUT / "protocol.json"
WORK.mkdir(parents=True, exist_ok=True)
if PROTOCOL.is_file():
    config = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    print("Reusing frozen protocol; device:", config["device"])
else:
    if OUT.exists():
        raise RuntimeError("Incomplete preparation: preserve this folder and choose a new RUN_NAME")
    config = json.loads((ROOT / "configs/bs5_launch_ab_v1.json").read_text(encoding="utf-8"))
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    CFG.write_text(json.dumps(config, indent=2), encoding="utf-8")
if config["device"].startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("This protocol needs CUDA. Enable a GPU accelerator before resuming.")
print("CWD:", Path.cwd())
print("Output:", OUT)
print("Device:", config["device"], "| torch:", torch.__version__)
```

## 3. Prepare once, then train or resume

```python
import subprocess

BASE = [sys.executable, "-B", "-u", "-m", "scripts.training.run_bs5_launch_ab"]


def run_bs5(*args):
    subprocess.run(BASE + list(args), cwd=ROOT, check=True, env=os.environ.copy())


if not PROTOCOL.is_file():
    run_bs5("--prepare", "--config", str(CFG), "--outdir", str(OUT))
protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
print("Frozen steps/run:", protocol["steps_per_run"], flush=True)
print("Runtime estimate:", protocol["runtime_estimate"], flush=True)
run_bs5("--run", "--resume", "--outdir", str(OUT))
```

## 4. Export ZIP bundles for download

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
        "scripts.training.run_bs5_launch_ab",
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

