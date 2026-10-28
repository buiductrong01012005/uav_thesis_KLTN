"""Per-notebook review/restart export for the four optional-Hybrid runs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

VERSION = "bs5_hybrid_training_1"
RUNS = {
    "CN_TC": ("C-N", "tc_priority"),
    "CR_TC": ("C-R", "tc_priority"),
    "CH_TC": ("C-H", "tc_priority"),
    "CH_BAL": ("C-H", "balanced"),
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export(runs, output, run_id):
    if run_id not in RUNS:
        raise ValueError("Unknown run ID")
    runs, output = Path(runs), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    statuses = {}
    accepted = []
    for path in sorted(runs.rglob("status.json")):
        status = json.loads(path.read_text())
        if status.get("training_env_version") != VERSION:
            continue
        if (status.get("arm"), status.get("preset")) != RUNS[run_id]:
            continue
        statuses[str(path.parent.relative_to(runs))] = status
        accepted.append(path.parent)
    files = sorted(
        {
            p
            for folder in accepted
            for p in folder.rglob("*")
            if p.is_file()
            and p.suffix not in {".lock", ".tmp", ".pyc", ".pyo", ".zip"}
            and "__pycache__" not in p.parts
            and p.name != ".DS_Store"
        }
    )
    artifacts = {}
    for suffix, include_resume in [("REVIEW", False), ("RESUME", True)]:
        dest = output / f"{run_id}_{suffix}.zip"
        temp = dest.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
            for path in files:
                if include_resume or path.name != "resume.pt":
                    z.write(path, path.relative_to(runs))
        with temp.open("r+b") as f:
            os.fsync(f.fileno())
        with zipfile.ZipFile(temp) as z:
            if z.testzip() is not None:
                raise IOError("Corrupt export")
        temp.replace(dest)
        artifacts[dest.name] = dict(bytes=dest.stat().st_size, sha256=sha(dest))
    index = dict(
        version="bs5_hybrid_four_export_1",
        run_id=run_id,
        status=statuses,
        files=artifacts,
        primary_selection="best.pt by mean weighted TCF on two fixed validation seeds; utility supplementary",
        completion="Pilot pause is not complete120k; continue same Dataset/runtime with pause_at_step0",
        resume="Full optimizer/replay/RNG at episode boundary, isolated per run ID; old resumes incompatible",
    )
    path = output / f"{run_id}_INDEX.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(index, indent=2) + "\n")
    temp.replace(path)
    print(json.dumps(index, indent=2), flush=True)
    return index


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-id", choices=list(RUNS), required=True)
    a = p.parse_args()
    export(a.runs, a.output, a.run_id)
