"""Version-checked review/resume exports for four-arm399600-step runs."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

VERSION = "bs5_fourarm400k_v1"


def export(runs, output, name):
    if not name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name
    ):
        raise ValueError("Use a simple alphanumeric export name")
    runs = runs.resolve()
    output.mkdir(parents=True, exist_ok=True)
    statuses = list(runs.rglob("status.json"))
    if not statuses:
        raise ValueError("No completed episode status found")
    for path in statuses:
        status = json.loads(path.read_text())
        if status["training_env_version"] != VERSION:
            raise ValueError("Old environment in output tree")
    files = [
        p
        for p in runs.rglob("*")
        if p.is_file()
        and p.suffix not in [".zip", ".tmp", ".lock", ".pyc", ".pyo"]
        and "__pycache__" not in p.parts
    ]
    result = {}
    for label in ["REVIEW", "RESUME"]:
        path = output / f"{name}_{label}.zip"
        temp = path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
            for p in files:
                if label == "REVIEW" and p.name == "resume.pt":
                    continue
                z.write(p, Path(name) / p.relative_to(runs))
        with zipfile.ZipFile(temp) as z:
            assert z.testzip() is None
        temp.replace(path)
        result[path.name] = dict(
            bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    (output / f"{name}_exports.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", default="DRL400K")
    a = p.parse_args()
    print(json.dumps(export(a.runs, a.output, a.name), indent=2))
