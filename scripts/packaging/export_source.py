"""Export a complete source ZIP and verify imports from its extracted contents."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CACHES = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
EXCLUDED_DIRS = CACHES | {".git", ".codegraph", ".venv"}
SOURCE_DIRS = ("src", "scripts", "tests", "configs")
ROOT_FILES = ("run.py", "AGENTS.md", "README.md", ".gitignore", "pyproject.toml")
REQUIRED = (
    "run.py", "AGENTS.md", "requirements.txt", "scripts/w00.py",
    "scripts/compat/windows/fcntl.py", "scripts/training/run_bs5_fourarm400k.py",
    "scripts/evaluation/eval_w00_checkpoint.py",
    "scripts/evaluation/run_fourarm400k_drl_benchmark.py",
    "scripts/evaluation/run_fourarm400k_heuristics.py",
    "configs/bs5_fourarm400k.json", "configs/bs5_fourarm400k_heuristic_benchmark.json",
    "src/envs/__init__.py", "src/envs/bs5_curriculum_env.py",
    "src/envs/bs5_fourarm400k.py", "src/envs/bs5_service_v2.py",
    "src/envs/bs5_tcf_env.py", "src/envs/bs5_energy_swap.py",
    "src/rl/bs5_fourarm400k_qmix.py", "src/rl/qmix.py",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def is_link(path):
    return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0)
                                     & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def source_tree(folder):
    """Do not infer virtual environments from names such as src/envs/."""
    if not folder.is_dir() or is_link(folder):
        raise ValueError(f"Missing or linked source directory: {folder}")

    def walk_error(error):
        raise error  # A denied directory must fail the export, never disappear silently.

    for current, directories, names in os.walk(folder, followlinks=False, onerror=walk_error):
        current = Path(current)
        selected = []
        for name in sorted(directories):
            child = current / name
            if name in EXCLUDED_DIRS or (child / "pyvenv.cfg").is_file():
                continue
            if is_link(child):
                raise ValueError(f"Linked source directory must be reviewed: {child}")
            selected.append(name)
        directories[:] = selected
        for name in sorted(names):
            path = current / name
            if path.suffix.lower() in {".pyc", ".pyo", ".zip"}:
                continue
            if is_link(path):
                raise ValueError(f"Linked source file must be reviewed: {path}")
            yield path


def collect_files(root):
    root = Path(root).resolve()
    files = [p for folder in SOURCE_DIRS for p in source_tree(root / folder)]
    files.extend(root / name for name in ROOT_FILES if (root / name).is_file())
    files.extend(root.glob("requirements*.txt"))
    selected = {p.relative_to(root).as_posix(): p for p in files}
    missing = set(REQUIRED) - selected.keys()
    if missing:
        raise ValueError(f"Missing train/eval source files: {sorted(missing)}")
    return dict(sorted(selected.items()))


def verify_archive(archive, root, expected):
    """Check CRC, every member hash, and envs independently of export selection."""
    root = Path(root).resolve()
    env_files = {p.relative_to(root).as_posix() for p in source_tree(root / "src/envs")}
    if not env_files:
        raise ValueError("src/envs/ is empty")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or z.testzip() is not None:
            raise ValueError("Corrupt ZIP or duplicate members")
        members = {"uav_thesis/" + name for name in expected}
        required = {"uav_thesis/" + name for name in set(REQUIRED) | env_files}
        if not required <= set(names):
            raise ValueError(f"ZIP is missing required modules: {sorted(required - set(names))}")
        if set(names) != members:
            raise ValueError("ZIP member list differs from the selected source files")
        for name, digest in expected.items():
            if sha(z.read("uav_thesis/" + name)) != digest:
                raise ValueError(f"ZIP member hash mismatch: {name}")
    return dict(members=len(expected), envs_files=len(env_files), crc="passed", member_hashes="passed")


IMPORT_CHECK = r'''
import importlib, json, pathlib, sys
root = pathlib.Path.cwd().resolve()
sys.path.insert(0, str(root))
from scripts import w00
w00.enable_windows()
for name in ("run", "scripts.training.run_bs5_fourarm400k",
             "scripts.evaluation.eval_w00_checkpoint"):
    importlib.import_module(name)
modules = {}
for name, module in tuple(sys.modules.items()):
    if name == "run" or name.startswith(("src.", "scripts.")):
        filename = getattr(module, "__file__", None)
        if filename:
            path = pathlib.Path(filename).resolve()
            assert path.is_relative_to(root), (name, str(path))
            modules[name] = path.relative_to(root).as_posix()
assert "src.envs.bs5_fourarm400k" in modules
assert "src.rl.bs5_fourarm400k_qmix" in modules
print(json.dumps(modules, indent=2))
'''


def verify_imports(archive, work):
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        # Only previously verified names are extracted into a fresh directory.
        for name in z.namelist():
            if not (work / name).resolve().is_relative_to(work):
                raise ValueError("ZIP path escapes the extraction directory")
        z.extractall(work)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-I", "-B", "-c", IMPORT_CHECK],
                            cwd=work / "uav_thesis", env=env, text=True,
                            encoding="utf-8", capture_output=True, check=True)
    return json.loads(result.stdout)


def export_source(root, output, check_dir):
    root, output = Path(root).resolve(), Path(output).resolve()
    selected = collect_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive mode preserves previous packages, even if an explicit name collides.
    hashes = {}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, path in selected.items():
            data = path.read_bytes()
            hashes[name] = sha(data)
            z.writestr("uav_thesis/" + name, data)
    receipt = verify_archive(output, root, hashes)
    modules = verify_imports(output, check_dir)
    receipt.update(archive=str(output), sha256=sha(output.read_bytes()), files_sha256=hashes,
                   extracted=str(Path(check_dir).resolve() / "uav_thesis"),
                   imported_modules=modules, interpreter=sys.executable)
    output.with_suffix(".verification.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def default_output():
    day = datetime.now(timezone(timedelta(hours=7))).date().isoformat()
    target = ROOT / f"archive/packages/uav_thesis_source_{day}.zip"
    suffix = 2
    while target.exists():
        target = ROOT / f"archive/packages/uav_thesis_source_{day}_{suffix}.zip"
        suffix += 1
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New ZIP filename; an existing file is never overwritten")
    parser.add_argument("--check-dir", type=Path, help="Fresh extraction directory retained for review")
    args = parser.parse_args(argv)
    check_dir = args.check_dir or ROOT / "runs/local" / ("source_import_" + uuid.uuid4().hex)
    check_dir = check_dir.resolve()
    if not check_dir.is_relative_to(ROOT / "runs") or check_dir == ROOT / "runs":
        parser.error("--check-dir must be a fresh directory below runs/")
    receipt = export_source(ROOT, args.output or default_output(), check_dir)
    print(json.dumps({k: receipt[k] for k in ("archive", "sha256", "members", "envs_files", "crc", "member_hashes", "extracted")}, indent=2))
    print(f"Imported {len(receipt['imported_modules'])} local modules from the extracted ZIP")


if __name__ == "__main__":
    main()
