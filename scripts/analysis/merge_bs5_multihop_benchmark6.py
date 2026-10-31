"""Merge per-algorithm or per-seed REVIEW archives without mixing experiments."""

from pathlib import Path
import argparse
import json
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.report_bs5_multihop_benchmark6 import generate


def merge(archives, output, phase, allow_diagnostic=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    choices = {}
    protocols = {}
    fingerprints = set()
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                q = Path(info.filename)
                if q.is_absolute() or ".." in q.parts or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("Unsafe archive member")
            for name in z.namelist():
                path = Path(name)
                if (
                    path.name != "summary.json"
                    or len(path.parts) != 3
                    or not path.parts[1].startswith("seed")
                ):
                    continue
                label = path.parts[0]
                p = json.loads(z.read(label + "/protocol.json"))
                if p["phase"] != phase:
                    continue
                if p["diagnostic"] and not allow_diagnostic:
                    raise ValueError("Diagnostic results are not a production benchmark")
                if label in protocols and protocols[label] != p:
                    raise ValueError("Conflicting label protocol")
                protocols[label] = p
                fingerprints.add(p["experiment_fingerprint"])
                s = json.loads(z.read(name))
                key = (label, s["seed"])
                if key in choices:
                    old = choices[key][0]
                    if s["slots"] == old["slots"] and (
                        s["ue_trace_sha256"] != old["ue_trace_sha256"]
                        or abs(s["throughput_mbps"] - old["throughput_mbps"]) > 1e-7
                    ):
                        raise ValueError("Conflicting duplicate seed result")
                    if s["slots"] <= old["slots"]:
                        continue
                choices[key] = (s, Path(archive), path.parent.as_posix() + "/")
    if len(fingerprints) != 1:
        raise ValueError("Need exactly one matching experiment identity")
    if not choices:
        raise ValueError("No results for selected phase")
    for label, p in protocols.items():
        (output / label).mkdir(exist_ok=True)
        (output / label / "protocol.json").write_text(json.dumps(p, indent=2) + "\n")
    for s, archive, prefix in choices.values():
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if (
                    name.startswith(prefix)
                    and not name.endswith("/")
                    and Path(name).name != "resume.pkl"
                ):
                    path = output / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(z.read(name))
    report = generate(output)
    dest = output.parent / ("MH_" + phase.upper() + "_COMPARISON.zip")
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(output))
    with zipfile.ZipFile(dest) as z:
        if z.testzip():
            raise ValueError("Invalid merged archive")
    print(report, "\n", dest, flush=True)
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--phase", choices=["pilot", "test"], default="pilot")
    p.add_argument("--allow-diagnostic", action="store_true")
    a = p.parse_args()
    archives = [p for b in a.input for p in ([b] if b.is_file() else b.rglob("*_BENCH_REVIEW.zip"))]
    merge(archives, a.output, a.phase, a.allow_diagnostic)
