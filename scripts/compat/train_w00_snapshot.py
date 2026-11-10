"""Isolated native launcher; the archived ablation source remains byte-identical."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--variant", choices=("mlp", "no_curriculum"), required=True)
    args, remaining = parser.parse_known_args()
    sys.path.insert(0, str(args.snapshot))
    if sys.platform == "win32":
        sys.path.insert(0, str(ROOT / "scripts/compat/windows"))
    path = args.snapshot / "scripts/training/run_bs5_fourarm400k.py"
    spec = importlib.util.spec_from_file_location("w00_ablation_training", path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    source_lock = runner.source_lock
    runner.source_lock = lambda: {k.replace("\\", "/"): v for k, v in source_lock().items()}
    original = runner.load_protocol

    def protocol(options):
        p = original(options)
        p["native_adapter"] = dict(version="w00_ablation_native_v1",
                                   sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        common = {k: v for k, v in p.items() if k not in
                  {"arm", "policy_actions", "runtime", "torch_version", "comparison_fingerprint"}}
        p["comparison_fingerprint"] = runner.digest(common)
        return p

    runner.load_protocol = protocol
    options = runner.parser().parse_args(remaining + (["--encoder", "mlp"] if args.variant == "mlp" else ["--no-curriculum"]))
    options.output = options.output.resolve()
    if options.arm != "C-R" or not options.output.is_relative_to((ROOT / "runs/local").resolve()):
        raise ValueError("W00 component ablations require C-R and output under runs/local/")
    signal.signal(signal.SIGTERM, runner.stop_handler)
    signal.signal(signal.SIGINT, runner.stop_handler)
    runner.run(options)


if __name__ == "__main__":
    main()
