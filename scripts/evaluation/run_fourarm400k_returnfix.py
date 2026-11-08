"""Run a frozen benchmark with the post-simulation Returning summary fix."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import run_fourarm400k_heuristics as core
from scripts.evaluation.fourarm400k_return_summary import extend_summary

ENTRYPOINTS = {
    "drl": "run_fourarm400k_drl_benchmark.py",
    "baseline": "run_fourarm400k_heuristics.py",
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ENTRYPOINTS:
        raise SystemExit("Usage: run_fourarm400k_returnfix.py {drl|baseline} [benchmark arguments]")
    mode = sys.argv.pop(1)
    core.extend_summary = extend_summary
    entry = Path(__file__).with_name(ENTRYPOINTS[mode])
    if mode == "baseline":
        # Invoke the already patched module, so the baseline CLI uses the
        # same function. DRL uses runpy to preserve FrozenActor.__main__ for
        # snapshots written by the original CLI.
        import signal

        args = core.parser().parse_args()
        if args.save_every < 1 or args.log_every < 1:
            raise ValueError("Positive log/save intervals required")
        signal.signal(signal.SIGTERM, core.stop_handler)
        signal.signal(signal.SIGINT, core.stop_handler)
        core.run(args)
    else:
        runpy.run_path(str(entry), run_name="__main__")
