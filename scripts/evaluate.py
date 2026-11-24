"""Evaluate a single checkpoint using its saved environment and architecture."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [str(ROOT)] + [
    entry for entry in sys.path
    if Path(entry or ".").resolve() not in {ROOT, ROOT / "scripts"}
]
sys.dont_write_bytecode = True


def main(argv=None):
    from scripts.w00 import enable_windows

    enable_windows()
    from scripts.evaluation.eval_w00_checkpoint import main as evaluate

    return evaluate(argv)


if __name__ == "__main__":
    main()
