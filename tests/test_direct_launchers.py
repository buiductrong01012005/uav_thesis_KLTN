"""Direct launchers must not shadow third-party packaging with the ZIP exporter."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("launcher", ["train.py", "evaluate.py"])
def test_direct_launcher_keeps_third_party_packaging(launcher):
    probe = (
        "import sys, runpy; sys.path.insert(0, 'scripts'); "
        f"runpy.run_path('scripts/{launcher}', run_name='launcher_probe'); "
        "import packaging.version, matplotlib; "
        "assert 'scripts/packaging' not in packaging.version.__file__.replace(chr(92), '/')"
    )
    subprocess.run([sys.executable, "-B", "-c", probe], cwd=ROOT, check=True,
                   capture_output=True, text=True)
