"""Validate local delivery tables and reject corruption without any rollout."""
from pathlib import Path
import shutil

import pandas as pd
import pytest

from scripts.verification.check_final400k import check_tables

TABLES = Path(__file__).resolve().parents[1] / "results/final400k/benchmark"
pytestmark = pytest.mark.skipif(not (TABLES / "per_run.csv").exists(),
                                reason="Restore the ignored final benchmark tables first")


def test_delivered_crossed_design_and_table_consistency():
    summary, _ = check_tables(TABLES)
    assert summary["runs"] == 150
    assert summary["slots"] == 1500000


@pytest.mark.parametrize("corruption", ["duplicate", "unpaired_trace", "wrong_aggregate"])
def test_delivery_rejects_corruption(tmp_path, corruption):
    for path in TABLES.iterdir():
        if path.suffix in (".csv", ".json"):
            shutil.copy2(path, tmp_path / path.name)
    name = "aggregate.csv" if corruption == "wrong_aggregate" else "per_run.csv"
    frame = pd.read_csv(tmp_path / name)
    if corruption == "duplicate":
        frame.iloc[1] = frame.iloc[0]
    elif corruption == "unpaired_trace":
        frame.loc[0, "ue_trace_sha256"] = "0" * 64
    else:
        frame.loc[0, "throughput_mbps"] += 10
    frame.to_csv(tmp_path / name, index=False)
    with pytest.raises(ValueError):
        check_tables(tmp_path)


def test_rebuild_adapter_writes_the_plot_input_schema(tmp_path, monkeypatch):
    # Exercise the adapter's complete branch using delivered tables as an IO
    # fixture. The full raw rebuild is validated separately by the delivery audit.
    from scripts.analysis import report_final400k as adapter

    def audited_report(input_dir, output, **kwargs):
        output.mkdir()
        for name in ("per_run.csv", "aggregate.csv"):
            shutil.copy2(TABLES / name, output / name)
        return dict(status="complete", expected_runs=150, completed_runs=150, missing=[])

    monkeypatch.setattr(adapter, "report", audited_report)
    monkeypatch.setattr(adapter, "temporal_table", lambda *_: pd.read_csv(TABLES / "temporal_1000slot.csv"))
    output = tmp_path / "rebuilt"
    adapter.rebuild("explicit-io-fixture", output)
    summary, _ = check_tables(output)
    assert summary["runs"] == 150


def test_partial_rebuild_removes_stale_final_tables(tmp_path, monkeypatch):
    from scripts.analysis import report_final400k as adapter

    monkeypatch.setattr(adapter, "report", lambda *_, **__: dict(status="partial"))
    names = ["audit.json", "temporal_1000slot.csv", "per_test_seed.csv", "per_training_seed.csv"]
    for name in names:
        (tmp_path / name).write_text("stale output")
    assert adapter.rebuild("explicit-partial-fixture", tmp_path)["status"] == "partial"
    assert not any((tmp_path / name).exists() for name in names)
