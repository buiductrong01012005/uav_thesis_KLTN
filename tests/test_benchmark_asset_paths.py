from pathlib import Path

import pytest

from scripts.evaluation.benchmark_asset_paths import benchmark_asset_dir, benchmark_asset_path


def test_results_layout_takes_priority_without_changing_manifest_paths(tmp_path):
    original = tmp_path / "benchmark_assets" / "fair300k"
    current = tmp_path / "results" / "benchmark_assets" / "fair300k"
    original.mkdir(parents=True)
    current.mkdir(parents=True)
    selected = benchmark_asset_dir(tmp_path, "fair300k")
    assert selected == current
    manifest_path = "benchmark_assets/fair300k/EMA/best.pt"
    assert benchmark_asset_path(selected, manifest_path) == (current / "EMA/best.pt").resolve()


def test_original_dataset_layout_still_works(tmp_path):
    original = tmp_path / "benchmark_assets" / "fair300k"
    original.mkdir(parents=True)
    assert benchmark_asset_dir(tmp_path, "fair300k") == original


def test_missing_assets_point_to_the_documented_results_layout(tmp_path):
    assert benchmark_asset_dir(tmp_path, "fair300k") == tmp_path / "results/benchmark_assets/fair300k"


@pytest.mark.parametrize("path", [
    "../model.pt",
    "/benchmark_assets/fair300k/EMA/best.pt",
    "benchmark_assets/fair300k/EMA/../../other.pt",
    "benchmark_assets/other/EMA/best.pt",
    "benchmark_assets\\fair300k\\EMA\\best.pt",
    "C:/benchmark_assets/fair300k/EMA/best.pt",
])
def test_manifest_cannot_read_outside_its_benchmark(path):
    with pytest.raises(ValueError):
        benchmark_asset_path(Path("results/benchmark_assets/fair300k"), path)
