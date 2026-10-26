"""Imported training provenance remains distinct from the integrated checkout."""
import hashlib
import json
from unittest.mock import patch

import pytest

from scripts.evaluation import run_bs5_v4_benchmark as benchmark


def test_integrated_source_checks_science_and_preserves_training_identity(tmp_path):
    science = tmp_path / "src/science.py"
    science.parent.mkdir()
    science.write_text("reward = 1\n", encoding="utf-8")
    receipt = tmp_path / "configs/bs5_return_v4_cpu_import.json"
    receipt.parent.mkdir()
    data = dict(original_source_sha256="training-source", science_files={
        "src/science.py": hashlib.sha256(science.read_bytes()).hexdigest(),
    })
    receipt.write_text(json.dumps(data), encoding="utf-8")
    with patch.object(benchmark, "ROOT", tmp_path), patch.object(
        benchmark, "source_identity", return_value=dict(sha256="integrated-source")
    ):
        result = benchmark.benchmark_source(dict(source_sha256="training-source"))
        assert result["source_sha256"] == "integrated-source"
        assert result["training_source_sha256"] == "training-source"
        assert result["source_import_receipt_sha256"] == benchmark.file_hash(receipt)
        with pytest.raises(AssertionError, match="Different training source"):
            benchmark.benchmark_source(dict(source_sha256="unrelated-training"))
        science.write_text("reward = 2\n", encoding="utf-8")
        with pytest.raises(AssertionError, match="Scientific source changed"):
            benchmark.benchmark_source(dict(source_sha256="training-source"))


def test_matching_source_needs_no_import_receipt():
    with patch.object(benchmark, "source_identity", return_value=dict(sha256="same")):
        assert benchmark.benchmark_source(dict(source_sha256="same")) == dict(
            source_sha256="same", training_source_sha256="same"
        )
