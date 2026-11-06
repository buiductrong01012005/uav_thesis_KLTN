"""Locate local benchmark inputs without rewriting their frozen metadata."""

from pathlib import Path, PurePosixPath


def benchmark_asset_dir(root: Path, name: str) -> Path:
    """Prefer results/ while accepting the original extracted Dataset layout."""
    current = root / "results" / "benchmark_assets" / name
    original = root / "benchmark_assets" / name
    return original if not current.exists() and original.is_dir() else current


def benchmark_asset_path(directory: Path, original_path: str) -> Path:
    """Resolve a manifest path inside its selected benchmark input directory."""
    source = PurePosixPath(original_path)
    if source.is_absolute() or ".." in source.parts or "\\" in original_path or ":" in original_path:
        raise ValueError(f"Invalid benchmark asset path: {original_path}")
    prefix = PurePosixPath("benchmark_assets") / directory.name
    relative = source.relative_to(prefix)
    target = (directory / relative).resolve()
    if not target.is_relative_to(directory.resolve()):
        raise ValueError(f"Benchmark asset path escapes its directory: {original_path}")
    return target
