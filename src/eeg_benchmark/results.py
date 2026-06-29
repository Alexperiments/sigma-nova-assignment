"""Result row construction and CSV persistence."""

import platform
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from eeg_benchmark.types import BenchmarkResult


def package_version(package_name: str) -> str:
    """Return an installed package version or unknown."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def runtime_metadata() -> dict[str, str]:
    """Return runtime versions captured with each benchmark row."""
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "braindecode_version": package_version("braindecode"),
        "moabb_version": package_version("moabb"),
    }


def result_row(
    result: BenchmarkResult,
    *,
    dataset: str,
    batch_size: int,
    epochs: int,
    lr: float,
    device: str,
    seed: int,
) -> dict[str, Any]:
    """Build one serializable benchmark result row."""
    return {
        "model": result.model_name,
        "model_repository": result.model_repository,
        "model_revision": result.model_revision,
        "dataset": dataset,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "device": device,
        "seed": seed,
        "embedding_shape": "x".join(str(size) for size in result.embedding_shape),
        "train_accuracy": result.train_accuracy,
        "test_accuracy": result.test_accuracy,
        "target_window_samples": result.target_window_samples,
        **runtime_metadata(),
    }


def save_results(rows: list[dict[str, Any]], output_dir: str | Path) -> Path:
    """Write benchmark result rows to a timestamped CSV file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"benchmark_results_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
