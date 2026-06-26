from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from eeg_benchmark.types import BenchmarkResult


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
    return {
        "model": result.model,
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
        "time_adjustment": result.time_adjustment,
    }


def save_results(rows: list[dict[str, Any]], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"benchmark_results_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
