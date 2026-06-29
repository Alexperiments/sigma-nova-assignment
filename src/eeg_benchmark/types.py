"""Shared benchmark data structures."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkResult:
    """Store metrics and metadata for one benchmark run."""

    model_name: str
    model_repository: str
    model_revision: str
    embedding_shape: tuple[int, ...]
    train_accuracy: float
    test_accuracy: float
    target_window_samples: int | None
