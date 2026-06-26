from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    embedding_shape: tuple[int, ...]
    train_accuracy: float
    test_accuracy: float
    target_window_samples: int | None
    time_adjustment: str
