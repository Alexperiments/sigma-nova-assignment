from eeg_benchmark.results import result_row
from eeg_benchmark.types import BenchmarkResult


def test_result_row_includes_adaptation_metadata() -> None:
    result = BenchmarkResult(
        model_name="labram",
        embedding_shape=(10, 200),
        train_accuracy=0.8,
        test_accuracy=0.5,
        target_window_samples=3000,
    )

    row = result_row(
        result,
        dataset="BNCI2014_001",
        batch_size=8,
        epochs=20,
        lr=1e-3,
        device="cpu",
        seed=0,
    )

    assert row["embedding_shape"] == "10x200"
    assert row["target_window_samples"] == 3000
