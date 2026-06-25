import unittest
from pathlib import Path

from eeg_benchmark.datasets import BNCI2014_001Dataset, DatasetConfig


class DatasetTests(unittest.TestCase):
    def test_evaluate_returns_expected_metrics(self) -> None:
        config = DatasetConfig(data_dir=Path("data/benchmarks/BCICIV_2a_gdf"), download=False)
        dataset = BNCI2014_001Dataset(config)

        metrics = dataset.evaluate([0, 1, 0, 1], [0, 1, 0, 1])

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["kappa"], 1.0)


if __name__ == "__main__":
    unittest.main()
