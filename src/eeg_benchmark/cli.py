"""Command-line interface for EEG benchmark runs."""

import argparse
import logging

from eeg_benchmark.datasets import DATASETS
from eeg_benchmark.models import MODELS
from eeg_benchmark.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    """Parse benchmark command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="BNCI2014_001")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> None:
    """Run the benchmark command-line entry point."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_pipeline(
        model_name_list=args.models,
        dataset_name_list=[args.dataset],
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
