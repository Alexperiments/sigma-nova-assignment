import logging
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from eeg_benchmark.adapters import adapt_windows_to_model
from eeg_benchmark.datasets import MOABBDatasetWrapper, split_train_test
from eeg_benchmark.models import FrozenFoundationModel, MODELS
from eeg_benchmark.probe import evaluate_linear_probe
from eeg_benchmark.results import result_row, save_results
from eeg_benchmark.types import BenchmarkResult

LOGGER = logging.getLogger(__name__)


def encode_dataset(
    dataset,
    model: FrozenFoundationModel,
    channel_names: list[str],
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    labels = []

    for inputs, batch_labels, _ in loader:
        embeddings.append(
            model.encode(inputs.float(), channel_names=channel_names).cpu()
        )
        labels.append(batch_labels.long())

    return torch.cat(embeddings), torch.cat(labels)


def run_model(
    model_name: str,
    dataset_name: str,
    *,
    batch_size: int,
    epochs: int,
    lr: float,
    device: str = "cpu",
) -> BenchmarkResult:
    model = MODELS[model_name].load(device=device)
    dataset = MOABBDatasetWrapper.load(dataset_name)

    windows = dataset.create_event_windows()
    target_window_samples = model.input_window_samples
    windows = adapt_windows_to_model(windows, model)
    train_windows, test_windows = split_train_test(windows)

    train_embeddings, train_labels = encode_dataset(
        train_windows,
        model,
        dataset.channel_names,
        batch_size,
    )
    test_embeddings, test_labels = encode_dataset(
        test_windows,
        model,
        dataset.channel_names,
        batch_size,
    )
    train_accuracy, test_accuracy = evaluate_linear_probe(
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels,
        n_classes=len(dataset.event_mapping),
        device=model.device,
        epochs=epochs,
        lr=lr,
    )

    return BenchmarkResult(
        model_name=model_name,
        model_repository=model.repository,
        model_revision=model.revision,
        embedding_shape=tuple(train_embeddings.shape),
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        target_window_samples=target_window_samples,
    )


def run_pipeline(
    model_name_list: list[str],
    dataset_name_list: list[str],
    seed: int,
    batch_size: int = 8,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    output_dir: str | Path = "results",
) -> None:
    torch.manual_seed(seed)

    result_rows = []
    for model_name in model_name_list:
        for dataset_name in dataset_name_list:
            result = run_model(
                model_name,
                dataset_name,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                device=device,
            )

            result_rows.append(
                result_row(
                    result,
                    dataset=dataset_name,
                    batch_size=batch_size,
                    epochs=epochs,
                    lr=lr,
                    device=device,
                    seed=seed,
                )
            )

    path = save_results(result_rows, output_dir)
    LOGGER.info(f"saved: {path}")
