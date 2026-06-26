import torch
from torch import Tensor
from torch.utils.data import DataLoader

from eeg_benchmark.adapters import adapt_windows_to_model
from eeg_benchmark.datasets import MOABBDatasetWrapper, split_train_test
from eeg_benchmark.models import FrozenFoundationModel, MODELS
from eeg_benchmark.probe import evaluate_linear_probe
from eeg_benchmark.types import BenchmarkResult


def encode_dataset(
    dataset: MOABBDatasetWrapper,
    model: FrozenFoundationModel,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    labels = []

    for inputs, batch_labels, _ in loader:
        embeddings.append(
            model.encode(inputs.float(), channel_names=dataset.channel_names).cpu()
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
    
    input_event_windows = dataset.create_event_windows()
    adapted_windows, target_window_samples, time_adjustment = adapt_windows_to_model(input_event_windows, model)
    train_windows, test_windows = split_train_test(adapted_windows)

    train_embeddings, train_labels = encode_dataset(
        train_windows,
        model,
        batch_size,
    )
    test_embeddings, test_labels = encode_dataset(
        test_windows,
        model,
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
        embedding_shape=tuple(train_embeddings.shape),
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        target_window_samples=target_window_samples,
        time_adjustment=time_adjustment,
    )
