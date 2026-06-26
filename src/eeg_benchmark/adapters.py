import numpy as np
from braindecode.datasets.base import BaseConcatDataset

from eeg_benchmark.models import FrozenFoundationModel


def pad_or_crop_time(x: np.ndarray, target_window_samples: int) -> np.ndarray:
    n_times = x.shape[-1]
    if n_times > target_window_samples:
        return x[..., :target_window_samples]
    if n_times < target_window_samples:
        pad_width = [(0, 0)] * x.ndim
        pad_width[-1] = (0, target_window_samples - n_times)
        return np.pad(x, pad_width, mode="constant")
    return x


def adapt_windows_to_model(
    windows: BaseConcatDataset,
    model: FrozenFoundationModel,
) -> tuple[BaseConcatDataset, int | None, str]:
    target_window_samples = model.input_window_samples
    if target_window_samples is None:
        return windows, None, "none"

    target_window_samples = int(target_window_samples)

    for window_dataset in windows.datasets:
        previous_transform = window_dataset.transform
        if previous_transform is None:
            window_dataset.transform = lambda x: pad_or_crop_time(x, target_window_samples)
        else:
            window_dataset.transform = lambda x, transform=previous_transform: (
                pad_or_crop_time(transform(x), target_window_samples)
            )

    return windows, target_window_samples, "pad_or_crop"
