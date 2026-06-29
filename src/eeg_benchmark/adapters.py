import numpy as np
from braindecode.datasets.base import BaseConcatDataset

from eeg_benchmark.models import FrozenFoundationModel


def pad_or_crop_sample(x: np.ndarray, target_window_samples: int) -> np.ndarray:
    """Force an EEG window to the model's expected sample length.

    Braindecode windows keep time on the last axis. Longer windows are
    truncated from the end; shorter windows are right-padded with zeros so the
    event-locked start stays aligned.
    """
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
) -> BaseConcatDataset:
    """Install lazy window adaptation for models with fixed input duration.

    Existing dataset transforms are preserved and run before the crop/pad
    compatibility step. Variable-length models leave windows unchanged.
    """
    target_window_samples = model.input_window_samples
    if target_window_samples is None:
        return windows

    target_window_samples = int(target_window_samples)

    for window_dataset in windows.datasets:
        previous_transform = window_dataset.transform
        if previous_transform is None:
            window_dataset.transform = lambda x: pad_or_crop_sample(x, target_window_samples)
        else:
            window_dataset.transform = lambda x, transform=previous_transform: (
                pad_or_crop_sample(transform(x), target_window_samples)
            )

    return windows
