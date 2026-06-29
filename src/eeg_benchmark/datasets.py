"""Dataset loading and windowing helpers."""

from functools import cached_property
from typing import Self

import numpy as np
from braindecode.datasets import MOABBDataset
from braindecode.datasets.base import BaseConcatDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)


MICROVOLTS_PER_VOLT = 1_000_000.0
EXTREME_AMPLITUDE_MICROVOLTS = 500.0


def volts_to_microvolts(x: np.ndarray) -> np.ndarray:
    """Convert EEG amplitudes from volts to microvolts."""
    return x * MICROVOLTS_PER_VOLT


def clip_extreme_microvolts(x: np.ndarray) -> np.ndarray:
    """Clip implausible EEG amplitudes in microvolts."""
    return np.clip(
        x,
        -EXTREME_AMPLITUDE_MICROVOLTS,
        EXTREME_AMPLITUDE_MICROVOLTS,
    )


class MOABBDatasetWrapper:
    """Wrap a Braindecode MOABB dataset with inferred metadata."""

    def __init__(self, dataset_name: str, dataset: BaseConcatDataset) -> None:
        """Create a wrapper around a loaded MOABB dataset."""
        self.dataset_name = dataset_name
        self.dataset = dataset

    @classmethod
    def load(cls, dataset_name: str) -> Self:
        """Load a MOABB dataset with minimal EEG preprocessing."""
        dataset = MOABBDataset(dataset_name=dataset_name)
        preprocess(
            dataset,
            [
                Preprocessor("pick", picks="eeg", apply_on_array=False),
                Preprocessor(volts_to_microvolts, apply_on_array=True),
                Preprocessor(clip_extreme_microvolts, apply_on_array=True),
            ],
        )

        return cls(dataset_name=dataset_name, dataset=dataset)

    @cached_property
    def channel_names(self) -> list[str]:
        """Return the EEG channel names from the first raw recording."""
        return list(self.dataset.datasets[0].raw.ch_names)

    @cached_property
    def event_mapping(self) -> dict[str, int]:
        """Return a deterministic event-name to class-index mapping."""
        event_names = sorted(
            {
                str(description)
                for raw_dataset in self.dataset.datasets
                for description in raw_dataset.raw.annotations.description
            }
        )
        return {event_name: index for index, event_name in enumerate(event_names)}

    def create_event_windows(self) -> BaseConcatDataset:
        """Create event-locked windows using each dataset event duration."""
        return create_windows_from_events(
            self.dataset,
            mapping=self.event_mapping,
            window_size_samples=None,
            window_stride_samples=None,
            drop_last_window=True,
            preload=False,
        )


def split_train_test(
    windows: BaseConcatDataset,
) -> tuple[BaseConcatDataset, BaseConcatDataset]:
    """Split MOABB windows into train and test sessions."""
    splits = windows.split("session")
    return splits["0train"], splits["1test"]


DATASETS = ["BNCI2014_001"]
