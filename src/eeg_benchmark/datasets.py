from functools import cached_property
from typing import Self

from braindecode.datasets import MOABBDataset
from braindecode.datasets.base import BaseConcatDataset
from braindecode.preprocessing import Preprocessor, create_windows_from_events, preprocess


class MOABBDatasetWrapper:
    def __init__(self, dataset_name: str, dataset: BaseConcatDataset) -> None:
        self.dataset_name = dataset_name
        self.dataset = dataset

    @classmethod
    def load(cls, dataset_name: str) -> Self:
        """Load a MOABB dataset and apply preprocessing to pick EEG channels."""
        dataset = MOABBDataset(dataset_name=dataset_name)
        preprocess(
            dataset,
            [Preprocessor("pick", picks="eeg", apply_on_array=False)],
        )

        return cls(dataset_name=dataset_name, dataset=dataset)

    @cached_property
    def channel_names(self) -> list[str]:
        return list(self.dataset.datasets[0].raw.ch_names)

    @cached_property
    def event_mapping(self) -> dict[str, int]:
        event_names = sorted(
            {
                str(description)
                for raw_dataset in self.dataset.datasets
                for description in raw_dataset.raw.annotations.description
            }
        )
        return {event_name: index for index, event_name in enumerate(event_names)}


    def create_event_windows(self) -> BaseConcatDataset:
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
    splits = windows.split("session")
    return splits["0train"], splits["1test"]


DATASETS = ["BNCI2014_001"]
