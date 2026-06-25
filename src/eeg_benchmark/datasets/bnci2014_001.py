from torch import Tensor

from .base import BaseDataset, DatasetConfig


class BNCI2014_001Dataset(BaseDataset):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)

    def load(self) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("Loading logic will be implemented once the dataset format is finalized.")
