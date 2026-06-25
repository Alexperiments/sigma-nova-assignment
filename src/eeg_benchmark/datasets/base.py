from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass
class DatasetConfig:
    data_dir: Path
    download: bool = True
    metrics: tuple[str, ...] = ("accuracy", "kappa")


class BaseDataset(ABC):
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def load(self) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def preprocess(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        return inputs, targets

    def evaluate(
        self,
        predictions: Tensor | list[int],
        targets: Tensor | list[int],
    ) -> dict[str, float]:
        preds = torch.as_tensor(predictions, dtype=torch.long).reshape(-1)
        targs = torch.as_tensor(targets, dtype=torch.long).reshape(-1)

        if preds.numel() != targs.numel():
            raise ValueError("Predictions and targets must have the same number of elements.")

        results: dict[str, float] = {}

        if "accuracy" in self.config.metrics:
            results["accuracy"] = float((preds == targs).float().mean().item())

        if "kappa" in self.config.metrics:
            observed = (preds == targs).float().mean().item()
            labels = torch.unique(torch.cat([preds, targs]))
            expected = 0.0
            for label in labels:
                pred_count = (preds == label).float().sum().item()
                target_count = (targs == label).float().sum().item()
                expected += (pred_count * target_count) / (preds.numel() ** 2)
            results["kappa"] = float((observed - expected) / (1.0 - expected)) if expected < 1.0 else 0.0

        return results
