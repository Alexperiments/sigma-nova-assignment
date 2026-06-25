from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class ModelAdapter(ABC):
    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()

    @abstractmethod
    def encode(
        self,
        inputs: Tensor,
        *,
        channel_names: list[str] | None = None,
    ) -> Tensor:
        """Return one embedding per input trial."""
