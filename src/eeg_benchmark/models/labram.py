import torch
from braindecode.models import Labram
from torch import Tensor

from eeg_benchmark.models.base import ModelAdapter


class LaBraMAdapter(ModelAdapter):
    REPOSITORY = "braindecode/labram-pretrained"

    @classmethod
    def load(cls, device: str = "cpu") -> "LaBraMAdapter":
        # This checkpoint has no classification head (n_outputs=0).
        model = Labram.from_pretrained(cls.REPOSITORY)
        return cls(model, device)

    @torch.inference_mode()
    def encode(
        self,
        inputs: Tensor,
        *,
        channel_names: list[str] | None = None,
    ) -> Tensor:
        # LaBraM uses channel names to select the matching positional embeddings.
        output = self.model(
            inputs.to(self.device),
            return_features=True,
            ch_names=channel_names,
        )
        # The checkpoint exposes one pooled 200-dimensional representation per trial.
        return output["cls_token"]
