import torch
from braindecode.models import CBraMod
from torch import Tensor

from eeg_benchmark.models.base import ModelAdapter


class CBraModAdapter(ModelAdapter):
    REPOSITORY = "braindecode/cbramod-pretrained"

    @classmethod
    def load(cls, device: str = "cpu") -> "CBraModAdapter":
        # The checkpoint has no output size, so load the encoder without a task head.
        model = CBraMod.from_pretrained(
            cls.REPOSITORY,
            return_encoder_output=True,
        )
        return cls(model, device)

    @torch.inference_mode()
    def encode(
        self,
        inputs: Tensor,
        *,
        channel_names: list[str] | None = None,
    ) -> Tensor:
        # CBraMod does not use channel names in its forward pass.
        del channel_names
        output = self.model(inputs.to(self.device), return_features=True)
        # Pool channel and patch features into one 200-dimensional vector per trial.
        return output["features"].mean(dim=(1, 2))
