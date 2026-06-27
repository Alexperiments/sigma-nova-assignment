from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from braindecode.models import CBraMod, Labram


class FrozenFoundationModel(ABC):
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        *,
        repository: str = "",
        revision: str = "",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.repository = repository
        self.revision = revision

    @property
    def input_window_samples(self) -> int | None:
        try:
            value = getattr(self.model, "n_times", None)
        except ValueError:
            return None
        return int(value) if value is not None else None
    
    @property
    def expected_channel_names(self) -> list[str] | None:
        return getattr(self.model, "ch_names", None)

    @abstractmethod
    def encode(
        self,
        inputs: Tensor,
        *,
        channel_names: list[str] | None = None,
    ) -> Tensor:
        """Return one embedding per input trial."""


class CBraModFoundationModel(FrozenFoundationModel):
    REPOSITORY = "braindecode/cbramod-pretrained"
    REVISION = "584cdc415913739a05d84bf0c1cb3db397764507"

    @classmethod
    def load(cls, device: str = "cpu") -> "CBraModFoundationModel":
        # The checkpoint has no output size, so load the encoder without a task head.
        model = CBraMod.from_pretrained(
            cls.REPOSITORY,
            revision=cls.REVISION,
            return_encoder_output=True,
        )
        return cls(
            model,
            device,
            repository=cls.REPOSITORY,
            revision=cls.REVISION,
        )

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



class LaBraMFoundationModel(FrozenFoundationModel):
    REPOSITORY = "braindecode/labram-pretrained"
    REVISION = "0563b6c626e7b40d9a36653b763715db94d945d7"

    @classmethod
    def load(cls, device: str = "cpu") -> "LaBraMFoundationModel":
        # This checkpoint has no classification head (n_outputs=0).
        model = Labram.from_pretrained(cls.REPOSITORY, revision=cls.REVISION)
        return cls(
            model,
            device,
            repository=cls.REPOSITORY,
            revision=cls.REVISION,
        )

    @torch.inference_mode()
    def encode(
        self,
        inputs: Tensor,
        *,
        channel_names: list[str] | None = None,
    ) -> Tensor:
        # LaBraM uses channel names to select the matching positional embeddings.

        if channel_names is None:
            raise ValueError("LaBraM requires channel names for encoding.")

        output = self.model(
            inputs.to(self.device),
            return_features=True,
            ch_names=channel_names,
        )
        # The checkpoint exposes one pooled 200-dimensional representation per trial.
        return output["cls_token"]


MODELS = {
    "labram": LaBraMFoundationModel,
    "cbramod": CBraModFoundationModel,
}
