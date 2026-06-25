import torch
from torch import nn

from eeg_benchmark.models.base import ModelAdapter


class DummyAdapter(ModelAdapter):
    def encode(self, inputs, *, channel_names=None):
        return inputs.mean(dim=-1)


def test_model_adapter_initializes_model_state() -> None:
    model = nn.Linear(4, 2)
    adapter = DummyAdapter(model, device="cpu")

    assert adapter.device.type == "cpu"
    assert adapter.model is model
    assert adapter.model.training is False
    assert all(not param.requires_grad for param in adapter.model.parameters())


def test_model_adapter_encode_returns_expected_shape() -> None:
    model = nn.Identity()
    adapter = DummyAdapter(model)

    inputs = torch.randn(3, 4, 8)
    embeddings = adapter.encode(inputs)

    assert embeddings.shape == (3, 4)
