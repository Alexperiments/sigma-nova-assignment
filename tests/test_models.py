import pytest
import torch
from torch import Tensor, nn

from eeg_benchmark import models
from eeg_benchmark.models import (
    CBraModFoundationModel,
    FrozenFoundationModel,
    LaBraMFoundationModel,
)


class FakeFoundationAdapter(FrozenFoundationModel):
    def encode(
        self,
        inputs: Tensor,
        *,
        channel_names: list[str] | None = None,
    ) -> Tensor:
        del channel_names
        return inputs


class FakeBackbone(nn.Module):
    def __init__(
        self,
        *,
        n_times=None,
        ch_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.n_times = n_times
        self.ch_names = ch_names


def test_frozen_foundation_model_moves_freezes_and_sets_eval_mode() -> None:
    backbone = FakeBackbone()

    adapter = FakeFoundationAdapter(backbone, device="cpu")

    assert adapter.device == torch.device("cpu")
    assert adapter.model is backbone
    assert adapter.model.training is False
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_input_window_samples_reads_braindecode_n_times() -> None:
    adapter = FakeFoundationAdapter(FakeBackbone(n_times="3000"))

    assert adapter.input_window_samples == 3000


def test_input_window_samples_returns_none_when_n_times_is_unavailable() -> None:
    class RaisesOnNTimes(nn.Module):
        @property
        def n_times(self):
            raise ValueError("n_times is not defined")

    adapter = FakeFoundationAdapter(RaisesOnNTimes())

    assert adapter.input_window_samples is None


def test_expected_channel_names_reads_model_channel_names() -> None:
    adapter = FakeFoundationAdapter(FakeBackbone(ch_names=["C3", "C4"]))

    assert adapter.expected_channel_names == ["C3", "C4"]


def test_cbramod_load_uses_encoder_checkpoint(monkeypatch) -> None:
    calls = {}
    backbone = FakeBackbone()

    def fake_from_pretrained(
        repository: str,
        *,
        revision: str,
        return_encoder_output: bool,
    ):
        calls["repository"] = repository
        calls["revision"] = revision
        calls["return_encoder_output"] = return_encoder_output
        return backbone

    monkeypatch.setattr(models.CBraMod, "from_pretrained", fake_from_pretrained)

    adapter = CBraModFoundationModel.load(device="cpu")

    assert adapter.model is backbone
    assert adapter.repository == "braindecode/cbramod-pretrained"
    assert adapter.revision == "584cdc415913739a05d84bf0c1cb3db397764507"
    assert calls == {
        "repository": "braindecode/cbramod-pretrained",
        "revision": "584cdc415913739a05d84bf0c1cb3db397764507",
        "return_encoder_output": True,
    }


def test_cbramod_encode_pools_channel_and_patch_features() -> None:
    class FakeCBraModBackbone(FakeBackbone):
        def forward(self, inputs, *, return_features: bool):
            assert return_features is True
            features = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(
                2,
                3,
                4,
                5,
            )
            return {"features": features.to(inputs.device)}

    adapter = CBraModFoundationModel(FakeCBraModBackbone(), device="cpu")
    result = adapter.encode(torch.ones(2, 3, 10), channel_names=["ignored"])

    expected = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(
        2,
        3,
        4,
        5,
    ).mean(dim=(1, 2))
    torch.testing.assert_close(result, expected)


def test_labram_load_uses_pretrained_checkpoint(monkeypatch) -> None:
    calls = {}
    backbone = FakeBackbone()

    def fake_from_pretrained(repository: str, *, revision: str):
        calls["repository"] = repository
        calls["revision"] = revision
        return backbone

    monkeypatch.setattr(models.Labram, "from_pretrained", fake_from_pretrained)

    adapter = LaBraMFoundationModel.load(device="cpu")

    assert adapter.model is backbone
    assert adapter.repository == "braindecode/labram-pretrained"
    assert adapter.revision == "0563b6c626e7b40d9a36653b763715db94d945d7"
    assert calls == {
        "repository": "braindecode/labram-pretrained",
        "revision": "0563b6c626e7b40d9a36653b763715db94d945d7",
    }


def test_labram_encode_requires_channel_names() -> None:
    adapter = LaBraMFoundationModel(FakeBackbone(), device="cpu")

    with pytest.raises(ValueError, match="requires channel names"):
        adapter.encode(torch.ones(2, 3, 10))


def test_labram_encode_passes_channel_names_and_returns_cls_token() -> None:
    class FakeLaBraMBackbone(FakeBackbone):
        def __init__(self) -> None:
            super().__init__()
            self.seen_channel_names = None

        def forward(self, inputs, *, return_features: bool, ch_names):
            assert return_features is True
            self.seen_channel_names = ch_names
            return {
                "cls_token": torch.full(
                    (inputs.shape[0], 5),
                    2.0,
                    device=inputs.device,
                )
            }

    backbone = FakeLaBraMBackbone()
    adapter = LaBraMFoundationModel(backbone, device="cpu")

    result = adapter.encode(
        torch.ones(2, 3, 10),
        channel_names=["C3", "Cz", "C4"],
    )

    assert backbone.seen_channel_names == ["C3", "Cz", "C4"]
    torch.testing.assert_close(result, torch.full((2, 5), 2.0))
