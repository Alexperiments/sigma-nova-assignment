from types import SimpleNamespace

import numpy as np

from eeg_benchmark import datasets
from eeg_benchmark.datasets import (
    MOABBDatasetWrapper,
    clip_extreme_microvolts,
    split_train_test,
    volts_to_microvolts,
)


def fake_moabb_dataset(
    channel_names: list[str] | None = None,
    descriptions: list[str] | None = None,
):
    raw = SimpleNamespace(
        ch_names=channel_names or ["C3", "Cz", "C4"],
        annotations=SimpleNamespace(
            description=descriptions or ["right_hand", "left_hand"],
        ),
    )
    return SimpleNamespace(datasets=[SimpleNamespace(raw=raw)])


def test_load_creates_moabb_dataset_and_applies_minimal_preprocessing(
    monkeypatch,
) -> None:
    created = {}
    preprocessed = {}

    class FakeMOABBDataset:
        def __init__(self, dataset_name: str) -> None:
            self.dataset_name = dataset_name
            created["dataset"] = self

    def fake_preprocess(dataset, preprocessors) -> None:
        preprocessed["dataset"] = dataset
        preprocessed["preprocessors"] = preprocessors

    monkeypatch.setattr(datasets, "MOABBDataset", FakeMOABBDataset)
    monkeypatch.setattr(datasets, "preprocess", fake_preprocess)

    wrapper = MOABBDatasetWrapper.load("BNCI2014_001")

    assert wrapper.dataset_name == "BNCI2014_001"
    assert wrapper.dataset is created["dataset"]
    assert preprocessed["dataset"] is wrapper.dataset
    preprocessors = preprocessed["preprocessors"]
    assert len(preprocessors) == 3
    assert preprocessors[0].fn == "pick"
    assert preprocessors[0].kwargs == {"picks": "eeg"}
    assert preprocessors[0].apply_on_array is False
    assert preprocessors[1].fn is volts_to_microvolts
    assert preprocessors[1].apply_on_array is True
    assert preprocessors[2].fn is clip_extreme_microvolts
    assert preprocessors[2].apply_on_array is True


def test_volts_to_microvolts_scales_values() -> None:
    values = np.array([[-1e-6, 0.0, 2e-6]])

    result = volts_to_microvolts(values)

    np.testing.assert_array_equal(result, np.array([[-1.0, 0.0, 2.0]]))


def test_clip_extreme_microvolts_clips_large_values() -> None:
    values = np.array([[-2_000.0, -20.0, 20.0, 2_000.0]])

    result = clip_extreme_microvolts(values)

    np.testing.assert_array_equal(result, np.array([[-1_000.0, -20.0, 20.0, 1_000.0]]))


def test_channel_names_come_from_first_raw_dataset() -> None:
    wrapper = MOABBDatasetWrapper(
        dataset_name="fake",
        dataset=fake_moabb_dataset(channel_names=["Fz", "Cz"]),
    )

    assert wrapper.channel_names == ["Fz", "Cz"]


def test_event_mapping_is_sorted_and_deduplicated() -> None:
    wrapper = MOABBDatasetWrapper(
        dataset_name="fake",
        dataset=fake_moabb_dataset(
            descriptions=["tongue", "left_hand", "tongue", "feet"],
        ),
    )

    assert wrapper.event_mapping == {
        "feet": 0,
        "left_hand": 1,
        "tongue": 2,
    }


def test_create_event_windows_uses_event_mapping(monkeypatch) -> None:
    wrapper = MOABBDatasetWrapper(
        dataset_name="fake",
        dataset=fake_moabb_dataset(descriptions=["right_hand", "left_hand"]),
    )
    captured = {}
    fake_windows = object()

    def fake_create_windows_from_events(dataset, **kwargs):
        captured["dataset"] = dataset
        captured["kwargs"] = kwargs
        return fake_windows

    monkeypatch.setattr(
        datasets,
        "create_windows_from_events",
        fake_create_windows_from_events,
    )

    windows = wrapper.create_event_windows()

    assert windows is fake_windows
    assert captured["dataset"] is wrapper.dataset
    assert captured["kwargs"]["mapping"] == {
        "left_hand": 0,
        "right_hand": 1,
    }
    assert captured["kwargs"]["window_size_samples"] is None
    assert captured["kwargs"]["window_stride_samples"] is None
    assert captured["kwargs"]["drop_last_window"] is True
    assert captured["kwargs"]["preload"] is False


def test_split_train_test_returns_expected_session_splits() -> None:
    class FakeWindows:
        def split(self, key: str):
            assert key == "session"
            return {
                "0train": "train-windows",
                "1test": "test-windows",
            }

    assert split_train_test(FakeWindows()) == (
        "train-windows",
        "test-windows",
    )
