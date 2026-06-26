from types import SimpleNamespace

from eeg_benchmark import datasets
from eeg_benchmark.datasets import MOABBDatasetWrapper


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


def test_load_creates_moabb_dataset_and_picks_eeg(monkeypatch) -> None:
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
    assert len(preprocessed["preprocessors"]) == 1


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

    wrapper = MOABBDatasetWrapper(
        dataset_name="fake",
        dataset=fake_moabb_dataset(),
    )

    assert wrapper.split_train_test(FakeWindows()) == (
        "train-windows",
        "test-windows",
    )
