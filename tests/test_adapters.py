import numpy as np

from eeg_benchmark.adapters import adapt_windows_to_model, pad_or_crop_time


class FakeModel:
    def __init__(self, input_window_samples: int | None) -> None:
        self.input_window_samples = input_window_samples


class FakeWindowDataset:
    def __init__(self, transform=None) -> None:
        self.transform = transform


class FakeWindows:
    def __init__(self, *datasets: FakeWindowDataset) -> None:
        self.datasets = list(datasets)


def test_pad_or_crop_time_crops_last_axis() -> None:
    x = np.arange(10).reshape(2, 5)

    result = pad_or_crop_time(x, target_window_samples=3)

    np.testing.assert_array_equal(result, x[:, :3])


def test_pad_or_crop_time_pads_last_axis_with_zeroes() -> None:
    x = np.ones((2, 3))

    result = pad_or_crop_time(x, target_window_samples=5)

    assert result.shape == (2, 5)
    np.testing.assert_array_equal(result[:, :3], x)
    np.testing.assert_array_equal(result[:, 3:], np.zeros((2, 2)))


def test_adapt_windows_to_model_records_no_time_adjustment() -> None:
    windows = FakeWindows(FakeWindowDataset())

    adapted, target_window_samples, time_adjustment = adapt_windows_to_model(
        windows,
        FakeModel(input_window_samples=None),
    )

    assert adapted is windows
    assert target_window_samples is None
    assert time_adjustment == "none"


def test_adapt_windows_to_model_preserves_existing_transform() -> None:
    window_dataset = FakeWindowDataset(transform=lambda x: x + 1)
    windows = FakeWindows(window_dataset)

    adapted, target_window_samples, time_adjustment = adapt_windows_to_model(
        windows,
        FakeModel(input_window_samples=2),
    )
    result = adapted.datasets[0].transform(np.array([[1, 2, 3]]))

    np.testing.assert_array_equal(result, np.array([[2, 3]]))
    assert target_window_samples == 2
    assert time_adjustment == "pad_or_crop"
