import numpy as np

from wearable_fall_detection.preprocessing import (
    downsample_pairwise,
    maximum_energy_window,
    mirror_pad,
    segment_static_recording,
)


def test_pairwise_downsampling_averages_adjacent_rows() -> None:
    source = np.arange(24, dtype=float).reshape(6, 4)
    expected = (source[::2] + source[1::2]) / 2
    np.testing.assert_allclose(downsample_pairwise(source), expected)


def test_mirror_padding_preserves_original_prefix() -> None:
    source = np.arange(18, dtype=float).reshape(3, 6)
    padded = mirror_pad(source, target_rows=7)
    np.testing.assert_array_equal(padded[:3], source)
    np.testing.assert_array_equal(padded[3], source[1])
    assert padded.shape == (7, 6)


def test_maximum_energy_window_finds_impulse() -> None:
    source = np.zeros((100, 6))
    source[70, :3] = 10
    selected = maximum_energy_window(source, target_rows=50)
    assert np.max(selected[:, :3]) == 10


def test_static_segmentation_returns_complete_frames() -> None:
    source = np.zeros((220, 9))
    frames = segment_static_recording(source)
    assert len(frames) == 2
    assert all(frame.shape == (50, 9) for frame in frames)
