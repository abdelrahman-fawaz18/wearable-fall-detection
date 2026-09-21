from pathlib import Path

import numpy as np
import pytest

from wearable_fall_detection.dataset import activity_from_filename, load_frame


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Fall 1.csv", "Fall"),
        ("dws 200.csv", "Downstairs"),
        ("ups 350.csv", "Upstairs"),
        ("wlk 800.csv", "Walking"),
        ("jog 50.csv", "Jogging"),
        ("sit 500.csv", "Sitting"),
        ("std 650.csv", "Standing"),
        ("lie 12.csv", "Lying"),
    ],
)
def test_activity_from_filename(filename: str, expected: str) -> None:
    assert activity_from_filename(filename) == expected


def test_load_frame_enforces_shape(tmp_path: Path) -> None:
    valid = tmp_path / "Fall 1.csv"
    np.savetxt(valid, np.zeros((50, 6)), delimiter=",")
    assert load_frame(valid).shape == (50, 6)

    invalid = tmp_path / "Fall 2.csv"
    np.savetxt(invalid, np.zeros((49, 6)), delimiter=",")
    with pytest.raises(ValueError, match="expected"):
        load_frame(invalid)
