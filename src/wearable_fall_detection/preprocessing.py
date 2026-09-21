"""Signal preparation for raw 100 Hz inertial recordings."""

from __future__ import annotations

import numpy as np


def downsample_pairwise(samples: np.ndarray) -> np.ndarray:
    """Reduce 100 Hz samples to 50 Hz by averaging adjacent rows."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("samples must be a two-dimensional array with at least two rows")
    even_length = values.shape[0] - values.shape[0] % 2
    paired = values[:even_length].reshape(-1, 2, values.shape[1])
    return paired.mean(axis=1)


def mirror_pad(samples: np.ndarray, target_rows: int = 50) -> np.ndarray:
    """Extend a short frame by reflecting samples from its trailing edge."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("samples must be a non-empty two-dimensional array")
    if values.shape[0] >= target_rows:
        return values[:target_rows].copy()

    result = values.copy()
    while result.shape[0] < target_rows:
        reflected = result[-2::-1] if result.shape[0] > 1 else result
        take = min(target_rows - result.shape[0], reflected.shape[0])
        result = np.vstack((result, reflected[:take]))
    return result


def maximum_energy_window(samples: np.ndarray, target_rows: int = 50) -> np.ndarray:
    """Select the contiguous frame with the highest acceleration-vector energy."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("samples must contain at least three acceleration columns")
    if values.shape[0] <= target_rows:
        return mirror_pad(values, target_rows)

    acceleration_energy = np.square(values[:, :3]).sum(axis=1)
    window_energy = np.convolve(
        acceleration_energy, np.ones(target_rows, dtype=np.float64), mode="valid"
    )
    start = int(np.argmax(window_energy))
    return values[start : start + target_rows].copy()


def prepare_dynamic_recording(raw_samples: np.ndarray) -> np.ndarray:
    """Downsample a dynamic recording and return its strongest 50-sample frame."""

    return maximum_energy_window(downsample_pairwise(raw_samples), target_rows=50)


def segment_static_recording(raw_samples: np.ndarray) -> list[np.ndarray]:
    """Downsample and split a static recording into complete one-second frames."""

    values = downsample_pairwise(raw_samples)
    complete_rows = values.shape[0] - values.shape[0] % 50
    return [frame.copy() for frame in values[:complete_rows].reshape(-1, 50, values.shape[1])]
