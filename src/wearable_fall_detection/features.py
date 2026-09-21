"""Autocorrelation features used by the support-vector classifiers."""

from __future__ import annotations

import numpy as np

from .constants import CHANNEL_NAMES, MAX_AUTOCORRELATION_LAG, MODEL_ROWS


def autocorrelation(series: np.ndarray, lag: int) -> float:
    """Calculate Pearson autocorrelation at a positive lag."""

    values = np.asarray(series, dtype=np.float64)
    if lag <= 0 or lag >= values.size:
        raise ValueError("lag must be positive and smaller than the series length")

    left = values[:-lag]
    right = values[lag:]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    correlation = float(np.corrcoef(left, right)[0, 1])
    return correlation if np.isfinite(correlation) else 0.0


def extract_features(
    frame: np.ndarray,
    *,
    sample_count: int = MODEL_ROWS,
    max_lag: int = MAX_AUTOCORRELATION_LAG,
) -> np.ndarray:
    """Extract lag-major autocorrelation values for all six channels."""

    values = np.asarray(frame, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CHANNEL_NAMES):
        raise ValueError(f"frame must have six columns; received shape {values.shape}")
    if sample_count <= max_lag or values.shape[0] < sample_count:
        raise ValueError("frame is too short for the requested sample count and lags")

    model_frame = values[:sample_count]
    return np.array(
        [
            autocorrelation(model_frame[:, channel], lag)
            for lag in range(1, max_lag + 1)
            for channel in range(model_frame.shape[1])
        ],
        dtype=np.float64,
    )


def build_feature_matrix(frames: list[np.ndarray]) -> np.ndarray:
    """Convert prepared frames to the 84-column model matrix."""

    if not frames:
        raise ValueError("at least one frame is required")
    return np.vstack([extract_features(frame) for frame in frames])


def feature_names(max_lag: int = MAX_AUTOCORRELATION_LAG) -> list[str]:
    """Return column names in the exact order produced by extract_features."""

    return [
        f"{channel}_autocorrelation_lag_{lag}"
        for lag in range(1, max_lag + 1)
        for channel in CHANNEL_NAMES
    ]
