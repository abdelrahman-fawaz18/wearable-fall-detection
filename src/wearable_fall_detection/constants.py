"""Shared dataset and model definitions."""

from __future__ import annotations

ACTIVITY_CLASSES = (
    "Fall",
    "Downstairs",
    "Upstairs",
    "Walking",
    "Jogging",
    "Sitting",
    "Standing",
    "Lying",
)

BINARY_CLASSES = ("Fall", "No fall")

CHANNEL_NAMES = (
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
)

PREFIX_TO_CLASS = {
    "fall": "Fall",
    "dws": "Downstairs",
    "ups": "Upstairs",
    "wlk": "Walking",
    "jog": "Jogging",
    "sit": "Sitting",
    "std": "Standing",
    "lie": "Lying",
}

FRAME_ROWS = 50
MODEL_ROWS = 46
MAX_AUTOCORRELATION_LAG = 14
