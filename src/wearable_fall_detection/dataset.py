"""Loading and validation for the prepared inertial frames."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import CHANNEL_NAMES, FRAME_ROWS, PREFIX_TO_CLASS


@dataclass(frozen=True)
class FrameRecord:
    """A validated sensor frame and its activity label."""

    path: Path
    activity: str
    values: np.ndarray


def activity_from_filename(path: str | Path) -> str:
    """Return the activity encoded by a prepared-frame filename."""

    prefix = Path(path).stem.split(maxsplit=1)[0].lower()
    try:
        return PREFIX_TO_CLASS[prefix]
    except KeyError as exc:
        raise ValueError(f"Unknown activity prefix in {Path(path).name!r}") from exc


def discover_frames(directory: str | Path) -> list[Path]:
    """Return CSV frame paths in stable name order."""

    paths = sorted(Path(directory).glob("*.csv"), key=lambda item: item.name.lower())
    if not paths:
        raise FileNotFoundError(f"No CSV frames found in {Path(directory)}")
    return paths


def load_frame(path: str | Path, expected_rows: int = FRAME_ROWS) -> np.ndarray:
    """Load one headerless frame and enforce its six-channel schema."""

    frame_path = Path(path)
    try:
        values = np.loadtxt(frame_path, delimiter=",", dtype=np.float64, ndmin=2)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric data in {frame_path}") from exc

    expected_shape = (expected_rows, len(CHANNEL_NAMES))
    if values.shape != expected_shape:
        raise ValueError(
            f"{frame_path.name} has shape {values.shape}; expected {expected_shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{frame_path.name} contains non-finite values")
    return values


def load_records(directory: str | Path) -> list[FrameRecord]:
    """Load every prepared frame in a dataset directory."""

    return [
        FrameRecord(path=path, activity=activity_from_filename(path), values=load_frame(path))
        for path in discover_frames(directory)
    ]


def class_counts(paths: Iterable[str | Path]) -> Counter[str]:
    """Count activities from frame filenames."""

    return Counter(activity_from_filename(path) for path in paths)


def write_manifest(directory: str | Path, output_path: str | Path) -> None:
    """Write a compact inventory of the prepared dataset."""

    rows = []
    for path in discover_frames(directory):
        values = load_frame(path)
        rows.append(
            {
                "file": path.name,
                "activity": activity_from_filename(path),
                "samples": values.shape[0],
                "channels": values.shape[1],
            }
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("file", "activity", "samples", "channels")
        )
        writer.writeheader()
        writer.writerows(rows)
