"""Repository figures generated from the prepared dataset and evaluation output."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constants import ACTIVITY_CLASSES
from .dataset import FrameRecord
from .modeling import EvaluationResult

NAVY = "#12304A"
BLUE = "#177E89"
ORANGE = "#E07A5F"
GOLD = "#F2CC8F"
GRID = "#D8E1E8"


def _finish(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_class_distribution(records: Sequence[FrameRecord], output_path: str | Path) -> None:
    """Plot the number of one-second frames in each activity class."""

    counts = Counter(record.activity for record in records)
    labels = list(ACTIVITY_CLASSES)
    values = [counts[label] for label in labels]
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    bars = axis.bar(labels, values, color=[ORANGE] + [BLUE] * (len(labels) - 1), width=0.72)
    axis.set_ylabel("One-second frames")
    axis.set_title("Prepared dataset: 1,191 labeled motion frames", loc="left", color=NAVY, weight="bold")
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="x", rotation=24)
    for bar, count in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, count + 2, str(count), ha="center", va="bottom", color=NAVY)
    _finish(figure, Path(output_path))


def plot_signal_comparison(records: Sequence[FrameRecord], output_path: str | Path) -> None:
    """Compare complete six-axis traces for representative fall and walking frames."""

    def motion_score(record: FrameRecord) -> float:
        acceleration_span = np.ptp(record.values[:, :3], axis=0).sum()
        angular_rate_span = np.ptp(record.values[:, 3:], axis=0).sum()
        return float(acceleration_span + angular_rate_span / 100.0)

    selected = {
        activity: max(
            (record for record in records if record.activity == activity),
            key=motion_score,
        )
        for activity in ("Fall", "Walking")
    }
    figure, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True)
    time_seconds = np.arange(50) / 50.0
    axis_colors = (ORANGE, BLUE, "#6A4C93")
    for column, activity in enumerate(("Fall", "Walking")):
        values = selected[activity].values
        for channel, color in enumerate(axis_colors):
            axes[0, column].plot(
                time_seconds,
                values[:, channel],
                color=color,
                linewidth=1.8,
                label=("X", "Y", "Z")[channel],
            )
            axes[1, column].plot(
                time_seconds,
                values[:, channel + 3],
                color=color,
                linewidth=1.8,
                label=("X", "Y", "Z")[channel],
            )
        axes[0, column].set_title(activity, color=NAVY, weight="bold")
        axes[1, column].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Acceleration (g)")
    axes[1, 0].set_ylabel("Angular rate (°/s)")
    axes[0, 1].legend(frameon=False, ncols=3, loc="upper right", title="Axis")
    for axis in axes.flat:
        axis.grid(color=GRID, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(0, 0.98)
    figure.suptitle("Representative six-axis wrist-motion frames", x=0.08, ha="left", color=NAVY, weight="bold", fontsize=14)
    _finish(figure, Path(output_path))


def plot_confusion(result: EvaluationResult, output_path: str | Path) -> None:
    """Plot an aggregate confusion matrix across evaluation splits."""

    matrix = result.confusion_matrix.astype(float)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0)
    size = 6.5 if result.task == "binary" else 9.5
    figure, axis = plt.subplots(figsize=(size, size * 0.78))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(result.labels)), labels=result.labels, rotation=35, ha="right")
    axis.set_yticks(range(len(result.labels)), labels=result.labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title(f"{result.task.capitalize()} SVC — aggregate of 10 stratified holdouts", loc="left", color=NAVY, weight="bold")
    threshold = 0.55
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{normalized[row, column]:.0%}\n{int(matrix[row, column])}",
                ha="center",
                va="center",
                color="white" if normalized[row, column] > threshold else NAVY,
                fontsize=8 if result.task == "multiclass" else 10,
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row-normalized share")
    _finish(figure, Path(output_path))


def plot_metrics(results: Sequence[EvaluationResult], output_path: str | Path) -> None:
    """Plot the mean benchmark metrics for both classification tasks."""

    metric_keys = ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted")
    metric_labels = ("Accuracy", "Precision", "Recall", "F1")
    x = np.arange(len(metric_keys))
    width = 0.34
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    for index, result in enumerate(results):
        values = [result.mean_metrics[key] * 100 for key in metric_keys]
        offset = (index - 0.5) * width
        bars = axis.bar(
            x + offset,
            values,
            width,
            label=result.task.capitalize(),
            color=ORANGE if result.task == "multiclass" else BLUE,
        )
        for bar, value in zip(bars, values, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.35, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    axis.set_xticks(x, metric_labels)
    axis.set_ylabel("Mean score (%)")
    axis.set_ylim(70, 101)
    axis.set_title("Reproducible SVC benchmark", loc="left", color=NAVY, weight="bold")
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(frameon=False, ncols=2, loc="lower right")
    _finish(figure, Path(output_path))
