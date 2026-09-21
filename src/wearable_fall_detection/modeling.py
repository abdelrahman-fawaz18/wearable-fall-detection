"""Reproducible support-vector classifier evaluation and serialization."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from .constants import ACTIVITY_CLASSES, BINARY_CLASSES


@dataclass(frozen=True)
class RunMetrics:
    task: str
    seed: int
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float


@dataclass(frozen=True)
class EvaluationResult:
    task: str
    labels: tuple[str, ...]
    runs: tuple[RunMetrics, ...]
    confusion_matrix: np.ndarray

    @property
    def mean_metrics(self) -> dict[str, float]:
        return {
            name: float(np.mean([getattr(run, name) for run in self.runs]))
            for name in ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted")
        }


def make_classifier() -> SVC:
    """Create the RBF SVC configuration used by the project."""

    return SVC(C=100, kernel="rbf", gamma=0.01)


def target_labels(activities: Sequence[str], task: str) -> np.ndarray:
    """Return multiclass activities or binary fall labels."""

    labels = np.asarray(activities, dtype=object)
    if task == "multiclass":
        return labels
    if task == "binary":
        return np.where(labels == "Fall", "Fall", "No fall")
    raise ValueError("task must be 'multiclass' or 'binary'")


def evaluate(
    features: np.ndarray,
    activities: Sequence[str],
    *,
    task: str,
    seeds: Sequence[int] = tuple(range(10)),
    test_size: float = 0.2,
) -> EvaluationResult:
    """Evaluate an SVC over repeated, stratified holdout splits."""

    y = target_labels(activities, task)
    labels = ACTIVITY_CLASSES if task == "multiclass" else BINARY_CLASSES
    aggregate_confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    runs: list[RunMetrics] = []

    for seed in seeds:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
        classifier = make_classifier()
        classifier.fit(x_train, y_train)
        predicted = classifier.predict(x_test)
        aggregate_confusion += confusion_matrix(y_test, predicted, labels=labels)
        runs.append(
            RunMetrics(
                task=task,
                seed=int(seed),
                accuracy=float(accuracy_score(y_test, predicted)),
                precision_weighted=float(
                    precision_score(y_test, predicted, average="weighted", zero_division=0)
                ),
                recall_weighted=float(
                    recall_score(y_test, predicted, average="weighted", zero_division=0)
                ),
                f1_weighted=float(
                    f1_score(y_test, predicted, average="weighted", zero_division=0)
                ),
            )
        )

    return EvaluationResult(
        task=task,
        labels=tuple(labels),
        runs=tuple(runs),
        confusion_matrix=aggregate_confusion,
    )


def fit_full_dataset(features: np.ndarray, activities: Sequence[str], task: str) -> SVC:
    """Fit one classifier on all available frames."""

    classifier = make_classifier()
    classifier.fit(features, target_labels(activities, task))
    return classifier


def save_model(classifier: SVC, output_path: str | Path, *, task: str) -> None:
    """Serialize a classifier with its feature and label metadata."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "estimator": classifier,
            "task": task,
            "feature_method": "autocorrelation",
            "sample_count": 46,
            "max_lag": 14,
        },
        destination,
    )


def write_evaluation(results: Sequence[EvaluationResult], output_directory: str | Path) -> None:
    """Write machine-readable metrics and confusion matrices."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)

    all_runs = [asdict(run) for result in results for run in result.runs]
    with (destination / "benchmark-runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(all_runs[0]))
        writer.writeheader()
        writer.writerows(all_runs)

    summary = {result.task: result.mean_metrics for result in results}
    (destination / "benchmark-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    for result in results:
        with (destination / f"confusion-{result.task}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(("actual/predicted", *result.labels))
            for label, row in zip(result.labels, result.confusion_matrix, strict=True):
                writer.writerow((label, *row.tolist()))
