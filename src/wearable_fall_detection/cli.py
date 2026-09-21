"""Command-line interface for validation, evaluation, and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .dataset import class_counts, discover_frames, load_frame, load_records, write_manifest
from .features import build_feature_matrix, extract_features
from .modeling import evaluate, fit_full_dataset, save_model, write_evaluation
from .visualization import (
    plot_class_distribution,
    plot_confusion,
    plot_metrics,
    plot_signal_comparison,
)


def validate_command(args: argparse.Namespace) -> None:
    paths = discover_frames(args.data)
    for path in paths:
        load_frame(path)
    write_manifest(args.data, args.manifest)
    print(json.dumps({"frames": len(paths), "classes": class_counts(paths)}, indent=2))


def evaluate_command(args: argparse.Namespace) -> None:
    records = load_records(args.data)
    features = build_feature_matrix([record.values for record in records])
    activities = [record.activity for record in records]
    results = [
        evaluate(features, activities, task="multiclass"),
        evaluate(features, activities, task="binary"),
    ]

    output = Path(args.output)
    write_evaluation(results, output)
    for result in results:
        classifier = fit_full_dataset(features, activities, result.task)
        save_model(classifier, output / f"model-{result.task}.joblib", task=result.task)

    if args.figures:
        figures = Path(args.figures)
        plot_class_distribution(records, figures / "class-distribution.png")
        plot_signal_comparison(records, figures / "signal-comparison.png")
        plot_metrics(results, figures / "benchmark-metrics.png")
        for result in results:
            plot_confusion(result, figures / f"confusion-{result.task}.png")

    print(json.dumps({result.task: result.mean_metrics for result in results}, indent=2))


def predict_command(args: argparse.Namespace) -> None:
    bundle = joblib.load(args.model)
    frame = load_frame(args.frame)
    prediction = bundle["estimator"].predict(extract_features(frame).reshape(1, -1))[0]
    print(json.dumps({"task": bundle["task"], "prediction": str(prediction)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate frames and write a manifest")
    validate.add_argument("--data", type=Path, default=Path("data/frames"))
    validate.add_argument("--manifest", type=Path, default=Path("data/dataset-manifest.csv"))
    validate.set_defaults(handler=validate_command)

    benchmark = subparsers.add_parser("evaluate", help="run both SVC benchmarks")
    benchmark.add_argument("--data", type=Path, default=Path("data/frames"))
    benchmark.add_argument("--output", type=Path, default=Path("artifacts"))
    benchmark.add_argument("--figures", type=Path, default=Path("docs/assets"))
    benchmark.set_defaults(handler=evaluate_command)

    predict = subparsers.add_parser("predict", help="classify one prepared frame")
    predict.add_argument("frame", type=Path)
    predict.add_argument("--model", type=Path, default=Path("artifacts/model-binary.joblib"))
    predict.set_defaults(handler=predict_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
