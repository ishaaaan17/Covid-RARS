#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from covid_rars.cross_dataset import harmonize_feature_columns
from covid_rars.metrics import evaluate_predictions, labels_to_binary
from covid_rars.models_ml import make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on one feature dataset and evaluate directly on another.")
    parser.add_argument("--source-features", required=True, type=Path)
    parser.add_argument("--external-features", required=True, type=Path)
    parser.add_argument("--modality", default="cough")
    parser.add_argument("--model-name", default="logistic_regression")
    parser.add_argument("--models-dir", type=Path, default=Path("data/outputs/models"))
    parser.add_argument("--predictions-output", type=Path, default=Path("data/outputs/metrics/cross_dataset_predictions.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/outputs/metrics/cross_dataset_metrics.csv"))
    parser.add_argument("--source-train-split", default="train")
    return parser.parse_args()


def _prediction_frame(external: pd.DataFrame, probabilities, model_name: str, modality: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recording_id": external["recording_id"].to_numpy(),
            "participant_id": external["participant_id"].to_numpy(),
            "dataset": external.get("dataset", pd.Series(["external"] * len(external))).to_numpy(),
            "modality": modality,
            "label_binary": external["label_binary"].to_numpy(),
            "split": "external",
            "model_name": model_name,
            "probability": probabilities,
        }
    )


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.source_features)
    external = pd.read_csv(args.external_features)
    source = source[
        (source["label_binary"].isin(["positive", "negative"]))
        & (source["modality"] == args.modality)
        & (source["split"] == args.source_train_split)
    ].copy()
    external = external[
        (external["label_binary"].isin(["positive", "negative"]))
        & (external["modality"] == args.modality)
    ].copy()
    if source.empty or external.empty:
        raise ValueError("Need non-empty source train rows and external labeled rows for selected modality")
    x_source, x_external, columns = harmonize_feature_columns(source, external)
    model = make_model(args.model_name)
    model.fit(x_source, labels_to_binary(source["label_binary"]))
    probabilities = model.predict_proba(x_external)[:, 1]
    predictions = _prediction_frame(external, probabilities, args.model_name, args.modality)
    metrics = evaluate_predictions(predictions)
    metrics["model_name"] = args.model_name
    metrics["modality"] = args.modality
    metrics["source_rows"] = len(source)
    metrics["external_rows"] = len(external)
    metrics["n_features"] = len(columns)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.models_dir / f"cross_dataset_{args.model_name}_{args.modality}.joblib")
    predictions.to_csv(args.predictions_output, index=False)
    metrics.to_csv(args.metrics_output, index=False)
    print(f"Wrote cross-dataset predictions: {args.predictions_output}")
    print(f"Wrote cross-dataset metrics: {args.metrics_output}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
