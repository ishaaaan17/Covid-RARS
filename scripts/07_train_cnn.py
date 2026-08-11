#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from covid_rars.train_cnn import train_cnn_for_modality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deep CNN/CRNN models on saved log-mel spectrograms.")
    parser.add_argument("--spectrogram-index", required=True, type=Path)
    parser.add_argument("--modality", default="cough")
    parser.add_argument("--architecture", choices=["compact_cnn", "residual_cnn", "cnn_bigru"], default="compact_cnn")
    parser.add_argument("--models-dir", type=Path, default=Path("data/outputs/models"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/outputs/metrics/cnn_metrics.csv"))
    parser.add_argument("--validation-output", type=Path, default=Path("data/outputs/metrics/cnn_logits_validation.csv"))
    parser.add_argument("--test-output", type=Path, default=Path("data/outputs/metrics/cnn_logits_test.csv"))
    parser.add_argument("--external-output", type=Path, default=Path("data/outputs/metrics/cnn_logits_external_test.csv"))
    parser.add_argument("--external-metrics-output", type=Path, default=Path("data/outputs/metrics/cnn_metrics_external_test.csv"))
    parser.add_argument("--history-output", type=Path, default=Path("data/outputs/metrics/cnn_training_history.csv"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_index = pd.read_csv(args.spectrogram_index)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)

    artifacts = train_cnn_for_modality(
        spec_index,
        modality=args.modality,
        architecture=args.architecture,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        augment=not args.no_augment,
    )
    torch.save(artifacts.model.state_dict(), args.models_dir / f"{args.architecture}_{args.modality}.pt")
    pd.DataFrame([artifacts.metrics]).to_csv(args.metrics_output, index=False)
    artifacts.validation_predictions.to_csv(args.validation_output, index=False)
    artifacts.test_predictions.to_csv(args.test_output, index=False)
    if artifacts.external_predictions is not None and artifacts.external_metrics is not None:
        args.external_output.parent.mkdir(parents=True, exist_ok=True)
        args.external_metrics_output.parent.mkdir(parents=True, exist_ok=True)
        artifacts.external_predictions.to_csv(args.external_output, index=False)
        pd.DataFrame([artifacts.external_metrics]).to_csv(args.external_metrics_output, index=False)
    artifacts.history.to_csv(args.history_output, index=False)
    print(f"Wrote CNN model and metrics for modality={args.modality}")


if __name__ == "__main__":
    main()
