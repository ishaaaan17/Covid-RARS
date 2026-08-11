#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.sota_ssl import train_sota_ssl_branch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one SOTA waveform SSL branch for one modality.")
    parser.add_argument("--segment-index", type=Path, default=Path("data/processed/sota_segment_index.csv"))
    parser.add_argument("--modality", required=True)
    parser.add_argument("--backend", choices=["hf_ssl", "debug_acoustic"], default="hf_ssl")
    parser.add_argument("--model-name", default="microsoft/wavlm-base-plus")
    parser.add_argument("--target-seconds", type=float, default=3.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--unfreeze-top-layers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-output", type=Path, default=None)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--history-output", type=Path, default=None)
    return parser.parse_args()


def _default_path(kind: str, modality: str, backend: str, model_name: str) -> Path:
    safe_model = model_name.replace("/", "_").replace(":", "_")
    if kind == "metrics":
        return Path(f"data/outputs/metrics/sota_ssl_metrics_{backend}_{safe_model}_{modality}.csv")
    if kind == "predictions":
        return Path(f"data/outputs/metrics/sota_ssl_predictions_{backend}_{safe_model}_{modality}.csv")
    if kind == "history":
        return Path(f"data/outputs/metrics/sota_ssl_history_{backend}_{safe_model}_{modality}.csv")
    raise ValueError(kind)


def main() -> None:
    args = parse_args()
    segment_index = pd.read_csv(args.segment_index)
    target_samples = int(round(float(args.target_seconds) * int(args.sample_rate)))
    result = train_sota_ssl_branch(
        segment_index,
        modality=args.modality,
        backend=args.backend,
        model_name=args.model_name,
        target_samples=target_samples,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        unfreeze_top_layers=args.unfreeze_top_layers,
        patience=args.patience,
        device=args.device,
        model_output=args.model_output,
    )
    metrics_output = args.metrics_output or _default_path("metrics", args.modality, args.backend, args.model_name)
    predictions_output = args.predictions_output or _default_path("predictions", args.modality, args.backend, args.model_name)
    history_output = args.history_output or _default_path("history", args.modality, args.backend, args.model_name)
    for path, frame in (
        (metrics_output, result.metrics),
        (predictions_output, result.predictions),
        (history_output, result.history),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    print(f"Wrote SOTA SSL metrics: {metrics_output} ({len(result.metrics)} rows)")
    print(f"Wrote SOTA SSL predictions: {predictions_output} ({len(result.predictions)} rows)")
    print(f"Wrote SOTA SSL history: {history_output} ({len(result.history)} rows)")


if __name__ == "__main__":
    main()
