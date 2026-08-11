#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covid_rars.sota_segments import build_sota_segment_index, validate_segment_index_no_leakage
from covid_rars.sota_ssl import train_sota_ssl_branch
from covid_rars.spectrograms import build_spectrogram_index
from covid_rars.train_cnn import train_cnn_for_modality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deep cough-only Coswara to COUGHVID external transfer for WavLM and CNN-BiGRU."
    )
    parser.add_argument("--source-metadata", type=Path, default=Path("data/processed/metadata_with_quality.csv"))
    parser.add_argument(
        "--external-metadata",
        type=Path,
        default=Path("data/processed/coughvid_metadata_compare_is10_external.csv"),
    )
    parser.add_argument(
        "--combined-metadata-output",
        type=Path,
        default=Path("data/processed/metadata_coswara_coughvid_cough_external.csv"),
    )
    parser.add_argument(
        "--segment-index-output",
        type=Path,
        default=Path("data/processed/sota_segment_index_coughvid_external.csv"),
    )
    parser.add_argument(
        "--segment-audit-output",
        type=Path,
        default=Path("reports/tables/sota_segment_index_coughvid_external_audit.csv"),
    )
    parser.add_argument(
        "--spectrogram-dir",
        type=Path,
        default=Path("data/processed/spectrograms_coughvid_external"),
    )
    parser.add_argument(
        "--spectrogram-index-output",
        type=Path,
        default=Path("data/processed/spectrogram_index_coughvid_external.csv"),
    )
    parser.add_argument("--quality-mode", choices=["all_samples", "quality_ok_only"], default="quality_ok_only")
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--max-segments-per-recording", type=int, default=4)
    parser.add_argument("--augment-train-copies", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-wavlm", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--ssl-backend", choices=["hf_ssl", "debug_acoustic"], default="hf_ssl")
    parser.add_argument("--ssl-model-name", default="microsoft/wavlm-base-plus")
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--unfreeze-top-layers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cnn-architecture", choices=["compact_cnn", "residual_cnn", "cnn_bigru"], default="cnn_bigru")
    parser.add_argument("--cnn-epochs", type=int, default=8)
    parser.add_argument("--cnn-batch-size", type=int, default=16)
    parser.add_argument("--cnn-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--wavlm-metrics-output",
        type=Path,
        default=Path("data/outputs/metrics/deep_external_wavlm_cough_metrics.csv"),
    )
    parser.add_argument(
        "--wavlm-predictions-output",
        type=Path,
        default=Path("data/outputs/metrics/deep_external_wavlm_cough_predictions.csv"),
    )
    parser.add_argument(
        "--wavlm-history-output",
        type=Path,
        default=Path("data/outputs/metrics/deep_external_wavlm_cough_history.csv"),
    )
    parser.add_argument(
        "--cnn-metrics-output",
        type=Path,
        default=Path("data/outputs/metrics/deep_external_cnn_bigru_cough_metrics.csv"),
    )
    parser.add_argument(
        "--cnn-validation-output",
        type=Path,
        default=Path("data/outputs/metrics/deep_external_cnn_bigru_cough_validation_predictions.csv"),
    )
    parser.add_argument(
        "--cnn-test-output",
        type=Path,
        default=Path("data/outputs/metrics/deep_external_cnn_bigru_cough_test_predictions.csv"),
    )
    parser.add_argument(
        "--cnn-external-output",
        type=Path,
        default=Path("data/outputs/metrics/deep_external_cnn_bigru_cough_external_predictions.csv"),
    )
    parser.add_argument(
        "--cnn-history-output",
        type=Path,
        default=Path("data/outputs/metrics/deep_external_cnn_bigru_cough_history.csv"),
    )
    return parser.parse_args()


def _prepare_combined_metadata(source_path: Path, external_path: Path) -> pd.DataFrame:
    source = pd.read_csv(source_path)
    external = pd.read_csv(external_path)
    source = source[source["label_binary"].isin(["positive", "negative"])].copy()
    source = source[source["modality"].astype(str).eq("cough")].copy()
    external = external[external["label_binary"].isin(["positive", "negative"])].copy()
    external["split"] = "external_test"
    external["dataset"] = "coughvid"
    external["modality"] = "cough"
    if "submodality" not in external.columns:
        external["submodality"] = "cough"
    external["quality_flag"] = external.get("quality_flag", "ok").fillna("ok")
    source["quality_flag"] = source.get("quality_flag", "ok").fillna("ok")
    return pd.concat([source, external], ignore_index=True, sort=False)


def _write_frame(frame: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {label}: {path} ({len(frame)} rows)")


def main() -> None:
    args = parse_args()
    combined = _prepare_combined_metadata(args.source_metadata, args.external_metadata)
    _write_frame(combined, args.combined_metadata_output, "combined cough metadata")

    if not args.skip_wavlm:
        segment_index = build_sota_segment_index(
            combined,
            modalities=["cough"],
            quality_mode=args.quality_mode,
            window_sec=args.window_sec,
            overlap=args.overlap,
            max_segments_per_recording=args.max_segments_per_recording,
            augment_train_copies=args.augment_train_copies,
            random_state=args.random_state,
        )
        audit = validate_segment_index_no_leakage(segment_index)
        _write_frame(segment_index, args.segment_index_output, "SOTA segment index with COUGHVID external")
        _write_frame(audit, args.segment_audit_output, "SOTA segment leakage audit")
        errors = audit[audit["severity"].astype(str).eq("error")]
        if not errors.empty:
            raise ValueError("Segment leakage audit failed; inspect the audit CSV before training.")

        wavlm = train_sota_ssl_branch(
            segment_index,
            modality="cough",
            backend=args.ssl_backend,
            model_name=args.ssl_model_name,
            target_samples=int(round(args.window_sec * 16000)),
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            head_learning_rate=args.head_learning_rate,
            unfreeze_top_layers=args.unfreeze_top_layers,
            patience=args.patience,
            device=args.device,
        )
        _write_frame(wavlm.metrics, args.wavlm_metrics_output, "WavLM external-transfer metrics")
        _write_frame(wavlm.predictions, args.wavlm_predictions_output, "WavLM external-transfer predictions")
        _write_frame(wavlm.history, args.wavlm_history_output, "WavLM external-transfer history")

    if not args.skip_cnn:
        spectrogram_index = build_spectrogram_index(combined, output_dir=args.spectrogram_dir)
        _write_frame(spectrogram_index, args.spectrogram_index_output, "spectrogram index with COUGHVID external")
        cnn = train_cnn_for_modality(
            spectrogram_index,
            modality="cough",
            architecture=args.cnn_architecture,
            max_epochs=args.cnn_epochs,
            patience=args.patience,
            batch_size=args.cnn_batch_size,
            learning_rate=args.cnn_learning_rate,
            augment=True,
            device=args.device,
        )
        metric_rows = [cnn.metrics]
        if cnn.external_metrics is not None:
            metric_rows.append(cnn.external_metrics)
        _write_frame(pd.DataFrame(metric_rows), args.cnn_metrics_output, "CNN-BiGRU external-transfer metrics")
        _write_frame(cnn.validation_predictions, args.cnn_validation_output, "CNN-BiGRU validation predictions")
        _write_frame(cnn.test_predictions, args.cnn_test_output, "CNN-BiGRU test predictions")
        if cnn.external_predictions is not None:
            _write_frame(cnn.external_predictions, args.cnn_external_output, "CNN-BiGRU external predictions")
        _write_frame(cnn.history, args.cnn_history_output, "CNN-BiGRU training history")


if __name__ == "__main__":
    main()
