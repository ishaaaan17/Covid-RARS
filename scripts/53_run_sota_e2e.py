#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command SOTA-targeted internal benchmark pipeline.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Coswara raw directory, used only if metadata is absent.")
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_with_quality.csv"))
    parser.add_argument("--segment-index", type=Path, default=Path("data/processed/sota_segment_index.csv"))
    parser.add_argument("--segment-audit", type=Path, default=Path("reports/tables/sota_segment_index_audit.csv"))
    parser.add_argument("--modalities", nargs="+", default=["cough", "breath", "speech"])
    parser.add_argument("--quality-mode", choices=["all_samples", "quality_ok_only"], default="quality_ok_only")
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--max-segments-per-recording", type=int, default=8)
    parser.add_argument("--augment-train-copies", type=int, default=2)
    parser.add_argument("--backend", choices=["hf_ssl", "debug_acoustic"], default="hf_ssl")
    parser.add_argument("--ssl-models", nargs="+", default=["microsoft/wavlm-base-plus"])
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--unfreeze-top-layers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fusion-top-k", type=int, default=8)
    parser.add_argument("--skip-report-refresh", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def _ensure_metadata(args: argparse.Namespace) -> None:
    metadata = PROJECT_ROOT / args.metadata
    if metadata.exists():
        return
    if args.raw_dir is None:
        raise FileNotFoundError(
            f"Metadata not found: {args.metadata}. Provide --raw-dir or run preprocessing first."
        )
    raw_dir = args.raw_dir
    _run([sys.executable, "scripts/01_build_coswara_index.py", "--raw-dir", str(raw_dir), "--output", "data/interim/coswara_index.csv"])
    _run(
        [
            sys.executable,
            "scripts/02_clean_metadata.py",
            "--index",
            "data/interim/coswara_index.csv",
            "--output",
            "data/processed/metadata_clean.csv",
            "--availability-output",
            "data/processed/modality_availability.csv",
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/03_create_splits.py",
            "--metadata",
            "data/processed/metadata_clean.csv",
            "--output",
            "data/processed/split_manifest.csv",
            "--metadata-output",
            "data/processed/metadata_with_split.csv",
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/04_quality_audit.py",
            "--metadata",
            "data/processed/metadata_with_split.csv",
            "--output",
            "data/processed/audio_quality.csv",
            "--metadata-output",
            str(args.metadata),
        ]
    )


def main() -> None:
    args = parse_args()
    _ensure_metadata(args)
    _run(
        [
            sys.executable,
            "scripts/50_build_sota_segment_index.py",
            "--metadata",
            str(args.metadata),
            "--output",
            str(args.segment_index),
            "--audit-output",
            str(args.segment_audit),
            "--modalities",
            *args.modalities,
            "--quality-mode",
            args.quality_mode,
            "--window-sec",
            str(args.window_sec),
            "--overlap",
            str(args.overlap),
            "--max-segments-per-recording",
            str(args.max_segments_per_recording),
            "--augment-train-copies",
            str(args.augment_train_copies),
        ]
    )

    metric_paths: list[str] = []
    prediction_paths: list[str] = []
    for model_name in args.ssl_models:
        safe_model = _safe_model_name(model_name)
        for modality in args.modalities:
            metrics_output = Path(f"data/outputs/metrics/sota_ssl_metrics_{args.backend}_{safe_model}_{modality}.csv")
            predictions_output = Path(f"data/outputs/metrics/sota_ssl_predictions_{args.backend}_{safe_model}_{modality}.csv")
            history_output = Path(f"data/outputs/metrics/sota_ssl_history_{args.backend}_{safe_model}_{modality}.csv")
            cmd = [
                sys.executable,
                "scripts/51_train_sota_ssl_branch.py",
                "--segment-index",
                str(args.segment_index),
                "--modality",
                modality,
                "--backend",
                args.backend,
                "--model-name",
                model_name,
                "--max-epochs",
                str(args.max_epochs),
                "--batch-size",
                str(args.batch_size),
                "--gradient-accumulation",
                str(args.gradient_accumulation),
                "--learning-rate",
                str(args.learning_rate),
                "--head-learning-rate",
                str(args.head_learning_rate),
                "--unfreeze-top-layers",
                str(args.unfreeze_top_layers),
                "--patience",
                str(args.patience),
                "--metrics-output",
                str(metrics_output),
                "--predictions-output",
                str(predictions_output),
                "--history-output",
                str(history_output),
            ]
            if args.device:
                cmd.extend(["--device", args.device])
            _run(cmd)
            metric_paths.append(str(metrics_output))
            prediction_paths.append(str(predictions_output))

    if len(metric_paths) >= 2 and len(prediction_paths) >= 2:
        _run(
            [
                sys.executable,
                "scripts/52_fuse_sota_predictions.py",
                "--metrics",
                *metric_paths,
                "--predictions",
                *prediction_paths,
                "--top-k",
                str(args.fusion_top_k),
                "--metrics-output",
                "data/outputs/metrics/sota_fusion_metrics.csv",
                "--predictions-output",
                "data/outputs/metrics/sota_fusion_predictions.csv",
            ]
        )
    else:
        print("Skipping SOTA fusion because fewer than two branch prediction sources were produced.")

    if not args.skip_report_refresh:
        _run([sys.executable, "scripts/20_make_paper_tables.py"])
        _run([sys.executable, "scripts/24_make_experiment_manifest.py"])
    if not args.skip_validation:
        index_path = PROJECT_ROOT / "data" / "interim" / "coswara_index.csv"
        quality_path = PROJECT_ROOT / "data" / "processed" / "audio_quality.csv"
        if index_path.exists() and quality_path.exists():
            _run(
                [
                    sys.executable,
                    "scripts/12_validate_artifacts.py",
                    "--index",
                    "data/interim/coswara_index.csv",
                    "--metadata",
                    str(args.metadata),
                    "--quality",
                    "data/processed/audio_quality.csv",
                    "--strict",
                ]
            )
        else:
            print("Skipping strict artifact validation because index/quality artifacts are absent.")

    print("\nSOTA E2E pipeline completed.")
    print("Primary outputs:")
    print(f"- {args.segment_index}")
    print("- data/outputs/metrics/sota_ssl_metrics_*.csv")
    print("- data/outputs/metrics/sota_ssl_predictions_*.csv")
    print("- data/outputs/metrics/sota_fusion_metrics.csv")


if __name__ == "__main__":
    main()
