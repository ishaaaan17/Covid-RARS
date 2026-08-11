#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command paper-comparable strong-baseline pipeline."
    )
    parser.add_argument("--raw-dir", type=Path, default=None, help="Coswara raw directory; required if features are absent.")
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_strong_acoustic.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_with_quality.csv"))
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--require-quality-ok", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true", help="Skip the compact spectrogram CNN branch.")
    parser.add_argument(
        "--cnn-architectures",
        nargs="+",
        default=["residual_cnn", "cnn_bigru"],
        choices=["compact_cnn", "residual_cnn", "cnn_bigru"],
    )
    parser.add_argument("--cnn-epochs", type=int, default=50)
    parser.add_argument("--cnn-batch-size", type=int, default=32)
    parser.add_argument("--optuna-trials", type=int, default=25)
    parser.add_argument("--ensemble-top-k", type=int, default=3)
    parser.add_argument("--augment-train-copies", type=int, default=0)
    parser.add_argument("--augmentation-seed", type=int, default=42)
    parser.add_argument("--feature-level-fusion", action="store_true")
    parser.add_argument("--global-stack-top-k", type=int, default=0)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--modalities", nargs="+", default=["cough", "breath", "speech"])
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=[
            "logistic_l2_f80",
            "logistic_smote_f80",
            "extra_trees_f100",
            "extra_trees_smote_f100",
            "random_forest_f80",
            "random_forest_smote_f80",
            "svc_rbf_f60",
            "xgboost_smote_f80",
            "lightgbm_smote_f80",
            "catboost_smote_f80",
            "optuna_validation_search",
        ],
    )
    return parser.parse_args()


def _run(args: list[str]) -> None:
    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def _path(path: Path) -> str:
    return str(path)


def _torch_available() -> bool:
    try:
        __import__("torch")
    except Exception:
        return False
    return True


def _ensure_processed_inputs(args: argparse.Namespace) -> None:
    features_path = PROJECT_ROOT / args.features
    metadata_path = PROJECT_ROOT / args.metadata
    if features_path.exists() and metadata_path.exists() and not args.rebuild_features:
        print(f"Using existing features: {args.features}")
        print(f"Using existing metadata: {args.metadata}")
        return
    legacy_features = PROJECT_ROOT / "data" / "processed" / "features_mfcc.csv"
    if args.raw_dir is None and legacy_features.exists() and metadata_path.exists() and not args.rebuild_features:
        print(
            "WARNING: extended features are absent and --raw-dir was not provided; "
            "falling back to data/processed/features_mfcc.csv."
        )
        args.features = Path("data/processed/features_mfcc.csv")
        return
    if args.raw_dir is None:
        raise FileNotFoundError(
            "Processed features/metadata are missing. Provide --raw-dir to build them, "
            "or run the preprocessing scripts manually first."
        )

    raw_dir = args.raw_dir
    _run([sys.executable, "scripts/01_build_coswara_index.py", "--raw-dir", _path(raw_dir), "--output", "data/interim/coswara_index.csv"])
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
            "data/processed/metadata_with_quality.csv",
        ]
    )
    extract_cmd = [
        sys.executable,
        "scripts/05_extract_features.py",
        "--metadata",
        "data/processed/metadata_with_quality.csv",
        "--features-output",
        "data/processed/features_mfcc.csv",
    ]
    if args.skip_cnn or not _torch_available():
        extract_cmd.append("--skip-spectrograms")
    _run(extract_cmd)
    _run(
        [
            sys.executable,
            "scripts/49_extract_strong_features.py",
            "--metadata",
            "data/processed/metadata_with_quality.csv",
            "--output",
            "data/processed/features_strong_acoustic.csv",
            "--augment-train-copies",
            str(args.augment_train_copies),
            "--augmentation-seed",
            str(args.augmentation_seed),
        ]
    )


def _run_cnn_branch(args: argparse.Namespace) -> None:
    if args.skip_cnn:
        print("Skipping compact CNN branch because --skip-cnn was provided.")
        return
    if not _torch_available():
        print("Skipping compact CNN branch because torch is not installed in this environment.")
        return
    spectrogram_index = PROJECT_ROOT / "data" / "processed" / "spectrogram_index.csv"
    if not spectrogram_index.exists():
        print("Skipping compact CNN branch because data/processed/spectrogram_index.csv is absent.")
        return
    for architecture in args.cnn_architectures:
        for modality in args.modalities:
            _run(
                [
                    sys.executable,
                    "scripts/07_train_cnn.py",
                    "--spectrogram-index",
                    "data/processed/spectrogram_index.csv",
                    "--modality",
                    str(modality),
                    "--architecture",
                    str(architecture),
                    "--metrics-output",
                    f"data/outputs/metrics/cnn_metrics_{architecture}_{modality}.csv",
                    "--validation-output",
                    f"data/outputs/metrics/cnn_logits_validation_{architecture}_{modality}.csv",
                    "--test-output",
                    f"data/outputs/metrics/cnn_logits_test_{architecture}_{modality}.csv",
                    "--history-output",
                    f"data/outputs/metrics/cnn_training_history_{architecture}_{modality}.csv",
                    "--epochs",
                    str(args.cnn_epochs),
                    "--batch-size",
                    str(args.cnn_batch_size),
                ]
            )


def main() -> None:
    args = parse_args()
    _ensure_processed_inputs(args)

    strong_cmd = [
        sys.executable,
        "scripts/47_run_strong_baseline.py",
        "--features",
        _path(args.features),
        "--metadata",
        _path(args.metadata),
        "--modalities",
        *args.modalities,
        "--model-names",
        *args.model_names,
        "--optuna-trials",
        str(args.optuna_trials),
        "--ensemble-top-k",
        str(args.ensemble_top_k),
    ]
    if args.feature_level_fusion:
        strong_cmd.append("--feature-level-fusion")
    if args.global_stack_top_k:
        strong_cmd.extend(["--global-stack-top-k", str(args.global_stack_top_k)])
    if args.require_quality_ok:
        strong_cmd.append("--require-quality-ok")
    _run(strong_cmd)
    _run_cnn_branch(args)

    _run([sys.executable, "scripts/20_make_paper_tables.py"])
    _run([sys.executable, "scripts/24_make_experiment_manifest.py"])
    core_validation_inputs = [
        PROJECT_ROOT / "data" / "interim" / "coswara_index.csv",
        PROJECT_ROOT / "data" / "processed" / "metadata_with_quality.csv",
    ]
    if not args.skip_validation and all(path.exists() for path in core_validation_inputs):
        _run(
            [
                sys.executable,
                "scripts/12_validate_artifacts.py",
                "--index",
                "data/interim/coswara_index.csv",
                "--metadata",
                "data/processed/metadata_with_quality.csv",
                "--quality",
                "data/processed/audio_quality.csv",
                "--strict",
            ]
        )
    elif not args.skip_validation:
        missing = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in core_validation_inputs if not path.exists())
        print(f"Skipping strict artifact validation because core preprocessing artifacts are absent: {missing}")

    print("\nStrong paper-comparable baseline pipeline completed.")
    print("Primary outputs:")
    print("- data/outputs/metrics/strong_baseline_metrics.csv")
    print("- data/outputs/metrics/strong_baseline_predictions.csv")
    print("- reports/tables/strong_baseline_model_selection.csv")
    print("- reports/tables/strong_baseline_protocol_audit.csv")
    print("- reports/tables/paper_metric_table.csv")


if __name__ == "__main__":
    main()
