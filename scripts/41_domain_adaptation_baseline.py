#!/usr/bin/env python
import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from covid_rars.domain_adaptation import run_domain_adaptation_baseline


@dataclass(frozen=True)
class RepresentationRun:
    representation: str
    source_features: Path
    external_features: Path
    feature_shift_report: Path | None = None


DEFAULT_REPRESENTATION_RUNS = [
    RepresentationRun(
        representation="mfcc",
        source_features=Path("data/processed/features_mfcc.csv"),
        external_features=Path("data/processed/coughvid_features_mfcc.csv"),
        feature_shift_report=Path("reports/tables/feature_shift_report.csv"),
    ),
    RepresentationRun(
        representation="opensmile_egemaps",
        source_features=Path("data/processed/features_opensmile_egemaps_coswara_cough.csv"),
        external_features=Path("data/processed/features_opensmile_egemaps_coughvid_cough.csv"),
        feature_shift_report=Path("reports/tables/feature_shift_opensmile_egemaps_cough.csv"),
    ),
    RepresentationRun(
        representation="beats",
        source_features=Path("data/processed/features_beats_coswara_cough.csv"),
        external_features=Path("data/processed/features_beats_coughvid_cough.csv"),
        feature_shift_report=Path("reports/tables/feature_shift_beats_cough.csv"),
    ),
    RepresentationRun(
        representation="panns",
        source_features=Path("data/processed/features_panns_coswara_cough.csv"),
        external_features=Path("data/processed/features_panns_coughvid_cough.csv"),
        feature_shift_report=Path("reports/tables/feature_shift_panns_cough.csv"),
    ),
]


def _resolve_project_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def default_representation_runs(project_root: Path = Path(".")) -> list[RepresentationRun]:
    runs: list[RepresentationRun] = []
    for run in DEFAULT_REPRESENTATION_RUNS:
        source_features = _resolve_project_path(project_root, run.source_features)
        external_features = _resolve_project_path(project_root, run.external_features)
        feature_shift_report = (
            _resolve_project_path(project_root, run.feature_shift_report)
            if run.feature_shift_report is not None
            else None
        )
        if source_features.exists() and external_features.exists():
            runs.append(
                RepresentationRun(
                    representation=run.representation,
                    source_features=source_features,
                    external_features=external_features,
                    feature_shift_report=feature_shift_report,
                )
            )
    return runs


def _requested_representation_run(args: argparse.Namespace) -> RepresentationRun | None:
    if args.source_features is None and args.external_features is None:
        return None
    if args.source_features is None or args.external_features is None:
        raise ValueError("--source-features and --external-features must be supplied together")
    representation = args.representation or "custom"
    return RepresentationRun(
        representation=representation,
        source_features=args.source_features,
        external_features=args.external_features,
        feature_shift_report=args.feature_shift_report,
    )


def _expected_default_paths(project_root: Path) -> str:
    rows = []
    for run in DEFAULT_REPRESENTATION_RUNS:
        rows.append(
            f"{run.representation}: "
            f"{_resolve_project_path(project_root, run.source_features)} and "
            f"{_resolve_project_path(project_root, run.external_features)}"
        )
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source-only versus CORAL unsupervised domain-adaptation baselines for external transfer."
    )
    parser.add_argument("--source-features", type=Path, default=None)
    parser.add_argument("--external-features", type=Path, default=None)
    parser.add_argument("--feature-shift-report", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=["logistic_regression"])
    parser.add_argument("--feature-strategies", nargs="+", default=["all", "drop_high_shift"])
    parser.add_argument("--modality", default="cough")
    parser.add_argument("--source-train-split", default="train")
    parser.add_argument("--smd-threshold", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--representation", default=None)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--regularization", type=float, default=1e-5)
    parser.add_argument("--n-mmd-samples", type=int, default=1000)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("data/outputs/metrics/domain_adaptation_baseline_metrics.csv"),
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data/outputs/metrics/domain_adaptation_baseline_predictions.csv"),
    )
    parser.add_argument(
        "--mmd-output",
        type=Path,
        default=Path("reports/tables/domain_adaptation_mmd.csv"),
    )
    parser.add_argument("--strict-models", action="store_true", help="Fail instead of skipping failed model/strategy runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_run = _requested_representation_run(args)
    runs = [requested_run] if requested_run is not None else default_representation_runs(args.project_root)
    if not runs:
        raise FileNotFoundError(
            "No default representation feature pairs were found. Expected one of:\n"
            f"{_expected_default_paths(args.project_root)}"
        )

    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    mmd_frames: list[pd.DataFrame] = []
    for run in runs:
        source = pd.read_csv(run.source_features)
        external = pd.read_csv(run.external_features)
        shift_report = (
            pd.read_csv(run.feature_shift_report)
            if run.feature_shift_report is not None and run.feature_shift_report.exists()
            else pd.DataFrame()
        )
        for model_name in args.models:
            for strategy in args.feature_strategies:
                try:
                    result = run_domain_adaptation_baseline(
                        source,
                        external,
                        model_name=model_name,
                        feature_strategy=strategy,
                        shift_report=shift_report,
                        modality=args.modality,
                        source_train_split=args.source_train_split,
                        random_state=args.random_state,
                        smd_threshold=args.smd_threshold,
                        representation=run.representation,
                        regularization=args.regularization,
                        n_mmd_samples=args.n_mmd_samples,
                    )
                except (RuntimeError, ValueError) as exc:
                    if args.strict_models:
                        raise
                    print(f"SKIP {run.representation}/{model_name}/{strategy}: {exc}")
                    continue
                metric_frames.append(result.metrics)
                prediction_frames.append(result.predictions)
                mmd_frames.append(result.mmd)
                best = result.metrics.sort_values("auroc", ascending=False).iloc[0]
                print(
                    f"Ran {run.representation}/{model_name}/{strategy}: "
                    f"best={best['adaptation_method']} AUROC={best['auroc']:.4f}, "
                    f"MMD before={best['mmd_before']:.4f}, MMD after={best['mmd_after']:.4f}"
                )

    if not metric_frames:
        raise RuntimeError("No domain-adaptation baseline rows were produced.")

    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    mmd = pd.concat(mmd_frames, ignore_index=True)

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.mmd_output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics_output, index=False)
    predictions.to_csv(args.predictions_output, index=False)
    mmd.to_csv(args.mmd_output, index=False)
    print(f"Wrote domain-adaptation metrics: {args.metrics_output} ({len(metrics)} rows)")
    print(f"Wrote domain-adaptation predictions: {args.predictions_output} ({len(predictions)} rows)")
    print(f"Wrote domain-adaptation MMD table: {args.mmd_output} ({len(mmd)} rows)")


if __name__ == "__main__":
    main()
