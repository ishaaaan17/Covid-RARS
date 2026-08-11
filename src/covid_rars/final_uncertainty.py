from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from covid_rars.calibration_shift import build_calibration_shift_report
from covid_rars.statistics import bootstrap_prediction_table


DEFAULT_FINAL_GROUP_COLUMNS = [
    "prediction_source",
    "evaluation_protocol",
    "analysis_family",
    "model_name",
    "modality",
    "submodality",
    "modality_combination",
    "fusion_method",
    "feature_strategy",
    "selected_feature_k",
    "metric_split",
    "split",
]


def _read_prediction_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"label_binary", "probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required prediction columns: {sorted(missing)}")
    frame = frame.copy()
    frame["prediction_source"] = path.stem
    if "metric_split" not in frame.columns and "split" in frame.columns:
        frame["metric_split"] = frame["split"].astype(str)
    return frame


def load_final_prediction_files(prediction_paths: list[Path]) -> pd.DataFrame:
    frames = [_read_prediction_file(Path(path)) for path in prediction_paths]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError("No non-empty prediction files were supplied")
    return pd.concat(frames, ignore_index=True, sort=False)


def build_final_uncertainty_and_calibration(
    prediction_paths: list[Path],
    group_columns: list[str] | None = None,
    bootstrap_metrics: list[str] | None = None,
    n_bootstraps: int = 1000,
    n_bins: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build bootstrap CI and calibration tables for final validation predictions."""
    predictions = load_final_prediction_files(prediction_paths)
    groups = [col for col in (group_columns or DEFAULT_FINAL_GROUP_COLUMNS) if col in predictions.columns]
    ci = bootstrap_prediction_table(
        predictions,
        metrics=bootstrap_metrics or ["auroc", "auprc", "brier", "ece"],
        group_columns=groups,
        n_bootstraps=n_bootstraps,
        random_state=random_state,
    )
    calibration_summary, calibration_bins = build_calibration_shift_report(
        predictions,
        group_columns=groups,
        n_bins=n_bins,
    )
    return ci, calibration_summary, calibration_bins


def save_calibration_curve_figure(
    calibration_bins: pd.DataFrame,
    output: Path,
    group_columns: list[str] | None = None,
    max_series: int = 12,
) -> None:
    """Save an SVG reliability curve from calibration-bin rows."""
    if calibration_bins.empty:
        raise ValueError("Cannot draw calibration curves from an empty calibration-bin table")
    import matplotlib.pyplot as plt

    bin_columns = {
        "bin_index",
        "probability_lower",
        "probability_upper",
        "n_samples",
        "n_positive",
        "n_negative",
        "mean_probability",
        "observed_positive_rate",
        "calibration_gap",
        "abs_calibration_gap",
    }
    groups = [col for col in (group_columns or DEFAULT_FINAL_GROUP_COLUMNS) if col in calibration_bins.columns]
    groups = [col for col in groups if col not in bin_columns]
    if not groups:
        groups = ["prediction_source"] if "prediction_source" in calibration_bins.columns else []

    grouped = list(calibration_bins.groupby(groups, dropna=False)) if groups else [((), calibration_bins)]
    ranked: list[tuple[float, object, pd.DataFrame]] = []
    for key, frame in grouped:
        n_samples = pd.to_numeric(frame.get("n_samples", 0), errors="coerce").fillna(0).sum()
        ranked.append((float(n_samples), key, frame))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = ranked[: max(1, int(max_series))]

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 5.8))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1.2, label="Perfect calibration")
    for _, key, frame in selected:
        frame = frame.sort_values("mean_probability")
        x = pd.to_numeric(frame["mean_probability"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(frame["observed_positive_rate"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            continue
        if groups:
            if not isinstance(key, tuple):
                key = (key,)
            label_parts = [str(value) for col, value in zip(groups, key) if col in {"model_name", "modality", "split", "metric_split", "fusion_method"}]
            label = " | ".join(label_parts) or " | ".join(str(value) for value in key)
        else:
            label = "model"
        plt.plot(x[finite], y[finite], marker="o", linewidth=1.2, markersize=3.5, label=label[:95])
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title("Final validation calibration curves")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25, linewidth=0.6)
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(output, format="svg")
    plt.close()
