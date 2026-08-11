from __future__ import annotations

import numpy as np
import pandas as pd

from covid_rars.calibration_shift import calibration_summary
from covid_rars.metrics import binary_metric_bundle, labels_to_binary


def _logit(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def intercept_recalibrate_probabilities(
    probabilities: np.ndarray | pd.Series,
    source_prevalence: float,
    target_prevalence: float,
) -> np.ndarray:
    """Apply prevalence-shift intercept correction without changing ranking."""
    source = float(np.clip(source_prevalence, 1e-6, 1.0 - 1e-6))
    target = float(np.clip(target_prevalence, 1e-6, 1.0 - 1e-6))
    offset = float(_logit(np.asarray([target]))[0] - _logit(np.asarray([source]))[0])
    return _sigmoid(_logit(np.asarray(probabilities, dtype=float)) + offset)


def _summary_row(
    group: pd.DataFrame,
    probabilities: np.ndarray,
    method: str,
    threshold: float,
    n_bins: int,
) -> dict[str, object]:
    labels = labels_to_binary(group["label_binary"])
    summary = calibration_summary(
        pd.DataFrame({"label_binary": group["label_binary"].to_numpy(), "probability": probabilities}),
        n_bins=n_bins,
    )
    metrics = binary_metric_bundle(labels, probabilities, threshold=threshold)
    row: dict[str, object] = {
        "recalibration_method": method,
        **summary,
        "abs_calibration_gap": abs(float(summary["calibration_gap"])),
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "f1": metrics["f1"],
        "threshold": float(threshold),
    }
    return row


def build_prevalence_recalibration_report(
    predictions: pd.DataFrame,
    group_columns: list[str] | None = None,
    probability_column: str = "probability",
    label_column: str = "label_binary",
    n_bins: int = 10,
    threshold: float = 0.5,
    target_prevalence: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {probability_column, label_column}
    if not required.issubset(predictions.columns):
        raise KeyError(f"Predictions missing required columns: {sorted(required - set(predictions.columns))}")
    frame = predictions[predictions[label_column].isin(["positive", "negative"])].copy()
    frame[probability_column] = pd.to_numeric(frame[probability_column], errors="coerce")
    frame = frame[np.isfinite(frame[probability_column])].copy()
    frame[probability_column] = frame[probability_column].clip(0.0, 1.0)
    if frame.empty:
        raise ValueError("No labeled finite prediction rows available for prevalence recalibration")

    group_columns = [col for col in (group_columns or []) if col in frame.columns]
    iterator = frame.groupby(group_columns, dropna=False) if group_columns else [((), frame)]
    summary_rows: list[dict[str, object]] = []
    recalibrated_frames: list[pd.DataFrame] = []
    for group_key, group in iterator:
        group_values: dict[str, object] = {}
        if group_columns:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_values = dict(zip(group_columns, group_key))
        original_prob = group[probability_column].to_numpy(dtype=float)
        observed_prevalence = float(labels_to_binary(group[label_column]).mean())
        source_prevalence = float(np.mean(original_prob))
        effective_target = observed_prevalence if target_prevalence is None else float(target_prevalence)
        corrected_prob = intercept_recalibrate_probabilities(
            original_prob,
            source_prevalence=source_prevalence,
            target_prevalence=effective_target,
        )

        original_row = _summary_row(group, original_prob, "source_calibrated", threshold=threshold, n_bins=n_bins)
        corrected_row = _summary_row(group, corrected_prob, "target_prevalence_intercept", threshold=threshold, n_bins=n_bins)
        for row in [original_row, corrected_row]:
            row.update(group_values)
            row["source_prevalence_for_offset"] = source_prevalence
            row["target_prevalence_for_offset"] = effective_target
            row["prevalence_offset"] = float(_logit(np.asarray([effective_target]))[0] - _logit(np.asarray([source_prevalence]))[0])
            summary_rows.append(row)

        out = group.copy()
        out["original_probability"] = original_prob
        out["prevalence_recalibrated_probability"] = corrected_prob
        out["source_prevalence_for_offset"] = source_prevalence
        out["target_prevalence_for_offset"] = effective_target
        recalibrated_frames.append(out)

    return (
        pd.DataFrame(summary_rows),
        pd.concat(recalibrated_frames, ignore_index=True, sort=False) if recalibrated_frames else pd.DataFrame(),
    )
