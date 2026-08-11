from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from covid_rars.clinical_operating_points import build_clinical_operating_points
from covid_rars.metrics import binary_metric_bundle, labels_to_binary
from covid_rars.statistics import _metric_value


LOWER_IS_BETTER_METRICS = {"brier", "ece", "nll"}


def _clean_predictions(
    predictions: pd.DataFrame,
    probability_column: str = "probability",
    label_column: str = "label_binary",
) -> pd.DataFrame:
    required = {probability_column, label_column}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"Predictions missing required columns: {sorted(missing)}")
    frame = predictions[predictions[label_column].isin(["positive", "negative"])].copy()
    frame[probability_column] = pd.to_numeric(frame[probability_column], errors="coerce")
    frame = frame[np.isfinite(frame[probability_column])].copy()
    frame[probability_column] = frame[probability_column].clip(0.0, 1.0)
    return frame


def _available_group_columns(frame: pd.DataFrame, group_columns: Iterable[str] | None) -> list[str]:
    return [col for col in (group_columns or []) if col in frame.columns]


def _iter_groups(frame: pd.DataFrame, group_columns: list[str]) -> Iterable[tuple[dict[str, object], pd.DataFrame]]:
    if not group_columns:
        yield {}, frame
        return
    for key, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        yield dict(zip(group_columns, key)), group


def build_shuffle_label_sanity(
    predictions: pd.DataFrame,
    group_columns: list[str] | None = None,
    metrics: list[str] | None = None,
    probability_column: str = "probability",
    label_column: str = "label_binary",
    n_permutations: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation-label sanity table for already materialized prediction groups.

    This does not replace a full retrain-with-shuffled-labels experiment, but it is
    a cheap leakage smoke test: a legitimate prediction table should lose ranking
    power when labels are randomly reassigned within the same endpoint.
    """
    frame = _clean_predictions(predictions, probability_column=probability_column, label_column=label_column)
    groups = _available_group_columns(frame, group_columns)
    metric_names = metrics or ["auroc", "auprc"]
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    for group_values, group in _iter_groups(frame, groups):
        if group[label_column].nunique() < 2:
            continue
        y_true = labels_to_binary(group[label_column])
        y_prob = group[probability_column].astype(float).to_numpy()
        for metric in metric_names:
            observed = _metric_value(y_true, y_prob, metric)
            values: list[float] = []
            for _ in range(max(1, int(n_permutations))):
                y_perm = rng.permutation(y_true)
                if len(np.unique(y_perm)) < 2:
                    continue
                value = _metric_value(y_perm, y_prob, metric)
                if np.isfinite(value):
                    values.append(float(value))
            arr = np.asarray(values, dtype=float)
            if arr.size:
                ci_low, ci_high = np.quantile(arr, [0.025, 0.975])
                if metric in LOWER_IS_BETTER_METRICS:
                    p_value = float((np.sum(arr <= observed) + 1) / (arr.size + 1))
                else:
                    p_value = float((np.sum(arr >= observed) + 1) / (arr.size + 1))
                permuted_mean = float(arr.mean())
            else:
                ci_low = ci_high = p_value = permuted_mean = float("nan")
            rows.append(
                {
                    **group_values,
                    "metric": metric,
                    "observed": float(observed),
                    "permuted_mean": permuted_mean,
                    "permuted_ci_low": float(ci_low),
                    "permuted_ci_high": float(ci_high),
                    "permutation_p_value": p_value,
                    "n_permutations": int(arr.size),
                    "n_samples": int(len(group)),
                    "n_positive": int(np.sum(y_true == 1)),
                    "n_negative": int(np.sum(y_true == 0)),
                    "sanity_check": "prediction_label_permutation",
                }
            )
    return pd.DataFrame(rows)


def build_fixed_sensitivity_table(
    predictions: pd.DataFrame,
    group_columns: list[str] | None = None,
    target_sensitivities: list[float] | None = None,
    probability_column: str = "probability",
    label_column: str = "label_binary",
) -> pd.DataFrame:
    frame = _clean_predictions(predictions, probability_column=probability_column, label_column=label_column)
    return build_clinical_operating_points(
        frame,
        group_columns=_available_group_columns(frame, group_columns),
        probability_column=probability_column,
        label_column=label_column,
        target_specificities=[],
        target_sensitivities=target_sensitivities or [0.90],
    )


def _correlation(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray, method: str = "pearson") -> float:
    l = pd.Series(left, dtype="float64")
    r = pd.Series(right, dtype="float64")
    mask = np.isfinite(l.to_numpy()) & np.isfinite(r.to_numpy())
    if int(mask.sum()) < 2:
        return float("nan")
    return float(l[mask].corr(r[mask], method=method))


def _participant_prediction_frame(
    frame: pd.DataFrame,
    group_columns: list[str],
    probability_column: str,
    label_column: str,
) -> pd.DataFrame:
    id_columns = [col for col in [*group_columns, "participant_id"] if col in frame.columns]
    if "participant_id" not in id_columns:
        raise KeyError("participant_id is required for residual-correlation analysis")
    out = (
        frame.groupby(id_columns, dropna=False)
        .agg(
            label_binary=(label_column, "first"),
            probability=(probability_column, "mean"),
        )
        .reset_index()
    )
    out["label_int"] = labels_to_binary(out["label_binary"])
    out["residual"] = out["label_int"] - out["probability"].astype(float)
    out["abs_error"] = out["residual"].abs()
    return out


def build_audio_metadata_residual_correlation(
    audio_predictions: pd.DataFrame,
    metadata_predictions: pd.DataFrame,
    group_columns: list[str] | None = None,
    probability_column: str = "probability",
    label_column: str = "label_binary",
) -> pd.DataFrame:
    """Align audio and metadata predictions and quantify shortcut co-movement."""
    audio = _clean_predictions(audio_predictions, probability_column=probability_column, label_column=label_column)
    metadata = _clean_predictions(metadata_predictions, probability_column=probability_column, label_column=label_column)
    groups = [col for col in (group_columns or []) if col in audio.columns and col in metadata.columns]
    audio_p = _participant_prediction_frame(audio, groups, probability_column, label_column)
    metadata_p = _participant_prediction_frame(metadata, groups, probability_column, label_column)
    merge_cols = [*groups, "participant_id"]
    merged = audio_p.merge(metadata_p, on=merge_cols, suffixes=("_audio", "_metadata"), how="inner")
    if merged.empty:
        return pd.DataFrame()
    if "label_binary_audio" in merged.columns and "label_binary_metadata" in merged.columns:
        merged = merged[merged["label_binary_audio"].astype(str).eq(merged["label_binary_metadata"].astype(str))].copy()
    rows: list[dict[str, object]] = []
    for group_values, group in _iter_groups(merged, groups):
        if group.empty:
            continue
        row = {
            **group_values,
            "n_aligned": int(len(group)),
            "probability_pearson": _correlation(group["probability_audio"], group["probability_metadata"]),
            "probability_spearman": _correlation(group["probability_audio"], group["probability_metadata"], method="spearman"),
            "residual_pearson": _correlation(group["residual_audio"], group["residual_metadata"]),
            "residual_spearman": _correlation(group["residual_audio"], group["residual_metadata"], method="spearman"),
            "audio_abs_error_vs_metadata_probability_pearson": _correlation(
                group["abs_error_audio"],
                group["probability_metadata"],
            ),
            "audio_abs_error_vs_metadata_probability_spearman": _correlation(
                group["abs_error_audio"],
                group["probability_metadata"],
                method="spearman",
            ),
            "mean_audio_probability": float(group["probability_audio"].mean()),
            "mean_metadata_probability": float(group["probability_metadata"].mean()),
            "mean_audio_abs_error": float(group["abs_error_audio"].mean()),
            "mean_metadata_abs_error": float(group["abs_error_metadata"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _logit(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _evaluate_recalibrated_group(
    group: pd.DataFrame,
    probabilities: np.ndarray,
    method: str,
    group_values: dict[str, object],
    n_calibration: int,
) -> dict[str, object]:
    metrics = binary_metric_bundle(labels_to_binary(group["label_binary"]), probabilities, threshold=0.5)
    return {
        **group_values,
        **metrics,
        "recalibration_method": method,
        "metric_split": "target_domain_recalibration_evaluation",
        "n_calibration": int(n_calibration),
        "n_evaluation": int(len(group)),
    }


def build_partial_target_recalibration(
    predictions: pd.DataFrame,
    group_columns: list[str] | None = None,
    probability_column: str = "probability",
    label_column: str = "label_binary",
    calibration_fraction: float = 0.25,
    random_state: int = 42,
    methods: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit Platt/isotonic calibration on a small target-domain slice and evaluate the rest."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")
    methods = methods or ["platt", "isotonic"]
    frame = _clean_predictions(predictions, probability_column=probability_column, label_column=label_column)
    groups = _available_group_columns(frame, group_columns)
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for group_values, group in _iter_groups(frame, groups):
        if group[label_column].nunique() < 2 or len(group) < 8:
            continue
        y = labels_to_binary(group[label_column])
        stratify = y if np.min(np.bincount(y)) >= 2 else None
        calibration_idx, evaluation_idx = train_test_split(
            np.arange(len(group)),
            train_size=calibration_fraction,
            random_state=random_state,
            stratify=stratify,
        )
        calibration = group.iloc[calibration_idx].copy()
        evaluation = group.iloc[evaluation_idx].copy()
        if calibration[label_column].nunique() < 2 or evaluation[label_column].nunique() < 2:
            continue

        cal_prob = calibration[probability_column].astype(float).to_numpy()
        eval_prob = evaluation[probability_column].astype(float).to_numpy()
        cal_y = labels_to_binary(calibration[label_column])
        metric_rows.append(
            _evaluate_recalibrated_group(
                evaluation,
                eval_prob,
                "original",
                group_values,
                n_calibration=len(calibration),
            )
        )
        original_eval = evaluation.copy()
        original_eval["original_probability"] = eval_prob
        original_eval["probability"] = eval_prob
        original_eval["recalibration_method"] = "original"
        prediction_frames.append(original_eval)

        if "platt" in methods:
            model = LogisticRegression(max_iter=1000)
            model.fit(_logit(cal_prob).reshape(-1, 1), cal_y)
            platt_prob = model.predict_proba(_logit(eval_prob).reshape(-1, 1))[:, 1]
            metric_rows.append(
                _evaluate_recalibrated_group(evaluation, platt_prob, "platt", group_values, n_calibration=len(calibration))
            )
            out = evaluation.copy()
            out["original_probability"] = eval_prob
            out["probability"] = platt_prob
            out["recalibration_method"] = "platt"
            prediction_frames.append(out)

        if "isotonic" in methods:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(cal_prob, cal_y)
            isotonic_prob = iso.predict(eval_prob)
            metric_rows.append(
                _evaluate_recalibrated_group(
                    evaluation,
                    isotonic_prob,
                    "isotonic",
                    group_values,
                    n_calibration=len(calibration),
                )
            )
            out = evaluation.copy()
            out["original_probability"] = eval_prob
            out["probability"] = isotonic_prob
            out["recalibration_method"] = "isotonic"
            prediction_frames.append(out)

    metrics = pd.DataFrame(metric_rows)
    recalibrated = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    return metrics, recalibrated
