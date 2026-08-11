from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from covid_rars.metrics import (
    best_threshold_by_balanced_accuracy,
    binary_metric_bundle,
    labels_to_binary,
)


SOTA_SOURCE_COLS = [
    "evaluation_protocol",
    "analysis_family",
    "model_name",
    "modality",
    "submodality",
    "modality_combination",
    "fusion_method",
]


def _ensure_optional_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    return out


def _source_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_optional_columns(frame, SOTA_SOURCE_COLS)
    out["sota_source_key"] = out[SOTA_SOURCE_COLS].fillna("").astype(str).agg("||".join, axis=1)
    return out


def aggregate_sota_predictions(predictions: pd.DataFrame, level: str = "participant") -> pd.DataFrame:
    """Aggregate segment-level branch predictions to recording or participant level."""
    if predictions.empty:
        return predictions.copy()
    if level not in {"recording", "participant"}:
        raise ValueError(f"Unknown SOTA aggregation level: {level}")
    df = _ensure_optional_columns(predictions, ["dataset", "submodality", "evaluation_protocol"])
    group_cols = [
        "participant_id",
        "label_binary",
        "split",
        "dataset",
        "modality",
        "submodality",
        "model_name",
        "analysis_family",
        "evaluation_protocol",
    ]
    if level == "recording":
        group_cols.insert(1, "recording_id")
    optional = ["modality_combination", "fusion_method"]
    group_cols.extend([col for col in optional if col in df.columns])
    aggregated = (
        df.groupby(group_cols, dropna=False)
        .agg(
            probability=("probability", "mean"),
            n_segments=("probability", "size"),
            n_recordings=("recording_id", "nunique") if "recording_id" in df.columns else ("probability", "size"),
        )
        .reset_index()
    )
    if level == "participant":
        aggregated["recording_id"] = aggregated["participant_id"].astype(str) + "::participant"
    aggregated["aggregation_level"] = level
    return aggregated


def _metric_row(
    frame: pd.DataFrame,
    threshold: float,
    extra: dict[str, object],
) -> dict[str, object]:
    y_true = labels_to_binary(frame["label_binary"])
    y_prob = frame["probability"].astype(float).to_numpy()
    row = binary_metric_bundle(y_true, y_prob, threshold=threshold)
    row.update(extra)
    row["n_participants"] = float(frame["participant_id"].nunique()) if "participant_id" in frame.columns else float(len(frame))
    return row


def evaluate_sota_prediction_table(
    predictions: pd.DataFrame,
    validation_split: str = "validation",
    test_split: str = "test",
) -> pd.DataFrame:
    """Evaluate prediction sources using validation-selected thresholds.

    The validation split is used only for threshold selection. Metrics are then
    reported for validation plus every available non-train evaluation split,
    including external_test when a cross-dataset prediction table is supplied.
    """
    if predictions.empty:
        return pd.DataFrame()
    df = _ensure_optional_columns(
        predictions,
        ["evaluation_protocol", "analysis_family", "model_name", "modality", "submodality"],
    )
    group_cols = [
        "evaluation_protocol",
        "analysis_family",
        "model_name",
        "modality",
        "submodality",
    ]
    for col in ("modality_combination", "fusion_method"):
        if col in df.columns:
            group_cols.append(col)

    rows: list[dict[str, object]] = []
    for group_key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_meta = dict(zip(group_cols, group_key))
        val = group[group["split"].astype(str).eq(validation_split)].copy()
        available_splits = [str(split) for split in group["split"].dropna().astype(str).unique()]
        evaluation_splits: list[str] = []
        for split_name in [validation_split, test_split, *sorted(available_splits)]:
            if split_name == "train" or split_name in evaluation_splits:
                continue
            if split_name in available_splits:
                evaluation_splits.append(split_name)
        if val.empty or val["label_binary"].nunique() < 2 or not evaluation_splits:
            rows.append(
                {
                    **group_meta,
                    "metric_split": "skipped",
                    "skipped": True,
                    "skip_reason": "missing validation/evaluation split or one-class validation labels",
                }
            )
            continue
        threshold = best_threshold_by_balanced_accuracy(
            labels_to_binary(val["label_binary"]),
            val["probability"].astype(float).to_numpy(),
        )
        for split_name in evaluation_splits:
            split_frame = group[group["split"].astype(str).eq(split_name)].copy()
            if split_frame.empty:
                continue
            rows.append(
                _metric_row(
                    split_frame,
                    threshold=threshold,
                    extra={
                        **group_meta,
                        "metric_split": split_name,
                        "threshold_source": "validation_balanced_accuracy",
                        "skipped": False,
                    },
                )
            )
    return pd.DataFrame(rows)


def _participant_matrix(predictions: pd.DataFrame, selected_keys: list[str], split: str) -> pd.DataFrame:
    pred = _source_key(predictions)
    pred = pred[pred["sota_source_key"].isin(selected_keys) & pred["split"].astype(str).eq(split)].copy()
    if pred.empty:
        return pd.DataFrame()
    participant = (
        pred.groupby(["participant_id", "label_binary", "sota_source_key"], dropna=False)
        .agg(probability=("probability", "mean"))
        .reset_index()
    )
    matrix = participant.pivot_table(
        index=["participant_id", "label_binary"],
        columns="sota_source_key",
        values="probability",
        aggfunc="mean",
    ).reset_index()
    matrix.columns.name = None
    matrix["split"] = split
    return matrix


def _participant_fusion_frame(
    source: pd.DataFrame,
    probabilities: np.ndarray,
    split: str,
    fusion_method: str,
    fusion_name: str,
    selected_keys: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recording_id": source["participant_id"].astype(str) + f"::{fusion_name}",
            "participant_id": source["participant_id"].astype(str),
            "dataset": "mixed",
            "modality": "multimodal",
            "submodality": fusion_name,
            "label_binary": source["label_binary"].to_numpy(),
            "split": split,
            "model_name": fusion_name,
            "analysis_family": "sota_prediction_fusion",
            "evaluation_protocol": "sota_internal_protocol",
            "modality_combination": "branch_prediction_stack",
            "fusion_method": fusion_method,
            "probability": probabilities,
            "ensemble_members": ";".join(selected_keys),
        }
    )


def fuse_sota_prediction_sources(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    top_k: int = 8,
    fusion_name: str = "sota_validation_stack",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fuse top validation prediction sources with simple, defensible stackers."""
    if metrics.empty or predictions.empty or top_k < 2:
        return pd.DataFrame(), pd.DataFrame()
    skipped = metrics["skipped"].fillna(False).astype(bool) if "skipped" in metrics.columns else False
    candidates = metrics[metrics["metric_split"].eq("validation") & ~skipped].copy()
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    candidates = _source_key(candidates)
    candidates["auroc"] = pd.to_numeric(candidates["auroc"], errors="coerce")
    candidates["auprc"] = pd.to_numeric(candidates["auprc"], errors="coerce")
    ranked = candidates.sort_values(["auroc", "auprc", "sota_source_key"], ascending=[False, False, True])
    selected_keys = list(dict.fromkeys(ranked["sota_source_key"].dropna().astype(str).head(top_k)))
    if len(selected_keys) < 2:
        return pd.DataFrame(), pd.DataFrame()

    val = _participant_matrix(predictions, selected_keys, split="validation").dropna(subset=selected_keys)
    test = _participant_matrix(predictions, selected_keys, split="test").dropna(subset=selected_keys)
    if val.empty or test.empty or val["label_binary"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()

    val_x = val[selected_keys].to_numpy(dtype=float)
    test_x = test[selected_keys].to_numpy(dtype=float)
    val_y = labels_to_binary(val["label_binary"])
    weight_map = ranked.drop_duplicates("sota_source_key").set_index("sota_source_key")["auroc"].to_dict()
    raw_weights = np.asarray([max(float(weight_map.get(key, 0.5)) - 0.5, 0.01) for key in selected_keys])
    weights = raw_weights / raw_weights.sum()

    fusion_specs: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("top_source_uniform_mean", val_x.mean(axis=1), test_x.mean(axis=1)),
        ("top_source_validation_weighted_auroc", np.average(val_x, axis=1, weights=weights), np.average(test_x, axis=1, weights=weights)),
    ]
    if len(val) >= 8:
        stacker = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=42)
        stacker.fit(val_x, val_y)
        fusion_specs.append(
            (
                "top_source_stacked_logistic_validation",
                stacker.predict_proba(val_x)[:, 1],
                stacker.predict_proba(test_x)[:, 1],
            )
        )

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for method, val_prob, test_prob in fusion_specs:
        threshold = best_threshold_by_balanced_accuracy(val_y, val_prob)
        for split_name, frame, probs in (("validation", val, val_prob), ("test", test, test_prob)):
            pred_frame = _participant_fusion_frame(
                frame,
                probs,
                split=split_name,
                fusion_method=method,
                fusion_name=fusion_name,
                selected_keys=selected_keys,
            )
            prediction_frames.append(pred_frame)
            metric_rows.append(
                _metric_row(
                    pred_frame,
                    threshold=threshold,
                    extra={
                        "evaluation_protocol": "sota_internal_protocol",
                        "analysis_family": "sota_prediction_fusion",
                        "model_name": fusion_name,
                        "modality": "multimodal",
                        "modality_combination": "branch_prediction_stack",
                        "fusion_method": method,
                        "metric_split": split_name,
                        "threshold_source": "validation_balanced_accuracy",
                        "skipped": False,
                        "n_ensemble_models": float(len(selected_keys)),
                        "ensemble_members": ";".join(selected_keys),
                    },
                )
            )
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)
