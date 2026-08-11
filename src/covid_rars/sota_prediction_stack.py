from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from covid_rars.metrics import (
    best_threshold_by_balanced_accuracy,
    binary_metric_bundle,
    labels_to_binary,
)


STACK_SOURCE_COLUMNS = [
    "source_run",
    "evaluation_protocol",
    "analysis_family",
    "model_name",
    "modality",
    "feature_strategy",
    "modality_combination",
    "fusion_method",
]


GATED_STACK_METRIC_COLUMNS = [
    "auroc",
    "auprc",
    "balanced_accuracy",
    "f1",
    "sensitivity",
    "specificity",
    "brier",
    "ece",
    "nll",
    "threshold",
    "n_samples",
    "evaluation_protocol",
    "analysis_family",
    "model_name",
    "modality",
    "modality_combination",
    "fusion_method",
    "metric_split",
    "threshold_source",
    "skipped",
    "n_ensemble_models",
    "ensemble_members",
    "n_participants",
]


GATED_STACK_PREDICTION_COLUMNS = [
    "recording_id",
    "participant_id",
    "dataset",
    "modality",
    "submodality",
    "label_binary",
    "split",
    "evaluation_protocol",
    "analysis_family",
    "model_name",
    "modality_combination",
    "fusion_method",
    "probability",
    "ensemble_members",
    "source_run",
]


@dataclass(frozen=True)
class GatedStackResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    candidates: pd.DataFrame


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    return out


def add_stack_source_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_columns(frame, STACK_SOURCE_COLUMNS)
    out["gated_stack_source_key"] = out[STACK_SOURCE_COLUMNS].fillna("").astype(str).agg("||".join, axis=1)
    return out


def build_gated_stack_candidates(
    metrics: pd.DataFrame,
    top_k: int = 16,
    max_validation_drop: float = 0.03,
    min_validation_auroc: float = 0.0,
    min_sources: int = 2,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    skipped = metrics["skipped"].fillna(False).astype(bool) if "skipped" in metrics.columns else False
    candidates = metrics[metrics["metric_split"].astype(str).eq("validation") & ~skipped].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates = add_stack_source_key(candidates)
    candidates["auroc"] = pd.to_numeric(candidates["auroc"], errors="coerce")
    candidates["auprc"] = pd.to_numeric(candidates["auprc"], errors="coerce")
    candidates = candidates.dropna(subset=["auroc"])
    if candidates.empty:
        return pd.DataFrame()
    candidates = candidates.sort_values(
        ["auroc", "auprc", "gated_stack_source_key"],
        ascending=[False, False, True],
    ).drop_duplicates("gated_stack_source_key")
    best_auroc = float(candidates["auroc"].iloc[0])
    floor = max(float(min_validation_auroc), best_auroc - float(max_validation_drop))
    candidates["validation_rank"] = np.arange(1, len(candidates) + 1, dtype=int)
    candidates["validation_floor"] = floor
    candidates["selected"] = (
        (candidates["validation_rank"] <= int(top_k))
        & (candidates["auroc"] >= floor)
    )
    if int(min_sources) > 0 and int(candidates["selected"].sum()) < int(min_sources):
        fallback_idx = candidates.index[: int(min_sources)]
        candidates.loc[fallback_idx, "selected"] = True
    candidates["reject_reason"] = ""
    candidates.loc[~candidates["selected"] & (candidates["validation_rank"] > int(top_k)), "reject_reason"] = "outside_top_k"
    candidates.loc[~candidates["selected"] & (candidates["auroc"] < floor), "reject_reason"] = "below_validation_gate"
    return candidates


def _empty_metrics() -> pd.DataFrame:
    return pd.DataFrame(columns=GATED_STACK_METRIC_COLUMNS)


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=GATED_STACK_PREDICTION_COLUMNS)


def _apply_prediction_availability_gate(candidates: pd.DataFrame, predictions: pd.DataFrame, top_k: int, min_sources: int) -> pd.DataFrame:
    out = candidates.copy()
    pred = add_stack_source_key(predictions)
    if "split" not in pred.columns:
        pred["split"] = ""
    validation_keys = set(pred[pred["split"].astype(str).eq("validation")]["gated_stack_source_key"].astype(str))
    test_keys = set(pred[pred["split"].astype(str).eq("test")]["gated_stack_source_key"].astype(str))
    keys = out["gated_stack_source_key"].astype(str)
    out["has_validation_predictions"] = keys.isin(validation_keys)
    out["has_test_predictions"] = keys.isin(test_keys)
    out["has_predictions"] = out["has_validation_predictions"] & out["has_test_predictions"]

    out["selected"] = False
    out["reject_reason"] = ""
    available_idx = out.index[out["has_predictions"]].tolist()
    out["available_validation_rank"] = np.nan
    if available_idx:
        out.loc[available_idx, "available_validation_rank"] = np.arange(1, len(available_idx) + 1, dtype=int)

    floor = out["validation_floor"].iloc[0] if "validation_floor" in out.columns and len(out) else float("-inf")
    eligible = out[out["has_predictions"] & (out["auroc"] >= floor)]
    selected_idx = eligible.index[: int(top_k)]
    out.loc[selected_idx, "selected"] = True
    if int(min_sources) > 0 and int(out["selected"].sum()) < int(min_sources):
        fallback_idx = out[out["has_predictions"]].index[: int(min_sources)]
        out.loc[fallback_idx, "selected"] = True

    missing_mask = ~out["has_predictions"]
    below_gate_mask = out["has_predictions"] & ~out["selected"] & (out["auroc"] < floor)
    outside_top_k_mask = out["has_predictions"] & ~out["selected"] & (out["auroc"] >= floor)
    out.loc[missing_mask, "reject_reason"] = "missing_predictions"
    out.loc[below_gate_mask, "reject_reason"] = "below_validation_gate"
    out.loc[outside_top_k_mask, "reject_reason"] = "outside_top_k"
    return out


def _participant_matrix(predictions: pd.DataFrame, selected_keys: list[str], split: str) -> pd.DataFrame:
    pred = add_stack_source_key(predictions)
    pred = pred[pred["gated_stack_source_key"].isin(selected_keys) & pred["split"].astype(str).eq(split)].copy()
    if pred.empty:
        return pd.DataFrame()
    participant = (
        pred.groupby(["participant_id", "label_binary", "gated_stack_source_key"], dropna=False)
        .agg(probability=("probability", "mean"))
        .reset_index()
    )
    matrix = participant.pivot_table(
        index=["participant_id", "label_binary"],
        columns="gated_stack_source_key",
        values="probability",
        aggfunc="mean",
    ).reset_index()
    matrix.columns.name = None
    matrix["split"] = split
    return matrix


def _prediction_frame(
    source: pd.DataFrame,
    probabilities: np.ndarray,
    split: str,
    fusion_method: str,
    selected_keys: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recording_id": source["participant_id"].astype(str) + "::sota_gated_stack",
            "participant_id": source["participant_id"].astype(str),
            "dataset": "coswara",
            "modality": "multimodal",
            "submodality": "gated_stack",
            "label_binary": source["label_binary"].to_numpy(),
            "split": split,
            "evaluation_protocol": "sota_gated_internal_protocol",
            "analysis_family": "sota_gated_prediction_stack",
            "model_name": "validation_gated_stack",
            "modality_combination": "all_selected_sources",
            "fusion_method": fusion_method,
            "probability": probabilities,
            "ensemble_members": ";".join(selected_keys),
            "source_run": "sota_gated_stack",
        }
    )


def _metric_row(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float, extra: dict[str, object]) -> dict[str, object]:
    row = binary_metric_bundle(labels_to_binary(frame["label_binary"]), probabilities, threshold=threshold)
    row.update(extra)
    row["n_participants"] = float(frame["participant_id"].nunique())
    return row


def run_gated_prediction_stack(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    top_k: int = 16,
    max_validation_drop: float = 0.03,
    min_validation_auroc: float = 0.0,
    min_sources: int = 2,
) -> GatedStackResult:
    if metrics.empty or predictions.empty:
        return GatedStackResult(_empty_metrics(), _empty_predictions(), pd.DataFrame())
    candidates = build_gated_stack_candidates(
        metrics,
        top_k=top_k,
        max_validation_drop=max_validation_drop,
        min_validation_auroc=min_validation_auroc,
        min_sources=min_sources,
    )
    if candidates.empty:
        return GatedStackResult(_empty_metrics(), _empty_predictions(), candidates)
    candidates = _apply_prediction_availability_gate(candidates, predictions, top_k=top_k, min_sources=min_sources)
    selected_keys = candidates[candidates["selected"]]["gated_stack_source_key"].astype(str).tolist()
    if not selected_keys:
        return GatedStackResult(_empty_metrics(), _empty_predictions(), candidates)

    val_matrix = _participant_matrix(predictions, selected_keys, split="validation")
    test_matrix = _participant_matrix(predictions, selected_keys, split="test")
    available_keys = [
        key for key in selected_keys
        if key in val_matrix.columns and key in test_matrix.columns
    ]
    missing_keys = sorted(set(selected_keys) - set(available_keys))
    if missing_keys:
        missing_mask = candidates["gated_stack_source_key"].astype(str).isin(missing_keys)
        candidates.loc[missing_mask, "selected"] = False
        candidates.loc[missing_mask, "reject_reason"] = "missing_predictions"
    selected_keys = available_keys
    if not selected_keys:
        return GatedStackResult(_empty_metrics(), _empty_predictions(), candidates)

    val = val_matrix.dropna(subset=selected_keys)
    test = test_matrix.dropna(subset=selected_keys)
    if val.empty or test.empty or val["label_binary"].nunique() < 2:
        return GatedStackResult(_empty_metrics(), _empty_predictions(), candidates)

    val_x = val[selected_keys].to_numpy(dtype=float)
    test_x = test[selected_keys].to_numpy(dtype=float)
    val_y = labels_to_binary(val["label_binary"])

    fusion_specs: list[tuple[str, np.ndarray, np.ndarray]] = []
    if len(selected_keys) == 1:
        fusion_specs.append(("single_best_validation_source", val_x[:, 0], test_x[:, 0]))
    else:
        weight_map = candidates.drop_duplicates("gated_stack_source_key").set_index("gated_stack_source_key")["auroc"].to_dict()
        raw_weights = np.asarray([max(float(weight_map.get(key, 0.5)) - 0.5, 0.01) for key in selected_keys])
        weights = raw_weights / raw_weights.sum()
        fusion_specs.extend(
            [
                ("gated_uniform_mean", val_x.mean(axis=1), test_x.mean(axis=1)),
                ("gated_validation_weighted_auroc", np.average(val_x, axis=1, weights=weights), np.average(test_x, axis=1, weights=weights)),
            ]
        )
        if len(val) >= 8:
            stacker = LogisticRegression(C=0.03, class_weight="balanced", max_iter=2000, random_state=42)
            stacker.fit(val_x, val_y)
            fusion_specs.append(
                (
                    "gated_stacked_logistic_validation",
                    stacker.predict_proba(val_x)[:, 1],
                    stacker.predict_proba(test_x)[:, 1],
                )
            )

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for method, val_prob, test_prob in fusion_specs:
        threshold = best_threshold_by_balanced_accuracy(val_y, val_prob)
        for split_name, frame, probs in (("validation", val, val_prob), ("test", test, test_prob)):
            pred_frame = _prediction_frame(frame, probs, split_name, method, selected_keys)
            prediction_frames.append(pred_frame)
            metric_rows.append(
                _metric_row(
                    frame,
                    probs,
                    threshold,
                    {
                        "evaluation_protocol": "sota_gated_internal_protocol",
                        "analysis_family": "sota_gated_prediction_stack",
                        "model_name": "validation_gated_stack",
                        "modality": "multimodal",
                        "modality_combination": "all_selected_sources",
                        "fusion_method": method,
                        "metric_split": split_name,
                        "threshold_source": "validation_balanced_accuracy",
                        "skipped": False,
                        "n_ensemble_models": float(len(selected_keys)),
                        "ensemble_members": ";".join(selected_keys),
                    },
                )
            )
    return GatedStackResult(
        metrics=pd.DataFrame(metric_rows, columns=GATED_STACK_METRIC_COLUMNS),
        predictions=pd.concat(prediction_frames, ignore_index=True) if prediction_frames else _empty_predictions(),
        candidates=candidates,
    )
