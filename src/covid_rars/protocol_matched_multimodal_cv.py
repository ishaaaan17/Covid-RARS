from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from covid_rars.compare_is10_rescue import rank_train_features
from covid_rars.protocol_matched_cv import (
    PROTOCOL_NAME as COUGH_ONLY_PROTOCOL_NAME,
    _has_two_classes,
    _participant_folds,
    _split_audit_row,
)
from covid_rars.strong_baseline import (
    _selected_predictions,
    run_global_prediction_stacker,
    run_strong_fusion,
    train_strong_modality_models,
)


PROTOCOL_NAME = "protocol_matched_multimodal_participant_10fold_cv"


@dataclass(frozen=True)
class ProtocolMatchedMultimodalCVResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_selection: pd.DataFrame
    branch_selection: pd.DataFrame
    split_audit: pd.DataFrame
    summary: pd.DataFrame


def _protocol_frame(frame: pd.DataFrame, *, fold: int, feature_strategy: str, selected_feature_k: int) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    out["evaluation_protocol"] = PROTOCOL_NAME
    out["fold"] = int(fold)
    out["fold_unit"] = "participant"
    out["feature_strategy"] = feature_strategy
    out["selected_feature_k"] = float(selected_feature_k)
    return out


def _assign_fold_splits(
    frame: pd.DataFrame,
    train_ids: set[str],
    validation_ids: set[str],
    test_ids: set[str],
) -> pd.DataFrame:
    out = frame.copy()
    participant_ids = out["participant_id"].astype(str)
    out["split"] = np.select(
        [
            participant_ids.isin(train_ids),
            participant_ids.isin(validation_ids),
            participant_ids.isin(test_ids),
        ],
        ["train", "validation", "test"],
        default="unused",
    )
    return out[out["split"].isin(["train", "validation", "test"])].copy()


def select_fold_feature_columns(
    fold_features: pd.DataFrame,
    *,
    k: int,
    ranker: str,
    selection_scope: str,
    random_state: int,
) -> tuple[list[str], pd.DataFrame]:
    ranking = rank_train_features(
        fold_features,
        ranker=ranker,
        selection_scope=selection_scope,
        random_state=random_state,
    )
    selected = ranking.head(min(int(k), len(ranking)))["feature"].astype(str).tolist()
    return selected, ranking


def aggregate_protocol_matched_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["metric_split"].astype(str).eq("test")].copy()
    if test.empty:
        return pd.DataFrame()

    optional_group_cols = ["modality_combination", "fusion_method", "ensemble_members"]
    if "feature_selection_scope" in test.columns:
        optional_group_cols.append("feature_selection_scope")
    group_cols = [
        "evaluation_protocol",
        "analysis_family",
        "model_name",
        "modality",
        *[col for col in optional_group_cols if col in test.columns],
        "feature_strategy",
        "selected_feature_k",
        "fold_unit",
        "threshold_source",
    ]
    numeric_metrics = [
        "auroc",
        "auprc",
        "balanced_accuracy",
        "f1",
        "sensitivity",
        "specificity",
        "brier",
        "ece",
        "nll",
        "n_participants",
    ]

    rows: list[dict[str, object]] = []
    for key, group in test.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["metric_split"] = "test_aggregate"
        row["fold"] = "aggregate"
        row["n_folds"] = float(group["fold"].nunique())
        row["n_samples"] = float(pd.to_numeric(group.get("n_samples"), errors="coerce").sum())
        for metric in numeric_metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            row[metric] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_protocol_matched_multimodal_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    aggregate = metrics[metrics["metric_split"].astype(str).eq("test_aggregate")].copy()
    if aggregate.empty:
        return pd.DataFrame()
    skipped = (
        aggregate["skipped"].fillna(False).astype(bool)
        if "skipped" in aggregate.columns
        else pd.Series(False, index=aggregate.index)
    )
    aggregate = aggregate[~skipped].copy()
    aggregate["auroc"] = pd.to_numeric(aggregate.get("auroc"), errors="coerce")
    aggregate["auprc"] = pd.to_numeric(aggregate.get("auprc"), errors="coerce")
    aggregate = aggregate.sort_values(
        ["auroc", "auprc", "analysis_family", "model_name"],
        ascending=[False, False, True, True],
    )
    return aggregate.reset_index(drop=True)


def _prepare_features(features: pd.DataFrame, modalities: Iterable[str]) -> pd.DataFrame:
    required = {"recording_id", "participant_id", "modality", "label_binary"}
    missing = required - set(features.columns)
    if missing:
        raise KeyError(f"features missing required columns: {sorted(missing)}")
    modality_set = {str(modality) for modality in modalities}
    df = features[
        features["modality"].astype(str).isin(modality_set)
        & features["label_binary"].isin(["positive", "negative"])
    ].copy()
    if "quality_flag" in df.columns:
        df = df[df["quality_flag"].astype(str).eq("ok")].copy()
    if df.empty or not _has_two_classes(df):
        raise ValueError("No two-class supervised rows are available for the requested modalities")
    return df


def run_protocol_matched_multimodal_cv(
    features: pd.DataFrame,
    *,
    modalities: Iterable[str] = ("cough", "breath", "speech"),
    n_splits: int = 10,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.125,
    top_k_values: Iterable[int] = (800,),
    ranker: str = "lightgbm",
    selection_scope: str = "per_modality_mean",
    model_names: Iterable[str] = ("lightgbm_smote_f80", "svc_rbf_f60", "catboost_smote_f80", "xgboost_smote_f80"),
    random_state: int = 42,
    optuna_trials: int = 0,
    ensemble_top_k: int = 5,
    global_stack_top_k: int = 0,
    feature_strategy_label: str | None = None,
) -> ProtocolMatchedMultimodalCVResult:
    df = _prepare_features(features, modalities)
    modality_list = [str(modality) for modality in modalities]

    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_selection_frames: list[pd.DataFrame] = []
    branch_selection_frames: list[pd.DataFrame] = []
    split_audit_rows: list[dict[str, object]] = []

    for fold_idx, train_ids, validation_ids, test_ids in _participant_folds(
        df,
        n_splits=n_splits,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        random_state=random_state,
    ):
        audit = _split_audit_row(fold_idx, train_ids, validation_ids, test_ids)
        audit["evaluation_protocol"] = PROTOCOL_NAME
        audit["source_protocol_template"] = COUGH_ONLY_PROTOCOL_NAME
        split_audit_rows.append(audit)
        fold_df = _assign_fold_splits(df, train_ids, validation_ids, test_ids)

        for raw_k in top_k_values:
            k = int(raw_k)
            selected_cols, ranking = select_fold_feature_columns(
                fold_df,
                k=k,
                ranker=ranker,
                selection_scope=selection_scope,
                random_state=random_state + fold_idx,
            )
            feature_strategy = feature_strategy_label or f"compare_is10_top{k}_{ranker}_{selection_scope}"
            feature_selection = ranking.copy()
            feature_selection["fold"] = int(fold_idx)
            feature_selection["fold_unit"] = "participant"
            feature_selection["evaluation_protocol"] = PROTOCOL_NAME
            feature_selection["feature_strategy"] = feature_strategy
            feature_selection["feature_selection_scope"] = selection_scope
            feature_selection["selected_feature_k"] = float(k)
            feature_selection["selected"] = feature_selection["feature"].isin(selected_cols)
            feature_selection_frames.append(feature_selection)

            id_cols = [col for col in fold_df.columns if col not in selected_cols and col in {
                "recording_id",
                "participant_id",
                "dataset",
                "modality",
                "submodality",
                "label_binary",
                "quality_flag",
                "split",
            }]
            fold_features = fold_df[[*id_cols, *selected_cols]].copy()
            modality_metrics, modality_predictions, branch_selection, _ = train_strong_modality_models(
                fold_features,
                modalities=modality_list,
                model_names=model_names,
                random_state=random_state + fold_idx,
                optuna_trials=optuna_trials,
                ensemble_top_k=ensemble_top_k,
            )
            selected_predictions = _selected_predictions(modality_predictions, branch_selection)
            fusion_metrics, fusion_predictions = run_strong_fusion(
                selected_predictions,
                branch_selection,
                modalities=modality_list,
            )

            metric_parts = [modality_metrics, fusion_metrics]
            prediction_parts = [modality_predictions, fusion_predictions]
            metrics = pd.concat([part for part in metric_parts if not part.empty], ignore_index=True, sort=False)
            predictions = pd.concat([part for part in prediction_parts if not part.empty], ignore_index=True, sort=False)

            if global_stack_top_k >= 2 and not metrics.empty and not predictions.empty:
                stack_metrics, stack_predictions = run_global_prediction_stacker(
                    metrics,
                    predictions,
                    top_k=global_stack_top_k,
                )
                if not stack_metrics.empty:
                    metrics = pd.concat([metrics, stack_metrics], ignore_index=True, sort=False)
                if not stack_predictions.empty:
                    predictions = pd.concat([predictions, stack_predictions], ignore_index=True, sort=False)

            metrics = _protocol_frame(
                metrics,
                fold=fold_idx,
                feature_strategy=feature_strategy,
                selected_feature_k=k,
            )
            if not metrics.empty:
                metrics["feature_selection_scope"] = selection_scope
            predictions = _protocol_frame(
                predictions,
                fold=fold_idx,
                feature_strategy=feature_strategy,
                selected_feature_k=k,
            )
            if not predictions.empty:
                predictions["feature_selection_scope"] = selection_scope
            if not branch_selection.empty:
                branch_selection = _protocol_frame(
                    branch_selection,
                    fold=fold_idx,
                    feature_strategy=feature_strategy,
                    selected_feature_k=k,
                )
                branch_selection["feature_selection_scope"] = selection_scope
                branch_selection["modality_set"] = "+".join(modality_list)
                branch_selection_frames.append(branch_selection)
            if not metrics.empty:
                metric_frames.append(metrics)
            if not predictions.empty:
                prediction_frames.append(predictions)

    metrics = pd.concat(metric_frames, ignore_index=True, sort=False) if metric_frames else pd.DataFrame()
    aggregate = aggregate_protocol_matched_metrics(metrics)
    if not aggregate.empty:
        metrics = pd.concat([metrics, aggregate], ignore_index=True, sort=False)
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    feature_selection = (
        pd.concat(feature_selection_frames, ignore_index=True, sort=False) if feature_selection_frames else pd.DataFrame()
    )
    branch_selection = (
        pd.concat(branch_selection_frames, ignore_index=True, sort=False) if branch_selection_frames else pd.DataFrame()
    )
    split_audit = pd.DataFrame(split_audit_rows)
    summary = build_protocol_matched_multimodal_summary(metrics)
    return ProtocolMatchedMultimodalCVResult(
        metrics=metrics,
        predictions=predictions,
        feature_selection=feature_selection,
        branch_selection=branch_selection,
        split_audit=split_audit,
        summary=summary,
    )
