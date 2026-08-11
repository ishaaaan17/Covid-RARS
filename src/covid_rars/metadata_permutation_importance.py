from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from covid_rars.metadata_confounding import (
    build_audit_feature_frame,
    feature_group_for_column,
)
from covid_rars.metrics import best_threshold_by_balanced_accuracy, binary_metric_bundle, labels_to_binary


@dataclass(frozen=True)
class MetadataPermutationImportanceResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    importance: pd.DataFrame
    group_summary: pd.DataFrame


def _align_feature_frames(frames: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], list[str]]:
    columns = sorted(set().union(*(frame.columns for frame in frames)))
    aligned = [frame.reindex(columns=columns, fill_value=0.0) for frame in frames]
    return aligned, columns


def _clean_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"participant_id", "label_binary", "split"}
    missing = required - set(metadata.columns)
    if missing:
        raise KeyError(f"metadata missing required columns: {sorted(missing)}")
    return metadata[
        metadata["label_binary"].isin(["positive", "negative"])
        & metadata["split"].isin(["train", "validation", "test"])
    ].copy()


def _prediction_frame(test: pd.DataFrame, probabilities: np.ndarray, feature_set: str, threshold: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": test["participant_id"].astype(str).to_numpy(),
            "recording_id": test.get("recording_id", pd.Series([""] * len(test), index=test.index)).astype(str).to_numpy(),
            "label_binary": test["label_binary"].to_numpy(),
            "split": "test",
            "analysis_family": "metadata_confounding",
            "model_name": "metadata_confounding_logistic_regression",
            "audit_model": feature_set,
            "feature_strategy": feature_set,
            "probability": probabilities,
            "threshold": float(threshold),
        }
    )


def _fit_metadata_model(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    random_state: int,
) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=3000, random_state=random_state),
            ),
        ]
    )
    model.fit(train_x, train_y)
    return model


def _summarize_groups(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame()
    grouped = (
        importance.groupby(["audit_model", "feature_group"], dropna=False)
        .agg(
            n_features=("feature", "count"),
            importance_mean_sum=("importance_mean", "sum"),
            importance_mean_max=("importance_mean", "max"),
            importance_std_mean=("importance_std", "mean"),
        )
        .reset_index()
    )
    totals = grouped.groupby("audit_model")["importance_mean_sum"].transform("sum").replace(0.0, np.nan)
    grouped["importance_share"] = (grouped["importance_mean_sum"] / totals).fillna(0.0)
    top = (
        importance.sort_values(["audit_model", "feature_group", "importance_mean"], ascending=[True, True, False])
        .groupby(["audit_model", "feature_group"], dropna=False)
        .head(1)[["audit_model", "feature_group", "feature", "importance_mean"]]
        .rename(columns={"feature": "top_feature", "importance_mean": "top_feature_importance_mean"})
    )
    return grouped.merge(top, on=["audit_model", "feature_group"], how="left").sort_values(
        ["audit_model", "importance_mean_sum"], ascending=[True, False]
    )


def _run_one_feature_set(
    metadata: pd.DataFrame,
    feature_set: str,
    n_repeats: int,
    random_state: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    train = metadata[metadata["split"].eq("train")].copy()
    validation = metadata[metadata["split"].eq("validation")].copy()
    test = metadata[metadata["split"].eq("test")].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Need train, validation, and test rows for metadata permutation importance")
    if train["label_binary"].nunique() < 2 or validation["label_binary"].nunique() < 2 or test["label_binary"].nunique() < 2:
        raise ValueError("Each split must contain positive and negative labels for metadata permutation importance")

    train_raw, train_groups = build_audit_feature_frame(train, feature_set=feature_set)
    validation_raw, validation_groups = build_audit_feature_frame(validation, feature_set=feature_set)
    test_raw, test_groups = build_audit_feature_frame(test, feature_set=feature_set)
    groups = {**test_groups, **validation_groups, **train_groups}
    (train_x, validation_x, test_x), columns = _align_feature_frames([train_raw, validation_raw, test_raw])
    varying_columns = [col for col in columns if train_x[col].nunique(dropna=False) > 1]
    if not varying_columns:
        raise ValueError(f"Feature set {feature_set} selected no train-varying columns")

    train_x = train_x[varying_columns]
    validation_x = validation_x[varying_columns]
    test_x = test_x[varying_columns]
    model = _fit_metadata_model(train_x, labels_to_binary(train["label_binary"]), random_state=random_state)
    validation_prob = model.predict_proba(validation_x)[:, 1]
    threshold = best_threshold_by_balanced_accuracy(labels_to_binary(validation["label_binary"]), validation_prob)
    test_prob = model.predict_proba(test_x)[:, 1]
    metrics = binary_metric_bundle(labels_to_binary(test["label_binary"]), test_prob, threshold=threshold)
    metrics.update(
        {
            "analysis_family": "metadata_confounding",
            "model_name": "metadata_confounding_logistic_regression",
            "audit_model": feature_set,
            "feature_strategy": feature_set,
            "modality": "metadata",
            "metric_split": "test",
            "n_features": float(len(varying_columns)),
            "n_permutation_repeats": int(n_repeats),
        }
    )

    permutation = permutation_importance(
        model,
        test_x,
        labels_to_binary(test["label_binary"]),
        n_repeats=max(1, int(n_repeats)),
        random_state=random_state,
        scoring="roc_auc",
    )
    importance = pd.DataFrame(
        {
            "audit_model": feature_set,
            "feature": varying_columns,
            "feature_group": [groups.get(col, feature_group_for_column(col)) for col in varying_columns],
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
            "n_repeats": int(n_repeats),
            "scoring": "roc_auc",
        }
    ).sort_values(["importance_mean", "feature"], ascending=[False, True])

    predictions = _prediction_frame(test, test_prob, feature_set=feature_set, threshold=threshold)
    return metrics, predictions, importance.reset_index(drop=True)


def run_metadata_permutation_importance(
    metadata: pd.DataFrame,
    feature_sets: list[str] | None = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> MetadataPermutationImportanceResult:
    clean = _clean_metadata(metadata)
    feature_sets = feature_sets or ["full_safe_metadata"]
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    for feature_set in feature_sets:
        metrics, predictions, importance = _run_one_feature_set(
            clean,
            feature_set=feature_set,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        metric_rows.append(metrics)
        prediction_frames.append(predictions)
        importance_frames.append(importance)
    importance_table = pd.concat(importance_frames, ignore_index=True, sort=False) if importance_frames else pd.DataFrame()
    return MetadataPermutationImportanceResult(
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame(),
        importance=importance_table,
        group_summary=_summarize_groups(importance_table),
    )
