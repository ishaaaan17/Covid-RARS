from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from covid_rars.compare_is10_rescue import rank_train_features
from covid_rars.features import feature_columns
from covid_rars.metrics import (
    best_threshold_by_balanced_accuracy,
    binary_metric_bundle,
    labels_to_binary,
)
from covid_rars.strong_baseline import (
    DEFAULT_MODEL_NAMES,
    _fit_optuna_model,
    _make_model,
    _predict_probability,
)


@dataclass(frozen=True)
class PaperComparableCVResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_selection: pd.DataFrame


def _has_two_classes(frame: pd.DataFrame) -> bool:
    return frame["label_binary"].isin(["positive", "negative"]).all() and frame["label_binary"].nunique() == 2


def select_fold_feature_columns(
    fold_features: pd.DataFrame,
    *,
    k: int,
    ranker: str = "lightgbm",
    random_state: int = 42,
) -> tuple[list[str], pd.DataFrame]:
    ranking = rank_train_features(
        fold_features,
        ranker=ranker,
        selection_scope="global",
        random_state=random_state,
    )
    selected = ranking.head(min(int(k), len(ranking)))["feature"].astype(str).tolist()
    return selected, ranking


def _inner_validation_split(
    train_validation: pd.DataFrame,
    *,
    validation_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = labels_to_binary(train_validation["label_binary"])
    if len(np.unique(y)) < 2:
        raise ValueError("Training fold does not contain both classes")
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(validation_fraction),
        random_state=random_state,
    )
    train_idx, validation_idx = next(splitter.split(np.zeros(len(y)), y))
    return train_idx, validation_idx


def _metric_row(
    labels: pd.Series,
    probabilities: np.ndarray,
    *,
    threshold: float,
    extra: dict[str, object],
) -> dict[str, object]:
    row = binary_metric_bundle(labels_to_binary(labels), probabilities, threshold=threshold)
    row.update(extra)
    return row


def _prediction_frame(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    split: str,
    fold: int,
    model_name: str,
    feature_strategy: str,
    selected_feature_k: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recording_id": frame["recording_id"].astype(str).to_numpy(),
            "participant_id": frame["participant_id"].astype(str).to_numpy(),
            "dataset": frame.get("dataset", pd.Series(["coswara"] * len(frame))).to_numpy(),
            "modality": frame["modality"].astype(str).to_numpy(),
            "submodality": frame.get("submodality", pd.Series(["unknown"] * len(frame))).to_numpy(),
            "label_binary": frame["label_binary"].to_numpy(),
            "split": split,
            "fold": int(fold),
            "fold_unit": "recording",
            "evaluation_protocol": "paper_comparable_10fold_cv",
            "analysis_family": "paper_comparable_cv",
            "model_name": model_name,
            "feature_strategy": feature_strategy,
            "selected_feature_k": float(selected_feature_k),
            "probability": probabilities,
        }
    )


def _fit_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
    *,
    random_state: int,
    optuna_trials: int,
):
    if model_name == "optuna_validation_search":
        return _fit_optuna_model(
            x_train,
            y_train,
            x_validation,
            y_validation,
            random_state=random_state,
            n_trials=optuna_trials,
        )
    return _make_model(model_name, random_state=random_state)


def _aggregate_test_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["metric_split"].eq("test")].copy()
    if test.empty:
        return pd.DataFrame()

    group_cols = [
        "evaluation_protocol",
        "analysis_family",
        "model_name",
        "modality",
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
    ]
    rows: list[dict[str, object]] = []
    for key, group in test.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row["metric_split"] = "test_aggregate"
        row["fold"] = "aggregate"
        row["n_folds"] = float(group["fold"].nunique())
        row["n_samples"] = float(group["n_samples"].sum())
        for metric in numeric_metrics:
            if metric in group.columns:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[metric] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run_paper_comparable_cv(
    features: pd.DataFrame,
    *,
    modality: str = "cough",
    n_splits: int = 10,
    validation_fraction: float = 0.2,
    top_k_values: Iterable[int] = (500, 800, 1200),
    ranker: str = "lightgbm",
    model_names: Iterable[str] = DEFAULT_MODEL_NAMES,
    random_state: int = 42,
    optuna_trials: int = 25,
) -> PaperComparableCVResult:
    df = features[
        features["modality"].astype(str).eq(str(modality))
        & features["label_binary"].isin(["positive", "negative"])
    ].copy()
    if "quality_flag" in df.columns:
        df = df[df["quality_flag"].astype(str).eq("ok")].copy()
    if df.empty or not _has_two_classes(df):
        raise ValueError(f"No two-class supervised rows are available for modality={modality}")

    y = labels_to_binary(df["label_binary"])
    class_counts = np.bincount(y, minlength=2)
    effective_splits = min(int(n_splits), int(class_counts.min()))
    if effective_splits < 2:
        raise ValueError("At least two positive and two negative rows are required for cross-validation")

    splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    selection_rows: list[pd.DataFrame] = []

    for fold_idx, (train_validation_idx, test_idx) in enumerate(splitter.split(np.zeros(len(df)), y), start=1):
        train_validation = df.iloc[train_validation_idx].copy()
        test = df.iloc[test_idx].copy()
        inner_train_idx, inner_validation_idx = _inner_validation_split(
            train_validation,
            validation_fraction=validation_fraction,
            random_state=random_state + fold_idx,
        )
        train = train_validation.iloc[inner_train_idx].copy()
        validation = train_validation.iloc[inner_validation_idx].copy()
        fold_df = pd.concat(
            [
                train.assign(split="train"),
                validation.assign(split="validation"),
                test.assign(split="test"),
            ],
            ignore_index=True,
            sort=False,
        )

        for k in top_k_values:
            selected_cols, ranking = select_fold_feature_columns(
                fold_df,
                k=int(k),
                ranker=ranker,
                random_state=random_state + fold_idx,
            )
            feature_strategy = f"compare_is10_top{int(k)}_{ranker}"
            ranking = ranking.copy()
            ranking["fold"] = int(fold_idx)
            ranking["feature_strategy"] = feature_strategy
            ranking["selected_feature_k"] = float(k)
            ranking["selected"] = ranking["feature"].isin(selected_cols)
            selection_rows.append(ranking)

            x_train = train[selected_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            y_train = labels_to_binary(train["label_binary"])
            x_validation = validation[selected_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            y_validation = labels_to_binary(validation["label_binary"])
            x_test = test[selected_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

            for model_name in model_names:
                try:
                    model = _fit_model(
                        str(model_name),
                        x_train,
                        y_train,
                        x_validation,
                        y_validation,
                        random_state=random_state + fold_idx,
                        optuna_trials=optuna_trials,
                    )
                    model.fit(x_train, y_train)
                    val_prob = _predict_probability(model, x_validation)
                    test_prob = _predict_probability(model, x_test)
                except Exception as exc:
                    metric_rows.append(
                        {
                            "evaluation_protocol": "paper_comparable_10fold_cv",
                            "analysis_family": "paper_comparable_cv",
                            "model_name": str(model_name),
                            "modality": modality,
                            "feature_strategy": feature_strategy,
                            "selected_feature_k": float(k),
                            "fold": int(fold_idx),
                            "fold_unit": "recording",
                            "metric_split": "skipped",
                            "skipped": True,
                            "skip_reason": str(exc),
                        }
                    )
                    continue

                threshold = best_threshold_by_balanced_accuracy(y_validation, val_prob)
                for split_name, split_frame, probs in (
                    ("validation", validation, val_prob),
                    ("test", test, test_prob),
                ):
                    prediction_frames.append(
                        _prediction_frame(
                            split_frame,
                            probs,
                            split=split_name,
                            fold=fold_idx,
                            model_name=str(model_name),
                            feature_strategy=feature_strategy,
                            selected_feature_k=int(k),
                        )
                    )
                    metric_rows.append(
                        _metric_row(
                            split_frame["label_binary"],
                            probs,
                            threshold=threshold,
                            extra={
                                "evaluation_protocol": "paper_comparable_10fold_cv",
                                "analysis_family": "paper_comparable_cv",
                                "model_name": str(model_name),
                                "modality": modality,
                                "feature_strategy": feature_strategy,
                                "selected_feature_k": float(k),
                                "fold": int(fold_idx),
                                "fold_unit": "recording",
                                "metric_split": split_name,
                                "threshold_source": "inner_validation_balanced_accuracy",
                                "skipped": False,
                                "n_selected_features": float(len(selected_cols)),
                            },
                        )
                    )

    metrics = pd.DataFrame(metric_rows)
    aggregate = _aggregate_test_metrics(metrics)
    if not aggregate.empty:
        metrics = pd.concat([metrics, aggregate], ignore_index=True, sort=False)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    feature_selection = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    return PaperComparableCVResult(
        metrics=metrics,
        predictions=predictions,
        feature_selection=feature_selection,
    )
