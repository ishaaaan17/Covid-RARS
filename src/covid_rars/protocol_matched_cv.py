from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from covid_rars.metrics import (
    best_threshold_by_balanced_accuracy,
    binary_metric_bundle,
    labels_to_binary,
)
from covid_rars.paper_comparable_cv import (
    _aggregate_test_metrics,
    _fit_model,
    select_fold_feature_columns,
)
from covid_rars.strong_baseline import _predict_probability


PROTOCOL_NAME = "protocol_matched_participant_10fold_cv"


@dataclass(frozen=True)
class ProtocolMatchedCVResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_selection: pd.DataFrame
    split_audit: pd.DataFrame


def _has_two_classes(frame: pd.DataFrame) -> bool:
    return frame["label_binary"].isin(["positive", "negative"]).all() and frame["label_binary"].nunique() == 2


def _participant_label_table(frame: pd.DataFrame) -> pd.DataFrame:
    if "participant_id" not in frame.columns:
        raise ValueError("participant_id is required for participant-level protocol matching")
    label_counts = frame.groupby("participant_id")["label_binary"].nunique()
    mixed = label_counts[label_counts > 1]
    if not mixed.empty:
        examples = ", ".join(map(str, mixed.index.astype(str).tolist()[:5]))
        raise ValueError(f"Participants with mixed labels cannot be split safely: {examples}")
    table = (
        frame[["participant_id", "label_binary"]]
        .drop_duplicates()
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    if not _has_two_classes(table):
        raise ValueError("Participant table must contain both classes")
    return table


def _inner_participant_validation_split(
    train_validation_participants: pd.DataFrame,
    *,
    validation_fraction: float,
    random_state: int,
) -> tuple[set[str], set[str]]:
    y = labels_to_binary(train_validation_participants["label_binary"])
    if len(np.unique(y)) < 2:
        raise ValueError("Training participant fold does not contain both classes")
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(validation_fraction),
        random_state=random_state,
    )
    train_idx, validation_idx = next(splitter.split(np.zeros(len(y)), y))
    train_ids = set(train_validation_participants.iloc[train_idx]["participant_id"].astype(str))
    validation_ids = set(train_validation_participants.iloc[validation_idx]["participant_id"].astype(str))
    return train_ids, validation_ids


def _participant_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    test_fraction: float,
    validation_fraction: float,
    random_state: int,
) -> Iterable[tuple[int, set[str], set[str], set[str]]]:
    participants = _participant_label_table(frame)
    y = labels_to_binary(participants["label_binary"])
    class_counts = np.bincount(y, minlength=2)
    if int(class_counts.min()) < 2:
        raise ValueError("At least two positive and two negative participants are required")

    splitter = StratifiedShuffleSplit(
        n_splits=int(n_splits),
        test_size=float(test_fraction),
        random_state=random_state,
    )
    for fold_idx, (train_validation_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(participants)), y),
        start=1,
    ):
        train_validation_participants = participants.iloc[train_validation_idx].copy()
        test_ids = set(participants.iloc[test_idx]["participant_id"].astype(str))
        train_ids, validation_ids = _inner_participant_validation_split(
            train_validation_participants,
            validation_fraction=validation_fraction,
            random_state=random_state + fold_idx,
        )
        yield fold_idx, train_ids, validation_ids, test_ids


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
    default_dataset = pd.Series(["coswara"] * len(frame), index=frame.index)
    default_submodality = pd.Series(["unknown"] * len(frame), index=frame.index)
    return pd.DataFrame(
        {
            "recording_id": frame["recording_id"].astype(str).to_numpy(),
            "participant_id": frame["participant_id"].astype(str).to_numpy(),
            "dataset": frame.get("dataset", default_dataset).astype(str).to_numpy(),
            "modality": frame["modality"].astype(str).to_numpy(),
            "submodality": frame.get("submodality", default_submodality).astype(str).to_numpy(),
            "label_binary": frame["label_binary"].to_numpy(),
            "split": split,
            "fold": int(fold),
            "fold_unit": "participant",
            "evaluation_protocol": PROTOCOL_NAME,
            "analysis_family": "protocol_matched_cv",
            "model_name": model_name,
            "feature_strategy": feature_strategy,
            "selected_feature_k": float(selected_feature_k),
            "probability": probabilities,
        }
    )


def _split_audit_row(
    fold: int,
    train_ids: set[str],
    validation_ids: set[str],
    test_ids: set[str],
) -> dict[str, object]:
    train_validation_overlap = train_ids & validation_ids
    train_test_overlap = train_ids & test_ids
    validation_test_overlap = validation_ids & test_ids
    overlap_count = len(train_validation_overlap | train_test_overlap | validation_test_overlap)
    return {
        "evaluation_protocol": PROTOCOL_NAME,
        "fold": int(fold),
        "fold_unit": "participant",
        "n_train_participants": float(len(train_ids)),
        "n_validation_participants": float(len(validation_ids)),
        "n_test_participants": float(len(test_ids)),
        "overlap_count": int(overlap_count),
        "train_validation_overlap": ",".join(sorted(train_validation_overlap)),
        "train_test_overlap": ",".join(sorted(train_test_overlap)),
        "validation_test_overlap": ",".join(sorted(validation_test_overlap)),
    }


def run_protocol_matched_cv(
    features: pd.DataFrame,
    *,
    modality: str = "cough",
    n_splits: int = 10,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.125,
    top_k_values: Iterable[int] = (800,),
    ranker: str = "lightgbm",
    model_names: Iterable[str] = ("svc_rbf_f60",),
    random_state: int = 42,
    optuna_trials: int = 0,
) -> ProtocolMatchedCVResult:
    df = features[
        features["modality"].astype(str).eq(str(modality))
        & features["label_binary"].isin(["positive", "negative"])
    ].copy()
    if "quality_flag" in df.columns:
        df = df[df["quality_flag"].astype(str).eq("ok")].copy()
    if df.empty or not _has_two_classes(df):
        raise ValueError(f"No two-class supervised rows are available for modality={modality}")

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    selection_rows: list[pd.DataFrame] = []
    split_audit_rows: list[dict[str, object]] = []

    for fold_idx, train_ids, validation_ids, test_ids in _participant_folds(
        df,
        n_splits=n_splits,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        random_state=random_state,
    ):
        split_audit_rows.append(_split_audit_row(fold_idx, train_ids, validation_ids, test_ids))
        participant_ids = df["participant_id"].astype(str)
        train = df[participant_ids.isin(train_ids)].copy()
        validation = df[participant_ids.isin(validation_ids)].copy()
        test = df[participant_ids.isin(test_ids)].copy()
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
            ranking["fold_unit"] = "participant"
            ranking["evaluation_protocol"] = PROTOCOL_NAME
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
                            "evaluation_protocol": PROTOCOL_NAME,
                            "analysis_family": "protocol_matched_cv",
                            "model_name": str(model_name),
                            "modality": modality,
                            "feature_strategy": feature_strategy,
                            "selected_feature_k": float(k),
                            "fold": int(fold_idx),
                            "fold_unit": "participant",
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
                                "evaluation_protocol": PROTOCOL_NAME,
                                "analysis_family": "protocol_matched_cv",
                                "model_name": str(model_name),
                                "modality": modality,
                                "feature_strategy": feature_strategy,
                                "selected_feature_k": float(k),
                                "fold": int(fold_idx),
                                "fold_unit": "participant",
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
    split_audit = pd.DataFrame(split_audit_rows)
    return ProtocolMatchedCVResult(
        metrics=metrics,
        predictions=predictions,
        feature_selection=feature_selection,
        split_audit=split_audit,
    )
