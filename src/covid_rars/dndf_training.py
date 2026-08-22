from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from covid_rars.dndf_models import DNDFClassifier, DNDFHyperparameters
from covid_rars.features import feature_columns
from covid_rars.metrics import (
    best_threshold_by_balanced_accuracy,
    binary_metric_bundle,
    labels_to_binary,
)

logger = logging.getLogger(__name__)

DEFAULT_DNDF_MODALITIES = ("cough", "breath", "speech")


@dataclass(frozen=True)
class DNDFTrainingResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    selection: pd.DataFrame
    history: list[dict[str, Any]]


def participant_average_predictions(
    frame: pd.DataFrame,
    id_column: str = "participant_id",
    prob_column: str = "probability",
    label_column: str = "label_binary",
) -> pd.DataFrame:
    """Average recording-level probabilities to participant-level predictions."""
    if frame.empty:
        return frame.copy()
    group_cols = [id_column]
    extra_cols = [c for c in ["split", "modality", "submodality", "model_name", "evaluation_protocol"] if c in frame.columns]
    group_cols.extend(extra_cols)

    agg_dict = {
        label_column: "first",
        prob_column: "mean",
    }
    if "threshold" in frame.columns:
        agg_dict["threshold"] = "first"

    grouped = frame.groupby(group_cols, as_index=False, dropna=False).agg(agg_dict)
    return grouped


def train_dndf_modality_models(
    features: pd.DataFrame,
    modalities: Iterable[str] = DEFAULT_DNDF_MODALITIES,
    model_types: Sequence[str] = ("dndf", "dndt"),
    num_trees: int = 20,
    depth: int = 4,
    used_features_rate: float = 0.8,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    max_epochs: int = 50,
    patience: int = 10,
    use_smote: bool = True,
    feature_selection: str = "none",
    random_state: int = 42,
    device: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train DNDT and DNDF models across specified modalities and evaluate on val/test splits.

    Returns:
        tuple of (metrics_df, predictions_df, selection_df)
    """
    feat_cols = feature_columns(features)
    if not feat_cols:
        raise ValueError("Features DataFrame has no numeric feature columns.")

    metrics_rows: list[dict[str, Any]] = []
    predictions_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []

    for modality in modalities:
        mod_data = features[features["modality"] == modality].copy()
        if mod_data.empty:
            logger.warning(f"No data found for modality: {modality}")
            continue

        train_data = mod_data[mod_data["split"] == "train"].copy()
        val_data = mod_data[mod_data["split"].isin(["val", "validation"])].copy()
        test_data = mod_data[mod_data["split"] == "test"].copy()

        if train_data.empty:
            logger.warning(f"No train split for modality: {modality}")
            continue

        X_train = train_data[feat_cols].to_numpy(dtype=np.float32)
        y_train = labels_to_binary(train_data["label_binary"])

        X_val = val_data[feat_cols].to_numpy(dtype=np.float32) if not val_data.empty else None
        y_val = labels_to_binary(val_data["label_binary"]) if not val_data.empty else None

        X_test = test_data[feat_cols].to_numpy(dtype=np.float32) if not test_data.empty else None
        y_test = labels_to_binary(test_data["label_binary"]) if not test_data.empty else None

        for m_type in model_types:
            model_name = f"{m_type}_depth{depth}"
            if m_type == "dndf":
                model_name = f"dndf_trees{num_trees}_depth{depth}"

            clf = DNDFClassifier(
                model_type=m_type,
                num_trees=num_trees,
                depth=depth,
                used_features_rate=used_features_rate,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                use_smote=use_smote,
                feature_selection=feature_selection,
                device=device,
                random_state=random_state,
            )

            print(f"\n  >> Training {model_name} on [{modality.upper()}] (Samples: train={len(X_train)}, val={len(X_val) if X_val is not None else 0}, test={len(X_test) if X_test is not None else 0})...")
            clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)

            # Predict on all available splits
            for split_name, s_data, s_x, s_y in [
                ("train", train_data, X_train, y_train),
                ("val", val_data, X_val, y_val),
                ("test", test_data, X_test, y_test),
            ]:
                if s_data.empty or s_x is None:
                    continue

                probs = clf.predict_proba(s_x)[:, 1]
                pred_df = s_data[["participant_id", "recording_id", "modality", "label_binary"]].copy()
                pred_df["split"] = split_name
                pred_df["probability"] = probs
                pred_df["threshold"] = clf.best_threshold_
                pred_df["model_name"] = model_name
                pred_df["model_family"] = "dndt_dndf"
                predictions_frames.append(pred_df)

                # Participant-level metrics
                part_df = participant_average_predictions(pred_df)
                y_true = labels_to_binary(part_df["label_binary"])
                y_prob = part_df["probability"].to_numpy(dtype=float)

                if len(np.unique(y_true)) > 1:
                    m_bundle = binary_metric_bundle(y_true, y_prob, threshold=clf.best_threshold_)
                else:
                    m_bundle = {
                        "auroc": 0.5,
                        "auprc": 0.5,
                        "balanced_accuracy": 0.5,
                        "sensitivity": 0.5,
                        "specificity": 0.5,
                        "brier": 0.25,
                        "ece": 0.0,
                        "nll": 0.693,
                    }
                metrics_rows.append({
                    "modality": modality,
                    "model_name": model_name,
                    "model_family": "dndt_dndf",
                    "split": split_name,
                    "n_participants": len(part_df),
                    "n_recordings": len(pred_df),
                    "threshold": clf.best_threshold_,
                    **m_bundle,
                })

            # Selection on validation AUROC
            val_rows = [r for r in metrics_rows if r["modality"] == modality and r["model_name"] == model_name and r["split"] == "val"]
            val_auroc = val_rows[0]["auroc"] if val_rows else 0.5
            selection_rows.append({
                "modality": modality,
                "model_name": model_name,
                "validation_auroc": val_auroc,
                "selected_threshold": clf.best_threshold_,
            })

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(predictions_frames, ignore_index=True) if predictions_frames else pd.DataFrame()
    selection_df = pd.DataFrame(selection_rows)

    return metrics_df, predictions_df, selection_df
