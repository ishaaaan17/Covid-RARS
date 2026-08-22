from __future__ import annotations

import logging
from itertools import combinations
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from covid_rars.metrics import (
    best_threshold_by_balanced_accuracy,
    binary_metric_bundle,
    labels_to_binary,
)

logger = logging.getLogger(__name__)


def run_dndf_multimodal_fusion(
    participant_predictions: pd.DataFrame,
    modalities: Sequence[str] = ("cough", "breath", "speech"),
    fusion_methods: Sequence[str] = ("uniform_mean", "validation_weighted", "stacked_logistic"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform participant-level multimodal probability fusion for DNDT/DNDF.

    Args:
        participant_predictions: DataFrame with columns:
            [participant_id, split, modality, model_name, label_binary, probability]
        modalities: Modalities to combine.
        fusion_methods: Fusion algorithms to apply.

    Returns:
        tuple of (fusion_metrics_df, fusion_predictions_df)
    """
    if participant_predictions.empty:
        return pd.DataFrame(), pd.DataFrame()

    available_mods = [m for m in modalities if m in participant_predictions["modality"].unique()]
    if len(available_mods) < 2:
        logger.info("Fewer than 2 modalities available for fusion.")
        return pd.DataFrame(), pd.DataFrame()

    mod_combinations: list[tuple[str, ...]] = []
    for k in range(2, len(available_mods) + 1):
        for combo in combinations(available_mods, k):
            mod_combinations.append(combo)

    metrics_rows: list[dict[str, Any]] = []
    fused_predictions_list: list[pd.DataFrame] = []

    # Pivot predictions by participant and modality
    pivot = participant_predictions.pivot_table(
        index=["participant_id", "split", "label_binary"],
        columns="modality",
        values="probability",
        aggfunc="mean",
    ).reset_index()

    for combo in mod_combinations:
        combo_name = "+".join(combo)
        cols_present = [m for m in combo if m in pivot.columns]
        if len(cols_present) < len(combo):
            continue

        # Complete-case subset (all modalities in combo must be present)
        complete_case = pivot.dropna(subset=cols_present).copy()
        if complete_case.empty:
            continue

        val_mask = complete_case["split"] == "val"
        test_mask = complete_case["split"] == "test"
        train_mask = complete_case["split"] == "train"

        # 1. Uniform Mean Fusion
        if "uniform_mean" in fusion_methods:
            uniform_prob = complete_case[list(combo)].mean(axis=1)
            uniform_df = complete_case[["participant_id", "split", "label_binary"]].copy()
            uniform_df["probability"] = uniform_prob
            uniform_df["modality_combination"] = combo_name
            uniform_df["fusion_method"] = "uniform_mean"
            uniform_df["model_family"] = "dndt_dndf_fusion"

            # Determine validation threshold
            val_sub = uniform_df[uniform_df["split"] == "val"]
            if not val_sub.empty and len(np.unique(val_sub["label_binary"])) > 1:
                y_val_true = labels_to_binary(val_sub["label_binary"])
                thresh = best_threshold_by_balanced_accuracy(y_val_true, val_sub["probability"].to_numpy(dtype=float))
            else:
                thresh = 0.5
            uniform_df["threshold"] = thresh
            fused_predictions_list.append(uniform_df)

            for split_name in ("train", "val", "test"):
                s_df = uniform_df[uniform_df["split"] == split_name]
                if not s_df.empty and len(np.unique(s_df["label_binary"])) > 1:
                    y_true = labels_to_binary(s_df["label_binary"])
                    y_prob = s_df["probability"].to_numpy(dtype=float)
                    bundle = binary_metric_bundle(y_true, y_prob, threshold=thresh)
                    metrics_rows.append({
                        "modality_combination": combo_name,
                        "fusion_method": "uniform_mean",
                        "model_family": "dndt_dndf_fusion",
                        "split": split_name,
                        "n_participants": len(s_df),
                        "threshold": thresh,
                        **bundle,
                    })

        # 2. Validation-AUPRC Weighted Fusion
        if "validation_weighted" in fusion_methods:
            weights = []
            for m in combo:
                m_preds = participant_predictions[(participant_predictions["modality"] == m) & (participant_predictions["split"] == "val")]
                if not m_preds.empty and len(np.unique(m_preds["label_binary"])) > 1:
                    y_m_true = labels_to_binary(m_preds["label_binary"])
                    y_m_prob = m_preds["probability"].to_numpy(dtype=float)
                    m_bundle = binary_metric_bundle(y_m_true, y_m_prob)
                    w = max(m_bundle["auprc"] - 0.5, 0.01)
                else:
                    w = 1.0 / len(combo)
                weights.append(w)

            total_w = sum(weights)
            norm_weights = [w / total_w for w in weights]

            weighted_prob = sum(complete_case[m] * w for m, w in zip(combo, norm_weights))
            weighted_df = complete_case[["participant_id", "split", "label_binary"]].copy()
            weighted_df["probability"] = weighted_prob
            weighted_df["modality_combination"] = combo_name
            weighted_df["fusion_method"] = "validation_weighted"
            weighted_df["model_family"] = "dndt_dndf_fusion"

            val_sub = weighted_df[weighted_df["split"] == "val"]
            if not val_sub.empty and len(np.unique(val_sub["label_binary"])) > 1:
                y_val_true = labels_to_binary(val_sub["label_binary"])
                thresh = best_threshold_by_balanced_accuracy(y_val_true, val_sub["probability"].to_numpy(dtype=float))
            else:
                thresh = 0.5
            weighted_df["threshold"] = thresh
            fused_predictions_list.append(weighted_df)

            for split_name in ("train", "val", "test"):
                s_df = weighted_df[weighted_df["split"] == split_name]
                if not s_df.empty and len(np.unique(s_df["label_binary"])) > 1:
                    y_true = labels_to_binary(s_df["label_binary"])
                    y_prob = s_df["probability"].to_numpy(dtype=float)
                    bundle = binary_metric_bundle(y_true, y_prob, threshold=thresh)
                    metrics_rows.append({
                        "modality_combination": combo_name,
                        "fusion_method": "validation_weighted",
                        "model_family": "dndt_dndf_fusion",
                        "split": split_name,
                        "n_participants": len(s_df),
                        "threshold": thresh,
                        **bundle,
                    })

        # 3. Stacked Logistic Regression on Validation Data
        if "stacked_logistic" in fusion_methods and np.sum(val_mask) >= 10:
            X_val_meta = complete_case.loc[val_mask, list(combo)].to_numpy(dtype=np.float32)
            y_val_meta = labels_to_binary(complete_case.loc[val_mask, "label_binary"])

            if len(np.unique(y_val_meta)) > 1:
                stacker = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced")
                stacker.fit(X_val_meta, y_val_meta)

                X_all_meta = complete_case[list(combo)].to_numpy(dtype=np.float32)
                stacked_probs = stacker.predict_proba(X_all_meta)[:, 1]

                stacked_df = complete_case[["participant_id", "split", "label_binary"]].copy()
                stacked_df["probability"] = stacked_probs
                stacked_df["modality_combination"] = combo_name
                stacked_df["fusion_method"] = "stacked_logistic"
                stacked_df["model_family"] = "dndt_dndf_fusion"

                val_sub = stacked_df[stacked_df["split"] == "val"]
                thresh = best_threshold_by_balanced_accuracy(y_val_meta, val_sub["probability"].to_numpy(dtype=float))
                stacked_df["threshold"] = thresh
                fused_predictions_list.append(stacked_df)

                for split_name in ("train", "val", "test"):
                    s_df = stacked_df[stacked_df["split"] == split_name]
                    if not s_df.empty and len(np.unique(s_df["label_binary"])) > 1:
                        y_true = labels_to_binary(s_df["label_binary"])
                        y_prob = s_df["probability"].to_numpy(dtype=float)
                        bundle = binary_metric_bundle(y_true, y_prob, threshold=thresh)
                        metrics_rows.append({
                            "modality_combination": combo_name,
                            "fusion_method": "stacked_logistic",
                            "model_family": "dndt_dndf_fusion",
                            "split": split_name,
                            "n_participants": len(s_df),
                            "threshold": thresh,
                            **bundle,
                        })

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(fused_predictions_list, ignore_index=True) if fused_predictions_list else pd.DataFrame()

    return metrics_df, predictions_df


def run_dndf_comparator_hybrid_fusion(
    dndf_predictions: pd.DataFrame,
    comparator_predictions: pd.DataFrame,
    modality: str = "cough",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fuse DNDT/DNDF predictions with Classical Comparator predictions (e.g. LightGBM / SVC).

    Returns:
        tuple of (hybrid_metrics_df, hybrid_predictions_df)
    """
    if dndf_predictions.empty or comparator_predictions.empty:
        return pd.DataFrame(), pd.DataFrame()

    d_sub = dndf_predictions[dndf_predictions["modality"] == modality].copy()
    c_sub = comparator_predictions[comparator_predictions["modality"] == modality].copy()

    merged = pd.merge(
        d_sub[["participant_id", "split", "label_binary", "probability"]].rename(columns={"probability": "prob_dndf"}),
        c_sub[["participant_id", "split", "probability"]].rename(columns={"probability": "prob_comparator"}),
        on=["participant_id", "split"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged["probability"] = 0.5 * merged["prob_dndf"] + 0.5 * merged["prob_comparator"]
    merged["modality"] = modality
    merged["fusion_method"] = "hybrid_dndf_comparator_uniform"
    merged["model_family"] = "hybrid_dndf_comparator"

    val_sub = merged[merged["split"] == "val"]
    if not val_sub.empty and len(np.unique(val_sub["label_binary"])) > 1:
        y_val_true = labels_to_binary(val_sub["label_binary"])
        thresh = best_threshold_by_balanced_accuracy(y_val_true, val_sub["probability"].to_numpy(dtype=float))
    else:
        thresh = 0.5
    merged["threshold"] = thresh

    metrics_rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        s_df = merged[merged["split"] == split_name]
        if not s_df.empty and len(np.unique(s_df["label_binary"])) > 1:
            y_true = labels_to_binary(s_df["label_binary"])
            y_prob = s_df["probability"].to_numpy(dtype=float)
            bundle = binary_metric_bundle(y_true, y_prob, threshold=thresh)
            metrics_rows.append({
                "modality": modality,
                "fusion_method": "hybrid_dndf_comparator_uniform",
                "model_family": "hybrid_dndf_comparator",
                "split": split_name,
                "n_participants": len(s_df),
                "threshold": thresh,
                **bundle,
            })

    return pd.DataFrame(metrics_rows), merged
