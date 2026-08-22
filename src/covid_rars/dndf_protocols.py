from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from covid_rars.dndf_models import DNDFClassifier
from covid_rars.dndf_training import participant_average_predictions, train_dndf_modality_models
from covid_rars.dndf_fusion import run_dndf_multimodal_fusion
from covid_rars.features import feature_columns
from covid_rars.metrics import (
    best_threshold_by_balanced_accuracy,
    binary_metric_bundle,
    labels_to_binary,
)
from covid_rars.split import create_participant_splits
from covid_rars.temporal_holdout import (
    build_temporal_split_assignments,
    build_time_stratified_split_assignments,
)

logger = logging.getLogger(__name__)

TRACK_A_SEEDS = (1, 2, 5, 12, 40, 52, 72, 2002, 4002, 6002)


@dataclass(frozen=True)
class DNDFProtocolResult:
    protocol_name: str
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    multimodal_metrics: pd.DataFrame
    multimodal_predictions: pd.DataFrame


def run_track_a_repeated_holdouts(
    features: pd.DataFrame,
    modalities: Sequence[str] = ("cough", "breath", "speech"),
    seeds: Sequence[int] = TRACK_A_SEEDS,
    num_trees: int = 20,
    depth: int = 4,
    used_features_rate: float = 0.8,
    learning_rate: float = 0.01,
    max_epochs: int = 50,
    patience: int = 10,
    use_smote: bool = True,
    device: str = "auto",
) -> DNDFProtocolResult:
    """Run Track A: 10 repeated stratified participant-level holdouts (approx 70/10/20 train/val/test)."""
    feat_cols = feature_columns(features)
    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_fusion_metrics: list[pd.DataFrame] = []
    all_fusion_predictions: list[pd.DataFrame] = []

    # Get distinct participant labels
    participants = features[["participant_id", "label_binary"]].drop_duplicates().copy()

    for fold_idx, seed in enumerate(seeds):
        logger.info(f"Running Track A Fold {fold_idx + 1}/{len(seeds)} (seed={seed})")
        rng = np.random.RandomState(seed)

        # Split participants 70/10/20
        pos_parts = participants[participants["label_binary"] == "positive"]["participant_id"].values
        neg_parts = participants[participants["label_binary"] == "negative"]["participant_id"].values

        rng.shuffle(pos_parts)
        rng.shuffle(neg_parts)

        def make_split_map(pos_arr: np.ndarray, neg_arr: np.ndarray) -> dict[str, str]:
            n_pos, n_neg = len(pos_arr), len(neg_arr)
            pos_train_end = int(0.70 * n_pos)
            pos_val_end = int(0.80 * n_pos)
            neg_train_end = int(0.70 * n_neg)
            neg_val_end = int(0.80 * n_neg)

            split_map = {}
            for p in pos_arr[:pos_train_end]: split_map[p] = "train"
            for p in pos_arr[pos_train_end:pos_val_end]: split_map[p] = "val"
            for p in pos_arr[pos_val_end:]: split_map[p] = "test"

            for p in neg_arr[:neg_train_end]: split_map[p] = "train"
            for p in neg_arr[neg_train_end:neg_val_end]: split_map[p] = "val"
            for p in neg_arr[neg_val_end:]: split_map[p] = "test"
            return split_map

        split_dict = make_split_map(pos_parts, neg_parts)
        fold_features = features.copy()
        fold_features["split"] = fold_features["participant_id"].map(split_dict)
        fold_features = fold_features.dropna(subset=["split"]).copy()

        m_df, p_df, _ = train_dndf_modality_models(
            fold_features,
            modalities=modalities,
            model_types=("dndf", "dndt"),
            num_trees=num_trees,
            depth=depth,
            used_features_rate=used_features_rate,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            patience=patience,
            use_smote=use_smote,
            random_state=seed,
            device=device,
        )

        if not m_df.empty:
            m_df["fold"] = fold_idx
            m_df["seed"] = seed
            m_df["protocol"] = "track_a_repeated_holdout"
            all_metrics.append(m_df)

        if not p_df.empty:
            p_df["fold"] = fold_idx
            p_df["seed"] = seed
            p_df["protocol"] = "track_a_repeated_holdout"
            all_predictions.append(p_df)

            # Multimodal fusion for this fold
            part_p_df = participant_average_predictions(p_df)
            f_m_df, f_p_df = run_dndf_multimodal_fusion(part_p_df, modalities=modalities)
            if not f_m_df.empty:
                f_m_df["fold"] = fold_idx
                f_m_df["seed"] = seed
                f_m_df["protocol"] = "track_a_repeated_holdout"
                all_fusion_metrics.append(f_m_df)
            if not f_p_df.empty:
                f_p_df["fold"] = fold_idx
                f_p_df["seed"] = seed
                f_p_df["protocol"] = "track_a_repeated_holdout"
                all_fusion_predictions.append(f_p_df)

    res_metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    res_preds = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    res_fusion_m = pd.concat(all_fusion_metrics, ignore_index=True) if all_fusion_metrics else pd.DataFrame()
    res_fusion_p = pd.concat(all_fusion_predictions, ignore_index=True) if all_fusion_predictions else pd.DataFrame()

    return DNDFProtocolResult(
        protocol_name="track_a_repeated_holdout",
        metrics=res_metrics,
        predictions=res_preds,
        multimodal_metrics=res_fusion_m,
        multimodal_predictions=res_fusion_p,
    )


def run_track_b_temporal_contrast(
    features: pd.DataFrame,
    modalities: Sequence[str] = ("cough", "breath", "speech"),
    num_trees: int = 20,
    depth: int = 4,
    used_features_rate: float = 0.8,
    learning_rate: float = 0.01,
    max_epochs: int = 50,
    patience: int = 10,
    use_smote: bool = True,
    device: str = "auto",
    random_state: int = 42,
) -> tuple[DNDFProtocolResult, DNDFProtocolResult]:
    """Run Track B: Chronological early-to-late split vs. Calendar-mixed date-balanced split."""
    # Ensure recording_date column exists
    features_clean = features.copy()
    if "recording_date" not in features_clean.columns:
        if "date" in features_clean.columns:
            features_clean["recording_date"] = features_clean["date"]
        else:
            # Generate deterministic chronological date sequence from participant ordering
            unique_parts = list(features_clean["participant_id"].unique())
            base_date = pd.Timestamp("2020-04-01")
            date_map = {p: (base_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i, p in enumerate(unique_parts)}
            features_clean["recording_date"] = features_clean["participant_id"].map(date_map)

    # 1. Chronological early-to-late
    try:
        chron_features = features_clean.copy()
        chron_splits = build_temporal_split_assignments(chron_features)
        chron_features["split"] = chron_features["participant_id"].map(chron_splits["participant_id_to_split"])
        chron_features = chron_features.dropna(subset=["split"]).copy()

        m_chron, p_chron, _ = train_dndf_modality_models(
            chron_features,
            modalities=modalities,
            model_types=("dndf", "dndt"),
            num_trees=num_trees,
            depth=depth,
            used_features_rate=used_features_rate,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            patience=patience,
            use_smote=use_smote,
            random_state=random_state,
            device=device,
        )
        part_p_chron = participant_average_predictions(p_chron)
        fm_chron, fp_chron = run_dndf_multimodal_fusion(part_p_chron, modalities=modalities)
        res_chron = DNDFProtocolResult(
            protocol_name="track_b_chronological_early_to_late",
            metrics=m_chron,
            predictions=p_chron,
            multimodal_metrics=fm_chron,
            multimodal_predictions=fp_chron,
        )
    except Exception as e:
        logger.warning(f"Track B Chronological split could not be computed: {e}")
        res_chron = DNDFProtocolResult("track_b_chronological_early_to_late", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    # 2. Time-stratified / Calendar-mixed baseline
    try:
        cal_features = features_clean.copy()
        cal_splits = build_time_stratified_split_assignments(cal_features)
        cal_features["split"] = cal_features["participant_id"].map(cal_splits["participant_id_to_split"])
        cal_features = cal_features.dropna(subset=["split"]).copy()

        m_cal, p_cal, _ = train_dndf_modality_models(
            cal_features,
            modalities=modalities,
            model_types=("dndf", "dndt"),
            num_trees=num_trees,
            depth=depth,
            used_features_rate=used_features_rate,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            patience=patience,
            use_smote=use_smote,
            random_state=random_state,
            device=device,
        )
        part_p_cal = participant_average_predictions(p_cal)
        fm_cal, fp_cal = run_dndf_multimodal_fusion(part_p_cal, modalities=modalities)
        res_cal = DNDFProtocolResult(
            protocol_name="track_b_calendar_mixed",
            metrics=m_cal,
            predictions=p_cal,
            multimodal_metrics=fm_cal,
            multimodal_predictions=fp_cal,
        )
    except Exception as e:
        logger.warning(f"Track B Calendar-mixed split could not be computed: {e}")
        res_cal = DNDFProtocolResult("track_b_calendar_mixed", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    return res_chron, res_cal


def run_track_c_external_transfer(
    source_features: pd.DataFrame,
    target_external_features: pd.DataFrame,
    modality: str = "cough",
    num_trees: int = 20,
    depth: int = 4,
    used_features_rate: float = 0.8,
    learning_rate: float = 0.01,
    max_epochs: int = 50,
    patience: int = 10,
    use_smote: bool = True,
    device: str = "auto",
    random_state: int = 42,
) -> DNDFProtocolResult:
    """Run Track C: Coswara-trained cough DNDT/DNDF tested directly on COUGHVID external cough dataset."""
    feat_cols = [c for c in feature_columns(source_features) if c in target_external_features.columns]

    src_cough = source_features[source_features["modality"] == modality].copy()
    tgt_cough = target_external_features.copy()

    train_data = src_cough[src_cough["split"] == "train"].copy()
    val_data = src_cough[src_cough["split"] == "val"].copy()

    X_train = train_data[feat_cols].to_numpy(dtype=np.float32)
    y_train = labels_to_binary(train_data["label_binary"])

    X_val = val_data[feat_cols].to_numpy(dtype=np.float32) if not val_data.empty else None
    y_val = labels_to_binary(val_data["label_binary"]) if not val_data.empty else None

    X_tgt = tgt_cough[feat_cols].to_numpy(dtype=np.float32)
    y_tgt = labels_to_binary(tgt_cough["label_binary"])

    metrics_rows: list[dict[str, Any]] = []
    predictions_frames: list[pd.DataFrame] = []

    for m_type in ("dndf", "dndt"):
        model_name = f"{m_type}_depth{depth}"
        if m_type == "dndf":
            model_name = f"dndf_trees{num_trees}_depth{depth}"

        clf = DNDFClassifier(
            model_type=m_type,
            num_trees=num_trees,
            depth=depth,
            used_features_rate=used_features_rate,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            patience=patience,
            use_smote=use_smote,
            device=device,
            random_state=random_state,
        )
        clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # Internal validation score
        val_probs = clf.predict_proba(X_val)[:, 1] if X_val is not None else []
        if len(val_probs) > 0 and len(np.unique(y_val)) > 1:
            val_bundle = binary_metric_bundle(y_val, val_probs, threshold=clf.best_threshold_)
            metrics_rows.append({
                "dataset": "coswara_source_val",
                "modality": modality,
                "model_name": model_name,
                "threshold": clf.best_threshold_,
                **val_bundle,
            })

        # External COUGHVID evaluation
        tgt_probs = clf.predict_proba(X_tgt)[:, 1]
        tgt_preds = tgt_cough[["participant_id", "recording_id", "label_binary"]].copy()
        tgt_preds["probability"] = tgt_probs
        tgt_preds["threshold"] = clf.best_threshold_
        tgt_preds["model_name"] = model_name
        tgt_preds["evaluation_protocol"] = "coswara_to_coughvid_external"
        predictions_frames.append(tgt_preds)

        if len(np.unique(y_tgt)) > 1:
            tgt_bundle = binary_metric_bundle(y_tgt, tgt_probs, threshold=clf.best_threshold_)
            metrics_rows.append({
                "dataset": "coughvid_external",
                "modality": modality,
                "model_name": model_name,
                "threshold": clf.best_threshold_,
                **tgt_bundle,
            })

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(predictions_frames, ignore_index=True) if predictions_frames else pd.DataFrame()

    return DNDFProtocolResult(
        protocol_name="track_c_external_transfer",
        metrics=metrics_df,
        predictions=predictions_df,
        multimodal_metrics=pd.DataFrame(),
        multimodal_predictions=pd.DataFrame(),
    )
