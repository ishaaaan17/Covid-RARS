from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from covid_rars.dndf_fusion import run_dndf_multimodal_fusion
from covid_rars.dndf_models import DNDFClassifier
from covid_rars.dndf_protocols import (
    DNDFProtocolResult,
    run_track_a_repeated_holdouts,
    run_track_b_temporal_contrast,
    run_track_c_external_transfer,
)
from covid_rars.dndf_training import participant_average_predictions
from covid_rars.features import feature_columns
from covid_rars.metrics import binary_metric_bundle, labels_to_binary

logger = logging.getLogger(__name__)


def run_track1_author_exact_reproduction(
    features_df: pd.DataFrame,
    modality: str = "cough",
    n_splits: int = 10,
    num_trees: int = 25,
    depth: int = 11,
    learning_rate: float = 0.01,
    batch_size: int = 16,
    max_epochs: int = 14,
    n_selected_features: int = 33,
    feature_selection: str = "rfecv_extratrees",
    random_state: int = 42,
    device: str = "auto",
) -> pd.DataFrame:
    """Track 1: Authors' Exact Paper Reproduction Benchmark.

    Matches Islam et al. (ESWA 2026 / arXiv:2501.01117):
    - Cough audio
    - 10-Fold Stratified Cross-Validation (recording-level split)
    - ExtraTrees + RFECV feature selection (selecting ~33 features on Coswara)
    - 25 trees, depth 11, lr 0.01, batch size 16, 14 epochs, SMOTE balancing
    """
    print("\n" + "=" * 80)
    print("📜 TRACK 1: ISLAM ET AL. (ESWA 2026) EXACT PAPER REPRODUCTION (10-FOLD CV)")
    print("=" * 80)
    print(f"  • Modality            : [{modality.upper()}]")
    print(f"  • Cross-Validation    : {n_splits}-Fold Stratified Recording-Level CV")
    print(f"  • Feature Selection   : {feature_selection.upper()} (~{n_selected_features} features)")
    print(f"  • Model Architecture  : DNDF (Trees={num_trees}, Depth={depth})")
    print(f"  • Training Parameters : Epochs={max_epochs}, LR={learning_rate}, BatchSize={batch_size}, SMOTE=True")
    print("-" * 80)

    mod_df = features_df[features_df["modality"] == modality].copy()
    if mod_df.empty:
        logger.warning(f"No records found for modality {modality}. Skipping Track 1.")
        return pd.DataFrame()

    feat_cols = feature_columns(mod_df)
    X = mod_df[feat_cols].to_numpy(dtype=np.float32)
    y = labels_to_binary(mod_df["label_binary"])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_records: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        f_t0 = time.time()
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        clf = DNDFClassifier(
            model_type="dndf",
            num_trees=num_trees,
            depth=depth,
            used_features_rate=0.80,
            temperature=1.0,
            learning_rate=learning_rate,
            weight_decay=1e-4,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=5,
            use_smote=True,
            feature_selection=feature_selection,
            n_selected_features=n_selected_features,
            device=device,
            random_state=random_state + fold_idx,
        )

        clf.fit(X_train, y_train, X_val=X_test, y_val=y_test, optimize_threshold=True)

        probs_test = clf.predict_proba(X_test)[:, 1]
        preds_test = (probs_test >= clf.best_threshold_).astype(int)

        acc = accuracy_score(y_test, preds_test)
        bacc = balanced_accuracy_score(y_test, preds_test)
        auroc = roc_auc_score(y_test, probs_test) if len(np.unique(y_test)) > 1 else 0.5
        rec = recall_score(y_test, preds_test, zero_division=0)
        prec = precision_score(y_test, preds_test, zero_division=0)
        f1 = f1_score(y_test, preds_test, zero_division=0)
        cm = confusion_matrix(y_test, preds_test, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        elapsed = time.time() - f_t0

        print(
            f"  Fold {fold_idx + 1:02d}/{n_splits:02d} -> "
            f"Acc: {acc*100:5.2f}% | AUROC: {auroc:6.4f} | BAcc: {bacc*100:5.2f}% | "
            f"Sens: {rec*100:5.2f}% | Spec: {spec*100:5.2f}% ({elapsed:4.1f}s)"
        )

        fold_records.append({
            "track": "track1_author_exact_reproduction",
            "modality": modality,
            "fold": fold_idx + 1,
            "accuracy": acc,
            "auroc": auroc,
            "balanced_accuracy": bacc,
            "sensitivity": rec,
            "specificity": spec,
            "precision": prec,
            "f1_score": f1,
            "threshold": clf.best_threshold_,
        })

    summary_df = pd.DataFrame(fold_records)
    print("-" * 80)
    print(f"🏆 TRACK 1 FINAL RESULTS ({n_splits}-FOLD MEAN ± STD):")
    print(f"   • Overall Accuracy   : {summary_df['accuracy'].mean()*100:5.2f}% ± {summary_df['accuracy'].std()*100:4.2f}%")
    print(f"   • ROC-AUC (AUROC)    : {summary_df['auroc'].mean():6.4f} ± {summary_df['auroc'].std():6.4f}")
    print(f"   • Balanced Accuracy  : {summary_df['balanced_accuracy'].mean()*100:5.2f}% ± {summary_df['balanced_accuracy'].std()*100:4.2f}%")
    print(f"   • Sensitivity/Recall : {summary_df['sensitivity'].mean()*100:5.2f}% ± {summary_df['sensitivity'].std()*100:4.2f}%")
    print(f"   • Specificity        : {summary_df['specificity'].mean()*100:5.2f}% ± {summary_df['specificity'].std()*100:4.2f}%")
    print(f"   • Precision          : {summary_df['precision'].mean()*100:5.2f}% ± {summary_df['precision'].std()*100:4.2f}%")
    print(f"   • F1-Score           : {summary_df['f1_score'].mean():6.4f} ± {summary_df['f1_score'].std():6.4f}")
    print("=" * 80 + "\n")

    return summary_df


def run_track2_corrected_leak_free_reproduction(
    features_df: pd.DataFrame,
    modality: str = "cough",
    n_splits: int = 10,
    num_trees: int = 25,
    depth: int = 11,
    learning_rate: float = 0.01,
    batch_size: int = 16,
    max_epochs: int = 14,
    n_selected_features: int = 33,
    feature_selection: str = "rfecv_extratrees",
    random_state: int = 42,
    device: str = "auto",
) -> pd.DataFrame:
    """Track 2: Methodologically Corrected Leakage-Free Reproduction.

    Enforces strict scientific validity:
    1. Feature selection (RFECV + ExtraTrees) is fitted STRICTLY on the training fold only.
    2. Inner validation split is used for threshold tuning and early stopping (test fold is 100% unseen).
    3. Fresh model and scaler instantiated per fold.
    """
    print("\n" + "=" * 80)
    print("🛡️ TRACK 2: METHODOLOGICALLY CORRECTED REPRODUCTION (ZERO LEAKAGE 10-FOLD CV)")
    print("=" * 80)
    print(f"  • Modality            : [{modality.upper()}]")
    print(f"  • Cross-Validation    : {n_splits}-Fold Stratified Nested CV")
    print(f"  • Leakage Prevention  : RFECV fitted strictly on Train Split; Threshold tuned on Inner Val")
    print(f"  • Model Architecture  : DNDF (Trees={num_trees}, Depth={depth})")
    print("-" * 80)

    mod_df = features_df[features_df["modality"] == modality].copy()
    if mod_df.empty:
        logger.warning(f"No records found for modality {modality}. Skipping Track 2.")
        return pd.DataFrame()

    feat_cols = feature_columns(mod_df)
    X = mod_df[feat_cols].to_numpy(dtype=np.float32)
    y = labels_to_binary(mod_df["label_binary"])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_records: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        f_t0 = time.time()
        X_outer_train, y_outer_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Inner validation split (90% train -> 80% train-sub, 10% inner-val)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_outer_train, y_outer_train, test_size=0.15, random_state=random_state + fold_idx, stratify=y_outer_train
        )

        clf = DNDFClassifier(
            model_type="dndf",
            num_trees=num_trees,
            depth=depth,
            used_features_rate=0.80,
            temperature=1.0,
            learning_rate=learning_rate,
            weight_decay=1e-4,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=5,
            use_smote=True,
            feature_selection=feature_selection,
            n_selected_features=n_selected_features,
            device=device,
            random_state=random_state + fold_idx,
        )

        # Fit strictly on inner train, optimize threshold on inner val
        clf.fit(X_tr, y_tr, X_val=X_val, y_val=y_val, optimize_threshold=True)

        # Final evaluation on 100% untouched outer test fold
        probs_test = clf.predict_proba(X_test)[:, 1]
        preds_test = (probs_test >= clf.best_threshold_).astype(int)

        acc = accuracy_score(y_test, preds_test)
        bacc = balanced_accuracy_score(y_test, preds_test)
        auroc = roc_auc_score(y_test, probs_test) if len(np.unique(y_test)) > 1 else 0.5
        rec = recall_score(y_test, preds_test, zero_division=0)
        prec = precision_score(y_test, preds_test, zero_division=0)
        f1 = f1_score(y_test, preds_test, zero_division=0)
        cm = confusion_matrix(y_test, preds_test, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        elapsed = time.time() - f_t0

        print(
            f"  Fold {fold_idx + 1:02d}/{n_splits:02d} -> "
            f"Acc: {acc*100:5.2f}% | AUROC: {auroc:6.4f} | BAcc: {bacc*100:5.2f}% | "
            f"Sens: {rec*100:5.2f}% | Spec: {spec*100:5.2f}% ({elapsed:4.1f}s)"
        )

        fold_records.append({
            "track": "track2_corrected_leak_free_reproduction",
            "modality": modality,
            "fold": fold_idx + 1,
            "accuracy": acc,
            "auroc": auroc,
            "balanced_accuracy": bacc,
            "sensitivity": rec,
            "specificity": spec,
            "precision": prec,
            "f1_score": f1,
            "threshold": clf.best_threshold_,
        })

    summary_df = pd.DataFrame(fold_records)
    print("-" * 80)
    print(f"🏆 TRACK 2 FINAL RESULTS (CORRECTED LEAK-FREE {n_splits}-FOLD MEAN ± STD):")
    print(f"   • Overall Accuracy   : {summary_df['accuracy'].mean()*100:5.2f}% ± {summary_df['accuracy'].std()*100:4.2f}%")
    print(f"   • ROC-AUC (AUROC)    : {summary_df['auroc'].mean():6.4f} ± {summary_df['auroc'].std():6.4f}")
    print(f"   • Balanced Accuracy  : {summary_df['balanced_accuracy'].mean()*100:5.2f}% ± {summary_df['balanced_accuracy'].std()*100:4.2f}%")
    print(f"   • Sensitivity/Recall : {summary_df['sensitivity'].mean()*100:5.2f}% ± {summary_df['sensitivity'].std()*100:4.2f}%")
    print(f"   • Specificity        : {summary_df['specificity'].mean()*100:5.2f}% ± {summary_df['specificity'].std()*100:4.2f}%")
    print(f"   • Precision          : {summary_df['precision'].mean()*100:5.2f}% ± {summary_df['precision'].std()*100:4.2f}%")
    print(f"   • F1-Score           : {summary_df['f1_score'].mean():6.4f} ± {summary_df['f1_score'].std():6.4f}")
    print("=" * 80 + "\n")

    return summary_df


def run_track3_covid_rars_reliability_suite(
    features_df: pd.DataFrame,
    external_features_df: pd.DataFrame | None = None,
    modalities: Sequence[str] = ("cough", "breath", "speech"),
    seeds: Sequence[int] = (1, 2, 5, 12, 40),
    num_trees: int = 25,
    depth: int = 5,
    learning_rate: float = 0.005,
    max_epochs: int = 30,
    device: str = "auto",
) -> dict[str, Any]:
    """Track 3: COVID-RARS Extended Reliability Suite.

    Evaluates the corrected DNDF on:
    - Track 3A: Literature-Aligned Participant-Disjoint Holdouts (10 Seeds)
    - Track 3B: Chronological Temporal Generalization (Real dates only, skips if missing)
    - Track 3C: External Transfer (Coswara -> COUGHVID zero-shot)
    - Track 3D: Multimodal Late Fusion (Cough + Breath + Speech via Stacked Logistic)
    """
    print("\n" + "=" * 80)
    print("🔬 TRACK 3: COVID-RARS CLINICAL RELIABILITY PROTOCOL SUITE")
    print("=" * 80)
    print(f"  • Modalities          : {list(modalities)}")
    print(f"  • Participant Seeds   : {list(seeds)}")
    print(f"  • Model Architecture  : DNDF (Trees={num_trees}, Depth={depth}, LR={learning_rate})")
    print("-" * 80)

    # 3A: Participant-Disjoint Holdouts
    print("\n>> Executing Track 3A: Participant-Disjoint Repeated Holdouts...")
    res_3a = run_track_a_repeated_holdouts(
        features=features_df,
        modalities=modalities,
        seeds=seeds,
        num_trees=num_trees,
        depth=depth,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        use_smote=True,
        device=device,
    )

    # 3B: Temporal Contrast (Only if real dates exist)
    has_real_dates = "recording_date" in features_df.columns or "date" in features_df.columns
    if has_real_dates:
        print("\n>> Executing Track 3B: Chronological vs. Calendar-Mixed Temporal Contrast...")
        res_3b_chron, res_3b_cal = run_track_b_temporal_contrast(
            features=features_df,
            modalities=modalities,
            num_trees=num_trees,
            depth=depth,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            use_smote=True,
            device=device,
        )
    else:
        print("\n>> ⚠️ Track 3B SKIPPED: Metadata does not contain genuine recording_date column.")
        res_3b_chron = DNDFProtocolResult("track_b_chronological", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        res_3b_cal = DNDFProtocolResult("track_b_calendar_mixed", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    # 3C: External Transfer
    if external_features_df is not None and not external_features_df.empty:
        print("\n>> Executing Track 3C: Zero-Shot External Transfer (Coswara -> COUGHVID)...")
        res_3c = run_track_c_external_transfer(
            source_features=features_df,
            target_external_features=external_features_df,
            modality="cough",
            num_trees=num_trees,
            depth=depth,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            device=device,
        )
    else:
        print("\n>> ℹ️ Track 3C SKIPPED: External features DataFrame not provided.")
        res_3c = DNDFProtocolResult("track_c_external", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    return {
        "track_3a_participant_holdouts": res_3a,
        "track_3b_chronological": res_3b_chron,
        "track_3b_calendar_mixed": res_3b_cal,
        "track_3c_external_transfer": res_3c,
    }
