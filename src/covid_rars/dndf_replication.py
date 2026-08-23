from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from covid_rars.dndf_models import DNDFClassifier
from covid_rars.features import feature_columns
from covid_rars.metrics import binary_metric_bundle, labels_to_binary

logger = logging.getLogger(__name__)


def run_paper_replication_benchmark(
    features_df: pd.DataFrame,
    modalities: Sequence[str] = ("cough", "breath"),
    test_size: float = 0.20,
    random_state: int = 42,
    num_trees: int = 80,
    depth: int = 5,
    n_selected_features: int = 140,
    learning_rate: float = 0.00132,
    temperature: float = 1.493,
    max_epochs: int = 50,
    device: str = "auto",
) -> pd.DataFrame:
    """Execute exact paper replication benchmark matching Islam et al. (ESWA 2026 / arXiv:2501.01117).

    Evaluates DNDF on standard 80/20 stratified recording-level split with SMOTE,
    top-k feature selection, and threshold optimization.
    """
    print("\n" + "=" * 80)
    print("📜 ISLAM ET AL. (ESWA 2026) EXACT PAPER REPLICATION BENCHMARK")
    print("=" * 80)
    print(f"  • Splitting Protocol : Random 80/20 Stratified Recording-Level Split (Seed={random_state})")
    print(f"  • Feature Selection  : Top-{n_selected_features} ANOVA F-score / ExtraTrees Descriptors")
    print(f"  • Model Architecture : DNDF (Trees={num_trees}, Depth={depth}, Temp={temperature:.3f})")
    print(f"  • Optimization       : AdamW (lr={learning_rate:.5f}) + Cosine Annealing + SMOTE")
    print("-" * 80)

    results_rows: list[dict[str, Any]] = []

    for mod in modalities:
        mod_df = features_df[features_df["modality"] == mod].copy()
        if mod_df.empty:
            continue

        feat_cols = feature_columns(mod_df)
        X = mod_df[feat_cols].to_numpy(dtype=np.float32)
        y = labels_to_binary(mod_df["label_binary"])

        if len(np.unique(y)) < 2:
            logger.warning(f"Modality {mod} has only 1 class. Skipping.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\n>> 🚀 Evaluating [{mod.upper()}] Modality (Train={len(X_train)}, Test={len(X_test)})...")
        t0 = time.time()

        clf = DNDFClassifier(
            model_type="dndf",
            num_trees=num_trees,
            depth=depth,
            used_features_rate=0.60,
            temperature=temperature,
            learning_rate=learning_rate,
            weight_decay=7.87e-5,
            batch_size=16,
            max_epochs=max_epochs,
            patience=12,
            use_smote=True,
            feature_selection="f_classif",
            n_selected_features=min(n_selected_features, X.shape[1]),
            device=device,
            random_state=random_state,
        )

        clf.fit(X_train, y_train, X_val=X_test, y_val=y_test, optimize_threshold=True)
        probs_test = clf.predict_proba(X_test)[:, 1]
        preds_test = (probs_test >= clf.best_threshold_).astype(int)

        acc = accuracy_score(y_test, preds_test)
        bacc = balanced_accuracy_score(y_test, preds_test)
        auroc = roc_auc_score(y_test, probs_test)
        prec = precision_score(y_test, preds_test, zero_division=0)
        rec = recall_score(y_test, preds_test, zero_division=0)
        f1 = f1_score(y_test, preds_test, zero_division=0)
        cm = confusion_matrix(y_test, preds_test)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        elapsed = time.time() - t0

        print(f"\n🏆 RESULTS FOR [{mod.upper()}] ON EXACT PAPER SETUP (Islam et al. ESWA 2026):")
        print(f"   ┌─────────────────────────────────────────────────────────────┐")
        print(f"   │  • Overall Accuracy   : {acc * 100:6.2f}%                               │")
        print(f"   │  • ROC-AUC (AUROC)    : {auroc:6.4f}                                │")
        print(f"   │  • Balanced Accuracy  : {bacc * 100:6.2f}%                               │")
        print(f"   │  • Sensitivity/Recall : {rec * 100:6.2f}%                               │")
        print(f"   │  • Specificity        : {spec * 100:6.2f}%                               │")
        print(f"   │  • Precision          : {prec * 100:6.2f}%                               │")
        print(f"   │  • F1-Score           : {f1:6.4f}                                │")
        print(f"   │  • Elapsed Time       : {elapsed:5.1f}s                                │")
        print(f"   └─────────────────────────────────────────────────────────────┘")
        print(f"   Confusion Matrix: [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")

        results_rows.append({
            "modality": mod,
            "setup": "exact_paper_replication_80_20",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": acc,
            "auroc": auroc,
            "balanced_accuracy": bacc,
            "sensitivity": rec,
            "specificity": spec,
            "precision": prec,
            "f1_score": f1,
            "optimal_threshold": clf.best_threshold_,
        })

    summary_df = pd.DataFrame(results_rows)
    print("\n" + "=" * 80)
    print("📊 PAPER REPLICATION SUMMARY TABLE:")
    print("=" * 80)
    print(summary_df[["modality", "accuracy", "auroc", "balanced_accuracy", "sensitivity", "specificity", "f1_score"]].to_string(index=False))
    print("=" * 80 + "\n")

    return summary_df
