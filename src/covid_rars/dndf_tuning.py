from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from covid_rars.dndf_models import DNDFClassifier
from covid_rars.metrics import binary_metric_bundle

logger = logging.getLogger(__name__)


def optimize_dndf_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 25,
    timeout_seconds: int = 600,
    metric: str = "auroc",
    device: str = "auto",
    random_state: int = 42,
) -> dict[str, Any]:
    """Perform Bayesian Hyperparameter Optimization using Optuna (TPE Sampler) with rich live feedback."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise RuntimeError("Optuna is required for hyperparameter tuning. Install via: pip install optuna") from exc

    best_score_holder = {"score": -float("inf"), "trial_num": 0}
    start_time = time.time()

    print("\n" + "=" * 78)
    print(f"🔍 STARTING OPTUNA BAYESIAN HYPERPARAMETER OPTIMIZATION ({n_trials} TRIALS)")
    print("=" * 78)
    print(f"  • Target Metric      : {metric.upper()}")
    print(f"  • Training Samples   : {len(X_train)} (Positive: {int(np.sum(y_train == 1))}, Negative: {int(np.sum(y_train == 0))})")
    print(f"  • Validation Samples : {len(X_val)} (Positive: {int(np.sum(y_val == 1))}, Negative: {int(np.sum(y_val == 0))})")
    print(f"  • Feature Pool Size  : {X_train.shape[1]} features")
    print(f"  • Execution Device   : {device}")
    print("-" * 78)

    def objective(trial: optuna.Trial) -> float:
        trial_t0 = time.time()
        num_trees = trial.suggest_int("num_trees", 30, 80, step=10)
        depth = trial.suggest_int("depth", 3, 6)
        used_features_rate = trial.suggest_float("used_features_rate", 0.5, 0.85, step=0.05)
        temperature = trial.suggest_float("temperature", 0.5, 2.0, log=True)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.015, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        n_selected_features = trial.suggest_int("n_selected_features", 40, 140, step=20)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        clf = DNDFClassifier(
            model_type="dndf",
            num_trees=num_trees,
            depth=depth,
            used_features_rate=used_features_rate,
            temperature=temperature,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=40,
            patience=8,
            use_smote=True,
            feature_selection="f_classif",
            n_selected_features=n_selected_features,
            device=device,
            random_state=random_state,
        )

        clf.fit(X_train, y_train, X_val=X_val, y_val=y_val, optimize_threshold=True)

        val_probs = clf.predict_proba(X_val)[:, 1]
        m_bundle = binary_metric_bundle(y_val, val_probs, threshold=clf.best_threshold_)

        target_val = float(m_bundle.get(metric, m_bundle.get("auroc", 0.5)))
        bacc = float(m_bundle.get("balanced_accuracy", 0.5))
        duration = time.time() - trial_t0

        is_best = target_val > best_score_holder["score"]
        if is_best:
            best_score_holder["score"] = target_val
            best_score_holder["trial_num"] = trial.number
            star_tag = " 🌟 [NEW BEST]"
        else:
            star_tag = ""

        print(
            f"  [Trial {trial.number + 1:02d}/{n_trials:02d}] "
            f"trees={num_trees:2d}, depth={depth}, lr={learning_rate:.4f}, k={n_selected_features:3d}, "
            f"sub={used_features_rate:.2f}, temp={temperature:.2f} | "
            f"Val {metric.upper()}: {target_val:.4f} | BAcc: {bacc*100:.1f}% ({duration:.1f}s){star_tag}"
        )

        return target_val

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, show_progress_bar=False)

    total_time = time.time() - start_time
    print("-" * 78)
    print(f"✅ OPTUNA TUNING COMPLETE in {total_time:.1f}s across {len(study.trials)} trials")
    print(f"🏆 BEST TRIAL #{study.best_trial.number + 1}: Peak Val {metric.upper()} = {study.best_value:.4f}")
    print("📋 BEST HYPERPARAMETERS:")
    for k, v in study.best_params.items():
        print(f"    • {k:<22s}: {v}")
    print("=" * 78 + "\n")

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "n_trials_completed": len(study.trials),
        "study": study,
    }
