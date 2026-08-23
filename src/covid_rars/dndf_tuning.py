from __future__ import annotations

import logging
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
    """Perform Bayesian Hyperparameter Optimization using Optuna (TPE Sampler)."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise RuntimeError("Optuna is required for hyperparameter tuning. Install via: pip install optuna") from exc

    def objective(trial: optuna.Trial) -> float:
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
        return target_val

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    logger.info(f"Starting Optuna search ({n_trials} trials, optimizing {metric.upper()})...")
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, show_progress_bar=False)

    logger.info(f"Optuna Best Trial #{study.best_trial.number}: {metric.upper()} = {study.best_value:.4f}")
    logger.info(f"Optuna Best Hyperparameters: {study.best_params}")

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "n_trials_completed": len(study.trials),
        "study": study,
    }
