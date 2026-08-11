from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from covid_rars.features import feature_columns
from covid_rars.metrics import (
    best_threshold_by_balanced_accuracy,
    binary_metric_bundle,
    labels_to_binary,
)


@dataclass(frozen=True)
class SwarmFeatureSearchResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    selection: pd.DataFrame


def _safe_f_scores(x: pd.DataFrame, y: np.ndarray) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        scores, _ = f_classif(x.fillna(0.0).to_numpy(dtype=float), y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)
    return pd.Series(scores, index=x.columns).sort_values(ascending=False)


def _make_classifier(classifier: str, random_state: int) -> object:
    if classifier == "logistic":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=4000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if classifier == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception:
            classifier = "extra_trees"
        else:
            return LGBMClassifier(
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=12,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=3.0,
                objective="binary",
                n_jobs=-1,
                random_state=random_state,
                verbosity=-1,
            )
    if classifier == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=600,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"Unknown swarm classifier: {classifier}")


def _predict_probability(model: object, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-score))
    return np.asarray(model.predict(x), dtype=float)


def _repair_mask(
    mask: np.ndarray,
    scores: np.ndarray,
    rng: np.random.Generator,
    min_features: int,
    max_features: int,
) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    if out.sum() < min_features:
        order = np.argsort(scores)[::-1]
        out[order[:min_features]] = True
    if out.sum() > max_features:
        selected = np.flatnonzero(out)
        ranked_selected = selected[np.argsort(scores[selected])[::-1]]
        keep = set(ranked_selected[:max_features])
        out[:] = False
        out[list(keep)] = True
    if not out.any():
        out[int(rng.integers(0, len(out)))] = True
    return out


def _binary_pso_select(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
    classifier: str,
    particles: int,
    iterations: int,
    min_features: int,
    max_features: int,
    sparsity_penalty: float,
    random_state: int,
    verbose: bool = False,
) -> tuple[list[str], float, list[dict[str, object]]]:
    rng = np.random.default_rng(random_state)
    n_features = x_train.shape[1]
    scores = _safe_f_scores(x_train, y_train).reindex(x_train.columns).fillna(0.0).to_numpy(dtype=float)
    if np.all(scores == 0):
        scores = np.ones_like(scores)

    positions = rng.random((particles, n_features)) < min(0.35, max_features / max(1, n_features))
    velocities = rng.normal(0.0, 0.5, size=(particles, n_features))
    positions = np.vstack([
        _repair_mask(mask, scores, rng, min_features=min_features, max_features=max_features)
        for mask in positions
    ])
    personal_best = positions.copy()
    personal_scores = np.full(particles, -np.inf)
    global_best = positions[0].copy()
    global_score = -np.inf
    cache: dict[tuple[int, ...], float] = {}
    history: list[dict[str, object]] = []

    def evaluate(mask: np.ndarray) -> float:
        repaired = _repair_mask(mask, scores, rng, min_features=min_features, max_features=max_features)
        key = tuple(np.flatnonzero(repaired).tolist())
        if key in cache:
            return cache[key]
        cols = list(x_train.columns[list(key)])
        model = _make_classifier(classifier, random_state=random_state)
        model.fit(x_train[cols].fillna(0.0), y_train)
        prob = _predict_probability(model, x_validation[cols].fillna(0.0))
        if len(np.unique(y_validation)) < 2:
            auc = 0.5
        else:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(y_validation, prob))
        score = auc - float(sparsity_penalty) * (len(cols) / max(1, n_features))
        cache[key] = score
        return score

    for iteration in range(1, int(iterations) + 1):
        for idx in range(int(particles)):
            score = evaluate(positions[idx])
            if score > personal_scores[idx]:
                personal_scores[idx] = score
                personal_best[idx] = positions[idx].copy()
            if score > global_score:
                global_score = score
                global_best = positions[idx].copy()
        history.append(
            {
                "iteration": iteration,
                "best_objective": float(global_score),
                "mean_personal_best": float(np.mean(personal_scores[np.isfinite(personal_scores)])),
                "selected_features": int(global_best.sum()),
            }
        )
        if verbose:
            print(
                f"  PSO iteration {iteration}/{int(iterations)}: "
                f"best_objective={float(global_score):.4f}, "
                f"selected_features={int(global_best.sum())}",
                flush=True,
            )
        r1 = rng.random(size=positions.shape)
        r2 = rng.random(size=positions.shape)
        velocities = (
            0.72 * velocities
            + 1.45 * r1 * (personal_best.astype(float) - positions.astype(float))
            + 1.45 * r2 * (global_best.astype(float) - positions.astype(float))
        )
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(velocities, -20, 20)))
        positions = rng.random(size=positions.shape) < probabilities
        positions = np.vstack([
            _repair_mask(mask, scores, rng, min_features=min_features, max_features=max_features)
            for mask in positions
        ])

    final_mask = _repair_mask(global_best, scores, rng, min_features=min_features, max_features=max_features)
    selected = x_train.columns[final_mask].tolist()
    return selected, float(global_score), history


def _participant_average(predictions: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "participant_id",
        "label_binary",
        "split",
        "evaluation_protocol",
        "analysis_family",
        "model_name",
        "modality",
        "feature_strategy",
    ]
    return (
        predictions.groupby(group_cols, dropna=False)
        .agg(probability=("probability", "mean"), n_recordings=("recording_id", "nunique"))
        .reset_index()
    )


def _metric_row(frame: pd.DataFrame, threshold: float, extra: dict[str, object]) -> dict[str, object]:
    row = binary_metric_bundle(
        labels_to_binary(frame["label_binary"]),
        frame["probability"].astype(float).to_numpy(),
        threshold=threshold,
    )
    row.update(extra)
    row["n_participants"] = float(frame["participant_id"].nunique())
    return row


def run_swarm_feature_search(
    features: pd.DataFrame,
    modalities: Iterable[str] = ("cough", "breath", "speech"),
    classifier: str = "lightgbm",
    particles: int = 12,
    iterations: int = 16,
    max_candidate_features: int = 256,
    min_selected_features: int = 8,
    max_selected_features: int = 64,
    sparsity_penalty: float = 0.01,
    random_state: int = 42,
    verbose: bool = False,
) -> SwarmFeatureSearchResult:
    cols = feature_columns(features)
    if not cols:
        raise ValueError("No numeric feature columns are available for swarm feature search")
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []

    for modality in modalities:
        df = features[
            features["modality"].astype(str).eq(str(modality))
            & features["label_binary"].isin(["positive", "negative"])
            & features["split"].isin(["train", "validation", "test"])
        ].copy()
        if df.empty:
            continue
        train = df[df["split"].eq("train")]
        validation = df[df["split"].eq("validation")]
        test = df[df["split"].eq("test")]
        if train.empty or validation.empty or test.empty or train["label_binary"].nunique() < 2:
            continue
        if verbose:
            print(
                f"Running swarm feature search for modality={modality} "
                f"(train={len(train)}, validation={len(validation)}, test={len(test)})",
                flush=True,
            )

        y_train = labels_to_binary(train["label_binary"])
        y_validation = labels_to_binary(validation["label_binary"])
        ranked = _safe_f_scores(train[cols], y_train)
        candidate_cols = ranked.head(int(max_candidate_features)).index.tolist()
        max_selected = min(int(max_selected_features), len(candidate_cols))
        min_selected = min(max(1, int(min_selected_features)), max_selected)
        selected_cols, objective, history = _binary_pso_select(
            train[candidate_cols],
            y_train,
            validation[candidate_cols],
            y_validation,
            classifier=classifier,
            particles=int(particles),
            iterations=int(iterations),
            min_features=min_selected,
            max_features=max_selected,
            sparsity_penalty=float(sparsity_penalty),
            random_state=int(random_state),
            verbose=verbose,
        )
        model = _make_classifier(classifier, random_state=random_state)
        model.fit(train[selected_cols].fillna(0.0), y_train)
        split_predictions = []
        for split_name, split_df in (("validation", validation), ("test", test)):
            prob = _predict_probability(model, split_df[selected_cols].fillna(0.0))
            split_predictions.append(
                pd.DataFrame(
                    {
                        "recording_id": split_df["recording_id"].astype(str).to_numpy(),
                        "participant_id": split_df["participant_id"].astype(str).to_numpy(),
                        "dataset": split_df.get("dataset", pd.Series(["coswara"] * len(split_df))).to_numpy(),
                        "modality": str(modality),
                        "submodality": split_df.get("submodality", pd.Series([str(modality)] * len(split_df))).to_numpy(),
                        "label_binary": split_df["label_binary"].to_numpy(),
                        "split": split_name,
                        "evaluation_protocol": "sota_swarm_internal_protocol",
                        "analysis_family": "sota_swarm_feature_search",
                        "model_name": f"binary_pso_{classifier}",
                        "feature_strategy": "binary_pso_validation_selected",
                        "probability": prob,
                    }
                )
            )
        pred = pd.concat(split_predictions, ignore_index=True)
        prediction_rows.append(pred)
        participant = _participant_average(pred)
        val_participant = participant[participant["split"].eq("validation")]
        test_participant = participant[participant["split"].eq("test")]
        threshold = best_threshold_by_balanced_accuracy(
            labels_to_binary(val_participant["label_binary"]),
            val_participant["probability"].astype(float).to_numpy(),
        )
        for split_name, group in (("validation", val_participant), ("test", test_participant)):
            metric_rows.append(
                _metric_row(
                    group,
                    threshold,
                    {
                        "evaluation_protocol": "sota_swarm_internal_protocol",
                        "analysis_family": "sota_swarm_feature_search",
                        "model_name": f"binary_pso_{classifier}",
                        "modality": str(modality),
                        "feature_strategy": "binary_pso_validation_selected",
                        "metric_split": split_name,
                        "threshold_source": "validation_balanced_accuracy",
                        "skipped": False,
                        "n_candidate_features": float(len(candidate_cols)),
                        "n_selected_features": float(len(selected_cols)),
                        "selected_features": ";".join(selected_cols),
                    },
                )
            )
        best_validation_auc = float(max((row["best_objective"] for row in history), default=objective))
        selection_rows.append(
            {
                "evaluation_protocol": "sota_swarm_internal_protocol",
                "analysis_family": "sota_swarm_feature_search",
                "model_name": f"binary_pso_{classifier}",
                "modality": str(modality),
                "feature_strategy": "binary_pso_validation_selected",
                "n_candidate_features": len(candidate_cols),
                "n_selected_features": len(selected_cols),
                "selected_features": ";".join(selected_cols),
                "best_objective": objective,
                "best_validation_objective": best_validation_auc,
                "particles": int(particles),
                "iterations": int(iterations),
            }
        )
        if verbose:
            print(
                f"Finished modality={modality}: selected {len(selected_cols)} features; "
                f"objective={objective:.4f}",
                flush=True,
            )

    return SwarmFeatureSearchResult(
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame(),
        selection=pd.DataFrame(selection_rows),
    )
