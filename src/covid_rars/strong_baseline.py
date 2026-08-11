from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from covid_rars.features import feature_columns
from covid_rars.metrics import (
    best_threshold_by_balanced_accuracy,
    binary_metric_bundle,
    labels_to_binary,
)
from covid_rars.strong_baseline_protocol import (
    DEFAULT_MODALITIES,
    ProtocolResult,
    build_clean_internal_protocol,
)


DEFAULT_MODEL_NAMES = (
    "logistic_l2_f80",
    "logistic_smote_f80",
    "extra_trees_f100",
    "extra_trees_smote_f100",
    "random_forest_f80",
    "random_forest_smote_f80",
    "svc_rbf_f60",
    "xgboost_smote_f80",
    "lightgbm_smote_f80",
    "catboost_smote_f80",
    "optuna_validation_search",
)


def _safe_f_classif(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Features .* are constant\.", category=UserWarning)
            warnings.filterwarnings("ignore", message=r"invalid value encountered in divide", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message=r"divide by zero encountered in divide", category=RuntimeWarning)
            scores, p_values = f_classif(x, y)
    max_float = np.finfo(np.float64).max
    scores = np.nan_to_num(scores, nan=0.0, posinf=max_float, neginf=0.0)
    p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=0.0)
    return scores, p_values


def _preprocess_steps(percentile: int, scale: bool = False) -> list[tuple[str, object]]:
    """Shared numeric preprocessing for high-dimensional acoustic features.

    VarianceThreshold removes exactly constant columns before the univariate
    F-score selector. Without this, sklearn emits repeated constant-feature and
    invalid-divide warnings for fallback acoustic descriptors that are all zero
    in a modality subset.
    """
    steps: list[tuple[str, object]] = [("variance", VarianceThreshold())]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("select", SelectPercentile(score_func=_safe_f_classif, percentile=int(percentile))))
    return steps


@dataclass(frozen=True)
class StrongBaselineResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    selection: pd.DataFrame
    protocol_audit: pd.DataFrame
    participant_audit: pd.DataFrame


def _make_model(model_name: str, random_state: int = 42) -> Pipeline:
    def smote_pipeline(preprocess_steps: list[tuple[str, object]], estimator: object) -> object:
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("imbalanced-learn is required for SMOTE strong-baseline models") from exc
        return ImbPipeline(
            [
                *preprocess_steps,
                ("smote", SMOTE(k_neighbors=3, random_state=random_state)),
                ("model", estimator),
            ]
        )

    def optional_classifier(name: str):
        if name == "xgboost":
            try:
                from xgboost import XGBClassifier
            except Exception as exc:  # pragma: no cover - depends on optional dependency
                raise RuntimeError("xgboost is required for xgboost_smote_f80") from exc
            return XGBClassifier(
                n_estimators=800,
                max_depth=3,
                learning_rate=0.025,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=2.0,
                reg_lambda=2.0,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=random_state,
            )
        if name == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except Exception as exc:  # pragma: no cover - depends on optional dependency
                raise RuntimeError("lightgbm is required for lightgbm_smote_f80") from exc
            return LGBMClassifier(
                n_estimators=900,
                learning_rate=0.025,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                objective="binary",
                n_jobs=-1,
                random_state=random_state,
                verbosity=-1,
            )
        if name == "catboost":
            try:
                from catboost import CatBoostClassifier
            except Exception as exc:  # pragma: no cover - depends on optional dependency
                raise RuntimeError("catboost is required for catboost_smote_f80") from exc
            return CatBoostClassifier(
                iterations=900,
                depth=5,
                learning_rate=0.025,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=random_state,
                verbose=False,
                allow_writing_files=False,
            )
        raise ValueError(f"Unknown optional classifier: {name}")

    if model_name == "logistic_l2_f80":
        return Pipeline(
            [
                *_preprocess_steps(percentile=80, scale=True),
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
    if model_name == "logistic_smote_f80":
        return smote_pipeline(
            _preprocess_steps(percentile=80, scale=True),
            LogisticRegression(
                C=1.0,
                max_iter=4000,
                random_state=random_state,
            ),
        )
    if model_name == "extra_trees_f100":
        return Pipeline(
            [
                *_preprocess_steps(percentile=100),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=700,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_name == "extra_trees_smote_f100":
        return smote_pipeline(
            _preprocess_steps(percentile=100),
            ExtraTreesClassifier(
                n_estimators=800,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=random_state,
            ),
        )
    if model_name == "random_forest_f80":
        return Pipeline(
            [
                *_preprocess_steps(percentile=80),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=700,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_name == "random_forest_smote_f80":
        return smote_pipeline(
            _preprocess_steps(percentile=80),
            RandomForestClassifier(
                n_estimators=800,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=random_state,
            ),
        )
    if model_name == "svc_rbf_f60":
        return Pipeline(
            [
                *_preprocess_steps(percentile=60, scale=True),
                (
                    "model",
                    CalibratedClassifierCV(
                        estimator=SVC(
                            C=2.0,
                            gamma="scale",
                            kernel="rbf",
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        )
    if model_name == "xgboost_smote_f80":
        return smote_pipeline(
            _preprocess_steps(percentile=80),
            optional_classifier("xgboost"),
        )
    if model_name == "lightgbm_smote_f80":
        return smote_pipeline(
            _preprocess_steps(percentile=80),
            optional_classifier("lightgbm"),
        )
    if model_name == "catboost_smote_f80":
        return smote_pipeline(
            _preprocess_steps(percentile=80),
            optional_classifier("catboost"),
        )
    raise ValueError(f"Unknown strong-baseline model: {model_name}")


def _predict_probability(model: object, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    raise ValueError("Model does not expose predict_proba or decision_function")


def _model_from_optuna_params(params: dict[str, object], random_state: int) -> Pipeline:
    family = str(params["family"])
    percentile = int(params["percentile"])
    if family == "logistic":
        return Pipeline(
            [
                *_preprocess_steps(percentile=percentile, scale=True),
                (
                    "model",
                    LogisticRegression(
                        C=float(params["logistic_c"]),
                        class_weight="balanced",
                        max_iter=4000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if family == "extra_trees":
        return Pipeline(
            [
                *_preprocess_steps(percentile=percentile),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=int(params["n_estimators"]),
                        min_samples_leaf=int(params["min_samples_leaf"]),
                        max_depth=None if params["max_depth"] == "none" else int(params["max_depth"]),
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if family == "random_forest":
        return Pipeline(
            [
                *_preprocess_steps(percentile=percentile),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=int(params["n_estimators"]),
                        min_samples_leaf=int(params["min_samples_leaf"]),
                        max_depth=None if params["max_depth"] == "none" else int(params["max_depth"]),
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown Optuna model family: {family}")


def _fit_optuna_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
    random_state: int,
    n_trials: int,
) -> Pipeline:
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("optuna is required for optuna_validation_search") from exc

    if n_trials <= 0:
        raise RuntimeError("optuna_validation_search requires --optuna-trials > 0")

    def objective(trial) -> float:
        family = trial.suggest_categorical("family", ["logistic", "extra_trees", "random_forest"])
        params: dict[str, object] = {
            "family": family,
            "percentile": trial.suggest_int("percentile", 40, 100, step=10),
        }
        if family == "logistic":
            params["logistic_c"] = trial.suggest_float("logistic_c", 1e-3, 30.0, log=True)
        else:
            params["n_estimators"] = trial.suggest_int("n_estimators", 300, 1000, step=100)
            params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 5)
            params["max_depth"] = trial.suggest_categorical("max_depth", ["none", "4", "8", "12", "16"])
        model = _model_from_optuna_params(params, random_state=random_state)
        model.fit(x_train, y_train)
        probabilities = _predict_probability(model, x_validation)
        if len(np.unique(y_validation)) < 2:
            return 0.5
        return float(roc_auc_score(y_validation, probabilities))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    best = _model_from_optuna_params(study.best_trial.params, random_state=random_state)
    best.fit(x_train, y_train)
    return best


def _participant_average(predictions: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "participant_id",
        "label_binary",
        "split",
        "model_name",
        "modality",
        "evaluation_protocol",
        "analysis_family",
    ]
    optional_cols = ["modality_combination", "fusion_method"]
    group_cols.extend([col for col in optional_cols if col in predictions.columns])
    return (
        predictions.groupby(group_cols, dropna=False)
        .agg(probability=("probability", "mean"), n_recordings=("recording_id", "nunique"))
        .reset_index()
    )


def _metric_row(
    participant_predictions: pd.DataFrame,
    threshold: float,
    extra: dict[str, object],
) -> dict[str, object]:
    y_true = labels_to_binary(participant_predictions["label_binary"])
    y_prob = participant_predictions["probability"].astype(float).to_numpy()
    row: dict[str, object] = binary_metric_bundle(y_true, y_prob, threshold=threshold)
    row.update(extra)
    row["n_participants"] = float(participant_predictions["participant_id"].nunique())
    return row


def _prediction_frame(
    source: pd.DataFrame,
    probabilities: np.ndarray,
    model_name: str,
    modality: str,
    split: str,
    evaluation_protocol: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recording_id": source["recording_id"].astype(str).to_numpy(),
            "participant_id": source["participant_id"].astype(str).to_numpy(),
            "dataset": source.get("dataset", pd.Series(["coswara"] * len(source))).to_numpy(),
            "modality": modality,
            "submodality": source.get("submodality", pd.Series(["unknown"] * len(source))).to_numpy(),
            "label_binary": source["label_binary"].to_numpy(),
            "split": split,
            "model_name": model_name,
            "analysis_family": "strong_audio_modality",
            "evaluation_protocol": evaluation_protocol,
            "probability": probabilities,
        }
    )


def _participant_prediction_frame(
    source: pd.DataFrame,
    probabilities: np.ndarray,
    model_name: str,
    split: str,
    analysis_family: str,
    modality_combination: str,
    fusion_method: str | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "recording_id": source["participant_id"].astype(str) + f"::{modality_combination}",
            "participant_id": source["participant_id"].astype(str).to_numpy(),
            "dataset": source.get("dataset", pd.Series(["coswara"] * len(source))).to_numpy(),
            "modality": "multimodal",
            "submodality": modality_combination,
            "label_binary": source["label_binary"].to_numpy(),
            "split": split,
            "model_name": model_name,
            "analysis_family": analysis_family,
            "evaluation_protocol": "clean_internal_protocol",
            "modality_combination": modality_combination,
            "probability": probabilities,
        }
    )
    if fusion_method is not None:
        out["fusion_method"] = fusion_method
    return out


def _has_two_classes(df: pd.DataFrame) -> bool:
    return df["label_binary"].isin(["positive", "negative"]).all() and df["label_binary"].nunique() == 2


def _top_k_modality_ensemble(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if top_k < 2 or metrics.empty or predictions.empty:
        return pd.DataFrame(), pd.DataFrame()

    skipped = metrics["skipped"].fillna(False).astype(bool) if "skipped" in metrics.columns else False
    candidates = metrics[
        metrics["analysis_family"].eq("strong_audio_modality")
        & metrics["metric_split"].eq("validation")
        & ~skipped
    ].copy()
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    candidates["auroc"] = pd.to_numeric(candidates["auroc"], errors="coerce")
    candidates["auprc"] = pd.to_numeric(candidates["auprc"], errors="coerce")

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for modality, group in candidates.groupby("modality", dropna=False):
        ranked = group.sort_values(["auroc", "auprc", "model_name"], ascending=[False, False, True])
        selected_models = [str(name) for name in ranked["model_name"].dropna().head(top_k)]
        if len(selected_models) < 2:
            continue
        source = predictions[
            predictions["analysis_family"].eq("strong_audio_modality")
            & predictions["modality"].astype(str).eq(str(modality))
            & predictions["model_name"].astype(str).isin(selected_models)
        ].copy()
        if source.empty:
            continue

        ensemble_name = f"top_{len(selected_models)}_validation_ensemble"
        group_cols = [
            "recording_id",
            "participant_id",
            "dataset",
            "modality",
            "submodality",
            "label_binary",
            "split",
            "analysis_family",
            "evaluation_protocol",
        ]
        averaged = (
            source.groupby(group_cols, dropna=False)
            .agg(
                probability=("probability", "mean"),
                n_ensemble_models=("model_name", "nunique"),
            )
            .reset_index()
        )
        averaged["model_name"] = ensemble_name
        averaged["ensemble_members"] = ",".join(selected_models)
        prediction_rows.append(averaged)

        participant_pred = _participant_average(averaged)
        validation_participant = participant_pred[participant_pred["split"].eq("validation")]
        test_participant = participant_pred[participant_pred["split"].eq("test")]
        if validation_participant.empty or test_participant.empty or validation_participant["label_binary"].nunique() < 2:
            continue
        threshold = best_threshold_by_balanced_accuracy(
            labels_to_binary(validation_participant["label_binary"]),
            validation_participant["probability"].astype(float).to_numpy(),
        )
        for split_name, split_predictions in (("validation", validation_participant), ("test", test_participant)):
            metric_rows.append(
                _metric_row(
                    split_predictions,
                    threshold=threshold,
                    extra={
                        "evaluation_protocol": "clean_internal_protocol",
                        "analysis_family": "strong_audio_modality",
                        "model_name": ensemble_name,
                        "modality": modality,
                        "metric_split": split_name,
                        "threshold_source": "validation_balanced_accuracy",
                        "skipped": False,
                        "n_ensemble_models": float(len(selected_models)),
                        "ensemble_members": ",".join(selected_models),
                    },
                )
            )

    ensemble_metrics = pd.DataFrame(metric_rows)
    ensemble_predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    return ensemble_metrics, ensemble_predictions


def _prepare_features_for_protocol(
    features: pd.DataFrame,
    metadata: pd.DataFrame | None,
    modalities: Iterable[str],
    require_quality_ok: bool,
) -> tuple[pd.DataFrame, ProtocolResult]:
    if metadata is None:
        metadata = features[[c for c in features.columns if c in {
            "recording_id",
            "participant_id",
            "dataset",
            "modality",
            "submodality",
            "label_binary",
            "split",
            "label_raw",
            "label_group",
            "quality_flag",
        }]].drop_duplicates("recording_id")

    protocol = build_clean_internal_protocol(
        metadata,
        modalities=modalities,
        require_quality_ok=require_quality_ok,
    )
    if protocol.metadata.empty:
        return features.iloc[0:0].copy(), protocol

    protocol_cols = [
        "recording_id",
        "label_binary",
        "split",
        "evaluation_protocol",
        "label_protocol",
    ]
    protocol_map = (
        protocol.metadata[protocol_cols]
        .drop_duplicates("recording_id")
        .rename(columns={"recording_id": "_protocol_recording_id"})
    )
    merge_features = features.copy()
    merge_features["_protocol_recording_id"] = merge_features.get(
        "source_recording_id",
        merge_features["recording_id"],
    ).fillna(merge_features["recording_id"]).astype(str)
    prepared = merge_features.drop(columns=[c for c in ["label_binary", "split"] if c in merge_features.columns]).merge(
        protocol_map,
        on="_protocol_recording_id",
        how="inner",
    ).drop(columns=["_protocol_recording_id"])
    return prepared, protocol


def train_strong_modality_models(
    features: pd.DataFrame,
    modalities: Iterable[str] = DEFAULT_MODALITIES,
    model_names: Iterable[str] = DEFAULT_MODEL_NAMES,
    random_state: int = 42,
    optuna_trials: int = 25,
    ensemble_top_k: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, str], object]]:
    cols = feature_columns(features)
    if not cols:
        raise ValueError("No numeric feature columns are available for strong-baseline training")

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    fitted: dict[tuple[str, str], object] = {}

    for modality in modalities:
        modality_df = features[
            (features["modality"].astype(str) == str(modality))
            & (features["label_binary"].isin(["positive", "negative"]))
            & (features["split"].isin(["train", "validation", "test"]))
        ].copy()
        if modality_df.empty:
            continue
        train = modality_df[modality_df["split"].eq("train")]
        validation = modality_df[modality_df["split"].eq("validation")]
        test = modality_df[modality_df["split"].eq("test")]
        if train.empty or validation.empty or test.empty or not _has_two_classes(train):
            continue

        for model_name in model_names:
            x_train = train[cols].fillna(0.0)
            y_train = labels_to_binary(train["label_binary"])
            x_validation = validation[cols].fillna(0.0)
            y_validation = labels_to_binary(validation["label_binary"])
            try:
                if str(model_name) == "optuna_validation_search":
                    model = _fit_optuna_model(
                        x_train,
                        y_train,
                        x_validation,
                        y_validation,
                        random_state=random_state,
                        n_trials=optuna_trials,
                    )
                else:
                    model = _make_model(str(model_name), random_state=random_state)
            except Exception as exc:
                metric_rows.append(
                    {
                        "evaluation_protocol": "clean_internal_protocol",
                        "analysis_family": "strong_audio_modality",
                        "model_name": model_name,
                        "modality": modality,
                        "metric_split": "skipped",
                        "skipped": True,
                        "skip_reason": str(exc),
                    }
                )
                continue
            try:
                model.fit(x_train, y_train)
            except Exception as exc:
                metric_rows.append(
                    {
                        "evaluation_protocol": "clean_internal_protocol",
                        "analysis_family": "strong_audio_modality",
                        "model_name": model_name,
                        "modality": modality,
                        "metric_split": "skipped",
                        "skipped": True,
                        "skip_reason": str(exc),
                    }
                )
                continue

            fitted[(str(modality), str(model_name))] = model
            split_predictions: list[pd.DataFrame] = []
            for split_name, split_df in (("validation", validation), ("test", test)):
                prob = _predict_probability(model, split_df[cols].fillna(0.0))
                split_predictions.append(
                    _prediction_frame(
                        split_df,
                        prob,
                        model_name=str(model_name),
                        modality=str(modality),
                        split=split_name,
                        evaluation_protocol="clean_internal_protocol",
                    )
                )
            all_pred = pd.concat(split_predictions, ignore_index=True)
            prediction_frames.append(all_pred)

            participant_pred = _participant_average(all_pred)
            validation_participant = participant_pred[participant_pred["split"].eq("validation")]
            test_participant = participant_pred[participant_pred["split"].eq("test")]
            if validation_participant.empty or test_participant.empty:
                continue
            threshold = best_threshold_by_balanced_accuracy(
                labels_to_binary(validation_participant["label_binary"]),
                validation_participant["probability"].astype(float).to_numpy(),
            )
            for split_name, group in (("validation", validation_participant), ("test", test_participant)):
                metric_rows.append(
                    _metric_row(
                        group,
                        threshold=threshold,
                        extra={
                            "evaluation_protocol": "clean_internal_protocol",
                            "analysis_family": "strong_audio_modality",
                            "model_name": model_name,
                            "modality": modality,
                            "metric_split": split_name,
                            "threshold_source": "validation_balanced_accuracy",
                            "skipped": False,
                            "n_features": float(len(cols)),
                        },
                    )
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    ensemble_metrics, ensemble_predictions = _top_k_modality_ensemble(
        metrics,
        predictions,
        top_k=ensemble_top_k,
    )
    if not ensemble_metrics.empty:
        metrics = pd.concat([metrics, ensemble_metrics], ignore_index=True, sort=False)
    if not ensemble_predictions.empty:
        predictions = pd.concat([predictions, ensemble_predictions], ignore_index=True, sort=False)
    selection = select_best_modality_models(metrics)
    return metrics, predictions, selection, fitted


def select_best_modality_models(metrics: pd.DataFrame, selection_metric: str = "auroc") -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "evaluation_protocol",
                "modality",
                "selected_model_name",
                "selection_metric",
                "validation_auroc",
                "validation_auprc",
            ]
        )
    skipped = metrics["skipped"].fillna(False).astype(bool) if "skipped" in metrics.columns else False
    candidates = metrics[
        metrics["metric_split"].eq("validation")
        & metrics["analysis_family"].eq("strong_audio_modality")
        & ~skipped
    ].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates[selection_metric] = pd.to_numeric(candidates[selection_metric], errors="coerce")
    candidates["auprc"] = pd.to_numeric(candidates.get("auprc"), errors="coerce")
    candidates = candidates.sort_values(
        ["modality", selection_metric, "auprc", "model_name"],
        ascending=[True, False, False, True],
    )
    selected = candidates.groupby("modality", dropna=False).head(1).copy()
    return pd.DataFrame(
        {
            "evaluation_protocol": selected["evaluation_protocol"].to_numpy(),
            "modality": selected["modality"].to_numpy(),
            "selected_model_name": selected["model_name"].to_numpy(),
            "selection_metric": selection_metric,
            "validation_auroc": selected["auroc"].to_numpy(),
            "validation_auprc": selected["auprc"].to_numpy(),
            "validation_balanced_accuracy": selected["balanced_accuracy"].to_numpy(),
            "threshold": selected["threshold"].to_numpy(),
        }
    )


def _participant_feature_matrix(features: pd.DataFrame, combo: tuple[str, ...], cols: list[str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for modality in combo:
        modality_df = features[
            features["modality"].astype(str).eq(str(modality))
            & features["label_binary"].isin(["positive", "negative"])
            & features["split"].isin(["train", "validation", "test"])
        ].copy()
        if modality_df.empty:
            return pd.DataFrame()
        grouped = (
            modality_df.groupby(["participant_id", "label_binary", "split"], dropna=False)[cols]
            .mean()
            .reset_index()
        )
        grouped = grouped.rename(columns={col: f"{modality}__{col}" for col in cols})
        if merged is None:
            merged = grouped
        else:
            merged = merged.merge(grouped, on=["participant_id", "label_binary", "split"], how="inner")
    return merged if merged is not None else pd.DataFrame()


def train_feature_level_fusion_models(
    features: pd.DataFrame,
    modalities: Iterable[str] = DEFAULT_MODALITIES,
    model_names: Iterable[str] = DEFAULT_MODEL_NAMES,
    random_state: int = 42,
    optuna_trials: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = feature_columns(features)
    if not cols:
        return pd.DataFrame(), pd.DataFrame()

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    modality_list = [str(m) for m in modalities]

    for size in range(2, len(modality_list) + 1):
        for combo in combinations(modality_list, size):
            matrix = _participant_feature_matrix(features, combo, cols)
            if matrix.empty:
                continue
            train = matrix[matrix["split"].eq("train")].copy()
            validation = matrix[matrix["split"].eq("validation")].copy()
            test = matrix[matrix["split"].eq("test")].copy()
            if train.empty or validation.empty or test.empty or not _has_two_classes(train):
                continue

            combo_name = "+".join(combo)
            x_cols = [col for col in matrix.columns if col not in {"participant_id", "label_binary", "split"}]
            x_train = train[x_cols].fillna(0.0)
            y_train = labels_to_binary(train["label_binary"])
            x_validation = validation[x_cols].fillna(0.0)
            y_validation = labels_to_binary(validation["label_binary"])
            x_test = test[x_cols].fillna(0.0)

            for model_name in model_names:
                try:
                    if str(model_name) == "optuna_validation_search":
                        model = _fit_optuna_model(
                            x_train,
                            y_train,
                            x_validation,
                            y_validation,
                            random_state=random_state,
                            n_trials=optuna_trials,
                        )
                    else:
                        model = _make_model(str(model_name), random_state=random_state)
                    model.fit(x_train, y_train)
                except Exception as exc:
                    metric_rows.append(
                        {
                            "evaluation_protocol": "clean_internal_protocol",
                            "analysis_family": "strong_feature_level_fusion",
                            "model_name": model_name,
                            "modality": "multimodal",
                            "modality_combination": combo_name,
                            "metric_split": "skipped",
                            "skipped": True,
                            "skip_reason": str(exc),
                        }
                    )
                    continue

                val_prob = _predict_probability(model, x_validation)
                test_prob = _predict_probability(model, x_test)
                if validation["label_binary"].nunique() < 2:
                    continue
                threshold = best_threshold_by_balanced_accuracy(y_validation, val_prob)
                for split_name, frame, probs in (
                    ("validation", validation, val_prob),
                    ("test", test, test_prob),
                ):
                    pred = _participant_prediction_frame(
                        frame,
                        probs,
                        model_name=str(model_name),
                        split=split_name,
                        analysis_family="strong_feature_level_fusion",
                        modality_combination=combo_name,
                    )
                    prediction_frames.append(pred)
                    metric_rows.append(
                        _metric_row(
                            pred,
                            threshold=threshold,
                            extra={
                                "evaluation_protocol": "clean_internal_protocol",
                                "analysis_family": "strong_feature_level_fusion",
                                "model_name": model_name,
                                "modality": "multimodal",
                                "modality_combination": combo_name,
                                "metric_split": split_name,
                                "threshold_source": "validation_balanced_accuracy",
                                "skipped": False,
                                "n_features": float(len(x_cols)),
                            },
                        )
                    )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return metrics, predictions


def _selected_predictions(predictions: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or selection.empty:
        return predictions.iloc[0:0].copy()
    selected_keys = set(zip(selection["modality"].astype(str), selection["selected_model_name"].astype(str)))
    mask = [
        (str(row.modality), str(row.model_name)) in selected_keys
        for row in predictions[["modality", "model_name"]].itertuples(index=False)
    ]
    out = predictions.loc[mask].copy()
    out["selected_for_fusion"] = True
    return out


def _participant_modality_matrix(predictions: pd.DataFrame, split: str) -> pd.DataFrame:
    selected = predictions[predictions["split"].eq(split)].copy()
    if selected.empty:
        return pd.DataFrame()
    participant = (
        selected.groupby(["participant_id", "label_binary", "modality"], dropna=False)
        .agg(probability=("probability", "mean"))
        .reset_index()
    )
    matrix = participant.pivot_table(
        index=["participant_id", "label_binary"],
        columns="modality",
        values="probability",
        aggfunc="mean",
    ).reset_index()
    matrix.columns.name = None
    matrix["split"] = split
    return matrix


def run_strong_fusion(
    selected_predictions: pd.DataFrame,
    selection: pd.DataFrame,
    modalities: Iterable[str] = DEFAULT_MODALITIES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if selected_predictions.empty:
        return pd.DataFrame(), pd.DataFrame()

    validation_matrix = _participant_modality_matrix(selected_predictions, "validation")
    test_matrix = _participant_modality_matrix(selected_predictions, "test")
    if validation_matrix.empty or test_matrix.empty:
        return pd.DataFrame(), pd.DataFrame()

    modality_list = [str(m) for m in modalities]
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    validation_weight_map = selection.set_index("modality")["validation_auprc"].to_dict() if not selection.empty else {}

    for size in range(2, len(modality_list) + 1):
        for combo in combinations(modality_list, size):
            combo_cols = list(combo)
            if not set(combo_cols).issubset(validation_matrix.columns) or not set(combo_cols).issubset(test_matrix.columns):
                continue
            val = validation_matrix.dropna(subset=combo_cols).copy()
            test = test_matrix.dropna(subset=combo_cols).copy()
            if val.empty or test.empty or val["label_binary"].nunique() < 2:
                continue

            combo_name = "+".join(combo_cols)
            fusion_specs: list[tuple[str, np.ndarray, np.ndarray]] = []
            fusion_specs.append(
                (
                    "uniform_mean",
                    val[combo_cols].mean(axis=1).to_numpy(dtype=float),
                    test[combo_cols].mean(axis=1).to_numpy(dtype=float),
                )
            )
            raw_weights = np.asarray(
                [max(float(validation_weight_map.get(col, 0.5)) - 0.5, 0.01) for col in combo_cols],
                dtype=float,
            )
            weights = raw_weights / raw_weights.sum()
            fusion_specs.append(
                (
                    "validation_weighted_auprc",
                    np.average(val[combo_cols].to_numpy(dtype=float), axis=1, weights=weights),
                    np.average(test[combo_cols].to_numpy(dtype=float), axis=1, weights=weights),
                )
            )

            if val.shape[0] >= 8 and val["label_binary"].nunique() == 2:
                stacker = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
                stacker.fit(val[combo_cols].to_numpy(dtype=float), labels_to_binary(val["label_binary"]))
                fusion_specs.append(
                    (
                        "stacked_logistic_validation",
                        stacker.predict_proba(val[combo_cols].to_numpy(dtype=float))[:, 1],
                        stacker.predict_proba(test[combo_cols].to_numpy(dtype=float))[:, 1],
                    )
                )

            for method, val_prob, test_prob in fusion_specs:
                threshold = best_threshold_by_balanced_accuracy(labels_to_binary(val["label_binary"]), val_prob)
                for split_name, frame, probs in (("validation", val, val_prob), ("test", test, test_prob)):
                    pred = pd.DataFrame(
                        {
                            "recording_id": frame["participant_id"].astype(str) + f"::{combo_name}",
                            "participant_id": frame["participant_id"].astype(str),
                            "dataset": "coswara",
                            "modality": "multimodal",
                            "submodality": combo_name,
                            "label_binary": frame["label_binary"].to_numpy(),
                            "split": split_name,
                            "model_name": "strong_baseline_selected_fusion",
                            "analysis_family": "strong_multimodal_fusion",
                            "evaluation_protocol": "clean_internal_protocol",
                            "modality_combination": combo_name,
                            "fusion_method": method,
                            "probability": probs,
                        }
                    )
                    prediction_rows.append(pred)
                    metric_rows.append(
                        _metric_row(
                            pred,
                            threshold=threshold,
                            extra={
                                "evaluation_protocol": "clean_internal_protocol",
                                "analysis_family": "strong_multimodal_fusion",
                                "model_name": "strong_baseline_selected_fusion",
                                "modality": "multimodal",
                                "modality_combination": combo_name,
                                "fusion_method": method,
                                "metric_split": split_name,
                                "threshold_source": "validation_balanced_accuracy",
                                "skipped": False,
                            },
                        )
                    )

    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    return metrics, predictions


STACK_SOURCE_COLS = [
    "analysis_family",
    "model_name",
    "modality",
    "modality_combination",
    "fusion_method",
]


def _add_stack_source_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in STACK_SOURCE_COLS:
        if col not in out.columns:
            out[col] = ""
    out["stack_source_key"] = out[STACK_SOURCE_COLS].fillna("").astype(str).agg("||".join, axis=1)
    return out


def run_global_prediction_stacker(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if top_k < 2 or metrics.empty or predictions.empty:
        return pd.DataFrame(), pd.DataFrame()

    skipped = metrics["skipped"].fillna(False).astype(bool) if "skipped" in metrics.columns else False
    candidates = metrics[
        metrics["metric_split"].eq("validation")
        & metrics["analysis_family"].isin(
            ["strong_audio_modality", "strong_multimodal_fusion", "strong_feature_level_fusion"]
        )
        & ~skipped
    ].copy()
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    candidates = _add_stack_source_key(candidates)
    candidates["auroc"] = pd.to_numeric(candidates["auroc"], errors="coerce")
    candidates["auprc"] = pd.to_numeric(candidates["auprc"], errors="coerce")
    ranked = candidates.sort_values(["auroc", "auprc", "stack_source_key"], ascending=[False, False, True])
    selected_keys = list(dict.fromkeys(ranked["stack_source_key"].dropna().astype(str).head(top_k)))
    if len(selected_keys) < 2:
        return pd.DataFrame(), pd.DataFrame()

    pred = _add_stack_source_key(predictions)
    pred = pred[pred["stack_source_key"].isin(selected_keys) & pred["split"].isin(["validation", "test"])].copy()
    if pred.empty:
        return pd.DataFrame(), pd.DataFrame()
    participant = (
        pred.groupby(["participant_id", "label_binary", "split", "stack_source_key"], dropna=False)
        .agg(probability=("probability", "mean"))
        .reset_index()
    )
    matrix = participant.pivot_table(
        index=["participant_id", "label_binary", "split"],
        columns="stack_source_key",
        values="probability",
        aggfunc="mean",
    ).reset_index()
    matrix.columns.name = None
    val = matrix[matrix["split"].eq("validation")].dropna(subset=selected_keys).copy()
    test = matrix[matrix["split"].eq("test")].dropna(subset=selected_keys).copy()
    if val.empty or test.empty or val["label_binary"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()

    val_x = val[selected_keys].to_numpy(dtype=float)
    test_x = test[selected_keys].to_numpy(dtype=float)
    val_y = labels_to_binary(val["label_binary"])
    ranked_weights = ranked.drop_duplicates("stack_source_key").set_index("stack_source_key")["auroc"].to_dict()
    raw_weights = np.asarray([max(float(ranked_weights.get(key, 0.5)) - 0.5, 0.01) for key in selected_keys])
    weights = raw_weights / raw_weights.sum()

    fusion_specs: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("top_global_uniform_mean", val_x.mean(axis=1), test_x.mean(axis=1)),
        ("top_global_validation_weighted_auroc", np.average(val_x, axis=1, weights=weights), np.average(test_x, axis=1, weights=weights)),
    ]
    if len(val) >= 8:
        stacker = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
        stacker.fit(val_x, val_y)
        fusion_specs.append(
            (
                "top_global_stacked_logistic_validation",
                stacker.predict_proba(val_x)[:, 1],
                stacker.predict_proba(test_x)[:, 1],
            )
        )

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    selected_label = ";".join(selected_keys)
    for method, val_prob, test_prob in fusion_specs:
        threshold = best_threshold_by_balanced_accuracy(val_y, val_prob)
        for split_name, frame, probs in (("validation", val, val_prob), ("test", test, test_prob)):
            pred_frame = _participant_prediction_frame(
                frame,
                probs,
                model_name=f"top_{len(selected_keys)}_global_stack",
                split=split_name,
                analysis_family="strong_global_stacking",
                modality_combination="global_prediction_stack",
                fusion_method=method,
            )
            pred_frame["ensemble_members"] = selected_label
            prediction_rows.append(pred_frame)
            metric_rows.append(
                _metric_row(
                    pred_frame,
                    threshold=threshold,
                    extra={
                        "evaluation_protocol": "clean_internal_protocol",
                        "analysis_family": "strong_global_stacking",
                        "model_name": f"top_{len(selected_keys)}_global_stack",
                        "modality": "multimodal",
                        "modality_combination": "global_prediction_stack",
                        "fusion_method": method,
                        "metric_split": split_name,
                        "threshold_source": "validation_balanced_accuracy",
                        "skipped": False,
                        "n_ensemble_models": float(len(selected_keys)),
                        "ensemble_members": selected_label,
                    },
                )
            )

    metrics_out = pd.DataFrame(metric_rows)
    predictions_out = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    return metrics_out, predictions_out


def run_strong_baseline(
    features: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    modalities: Iterable[str] = DEFAULT_MODALITIES,
    model_names: Iterable[str] = DEFAULT_MODEL_NAMES,
    require_quality_ok: bool = False,
    random_state: int = 42,
    optuna_trials: int = 25,
    ensemble_top_k: int = 3,
    enable_feature_level_fusion: bool = False,
    global_stack_top_k: int = 0,
) -> StrongBaselineResult:
    prepared, protocol = _prepare_features_for_protocol(
        features,
        metadata=metadata,
        modalities=modalities,
        require_quality_ok=require_quality_ok,
    )
    if prepared.empty:
        return StrongBaselineResult(
            metrics=pd.DataFrame(),
            predictions=pd.DataFrame(),
            selection=pd.DataFrame(),
            protocol_audit=protocol.audit,
            participant_audit=protocol.participant_audit,
        )

    modality_metrics, modality_predictions, selection, fitted = train_strong_modality_models(
        prepared,
        modalities=modalities,
        model_names=model_names,
        random_state=random_state,
        optuna_trials=optuna_trials,
        ensemble_top_k=ensemble_top_k,
    )
    selected = _selected_predictions(modality_predictions, selection)
    fusion_metrics, fusion_predictions = run_strong_fusion(selected, selection, modalities=modalities)
    metric_frames = [modality_metrics, fusion_metrics]
    prediction_frames = [modality_predictions, fusion_predictions]

    if enable_feature_level_fusion:
        early_metrics, early_predictions = train_feature_level_fusion_models(
            prepared,
            modalities=modalities,
            model_names=model_names,
            random_state=random_state,
            optuna_trials=optuna_trials,
        )
        metric_frames.append(early_metrics)
        prediction_frames.append(early_predictions)

    metrics = pd.concat([frame for frame in metric_frames if not frame.empty], ignore_index=True, sort=False)
    predictions = pd.concat([frame for frame in prediction_frames if not frame.empty], ignore_index=True, sort=False)

    if global_stack_top_k >= 2:
        stack_metrics, stack_predictions = run_global_prediction_stacker(
            metrics,
            predictions,
            top_k=global_stack_top_k,
        )
        if not stack_metrics.empty:
            metrics = pd.concat([metrics, stack_metrics], ignore_index=True, sort=False)
        if not stack_predictions.empty:
            predictions = pd.concat([predictions, stack_predictions], ignore_index=True, sort=False)

    return StrongBaselineResult(
        metrics=metrics,
        predictions=predictions,
        selection=selection,
        protocol_audit=protocol.audit,
        participant_audit=protocol.participant_audit,
    )


def save_strong_baseline_result(
    result: StrongBaselineResult,
    metrics_output: Path,
    predictions_output: Path,
    selection_output: Path,
    protocol_audit_output: Path,
    participant_audit_output: Path,
) -> None:
    outputs = [
        metrics_output,
        predictions_output,
        selection_output,
        protocol_audit_output,
        participant_audit_output,
    ]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(metrics_output, index=False)
    result.predictions.to_csv(predictions_output, index=False)
    result.selection.to_csv(selection_output, index=False)
    result.protocol_audit.to_csv(protocol_audit_output, index=False)
    result.participant_audit.to_csv(participant_audit_output, index=False)
