from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from covid_rars.feature_selection import build_feature_selection_pipeline

from covid_rars.features import feature_columns
from covid_rars.metrics import binary_metric_bundle, labels_to_binary


try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency may be missing
    XGBClassifier = None


@dataclass
class TrainResult:
    model_name: str
    metrics: dict[str, float | str]
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    model: object
    validation_metrics: dict[str, float | str] | None = None


def make_model(model_name: str, random_state: int = 42) -> object:
    if model_name == "dummy_most_frequent":
        return DummyClassifier(strategy="most_frequent")
    if model_name == "dummy_stratified":
        return DummyClassifier(strategy="stratified", random_state=random_state)
    if model_name == "logistic_regression":
        return Pipeline(
            [
                ("features", build_feature_selection_pipeline(method="select_k_best", k=80)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_name == "random_forest":
        return Pipeline([
            ("features", build_feature_selection_pipeline(method="select_k_best", k=80)),
            ("model", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )),
        ])
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed")
        return Pipeline([
            ("features", build_feature_selection_pipeline(method="select_k_best", k=80)),
            ("model", XGBClassifier(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
        )),
        ])
    raise ValueError(f"Unknown model name: {model_name}")


def _predict_prob(model: object, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))
    raise ValueError("Model does not expose predict_proba or decision_function")


def _prediction_frame(source: pd.DataFrame, probabilities: np.ndarray, model_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recording_id": source["recording_id"].to_numpy(),
            "participant_id": source["participant_id"].to_numpy(),
            "dataset": source["dataset"].to_numpy(),
            "modality": source["modality"].to_numpy(),
            "label_binary": source["label_binary"].to_numpy(),
            "split": source["split"].to_numpy(),
            "model_name": model_name,
            "probability": probabilities,
        }
    )


def train_single_model(
    features: pd.DataFrame,
    model_name: str,
    modality: str,
    random_state: int = 42,
) -> TrainResult:
    df = features[
        (features["label_binary"].isin(["positive", "negative"]))
        & (features["modality"] == modality)
        & (features["split"].isin(["train", "validation", "test"]))
    ].copy()
    if df.empty:
        raise ValueError(f"No rows available for modality={modality}")

    cols = feature_columns(df)
    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]
    if train.empty or validation.empty or test.empty:
        raise ValueError(f"Need train/validation/test rows for modality={modality}")

    x_train = train[cols].fillna(0.0)
    y_train = labels_to_binary(train["label_binary"])
    model = make_model(model_name, random_state=random_state)
    model.fit(x_train, y_train)

    val_prob = _predict_prob(model, validation[cols].fillna(0.0))
    test_prob = _predict_prob(model, test[cols].fillna(0.0))
    val_pred = _prediction_frame(validation, val_prob, model_name)
    test_pred = _prediction_frame(test, test_prob, model_name)

    y_validation = labels_to_binary(validation["label_binary"])
    validation_metrics = binary_metric_bundle(y_validation, val_prob)
    validation_metrics.update({"model_name": model_name, "modality": modality, "metric_split": "validation"})

    y_test = labels_to_binary(test["label_binary"])
    metrics = binary_metric_bundle(y_test, test_prob)
    metrics.update({"model_name": model_name, "modality": modality})
    return TrainResult(
        model_name=model_name,
        metrics=metrics,
        validation_predictions=val_pred,
        test_predictions=test_pred,
        model=model,
        validation_metrics=validation_metrics,
    )


def save_model(model: object, path: str) -> None:
    joblib.dump(model, path)
