from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureSelectionResult:
    selected_columns: list[str]
    transformed_train: np.ndarray
    transformed_other: np.ndarray | None
    pipeline: Pipeline
    method: str


def remove_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
) -> list[str]:
    if df.empty:
        return []
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = {column for column in upper.columns if any(upper[column] > threshold)}
    return [column for column in df.columns if column not in drop]


def build_feature_selection_pipeline(
    method: str = "select_k_best",
    k: int = 80,
    pca_variance: float = 0.95,
) -> Pipeline:
    if method == "select_k_best":
        return Pipeline(
            [
                ("variance", VarianceThreshold()),
                ("scaler", StandardScaler()),
                ("select", SelectPercentile(score_func=mutual_info_classif, percentile=min(100, max(5, int(k))))),
            ]
        )
    if method == "pca":
        return Pipeline(
            [
                ("variance", VarianceThreshold()),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_variance, svd_solver="full")),
            ]
        )
    if method == "scale_only":
        return Pipeline([("variance", VarianceThreshold()), ("scaler", StandardScaler())])
    raise ValueError(f"Unknown feature-selection method: {method}")


def fit_transform_features(
    features: pd.DataFrame,
    labels: pd.Series,
    other: pd.DataFrame | None = None,
    method: str = "select_k_best",
    k: int = 80,
) -> FeatureSelectionResult:
    non_constant_columns = [
        col
        for col in features.columns
        if pd.to_numeric(features[col], errors="coerce").nunique(dropna=False) > 1
    ]
    if not non_constant_columns:
        raise ValueError("No non-constant feature columns available")
    features = features[non_constant_columns]
    other = other[non_constant_columns] if other is not None else None
    corr_columns = remove_highly_correlated_features(features)
    features_corr = features[corr_columns]
    other_corr = other[corr_columns] if other is not None else None
    effective_k = min(k, max(1, features_corr.shape[1]))
    pipeline = build_feature_selection_pipeline(method=method, k=effective_k)
    transformed_train = pipeline.fit_transform(features_corr, labels)
    transformed_other = pipeline.transform(other_corr) if other_corr is not None else None
    return FeatureSelectionResult(
        selected_columns=corr_columns,
        transformed_train=transformed_train,
        transformed_other=transformed_other,
        pipeline=pipeline,
        method=method,
    )
