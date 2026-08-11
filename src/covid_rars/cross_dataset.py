from __future__ import annotations

import pandas as pd


DEFAULT_ID_COLUMNS = {
    "recording_id",
    "participant_id",
    "dataset",
    "modality",
    "submodality",
    "label_binary",
    "split",
    "segmentation_method",
}


def numeric_feature_columns(df: pd.DataFrame, id_columns: list[str] | None = None) -> list[str]:
    excluded = set(id_columns or []) | DEFAULT_ID_COLUMNS
    return [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]


def harmonize_feature_columns(
    train_features: pd.DataFrame,
    external_features: pd.DataFrame,
    id_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Align numeric feature matrices for direct cross-dataset evaluation."""
    train_cols = set(numeric_feature_columns(train_features, id_columns=id_columns))
    external_cols = set(numeric_feature_columns(external_features, id_columns=id_columns))
    columns = sorted(train_cols | external_cols)
    train_x = train_features.reindex(columns=columns, fill_value=0.0).fillna(0.0)
    external_x = external_features.reindex(columns=columns, fill_value=0.0).fillna(0.0)
    return train_x, external_x, columns


def add_external_split(features: pd.DataFrame, split_name: str = "external") -> pd.DataFrame:
    out = features.copy()
    out["split"] = split_name
    return out
