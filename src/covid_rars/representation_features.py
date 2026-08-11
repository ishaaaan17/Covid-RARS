from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPRESENTATION_ID_COLUMNS = {
    "recording_id",
    "participant_id",
    "dataset",
    "modality",
    "submodality",
    "label_binary",
    "split",
    "segmentation_method",
    "manual_quality_label",
    "quality_flag",
    "representation",
}

REQUIRED_FEATURE_TABLE_COLUMNS = {
    "recording_id",
    "participant_id",
    "dataset",
    "modality",
    "label_binary",
}


def read_feature_table(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(resolved)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(resolved)
    raise ValueError(f"Unsupported feature table format: {resolved.suffix}")


def write_feature_table(features: pd.DataFrame, path: str | Path) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        features.to_csv(resolved, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        features.to_parquet(resolved, index=False)
        return
    raise ValueError(f"Unsupported feature table format: {resolved.suffix}")


def representation_feature_columns(
    features: pd.DataFrame,
    id_columns: set[str] | list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    excluded = REPRESENTATION_ID_COLUMNS | set(id_columns or [])
    return [
        col
        for col in features.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(features[col])
    ]


def validate_feature_table(
    features: pd.DataFrame,
    *,
    require_split: bool = True,
    id_columns: set[str] | list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    required = set(REQUIRED_FEATURE_TABLE_COLUMNS)
    if require_split:
        required.add("split")
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Feature table missing required columns: {missing}")

    cols = representation_feature_columns(features, id_columns=id_columns)
    if not cols:
        raise ValueError("Feature table has no numeric representation feature columns")

    values = features[cols].to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        bad_cols = [
            col
            for col in cols
            if not np.isfinite(features[col].to_numpy(dtype=float, copy=False)).all()
        ]
        raise ValueError(f"Feature table contains non-finite feature values in: {bad_cols[:10]}")
    return cols
