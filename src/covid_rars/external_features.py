from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_EXTERNAL_QUALITY_OK = {"ok"}


def _quality_series(df: pd.DataFrame) -> pd.Series:
    if "quality_flag" in df.columns:
        return df["quality_flag"]
    if "manual_quality_label" in df.columns:
        return df["manual_quality_label"]
    return pd.Series(["unknown"] * len(df), index=df.index)


def prepare_external_feature_metadata(
    index: pd.DataFrame,
    split_name: str = "external",
    labeled_only: bool = True,
    quality_ok_only: bool = False,
    accepted_quality_flags: set[str] | None = None,
) -> pd.DataFrame:
    """Prepare an external dataset index for feature extraction.

    External rows are forced into a single `external` split so they cannot leak into
    source train/validation/test flows. By default only positive/negative labels are
    kept because downstream cross-dataset metrics require binary labels.
    """
    df = index.copy()
    if df.empty:
        out = df.copy()
        out["split"] = []
        if "quality_flag" not in out.columns:
            out["quality_flag"] = []
        return out

    df["split"] = split_name
    df["quality_flag"] = _quality_series(df).fillna("unknown").astype(str).str.lower()
    if labeled_only and "label_binary" in df.columns:
        df = df[df["label_binary"].isin(["positive", "negative"])].copy()
    if quality_ok_only:
        accepted = accepted_quality_flags or DEFAULT_EXTERNAL_QUALITY_OK
        accepted_normalized = {str(flag).lower() for flag in accepted}
        df = df[df["quality_flag"].isin(accepted_normalized)].copy()
    return df.reset_index(drop=True)


def extract_external_feature_table(
    index: pd.DataFrame,
    split_name: str = "external",
    labeled_only: bool = True,
    quality_ok_only: bool = False,
) -> pd.DataFrame:
    """Extract MFCC/acoustic features from an external dataset index."""
    from covid_rars.features import extract_feature_table

    prepared = prepare_external_feature_metadata(
        index,
        split_name=split_name,
        labeled_only=labeled_only,
        quality_ok_only=quality_ok_only,
    )
    return extract_feature_table(prepared, quality_mode="all_samples")


def write_external_feature_table(
    index_path: str | Path,
    output_path: str | Path,
    split_name: str = "external",
    labeled_only: bool = True,
    quality_ok_only: bool = False,
    max_rows: int | None = None,
) -> pd.DataFrame:
    index = pd.read_csv(index_path)
    prepared = prepare_external_feature_metadata(
        index,
        split_name=split_name,
        labeled_only=labeled_only,
        quality_ok_only=quality_ok_only,
    )
    if max_rows is not None:
        prepared = prepared.head(max_rows).copy()
    from covid_rars.features import extract_feature_table

    features = extract_feature_table(prepared, quality_mode="all_samples")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    return features
