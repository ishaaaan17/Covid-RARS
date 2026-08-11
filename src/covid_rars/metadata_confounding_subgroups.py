from __future__ import annotations

import numpy as np
import pandas as pd

from covid_rars.metrics import binary_metric_bundle, labels_to_binary


DEVICE_CANDIDATES = [
    "device",
    "device_type",
    "recording_device",
    "platform",
    "phone_model",
    "browser",
    "os",
]


def _derive_temporal_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    frame = metadata.copy()
    if ("recording_year" not in frame.columns or "recording_month" not in frame.columns) and "recording_date" in frame.columns:
        date = pd.to_datetime(frame["recording_date"], errors="coerce")
        if "recording_year" not in frame.columns:
            frame["recording_year"] = date.dt.year
        if "recording_month" not in frame.columns:
            frame["recording_month"] = date.dt.month
    return frame


def _resolve_subgroup(metadata: pd.DataFrame, subgroup: str) -> tuple[str | None, pd.Series | None]:
    frame = _derive_temporal_columns(metadata)
    if subgroup == "device":
        for candidate in DEVICE_CANDIDATES:
            if candidate in frame.columns:
                return candidate, frame[candidate]
        return None, None
    if subgroup in frame.columns:
        return subgroup, frame[subgroup]
    return None, None


def _availability_row(metadata: pd.DataFrame, subgroup: str) -> dict[str, object]:
    source_column, values = _resolve_subgroup(metadata, subgroup)
    if source_column is None or values is None:
        return {
            "subgroup": subgroup,
            "available": False,
            "source_column": "",
            "n_non_missing": 0,
            "n_levels": 0,
            "reason": "no matching metadata column",
        }
    clean = values.dropna().astype(str).str.strip()
    clean = clean[clean.ne("")]
    return {
        "subgroup": subgroup,
        "available": bool(not clean.empty),
        "source_column": source_column,
        "n_non_missing": int(len(clean)),
        "n_levels": int(clean.nunique()),
        "reason": "" if not clean.empty else "column exists but has no usable values",
    }


def _subgroup_frame(metadata: pd.DataFrame, subgroup_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _derive_temporal_columns(metadata)
    availability = pd.DataFrame([_availability_row(frame, subgroup) for subgroup in subgroup_columns])
    rows: list[dict[str, object]] = []
    for _, available in availability[availability["available"].astype(bool)].iterrows():
        subgroup = str(available["subgroup"])
        source_column = str(available["source_column"])
        values = frame[source_column]
        usable = frame[frame["label_binary"].isin(["positive", "negative"])].copy()
        usable["_subgroup_level"] = values.loc[usable.index].fillna("missing").astype(str).str.strip().replace("", "missing")
        for level, group in usable.groupby("_subgroup_level", dropna=False):
            y = labels_to_binary(group["label_binary"])
            rows.append(
                {
                    "subgroup": subgroup,
                    "source_column": source_column,
                    "level": str(level),
                    "n_recordings": int(len(group)),
                    "n_participants": int(group["participant_id"].nunique()) if "participant_id" in group.columns else int(len(group)),
                    "n_positive": int(y.sum()),
                    "n_negative": int(len(y) - y.sum()),
                    "positive_prevalence": float(y.mean()) if len(y) else float("nan"),
                    "splits": ",".join(sorted(group["split"].dropna().astype(str).unique())) if "split" in group.columns else "",
                }
            )
    return availability, pd.DataFrame(rows)


def _merge_predictions_metadata(predictions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    meta = _derive_temporal_columns(metadata).copy()
    preferred_cols = [
        "recording_id",
        "participant_id",
        "country",
        "recording_year",
        "recording_month",
        *DEVICE_CANDIDATES,
    ]
    keep = [col for col in preferred_cols if col in meta.columns]
    if "recording_id" in predictions.columns and "recording_id" in meta.columns:
        return predictions.merge(meta[keep].drop_duplicates("recording_id"), on="recording_id", how="left", suffixes=("", "_metadata"))
    if "participant_id" in predictions.columns and "participant_id" in meta.columns:
        keep = [col for col in keep if col != "recording_id"]
        return predictions.merge(meta[keep].drop_duplicates("participant_id"), on="participant_id", how="left", suffixes=("", "_metadata"))
    return predictions.copy()


def _subgroup_metrics(
    metadata: pd.DataFrame,
    predictions: pd.DataFrame,
    subgroup_columns: list[str],
    min_samples: int,
) -> pd.DataFrame:
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    merged = _merge_predictions_metadata(predictions, metadata)
    rows: list[dict[str, object]] = []
    for subgroup in subgroup_columns:
        source_column, values = _resolve_subgroup(merged, subgroup)
        if source_column is None or values is None:
            continue
        work = merged[merged["label_binary"].isin(["positive", "negative"])].copy()
        work["probability"] = pd.to_numeric(work["probability"], errors="coerce")
        work = work[np.isfinite(work["probability"])]
        work["_subgroup_level"] = values.loc[work.index].fillna("missing").astype(str).str.strip().replace("", "missing")
        group_cols = ["_subgroup_level"]
        for col in ["audit_model", "model_name", "feature_strategy", "split"]:
            if col in work.columns:
                group_cols.append(col)
        for group_key, group in work.groupby(group_cols, dropna=False):
            if len(group) < int(min_samples) or group["label_binary"].nunique() < 2:
                continue
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            meta_values = dict(zip(group_cols, group_key))
            threshold = float(pd.to_numeric(group.get("threshold", pd.Series([0.5])), errors="coerce").dropna().median())
            if not np.isfinite(threshold):
                threshold = 0.5
            row = binary_metric_bundle(
                labels_to_binary(group["label_binary"]),
                group["probability"].astype(float).to_numpy(),
                threshold=threshold,
            )
            row.update(
                {
                    "subgroup": subgroup,
                    "source_column": source_column,
                    "level": str(meta_values.pop("_subgroup_level")),
                    "n_participants": int(group["participant_id"].nunique()) if "participant_id" in group.columns else int(len(group)),
                    **meta_values,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_metadata_confounding_subgroup_tables(
    metadata: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    subgroup_columns: list[str] | None = None,
    min_samples: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subgroup_columns = subgroup_columns or ["country", "device", "recording_year", "recording_month"]
    availability, breakdown = _subgroup_frame(metadata, subgroup_columns)
    metrics = _subgroup_metrics(metadata, predictions if predictions is not None else pd.DataFrame(), subgroup_columns, min_samples)
    return availability, breakdown, metrics
