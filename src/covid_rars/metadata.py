from __future__ import annotations

import json

import pandas as pd

from covid_rars.data_index import build_modality_availability
from covid_rars.labels import normalize_label
from covid_rars.schemas import METADATA_COLUMNS


def clean_metadata(index_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = index_df.copy()
    if df.empty:
        return pd.DataFrame(columns=METADATA_COLUMNS), pd.DataFrame()

    for column in ["participant_id", "recording_id", "dataset", "modality", "submodality", "audio_path"]:
        if column not in df.columns:
            raise ValueError(f"Missing required index column: {column}")

    df["participant_id"] = df["participant_id"].astype(str).str.strip()
    df["recording_id"] = df["recording_id"].astype(str).str.strip()
    df["dataset"] = df.get("dataset", "coswara").fillna("coswara").astype(str).str.strip()
    df["modality"] = df["modality"].fillna("unknown").astype(str).str.strip()
    df["submodality"] = df["submodality"].fillna("unknown").astype(str).str.strip()
    df["label_raw"] = df.get("label_raw", "unknown").fillna("unknown").astype(str)
    df["label_binary"] = df["label_raw"].map(normalize_label)
    df["label_group"] = df["label_binary"]
    df["recording_date"] = df.get("recording_date", "").fillna("")
    df["age"] = df.get("age", "").fillna("")
    df["gender"] = df.get("gender", "").fillna("")
    df["country"] = df.get("country", "").fillna("")
    df["symptoms_json"] = df.get("symptoms_json", json.dumps({}))
    df["comorbidities_json"] = df.get("comorbidities_json", json.dumps({}))
    df["duration_sec"] = df.get("duration_sec", "")
    df["sample_rate_original"] = df.get("sample_rate_original", "")
    df["quality_flag"] = df.get("quality_flag", "not_audited")
    df["manual_quality_score"] = df.get("manual_quality_score", "")
    df["manual_quality_label"] = df.get("manual_quality_label", "unknown")
    df["split"] = df.get("split", "unused")

    df = df.drop_duplicates(subset=["recording_id"], keep="first")
    availability = build_modality_availability(df)
    return df[METADATA_COLUMNS], availability


def metadata_audit(metadata: pd.DataFrame) -> pd.DataFrame:
    if metadata.empty:
        return pd.DataFrame(
            columns=["dataset", "modality", "label_binary", "n_recordings", "n_participants"]
        )
    audit = (
        metadata.groupby(["dataset", "modality", "label_binary"], dropna=False)
        .agg(n_recordings=("recording_id", "nunique"), n_participants=("participant_id", "nunique"))
        .reset_index()
        .sort_values(["dataset", "modality", "label_binary"])
    )
    return audit
