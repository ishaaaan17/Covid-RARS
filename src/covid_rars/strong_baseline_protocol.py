from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


CLEAN_POSITIVE_HINTS = {
    "positive",
    "positive mild",
    "positive moderate",
    "positive asymp",
    "positive asymptomatic",
    "covid positive",
    "covid 19 positive",
    "sars cov 2 positive",
}

CLEAN_NEGATIVE_HINTS = {
    "healthy",
    "negative",
    "covid negative",
    "covid 19 negative",
    "sars cov 2 negative",
    "normal",
}

AMBIGUOUS_HINTS = {
    "recovered",
    "resp illness",
    "respiratory illness",
    "no resp illness exposed",
    "not identified",
    "unknown",
}

DEFAULT_MODALITIES = ("cough", "breath", "speech")
DEFAULT_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class ProtocolResult:
    metadata: pd.DataFrame
    audit: pd.DataFrame
    participant_audit: pd.DataFrame


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def _contains_any(text: str, hints: Iterable[str]) -> bool:
    return any(hint in text for hint in hints)


def infer_clean_label_from_raw(row: pd.Series) -> tuple[str, str]:
    """Infer a paper-comparable active-positive/healthy-negative label.

    If raw label text is informative, ambiguous recovered/exposed/respiratory
    illness groups are excluded. If raw text is unavailable, the existing
    normalized binary label is used as a conservative fallback so older feature
    tables remain runnable.
    """
    label_binary = _clean_text(row.get("label_binary", ""))
    raw = _clean_text(row.get("label_raw", ""))
    label_group = _clean_text(row.get("label_group", ""))
    combined = " ".join(part for part in (raw, label_group) if part)

    if combined:
        if _contains_any(combined, AMBIGUOUS_HINTS):
            return "exclude", "ambiguous_raw_label"
        if _contains_any(combined, CLEAN_POSITIVE_HINTS):
            return "positive", "raw_clean_positive"
        if _contains_any(combined, CLEAN_NEGATIVE_HINTS):
            return "negative", "raw_clean_negative"
        return "exclude", "raw_label_not_clean_protocol"

    if label_binary in {"positive", "negative"}:
        return label_binary, "fallback_binary_label"
    return "exclude", "unknown_binary_label"


def build_clean_internal_protocol(
    metadata: pd.DataFrame,
    modalities: Iterable[str] = DEFAULT_MODALITIES,
    splits: Iterable[str] = DEFAULT_SPLITS,
    require_quality_ok: bool = False,
) -> ProtocolResult:
    """Filter metadata to a paper-comparable clean internal protocol.

    The filter is intentionally explicit: active COVID-positive versus healthy
    or clearly negative controls, known supervised labels, supported modalities,
    participant-level label consistency, and train/validation/test rows.
    """
    if metadata.empty:
        empty = metadata.copy()
        return ProtocolResult(empty, _audit_table(empty), _participant_audit(empty))

    required = {"participant_id", "recording_id", "label_binary", "modality"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required protocol columns: {sorted(missing)}")

    allowed_modalities = {str(m) for m in modalities}
    allowed_splits = {str(s) for s in splits}
    df = metadata.copy()
    if "split" not in df.columns:
        df["split"] = "unused"

    inferred = df.apply(infer_clean_label_from_raw, axis=1, result_type="expand")
    inferred.columns = ["clean_label_binary", "clean_label_source"]
    df = pd.concat([df, inferred], axis=1)
    df["protocol_exclusion_reason"] = ""

    modality_mask = df["modality"].astype(str).isin(allowed_modalities)
    split_mask = df["split"].astype(str).isin(allowed_splits)
    label_mask = df["clean_label_binary"].isin(["positive", "negative"])
    quality_mask = pd.Series(True, index=df.index)
    if require_quality_ok and "quality_flag" in df.columns:
        quality_mask = df["quality_flag"].astype(str).str.lower().eq("ok")

    df.loc[~modality_mask, "protocol_exclusion_reason"] = "unsupported_modality"
    df.loc[modality_mask & ~split_mask, "protocol_exclusion_reason"] = "unused_split"
    df.loc[modality_mask & split_mask & ~label_mask, "protocol_exclusion_reason"] = df.loc[
        modality_mask & split_mask & ~label_mask,
        "clean_label_source",
    ]
    df.loc[modality_mask & split_mask & label_mask & ~quality_mask, "protocol_exclusion_reason"] = "quality_not_ok"

    included = df[
        modality_mask
        & split_mask
        & label_mask
        & quality_mask
    ].copy()
    df.loc[modality_mask & split_mask & label_mask & quality_mask, "protocol_exclusion_reason"] = "included"

    label_counts = included.groupby("participant_id")["clean_label_binary"].nunique()
    inconsistent = set(label_counts[label_counts > 1].index.astype(str))
    if inconsistent:
        included = included[~included["participant_id"].astype(str).isin(inconsistent)].copy()
        df.loc[df["participant_id"].astype(str).isin(inconsistent), "protocol_exclusion_reason"] = (
            "participant_label_conflict"
        )

    included["label_binary"] = included["clean_label_binary"]
    included["evaluation_protocol"] = "clean_internal_protocol"
    included["label_protocol"] = "active_positive_vs_clean_negative"
    included["protocol_exclusion_reason"] = "included"
    return ProtocolResult(
        metadata=included.reset_index(drop=True),
        audit=_audit_table(df),
        participant_audit=_participant_audit(included),
    )


def _audit_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "evaluation_protocol",
                "protocol_exclusion_reason",
                "split",
                "modality",
                "label_binary",
                "n_rows",
                "n_participants",
            ]
        )
    audit = (
        df.assign(
            protocol_exclusion_reason=df.get("protocol_exclusion_reason", "included"),
            split=df.get("split", "unused"),
            modality=df.get("modality", "unknown"),
            label_binary=df.get("clean_label_binary", df.get("label_binary", "unknown")),
        )
        .groupby(["protocol_exclusion_reason", "split", "modality", "label_binary"], dropna=False)
        .agg(n_rows=("recording_id", "nunique"), n_participants=("participant_id", "nunique"))
        .reset_index()
        .sort_values(["protocol_exclusion_reason", "split", "modality", "label_binary"])
    )
    audit.insert(0, "evaluation_protocol", "clean_internal_protocol")
    return audit


def _participant_audit(metadata: pd.DataFrame) -> pd.DataFrame:
    if metadata.empty:
        return pd.DataFrame(
            columns=[
                "evaluation_protocol",
                "split",
                "label_binary",
                "n_participants",
                "modalities_available",
            ]
        )

    def join_modalities(values: pd.Series) -> str:
        return ",".join(sorted(set(values.dropna().astype(str))))

    table = (
        metadata.groupby(["participant_id", "split", "label_binary"], dropna=False)
        .agg(modalities_available=("modality", join_modalities))
        .reset_index()
    )
    out = (
        table.groupby(["split", "label_binary", "modalities_available"], dropna=False)
        .agg(n_participants=("participant_id", "nunique"))
        .reset_index()
        .sort_values(["split", "label_binary", "modalities_available"])
    )
    out.insert(0, "evaluation_protocol", "clean_internal_protocol")
    return out
