from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def read_existing_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        p = Path(path)
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _comparison_type(target: pd.Series) -> str:
    split_style = str(target.get("split_style", "")).lower()
    paper_id = str(target.get("paper_id", "")).lower()
    if "cross-dataset" in split_style or "cross_" in paper_id:
        return "cross_dataset_transfer"
    if "chronological" in split_style or "postdevelopment" in paper_id:
        return "temporal_transfer"
    if "matched" in split_style:
        return "confounder_matched"
    return "paper_style_internal"


def _best_protocol_row(protocol_metrics: pd.DataFrame, modality: str) -> pd.Series | None:
    if protocol_metrics.empty:
        return None
    frame = protocol_metrics.copy()
    if "metric_split" in frame.columns:
        frame = frame[frame["metric_split"].astype(str).eq("test_aggregate")].copy()
    if "skipped" in frame.columns:
        skipped = frame["skipped"].astype(str).str.lower().isin(["true", "1", "yes"])
        frame = frame[~skipped].copy()
    if "modality" in frame.columns:
        frame = frame[frame["modality"].astype(str).eq(str(modality))].copy()
    if frame.empty or "auroc" not in frame.columns:
        return None
    frame["auroc"] = pd.to_numeric(frame["auroc"], errors="coerce")
    frame = frame.dropna(subset=["auroc"]).sort_values(["auroc"], ascending=False)
    return None if frame.empty else frame.iloc[0]


def _best_external_row(external_transfer_summary: pd.DataFrame) -> pd.Series | None:
    if external_transfer_summary.empty or "external_auroc" not in external_transfer_summary.columns:
        return None
    frame = external_transfer_summary.copy()
    if "skipped" in frame.columns:
        skipped = frame["skipped"].astype(str).str.lower().isin(["true", "1", "yes"])
        frame = frame[~skipped].copy()
    frame["external_auroc"] = pd.to_numeric(frame["external_auroc"], errors="coerce")
    frame = frame.dropna(subset=["external_auroc"]).sort_values(["external_auroc"], ascending=False)
    return None if frame.empty else frame.iloc[0]


def _final_ladder_values(final_validation_summary: pd.DataFrame | None) -> dict[str, float]:
    if final_validation_summary is None or final_validation_summary.empty:
        return {}
    values: dict[str, float] = {}
    protocol_map = {
        "strict_internal_auroc": "existing_participant_split",
        "time_stratified_auroc": "time_stratified",
        "temporal_early_to_late_auroc": "temporal_early_to_late",
    }
    for output_col, needle in protocol_map.items():
        if "evaluation_protocol" not in final_validation_summary.columns:
            continue
        rows = final_validation_summary[
            final_validation_summary["evaluation_protocol"].astype(str).str.contains(needle, case=False, na=False)
        ].copy()
        if rows.empty or "auroc" not in rows.columns:
            continue
        rows["auroc"] = pd.to_numeric(rows["auroc"], errors="coerce")
        rows = rows.dropna(subset=["auroc"]).sort_values("auroc", ascending=False)
        if not rows.empty:
            values[output_col] = float(rows.iloc[0]["auroc"])
    return values


def build_protocol_matched_gap_summary(
    targets: pd.DataFrame,
    protocol_metrics: pd.DataFrame,
    *,
    final_validation_summary: pd.DataFrame | None = None,
    external_transfer_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ladder_values = _final_ladder_values(final_validation_summary)
    best_external = _best_external_row(
        external_transfer_summary if external_transfer_summary is not None else pd.DataFrame()
    )
    rows: list[dict[str, object]] = []
    for _, target in targets.iterrows():
        target_type = _comparison_type(target)
        reported_value = float(target["reported_value"])
        modality = str(target.get("modality", "cough"))
        row: dict[str, object] = {
            "paper_id": target.get("paper_id"),
            "paper_name": target.get("paper_name"),
            "modality": modality,
            "dataset": target.get("dataset"),
            "comparison_type": target_type,
            "paper_split_style": target.get("split_style"),
            "reported_metric": target.get("reported_metric"),
            "reported_value": reported_value,
            "our_protocol_matched_auroc": np.nan,
            "paper_minus_our_protocol_matched_auroc": np.nan,
            "within_1pct": False,
            "our_external_auroc": np.nan,
            "paper_minus_our_external_auroc": np.nan,
            "matched_model_name": "",
            "matched_feature_strategy": "",
            "skipped": False,
            "skip_reason": "",
            **ladder_values,
        }

        if target_type == "paper_style_internal":
            best_protocol = _best_protocol_row(protocol_metrics, modality)
            if best_protocol is None:
                row["skipped"] = True
                row["skip_reason"] = "no protocol-matched aggregate row available"
            else:
                our_auroc = float(best_protocol["auroc"])
                gap = reported_value - our_auroc
                row["our_protocol_matched_auroc"] = our_auroc
                row["paper_minus_our_protocol_matched_auroc"] = gap
                row["within_1pct"] = abs(gap) <= 0.01
                row["matched_model_name"] = best_protocol.get("model_name", "")
                row["matched_feature_strategy"] = best_protocol.get("feature_strategy", "")
        elif target_type == "cross_dataset_transfer":
            dataset = str(target.get("dataset", "")).lower()
            if "coswara to coughvid" not in dataset:
                row["skipped"] = True
                row["skip_reason"] = "no matching reverse external transfer row available"
            elif best_external is None:
                row["skipped"] = True
                row["skip_reason"] = "no external transfer summary available"
            else:
                our_external = float(best_external["external_auroc"])
                gap = reported_value - our_external
                row["our_external_auroc"] = our_external
                row["paper_minus_our_external_auroc"] = gap
                row["within_1pct"] = abs(gap) <= 0.01
                row["matched_model_name"] = best_external.get("family_model", "")
                row["matched_feature_strategy"] = best_external.get("model_family", "")
        else:
            row["skipped"] = True
            row["skip_reason"] = f"{target_type} target documented but not protocol-matched by this script"

        rows.append(row)
    return pd.DataFrame(rows)
