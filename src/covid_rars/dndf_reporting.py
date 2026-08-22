from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_dndf_summary_table(
    track_a_metrics: pd.DataFrame,
    track_b_chron_metrics: pd.DataFrame,
    track_b_cal_metrics: pd.DataFrame,
    track_c_external_metrics: pd.DataFrame,
    track_a_fusion_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build standardized comparison summary table for DNDT/DNDF models across all evaluation tracks."""
    rows: list[dict[str, Any]] = []

    # 1. Track A (Literature-Aligned 10 Repeated Holdouts)
    if not track_a_metrics.empty:
        test_rows = track_a_metrics[track_a_metrics["split"] == "test"]
        for (mod, model), group in test_rows.groupby(["modality", "model_name"]):
            rows.append({
                "track": "Track A (Repeated Holdouts)",
                "modality": mod,
                "model_name": model,
                "mean_auroc": float(group["auroc"].mean()),
                "std_auroc": float(group["auroc"].std()),
                "mean_auprc": float(group["auprc"].mean()),
                "std_auprc": float(group["auprc"].std()),
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "n_evaluations": len(group),
            })

    # Track A Multimodal Fusion
    if track_a_fusion_metrics is not None and not track_a_fusion_metrics.empty:
        f_test = track_a_fusion_metrics[track_a_fusion_metrics["split"] == "test"]
        for (combo, f_method), group in f_test.groupby(["modality_combination", "fusion_method"]):
            rows.append({
                "track": "Track A (Multimodal Fusion)",
                "modality": combo,
                "model_name": f_method,
                "mean_auroc": float(group["auroc"].mean()),
                "std_auroc": float(group["auroc"].std()),
                "mean_auprc": float(group["auprc"].mean()),
                "std_auprc": float(group["auprc"].std()),
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "n_evaluations": len(group),
            })

    # 2. Track B (Chronological early-to-late)
    if not track_b_chron_metrics.empty:
        c_test = track_b_chron_metrics[track_b_chron_metrics["split"] == "test"]
        for (mod, model), group in c_test.groupby(["modality", "model_name"]):
            rows.append({
                "track": "Track B (Chronological Early->Late)",
                "modality": mod,
                "model_name": model,
                "mean_auroc": float(group["auroc"].mean()),
                "std_auroc": 0.0,
                "mean_auprc": float(group["auprc"].mean()),
                "std_auprc": 0.0,
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "n_evaluations": len(group),
            })

    # Track B Calendar-mixed baseline
    if not track_b_cal_metrics.empty:
        cal_test = track_b_cal_metrics[track_b_cal_metrics["split"] == "test"]
        for (mod, model), group in cal_test.groupby(["modality", "model_name"]):
            rows.append({
                "track": "Track B (Calendar-Mixed Baseline)",
                "modality": mod,
                "model_name": model,
                "mean_auroc": float(group["auroc"].mean()),
                "std_auroc": 0.0,
                "mean_auprc": float(group["auprc"].mean()),
                "std_auprc": 0.0,
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "n_evaluations": len(group),
            })

    # 3. Track C (COUGHVID External Transfer)
    if not track_c_external_metrics.empty:
        for (dset, mod, model), group in track_c_external_metrics.groupby(["dataset", "modality", "model_name"]):
            rows.append({
                "track": f"Track C ({dset})",
                "modality": mod,
                "model_name": model,
                "mean_auroc": float(group["auroc"].mean()),
                "std_auroc": 0.0,
                "mean_auprc": float(group["auprc"].mean()),
                "std_auprc": 0.0,
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "n_evaluations": len(group),
            })

    return pd.DataFrame(rows)


def build_comparative_table_dndf_vs_hst_vs_baselines(
    dndf_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame | None = None,
    hst_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine DNDT/DNDF metrics with Classical baseline and HST transformer numbers for publication."""
    out = dndf_summary.copy()
    out["model_family"] = "DNDT / DNDF (Islam et al. ESWA 2026)"

    combined_list = [out]
    if baseline_summary is not None and not baseline_summary.empty:
        b_df = baseline_summary.copy()
        b_df["model_family"] = "Classical ML / ComParE+IS10"
        combined_list.append(b_df)

    if hst_summary is not None and not hst_summary.empty:
        h_df = hst_summary.copy()
        h_df["model_family"] = "HST Transformer"
        combined_list.append(h_df)

    return pd.concat(combined_list, ignore_index=True)
