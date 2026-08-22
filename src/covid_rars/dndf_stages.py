from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from covid_rars.dndf_protocols import (
    run_track_a_repeated_holdouts,
    run_track_b_temporal_contrast,
    run_track_c_external_transfer,
)
from covid_rars.dndf_reliability import (
    compute_dndf_calibration_summary,
    compute_dndf_decision_curve_analysis,
    compute_dndf_fixed_sensitivity_operating_points,
    run_dndf_bootstrap_uncertainty,
)
from covid_rars.dndf_reporting import build_dndf_summary_table

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DNDFPipelineArtifacts:
    track_a_metrics: pd.DataFrame
    track_a_predictions: pd.DataFrame
    track_a_fusion_metrics: pd.DataFrame
    track_a_fusion_predictions: pd.DataFrame
    track_b_chron_metrics: pd.DataFrame
    track_b_cal_metrics: pd.DataFrame
    track_c_external_metrics: pd.DataFrame
    track_c_external_predictions: pd.DataFrame
    calibration_summary: pd.DataFrame
    operating_points: pd.DataFrame
    dca_summary: pd.DataFrame
    bootstrap_ci: pd.DataFrame
    final_summary_table: pd.DataFrame


def run_dndf_reliability_pipeline(
    features_df: pd.DataFrame,
    external_features_df: pd.DataFrame | None = None,
    modalities: Sequence[str] = ("cough", "breath", "speech"),
    seeds: Sequence[int] = (1, 2, 5, 12, 40),
    num_trees: int = 20,
    depth: int = 4,
    used_features_rate: float = 0.8,
    learning_rate: float = 0.01,
    max_epochs: int = 50,
    patience: int = 10,
    use_smote: bool = True,
    device: str = "auto",
    output_dir: Path | str | None = None,
) -> DNDFPipelineArtifacts:
    """Execute end-to-end DNDT and DNDF reliability study pipeline."""
    out_path = Path(output_dir) if output_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting DNDT / DNDF Reliability Pipeline ===")

    # 1. Track A: Repeated Stratified Holdouts
    logger.info("Running Track A: Literature-Aligned Repeated Holdouts...")
    track_a_res = run_track_a_repeated_holdouts(
        features=features_df,
        modalities=modalities,
        seeds=seeds,
        num_trees=num_trees,
        depth=depth,
        used_features_rate=used_features_rate,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        patience=patience,
        use_smote=use_smote,
        device=device,
    )

    # 2. Track B: Temporal Contrast
    logger.info("Running Track B: Chronological vs Calendar-Mixed Contrast...")
    track_b_chron, track_b_cal = run_track_b_temporal_contrast(
        features=features_df,
        modalities=modalities,
        num_trees=num_trees,
        depth=depth,
        used_features_rate=used_features_rate,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        patience=patience,
        use_smote=use_smote,
        device=device,
        random_state=42,
    )

    # 3. Track C: External COUGHVID Transfer (if external data provided)
    if external_features_df is not None and not external_features_df.empty:
        logger.info("Running Track C: COUGHVID External Transfer...")
        track_c_res = run_track_c_external_transfer(
            source_features=features_df,
            target_external_features=external_features_df,
            modality="cough",
            num_trees=num_trees,
            depth=depth,
            used_features_rate=used_features_rate,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            patience=patience,
            use_smote=use_smote,
            device=device,
            random_state=42,
        )
    else:
        logger.info("No external features provided. Skipping Track C.")
        track_c_res = None

    track_c_metrics = track_c_res.metrics if track_c_res else pd.DataFrame()
    track_c_preds = track_c_res.predictions if track_c_res else pd.DataFrame()

    # 4. Reliability Audits (Calibration, Operating Points, DCA, Bootstrap CIs)
    logger.info("Running Reliability Audits (Calibration, Operating Points, DCA, Bootstraps)...")
    all_eval_preds = []
    if not track_a_res.predictions.empty:
        all_eval_preds.append(track_a_res.predictions)
    if not track_a_res.multimodal_predictions.empty:
        all_eval_preds.append(track_a_res.multimodal_predictions)
    if not track_b_chron.predictions.empty:
        all_eval_preds.append(track_b_chron.predictions)
    if not track_b_cal.predictions.empty:
        all_eval_preds.append(track_b_cal.predictions)
    if not track_c_preds.empty:
        all_eval_preds.append(track_c_preds)

    merged_preds = pd.concat(all_eval_preds, ignore_index=True) if all_eval_preds else pd.DataFrame()

    cal_df = compute_dndf_calibration_summary(merged_preds)
    op_df = compute_dndf_fixed_sensitivity_operating_points(merged_preds, min_sensitivity=0.90)
    dca_df = compute_dndf_decision_curve_analysis(merged_preds)
    boot_df = run_dndf_bootstrap_uncertainty(merged_preds, n_bootstraps=200)

    # 5. Build Final Summary Table
    logger.info("Building Final Summary Tables...")
    summary_table = build_dndf_summary_table(
        track_a_metrics=track_a_res.metrics,
        track_b_chron_metrics=track_b_chron.metrics,
        track_b_cal_metrics=track_b_cal.metrics,
        track_c_external_metrics=track_c_metrics,
        track_a_fusion_metrics=track_a_res.multimodal_metrics,
    )

    artifacts = DNDFPipelineArtifacts(
        track_a_metrics=track_a_res.metrics,
        track_a_predictions=track_a_res.predictions,
        track_a_fusion_metrics=track_a_res.multimodal_metrics,
        track_a_fusion_predictions=track_a_res.multimodal_predictions,
        track_b_chron_metrics=track_b_chron.metrics,
        track_b_cal_metrics=track_b_cal.metrics,
        track_c_external_metrics=track_c_metrics,
        track_c_external_predictions=track_c_preds,
        calibration_summary=cal_df,
        operating_points=op_df,
        dca_summary=dca_df,
        bootstrap_ci=boot_df,
        final_summary_table=summary_table,
    )

    if out_path:
        summary_table.to_csv(out_path / "dndf_final_validation_summary.csv", index=False)
        cal_df.to_csv(out_path / "dndf_calibration_summary.csv", index=False)
        op_df.to_csv(out_path / "dndf_operating_points.csv", index=False)
        dca_df.to_csv(out_path / "dndf_decision_curves.csv", index=False)
        boot_df.to_csv(out_path / "dndf_bootstrap_ci.csv", index=False)
        logger.info(f"Artifacts successfully written to: {out_path}")

    return artifacts
