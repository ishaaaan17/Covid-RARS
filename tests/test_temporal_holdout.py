from __future__ import annotations

import numpy as np
import pandas as pd


def _metadata_frame(n_participants: int = 36) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(n_participants):
        label = "positive" if idx % 3 != 0 else "negative"
        current_split = "train" if idx % 6 < 4 else "validation" if idx % 6 == 4 else "test"
        rows.append(
            {
                "recording_id": f"rec_{idx}",
                "participant_id": f"p_{idx:03d}",
                "dataset": "coswara",
                "modality": "cough",
                "submodality": "cough-heavy",
                "label_raw": "positive_mild" if label == "positive" else "healthy",
                "label_binary": label,
                "label_group": label,
                "recording_date": f"2020-{(idx // 3) + 1:02d}-{(idx % 27) + 1:02d}",
                "age": 25 + (idx % 20),
                "gender": "female" if idx % 2 else "male",
                "country": "India" if idx % 4 else "United States",
                "symptoms_json": '{"cough": true, "fever": true}' if label == "positive" else '{"cough": false}',
                "comorbidities_json": '{"asthma": true}' if idx % 5 == 0 else '{"asthma": false}',
                "manual_quality_score": np.nan,
                "manual_quality_label": "unknown",
                "split": current_split,
                "duration_sec": 4.0 + (idx % 7),
                "sample_rate_original": 48000,
                "quality_flag": "ok",
            }
        )
    return pd.DataFrame(rows)


def _feature_frame(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    modality_offsets = {"cough": 0.0, "breath": 0.25, "speech": -0.25}
    for _, row in metadata.iterrows():
        signal = 1.0 if row["label_binary"] == "positive" else -1.0
        idx = int(str(row["participant_id"]).split("_")[-1])
        for modality, offset in modality_offsets.items():
            rows.append(
                {
                    "recording_id": f"{row['recording_id']}_{modality}",
                    "participant_id": row["participant_id"],
                    "dataset": "coswara",
                    "modality": modality,
                    "submodality": modality,
                    "label_binary": row["label_binary"],
                    "split": row["split"],
                    "feat_signal": signal + offset + idx * 0.001,
                    "feat_aux": signal * 0.3 + offset,
                    "feat_noise": (idx % 5) * 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_temporal_split_is_participant_level_and_ordered() -> None:
    from covid_rars.temporal_holdout import build_temporal_split_assignments

    metadata = _metadata_frame()
    assignments, summary = build_temporal_split_assignments(
        metadata,
        train_fraction=0.5,
        validation_fraction=0.25,
    )

    assert assignments["participant_id"].is_unique
    assert set(assignments["temporal_split"]) == {"train", "validation", "test"}

    ranges = summary.set_index("temporal_split")
    assert pd.Timestamp(ranges.loc["train", "date_max"]) <= pd.Timestamp(ranges.loc["validation", "date_min"])
    assert pd.Timestamp(ranges.loc["validation", "date_max"]) <= pd.Timestamp(ranges.loc["test", "date_min"])
    assert {"n_participants", "n_positive", "n_negative", "positive_prevalence"}.issubset(summary.columns)


def test_temporal_holdout_audit_reports_audio_metadata_and_fusion() -> None:
    from covid_rars.temporal_holdout import run_temporal_holdout_audit

    metadata = _metadata_frame()
    features = _feature_frame(metadata)

    result = run_temporal_holdout_audit(
        metadata=metadata,
        features=features,
        modalities=["cough", "breath", "speech"],
        model_names=["logistic_regression"],
        include_existing_split_reference=True,
        bootstrap_samples=25,
        random_state=0,
    )

    assert {
        "temporal_early_to_late",
        "existing_participant_split",
        "time_stratified_participant_split",
    }.issubset(
        set(result.metrics["evaluation_protocol"])
    )
    assert {"audio_modality", "multimodal_fusion", "metadata_confounding"}.issubset(
        set(result.metrics["analysis_family"])
    )
    audio = result.metrics[result.metrics["analysis_family"].eq("audio_modality")]
    assert {"cough", "breath", "speech"}.issubset(set(audio["modality"]))
    fusion = result.metrics[result.metrics["analysis_family"].eq("multimodal_fusion")]
    assert not fusion.empty
    assert fusion["fusion_method"].notna().all()
    expected_combinations = {
        "breath+cough",
        "cough+speech",
        "breath+speech",
        "breath+cough+speech",
    }
    assert expected_combinations.issubset(set(fusion["modality_combination"]))
    assert "test_positive_prevalence" in result.metrics.columns
    assert "auprc_lift_over_prevalence" in result.metrics.columns
    assert fusion["test_positive_prevalence"].notna().all()
    finite_auprc = fusion["auprc"].notna()
    assert fusion.loc[finite_auprc, "auprc_lift_over_prevalence"].notna().all()
    assert result.metrics["test_positive_prevalence"].notna().any()
    assert result.metrics["auprc_lift_over_prevalence"].notna().any()
    metadata_rows = result.metrics[result.metrics["analysis_family"].eq("metadata_confounding")]
    assert {"symptoms_only", "demographic_protocol_only", "full_safe_metadata"}.issubset(
        set(metadata_rows["audit_model"])
    )
    assert not result.predictions.empty
    assert not result.split_summary.empty
    assert not result.modality_coverage.empty


def test_temporal_robustness_outputs_ablation_stability_and_bootstrap() -> None:
    from covid_rars.temporal_holdout import run_temporal_holdout_audit

    metadata = _metadata_frame()
    features = _feature_frame(metadata)

    result = run_temporal_holdout_audit(
        metadata=metadata,
        features=features,
        modalities=["cough", "breath", "speech"],
        model_names=["logistic_regression"],
        bootstrap_samples=25,
        random_state=0,
    )

    assert not result.metadata_ablation.empty
    ablations = set(result.metadata_ablation["ablation_name"].astype(str))
    assert {
        "demographic_protocol_full",
        "demographic_protocol_no_recording_year",
        "demographic_protocol_no_recording_month",
        "demographic_protocol_no_recording_year_month",
        "recording_year_only",
        "recording_month_only",
        "recording_year_month_only",
    }.issubset(ablations)
    assert {"removed_features", "n_features", "auroc", "auprc"}.issubset(result.metadata_ablation.columns)

    assert not result.stability_summary.empty
    assert {
        "delta_auroc_temporal_minus_existing",
        "delta_ece_temporal_minus_existing",
        "delta_brier_temporal_minus_existing",
        "delta_auprc_lift_temporal_minus_existing",
    }.issubset(result.stability_summary.columns)

    assert not result.bootstrap_ci.empty
    assert {"auroc", "auprc", "balanced_accuracy", "brier", "ece"}.issubset(
        set(result.bootstrap_ci["metric"])
    )
    assert result.bootstrap_ci["ci_low"].notna().any()
    assert result.bootstrap_ci["ci_high"].notna().any()

    assert not result.external_unification.empty
    assert {"participant_internal", "temporal_holdout"}.issubset(
        set(result.external_unification["stress_test"])
    )
