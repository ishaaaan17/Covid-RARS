from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _metadata_frame(n_participants: int = 36) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(n_participants):
        month = 1 + idx // 3
        label = "positive" if month >= 7 or idx % 5 == 0 else "negative"
        split = "train" if idx < 20 else "validation" if idx < 28 else "test"
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
                "recording_date": f"2020-{month:02d}-{(idx % 27) + 1:02d}",
                "age": 24 + (idx % 16),
                "gender": "female" if idx % 2 else "male",
                "country": "India" if idx % 4 else "United States",
                "symptoms_json": '{"cough": true, "fever": true}' if label == "positive" else '{"cough": false, "fever": false}',
                "comorbidities_json": '{"asthma": false}',
                "manual_quality_score": np.nan,
                "manual_quality_label": "unknown",
                "split": split,
                "duration_sec": 4.0 + (idx % 6),
                "sample_rate_original": 48000,
                "quality_flag": "ok" if idx % 7 else "mostly_silence",
            }
        )
    return pd.DataFrame(rows)


def _feature_frame(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in metadata.iterrows():
        signal = 1.0 if row["label_binary"] == "positive" else -1.0
        idx = int(str(row["participant_id"]).split("_")[-1])
        for modality, offset in {"cough": 0.0, "breath": 0.15, "speech": -0.15}.items():
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
                    "feat_aux": signal * 0.25 + offset,
                }
            )
    return pd.DataFrame(rows)


def _prediction_frame(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for protocol in ["existing_participant_split", "temporal_early_to_late"]:
        for _, row in metadata.tail(8).iterrows():
            prob = 0.82 if row["label_binary"] == "positive" else 0.18
            if protocol == "temporal_early_to_late" and int(str(row["participant_id"]).split("_")[-1]) % 3 == 0:
                prob = 1.0 - prob
            rows.append(
                {
                    "recording_id": row["recording_id"],
                    "participant_id": row["participant_id"],
                    "label_binary": row["label_binary"],
                    "split": "test",
                    "evaluation_protocol": protocol,
                    "analysis_family": "multimodal_fusion",
                    "model_name": "logistic_regression",
                    "modality": "multimodal",
                    "modality_combination": "breath+cough+speech",
                    "fusion_method": "uniform_mean",
                    "probability": prob,
                    "threshold": 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_month_causal_audit_summarizes_label_covariate_and_effect_size() -> None:
    from covid_rars.temporal_month_causal import (
        build_month_ablation_effect_sizes,
        build_month_covariate_shift,
        build_month_label_shift,
    )

    metadata = _metadata_frame()
    ablation = pd.DataFrame(
        [
            {"metadata_configuration": "Full metadata", "temporal_auroc": 0.531},
            {"metadata_configuration": "Remove year", "temporal_auroc": 0.531},
            {"metadata_configuration": "Remove month", "temporal_auroc": 0.779},
            {"metadata_configuration": "Remove year + month", "temporal_auroc": 0.779},
        ]
    )

    labels = build_month_label_shift(metadata)
    covariates = build_month_covariate_shift(metadata)
    effects = build_month_ablation_effect_sizes(ablation)

    assert {"recording_year_month", "positive_prevalence", "symptom_count_mean"}.issubset(labels.columns)
    assert labels["positive_prevalence"].between(0, 1).all()
    assert "symptom_share_fever" in set(covariates["covariate"])
    remove_month = effects[effects["comparison_configuration"].eq("Remove month")].iloc[0]
    assert round(remove_month["delta_auroc"], 3) == 0.248
    assert remove_month["interpretation"] == "improves_temporal_generalization"



def test_symptom_count_treats_string_false_as_false() -> None:
    from covid_rars.temporal_month_causal import build_month_label_shift

    metadata = pd.DataFrame(
        [
            {
                "participant_id": "p1",
                "label_binary": "negative",
                "recording_date": "2020-01-01",
                "age": 30,
                "gender": "male",
                "country": "India",
                "quality_flag": "ok",
                "duration_sec": 4.0,
                "symptoms_json": '{"cough": "False", "fever": "no", "fatigue": "0"}',
            },
            {
                "participant_id": "p2",
                "label_binary": "positive",
                "recording_date": "2020-01-02",
                "age": 31,
                "gender": "female",
                "country": "India",
                "quality_flag": "ok",
                "duration_sec": 5.0,
                "symptoms_json": '{"cough": "True", "fever": "yes", "fatigue": "1"}',
            },
        ]
    )

    summary = build_month_label_shift(metadata)

    assert summary["symptom_count_mean"].iloc[0] == 1.5



def test_matched_failure_and_uncertainty_outputs_are_estimable() -> None:
    from covid_rars.temporal_month_causal import (
        build_failure_mode_summary,
        build_uncertainty_summary,
        evaluate_matched_temporal_audio,
    )

    metadata = _metadata_frame(48)
    features = _feature_frame(metadata)
    predictions = _prediction_frame(metadata)

    matched = evaluate_matched_temporal_audio(metadata, features, random_state=0)
    failures = build_failure_mode_summary(metadata, predictions)
    uncertainty = build_uncertainty_summary(predictions)

    assert not matched.empty
    assert {"matched_audio_modality", "matched_multimodal_fusion"}.intersection(set(matched["analysis_family"]))
    assert not failures.empty
    assert {"prediction_outcome", "top_month", "mean_symptom_count"}.issubset(failures.columns)
    assert not uncertainty.empty
    assert {"mean_confidence", "high_confidence_error_rate"}.issubset(uncertainty.columns)


def test_temporal_month_causal_cli_writes_expected_outputs(tmp_path: Path) -> None:
    metadata = _metadata_frame(48)
    features = _feature_frame(metadata)
    predictions = _prediction_frame(metadata)
    ablation = pd.DataFrame(
        [
            {"metadata_configuration": "Full metadata", "temporal_auroc": 0.531},
            {"metadata_configuration": "Remove month", "temporal_auroc": 0.779},
        ]
    )
    metadata_path = tmp_path / "metadata.csv"
    features_path = tmp_path / "features.csv"
    predictions_path = tmp_path / "predictions.csv"
    ablation_path = tmp_path / "ablation.csv"
    metadata.to_csv(metadata_path, index=False)
    features.to_csv(features_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    ablation.to_csv(ablation_path, index=False)

    outputs = {
        "--month-label-shift-output": tmp_path / "labels.csv",
        "--month-covariate-shift-output": tmp_path / "covariates.csv",
        "--matched-cohort-output": tmp_path / "matched.csv",
        "--failure-modes-output": tmp_path / "failures.csv",
        "--uncertainty-output": tmp_path / "uncertainty.csv",
        "--effect-sizes-output": tmp_path / "effects.csv",
        "--dag-output": tmp_path / "dag.md",
        "--theory-output": tmp_path / "theory.md",
    }

    script = Path(__file__).parents[1] / "scripts" / "46_temporal_month_causal_audit.py"
    spec = importlib.util.spec_from_file_location("temporal_month_causal_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv = [
        "46_temporal_month_causal_audit.py",
        "--metadata",
        str(metadata_path),
        "--features",
        str(features_path),
        "--temporal-predictions",
        str(predictions_path),
        "--month-ablation",
        str(ablation_path),
    ]
    for flag, path in outputs.items():
        argv.extend([flag, str(path)])
    old_argv = sys.argv
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = old_argv

    for path in outputs.values():
        assert path.exists()
    assert "Calendar month" in outputs["--dag-output"].read_text()
    assert "collection-period shortcut" in outputs["--theory-output"].read_text()
