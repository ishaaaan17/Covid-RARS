from __future__ import annotations

import pandas as pd


def _metadata() -> pd.DataFrame:
    rows = []
    labels = {
        "tr0": "negative",
        "tr1": "positive",
        "tr2": "negative",
        "tr3": "positive",
        "v0": "negative",
        "v1": "positive",
        "v2": "negative",
        "v3": "positive",
        "t0": "negative",
        "t1": "positive",
        "t2": "negative",
        "t3": "positive",
    }
    for participant_id, label in labels.items():
        if participant_id.startswith("tr"):
            split = "train"
        elif participant_id.startswith("v"):
            split = "validation"
        else:
            split = "test"
        positive = label == "positive"
        rows.append(
            {
                "participant_id": participant_id,
                "recording_id": f"{participant_id}_rec",
                "label_binary": label,
                "split": split,
                "age": 55 if positive else 25,
                "gender": "female" if positive else "male",
                "country": "A" if positive else "B",
                "recording_date": "2020-07-01" if positive else "2020-04-01",
                "symptoms_json": '{"cough": true}' if positive else '{"cough": false}',
                "comorbidities_json": "{}",
                "duration_sec": 4.5 if positive else 2.0,
                "sample_rate_original": 48000,
                "quality_flag": "ok",
            }
        )
    return pd.DataFrame(rows)


def _audio_predictions() -> pd.DataFrame:
    rows = []
    label_by_id = {row["participant_id"]: row["label_binary"] for row in _metadata().to_dict("records")}
    for source_name, has_validation, val_good, test_good in [
        ("usable_audio", True, True, True),
        ("test_only_audio", False, True, True),
        ("weak_audio", True, False, False),
    ]:
        for participant_id, label in label_by_id.items():
            if participant_id.startswith("tr"):
                continue
            split = "validation" if participant_id.startswith("v") else "test"
            if split == "validation" and not has_validation:
                continue
            positive = label == "positive"
            good = val_good if split == "validation" else test_good
            if good:
                probability = 0.85 if positive else 0.15
            else:
                probability = 0.45 if positive else 0.55
            rows.append(
                {
                    "participant_id": participant_id,
                    "recording_id": f"{participant_id}_{source_name}",
                    "label_binary": label,
                    "split": split,
                    "evaluation_protocol": "synthetic_protocol",
                    "analysis_family": "audio_family",
                    "model_name": source_name,
                    "modality": "cough",
                    "submodality": "cough",
                    "feature_strategy": "synthetic_audio",
                    "probability": probability,
                }
            )
    return pd.DataFrame(rows)


def test_build_audio_source_candidates_requires_validation_and_test_overlap() -> None:
    from covid_rars.incremental_value import build_audio_source_candidates, build_metadata_probability_table

    metadata_predictions = build_metadata_probability_table(_metadata(), feature_set="symptoms_only")
    candidates = build_audio_source_candidates(_audio_predictions(), metadata_predictions, top_k=5)

    assert not candidates.empty
    assert "usable_audio" in set(candidates["model_name"])
    assert "weak_audio" in set(candidates["model_name"])
    assert "test_only_audio" not in set(candidates["model_name"])
    assert candidates.iloc[0]["model_name"] == "usable_audio"
    assert candidates.iloc[0]["validation_auroc"] > 0.9


def test_incremental_value_compares_same_test_participants() -> None:
    from covid_rars.incremental_value import build_incremental_audio_metadata_value

    metrics, predictions, candidates = build_incremental_audio_metadata_value(
        metadata=_metadata(),
        audio_predictions=_audio_predictions(),
        metadata_feature_sets=["symptoms_only"],
        top_k_audio_sources=1,
        n_bootstraps=25,
        random_state=0,
    )

    assert not candidates.empty
    assert set(metrics["nested_model"]) == {"metadata_only", "audio_only", "metadata_plus_audio"}
    assert metrics["n_validation_aligned"].eq(4).all()
    assert metrics["n_test_aligned"].eq(4).all()
    assert predictions.groupby("nested_model")["participant_id"].nunique().eq(4).all()

    wide = metrics.set_index("nested_model")
    assert wide.loc["metadata_plus_audio", "delta_auroc_vs_metadata"] >= 0
    assert wide.loc["metadata_plus_audio", "delta_auroc_ci_low_vs_metadata"] <= wide.loc["metadata_plus_audio", "delta_auroc_vs_metadata"]
    assert wide.loc["metadata_plus_audio", "delta_auroc_ci_high_vs_metadata"] >= wide.loc["metadata_plus_audio", "delta_auroc_vs_metadata"]


def test_external_model_family_summary_uses_internal_and_external_rows() -> None:
    from covid_rars.incremental_value import build_external_model_family_transfer_summary

    compare_external = pd.DataFrame(
        [
            {
                "analysis_family": "compare_is10_external_transfer",
                "model_name": "lightgbm_smote_f80",
                "modality": "cough",
                "metric_split": "external_test",
                "auroc": 0.54,
                "auprc": 0.04,
            }
        ]
    )
    compare_internal = pd.DataFrame(
        [
            {
                "evaluation_protocol": "compare_is10_existing_participant_split",
                "analysis_family": "strong_audio_modality",
                "model_name": "lightgbm_smote_f80",
                "modality": "cough",
                "metric_split": "test",
                "auroc": 0.84,
                "auprc": 0.77,
            }
        ]
    )
    wavlm = pd.DataFrame(
        [
            {
                "analysis_family": "deep_external_transfer",
                "model_name": "wavlm_base_plus_pooled_cough",
                "modality": "cough",
                "metric_split": "test",
                "auroc": 0.81,
                "auprc": 0.73,
            },
            {
                "analysis_family": "deep_external_transfer",
                "model_name": "wavlm_base_plus_pooled_cough",
                "modality": "cough",
                "metric_split": "external_test",
                "auroc": 0.48,
                "auprc": 0.03,
            },
        ]
    )

    summary = build_external_model_family_transfer_summary(
        compare_internal_metrics=compare_internal,
        compare_external_metrics=compare_external,
        wavlm_metrics=wavlm,
        cnn_metrics=pd.DataFrame(),
    )

    assert {"compare_is10_lightgbm_smote_f80", "wavlm_base_plus_pooled_cough"} <= set(summary["family_model"])
    compare_row = summary[summary["family_model"].eq("compare_is10_lightgbm_smote_f80")].iloc[0]
    assert compare_row["internal_auroc"] == 0.84
    assert compare_row["external_auroc"] == 0.54
    assert compare_row["delta_auroc_internal_minus_external"] == 0.30
