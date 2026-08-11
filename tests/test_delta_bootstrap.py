from __future__ import annotations

import pandas as pd


def _predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source, split, pos_prob, neg_prob in [
        ("internal", "test", 0.82, 0.18),
        ("external", "external_test", 0.58, 0.46),
    ]:
        for idx in range(40):
            label = "positive" if idx % 2 else "negative"
            rows.append(
                {
                    "recording_id": f"{source}_{idx}",
                    "participant_id": f"{source}_p{idx}",
                    "label_binary": label,
                    "probability": pos_prob if label == "positive" else neg_prob,
                    "split": split,
                    "evaluation_protocol": source,
                    "model_name": "matched_model",
                    "modality": "cough",
                }
            )
    return pd.DataFrame(rows)


def test_delta_bootstrap_reports_metric_drops_with_ci() -> None:
    from covid_rars.delta_bootstrap import DeltaComparison, build_delta_bootstrap_table

    table = build_delta_bootstrap_table(
        _predictions(),
        comparisons=[
            DeltaComparison(
                comparison_id="internal_minus_external",
                left_name="internal",
                right_name="external",
                left_selector={"evaluation_protocol": "internal"},
                right_selector={"evaluation_protocol": "external"},
            )
        ],
        metrics=["auroc", "auprc"],
        n_bootstraps=25,
        random_state=0,
    )

    assert set(table["metric"]) == {"auroc", "auprc"}
    assert table["comparison_id"].eq("internal_minus_external").all()
    assert {"left_point", "right_point", "delta", "ci_low", "ci_high", "paired"}.issubset(table.columns)
    assert table["delta"].notna().all()
    assert (table["left_n"] == 40).all()
    assert (table["right_n"] == 40).all()


def test_delta_bootstrap_uses_paired_participant_points_when_paired() -> None:
    from covid_rars.delta_bootstrap import DeltaComparison, build_delta_bootstrap_table

    rows = []
    labels = {
        "p1": "positive",
        "p2": "positive",
        "p3": "negative",
        "p4": "negative",
    }
    left_probs = {
        "p1": [0.95, 0.05],
        "p2": [0.90, 0.90],
        "p3": [0.20, 0.20],
        "p4": [0.85, 0.15],
    }
    right_probs = {
        "p1": [0.70, 0.70],
        "p2": [0.65, 0.65],
        "p3": [0.35, 0.35],
        "p4": [0.30, 0.30],
    }
    for source, probs_by_participant in [("left", left_probs), ("right", right_probs)]:
        for participant_id, probs in probs_by_participant.items():
            for recording_idx, prob in enumerate(probs):
                rows.append(
                    {
                        "recording_id": f"{source}_{participant_id}_{recording_idx}",
                        "participant_id": participant_id,
                        "label_binary": labels[participant_id],
                        "probability": prob,
                        "evaluation_protocol": source,
                    }
                )
    predictions = pd.DataFrame(rows)

    table = build_delta_bootstrap_table(
        predictions,
        comparisons=[
            DeltaComparison(
                comparison_id="left_minus_right",
                left_name="left",
                right_name="right",
                left_selector={"evaluation_protocol": "left"},
                right_selector={"evaluation_protocol": "right"},
            )
        ],
        metrics=["brier"],
        n_bootstraps=25,
        random_state=0,
    )

    row = table.iloc[0]
    left_participant_brier = ((1 - 0.50) ** 2 + (1 - 0.90) ** 2 + 0.20**2 + 0.50**2) / 4
    right_participant_brier = ((1 - 0.70) ** 2 + (1 - 0.65) ** 2 + 0.35**2 + 0.30**2) / 4

    assert row["paired"]
    assert row["paired_n"] == 4
    assert row["left_point"] == left_participant_brier
    assert row["right_point"] == right_participant_brier
    assert row["delta"] == left_participant_brier - right_participant_brier


def test_independent_delta_uses_participant_means_on_clustered_source() -> None:
    from covid_rars.delta_bootstrap import DeltaComparison, build_delta_bootstrap_table

    rows: list[dict[str, object]] = []
    source = {
        "p1": ("positive", [0.95, 0.05]),
        "p2": ("positive", [0.90]),
        "p3": ("negative", [0.20]),
        "p4": ("negative", [0.85, 0.15]),
    }
    for participant_id, (label, probabilities) in source.items():
        for recording_idx, probability in enumerate(probabilities):
            rows.append(
                {
                    "recording_id": f"source_{participant_id}_{recording_idx}",
                    "participant_id": participant_id,
                    "label_binary": label,
                    "probability": probability,
                    "evaluation_protocol": "source",
                }
            )
    for idx, (label, probability) in enumerate(
        [("positive", 0.70), ("positive", 0.65), ("negative", 0.35), ("negative", 0.30)]
    ):
        rows.append(
            {
                "recording_id": f"target_{idx}",
                "participant_id": f"target_{idx}",
                "label_binary": label,
                "probability": probability,
                "evaluation_protocol": "target",
            }
        )

    table = build_delta_bootstrap_table(
        pd.DataFrame(rows),
        comparisons=[
            DeltaComparison(
                comparison_id="source_minus_target",
                left_name="source",
                right_name="target",
                left_selector={"evaluation_protocol": "source"},
                right_selector={"evaluation_protocol": "target"},
                paired_on=(),
                left_cluster_on=("participant_id",),
            )
        ],
        metrics=["brier"],
        n_bootstraps=25,
        random_state=0,
    )

    row = table.iloc[0]
    source_participant_brier = ((1 - 0.50) ** 2 + (1 - 0.90) ** 2 + 0.20**2 + 0.50**2) / 4
    target_row_brier = ((1 - 0.70) ** 2 + (1 - 0.65) ** 2 + 0.35**2 + 0.30**2) / 4

    assert not row["paired"]
    assert row["resampling_scheme"] == "independent_two_sample"
    assert row["left_resampling_level"] == "participant_id"
    assert row["right_resampling_level"] == "row"
    assert row["left_n"] == 6
    assert row["left_analysis_n"] == 4
    assert row["right_analysis_n"] == 4
    assert row["left_point"] == source_participant_brier
    assert row["right_point"] == target_row_brier
    assert row["delta"] == source_participant_brier - target_row_brier


def test_independent_cluster_bootstrap_is_invariant_to_duplicate_source_recordings() -> None:
    from covid_rars.delta_bootstrap import DeltaComparison, build_delta_bootstrap_table

    base_rows = [
        {"participant_id": "p1", "label_binary": "positive", "probability": 0.9},
        {"participant_id": "p2", "label_binary": "positive", "probability": 0.8},
        {"participant_id": "p3", "label_binary": "negative", "probability": 0.3},
        {"participant_id": "p4", "label_binary": "negative", "probability": 0.2},
    ]
    target_rows = [
        {"participant_id": "t1", "label_binary": "positive", "probability": 0.7},
        {"participant_id": "t2", "label_binary": "positive", "probability": 0.6},
        {"participant_id": "t3", "label_binary": "negative", "probability": 0.4},
        {"participant_id": "t4", "label_binary": "negative", "probability": 0.3},
    ]

    comparison = DeltaComparison(
        comparison_id="source_minus_target",
        left_name="source",
        right_name="target",
        left_selector={"evaluation_protocol": "source"},
        right_selector={"evaluation_protocol": "target"},
        paired_on=(),
        left_cluster_on=("participant_id",),
    )

    def _table(source_rows: list[dict[str, object]]) -> pd.DataFrame:
        source_frame = pd.DataFrame(source_rows).assign(evaluation_protocol="source")
        target_frame = pd.DataFrame(target_rows).assign(evaluation_protocol="target")
        predictions = pd.concat([source_frame, target_frame], ignore_index=True)
        return build_delta_bootstrap_table(
            predictions,
            comparisons=[comparison],
            metrics=["brier"],
            n_bootstraps=100,
            random_state=7,
        )

    baseline = _table(base_rows).iloc[0]
    duplicated = _table(base_rows + [base_rows[0].copy() for _ in range(50)]).iloc[0]

    assert duplicated["left_n"] == baseline["left_n"] + 50
    assert duplicated["left_analysis_n"] == baseline["left_analysis_n"] == 4
    for column in ["left_point", "right_point", "delta", "bootstrap_mean", "ci_low", "ci_high"]:
        assert duplicated[column] == baseline[column]


def test_auto_ladder_comparisons_include_matched_cough_models() -> None:
    from covid_rars.delta_bootstrap import build_auto_reviewer_comparisons

    final_summary = pd.DataFrame(
        [
            {
                "evaluation_protocol": "compare_is10_existing_participant_split",
                "analysis_family": "strong_multimodal_fusion",
                "model_name": "fusion",
                "modality": "multimodal",
                "modality_combination": "cough+speech",
                "fusion_method": "stacked_logistic_validation",
                "auroc": 0.897,
            },
            {
                "evaluation_protocol": "compare_is10_time_stratified_participant_split",
                "analysis_family": "strong_multimodal_fusion",
                "model_name": "fusion",
                "modality": "multimodal",
                "modality_combination": "cough+breath+speech",
                "fusion_method": "uniform_mean",
                "auroc": 0.849,
            },
            {
                "evaluation_protocol": "compare_is10_temporal_early_to_late",
                "analysis_family": "strong_audio_modality",
                "model_name": "top_4_validation_ensemble",
                "modality": "breath",
                "auroc": 0.698,
            },
        ]
    )
    final_metrics = pd.DataFrame(
        [
            {
                "evaluation_protocol": "compare_is10_existing_participant_split",
                "analysis_family": "strong_audio_modality",
                "model_name": "lightgbm_smote_f80",
                "modality": "cough",
                "auroc": 0.868,
            }
        ]
    )
    external_metrics = pd.DataFrame(
        [
            {
                "evaluation_protocol": "coswara_to_coughvid_compare_is10_external",
                "analysis_family": "compare_is10_external_transfer",
                "model_name": "lightgbm_smote_f80",
                "modality": "cough",
                "auroc": 0.543,
                "skipped": False,
            }
        ]
    )

    comparisons = build_auto_reviewer_comparisons(final_summary, final_metrics, external_metrics)
    ids = {comparison.comparison_id for comparison in comparisons}

    assert "existing_participant_split_minus_time_stratified" in ids
    assert "time_stratified_minus_temporal_early_to_late" in ids
    assert "existing_cough_lightgbm_smote_f80_minus_coughvid_external" in ids
    matched = next(
        comparison
        for comparison in comparisons
        if comparison.comparison_id == "existing_cough_lightgbm_smote_f80_minus_coughvid_external"
    )
    assert matched.paired_on == ()
    assert matched.left_cluster_on == ("participant_id",)
    assert matched.right_cluster_on == ()
