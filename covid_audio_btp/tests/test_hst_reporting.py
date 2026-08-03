from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _predictions(prefix: str, probabilities: list[float]) -> pd.DataFrame:
    labels = ["negative", "negative", "positive", "positive"]
    return pd.DataFrame(
        {
            "participant_key": [f"{prefix}::{index}" for index in range(4)],
            "label_binary": labels,
            "probability": probabilities,
            "fold": 1,
            "split": "test",
        }
    )


def test_external_delta_is_independent_two_sample_bootstrap() -> None:
    from covid_audio_btp.hst_reporting import external_transfer_delta

    result = external_transfer_delta(
        _predictions("source", [0.1, 0.2, 0.8, 0.9]),
        _predictions("target", [0.2, 0.7, 0.3, 0.8]),
        metric="auroc",
        n_bootstrap=50,
        seed=42,
    )
    assert result["bootstrap_design"] == "independent_label_stratified_participants"
    assert result["paired"] is False
    assert result["resampling_unit"] == "participant_key"
    assert result["valid_replicates"] == 50
    assert np.isnan(result["p_value"])
    assert result["hypothesis_test"] == "not_tested_bootstrap_ci_only"


def test_external_delta_rejects_missing_recording_uuid_provenance() -> None:
    from covid_audio_btp.hst_reporting import external_transfer_delta

    source = _predictions("source", [0.1, 0.2, 0.8, 0.9]).assign(
        dataset="coswara",
        split="test",
    )
    target = _predictions("target", [0.2, 0.7, 0.3, 0.8]).assign(
        dataset="coughvid",
        split="external_test",
    )

    with pytest.raises(ValueError, match="analysis-unit provenance"):
        external_transfer_delta(
            source,
            target,
            metric="auroc",
            n_bootstrap=20,
            seed=42,
        )


def test_repeated_fold_external_delta_preserves_fold_structure_and_shared_target_resample() -> None:
    from covid_audio_btp.hst_reporting import external_repeated_fold_delta

    source_rows: list[pd.DataFrame] = []
    target_rows: list[pd.DataFrame] = []
    for fold in range(10):
        source_rows.append(
            _predictions(
                "source",
                [0.1 + fold / 1000, 0.2, 0.8, 0.9 - fold / 1000],
            ).assign(fold=fold, dataset="coswara", split="test")
        )
        target_rows.append(
            _predictions(
                "target",
                [0.2 + fold / 1000, 0.7, 0.3, 0.8 - fold / 1000],
            ).assign(
                fold=fold,
                dataset="coughvid",
                split="external_test",
                analysis_unit_type="recording_uuid",
                subject_linkage_available=False,
            )
        )

    result = external_repeated_fold_delta(
        pd.concat(source_rows, ignore_index=True),
        pd.concat(target_rows, ignore_index=True),
        metric="auroc",
        n_bootstrap=40,
        seed=42,
    )

    assert result["bootstrap_design"] == (
        "independent_source_target_label_stratified_participant_clusters_across_folds"
    )
    assert result["source_fold_count"] == 10
    assert result["external_fold_count"] == 10
    assert result["same_source_resample_across_repeated_folds"] is True
    assert result["same_target_resample_across_external_folds"] is True
    assert result["independent_row_pooling"] is False
    assert result["resampling_unit"] == (
        "source_participant_cluster_across_folds_and_"
        "target_participant_cluster_across_folds"
    )
    assert result["source_analysis_unit_type"] == "participant"
    assert result["target_analysis_unit_type"] == "recording_uuid"
    assert result["target_subject_linkage_available"] is False
    assert result["source_point"] == pytest.approx(1.0)
    assert result["delta"] == pytest.approx(
        result["source_point"] - result["target_point"]
    )
    assert np.isfinite(result["ci_low"])
    assert np.isfinite(result["ci_high"])
    assert result["ci_low"] <= result["ci_high"]
    assert np.isnan(result["p_value"])
    assert result["hypothesis_test"] == "not_tested_bootstrap_ci_only"


def test_repeated_fold_external_delta_resamples_recurring_source_participants_jointly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_reporting as reporting

    source = pd.concat(
        [
            _predictions("source", [0.10, 0.35, 0.65, 0.90]).assign(
                fold=fold,
                dataset="coswara",
                split="test",
            )
            for fold in range(3)
        ],
        ignore_index=True,
    )
    target = pd.concat(
        [
            _predictions("target", [0.20, 0.70, 0.30, 0.80]).assign(
                fold=fold,
                dataset="coughvid",
                split="external_test",
                analysis_unit_type="recording_uuid",
                subject_linkage_available=False,
            )
            for fold in range(3)
        ],
        ignore_index=True,
    )
    real_sample = reporting._stratified_sample
    sampled_calls: list[tuple[str, tuple[int, ...], pd.DataFrame]] = []

    def tracked_sample(
        frame: pd.DataFrame,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        sampled = real_sample(frame, rng)
        sampled_calls.append(
            (
                str(frame["dataset"].iloc[0]),
                tuple(sorted(frame["fold"].unique().tolist())),
                sampled.copy(),
            )
        )
        return sampled

    monkeypatch.setattr(reporting, "_stratified_sample", tracked_sample)
    result = reporting.external_repeated_fold_delta(
        source,
        target,
        metric="auroc",
        n_bootstrap=12,
        seed=19,
    )

    source_calls = [call for call in sampled_calls if call[0] == "coswara"]
    target_calls = [call for call in sampled_calls if call[0] == "coughvid"]
    assert len(source_calls) == result["attempts"]
    assert len(target_calls) == result["attempts"]
    assert all(folds == (0, 1, 2) for _, folds, _ in source_calls)
    assert all(folds == (0, 1, 2) for _, folds, _ in target_calls)

    for _, _, sampled_source in source_calls:
        sampled_source = sampled_source.assign(
            original_participant=sampled_source["participant_key"].str.split(
                "::bootstrap::", n=1
            ).str[0]
        )
        fold_multiplicities = sampled_source.groupby(
            ["original_participant", "fold"], sort=True
        ).size().unstack(fill_value=0)
        assert (fold_multiplicities.nunique(axis=1) == 1).all()


def test_repeated_fold_external_delta_is_deterministic_and_target_does_not_change_source_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_reporting as reporting

    source = pd.concat(
        [
            _predictions("source", [0.10, 0.40, 0.60, 0.90]).assign(
                fold=fold,
                dataset="coswara",
                split="test",
            )
            for fold in range(3)
        ],
        ignore_index=True,
    )
    target = pd.concat(
        [
            _predictions("target", [0.20, 0.70, 0.30, 0.80]).assign(
                fold=fold,
                dataset="coughvid",
                split="external_test",
                analysis_unit_type="recording_uuid",
                subject_linkage_available=False,
            )
            for fold in range(3)
        ],
        ignore_index=True,
    )
    real_sample = reporting._stratified_sample

    def run_and_capture(target_frame: pd.DataFrame) -> tuple[dict[str, object], list[tuple[str, ...]]]:
        source_draws: list[tuple[str, ...]] = []

        def tracked_sample(
            frame: pd.DataFrame,
            rng: np.random.Generator,
        ) -> pd.DataFrame:
            sampled = real_sample(frame, rng)
            if str(frame["dataset"].iloc[0]) == "coswara":
                source_draws.append(tuple(sampled["participant_key"].astype(str)))
            return sampled

        monkeypatch.setattr(reporting, "_stratified_sample", tracked_sample)
        result = reporting.external_repeated_fold_delta(
            source,
            target_frame,
            metric="auroc",
            n_bootstrap=20,
            seed=314,
        )
        return result, source_draws

    first, first_source_draws = run_and_capture(target)
    second, second_source_draws = run_and_capture(target.copy())
    changed_target = target.copy()
    changed_target["probability"] = 1.0 - changed_target["probability"]
    changed, changed_source_draws = run_and_capture(changed_target)

    assert first == second
    assert first_source_draws == second_source_draws
    assert first_source_draws == changed_source_draws
    assert changed["source_point"] == first["source_point"]
    assert changed["target_point"] != first["target_point"]
    assert changed["delta"] == pytest.approx(
        changed["source_point"] - changed["target_point"]
    )


def test_equal_fold_probability_ensemble_is_a_separate_participant_endpoint() -> None:
    from covid_audio_btp.hst_reporting import equal_fold_probability_ensemble_ci

    folds = []
    for fold in range(10):
        folds.append(
            _predictions(
                "target",
                [0.1 + fold / 1000, 0.2, 0.8, 0.9 - fold / 1000],
            ).assign(
                fold=fold,
                dataset="coughvid",
                split="external_test",
                analysis_unit_type="recording_uuid",
                subject_linkage_available=False,
            )
        )
    result = equal_fold_probability_ensemble_ci(
        pd.concat(folds, ignore_index=True),
        metric="auroc",
        n_bootstrap=30,
        seed=42,
    )

    assert result["endpoint"] == "equal_source_fold_probability_ensemble"
    assert result["source_fold_count"] == 10
    assert result["n_participants"] == 4
    assert result["independent_row_pooling"] is False
    assert result["point"] == pytest.approx(1.0)


def test_screening_threshold_is_validation_only() -> None:
    from covid_audio_btp.hst_reporting import apply_screening_operating_point, fit_screening_operating_point

    validation = _predictions("validation", [0.1, 0.4, 0.6, 0.9])
    test = _predictions("test", [0.2, 0.3, 0.7, 0.8])
    operating_point = fit_screening_operating_point(validation, target_sensitivity=0.90)
    first = apply_screening_operating_point(test, operating_point)
    reversed_labels = test.assign(label_binary=test["label_binary"].iloc[::-1].to_numpy())
    second = apply_screening_operating_point(reversed_labels, operating_point)
    assert first["threshold"] == operating_point.threshold
    assert second["threshold"] == operating_point.threshold


def test_reporting_contract_is_frozen() -> None:
    from covid_audio_btp.hst_reporting import REPORTING_CONTRACT

    assert REPORTING_CONTRACT["bootstrap_replicates"] == 1000
    assert REPORTING_CONTRACT["bootstrap_seed"] == 42
    assert REPORTING_CONTRACT["ece_bins"] == 10
    assert REPORTING_CONTRACT["fixed_sensitivity"] == 0.90
    assert REPORTING_CONTRACT["decision_thresholds"] == [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]


def test_decision_curve_uses_standard_net_benefit_formula() -> None:
    from covid_audio_btp.hst_reporting import build_decision_curve

    predictions = _predictions("test", [0.1, 0.2, 0.8, 0.9])
    curve = build_decision_curve(predictions, thresholds=[0.5])
    assert curve.loc[0, "model_net_benefit"] == pytest.approx(0.5)
    assert curve.loc[0, "treat_none_net_benefit"] == 0.0


def test_repeated_holdout_rejects_participant_label_changes_across_folds() -> None:
    from covid_audio_btp.hst_reporting import repeated_holdout_cluster_ci

    predictions = pd.DataFrame(
        {
            "participant_key": ["coswara::p1", "coswara::p1", "coswara::p2", "coswara::p2"],
            "label_binary": ["negative", "positive", "positive", "positive"],
            "probability": [0.1, 0.9, 0.8, 0.7],
            "fold": [0, 1, 0, 1],
            "split": ["test"] * 4,
        }
    )
    with pytest.raises(ValueError, match="conflicting labels"):
        repeated_holdout_cluster_ci(
            predictions,
            metric="auroc",
            n_bootstrap=10,
            seed=42,
        )


def test_repeated_holdout_absolute_metric_does_not_test_against_zero() -> None:
    from covid_audio_btp.hst_reporting import repeated_holdout_cluster_ci

    predictions = pd.concat(
        [
            _predictions("source", [0.5, 0.5, 0.5, 0.5]).assign(fold=fold)
            for fold in (1, 2)
        ],
        ignore_index=True,
    )

    result = repeated_holdout_cluster_ci(
        predictions,
        metric="auroc",
        n_bootstrap=20,
        seed=42,
    )

    assert np.isnan(result["p_value"])
    assert result["hypothesis_test"] == "not_tested_absolute_metric"


def test_external_fold_bootstrap_requires_same_target_cohort_in_every_fold() -> None:
    from covid_audio_btp.hst_reporting import external_fold_cluster_bootstrap

    predictions = pd.DataFrame(
        {
            "participant_key": ["coughvid::p1", "coughvid::p2", "coughvid::p1"],
            "label_binary": ["negative", "positive", "negative"],
            "probability": [0.1, 0.8, 0.2],
            "fold": [0, 0, 1],
            "split": ["external_test"] * 3,
        }
    )
    with pytest.raises(ValueError, match="same target cohort"):
        external_fold_cluster_bootstrap(
            predictions,
            metric="auroc",
            n_bootstrap=10,
            seed=42,
        )


def test_paired_delta_rejects_different_scientific_contexts() -> None:
    from covid_audio_btp.hst_reporting import paired_model_cluster_delta

    left = _predictions("same", [0.1, 0.2, 0.8, 0.9]).assign(
        protocol="internal",
        modality="cough",
    )
    right = left.assign(
        probability=[0.2, 0.3, 0.7, 0.8],
        protocol="temporal",
    )
    with pytest.raises(ValueError, match="scientific context"):
        paired_model_cluster_delta(
            left,
            right,
            metric="auroc",
            n_bootstrap=10,
            seed=42,
        )


def test_paired_delta_reports_cluster_ci_without_invalid_bootstrap_p_value() -> None:
    from covid_audio_btp.hst_reporting import paired_model_cluster_delta

    left = _predictions("same", [0.1, 0.2, 0.8, 0.9])
    right = left.assign(probability=[0.2, 0.7, 0.3, 0.8])
    result = paired_model_cluster_delta(
        left,
        right,
        metric="auroc",
        n_bootstrap=30,
        seed=42,
    )
    assert np.isnan(result["p_value"])
    assert result["hypothesis_test"] == "not_tested_bootstrap_ci_only"


def test_split_policy_delta_handles_partially_overlapping_test_cohorts() -> None:
    from covid_audio_btp.hst_reporting import split_policy_delta

    left = _predictions("p", [0.1, 0.2, 0.8, 0.9])
    right = _predictions("p", [0.7, 0.8, 0.3, 0.4])
    right["participant_key"] = ["p::2", "p::3", "p::4", "p::5"]
    right["label_binary"] = ["positive", "positive", "negative", "negative"]

    result = split_policy_delta(
        (left, right),
        common_test=False,
        metric="auroc",
        n_bootstrap=20,
        seed=42,
    )

    assert result["bootstrap_design"] == "partially_paired_label_stratified_participants"
    assert result["overlap_participants"] == 2
    assert result["left_only_participants"] == 2
    assert result["right_only_participants"] == 2


def test_common_test_split_policy_allows_only_the_expected_protocol_difference() -> None:
    from covid_audio_btp.hst_reporting import split_policy_delta

    left = _predictions("same", [0.1, 0.2, 0.8, 0.9]).assign(
        dataset="coswara",
        split="test",
        protocol="common_late_calendar_mixed",
        modality="cough",
        cohort="common_late",
    )
    right = left.assign(
        probability=[0.2, 0.3, 0.7, 0.8],
        protocol="common_late_chronological",
    )

    result = split_policy_delta(
        (left, right),
        common_test=True,
        metric="auroc",
        n_bootstrap=20,
        seed=42,
    )

    assert result["paired"]
    assert result["bootstrap_design"] == "paired_participant"

    with pytest.raises(ValueError, match="scientific context"):
        split_policy_delta(
            (left, right.assign(cohort="different")),
            common_test=True,
            metric="auroc",
            n_bootstrap=20,
            seed=42,
        )


def test_paired_delong_accepts_only_one_exact_nonexternal_test_cohort() -> None:
    from covid_audio_btp.hst_reporting import paired_delong_auc_test

    left = _predictions("same", [0.1, 0.2, 0.8, 0.9]).drop(columns="fold").assign(
        dataset="coswara",
        protocol="track_a_internal",
        modality="cough",
        manifest_sha256="a" * 64,
    )
    right = left.assign(probability=[0.2, 0.7, 0.3, 0.8])
    result = paired_delong_auc_test(left, right)

    assert result["method"] == "paired_delong"
    assert result["n"] == 4
    assert result["delta"] == pytest.approx(result["left_auroc"] - result["right_auroc"])
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["ci_low"] <= result["delta"] <= result["ci_high"]

    with pytest.raises(ValueError, match="repeated folds|single exact"):
        paired_delong_auc_test(left.assign(fold=0), right.assign(fold=0))
    with pytest.raises(ValueError, match="external"):
        paired_delong_auc_test(
            left.assign(split="external_test", dataset="coughvid"),
            right.assign(split="external_test", dataset="coughvid"),
        )
    with pytest.raises(ValueError, match="identical participant keys"):
        paired_delong_auc_test(
            left,
            right.assign(participant_key=["other-0", "other-1", "other-2", "other-3"]),
        )


def test_source_platt_calibration_is_fold_local_and_never_uses_evaluation_labels() -> None:
    from covid_audio_btp.hst_reporting import apply_source_platt_calibration

    validation = pd.concat(
        [
            _predictions(
                f"validation-{fold}",
                [0.05, 0.35, 0.65, 0.95],
            ).assign(fold=fold, split="validation")
            for fold in (1, 2)
        ],
        ignore_index=True,
    )
    evaluation = pd.concat(
        [
            _predictions(
                f"evaluation-{fold}",
                [0.15, 0.45, 0.55, 0.85],
            ).assign(fold=fold, split="external_test")
            for fold in (1, 2)
        ],
        ignore_index=True,
    )

    calibrated_validation, calibrated_evaluation, audit = (
        apply_source_platt_calibration(validation, evaluation)
    )
    _, relabeled_evaluation, _ = apply_source_platt_calibration(
        validation,
        evaluation.assign(
            label_binary=evaluation["label_binary"].map(
                {"negative": "positive", "positive": "negative"}
            )
        ),
    )

    assert set(audit["fold"]) == {1, 2}
    assert not audit["skipped"].any()
    assert calibrated_validation["probability_scale"].eq(
        "source_validation_platt"
    ).all()
    assert calibrated_evaluation["probability_scale"].eq(
        "source_validation_platt"
    ).all()
    assert np.allclose(
        calibrated_evaluation["probability"],
        relabeled_evaluation["probability"],
    )
    assert np.allclose(
        calibrated_evaluation["raw_probability"],
        evaluation["probability"],
    )


def test_source_platt_calibration_fails_closed_to_raw_on_one_class_validation() -> None:
    from covid_audio_btp.hst_reporting import apply_source_platt_calibration

    validation = _predictions("validation", [0.1, 0.2, 0.3, 0.4]).assign(
        split="validation",
        label_binary="negative",
    )
    evaluation = _predictions("evaluation", [0.2, 0.4, 0.6, 0.8]).assign(
        split="test"
    )

    calibrated_validation, calibrated_evaluation, audit = (
        apply_source_platt_calibration(validation, evaluation)
    )

    assert audit.loc[0, "skipped"]
    assert audit.loc[0, "skip_reason"] == "one_class_source_validation"
    assert calibrated_validation["probability_scale"].eq("raw_fallback").all()
    assert calibrated_evaluation["probability_scale"].eq("raw_fallback").all()
    assert np.allclose(
        calibrated_evaluation["probability"],
        evaluation["probability"],
    )


def test_source_platt_calibration_rejects_missing_fold_identity() -> None:
    from covid_audio_btp.hst_reporting import apply_source_platt_calibration

    validation = _predictions("validation", [0.1, 0.2, 0.8, 0.9]).assign(
        fold=[1, 1, np.nan, np.nan],
        split="validation",
    )
    evaluation = _predictions("evaluation", [0.2, 0.3, 0.7, 0.8]).assign(
        fold=[1, 1, np.nan, np.nan],
        split="external_test",
    )

    with pytest.raises(ValueError, match="non-null folds"):
        apply_source_platt_calibration(validation, evaluation)
