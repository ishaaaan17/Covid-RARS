from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pandas as pd
import pytest


KEY_COLUMNS = [
    "run_id",
    "protocol",
    "fold",
    "dataset",
    "participant_key",
    "split",
]


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _recording_intersection_digest(frame: pd.DataFrame) -> str:
    modalities = set(frame["modality"].astype(str))
    contract = (
        frame.loc[frame["modality"].isin(("cough", "speech"))]
        if {"cough", "speech"}.issubset(modalities)
        else frame
    )
    records = (
        contract[["split", "recording_key", "audio_content_sha256", "modality"]]
        .drop_duplicates()
        .sort_values(
            ["split", "recording_key", "audio_content_sha256", "modality"],
            kind="stable",
        )
        .to_dict(orient="records")
    )
    payload = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _refresh_recording_intersection(frame: pd.DataFrame) -> pd.DataFrame:
    refreshed = frame.copy()
    for _context, index in refreshed.groupby(
        ["run_id", "protocol", "fold", "dataset"],
        sort=False,
    ).groups.items():
        selected = refreshed.loc[index]
        refreshed.loc[index, "recording_intersection_sha256"] = (
            _recording_intersection_digest(selected)
        )
    return refreshed


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _authenticated_binding(
    hst_predictions: pd.DataFrame,
    comparator_predictions: pd.DataFrame,
):
    from covid_rars.hst_fusion import (
        AuthenticatedFusionBinding,
        BRANCH_IDENTITY_COLUMNS,
        _prediction_artifact_hash,
        _validate_predictions,
    )

    hst = _validate_predictions(hst_predictions, name="trusted test HST predictions")
    comparator = _validate_predictions(
        comparator_predictions,
        name="trusted test comparator predictions",
    )
    contexts: list[dict[str, object]] = []
    context_columns = ["run_id", "protocol", "fold", "dataset"]
    context_values = hst[context_columns].drop_duplicates().sort_values(context_columns)
    for context_key in context_values.itertuples(index=False, name=None):
        hst_context = hst.loc[
            np.logical_and.reduce(
                [hst[column].eq(value) for column, value in zip(context_columns, context_key)]
            )
            & hst["modality"].isin(("cough", "speech"))
        ].copy()
        comparator_context = comparator.loc[
            np.logical_and.reduce(
                [
                    comparator[column].eq(value)
                    for column, value in zip(context_columns, context_key)
                ]
            )
            & comparator["modality"].isin(("cough", "speech"))
        ].copy()

        def branch_receipts(frame: pd.DataFrame) -> dict[str, object]:
            return {
                modality: {
                    **{
                        column: frame.loc[frame["modality"].eq(modality), column].iloc[0]
                        for column in BRANCH_IDENTITY_COLUMNS
                    },
                    "branch_provenance_hash": frame.loc[
                        frame["modality"].eq(modality), "branch_provenance_hash"
                    ].iloc[0],
                }
                for modality in ("cough", "speech")
            }

        context = dict(zip(context_columns, context_key))
        manifest_sha256 = hst_context["manifest_sha256"].iloc[0]
        intersection_sha256 = hst_context["recording_intersection_sha256"].iloc[0]
        contexts.append(
            {
                **context,
                "manifest_receipt": {
                    "receipt_id": f"manifest:{context_key}",
                    "receipt_sha256": _digest(f"trusted-manifest-receipt:{context_key}"),
                    "manifest_sha256": manifest_sha256,
                    "recording_intersection_sha256": intersection_sha256,
                },
                "hst": {
                    "prediction_artifact_sha256": _prediction_artifact_hash(hst_context),
                    "branches": branch_receipts(hst_context),
                },
                "comparator": {
                    "generation_id": f"approved-comparator-generation:{context_key}",
                    "generation_receipt_sha256": _digest(
                        f"trusted-comparator-generation:{context_key}"
                    ),
                    "prediction_artifact_sha256": _prediction_artifact_hash(
                        comparator_context
                    ),
                    "branches": branch_receipts(comparator_context),
                },
            }
        )
    receipt = {
        "schema_version": 1,
        "receipt_type": "hst_fusion_authenticated_registry",
        "registry_authority": "unit-test-trusted-registry",
        "receipt_id": "unit-test-hst-fusion-binding-v1",
        "contexts": contexts,
    }
    return AuthenticatedFusionBinding.from_registry_receipt(
        receipt,
        trusted_receipt_sha256=_canonical_digest(receipt),
    )


def _run_confirmatory_bank(
    hst_predictions: pd.DataFrame,
    comparator_predictions: pd.DataFrame,
):
    from covid_rars.hst_fusion import run_hst_fusion_bank

    return run_hst_fusion_bank(
        hst_predictions,
        comparator_predictions,
        analysis_mode="confirmatory",
        authenticated_binding=_authenticated_binding(
            hst_predictions,
            comparator_predictions,
        ),
    )


def _branch_predictions(
    source_family: str,
    *,
    folds: tuple[int, ...] = (0,),
    modalities: tuple[str, ...] = ("cough", "speech"),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    family_offset = 0.02 if source_family == "hst" else -0.02
    modality_offset = {"cough": 0.08, "breath": -0.04, "speech": 0.0}
    for fold in folds:
        for split, count in (("validation", 12), ("test", 6)):
            for index in range(count):
                label = "positive" if index % 2 else "negative"
                base = 0.72 if label == "positive" else 0.28
                participant_key = f"coswara::{split}_f{fold}_p{index:02d}"
                for modality in modalities:
                    recording_key = f"{participant_key}::{modality}::r00"
                    probability = float(
                        np.clip(
                            base + family_offset + modality_offset[modality] + fold * 0.001,
                            0.001,
                            0.999,
                        )
                    )
                    rows.append(
                        {
                            "run_id": "run-1",
                            "protocol": "repeated_holdout",
                            "fold": fold,
                            "dataset": "coswara",
                            "participant_key": participant_key,
                            "recording_key": recording_key,
                            "audio_content_sha256": _digest(
                                f"{participant_key}:{modality}:audio"
                            ),
                            "split": split,
                            "modality": modality,
                            "label_binary": label,
                            "probability": probability,
                            "eligible": True,
                            "manifest_sha256": _digest(
                                f"run-1:repeated_holdout:{fold}:coswara:manifest"
                            ),
                            "recording_intersection_sha256": "pending",
                            "feature_artifact_sha256": _digest(
                                f"{source_family}:{fold}:{modality}:feature-artifact"
                            ),
                            "feature_approval_id": (
                                f"approved:{source_family}:{modality}:v1"
                            ),
                            "preprocessing_sha256": _digest(
                                f"{source_family}:{modality}:preprocessing"
                            ),
                            "source_family": source_family,
                            "model": f"{source_family}_{modality}",
                            "checkpoint_hash": _digest(
                                f"{source_family}:{fold}:{modality}:checkpoint"
                            ),
                            "representation": source_family,
                        }
                    )
    return _refresh_recording_intersection(pd.DataFrame(rows))


def test_legacy_auprc_weights_use_validation_only_and_are_normalized() -> None:
    from covid_rars.hst_fusion import legacy_validation_auprc_weights

    metrics = pd.DataFrame(
        {
            "run_id": ["run-1"] * 6,
            "protocol": ["repeated_holdout"] * 6,
            "fold": [0] * 6,
            "dataset": ["coswara"] * 6,
            "split": ["validation"] * 3 + ["test"] * 3,
            "modality": ["cough", "breath", "speech"] * 2,
            "auprc": [0.80, 0.70, 0.50, 0.01, 0.99, 1.00],
        }
    )

    weights = legacy_validation_auprc_weights(metrics)
    changed_test = metrics.copy()
    changed_test.loc[changed_test["split"].eq("test"), "auprc"] = [1.0, 0.0, 0.0]

    assert weights == legacy_validation_auprc_weights(changed_test)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["cough"] > weights["breath"] > weights["speech"]
    assert weights["speech"] > 0.0


def test_legacy_auprc_weights_reject_missing_validation_or_cross_fold_pooling() -> None:
    from covid_rars.hst_fusion import legacy_validation_auprc_weights

    only_test = pd.DataFrame(
        {
            "run_id": ["run-1"],
            "protocol": ["repeated_holdout"],
            "fold": [0],
            "dataset": ["coswara"],
            "split": ["test"],
            "modality": ["cough"],
            "auprc": [0.99],
        }
    )
    pooled = pd.DataFrame(
        {
            "run_id": ["run-1", "run-1"],
            "protocol": ["repeated_holdout", "repeated_holdout"],
            "fold": [0, 1],
            "dataset": ["coswara", "coswara"],
            "split": ["validation", "validation"],
            "modality": ["cough", "cough"],
            "auprc": [0.8, 0.7],
        }
    )

    with pytest.raises(ValueError, match="validation"):
        legacy_validation_auprc_weights(only_test)
    with pytest.raises(ValueError, match="fold"):
        legacy_validation_auprc_weights(pooled)


def test_primary_uniform_fusion_is_complete_case() -> None:
    from covid_rars.hst_fusion import fuse_uniform_complete_case

    predictions = _branch_predictions(
        "hst", modalities=("cough", "breath", "speech")
    )
    missing_key = "coswara::validation_f0_p00"
    predictions = predictions.loc[
        ~(
            predictions["participant_key"].eq(missing_key)
            & predictions["modality"].eq("breath")
        )
    ].copy()
    predictions = _refresh_recording_intersection(predictions)

    fused = fuse_uniform_complete_case(predictions)

    assert missing_key not in set(fused["participant_key"])
    assert fused["complete_case"].all()
    assert fused["n_modalities"].eq(3).all()
    assert fused["available_modalities"].eq("breath,cough,speech").all()
    row = fused.loc[fused["participant_key"].eq("coswara::test_f0_p01")].iloc[0]
    source = predictions.loc[
        predictions["participant_key"].eq("coswara::test_f0_p01"), "probability"
    ]
    assert row["probability"] == pytest.approx(source.mean())


def test_fixed_weights_are_complete_case_while_available_modality_is_sensitivity() -> None:
    from covid_rars.hst_fusion import (
        _validate_predictions,
        fuse_available_modalities_sensitivity,
        fuse_with_fixed_weights,
        legacy_validation_auprc_weights,
    )

    predictions = _branch_predictions("hst")
    missing_key = "coswara::test_f0_p00"
    predictions = predictions.loc[
        ~(
            predictions["participant_key"].eq(missing_key)
            & predictions["modality"].eq("speech")
        )
    ].copy()
    predictions = _refresh_recording_intersection(predictions)
    validated = _validate_predictions(predictions, name="fixed-weight test predictions")
    branch_hashes = validated.groupby("modality")[
        "branch_provenance_hash"
    ].first()
    weights = legacy_validation_auprc_weights(
        pd.DataFrame(
            {
                "run_id": ["run-1", "run-1"],
                "protocol": ["repeated_holdout", "repeated_holdout"],
                "fold": [0, 0],
                "dataset": ["coswara", "coswara"],
                "split": ["validation", "validation"],
                "modality": ["cough", "speech"],
                "auprc": [0.80, 0.60],
                "branch_provenance_hash": [
                    branch_hashes["cough"],
                    branch_hashes["speech"],
                ],
            }
        )
    )

    fixed = fuse_with_fixed_weights(predictions, weights)
    available = fuse_available_modalities_sensitivity(predictions, weights)

    with pytest.raises(ValueError, match="validation-derived"):
        fuse_with_fixed_weights(predictions, {"cough": 0.75, "speech": 0.25})

    assert missing_key not in set(fixed["participant_key"])
    retained = available.loc[available["participant_key"].eq(missing_key)].iloc[0]
    cough = predictions.loc[
        predictions["participant_key"].eq(missing_key), "probability"
    ].iloc[0]
    assert retained["probability"] == pytest.approx(cough)
    assert retained["complete_case"] == False  # noqa: E712
    assert retained["available_modalities"] == "cough"


def test_logistic_stacker_is_balanced_l2_validation_only_and_fold_frozen() -> None:
    from covid_rars.hst_fusion import (
        apply_validation_logistic_stacker,
        fit_validation_logistic_stacker,
    )

    predictions = _branch_predictions("hst")
    validation = predictions.loc[predictions["split"].eq("validation")].copy()
    test = predictions.loc[predictions["split"].eq("test")].copy()

    stacker = fit_validation_logistic_stacker(validation, random_state=42)
    applied = apply_validation_logistic_stacker(stacker, test)

    assert stacker.source_split == "validation"
    assert stacker.fold == 0
    assert stacker.feature_names == ("cough", "speech")
    assert stacker.estimator.C == 1.0
    assert stacker.estimator.class_weight == "balanced"
    assert stacker.estimator.l1_ratio == 0.0
    assert applied["fusion_method"].eq("stacked_logistic_validation").all()
    assert applied["probability"].between(0.0, 1.0).all()
    expected_matrix = test.pivot(
        index=KEY_COLUMNS + ["label_binary"],
        columns="modality",
        values="probability",
    ).reset_index()
    expected_matrix.columns.name = None
    expected_matrix = expected_matrix.sort_values(KEY_COLUMNS, kind="stable")
    expected = stacker.estimator.predict_proba(
        expected_matrix[list(stacker.feature_names)].to_numpy(dtype=float)
    )[:, 1]
    np.testing.assert_allclose(applied["probability"], expected, rtol=0.0, atol=1e-15)

    with pytest.raises(ValueError, match="validation"):
        fit_validation_logistic_stacker(predictions, random_state=42)
    with pytest.raises(ValueError, match="fold"):
        apply_validation_logistic_stacker(
            stacker,
            _branch_predictions("hst", folds=(1,)).query("split == 'test'"),
        )


def test_four_branch_hybrid_has_exact_schema_and_uniform_quarter_weights() -> None:
    from covid_rars.hst_fusion import (
        FOUR_BRANCH_COLUMNS,
        _hybrid_to_long,
        build_four_branch_hybrid_inputs,
        fuse_uniform_complete_case,
    )

    hst = _branch_predictions("hst")
    comparator = _branch_predictions("comparator")
    hybrid = build_four_branch_hybrid_inputs(hst, comparator)
    long = _hybrid_to_long(hybrid)

    assert list(hybrid.columns[-4:]) == list(FOUR_BRANCH_COLUMNS)
    fused = fuse_uniform_complete_case(long)
    row = hybrid.iloc[0]
    fused_row = fused.loc[
        fused["participant_key"].eq(row["participant_key"])
        & fused["split"].eq(row["split"])
    ].iloc[0]
    assert fused_row["probability"] == pytest.approx(
        sum(float(row[column]) * 0.25 for column in FOUR_BRANCH_COLUMNS)
    )


def test_four_branch_hybrid_rejects_cohort_or_fold_mismatch() -> None:
    from covid_rars.hst_fusion import build_four_branch_hybrid_inputs

    hst = _branch_predictions("hst")
    comparator = _branch_predictions("comparator")
    comparator = comparator.loc[
        ~comparator["participant_key"].eq("coswara::test_f0_p00")
    ].copy()
    comparator = _refresh_recording_intersection(comparator)
    with pytest.raises(ValueError, match="cohort|recording.*intersection"):
        build_four_branch_hybrid_inputs(hst, comparator)

    comparator = _branch_predictions("comparator")
    comparator["fold"] = 1
    with pytest.raises(ValueError, match="fold|cohort"):
        build_four_branch_hybrid_inputs(hst, comparator)


def test_fusion_bank_saves_fold_local_weights_and_all_prespecified_methods(
    tmp_path: Path,
) -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    hst = _branch_predictions("hst", folds=(0, 1))
    comparator = _branch_predictions("comparator", folds=(0, 1))

    result = _run_confirmatory_bank(hst, comparator)

    expected_methods = {
        "uniform_mean",
        "legacy_validation_weighted_auprc",
        "stacked_logistic_validation",
        "available_modalities_validation_weighted_auprc",
        "hybrid_uniform_four_branch",
        "hybrid_legacy_validation_weighted_auprc",
        "hybrid_stacked_logistic_validation",
    }
    assert expected_methods.issubset(set(result.predictions["fusion_method"]))
    assert set(result.weights["source_split"]) == {"prespecified", "validation"}
    assert set(result.weights["fold"]) == {0, 1}
    assert result.weights.groupby(
        ["fold", "source_family", "modality_combination", "fusion_method"]
    )["normalized_weight"].sum().eq(1.0).all()
    assert set(result.stacker_parameters["source_split"]) == {"validation"}
    assert set(result.metrics["split"]) == {"validation", "test"}
    assert not result.complete_case_counts.empty
    assert set(result.paired_deltas["reference_family"]) == {"hst", "comparator"}
    assert set(result.paired_deltas["metric"]) == {"auroc", "auprc"}

    weight_path = tmp_path / "hst_fusion_weights.csv"
    stacker_path = tmp_path / "hst_fusion_stacker_parameters.csv"
    result.save_weights(weight_path)
    result.save_stacker_parameters(stacker_path)
    saved_weights = pd.read_csv(weight_path)
    saved_stackers = pd.read_csv(stacker_path)
    pd.testing.assert_frame_equal(saved_weights, result.weights, check_dtype=False)
    pd.testing.assert_frame_equal(
        saved_stackers, result.stacker_parameters, check_dtype=False
    )


def test_fusion_bank_rejects_duplicate_branch_rows_instead_of_pooling() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    hst = _branch_predictions("hst")
    duplicate = pd.concat([hst, hst.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        run_hst_fusion_bank(duplicate, analysis_mode="exploratory")


def test_fusion_bank_labels_cough_breath_as_secondary_hst_sensitivity() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    hst = _branch_predictions("hst", modalities=("cough", "breath", "speech"))
    comparator = _branch_predictions("comparator")

    result = _run_confirmatory_bank(hst, comparator)
    cough_breath = result.predictions.loc[
        result.predictions["source_family"].eq("hst")
        & result.predictions["modality_combination"].eq("cough+breath")
    ]

    assert not cough_breath.empty
    assert cough_breath.loc[
        cough_breath["split"].eq("validation"), "analysis_role"
    ].eq("secondary").all()
    assert cough_breath.loc[
        cough_breath["split"].eq("validation"), "analysis_scope"
    ].eq("selection").all()
    assert cough_breath.loc[cough_breath["split"].eq("test"), "analysis_role"].eq(
        "sensitivity"
    ).all()
    assert {
        "uniform_mean",
        "legacy_validation_weighted_auprc",
        "stacked_logistic_validation",
        "available_modalities_validation_weighted_auprc",
    }.issubset(set(cough_breath["fusion_method"]))


def test_fusion_bank_rejects_participant_overlap_between_validation_and_test() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    hst = _branch_predictions("hst")
    overlap = hst["participant_key"].eq("coswara::test_f0_p00")
    hst.loc[overlap, "participant_key"] = "coswara::validation_f0_p00"

    with pytest.raises(ValueError, match="overlap.*validation.*test|validation.*test.*overlap"):
        run_hst_fusion_bank(hst, analysis_mode="exploratory")


def test_fusion_rejects_recording_key_reused_across_validation_and_test() -> None:
    from covid_rars.hst_fusion import fuse_uniform_complete_case

    predictions = _branch_predictions("hst")
    validation = predictions.loc[
        predictions["split"].eq("validation")
        & predictions["modality"].eq("cough")
    ].iloc[0]
    test_index = predictions.loc[
        predictions["split"].eq("test")
        & predictions["modality"].eq("cough")
    ].index[0]
    predictions.loc[test_index, "recording_key"] = validation["recording_key"]
    predictions = _refresh_recording_intersection(predictions)

    with pytest.raises(ValueError, match="recording_key.*validation.*test|validation.*test.*recording_key"):
        fuse_uniform_complete_case(predictions)


def test_fusion_rejects_audio_content_reused_under_different_recording_keys() -> None:
    from covid_rars.hst_fusion import fuse_uniform_complete_case

    predictions = _branch_predictions("hst")
    validation = predictions.loc[
        predictions["split"].eq("validation")
        & predictions["modality"].eq("speech")
    ].iloc[0]
    test_index = predictions.loc[
        predictions["split"].eq("test")
        & predictions["modality"].eq("speech")
    ].index[0]
    predictions.loc[test_index, "audio_content_sha256"] = validation[
        "audio_content_sha256"
    ]

    with pytest.raises(ValueError, match="audio.*content.*validation.*test|validation.*test.*audio.*content"):
        fuse_uniform_complete_case(predictions)


def test_validation_rules_are_bound_to_the_exact_full_context() -> None:
    from covid_rars.hst_fusion import (
        apply_validation_logistic_stacker,
        fit_validation_logistic_stacker,
        fuse_with_fixed_weights,
        legacy_validation_auprc_weights,
    )

    predictions = _branch_predictions("hst")
    validation = predictions.loc[predictions["split"].eq("validation")].copy()
    test = predictions.loc[predictions["split"].eq("test")].copy()
    metrics = pd.DataFrame(
        {
            "run_id": ["run-1", "run-1"],
            "protocol": ["repeated_holdout", "repeated_holdout"],
            "fold": [0, 0],
            "dataset": ["coswara", "coswara"],
            "split": ["validation", "validation"],
            "modality": ["cough", "speech"],
            "auprc": [0.8, 0.7],
        }
    )
    weights = legacy_validation_auprc_weights(metrics)
    stacker = fit_validation_logistic_stacker(validation, random_state=42)

    assert (weights.run_id, weights.protocol, weights.fold, weights.dataset) == (
        "run-1",
        "repeated_holdout",
        0,
        "coswara",
    )
    changed = test.copy()
    changed["run_id"] = "run-2"
    with pytest.raises(ValueError, match="run_id"):
        fuse_with_fixed_weights(changed, weights)

    changed = test.copy()
    changed["dataset"] = "coughvid"
    with pytest.raises(ValueError, match="dataset"):
        apply_validation_logistic_stacker(stacker, changed)


def test_fusion_bank_rejects_unequal_primary_complete_case_cohorts() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    hst = _branch_predictions("hst")
    comparator = _branch_predictions("comparator")
    comparator = comparator.loc[
        ~comparator["participant_key"].eq("coswara::test_f0_p00")
    ].copy()
    comparator = _refresh_recording_intersection(comparator)

    with pytest.raises(
        ValueError,
        match="primary.*cohort|cohort.*primary|recording.*intersection",
    ):
        run_hst_fusion_bank(hst, comparator, analysis_mode="exploratory")


def test_primary_comparison_rejects_recording_mismatch_before_participant_aggregation() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    hst = _branch_predictions("hst")
    comparator = _branch_predictions("comparator")
    changed = comparator.index[comparator["split"].eq("test")][0]
    comparator.loc[changed, "recording_key"] += "-substituted"
    comparator = _refresh_recording_intersection(comparator)

    with pytest.raises(ValueError, match="recording.*intersection|intersection.*recording"):
        run_hst_fusion_bank(hst, comparator, analysis_mode="exploratory")


def test_primary_comparison_aggregates_only_after_exact_recording_alignment() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    hst = _branch_predictions("hst")
    comparator = _branch_predictions("comparator")
    participant = "coswara::test_f0_p01"
    for name, frame in (("hst", hst), ("comparator", comparator)):
        duplicate = frame.loc[frame["participant_key"].eq(participant)].copy()
        duplicate["recording_key"] = duplicate["recording_key"].str.replace(
            "::r00", "::r01", regex=False
        )
        duplicate["probability"] = np.clip(
            duplicate["probability"].to_numpy(dtype=float) - 0.20,
            0.0,
            1.0,
        )
        if name == "hst":
            hst = _refresh_recording_intersection(
                pd.concat([frame, duplicate], ignore_index=True)
            )
        else:
            comparator = _refresh_recording_intersection(
                pd.concat([frame, duplicate], ignore_index=True)
            )

    result = _run_confirmatory_bank(hst, comparator)
    primary_hst = result.predictions.loc[
        result.predictions["source_family"].eq("hst")
        & result.predictions["fusion_method"].eq("uniform_mean")
        & result.predictions["split"].eq("test")
        & result.predictions["participant_key"].eq(participant)
    ]

    assert len(primary_hst) == 1
    expected = hst.loc[
        hst["participant_key"].eq(participant), "probability"
    ].mean()
    assert primary_hst["probability"].iloc[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("column", "mutation"),
    [
        ("manifest_sha256", "sha256"),
        ("recording_intersection_sha256", "sha256"),
        ("feature_artifact_sha256", "sha256"),
        ("feature_approval_id", "string"),
        ("preprocessing_sha256", "sha256"),
    ],
)
def test_primary_inputs_reject_conflicting_upstream_identity(
    column: str,
    mutation: str,
) -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    hst = _branch_predictions("hst")
    changed = hst["split"].eq("test") & hst["modality"].eq("cough")
    hst.loc[changed, column] = (
        _digest(f"substituted:{column}") if mutation == "sha256" else "substituted-approval"
    )

    with pytest.raises(ValueError, match="identity|manifest|intersection|provenance"):
        run_hst_fusion_bank(
            hst,
            _branch_predictions("comparator"),
            analysis_mode="exploratory",
        )


def test_primary_inputs_reject_ineligible_recording_predictions() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    hst = _branch_predictions("hst")
    hst.loc[hst.index[0], "eligible"] = False

    with pytest.raises(ValueError, match="eligible|eligibility"):
        run_hst_fusion_bank(
            hst,
            _branch_predictions("comparator"),
            analysis_mode="exploratory",
        )


def test_hybrid_is_secondary_and_only_aligned_uniform_pair_is_primary() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    result = _run_confirmatory_bank(
        _branch_predictions("hst"),
        _branch_predictions("comparator"),
    )
    hybrid = result.predictions.loc[result.predictions["source_family"].eq("hybrid")]
    roles = hybrid.groupby("fusion_method")["analysis_role"].unique().to_dict()
    primary = result.predictions.loc[result.predictions["analysis_role"].eq("primary")]

    assert roles["hybrid_uniform_four_branch"].tolist() == ["secondary"]
    assert roles["hybrid_legacy_validation_weighted_auprc"].tolist() == ["secondary"]
    assert roles["hybrid_stacked_logistic_validation"].tolist() == ["secondary"]
    assert set(primary["source_family"]) == {"hst", "comparator"}
    assert primary["fusion_method"].eq("uniform_mean").all()
    assert primary["modality_combination"].eq("cough+speech").all()
    assert primary["analysis_scope"].eq("confirmatory").all()
    assert primary["estimand_id"].eq(
        "primary_hst_vs_comparator_uniform_cough_speech_auroc"
    ).all()
    assert primary["comparison_binding_hash"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert primary.groupby(KEY_COLUMNS)["comparison_binding_hash"].nunique().eq(1).all()


def test_validation_weight_raw_values_and_provenance_are_immutable() -> None:
    from covid_rars.hst_fusion import legacy_validation_auprc_weights

    metrics = pd.DataFrame(
        {
            "run_id": ["run-1", "run-1"],
            "protocol": ["repeated_holdout", "repeated_holdout"],
            "fold": [0, 0],
            "dataset": ["coswara", "coswara"],
            "split": ["validation", "validation"],
            "modality": ["cough", "speech"],
            "auprc": [0.8, 0.7],
        }
    )
    weights = legacy_validation_auprc_weights(metrics)

    with pytest.raises(TypeError, match="immutable"):
        weights.raw_weights["cough"] = 999.0
    with pytest.raises((AttributeError, TypeError)):
        weights.run_id = "tampered"


def test_weight_export_detects_result_or_existing_artifact_tampering(tmp_path: Path) -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    result = _run_confirmatory_bank(
        _branch_predictions("hst"),
        _branch_predictions("comparator"),
    )
    result.weights.loc[0, "normalized_weight"] = 999.0
    with pytest.raises(ValueError, match="tamper|hash|mutat"):
        result.save_weights(tmp_path / "mutated.csv")

    clean = _run_confirmatory_bank(
        _branch_predictions("hst"),
        _branch_predictions("comparator"),
    )
    output = tmp_path / "weights.csv"
    clean.save_weights(output)
    clean.save_weights(output)
    output.write_text("different bytes\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="different|overwrite"):
        clean.save_weights(output)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("fold", "0", "fold.*integer"),
        ("fold", True, "fold.*integer"),
        ("run_id", 1, "run_id.*string"),
        ("protocol", 7, "protocol.*string"),
        ("dataset", 9, "dataset.*string"),
    ],
)
def test_fusion_context_requires_canonical_types(
    column: str,
    value: object,
    message: str,
) -> None:
    from covid_rars.hst_fusion import fuse_uniform_complete_case

    predictions = _branch_predictions("hst")
    predictions[column] = value

    with pytest.raises((TypeError, ValueError), match=message):
        fuse_uniform_complete_case(predictions)


def test_hst_primary_contract_is_only_cough_speech_and_cough_breath_is_explicit_sensitivity() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    result = _run_confirmatory_bank(
        _branch_predictions("hst", modalities=("cough", "breath", "speech")),
        _branch_predictions("comparator"),
    )
    hst_uniform = result.predictions.loc[
        result.predictions["source_family"].eq("hst")
        & result.predictions["fusion_method"].eq("uniform_mean")
    ]
    primary = hst_uniform.loc[hst_uniform["analysis_role"].eq("primary")]
    cough_breath = hst_uniform.loc[
        hst_uniform["modality_combination"].eq("cough+breath")
    ]

    assert set(primary["modality_combination"]) == {"cough+speech"}
    assert primary["analysis_scope"].eq("confirmatory").all()
    assert cough_breath.loc[
        cough_breath["split"].eq("validation"), "analysis_role"
    ].eq("secondary").all()
    assert cough_breath.loc[
        cough_breath["split"].eq("validation"), "analysis_scope"
    ].eq("selection").all()
    assert cough_breath.loc[cough_breath["split"].eq("test"), "analysis_role"].eq(
        "sensitivity"
    ).all()
    assert cough_breath.loc[cough_breath["split"].eq("test"), "analysis_scope"].eq(
        "sensitivity"
    ).all()


def test_exported_uniform_weights_are_prespecified_not_validation_derived() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    result = _run_confirmatory_bank(
        _branch_predictions("hst"),
        _branch_predictions("comparator"),
    )
    family_uniform = result.weights.loc[
        result.weights["fusion_method"].eq("uniform_mean")
    ]
    hybrid_uniform = result.weights.loc[
        result.weights["fusion_method"].eq("hybrid_uniform_four_branch")
    ]
    validation_weighted = result.weights.loc[
        result.weights["fusion_method"].isin(
            [
                "legacy_validation_weighted_auprc",
                "hybrid_legacy_validation_weighted_auprc",
            ]
        )
    ]

    assert family_uniform["source_split"].eq("prespecified").all()
    assert family_uniform["normalized_weight"].eq(0.5).all()
    assert hybrid_uniform["source_split"].eq("prespecified").all()
    assert hybrid_uniform["normalized_weight"].eq(0.25).all()
    assert validation_weighted["source_split"].eq("validation").all()
    assert result.stacker_parameters["source_split"].eq("validation").all()


@pytest.mark.parametrize(
    "mutated_field",
    [
        "coef",
        "coef_dtype",
        "intercept",
        "classes",
        "classes_dtype",
        "n_features",
    ],
)
def test_logistic_stacker_rejects_mutated_fitted_state_before_apply_or_export(
    mutated_field: str,
) -> None:
    from covid_rars.hst_fusion import (
        _stacker_parameter_rows,
        apply_validation_logistic_stacker,
        fit_validation_logistic_stacker,
    )

    predictions = _branch_predictions("hst")
    validation = predictions.loc[predictions["split"].eq("validation")].copy()
    test = predictions.loc[predictions["split"].eq("test")].copy()
    stacker = fit_validation_logistic_stacker(validation, random_state=42)
    if mutated_field == "coef":
        stacker.estimator.coef_[0, 0] += 1.0
    elif mutated_field == "coef_dtype":
        stacker.estimator.coef_ = stacker.estimator.coef_.astype(np.float32)
    elif mutated_field == "intercept":
        stacker.estimator.intercept_[0] += 1.0
    elif mutated_field == "classes":
        stacker.estimator.classes_[0] = 7
    elif mutated_field == "classes_dtype":
        stacker.estimator.classes_ = stacker.estimator.classes_.astype(float)
    else:
        stacker.estimator.n_features_in_ += 1

    with pytest.raises(ValueError, match="stacker.*tamper|tamper.*stacker"):
        apply_validation_logistic_stacker(stacker, test)
    with pytest.raises(ValueError, match="stacker.*tamper|tamper.*stacker"):
        _stacker_parameter_rows(
            stacker,
            source_family="hst",
            modality_combination="cough+speech",
            fusion_method="stacked_logistic_validation",
        )


def test_participant_label_is_invariant_across_folds_and_splits() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    hst = _branch_predictions("hst", folds=(0, 1))
    contradictory = (
        hst["fold"].eq(1)
        & hst["split"].eq("validation")
        & hst["participant_key"].eq("coswara::validation_f1_p00")
    )
    hst.loc[contradictory, "participant_key"] = "coswara::validation_f0_p00"
    hst.loc[contradictory, "label_binary"] = "positive"

    with pytest.raises(ValueError, match="participant.*label|label.*participant"):
        run_hst_fusion_bank(hst, analysis_mode="exploratory")


def test_legacy_auprc_weight_normalization_is_finite_and_overflow_safe() -> None:
    from covid_rars.hst_fusion import legacy_validation_auprc_weights

    metrics = pd.DataFrame(
        {
            "run_id": ["run-1", "run-1"],
            "protocol": ["repeated_holdout", "repeated_holdout"],
            "fold": [0, 0],
            "dataset": ["coswara", "coswara"],
            "split": ["validation", "validation"],
            "modality": ["cough", "speech"],
            "auprc": [0.8, 0.7],
        }
    )

    stable = legacy_validation_auprc_weights(metrics, floor=1e308)
    stable_values = np.asarray(list(stable.values()), dtype=float)
    assert np.isfinite(stable_values).all()
    assert stable_values.sum() == pytest.approx(1.0)

    with pytest.raises(ValueError, match="nonfinite|overflow|finite"):
        legacy_validation_auprc_weights(
            metrics,
            reference=-np.inf,
        )


def test_available_modality_predictions_export_their_validation_weight_provenance() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    result = _run_confirmatory_bank(
        _branch_predictions("hst"),
        _branch_predictions("comparator"),
    )
    available = result.weights.loc[
        result.weights["fusion_method"].eq(
            "available_modalities_validation_weighted_auprc"
        )
    ]

    assert not available.empty
    assert available["source_split"].eq("validation").all()
    assert available["branch_provenance_hash"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert available.groupby(
        ["run_id", "protocol", "fold", "dataset", "source_family"]
    )["normalized_weight"].sum().eq(1.0).all()


def test_validation_weight_map_is_an_immutable_mapping_not_a_dict_subclass() -> None:
    from covid_rars.hst_fusion import legacy_validation_auprc_weights

    metrics = pd.DataFrame(
        {
            "run_id": ["run-1", "run-1"],
            "protocol": ["repeated_holdout", "repeated_holdout"],
            "fold": [0, 0],
            "dataset": ["coswara", "coswara"],
            "split": ["validation", "validation"],
            "modality": ["cough", "speech"],
            "auprc": [0.8, 0.7],
        }
    )
    weights = legacy_validation_auprc_weights(metrics)

    assert isinstance(weights, Mapping)
    assert not isinstance(weights, dict)
    with pytest.raises(TypeError):
        weights["cough"] = 1.0
    with pytest.raises(TypeError):
        weights.raw_weights["cough"] = 1.0


def test_empty_paired_delta_table_has_a_stable_export_schema() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    result = run_hst_fusion_bank(
        _branch_predictions("hst"),
        analysis_mode="exploratory",
    )

    assert result.paired_deltas.empty
    assert list(result.paired_deltas.columns) == [
        "run_id",
        "protocol",
        "fold",
        "dataset",
        "split",
        "candidate_family",
        "reference_family",
        "metric",
        "hybrid_value",
        "reference_value",
        "delta",
        "paired_participants",
        "comparison_binding_hash",
        "authenticated_registry_receipt_sha256",
        "authenticated_context_binding_sha256",
        "analysis_scope",
        "analysis_role",
        "estimand_id",
        "multiplicity_family",
    ]


@pytest.mark.parametrize("invalid_label", [True, False, 0, 1, 0.0, 1.0])
def test_prediction_labels_require_canonical_binary_strings(
    invalid_label: object,
) -> None:
    from covid_rars.hst_fusion import fuse_uniform_complete_case

    predictions = _branch_predictions("hst")
    predictions["label_binary"] = predictions["label_binary"].astype(object)
    participant = predictions.loc[0, "participant_key"]
    predictions.loc[
        predictions["participant_key"].eq(participant), "label_binary"
    ] = invalid_label

    with pytest.raises((TypeError, ValueError), match="label.*canonical|canonical.*label"):
        fuse_uniform_complete_case(predictions)


def test_validation_weight_function_has_its_concrete_return_annotation() -> None:
    from covid_rars.hst_fusion import (
        ValidationWeightMap,
        legacy_validation_auprc_weights,
    )

    assert (
        get_type_hints(legacy_validation_auprc_weights)["return"]
        is ValidationWeightMap
    )


@pytest.mark.parametrize("identity_column", ["model", "checkpoint_hash", "representation"])
def test_branch_identity_cannot_change_between_validation_and_test(
    identity_column: str,
) -> None:
    from covid_rars.hst_fusion import fuse_uniform_complete_case

    predictions = _branch_predictions("hst")
    predictions.loc[predictions["split"].eq("test"), identity_column] = (
        f"substituted-{identity_column}"
    )

    with pytest.raises(ValueError, match="branch.*identity|identity.*branch"):
        fuse_uniform_complete_case(predictions)


def test_validation_weights_reject_test_only_branch_substitution() -> None:
    from covid_rars.hst_fusion import (
        _branch_validation_metrics,
        _validate_predictions,
        fuse_with_fixed_weights,
        legacy_validation_auprc_weights,
    )

    predictions = _validate_predictions(
        _branch_predictions("hst"),
        name="test branch predictions",
    )
    weights = legacy_validation_auprc_weights(
        _branch_validation_metrics(predictions)
    )
    substituted_test = predictions.loc[predictions["split"].eq("test")].copy()
    substituted_test["checkpoint_hash"] = _digest("substituted-test-checkpoint")

    with pytest.raises(ValueError, match="branch.*identity|provenance|substitution"):
        fuse_with_fixed_weights(substituted_test, weights)


def test_fused_predictions_carry_content_and_branch_provenance_hashes() -> None:
    from covid_rars.hst_fusion import fuse_uniform_complete_case

    fused = fuse_uniform_complete_case(_branch_predictions("hst"))

    required = {
        "source_prediction_artifact_hash",
        "source_branch_provenance_hashes",
    }
    assert required.issubset(fused.columns)
    assert fused["source_prediction_artifact_hash"].str.fullmatch(r"[0-9a-f]{64}").all()
    branch_hashes = fused["source_branch_provenance_hashes"].map(json.loads)
    assert branch_hashes.map(set).eq({"cough", "speech"}).all()
    assert branch_hashes.map(
        lambda values: all(
            isinstance(value, str)
            and len(value) == 64
            and set(value) <= set("0123456789abcdef")
            for value in values.values()
        )
    ).all()


def test_hybrid_rejects_swapped_or_unprovenanced_source_families() -> None:
    from covid_rars.hst_fusion import build_four_branch_hybrid_inputs

    hst = _branch_predictions("hst")
    comparator = _branch_predictions("comparator")

    with pytest.raises(ValueError, match="source_family|source family"):
        build_four_branch_hybrid_inputs(comparator, hst)
    with pytest.raises(ValueError, match="source_family|provenance"):
        build_four_branch_hybrid_inputs(hst.drop(columns="source_family"), comparator)


def test_hybrid_propagates_actual_source_artifact_and_branch_identity_hashes() -> None:
    from covid_rars.hst_fusion import (
        FOUR_BRANCH_COLUMNS,
        build_four_branch_hybrid_inputs,
    )

    hst = _branch_predictions("hst")
    comparator = _branch_predictions("comparator")
    hybrid = build_four_branch_hybrid_inputs(hst, comparator)

    artifact_columns = {
        "hst_prediction_artifact_hash",
        "comparator_prediction_artifact_hash",
    }
    branch_columns = {
        f"{branch}_{field}"
        for branch in FOUR_BRANCH_COLUMNS
        for field in (
            "model",
            "checkpoint_hash",
            "representation",
            "branch_provenance_hash",
        )
    }
    assert artifact_columns | branch_columns <= set(hybrid.columns)
    for column in artifact_columns | {
        f"{branch}_branch_provenance_hash" for branch in FOUR_BRANCH_COLUMNS
    }:
        assert hybrid[column].str.fullmatch(r"[0-9a-f]{64}").all()
    assert hybrid["hst_cough_checkpoint_hash"].eq(
        hst.loc[hst["modality"].eq("cough"), "checkpoint_hash"].iloc[0]
    ).all()
    assert hybrid["comparator_speech_representation"].eq(
        comparator.loc[
            comparator["modality"].eq("speech"), "representation"
        ].iloc[0]
    ).all()


def test_hybrid_final_provenance_binds_actual_upstream_artifact_and_branch_hashes() -> None:
    from covid_rars.hst_fusion import (
        FOUR_BRANCH_COLUMNS,
        _hybrid_to_long,
        build_four_branch_hybrid_inputs,
        fuse_uniform_complete_case,
    )

    hybrid = build_four_branch_hybrid_inputs(
        _branch_predictions("hst"),
        _branch_predictions("comparator"),
    )
    fused = fuse_uniform_complete_case(_hybrid_to_long(hybrid))
    artifact_hashes = fused["upstream_source_prediction_artifact_hashes"].map(json.loads)
    branch_hashes = fused["upstream_branch_provenance_hashes"].map(json.loads)

    expected_artifacts = {
        "hst": hybrid["hst_prediction_artifact_hash"].iloc[0],
        "comparator": hybrid["comparator_prediction_artifact_hash"].iloc[0],
    }
    expected_branches = {
        branch: hybrid[f"{branch}_branch_provenance_hash"].iloc[0]
        for branch in FOUR_BRANCH_COLUMNS
    }
    assert artifact_hashes.map(lambda value: value == expected_artifacts).all()
    assert branch_hashes.map(lambda value: value == expected_branches).all()


@pytest.mark.parametrize(
    "provenance_column",
    ["hst_prediction_artifact_hash", "hst_cough_branch_provenance_hash"],
)
def test_hybrid_upstream_provenance_mutation_changes_final_fused_identity(
    provenance_column: str,
) -> None:
    from covid_rars.hst_fusion import (
        _hybrid_to_long,
        build_four_branch_hybrid_inputs,
        fuse_uniform_complete_case,
    )

    hybrid = build_four_branch_hybrid_inputs(
        _branch_predictions("hst"),
        _branch_predictions("comparator"),
    )
    original = fuse_uniform_complete_case(_hybrid_to_long(hybrid))
    mutated = hybrid.copy()
    mutated[provenance_column] = _digest(f"mutated:{provenance_column}")
    changed = fuse_uniform_complete_case(_hybrid_to_long(mutated))

    assert changed["checkpoint_hash"].iloc[0] != original["checkpoint_hash"].iloc[0]
    assert (
        changed["source_prediction_artifact_hash"].iloc[0]
        != original["source_prediction_artifact_hash"].iloc[0]
    )


def test_every_fusion_result_row_has_enforceable_analysis_hierarchy() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    result = _run_confirmatory_bank(
        _branch_predictions("hst", modalities=("cough", "breath", "speech")),
        _branch_predictions("comparator"),
    )
    hierarchy_columns = {
        "analysis_scope",
        "analysis_role",
        "estimand_id",
        "multiplicity_family",
    }
    tables = {
        "predictions": result.predictions,
        "metrics": result.metrics,
        "weights": result.weights,
        "stacker_parameters": result.stacker_parameters,
        "complete_case_counts": result.complete_case_counts,
        "paired_deltas": result.paired_deltas,
    }
    for name, table in tables.items():
        assert hierarchy_columns <= set(table.columns), name
        assert table[list(hierarchy_columns)].notna().all(axis=None), name
        assert (
            table[list(hierarchy_columns)]
            .apply(lambda column: column.astype(str).str.strip().ne("").all())
            .all()
        ), name

    allowed_roles = {"primary", "secondary", "sensitivity", "exploratory"}
    allowed_scopes = {
        "confirmatory",
        "selection",
        "secondary",
        "sensitivity",
        "exploratory",
    }
    assert set(result.predictions["analysis_role"]) <= allowed_roles
    assert set(result.predictions["analysis_scope"]) <= allowed_scopes

    available = result.predictions.loc[
        result.predictions["fusion_method"].eq(
            "available_modalities_validation_weighted_auprc"
        )
    ]
    available_validation = available.loc[available["split"].eq("validation")]
    available_test = available.loc[available["split"].eq("test")]
    assert available_validation["analysis_role"].eq("secondary").all()
    assert available_validation["analysis_scope"].eq("selection").all()
    assert available_test["analysis_role"].eq("sensitivity").all()
    assert available_test["analysis_scope"].eq("sensitivity").all()
    hybrid = result.predictions.loc[result.predictions["source_family"].eq("hybrid")]
    assert hybrid["analysis_role"].eq("secondary").all()
    assert hybrid.loc[hybrid["split"].eq("validation"), "analysis_scope"].eq(
        "selection"
    ).all()
    assert hybrid.loc[hybrid["split"].eq("test"), "analysis_scope"].eq(
        "secondary"
    ).all()

    primary_delta = result.paired_deltas.loc[
        result.paired_deltas["analysis_role"].eq("primary")
    ]
    assert len(primary_delta) == 1
    assert primary_delta["split"].eq("test").all()
    assert primary_delta["candidate_family"].eq("hst").all()
    assert primary_delta["reference_family"].eq("comparator").all()
    assert primary_delta["metric"].eq("auroc").all()
    assert primary_delta["estimand_id"].eq(
        "primary_hst_vs_comparator_uniform_cough_speech_auroc"
    ).all()


def test_validation_rows_are_secondary_selection_evidence_not_primary() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    result = _run_confirmatory_bank(
        _branch_predictions("hst"),
        _branch_predictions("comparator"),
    )
    validation = result.predictions.loc[result.predictions["split"].eq("validation")]
    test_primary = result.predictions.loc[
        result.predictions["split"].eq("test")
        & result.predictions["analysis_role"].eq("primary")
    ]

    assert not validation.empty
    assert not validation["analysis_role"].eq("primary").any()
    selected_uniform = validation.loc[
        validation["source_family"].isin(["hst", "comparator"])
        & validation["fusion_method"].eq("uniform_mean")
        & validation["modality_combination"].eq("cough+speech")
    ]
    assert selected_uniform["analysis_role"].eq("secondary").all()
    assert selected_uniform["analysis_scope"].eq("selection").all()
    assert set(test_primary["source_family"]) == {"hst", "comparator"}


def test_validation_only_bank_never_emits_confirmatory_primary_evidence() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    hst = _branch_predictions("hst")
    comparator = _branch_predictions("comparator")
    hst = _refresh_recording_intersection(hst.loc[hst["split"].eq("validation")])
    comparator = _refresh_recording_intersection(
        comparator.loc[comparator["split"].eq("validation")]
    )

    result = _run_confirmatory_bank(hst, comparator)

    for table in (result.predictions, result.metrics, result.paired_deltas):
        assert not table["analysis_role"].eq("primary").any()
        assert not table["analysis_scope"].eq("confirmatory").any()


def test_confirmatory_bank_fails_closed_without_authenticated_binding() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    with pytest.raises(ValueError, match="authenticated.*binding|trusted.*receipt"):
        run_hst_fusion_bank(
            _branch_predictions("hst"),
            _branch_predictions("comparator"),
            analysis_mode="confirmatory",
        )


def test_explicit_exploratory_bank_never_promotes_primary_without_binding() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    result = run_hst_fusion_bank(
        _branch_predictions("hst"),
        _branch_predictions("comparator"),
        analysis_mode="exploratory",
    )

    for table in (result.predictions, result.metrics, result.paired_deltas):
        assert not table["analysis_role"].eq("primary").any()
        assert not table["analysis_scope"].eq("confirmatory").any()


def test_confirmatory_bank_requires_exact_authenticated_registry_binding() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    hst = _branch_predictions("hst")
    comparator = _branch_predictions("comparator")
    binding = _authenticated_binding(hst, comparator)

    result = run_hst_fusion_bank(
        hst,
        comparator,
        analysis_mode="confirmatory",
        authenticated_binding=binding,
    )

    primary = result.predictions.loc[result.predictions["analysis_role"].eq("primary")]
    assert not primary.empty
    assert primary["split"].eq("test").all()
    assert primary["authenticated_registry_receipt_sha256"].eq(
        binding.receipt_sha256
    ).all()


@pytest.mark.parametrize(
    ("family", "column"),
    [
        ("hst", "checkpoint_hash"),
        ("hst", "feature_artifact_sha256"),
        ("comparator", "feature_approval_id"),
        ("comparator", "preprocessing_sha256"),
    ],
)
def test_confirmatory_bank_rejects_identity_not_approved_by_registry(
    family: str,
    column: str,
) -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    hst = _branch_predictions("hst")
    comparator = _branch_predictions("comparator")
    binding = _authenticated_binding(hst, comparator)
    changed = hst if family == "hst" else comparator
    changed.loc[changed["modality"].eq("cough"), column] = (
        "unapproved-identity"
        if column == "feature_approval_id"
        else _digest(f"unapproved:{family}:{column}")
    )

    with pytest.raises(ValueError, match="authenticated|registry|approved|identity"):
        run_hst_fusion_bank(
            hst,
            comparator,
            analysis_mode="confirmatory",
            authenticated_binding=binding,
        )


def test_confirmatory_bank_rejects_tampered_authenticated_binding_state() -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    hst = _branch_predictions("hst")
    comparator = _branch_predictions("comparator")
    binding = _authenticated_binding(hst, comparator)
    object.__setattr__(binding, "_receipt_json", binding.receipt_json + " ")

    with pytest.raises(ValueError, match="authenticated.*tamper|receipt.*hash|canonical"):
        run_hst_fusion_bank(
            hst,
            comparator,
            analysis_mode="confirmatory",
            authenticated_binding=binding,
        )


def test_standalone_fusion_predictions_are_explicitly_nonconfirmatory() -> None:
    from covid_rars.hst_fusion import fuse_uniform_complete_case

    fused = fuse_uniform_complete_case(_branch_predictions("hst"))

    assert {
        "analysis_scope",
        "analysis_role",
        "estimand_id",
        "multiplicity_family",
    } <= set(fused.columns)
    validation = fused.loc[fused["split"].eq("validation")]
    test = fused.loc[fused["split"].eq("test")]
    assert validation["analysis_role"].eq("secondary").all()
    assert validation["analysis_scope"].eq("selection").all()
    assert test["analysis_role"].eq("exploratory").all()
    assert test["analysis_scope"].eq("exploratory").all()


def test_fusion_result_rejects_any_export_table_without_analysis_hierarchy() -> None:
    from covid_rars.hst_fusion import HSTFusionResult, run_hst_fusion_bank

    result = _run_confirmatory_bank(
        _branch_predictions("hst"),
        _branch_predictions("comparator"),
    )
    invalid_metrics = result.metrics.drop(columns="estimand_id")

    with pytest.raises(ValueError, match="hierarchy|estimand_id"):
        HSTFusionResult(
            predictions=result.predictions,
            metrics=invalid_metrics,
            weights=result.weights,
            stacker_parameters=result.stacker_parameters,
            complete_case_counts=result.complete_case_counts,
            paired_deltas=result.paired_deltas,
        )


def test_paired_delta_rejects_non_sha_comparison_binding() -> None:
    from covid_rars.hst_fusion import _paired_delta_rows, run_hst_fusion_bank

    result = _run_confirmatory_bank(
        _branch_predictions("hst"),
        _branch_predictions("comparator"),
    )
    tampered = result.predictions.copy()
    tampered["comparison_binding_hash"] = "tampered"

    with pytest.raises(ValueError, match="binding.*SHA|SHA.*binding"):
        _paired_delta_rows(tampered)


def test_export_rejects_secondary_confirmatory_role_scope_pair() -> None:
    from covid_rars.hst_fusion import HSTFusionResult

    result = _run_confirmatory_bank(
        _branch_predictions("hst"),
        _branch_predictions("comparator"),
    )
    predictions = result.predictions.copy()
    row = predictions.index[predictions["analysis_role"].eq("secondary")][0]
    predictions.loc[row, "analysis_scope"] = "confirmatory"

    with pytest.raises(ValueError, match="role/scope|secondary.*confirmatory"):
        HSTFusionResult(
            predictions=predictions,
            metrics=result.metrics,
            weights=result.weights,
            stacker_parameters=result.stacker_parameters,
            complete_case_counts=result.complete_case_counts,
            paired_deltas=result.paired_deltas,
        )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("split", "validation"),
        ("source_family", "hybrid"),
        ("fusion_method", "legacy_validation_weighted_auprc"),
        ("modality_combination", "cough+breath"),
        ("complete_case", False),
        ("comparison_binding_hash", "not_applicable"),
    ],
)
def test_export_rejects_primary_rows_outside_exact_confirmatory_contract(
    column: str,
    value: object,
) -> None:
    from covid_rars.hst_fusion import HSTFusionResult

    result = _run_confirmatory_bank(
        _branch_predictions("hst"),
        _branch_predictions("comparator"),
    )
    predictions = result.predictions.copy()
    row = predictions.index[predictions["analysis_role"].eq("primary")][0]
    predictions.loc[row, column] = value

    with pytest.raises(ValueError, match="Primary|primary|confirmatory|binding"):
        HSTFusionResult(
            predictions=predictions,
            metrics=result.metrics,
            weights=result.weights,
            stacker_parameters=result.stacker_parameters,
            complete_case_counts=result.complete_case_counts,
            paired_deltas=result.paired_deltas,
        )


def test_export_rejects_primary_delta_with_wrong_source_families() -> None:
    from covid_rars.hst_fusion import HSTFusionResult

    result = _run_confirmatory_bank(
        _branch_predictions("hst"),
        _branch_predictions("comparator"),
    )
    paired_deltas = result.paired_deltas.copy()
    row = paired_deltas.index[paired_deltas["analysis_role"].eq("primary")][0]
    paired_deltas.loc[row, "candidate_family"] = "hybrid"

    with pytest.raises(ValueError, match="Primary|primary|source|family"):
        HSTFusionResult(
            predictions=result.predictions,
            metrics=result.metrics,
            weights=result.weights,
            stacker_parameters=result.stacker_parameters,
            complete_case_counts=result.complete_case_counts,
            paired_deltas=paired_deltas,
        )


@pytest.mark.parametrize(
    "normalized",
    [
        {"cough": 1e308, "speech": 1e308},
        {"cough": np.nan, "speech": 1.0},
        {"cough": np.inf, "speech": 1.0},
    ],
)
def test_direct_validation_weight_map_rejects_nonfinite_or_overflowed_totals(
    normalized: dict[str, float],
) -> None:
    from covid_rars.hst_fusion import ValidationWeightMap

    with pytest.raises(ValueError, match="weight.*finite|weight.*overflow|sum.*one"):
        ValidationWeightMap(
            normalized,
            raw_weights={"cough": 1.0, "speech": 1.0},
            run_id="run-1",
            protocol="repeated_holdout",
            fold=0,
            dataset="coswara",
            reference=0.5,
            floor=0.01,
        )


def test_available_modality_fixed_weights_raise_instead_of_producing_nan() -> None:
    from covid_rars.hst_fusion import (
        ValidationWeightMap,
        _validate_predictions,
        fuse_available_modalities_sensitivity,
    )

    predictions = _branch_predictions("hst")
    missing_speech = "coswara::test_f0_p00"
    predictions = predictions.loc[
        ~(
            predictions["participant_key"].eq(missing_speech)
            & predictions["modality"].eq("speech")
        )
    ].copy()
    predictions = _refresh_recording_intersection(predictions)
    validated = _validate_predictions(
        predictions,
        name="zero-denominator test predictions",
    )
    branch_hashes = validated.groupby("modality")[
        "branch_provenance_hash"
    ].first().to_dict()
    weights = ValidationWeightMap(
        {"cough": 0.0, "speech": 1.0},
        raw_weights={"cough": 0.0, "speech": 1.0},
        run_id="run-1",
        protocol="repeated_holdout",
        fold=0,
        dataset="coswara",
        reference=0.5,
        floor=0.01,
        branch_provenance_hashes=branch_hashes,
    )

    with pytest.raises(ValueError, match="zero.*denominator|finite.*probabil"):
        fuse_available_modalities_sensitivity(predictions, weights)


def test_complete_fusion_generation_export_is_atomic_checksummed_and_current(
    tmp_path: Path,
) -> None:
    from covid_rars.hst_fusion import run_hst_fusion_bank

    result = _run_confirmatory_bank(
        _branch_predictions("hst"),
        _branch_predictions("comparator"),
    )
    receipt = result.save_generation(tmp_path)

    expected_tables = {
        "predictions",
        "metrics",
        "weights",
        "stacker_parameters",
        "complete_case_counts",
        "paired_deltas",
    }
    assert receipt["status"] == "success"
    assert set(receipt["artifacts"]) == expected_tables
    generation = tmp_path / receipt["generation_path"]
    assert generation.is_dir()
    assert set(path.name for path in generation.iterdir()) == {
        *(f"{name}.csv" for name in expected_tables),
        "checksums.json",
    }
    for artifact in receipt["artifacts"].values():
        artifact_path = generation / artifact["relative_path"]
        assert sha256(artifact_path.read_bytes()).hexdigest() == artifact["sha256"]
    assert json.loads((generation / "checksums.json").read_text(encoding="utf-8")) == receipt
    assert json.loads((tmp_path / "current.json").read_text(encoding="utf-8")) == receipt
    assert result.save_generation(tmp_path) == receipt
    assert not list(tmp_path.glob(".hst-fusion-staging-*"))


def test_generation_export_failure_never_publishes_partial_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_rars.hst_fusion as fusion_module

    result = _run_confirmatory_bank(
        _branch_predictions("hst"),
        _branch_predictions("comparator"),
    )
    real_replace = fusion_module.os.replace

    def fail_generation_promotion(source: object, destination: object) -> None:
        if Path(destination).parent.name == "generations":
            raise OSError("intentional generation promotion failure")
        real_replace(source, destination)

    monkeypatch.setattr(fusion_module.os, "replace", fail_generation_promotion)

    with pytest.raises(OSError, match="intentional generation promotion failure"):
        result.save_generation(tmp_path)
    assert not (tmp_path / "current.json").exists()
    assert not list((tmp_path / "generations").glob("*"))
    assert not list(tmp_path.glob(".hst-fusion-staging-*"))
