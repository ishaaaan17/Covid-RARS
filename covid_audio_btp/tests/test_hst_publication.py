from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _predictions(
    probabilities: list[float],
    *,
    prefix: str = "p",
    split: str = "test",
    fold: int | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "participant_key": [f"{prefix}{index}" for index in range(8)],
            "label_binary": ["negative"] * 4 + ["positive"] * 4,
            "probability": probabilities,
            "split": split,
            "dataset": "coswara",
            "protocol": "track_a_internal",
            "modality": "multimodal",
        }
    )
    if fold is not None:
        frame["fold"] = fold
    return frame


def _authenticated(frame: pd.DataFrame, name: str = "predictions"):
    from covid_audio_btp.hst_publication import authenticate_table

    return authenticate_table(
        frame,
        source_name=name,
        manifest_sha256=_digest(f"manifest::{name}"),
        test_mode=True,
    )


def _analysis_plan(*estimand_ids: str):
    from covid_audio_btp.hst_publication import ANALYSIS_SCOPE_REGISTRY, bind_analysis_plan

    rows = [_primary_plan().iloc[0].to_dict()]
    for estimand_id in estimand_ids:
        if estimand_id == rows[0]["estimand_id"]:
            continue
        scope = ANALYSIS_SCOPE_REGISTRY.get(estimand_id)
        if scope is None:
            rows.append(
                {
                    **rows[0],
                    "estimand_id": estimand_id,
                    "analysis_role": "exploratory",
                    "analysis_scope": "exploratory",
                    "multiplicity_family": "exploratory_other",
                    "comparison_design": "paired_model",
                    "candidate_family": "exploratory",
                    "reference_family": "exploratory",
                }
            )
            continue
        rows.append(
            {
                **rows[0],
                "estimand_id": estimand_id,
                "analysis_role": scope.role,
                "analysis_scope": scope.scope,
                "multiplicity_family": scope.family,
                "metric": scope.metric,
                "comparison_design": scope.design,
                "candidate_family": "hst",
                "reference_family": "comparator",
                "split": "test",
            }
        )
    return bind_analysis_plan(
        _authenticated(pd.DataFrame(rows), "analysis-plan"),
        test_mode=True,
    )


def _write_receipted_table(
    tmp_path: Path,
    *,
    stage: str,
    relative_path: str,
    frame: pd.DataFrame,
    receipt_type: str = "hst_stage",
) -> tuple[Path, Path, str]:
    from covid_audio_btp.hst_runtime import canonical_json_sha256, stable_file_sha256

    run_root = tmp_path / "trusted-run"
    table_path = run_root / relative_path
    table_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table_path, index=False)
    receipt = {
        "version": 1,
        "receipt_type": receipt_type,
        "run_id": run_root.name,
        "stage": stage,
        "status": "success",
        "output_paths": [relative_path],
        "output_checksums": {relative_path: stable_file_sha256(table_path)},
        "row_counts": {relative_path: len(frame)},
        "metadata": {},
        "error": None,
    }
    receipt["record_hash"] = canonical_json_sha256(receipt)
    receipt_path = run_root / "runtime" / "stages" / f"{stage}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        encoding="ascii",
    )
    return run_root, table_path, hashlib.sha256(receipt_path.read_bytes()).hexdigest()


def _primary_prediction_pair():
    from covid_audio_btp.hst_publication import PRIMARY_ESTIMAND_ID

    base = _predictions([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]).assign(
        analysis_role="primary",
        analysis_scope="confirmatory",
        estimand_id=PRIMARY_ESTIMAND_ID,
        multiplicity_family="confirmatory_primary_single",
        fusion_method="uniform_mean",
        modality_combination="cough+speech",
        complete_case=True,
        comparison_binding_hash=_digest("comparison"),
        authenticated_registry_receipt_sha256=_digest("registry"),
        authenticated_context_binding_sha256=_digest("context"),
    )
    left = _authenticated(base.assign(source_family="hst"), "hst")
    right = _authenticated(
        base.assign(
            source_family="comparator",
            probability=[0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8],
        ),
        "comparator",
    )
    return left, right


def _primary_plan() -> pd.DataFrame:
    from covid_audio_btp.hst_publication import PRIMARY_ESTIMAND_ID

    return pd.DataFrame(
        [
            {
                "estimand_id": PRIMARY_ESTIMAND_ID,
                "analysis_role": "primary",
                "analysis_scope": "confirmatory",
                "multiplicity_family": "confirmatory_primary_single",
                "metric": "auroc",
                "comparison_design": "paired_model",
                "candidate_family": "hst",
                "reference_family": "comparator",
                "split": "test",
                "fusion_method": "uniform_mean",
                "modality_combination": "cough+speech",
                "complete_case": True,
            }
        ]
    )


def test_analysis_registry_is_immutable_and_has_exactly_one_primary() -> None:
    from covid_audio_btp.hst_publication import ANALYSIS_SCOPE_REGISTRY

    assert isinstance(ANALYSIS_SCOPE_REGISTRY, MappingProxyType)
    primary = [item for item in ANALYSIS_SCOPE_REGISTRY.values() if item.role == "primary"]
    assert len(primary) == 1
    assert primary[0].metric == "auroc"
    assert primary[0].design == "paired_model"
    assert {
        "split_policy_temporal_contrast",
        "common_late_temporal_contrast",
        "coswara_to_coughvid_external_transfer",
        "secondary_hybrid_vs_hst_auroc",
        "secondary_hybrid_vs_comparator_auroc",
        "secondary_fusion_vs_best_constituent_auroc",
    } <= set(ANALYSIS_SCOPE_REGISTRY)
    with pytest.raises(TypeError):
        ANALYSIS_SCOPE_REGISTRY["invented"] = primary[0]  # type: ignore[index]


def test_analysis_plan_rejects_promoted_unknown_and_contradictory_primary() -> None:
    from covid_audio_btp.hst_publication import freeze_analysis_plan

    promoted = pd.concat(
        [
            _primary_plan(),
            pd.DataFrame(
                [
                    {
                        **_primary_plan().iloc[0].to_dict(),
                        "estimand_id": "best_after_looking_at_results",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="unregistered|primary"):
        freeze_analysis_plan(promoted)

    contradictory = _primary_plan().assign(fusion_method="validation_weighted")
    with pytest.raises(ValueError, match="uniform_mean"):
        freeze_analysis_plan(contradictory)

    string_boolean = _primary_plan().assign(complete_case="False")
    with pytest.raises(ValueError, match="boolean|complete_case"):
        freeze_analysis_plan(string_boolean)


def test_unknown_analyses_are_only_sensitivity_or_exploratory() -> None:
    from covid_audio_btp.hst_publication import freeze_analysis_plan

    exploratory = pd.DataFrame(
        [
            {
                **_primary_plan().iloc[0].to_dict(),
                "estimand_id": "unregistered_ablation",
                "analysis_role": "exploratory",
                "analysis_scope": "exploratory",
                "multiplicity_family": "exploratory_other",
                "comparison_design": "paired_model",
                "candidate_family": "hst_ablation",
                "reference_family": "hst",
                "fusion_method": "ablation",
                "complete_case": False,
            }
        ]
    )
    frozen = freeze_analysis_plan(pd.concat([_primary_plan(), exploratory], ignore_index=True))
    assert frozen.loc[frozen["estimand_id"].eq("unregistered_ablation"), "analysis_role"].item() == "exploratory"


def test_bootstrap_evidence_routes_to_frozen_reporting_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    import covid_audio_btp.hst_publication as publication

    calls: list[tuple[int, int, str]] = []

    def fake_paired(
        left,
        right,
        *,
        metric,
        n_bootstrap,
        seed,
        allow_model_input_context_difference=False,
    ):
        assert allow_model_input_context_difference is False
        calls.append((n_bootstrap, seed, metric))
        return {
            "metric": metric,
            "delta": 0.1,
            "ci_low": 0.01,
            "ci_high": 0.2,
            "paired": True,
            "bootstrap_design": "paired_participant_cluster",
            "valid_replicates": n_bootstrap,
            "attempts": n_bootstrap,
        }

    monkeypatch.setattr(publication.hst_reporting, "paired_model_cluster_delta", fake_paired)
    left, right = _primary_prediction_pair()
    request = publication.PublicationComparison(
        estimand_id=publication.PRIMARY_ESTIMAND_ID,
        left=left,
        right=right,
    )
    result = publication.build_bootstrap_evidence(
        [request], analysis_plan=_analysis_plan(), test_mode=True
    )
    assert calls == [(1000, 42, "auroc")]
    assert result.loc[0, "valid_replicates"] == 1000
    assert {
        "left_source_artifact_sha256",
        "left_source_manifest_sha256",
        "left_source_receipt_sha256",
        "left_source_receipt_record_hash",
        "left_source_stage",
        "left_source_relative_path",
        "analysis_plan_source_artifact_sha256",
        "analysis_plan_source_stage",
        "analysis_plan_source_relative_path",
    } <= set(result.columns)


def test_primary_comparison_rejects_unpaired_participants() -> None:
    from covid_audio_btp.hst_publication import (
        PRIMARY_ESTIMAND_ID,
        PublicationComparison,
        build_bootstrap_evidence,
    )

    left, valid_right = _primary_prediction_pair()
    right_frame = valid_right.frame.assign(
        participant_key=[f"q{index}" for index in range(len(valid_right.frame))]
    )
    right = _authenticated(right_frame, "comparator_unpaired")
    with pytest.raises(ValueError, match="identical participant keys"):
        build_bootstrap_evidence(
            [PublicationComparison(estimand_id=PRIMARY_ESTIMAND_ID, left=left, right=right)],
            analysis_plan=_analysis_plan(),
            test_mode=True,
        )


def test_primary_bootstrap_rejects_nonuniform_or_unauthenticated_claim() -> None:
    from covid_audio_btp.hst_publication import (
        PRIMARY_ESTIMAND_ID,
        PublicationComparison,
        build_bootstrap_evidence,
    )

    left, right = _primary_prediction_pair()
    invalid = _authenticated(
        left.frame.assign(fusion_method="validation_weighted"),
        "invalid_hst",
    )
    with pytest.raises(ValueError, match="uniform_mean"):
        build_bootstrap_evidence(
            [
                PublicationComparison(
                    estimand_id=PRIMARY_ESTIMAND_ID,
                    left=invalid,
                    right=right,
                )
            ],
            analysis_plan=_analysis_plan(),
            test_mode=True,
        )


def test_external_comparison_is_independent_bootstrap_and_never_delong() -> None:
    from covid_audio_btp.hst_publication import PublicationComparison, build_bootstrap_evidence

    source = _authenticated(_predictions([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]), "source")
    target_frame = _predictions(
        [0.1, 0.8, 0.3, 0.6, 0.2, 0.7, 0.4, 0.9],
        prefix="external",
    ).assign(
        dataset="coughvid",
        protocol="external_transfer",
        modality="cough",
        analysis_unit_type="recording_uuid",
        subject_linkage_available=False,
    )
    target = _authenticated(target_frame, "target")
    result = build_bootstrap_evidence(
        [
            PublicationComparison(
                estimand_id="coswara_to_coughvid_external_transfer",
                left=source,
                right=target,
            )
        ],
        analysis_plan=_analysis_plan("coswara_to_coughvid_external_transfer"),
        test_mode=True,
    )
    assert result.loc[0, "bootstrap_design"] == "independent_label_stratified_participants"
    assert result.loc[0, "paired"] == False  # noqa: E712
    assert result.loc[0, "test_method"] == (
        "independent_participant_bootstrap_ci_only"
    )
    assert np.isnan(result.loc[0, "p_value"])


def test_external_delta_rejects_incomplete_repeated_fold_rows_instead_of_pooling() -> None:
    from covid_audio_btp.hst_publication import PublicationComparison, build_bootstrap_evidence

    source_fold = _predictions(
        [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], fold=0
    )
    source = _authenticated(
        pd.concat([source_fold, source_fold.assign(fold=1)], ignore_index=True),
        "source_repeated",
    )
    target_fold = source_fold.assign(
        participant_key=[f"external{index}" for index in range(8)],
        dataset="coughvid",
        protocol="external_transfer",
        modality="cough",
    )
    target = _authenticated(
        pd.concat([target_fold, target_fold.assign(fold=1)], ignore_index=True),
        "target_repeated",
    )
    with pytest.raises(ValueError, match="ten source folds"):
        build_bootstrap_evidence(
            [
                PublicationComparison(
                    estimand_id="coswara_to_coughvid_external_transfer",
                    left=source,
                    right=target,
                )
            ],
            analysis_plan=_analysis_plan("coswara_to_coughvid_external_transfer"),
            test_mode=True,
        )


def test_repeated_fold_summary_is_clustered_not_row_pooled(monkeypatch: pytest.MonkeyPatch) -> None:
    import covid_audio_btp.hst_publication as publication

    calls: list[int] = []

    def fake_repeated(frame, *, metric, n_bootstrap, seed):
        calls.append(len(frame))
        return {
            "metric": metric,
            "point": 0.75,
            "ci_low": 0.70,
            "ci_high": 0.80,
            "estimand": "mean_repetition_metric",
            "independent_row_pooling": False,
            "resampling_unit": "participant_key_cluster_across_folds",
            "valid_replicates": n_bootstrap,
            "attempts": n_bootstrap,
        }

    monkeypatch.setattr(publication.hst_reporting, "repeated_holdout_cluster_ci", fake_repeated)
    first = _predictions([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], fold=0)
    second = _predictions([0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8], fold=1)
    table = _authenticated(pd.concat([first, second], ignore_index=True), "repeated")
    result = publication.build_repeated_fold_evidence(
        table,
        estimand_id="exploratory_repeated_holdout",
        metric="auroc",
        analysis_plan=_analysis_plan("exploratory_repeated_holdout"),
        test_mode=True,
    )
    assert calls == [16]
    assert result["independent_row_pooling"] is False


def test_holm_adjustment_is_within_declared_secondary_families() -> None:
    from covid_audio_btp.hst_publication import adjust_secondary_holm

    evidence = pd.DataFrame(
        {
            "estimand_id": [
                "split_policy_temporal_contrast",
                "common_late_temporal_contrast",
                "secondary_hybrid_vs_hst_auroc",
                "primary_hst_vs_comparator_uniform_cough_speech_auroc",
            ],
            "analysis_role": ["secondary", "secondary", "secondary", "primary"],
            "multiplicity_family": [
                "prespecified_reliability",
                "prespecified_reliability",
                "secondary_hybrid_deltas",
                "confirmatory_primary_single",
            ],
            "p_value": [0.01, 0.04, 0.03, 0.001],
        }
    )
    adjusted = adjust_secondary_holm(
        evidence,
        analysis_plan=_analysis_plan(
            "split_policy_temporal_contrast",
            "common_late_temporal_contrast",
            "secondary_hybrid_vs_hst_auroc",
        ),
        test_mode=True,
    )
    values = adjusted.set_index("estimand_id")["p_value_holm"].to_dict()
    assert values["split_policy_temporal_contrast"] == pytest.approx(0.02)
    assert values["common_late_temporal_contrast"] == pytest.approx(0.04)
    assert values["secondary_hybrid_vs_hst_auroc"] == pytest.approx(0.03)
    assert np.isnan(
        values["primary_hst_vs_comparator_uniform_cough_speech_auroc"]
    )


def test_holm_adjustment_requires_estimand_identity() -> None:
    from covid_audio_btp.hst_publication import adjust_secondary_holm

    evidence = pd.DataFrame(
        {
            "analysis_role": ["secondary"],
            "multiplicity_family": ["prespecified_reliability"],
            "p_value": [0.02],
        }
    )
    with pytest.raises(ValueError, match="estimand_id"):
        adjust_secondary_holm(
            evidence,
            analysis_plan=_analysis_plan("split_policy_temporal_contrast"),
            test_mode=True,
        )


def test_fusion_vs_best_constituent_is_selected_on_validation_only() -> None:
    from covid_audio_btp.hst_publication import (
        build_fusion_vs_best_constituent_evidence,
    )

    validation_a = _authenticated(
        _predictions(
            [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
            split="validation",
        ).assign(modality="cough"),
        "validation-a",
    )
    validation_b = _authenticated(
        _predictions(
            [0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.7, 0.8],
            split="validation",
        ).assign(modality="speech"),
        "validation-b",
    )
    test_a = _authenticated(
        _predictions([0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8]).assign(
            modality="cough"
        ),
        "test-a",
    )
    test_b = _authenticated(
        _predictions([0.01, 0.02, 0.03, 0.04, 0.96, 0.97, 0.98, 0.99]).assign(
            modality="speech"
        ),
        "test-b",
    )
    fusion = _authenticated(
        _predictions([0.1, 0.15, 0.2, 0.3, 0.7, 0.8, 0.85, 0.9]),
        "fusion-test",
    )
    evidence, selection = build_fusion_vs_best_constituent_evidence(
        fusion,
        {"cough": validation_a, "speech": validation_b},
        {"cough": test_a, "speech": test_b},
        analysis_plan=_analysis_plan("secondary_fusion_vs_best_constituent_auroc"),
        test_mode=True,
    )

    assert evidence.loc[0, "selected_constituent"] == "cough"
    assert evidence.loc[0, "selection_split"] == "validation"
    assert np.isnan(evidence.loc[0, "p_value"])
    assert evidence.loc[0, "hypothesis_test"] == "not_tested_bootstrap_ci_only"
    assert selection.loc[selection["selected"], "constituent"].tolist() == ["cough"]


def test_fusion_vs_best_constituent_selection_is_fold_local() -> None:
    from covid_audio_btp.hst_publication import (
        build_fusion_vs_best_constituent_evidence,
    )

    perfect = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    inverse = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]

    def repeated(values_by_fold: dict[int, list[float]], split: str, modality: str):
        return _authenticated(
            pd.concat(
                [
                    _predictions(values, split=split, fold=fold).assign(
                        modality=modality
                    )
                    for fold, values in values_by_fold.items()
                ],
                ignore_index=True,
            ),
            f"{modality}-{split}",
        )

    cough_validation = repeated({1: perfect, 2: inverse}, "validation", "cough")
    speech_validation = repeated({1: inverse, 2: perfect}, "validation", "speech")
    cough_test = repeated({1: perfect, 2: inverse}, "test", "cough")
    speech_test = repeated({1: inverse, 2: perfect}, "test", "speech")
    fusion = _authenticated(
        pd.concat(
            [
                _predictions(perfect, split="test", fold=1),
                _predictions(perfect, split="test", fold=2),
            ],
            ignore_index=True,
        ),
        "fusion-test-fold-local",
    )

    evidence, selection = build_fusion_vs_best_constituent_evidence(
        fusion,
        {"cough": cough_validation, "speech": speech_validation},
        {"cough": cough_test, "speech": speech_test},
        analysis_plan=_analysis_plan("secondary_fusion_vs_best_constituent_auroc"),
        test_mode=True,
    )

    selected = selection.loc[selection["selected"], ["fold", "constituent"]]
    assert selected.to_dict(orient="records") == [
        {"fold": 1, "constituent": "cough"},
        {"fold": 2, "constituent": "speech"},
    ]
    assert evidence.loc[0, "selection_policy"] == "fold_local_validation"
    assert evidence.loc[0, "selected_constituent"] == "fold_local"


def test_calibration_operating_point_and_dca_keep_folds_separate() -> None:
    from covid_audio_btp.hst_publication import (
        build_calibration_evidence,
        build_decision_curve_evidence,
        build_fixed_sensitivity_evidence,
    )

    identity = {
        "checkpoint_hash": _digest("checkpoint"),
        "source_protocol": "track_a_internal",
        "source_manifest_sha256": _digest("source-manifest"),
    }
    fold0 = _predictions(
        [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
        split="validation",
        fold=0,
    ).assign(**identity)
    fold1 = _predictions(
        [0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8],
        split="validation",
        fold=1,
    ).assign(**identity)
    validation = _authenticated(pd.concat([fold0, fold1], ignore_index=True), "validation")
    test = _authenticated(
        pd.concat(
            [fold0.assign(split="test"), fold1.assign(split="test")],
            ignore_index=True,
        ),
        "test",
    )

    plan = _analysis_plan()
    estimands = {"internal": "primary_hst_vs_comparator_uniform_cough_speech_auroc"}
    bins, summary = build_calibration_evidence(
        {"internal": test},
        analysis_plan=plan,
        evidence_estimand_ids=estimands,
        test_mode=True,
    )
    assert set(summary["fold"]) == {0, 1}
    assert {"brier", "ece", "nll"} <= set(summary)
    assert set(bins["fold"]) == {0, 1}

    operating = build_fixed_sensitivity_evidence(
        validation,
        {"internal": test},
        analysis_plan=plan,
        evidence_estimand_ids=estimands,
        test_mode=True,
    )
    assert set(operating["fold"]) == {0, 1}
    assert operating["target_sensitivity"].eq(0.90).all()

    dca = build_decision_curve_evidence(
        {"internal": test},
        analysis_plan=plan,
        evidence_estimand_ids=estimands,
        test_mode=True,
    )
    assert set(dca["fold"]) == {0, 1}
    assert sorted(dca["threshold"].unique().tolist()) == [
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

    mixed_identity = pd.concat(
        [fold0.assign(split="test"), fold1.assign(split="test")],
        ignore_index=True,
    )
    mixed_identity.loc[mixed_identity.index[0], "checkpoint_hash"] = _digest(
        "different-checkpoint"
    )
    mixed = _authenticated(mixed_identity, "mixed-checkpoint")
    with pytest.raises(ValueError, match="identity|checkpoint_hash"):
        build_calibration_evidence(
            {"internal": mixed},
            analysis_plan=plan,
            evidence_estimand_ids=estimands,
            test_mode=True,
        )
    with pytest.raises(ValueError, match="identity|checkpoint_hash"):
        build_decision_curve_evidence(
            {"internal": mixed},
            analysis_plan=plan,
            evidence_estimand_ids=estimands,
            test_mode=True,
        )


def test_publication_fold_reports_reject_missing_fold_identity() -> None:
    import covid_audio_btp.hst_publication as publication

    frame = _predictions(
        [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
        split="test",
        fold=1,
    )
    frame.loc[frame.index[-1], "fold"] = np.nan

    with pytest.raises(ValueError, match="non-null fold"):
        list(publication._fold_groups(frame))


def test_external_publication_requires_recording_uuid_analysis_unit_provenance() -> None:
    import covid_audio_btp.hst_publication as publication

    frame = _predictions(
        [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
        split="external_test",
        fold=1,
    ).assign(
        dataset="coughvid",
        protocol="coswara_to_coughvid_hst_external",
        source_protocol="track_a_internal",
        checkpoint_hash=_digest("external-checkpoint"),
        source_manifest_sha256=_digest("external-source-manifest"),
        modality="cough",
    )

    with pytest.raises(ValueError, match="analysis-unit provenance"):
        publication._validate_evaluation_frame(frame, name="external")

    accepted = publication._validate_evaluation_frame(
        frame.assign(
            analysis_unit_type="recording_uuid",
            subject_linkage_available=False,
        ),
        name="external",
    )
    assert len(accepted) == len(frame)


def test_source_platt_pair_derivation_is_authenticated_and_fold_local() -> None:
    import covid_audio_btp.hst_publication as publication

    identity = {
        "checkpoint_hash": _digest("checkpoint-platt"),
        "source_protocol": "track_a_internal",
        "source_manifest_sha256": _digest("source-manifest-platt"),
    }
    validation_rows: list[pd.DataFrame] = []
    evaluation_rows: list[pd.DataFrame] = []
    for fold in (0, 1):
        validation_rows.append(
            _predictions(
                [0.05, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.95],
                split="validation",
                fold=fold,
            ).assign(**identity)
        )
        evaluation_rows.append(
            _predictions(
                [0.10, 0.30, 0.40, 0.48, 0.52, 0.60, 0.70, 0.90],
                split="test",
                fold=fold,
            ).assign(**identity)
        )
    validation = _authenticated(pd.concat(validation_rows), "platt-validation")
    evaluation = _authenticated(pd.concat(evaluation_rows), "platt-evaluation")
    plan = _analysis_plan()

    calibrated_validation, calibrated_evaluation, audit = (
        publication.derive_source_platt_calibrated_pair(
            validation,
            evaluation,
            source_name="internal_hst_platt",
            analysis_plan=plan,
            test_mode=True,
        )
    )

    assert calibrated_validation.test_mode
    assert calibrated_evaluation.test_mode
    assert calibrated_evaluation.derivation_sha256 is not None
    assert calibrated_evaluation.frame["probability_scale"].eq(
        "source_validation_platt"
    ).all()
    assert set(audit["fold"]) == {0, 1}
    assert not audit["skipped"].any()


def test_aligned_comparator_audit_and_runtime_gpu_tables() -> None:
    from covid_audio_btp.hst_publication import (
        audit_aligned_comparator,
        build_runtime_gpu_tables,
    )

    frame = _predictions([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    audit = audit_aligned_comparator(
        _authenticated(frame, "hst"),
        _authenticated(frame.assign(probability=lambda value: 1 - value["probability"]), "comparator"),
    )
    assert audit.loc[0, "identical_participants"]
    assert audit.loc[0, "identical_labels"]
    assert audit.loc[0, "n_aligned_participants"] == 8

    events = pd.DataFrame(
        {
            "run_id": ["r1", "r1", "r1"],
            "stage": ["train", "external", "cpu_audit"],
            "elapsed_seconds": [12.5, 4.0, 1.0],
            "gpu_memory_measured": [True, True, False],
            "peak_gpu_memory_allocated_mb": [1536.0, 768.0, 0.0],
            "peak_gpu_memory_reserved_mb": [2048.0, 1024.0, 0.0],
            "peak_gpu_memory_mb": [1536.0, 768.0, 0.0],
            "gpu_uuid": ["GPU-1", "GPU-1", ""],
            "status": ["success", "success", "success"],
        }
    )
    stage_table, summary = build_runtime_gpu_tables(events)
    assert stage_table["stage"].tolist() == ["cpu_audit", "external", "train"]
    cpu_row = stage_table.loc[stage_table["stage"].eq("cpu_audit")].iloc[0]
    assert not cpu_row["gpu_memory_measured"]
    assert pd.isna(cpu_row["peak_gpu_memory_allocated_mb"])
    assert pd.isna(cpu_row["peak_gpu_memory_reserved_mb"])
    assert pd.isna(cpu_row["peak_gpu_memory_mb"])
    assert summary.loc[0, "total_elapsed_seconds"] == pytest.approx(17.5)
    assert summary.loc[0, "peak_gpu_memory_allocated_mb"] == pytest.approx(1536.0)
    assert summary.loc[0, "peak_gpu_memory_reserved_mb"] == pytest.approx(2048.0)
    assert summary.loc[0, "gpu_memory_measured_stages"] == 2


def test_runtime_figure_plots_memory_only_for_measured_stages() -> None:
    from covid_audio_btp.hst_publication import _plot_runtime

    frame = pd.DataFrame(
        {
            "stage": ["CPU audit", "CUDA train", "Unmeasured"],
            "elapsed_seconds": [10.0, 20.0, 30.0],
            "gpu_memory_measured": [False, True, False],
            "peak_gpu_memory_allocated_mb": [0.0, 2048.0, float("nan")],
            "peak_gpu_memory_reserved_mb": [0.0, 3072.0, float("nan")],
            "peak_gpu_memory_mb": [0.0, 2048.0, float("nan")],
        }
    )

    figure = _plot_runtime(frame)

    memory_axis = figure.axes[1]
    assert memory_axis.get_ylabel() == "Peak GPU memory (GiB; measured stages only)"
    assert [line.get_label() for line in memory_axis.lines] == [
        "Peak allocated",
        "Peak reserved",
    ]
    assert list(memory_axis.lines[0].get_xdata()) == [1]
    assert list(memory_axis.lines[0].get_ydata()) == [pytest.approx(2.0)]
    assert list(memory_axis.lines[1].get_xdata()) == [1]
    assert list(memory_axis.lines[1].get_ydata()) == [pytest.approx(3.0)]


def test_legacy_runtime_peak_does_not_fabricate_reserved_memory() -> None:
    from covid_audio_btp.hst_publication import (
        _plot_runtime,
        build_runtime_gpu_tables,
    )

    events = pd.DataFrame(
        {
            "run_id": ["legacy"],
            "stage": ["CUDA train"],
            "elapsed_seconds": [20.0],
            "peak_gpu_memory_mb": [2048.0],
            "gpu_uuid": ["GPU-1"],
            "status": ["success"],
        }
    )

    stage_table, summary = build_runtime_gpu_tables(events)
    figure = _plot_runtime(events)

    assert stage_table.loc[0, "gpu_memory_measured"]
    assert stage_table.loc[0, "peak_gpu_memory_allocated_mb"] == pytest.approx(2048.0)
    assert pd.isna(stage_table.loc[0, "peak_gpu_memory_reserved_mb"])
    assert pd.isna(summary.loc[0, "peak_gpu_memory_reserved_mb"])
    assert [line.get_label() for line in figure.axes[1].lines] == ["Peak allocated"]


def _figure_tables():
    from covid_audio_btp.hst_publication import authenticate_table

    raw = {
        "branch_fusion_performance": pd.DataFrame(
            {
                "label": ["Cough", "Speech", "Fusion"],
                "auroc": [0.82, 0.85, 0.90],
                "ci_low": [0.78, 0.81, 0.87],
                "ci_high": [0.86, 0.89, 0.93],
                "kind": ["branch", "branch", "fusion"],
            }
        ),
        "paired_comparison": pd.DataFrame(
            {
                "label": ["Fold 1", "Fold 2", "Fold 3"],
                "hst_auroc": [0.88, 0.90, 0.91],
                "comparator_auroc": [0.85, 0.86, 0.87],
            }
        ),
        "validation_ladder": pd.DataFrame(
            {
                "stage": ["Internal", "Temporal", "External"],
                "auroc": [0.90, 0.70, 0.54],
                "ci_low": [0.87, 0.65, 0.50],
                "ci_high": [0.93, 0.75, 0.58],
            }
        ),
        "calibration": pd.DataFrame(
            {
                "series": ["Internal"] * 3 + ["External"] * 3,
                "mean_probability": [0.1, 0.5, 0.9, 0.1, 0.5, 0.9],
                "observed_prevalence": [0.08, 0.52, 0.88, 0.03, 0.08, 0.20],
            }
        ),
        "decision_curve": pd.DataFrame(
            {
                "series": ["Internal"] * 3,
                "threshold": [0.05, 0.25, 0.50],
                "model_net_benefit": [0.20, 0.15, 0.05],
                "treat_all_net_benefit": [0.18, 0.02, -0.20],
                "treat_none_net_benefit": [0.0, 0.0, 0.0],
            }
        ),
        "runtime_gpu": pd.DataFrame(
            {
                "stage": ["Cache", "Train", "External"],
                "elapsed_seconds": [60.0, 600.0, 120.0],
                "peak_gpu_memory_mb": [0.0, 4096.0, 3072.0],
            }
        ),
    }
    return {
        name: authenticate_table(
            table,
            source_name=name,
            manifest_sha256=_digest(f"manifest::{name}"),
            test_mode=True,
        )
        for name, table in raw.items()
    }


def test_publication_figures_are_nonblank_checksummed_and_deterministic(tmp_path: Path) -> None:
    from covid_audio_btp.hst_publication import build_publication_figures

    first = build_publication_figures(
        tmp_path / "first",
        _figure_tables(),
        analysis_plan=_analysis_plan(),
        test_mode=True,
    )
    second = build_publication_figures(
        tmp_path / "second",
        _figure_tables(),
        analysis_plan=_analysis_plan(),
        test_mode=True,
    )
    assert len(first) == 12
    assert set(first["format"]) == {"svg", "png"}
    assert first[["figure_id", "format", "sha256", "source_table_sha256"]].equals(
        second[["figure_id", "format", "sha256", "source_table_sha256"]]
    )
    assert first["figure_manifest_sha256"].nunique() == 1
    assert first["figure_manifest_sha256"].iloc[0] == second["figure_manifest_sha256"].iloc[0]
    expected_sources = {
        table.table_sha256 for table in _figure_tables().values()
    }
    assert set(first["source_table_sha256"]) == expected_sources
    for path_text in first["path"]:
        path = Path(path_text)
        assert path.stat().st_size > 1000
        if path.suffix == ".png":
            pixels = mpimg.imread(path)
            assert float(np.std(pixels)) > 0.01


def test_authentication_detects_table_mutation() -> None:
    from covid_audio_btp.hst_publication import authenticate_table, dataframe_sha256

    frame = pd.DataFrame({"value": [1, 2]})
    expected = dataframe_sha256(frame)
    frame.loc[0, "value"] = 99
    with pytest.raises(ValueError, match="checksum"):
        authenticate_table(
            frame,
            source_name="mutated",
            manifest_sha256=_digest("manifest"),
            expected_table_sha256=expected,
            test_mode=True,
        )


def test_confirmatory_publication_requires_independently_receipted_table(
    tmp_path: Path,
) -> None:
    import covid_audio_btp.hst_publication as publication

    frame = _predictions([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]).assign(
        manifest_sha256=_digest("scientific-manifest")
    )
    with pytest.raises(ValueError, match="test_mode=True"):
        publication.authenticate_table(
            frame,
            source_name="ad-hoc",
            manifest_sha256=_digest("scientific-manifest"),
        )

    run_root, table_path, receipt_sha256 = _write_receipted_table(
        tmp_path,
        stage="internal_cv",
        relative_path="scientific/internal_cv/participant_predictions.csv",
        frame=frame,
    )
    loaded = publication.load_receipted_table(
        run_root=run_root,
        stage="internal_cv",
        relative_path=table_path.relative_to(run_root),
        expected_receipt_sha256=receipt_sha256,
    )
    assert loaded.provenance_verified is True
    assert loaded.test_mode is False
    assert loaded.stage == "internal_cv"
    assert loaded.receipt_sha256 == receipt_sha256
    assert loaded.artifact_sha256 == hashlib.sha256(table_path.read_bytes()).hexdigest()

    table_path.write_text("tampered\n", encoding="ascii")
    with pytest.raises(ValueError, match="checksum"):
        publication.load_receipted_table(
            run_root=run_root,
            stage="internal_cv",
            relative_path=table_path.relative_to(run_root),
            expected_receipt_sha256=receipt_sha256,
        )


def test_receipted_loader_rejects_non_stage_receipt_and_stage_path_traversal(
    tmp_path: Path,
) -> None:
    import covid_audio_btp.hst_publication as publication

    frame = pd.DataFrame({"manifest_sha256": [_digest("manifest")], "value": [1.0]})
    forged_root, forged_path, forged_receipt = _write_receipted_table(
        tmp_path / "forged",
        stage="statistics",
        relative_path="scientific/statistics/table.csv",
        frame=frame,
        receipt_type="caller_forged",
    )
    with pytest.raises(ValueError, match="receipt_type|HST stage receipt"):
        publication.load_receipted_table(
            run_root=forged_root,
            stage="statistics",
            relative_path=forged_path.relative_to(forged_root),
            expected_receipt_sha256=forged_receipt,
        )

    escaped_root, escaped_path, escaped_receipt = _write_receipted_table(
        tmp_path / "escaped",
        stage="../forged_stage",
        relative_path="scientific/statistics/table.csv",
        frame=frame,
    )
    with pytest.raises(ValueError, match="stage.*identity|stage.*name"):
        publication.load_receipted_table(
            run_root=escaped_root,
            stage="../forged_stage",
            relative_path=escaped_path.relative_to(escaped_root),
            expected_receipt_sha256=escaped_receipt,
        )


def test_ad_hoc_test_table_cannot_cross_confirmatory_evidence_boundary() -> None:
    import covid_audio_btp.hst_publication as publication

    left, right = _primary_prediction_pair()
    comparison = publication.PublicationComparison(
        estimand_id=publication.PRIMARY_ESTIMAND_ID,
        left=left,
        right=right,
    )
    with pytest.raises(ValueError, match="[Cc]onfirmatory.*receipt|test-mode"):
        publication.build_bootstrap_evidence(
            [comparison],
            analysis_plan=_analysis_plan(),
        )
    exploratory = publication.build_bootstrap_evidence(
        [comparison],
        analysis_plan=_analysis_plan(),
        test_mode=True,
    )
    assert exploratory["confirmatory"].eq(False).all()  # noqa: E712
    assert exploratory["execution_class"].eq("exploratory_test_only").all()


def test_authenticated_table_cannot_self_assert_confirmatory_provenance() -> None:
    import covid_audio_btp.hst_publication as publication

    plan = _primary_plan()
    forged = publication.AuthenticatedTable(
        frame=plan,
        source_name="forged-plan",
        table_sha256=publication.dataframe_sha256(plan),
        manifest_sha256=_digest("manifest"),
        artifact_sha256=_digest("artifact"),
        receipt_sha256=_digest("receipt"),
        receipt_record_hash=_digest("record"),
        run_root="/forged",
        relative_path="scientific/fusion/analysis_plan.csv",
        stage="fusion",
        provenance_verified=True,
        test_mode=False,
    )
    with pytest.raises(ValueError, match="independently verified|authenticator"):
        publication.bind_analysis_plan(forged)


def test_derived_table_requires_authenticated_sources_and_explicit_test_mode() -> None:
    import covid_audio_btp.hst_publication as publication

    source = _authenticated(_predictions([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]))
    plan = _analysis_plan()
    derived = publication.derive_authenticated_table(
        pd.DataFrame({"label": ["HST"], "auroc": [0.8]}),
        source_name="branch_fusion_performance",
        sources=[source],
        analysis_plan=plan,
        test_mode=True,
    )
    assert derived.test_mode is True
    assert derived.provenance_verified is False
    assert derived.artifact_sha256 == publication.dataframe_sha256(derived.frame)
    with pytest.raises(ValueError, match="test_mode|receipt-backed"):
        publication.derive_authenticated_table(
            pd.DataFrame({"label": ["HST"], "auroc": [0.8]}),
            source_name="branch_fusion_performance",
            sources=[source],
            analysis_plan=plan,
        )


def test_analysis_plan_binding_rejects_unregistered_evidence_choice() -> None:
    import covid_audio_btp.hst_publication as publication

    source = _authenticated(
        _predictions([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]),
        "source-plan-bound",
    )
    target = _authenticated(
        _predictions(
            [0.1, 0.8, 0.3, 0.6, 0.2, 0.7, 0.4, 0.9],
            prefix="external",
        ).assign(dataset="coughvid", protocol="external_transfer", modality="cough"),
        "target-plan-bound",
    )
    with pytest.raises(ValueError, match="frozen analysis plan"):
        publication.build_bootstrap_evidence(
            [
                publication.PublicationComparison(
                    estimand_id="coswara_to_coughvid_external_transfer",
                    left=source,
                    right=target,
                )
            ],
            analysis_plan=_analysis_plan(),
            test_mode=True,
        )


def test_fold_aware_external_publication_reports_mean_fold_and_equal_fold_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_publication as publication

    monkeypatch.setitem(publication.hst_reporting.REPORTING_CONTRACT, "bootstrap_replicates", 20)

    source_rows: list[pd.DataFrame] = []
    target_rows: list[pd.DataFrame] = []
    for fold in range(10):
        source_rows.append(
            _predictions(
                [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
                prefix=f"source-{fold}-",
                fold=fold,
            ).assign(modality="cough")
        )
        target_rows.append(
            _predictions(
                [0.1, 0.8, 0.3, 0.6, 0.2, 0.7, 0.4, 0.9],
                prefix="target-",
                fold=fold,
            ).assign(
                dataset="coughvid",
                protocol="external_transfer",
                modality="cough",
                split="external_test",
                analysis_unit_type="recording_uuid",
                subject_linkage_available=False,
            )
        )
    target_frame = pd.concat(target_rows, ignore_index=True)
    calibrated_target = target_frame.copy()
    calibrated_target["raw_probability"] = calibrated_target["probability"]
    calibrated_target["probability"] = np.tile(
        [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1],
        10,
    )
    calibrated_target["probability_scale"] = "source_validation_platt"
    result = publication.build_bootstrap_evidence(
        [
            publication.PublicationComparison(
                estimand_id="coswara_to_coughvid_external_transfer",
                left=_authenticated(pd.concat(source_rows, ignore_index=True), "source-folds"),
                right=_authenticated(target_frame, "target-folds"),
                ensemble_right=_authenticated(
                    calibrated_target,
                    "target-folds-source-validation-platt",
                ),
            )
        ],
        analysis_plan=_analysis_plan("coswara_to_coughvid_external_transfer"),
        test_mode=True,
    )
    assert set(result["endpoint"]) == {
        "mean_source_fold_vs_mean_external_fold_delta",
        "equal_source_fold_probability_ensemble",
    }
    delta = result.loc[
        result["endpoint"].eq("mean_source_fold_vs_mean_external_fold_delta")
    ].iloc[0]
    assert delta["source_fold_count"] == 10
    assert delta["external_fold_count"] == 10
    assert delta["independent_row_pooling"] == False  # noqa: E712
    ensemble = result.loc[
        result["endpoint"].eq("equal_source_fold_probability_ensemble")
    ].iloc[0]
    assert ensemble["point"] == pytest.approx(0.0)
    assert ensemble["probability_scale"] == "source_validation_platt"


def test_operating_calibration_and_dca_enforce_evaluation_provenance() -> None:
    import covid_audio_btp.hst_publication as publication

    identity = {
        "checkpoint_hash": _digest("checkpoint"),
        "source_protocol": "track_a_internal",
        "source_manifest_sha256": _digest("source-manifest"),
        "modality": "cough",
    }
    validation = _authenticated(
        _predictions(
            [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
            split="validation",
            fold=0,
        ).assign(**identity),
        "validation-provenance",
    )
    evaluation = _authenticated(
        _predictions(
            [0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8],
            split="test",
            fold=0,
        ).assign(**identity),
        "test-provenance",
    )
    plan = _analysis_plan()
    operating = publication.build_fixed_sensitivity_evidence(
        validation,
        {"internal": evaluation},
        analysis_plan=plan,
        evidence_estimand_ids={"internal": publication.PRIMARY_ESTIMAND_ID},
        test_mode=True,
    )
    assert operating["threshold_source"].eq("source_validation_fixed_sensitivity").all()

    wrong_checkpoint = _authenticated(
        evaluation.frame.assign(checkpoint_hash=_digest("other-checkpoint")),
        "wrong-checkpoint",
    )
    with pytest.raises(ValueError, match="checkpoint_hash"):
        publication.build_fixed_sensitivity_evidence(
            validation,
            {"internal": wrong_checkpoint},
            analysis_plan=plan,
            evidence_estimand_ids={"internal": publication.PRIMARY_ESTIMAND_ID},
            test_mode=True,
        )
    with pytest.raises(ValueError, match="evaluation-only|validation"):
        publication.build_calibration_evidence(
            {"invalid": validation},
            analysis_plan=plan,
            evidence_estimand_ids={"invalid": publication.PRIMARY_ESTIMAND_ID},
            test_mode=True,
        )
    with pytest.raises(ValueError, match="evaluation-only|validation"):
        publication.build_decision_curve_evidence(
            {"invalid": validation},
            analysis_plan=plan,
            evidence_estimand_ids={"invalid": publication.PRIMARY_ESTIMAND_ID},
            test_mode=True,
        )


def test_publication_delong_is_plan_bound_and_rejects_external_or_repeated() -> None:
    import covid_audio_btp.hst_publication as publication

    left, _ = _primary_prediction_pair()
    right = _authenticated(
        left.frame.assign(
            source_family="comparator",
            probability=[0.2, 0.3, 0.7, 0.8, 0.4, 0.5, 0.6, 0.9],
        ),
        "delong-right",
    )
    result = publication.build_paired_delong_evidence(
        publication.PublicationComparison(
            estimand_id=publication.PRIMARY_ESTIMAND_ID,
            left=left,
            right=right,
        ),
        analysis_plan=_analysis_plan(),
        test_mode=True,
    )
    assert result.loc[0, "test_method"] == "paired_delong"
    assert result.loc[0, "n"] == 8


def test_primary_repeated_holdouts_skip_pooled_delong_with_audited_reason() -> None:
    import covid_audio_btp.hst_publication as publication

    base_left, base_right = _primary_prediction_pair()
    left = _authenticated(
        pd.concat(
            [base_left.frame.assign(fold=fold) for fold in range(10)],
            ignore_index=True,
        ),
        "primary-hst-10-fold",
    )
    right = _authenticated(
        pd.concat(
            [base_right.frame.assign(fold=fold) for fold in range(10)],
            ignore_index=True,
        ),
        "primary-comparator-10-fold",
    )

    result = publication.build_paired_delong_evidence(
        publication.PublicationComparison(
            estimand_id=publication.PRIMARY_ESTIMAND_ID,
            left=left,
            right=right,
        ),
        analysis_plan=_analysis_plan(),
        test_mode=True,
    )

    assert len(result) == 1
    assert bool(result.loc[0, "skipped"])
    assert result.loc[0, "fold_count"] == 10
    assert not bool(result.loc[0, "pooled_repeated_rows"])
    assert result.loc[0, "analysis_role"] == "descriptive"
    assert result.loc[0, "multiplicity_family"] == "not_applicable_descriptive_skip"
    assert np.isnan(result.loc[0, "p_value"])
    assert "one exact paired test set" in result.loc[0, "skip_reason"]
