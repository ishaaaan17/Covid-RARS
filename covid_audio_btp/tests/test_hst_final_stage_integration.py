from __future__ import annotations

import json
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


FINAL_STAGES = {
    "aligned_comparator",
    "fusion",
    "statistics",
    "gradcam",
    "evidence_pack",
}


def test_training_public_api_uses_frozen_evaluation_registry_name() -> None:
    from covid_audio_btp.hst_training import train_hst_fold

    parameters = inspect.signature(train_hst_fold).parameters
    assert "evaluation_registry_root" in parameters
    assert "project_evaluation_registry_root" not in parameters


def _sha256(path: Path) -> str:
    from covid_audio_btp.hst_runtime import stable_file_sha256

    return stable_file_sha256(path)


def _pipeline(tmp_path: Path) -> SimpleNamespace:
    from covid_audio_btp.hst_publication import PRIMARY_ESTIMAND_ID
    from covid_audio_btp.hst_reporting import REPORTING_CONTRACT

    workspace = tmp_path / "covid_audio_btp"
    workspace.mkdir()
    features = workspace / "features.csv"
    pd.DataFrame(
        {
            "dataset": ["coswara", "coswara"],
            "participant_key": ["coswara::p1", "coswara::p2"],
            "recording_key": ["coswara::r1", "coswara::r2"],
            "modality": ["cough", "cough"],
            "feature_1": [0.1, 0.2],
        }
    ).to_csv(features, index=False)
    config = SimpleNamespace(
        workspace_root=workspace,
        mode="full",
        scientific_config={
            "paths": {"compare_is10_features": "features.csv"},
            "comparator": {
                "selected_feature_count": 800,
                "ranker": "lightgbm",
                "selection_scope": "fold_training_only_per_modality_mean",
                "ensemble_cap": 5,
                "primary_endpoint": "validation_selected_candidate",
            },
            "reporting": {
                **REPORTING_CONTRACT,
                "primary_estimand_id": PRIMARY_ESTIMAND_ID,
            },
            "performance_objectives": {
                "metric": "participant_auroc",
                "engineering_targets_not_guarantees": True,
                "test_set_is_not_a_stopping_rule": True,
                "references": {
                    "cough": 0.868,
                    "breath": 0.842,
                    "speech": 0.891,
                    "cough_speech_fusion": 0.897,
                },
            },
        },
        accepted_hashes={
            "data_contracts_freeze": "a" * 64,
            "pilot_freeze": "b" * 64,
            "environment_lock": "c" * 64,
        },
        device="cpu",
        resume=True,
    )
    run_root = workspace / "data" / "outputs" / "hst" / "hst-test-run"
    run_root.mkdir(parents=True)
    return SimpleNamespace(config=config, run_root=run_root, run_id="hst-test-run")


def _write_stage_receipt(
    pipeline: SimpleNamespace,
    stage: str,
    outputs: list[Path],
) -> Path:
    from covid_audio_btp.hst_runtime import canonical_json_sha256

    relative = [path.resolve().relative_to(pipeline.run_root.resolve()).as_posix() for path in outputs]
    payload: dict[str, object] = {
        "schema_version": 1,
        "receipt_type": "hst_stage",
        "run_id": pipeline.run_id,
        "stage": stage,
        "status": "success",
        "output_paths": relative,
        "output_checksums": {
            value: _sha256(path) for value, path in zip(relative, outputs, strict=True)
        },
    }
    payload["record_hash"] = canonical_json_sha256(payload)
    path = pipeline.run_root / "runtime" / "stages" / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_runtime_figure_frame_preserves_unmeasured_gpu_memory_as_missing(
    tmp_path: Path,
) -> None:
    import covid_audio_btp.hst_stages as stages
    from covid_audio_btp.hst_runtime import canonical_json_sha256

    pipeline = _pipeline(tmp_path)
    output = pipeline.run_root / "artifacts" / "preflight.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}\n", encoding="utf-8")
    receipt_path = _write_stage_receipt(pipeline, "preflight", [output])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(
        {
            "started_at": "2026-08-02T00:00:00+00:00",
            "completed_at": "2026-08-02T00:00:05+00:00",
            "metadata": {
                "gpu_memory_measured": False,
                "peak_gpu_memory_allocated_mb": None,
                "peak_gpu_memory_reserved_mb": None,
            },
        }
    )
    receipt.pop("record_hash")
    receipt["record_hash"] = canonical_json_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    frame = stages._runtime_figure_frame(pipeline)

    assert not bool(frame.loc[0, "gpu_memory_measured"])
    assert pd.isna(frame.loc[0, "peak_gpu_memory_allocated_mb"])
    assert pd.isna(frame.loc[0, "peak_gpu_memory_reserved_mb"])
    assert pd.isna(frame.loc[0, "peak_gpu_memory_mb"])


def test_runtime_figure_frame_uses_resource_pilot_child_peak_measurement(
    tmp_path: Path,
) -> None:
    import covid_audio_btp.hst_stages as stages
    from covid_audio_btp.hst_runtime import canonical_json_sha256

    pipeline = _pipeline(tmp_path)
    output = pipeline.run_root / "audits" / "base_resource_pilot_freeze.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}\n", encoding="utf-8")
    receipt_path = _write_stage_receipt(
        pipeline, "base_resource_pilot", [output]
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(
        {
            "started_at": "2026-08-02T00:00:00+00:00",
            "completed_at": "2026-08-02T00:02:00+00:00",
            "metadata": {
                "gpu_memory_measured": False,
                "peak_gpu_memory_allocated_mb": None,
                "peak_gpu_memory_reserved_mb": None,
                "child_gpu_memory_measured": True,
                "child_peak_gpu_memory_allocated_mb": 2048.0,
                "child_peak_gpu_memory_reserved_mb": 3072.0,
                "gpu_memory_measurement_scope": "selected_resource_pilot_child_process",
            },
        }
    )
    receipt.pop("record_hash")
    receipt["record_hash"] = canonical_json_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    frame = stages._runtime_figure_frame(pipeline)
    row = frame.loc[frame["stage"].eq("base_resource_pilot")].iloc[0]

    assert bool(row["gpu_memory_measured"])
    assert row["peak_gpu_memory_allocated_mb"] == pytest.approx(2048.0)
    assert row["peak_gpu_memory_reserved_mb"] == pytest.approx(3072.0)


def _manifest_frame(*, include_target_train: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, dataset in (
        ("train", "coswara"),
        ("validation", "coswara"),
        ("test", "coswara"),
    ):
        rows.append(
            {
                "protocol": "hst_literature_aligned_repeated_holdout",
                "fold": 1,
                "cohort": "joint_complete_case",
                "manifest_sha256": "d" * 64,
                "dataset": dataset,
                "participant_key": f"{dataset}::p-{split}",
                "recording_key": f"{dataset}::r-{split}",
                "split": split,
                "modality": "cough",
                "label_binary": "negative" if split != "test" else "positive",
            }
        )
    if include_target_train:
        rows.append(
            {
                **rows[0],
                "dataset": "coughvid",
                "participant_key": "coughvid::p-target",
                "recording_key": "coughvid::r-target",
            }
        )
    return pd.DataFrame(rows)


def _minimal_compare_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["coswara", "coswara"],
            "participant_key": ["coswara::p1", "coswara::p2"],
            "recording_key": ["coswara::r1", "coswara::r2"],
            "modality": ["cough", "cough"],
            "feature_1": [0.1, 0.2],
        }
    )


def _fusion_predictions(source_family: str) -> pd.DataFrame:
    import covid_audio_btp.hst_fusion as fusion

    rows: list[dict[str, object]] = []
    for split, prefix in (("validation", "v"), ("test", "t")):
        for index, label in enumerate(("negative", "negative", "positive", "positive")):
            for modality in ("cough", "speech"):
                recording = f"coswara::{prefix}{index}-{modality}"
                rows.append(
                    {
                        "run_id": "hst-test-run",
                        "protocol": "hst_literature_aligned_repeated_holdout",
                        "fold": 1,
                        "dataset": "coswara",
                        "participant_key": f"coswara::{prefix}{index}",
                        "cohort": "joint_complete_case",
                        "split": split,
                        "recording_key": recording,
                        "audio_content_sha256": __import__("hashlib").sha256(
                            recording.encode("ascii")
                        ).hexdigest(),
                        "eligible": True,
                        "manifest_sha256": "d" * 64,
                        "recording_intersection_sha256": "0" * 64,
                        "modality": modality,
                        "label_binary": label,
                        "probability": 0.15 + 0.2 * index,
                        "source_family": source_family,
                        "model": (
                            "hst_base"
                            if source_family == "hst"
                            else "validation_selected_candidate"
                        ),
                        "checkpoint_hash": ("e" if source_family == "hst" else "f") * 64,
                        "representation": "paper_logmel_224" if source_family == "hst" else "compare_is10_top800",
                        "feature_artifact_sha256": ("1" if source_family == "hst" else "2") * 64,
                        "feature_approval_id": "hst-data-contract" if source_family == "hst" else "approved-comparator",
                        "preprocessing_sha256": ("3" if source_family == "hst" else "4") * 64,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["recording_intersection_sha256"] = fusion._recording_intersection_hash(frame)
    return fusion._validate_predictions(frame, name=f"{source_family} fixture")


def test_final_scientific_stages_are_all_wired() -> None:
    import covid_audio_btp.hst_stages as stages

    assert FINAL_STAGES <= set(stages._IMPLEMENTED_HANDLERS)
    assert all(
        stages._IMPLEMENTED_HANDLERS[name] is not stages._unwired_scientific_stage
        for name in FINAL_STAGES
    )


def test_statistics_plan_freezes_one_primary_and_declared_secondary_families() -> None:
    import covid_audio_btp.hst_stages as stages
    from covid_audio_btp.hst_publication import PRIMARY_ESTIMAND_ID

    plan = stages._frozen_publication_analysis_plan()

    primary = plan.loc[plan["analysis_role"].eq("primary")]
    assert primary["estimand_id"].tolist() == [PRIMARY_ESTIMAND_ID]
    assert primary["complete_case"].tolist() == [True]
    assert primary["fusion_method"].tolist() == ["uniform_mean"]
    assert set(plan.loc[plan["analysis_role"].eq("secondary"), "multiplicity_family"])


def test_statistics_routes_every_reporting_endpoint_through_source_platt() -> None:
    import inspect
    import covid_audio_btp.hst_stages as stages

    source = inspect.getsource(stages._statistics)
    for series in (
        '"internal_hst"',
        '"aligned_comparator"',
        '"temporal_hst"',
        '"external_hst"',
    ):
        assert series in source
    assert "derive_source_platt_calibrated_pair" in source
    assert 'ensemble_right=calibrated_evaluations["external_hst"]' in source
    assert '"source_platt_calibration_audit.csv"' in source
    assert 'f"{series}_platt"' in source
    assert 'f"{series}_raw"' in source
    assert "calibrated_validations[series]" in source
    assert "calibrated_evaluations[series]" in source


def test_prepare_fusion_source_restricts_comparator_to_manifest_protocol(
    tmp_path: Path,
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path)
    comparator = _fusion_predictions("comparator")
    comparator["feature_contract_hash"] = "5" * 64
    comparator["approval_id"] = "approved-comparator"
    unrelated = comparator.copy()
    unrelated["protocol"] = "unrelated_temporal_protocol"
    secondary = comparator.copy()
    secondary["model"] = "top_4_validation_ensemble"
    raw = pd.concat([comparator, unrelated, secondary], ignore_index=True)
    manifest = comparator[
        [
            "protocol",
            "fold",
            "dataset",
            "participant_key",
            "recording_key",
            "split",
            "modality",
            "label_binary",
            "cohort",
            "manifest_sha256",
            "audio_content_sha256",
        ]
    ].copy()

    selected = stages._prepare_fusion_source(
        raw,
        manifest,
        pipeline=pipeline,
        source_family="comparator",
    )

    assert selected["protocol"].unique().tolist() == [
        "hst_literature_aligned_repeated_holdout"
    ]
    assert len(selected) == len(manifest)
    assert selected["model"].unique().tolist() == ["validation_selected_candidate"]


def test_estimand_execution_audit_requires_every_frozen_estimand() -> None:
    import covid_audio_btp.hst_stages as stages

    plan = stages._frozen_publication_analysis_plan()
    complete = pd.DataFrame(
        {
            "estimand_id": plan["estimand_id"],
            "skipped": False,
            "skip_reason": "",
        }
    )

    audit = stages._build_estimand_execution_audit(plan, complete)
    assert audit["executed_or_explicitly_skipped"].all()
    assert set(audit["estimand_id"]) == set(plan["estimand_id"])

    with pytest.raises(ValueError, match="Frozen estimands were not executed"):
        stages._build_estimand_execution_audit(  # type: ignore[attr-defined]
            plan,
            complete.iloc[:-1].copy(),
        )


def test_temporal_publication_table_applies_complete_case_uniform_fusion() -> None:
    import covid_audio_btp.hst_stages as stages
    from covid_audio_btp.hst_publication import authenticate_table, bind_analysis_plan

    plan_frame = stages._frozen_publication_analysis_plan()
    plan_table = authenticate_table(
        plan_frame,
        source_name="plan",
        manifest_sha256="a" * 64,
        test_mode=True,
    )
    plan = bind_analysis_plan(plan_table, test_mode=True)
    rows: list[dict[str, object]] = []
    for participant, label, cough, speech in (
        ("p1", "negative", 0.1, 0.3),
        ("p2", "negative", 0.2, 0.4),
        ("p3", "positive", 0.7, 0.9),
        ("p4", "positive", 0.8, 1.0),
    ):
        for modality, probability in (("cough", cough), ("speech", speech)):
            rows.append(
                {
                    "fold": 1,
                    "participant_key": participant,
                    "dataset": "coswara",
                    "split": "test",
                    "protocol": "early_to_late",
                    "cohort": "joint_complete_case",
                    "manifest_sha256": "b" * 64,
                    "label_binary": label,
                    "modality": modality,
                    "model": "hst_base",
                    "probability": probability,
                }
            )
    branches = authenticate_table(
        pd.DataFrame(rows),
        source_name="temporal-branches",
        manifest_sha256="b" * 64,
        test_mode=True,
    )

    fused = stages._publication_uniform_cough_speech_table(
        branches,
        source_name="temporal-fusion",
        analysis_plan=plan,
    )

    assert len(fused.frame) == 4
    assert fused.frame["complete_case"].all()
    assert fused.frame["fusion_method"].eq("uniform_mean").all()
    assert fused.frame.set_index("participant_key").loc["p1", "probability"] == pytest.approx(
        0.2
    )


def test_gradcam_selection_uses_one_frozen_context_and_deterministic_cells() -> None:
    import covid_audio_btp.hst_stages as stages

    rows = []
    cases = [
        ("tp", "positive", 0.9),
        ("tn", "negative", 0.1),
        ("fp", "negative", 0.8),
        ("fn", "positive", 0.2),
    ]
    for key, label, probability in cases:
        rows.append(
            {
                "participant_key": f"coswara::{key}",
                "recording_key": f"coswara::{key}-recording",
                "label_binary": label,
                "probability": probability,
                "split": "test",
                "fold": 1,
                "protocol": "hst_literature_aligned_repeated_holdout",
                "modality": "cough",
            }
        )
    rows.append(
        {
            **rows[0],
            "participant_key": "coswara::other-fold",
            "recording_key": "coswara::other-fold-recording",
            "fold": 2,
            "probability": 0.999,
        }
    )

    selected = stages._select_frozen_gradcam_examples(
        pd.DataFrame(rows), threshold=0.5
    )

    assert selected["outcome"].tolist() == ["FN", "FP", "TN", "TP"]
    assert selected["fold"].eq(1).all()
    assert "coswara::other-fold" not in set(selected["participant_key"])


def test_gradcam_missing_cells_are_audited_without_changing_frozen_fold() -> None:
    import covid_audio_btp.hst_stages as stages

    predictions = pd.DataFrame(
        [
            {
                "participant_key": "coswara::positive",
                "recording_key": "coswara::positive-1",
                "label_binary": "positive",
                "probability": 0.9,
                "split": "test",
                "fold": 1,
                "protocol": "hst_literature_aligned_repeated_holdout",
                "modality": "cough",
            },
            {
                "participant_key": "coswara::negative",
                "recording_key": "coswara::negative-1",
                "label_binary": "negative",
                "probability": 0.1,
                "split": "test",
                "fold": 1,
                "protocol": "hst_literature_aligned_repeated_holdout",
                "modality": "cough",
            },
        ]
    )

    context = stages._frozen_gradcam_context(predictions, threshold=0.5)
    selected = stages._select_frozen_gradcam_examples(predictions, threshold=0.5)
    audit = stages._gradcam_showcase_cell_audit(context, selected)

    assert selected["outcome"].tolist() == ["TN", "TP"]
    assert audit["outcome"].tolist() == ["TP", "TN", "FP", "FN"]
    assert audit.set_index("outcome").loc["FP", "available"] == False  # noqa: E712
    assert audit.set_index("outcome").loc["FN", "available"] == False  # noqa: E712
    assert audit["frozen_fold"].eq(1).all()


def test_gradcam_group_rows_use_all_correct_rows_and_cluster_by_participant() -> None:
    import numpy as np
    import covid_audio_btp.hst_stages as stages
    from covid_audio_btp.hst_gradcam import build_participant_gradcam_summary

    rows: list[dict[str, object]] = []
    for label, probability, participant, recordings in (
        ("positive", 0.9, "coswara::p1", ("p1-a", "p1-b")),
        ("positive", 0.8, "coswara::p2", ("p2-a",)),
        ("negative", 0.1, "coswara::n1", ("n1-a", "n1-b")),
        ("negative", 0.2, "coswara::n2", ("n2-a",)),
    ):
        for recording in recordings:
            rows.append(
                {
                    "participant_key": participant,
                    "recording_key": f"coswara::{recording}",
                    "label_binary": label,
                    "probability": probability,
                    "split": "test",
                    "fold": 1,
                    "protocol": "hst_literature_aligned_repeated_holdout",
                    "modality": "cough",
                }
            )
    # These errors may be showcased but must not enter the correctly classified summary.
    rows.extend(
        [
            {**rows[0], "participant_key": "coswara::fn", "recording_key": "coswara::fn", "probability": 0.1},
            {**rows[-1], "participant_key": "coswara::fp", "recording_key": "coswara::fp", "probability": 0.9},
        ]
    )
    predictions = pd.DataFrame(rows)

    showcase = stages._select_frozen_gradcam_examples(predictions, threshold=0.5)
    group = stages._frozen_gradcam_group_rows(predictions, threshold=0.5)

    assert len(showcase) == 4
    assert len(group) == 6
    assert set(group["outcome"]) == {"TP", "TN"}
    heatmaps = group.copy()
    heatmaps["heatmap"] = [np.full((2, 2), index / 10.0) for index in range(1, 7)]
    summary = build_participant_gradcam_summary(
        heatmaps,
        bootstrap_replicates=10,
        seed=42,
    )
    assert len(summary.participant_heatmaps) == 4
    assert sorted(summary.participant_heatmaps["n_recordings"].tolist()) == [1, 1, 2, 2]


def test_engineering_objectives_are_descriptive_post_selection_audits() -> None:
    import covid_audio_btp.hst_stages as stages

    branch_rows: list[dict[str, object]] = []
    fusion_rows: list[dict[str, object]] = []
    for fold in range(1, 11):
        for index, label in enumerate(("negative", "negative", "positive", "positive")):
            for modality in ("cough", "breath", "speech"):
                branch_rows.append(
                    {
                        "fold": fold,
                        "participant_key": f"coswara::{fold}-{index}-{modality}",
                        "label_binary": label,
                        "probability": (0.1, 0.2, 0.8, 0.9)[index],
                        "modality": modality,
                        "split": "test",
                        "protocol": "hst_literature_aligned_repeated_holdout",
                    }
                )
            fusion_rows.append(
                {
                    "fold": fold,
                    "participant_key": f"coswara::{fold}-{index}",
                    "label_binary": label,
                    "probability": (0.1, 0.2, 0.8, 0.9)[index],
                    "split": "test",
                    "protocol": "hst_literature_aligned_repeated_holdout",
                    "fusion_method": "uniform_mean",
                    "modality_combination": "cough+speech",
                    "complete_case": True,
                }
            )

    audit = stages._build_engineering_objective_audit(
        pd.DataFrame(branch_rows),
        pd.DataFrame(fusion_rows),
    )

    assert audit["branch"].tolist() == ["cough", "breath", "speech", "cough_speech_fusion"]
    assert audit["reference_auroc"].tolist() == [0.868, 0.842, 0.891, 0.897]
    assert audit["observed_auroc"].eq(1.0).all()
    assert audit["achieved"].all()
    assert audit["targets_not_selection_rules"].eq(True).all()  # noqa: E712
    assert audit["generated_after_model_selection"].eq(True).all()  # noqa: E712
    assert audit["test_set_is_not_a_stopping_rule"].eq(True).all()  # noqa: E712


def test_statistics_fails_closed_without_receipted_upstream_tables(tmp_path: Path) -> None:
    import covid_audio_btp.hst_stages as stages

    with pytest.raises(FileNotFoundError, match="receipt"):
        stages._statistics(_pipeline(tmp_path), "statistics")


def test_aligned_comparator_rejects_target_rows_before_any_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path)
    monkeypatch.setattr(
        stages,
        "_load_indexed_manifest",
        lambda *_args: (tmp_path / "internal.csv", _manifest_frame(include_target_train=True), "d" * 64),
    )
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("target data entered comparator fitting")

    monkeypatch.setattr(stages, "run_aligned_compare_is10", forbidden, raising=False)
    with pytest.raises(ValueError, match="COUGHVID|source-only|target"):
        stages._aligned_comparator(pipeline, "aligned_comparator")
    assert called is False


def test_aligned_comparator_cannot_create_or_self_approve_freezes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path)
    monkeypatch.setattr(
        stages,
        "_load_indexed_manifest",
        lambda *_args: (
            tmp_path / "aligned_comparator.csv",
            _manifest_frame(),
            "d" * 64,
        ),
    )
    monkeypatch.setattr(stages, "_source_only_comparator_manifest", lambda _frame: None)
    with pytest.raises(FileNotFoundError, match="Canonical comparator"):
        stages._aligned_comparator(pipeline, "aligned_comparator")
    assert not list(pipeline.run_root.rglob("*.approved.json"))


def _write_comparator_trust_files(pipeline: SimpleNamespace) -> tuple[Path, Path]:
    config_root = pipeline.config.workspace_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    approval = config_root / "hst_compare_is10_approval.approved.json"
    accepted = config_root / "hst_comparator_accepted_freezes.approved.json"
    approval.write_text('{"approval":"frozen"}\n', encoding="ascii")
    accepted.write_text(
        '{"accepted_generation_manifests":{}}\n', encoding="ascii"
    )
    return approval, accepted


def test_comparator_generation_acceptance_does_not_change_run_identity(
    tmp_path: Path,
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path)
    _approval, accepted = _write_comparator_trust_files(pipeline)
    before = pipeline.run_id
    first = stages._comparator_trust_inputs(pipeline)

    accepted.write_text(
        '{"accepted_generation_manifests":{"generation-1":"'
        + "a" * 64
        + '"}}\n',
        encoding="ascii",
    )
    second = stages._comparator_trust_inputs(pipeline)

    assert pipeline.run_id == before
    assert first["accepted_freezes_sha256"] != second["accepted_freezes_sha256"]
    assert set(pipeline.config.accepted_hashes) == {
        "data_contracts_freeze",
        "pilot_freeze",
        "environment_lock",
    }


def test_comparator_first_call_generates_then_requires_manual_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_stages as stages
    from covid_audio_btp.hst_comparators import compare_is10_feature_artifact_sha256

    pipeline = _pipeline(tmp_path)
    _write_comparator_trust_files(pipeline)
    manifest = _manifest_frame()
    monkeypatch.setattr(
        stages,
        "_load_indexed_manifest",
        lambda *_args: (tmp_path / "aligned.csv", manifest, "d" * 64),
    )
    monkeypatch.setattr(stages, "_source_only_comparator_manifest", lambda _frame: None)
    features = pd.DataFrame(
        {
            "dataset": ["coswara", "coswara"],
            "participant_key": ["coswara::p1", "coswara::p2"],
            "recording_key": ["coswara::r1", "coswara::r2"],
            "modality": ["cough", "cough"],
            "feature_1": [0.1, 0.2],
        }
    )
    monkeypatch.setattr(
        stages,
        "_complete_comparator_features",
        lambda *_args: (
            features,
            SimpleNamespace(schema_sha256="f" * 64),
            pipeline.config.workspace_root / "features.csv",
        ),
    )
    approval_arguments: dict[str, object] = {}

    def approve(*_args: object, **kwargs: object) -> dict[str, object]:
        approval_arguments.update(kwargs)
        return {}

    monkeypatch.setattr(stages, "load_frozen_compare_is10_approval", approve)

    def generate(*_args: object, **kwargs: object) -> object:
        audit_root = Path(str(kwargs["audit_dir"]))
        generation = audit_root / "generations" / "generation-1"
        generation.mkdir(parents=True)
        generation_manifest = generation / "manifest.json"
        generation_manifest.write_text(
            json.dumps({"generation_id": "generation-1", "files": {}}),
            encoding="ascii",
        )
        current = {
            "generation_id": "generation-1",
            "generation_manifest_sha256": _sha256(generation_manifest),
        }
        from covid_audio_btp.hst_runtime import canonical_json_sha256

        current["receipt_sha256"] = canonical_json_sha256(current)
        (audit_root / "current.json").write_text(
            json.dumps(current, sort_keys=True), encoding="ascii"
        )
        return object()

    monkeypatch.setattr(stages, "run_aligned_compare_is10", generate)
    monkeypatch.setattr(
        stages,
        "_verify_comparator_generation",
        lambda audit_root: (
            audit_root / "current.json",
            audit_root / "generations" / "generation-1" / "manifest.json",
            {"generation_id": "generation-1"},
            {"files": {}},
        ),
    )

    with pytest.raises(
        stages.ManualComparatorGenerationAcceptanceRequired,
        match="manual.*generation.*acceptance",
    ):
        stages._aligned_comparator(pipeline, "aligned_comparator")

    audit = pipeline.run_root / "scientific" / "aligned_comparator" / "audit"
    assert (audit / "current.json").is_file()
    assert not (pipeline.run_root / "scientific" / "aligned_comparator" / "metrics.csv").exists()
    assert approval_arguments["feature_artifact_sha256"] == (
        compare_is10_feature_artifact_sha256(features)
    )
    assert approval_arguments["feature_artifact_sha256"] != _sha256(
        pipeline.config.workspace_root / "features.csv"
    )


def test_comparator_second_call_reuses_and_authenticates_accepted_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path)
    _write_comparator_trust_files(pipeline)
    manifest = _manifest_frame()
    monkeypatch.setattr(
        stages,
        "_load_indexed_manifest",
        lambda *_args: (tmp_path / "aligned.csv", manifest, "d" * 64),
    )
    monkeypatch.setattr(stages, "_source_only_comparator_manifest", lambda _frame: None)
    monkeypatch.setattr(
        stages,
        "_complete_comparator_features",
        lambda *_args: (
            _minimal_compare_features(),
            SimpleNamespace(schema_sha256="f" * 64),
            pipeline.config.workspace_root / "features.csv",
        ),
    )
    monkeypatch.setattr(stages, "load_frozen_compare_is10_approval", lambda *_args, **_kwargs: {})
    audit = pipeline.run_root / "scientific" / "aligned_comparator" / "audit"
    current = audit / "current.json"
    generation_manifest = audit / "generations" / "generation-1" / "manifest.json"
    generation_manifest.parent.mkdir(parents=True)
    generation_manifest.write_text("{}", encoding="ascii")
    generation_prediction = generation_manifest.parent / "comparator_predictions.csv"
    generation_prediction.write_text("x\n1\n", encoding="ascii")
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("{}", encoding="ascii")
    monkeypatch.setattr(
        stages,
        "_verify_comparator_generation",
        lambda _root: (
            current,
            generation_manifest,
            {"generation_id": "generation-1"},
            {"files": {"comparator_predictions.csv": {}}},
        ),
    )
    authenticated = {
        "recording_predictions": pd.DataFrame({"x": [1]}),
        "participant_predictions": pd.DataFrame({"x": [1]}),
        "metrics": pd.DataFrame({"model_name": ["m"], "auroc": [0.5], "analysis_scope": ["confirmatory"]}),
        "feature_selection": pd.DataFrame({"x": [1]}),
        "model_audit": pd.DataFrame({"x": [1]}),
        "alignment_audit": pd.DataFrame({"x": [1]}),
        "candidate_selection": pd.DataFrame({"x": [1]}),
    }
    monkeypatch.setattr(
        stages,
        "_authenticate_existing_comparator_generation",
        lambda **_kwargs: (authenticated, 7),
    )
    monkeypatch.setattr(
        stages,
        "run_aligned_compare_is10",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("accepted generation must not retrain")
        ),
    )

    result = stages._aligned_comparator(pipeline, "aligned_comparator")

    assert result["metadata"]["generation_reused"] is True
    assert result["metadata"]["authenticated_model_bundles"] == 7
    assert (pipeline.run_root / "scientific" / "aligned_comparator" / "metrics.csv").is_file()
    assert generation_prediction.resolve() in {
        Path(path).resolve() for path in result["output_paths"]
    }


def test_comparator_tampering_fails_before_retraining_or_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path)
    _write_comparator_trust_files(pipeline)
    monkeypatch.setattr(
        stages,
        "_load_indexed_manifest",
        lambda *_args: (tmp_path / "aligned.csv", _manifest_frame(), "d" * 64),
    )
    monkeypatch.setattr(stages, "_source_only_comparator_manifest", lambda _frame: None)
    monkeypatch.setattr(
        stages,
        "_complete_comparator_features",
        lambda *_args: (
            _minimal_compare_features(),
            SimpleNamespace(schema_sha256="f" * 64),
            pipeline.config.workspace_root / "features.csv",
        ),
    )
    monkeypatch.setattr(stages, "load_frozen_compare_is10_approval", lambda *_args, **_kwargs: {})
    audit = pipeline.run_root / "scientific" / "aligned_comparator" / "audit"
    audit.mkdir(parents=True)
    (audit / "current.json").write_text("tampered", encoding="ascii")
    monkeypatch.setattr(
        stages,
        "run_aligned_compare_is10",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("tampered generation must not retrain")
        ),
    )

    with pytest.raises(ValueError, match="corrupt|checksum|tamper"):
        stages._aligned_comparator(pipeline, "aligned_comparator")
    assert not (pipeline.run_root / "scientific" / "aligned_comparator" / "metrics.csv").exists()


def test_fusion_registry_binding_is_derived_from_actual_stage_receipts(
    tmp_path: Path,
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path)
    manifest = pipeline.run_root / "manifests" / "internal.csv"
    manifest.parent.mkdir(parents=True)
    _manifest_frame().to_csv(manifest, index=False)
    (manifest.parent / "manifest_index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifests": {
                    "internal": {
                        "path": "manifests/internal.csv",
                        "sha256": _sha256(manifest),
                        "rows": len(_manifest_frame()),
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    hst_path = pipeline.run_root / "scientific" / "internal_cv" / "recording_predictions.csv"
    comparator_path = (
        pipeline.run_root / "scientific" / "aligned_comparator" / "recording_predictions.csv"
    )
    hst_path.parent.mkdir(parents=True)
    comparator_path.parent.mkdir(parents=True)
    hst = _fusion_predictions("hst")
    comparator = _fusion_predictions("comparator")
    hst.to_csv(hst_path, index=False)
    comparator.to_csv(comparator_path, index=False)
    manifest_receipt = _write_stage_receipt(pipeline, "manifests", [manifest])
    hst_receipt = _write_stage_receipt(pipeline, "internal_cv", [hst_path])
    comparator_current = pipeline.run_root / "scientific" / "aligned_comparator" / "current.json"
    comparator_current.write_text(
        json.dumps(
            {
                "generation_id": "generation-1",
                "generation_manifest_sha256": "5" * 64,
                "receipt_sha256": "6" * 64,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    comparator_receipt = _write_stage_receipt(
        pipeline, "aligned_comparator", [comparator_path, comparator_current]
    )

    binding, receipt = stages._build_authenticated_fusion_binding(
        pipeline,
        hst,
        comparator,
        manifest_name="internal",
        comparator_current_path=comparator_current,
    )

    verified = binding.verified_receipt()
    assert verified == receipt
    assert verified["registry_authority"] == "covid_audio_btp.HSTPipeline.receipt_chain"
    context = verified["contexts"][0]
    assert context["manifest_receipt"]["receipt_sha256"] == _sha256(manifest_receipt)
    assert context["comparator"]["generation_receipt_sha256"] == _sha256(
        comparator_current
    )
    assert verified["receipt_id"] == stages.canonical_json_sha256(
        {
            "manifests": _sha256(manifest_receipt),
            "internal_cv": _sha256(hst_receipt),
            "aligned_comparator": _sha256(comparator_receipt),
        }
    )


def test_fusion_reads_only_authenticated_internal_track_a_jobs(tmp_path: Path) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path)
    stage_root = pipeline.run_root / "scientific" / "internal_cv"
    internal_job = stage_root / "jobs" / "internal"
    task2_job = stage_root / "jobs" / "task2"
    internal_job.mkdir(parents=True)
    task2_job.mkdir(parents=True)
    internal_predictions = internal_job / "recording_predictions.csv"
    task2_predictions = task2_job / "recording_predictions.csv"
    pd.DataFrame({"recording_key": ["internal"]}).to_csv(internal_predictions, index=False)
    pd.DataFrame({"recording_key": ["task2"]}).to_csv(task2_predictions, index=False)

    def job_receipt(job_root: Path, job_id: str, predictions: Path) -> Path:
        receipt = job_root / "job_receipt.json"
        payload = {
            "schema_version": 1,
            "receipt_type": "hst_scientific_job",
            "status": "success",
            "run_id": pipeline.run_id,
            "job_id": job_id,
            "job_spec_sha256": ("1" if job_id == "internal" else "2") * 64,
            "outputs": [
                {
                    "path": predictions.relative_to(stage_root).as_posix(),
                    "sha256": _sha256(predictions),
                    "size_bytes": predictions.stat().st_size,
                }
            ],
        }
        from covid_audio_btp.hst_runtime import canonical_json_sha256

        payload["record_hash"] = canonical_json_sha256(payload)
        receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return receipt

    internal_receipt = job_receipt(internal_job, "internal", internal_predictions)
    task2_receipt = job_receipt(task2_job, "task2", task2_predictions)
    source_index = stage_root / "source_checkpoints.csv"
    pd.DataFrame(
        [
            {
                "training_job_id": "internal",
                "training_job_spec_sha256": "1" * 64,
                "manifest_name": "internal",
                "source_job_receipt_path": internal_receipt.as_posix(),
                "source_job_receipt_sha256": _sha256(internal_receipt),
            },
            {
                "training_job_id": "task2",
                "training_job_spec_sha256": "2" * 64,
                "manifest_name": "task2_like_cough",
                "source_job_receipt_path": task2_receipt.as_posix(),
                "source_job_receipt_sha256": _sha256(task2_receipt),
            },
        ]
    ).to_csv(source_index, index=False)
    stage_receipt_path = _write_stage_receipt(
        pipeline,
        "internal_cv",
        [source_index, internal_receipt, internal_predictions, task2_receipt, task2_predictions],
    )
    stage_receipt = json.loads(stage_receipt_path.read_text(encoding="utf-8"))

    selected = stages._internal_track_a_recording_prediction_paths(
        pipeline,
        stage_receipt,
    )

    assert selected == [internal_predictions.resolve()]


@pytest.mark.parametrize("receipt_location", ["stale", "quarantine"])
def test_evidence_pack_excludes_a_stale_future_evidence_receipt(
    tmp_path: Path,
    receipt_location: str,
) -> None:
    from covid_audio_btp.hst_reliability import HSTPipeline
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path)
    for stage in HSTPipeline.STAGES[:-1]:
        if stage == "statistics":
            artifact = (
                pipeline.run_root
                / "scientific"
                / "statistics"
                / "tables"
                / "engineering_objective_audit.csv"
            )
            artifact.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "branch": ["cough", "breath", "speech", "cough_speech_fusion"],
                    "targets_not_selection_rules": [True] * 4,
                    "generated_after_model_selection": [True] * 4,
                    "test_set_is_not_a_stopping_rule": [True] * 4,
                }
            ).to_csv(artifact, index=False)
        else:
            artifact = pipeline.run_root / "artifacts" / f"{stage}.txt"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if stage != "statistics":
            artifact.write_text(stage, encoding="utf-8")
        _write_stage_receipt(pipeline, stage, [artifact])
    stale = pipeline.run_root / "artifacts" / "stale-evidence.txt"
    stale.write_text("stale", encoding="utf-8")
    stale_receipt = _write_stage_receipt(pipeline, "evidence_pack", [stale])
    if receipt_location == "quarantine":
        os.replace(
            stale_receipt,
            stale_receipt.parent / ".evidence_pack.previous",
        )

    result = stages._evidence_pack(pipeline, "evidence_pack")
    manifest_path = pipeline.run_root / "evidence" / "hst_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path in result["output_paths"]
    assert "evidence_pack" not in manifest["stages"]
    assert manifest["stage_count"] == len(HSTPipeline.STAGES) - 1
    engineering_artifact = next(
        item
        for item in manifest["artifacts"]
        if item["path"].endswith("/engineering_objective_audit.csv")
    )
    assert engineering_artifact["producer_stages"] == ["statistics"]
    assert result["metadata"]["engineering_targets_not_selection_rules"] is True

    _write_stage_receipt(pipeline, "evidence_pack", list(result["output_paths"]))
    from covid_audio_btp.hst_evidence import publish_hst_latest

    latest_path = pipeline.run_root.parent / "latest.json"
    latest = publish_hst_latest(
        run_root=pipeline.run_root,
        evidence_manifest_path=pipeline.run_root / "evidence" / "hst_evidence_manifest.json",
        latest_path=latest_path,
    )
    assert latest["evidence_manifest_path"] == manifest_path.relative_to(
        pipeline.run_root
    ).as_posix()
    assert latest["evidence_manifest_path_base"] == "run_root"
