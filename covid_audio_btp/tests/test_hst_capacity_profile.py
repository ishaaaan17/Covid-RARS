from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_capacity_notebook_is_clean_and_uses_the_reduced_config() -> None:
    path = PROJECT_ROOT / "notebooks" / "10_HST_CAPACITY_INTERNAL_FUSION.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )
    assert notebook["nbformat"] == 4
    assert "hst_capacity_internal_fusion.json" in source
    assert "RUN_CONFIRMATORY = False" in source
    assert "--mode', 'pilot'" in source
    assert "--through', 'evidence_pack'" in source
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_capacity_config_is_a_distinct_twenty_job_internal_fusion_contract() -> None:
    from covid_audio_btp.hst_reliability import (
        HSTCapacityInternalFusionPipeline,
        _read_scientific_config,
        pipeline_class_for_config,
    )
    from covid_audio_btp.hst_workloads import workload_profile_from_scientific_config

    config = _read_scientific_config(
        PROJECT_ROOT / "configs" / "hst_capacity_internal_fusion.json"
    )
    profile = workload_profile_from_scientific_config(config)

    assert profile.total_training_jobs == 20
    assert dict(profile.training_jobs_by_modality) == {"cough": 10, "speech": 10}
    assert dict(profile.training_jobs_by_stage) == {"internal_cv": 20}
    assert pipeline_class_for_config(config) is HSTCapacityInternalFusionPipeline
    assert "split_policy_contrast" not in HSTCapacityInternalFusionPipeline.STAGES
    assert "reverse_temporal" not in HSTCapacityInternalFusionPipeline.STAGES
    assert "external_transfer" not in HSTCapacityInternalFusionPipeline.STAGES
    assert "aligned_comparator" not in HSTCapacityInternalFusionPipeline.STAGES
    assert HSTCapacityInternalFusionPipeline.STAGES[-3:] == (
        "fusion",
        "gradcam",
        "evidence_pack",
    )
    assert set(config["performance_objectives"]["references"]) == {  # type: ignore[index]
        "cough",
        "speech",
        "cough_speech_fusion",
    }


def test_capacity_runtime_projection_is_bound_to_twenty_jobs() -> None:
    from covid_audio_btp.hst_acceptance import _validate_conservative_runtime_projection
    from covid_audio_btp.hst_resource_pilot import project_full_training_runtime
    from covid_audio_btp.hst_workloads import CAPACITY_INTERNAL_FUSION_PROFILE

    projection = project_full_training_runtime(
        workload_profile=CAPACITY_INTERNAL_FUSION_PROFILE,
        selected_trial_seconds=46.0,
        selected_trial_optimizer_updates=100,
        optimizer_updates_per_epoch_by_modality={"cough": 359, "speech": 359},
        planned_training_jobs_by_modality={"cough": 10, "speech": 10},
        confirmatory_epochs=100,
        end_to_end_overhead_multiplier=1.5,
        maximum_approved_runtime_hours=168.0,
    )

    assert projection["planned_training_jobs"] == 20
    assert projection["estimated_optimizer_updates"] == 718_000
    assert projection["within_approved_runtime_ceiling"] is True
    estimated, ceiling = _validate_conservative_runtime_projection(projection)
    assert estimated == pytest.approx(projection["estimated_serial_gpu_hours"])
    assert ceiling == 168.0


def test_capacity_workload_rejects_full_profile_modalities() -> None:
    from covid_audio_btp.hst_stages import _frozen_runtime_projection_workload
    from covid_audio_btp.hst_workloads import CAPACITY_INTERNAL_FUSION_PROFILE

    metadata = pd.DataFrame(
        {
            "dataset": ["coswara"] * 8,
            "participant_key": [f"p{index}" for index in range(8)],
            "label_binary": ["negative", "positive"] * 4,
            "modality": ["cough"] * 4 + ["speech"] * 4,
        }
    )
    seeds = (1, 2, 5, 12, 40, 52, 72, 2002, 4002, 6002)

    updates, jobs = _frozen_runtime_projection_workload(
        metadata,
        workload_profile=CAPACITY_INTERNAL_FUSION_PROFILE,
        project_seeds=seeds,
        primary_modalities=("cough", "speech"),
        secondary_modalities=(),
        effective_batch_size=8,
    )
    assert set(updates) == {"cough", "speech"}
    assert jobs == {"cough": 10, "speech": 10}

    with pytest.raises(ValueError, match="modalities"):
        _frozen_runtime_projection_workload(
            metadata,
            workload_profile=CAPACITY_INTERNAL_FUSION_PROFILE,
            project_seeds=seeds,
            primary_modalities=("cough", "speech"),
            secondary_modalities=("breath",),
            effective_batch_size=8,
        )


def test_capacity_full_preprocessing_is_coswara_cough_and_speech_only() -> None:
    from covid_audio_btp.hst_stages import _metadata_for_spectrogram_stage
    from covid_audio_btp.hst_workloads import CAPACITY_INTERNAL_FUSION_PROFILE

    metadata = pd.DataFrame(
        {
            "dataset": ["coswara", "coswara", "coswara", "coughvid"],
            "modality": ["cough", "speech", "breath", "cough"],
            "recording_key": ["c1", "s1", "b1", "e1"],
        }
    )

    selected = _metadata_for_spectrogram_stage(
        metadata,
        mode="full",
        workload_profile=CAPACITY_INTERNAL_FUSION_PROFILE,
    )

    assert set(selected["dataset"]) == {"coswara"}
    assert set(selected["modality"]) == {"cough", "speech"}
    assert set(selected["recording_key"]) == {"c1", "s1"}
