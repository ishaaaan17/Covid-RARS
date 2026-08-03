from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
import stat
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.hst_test_helpers import make_recording_predictions


def _write_checkpoint_marker(path: str) -> dict[str, object]:
    Path(path).write_text("executed", encoding="ascii")
    return {}


class _MaliciousCheckpointValue:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return _write_checkpoint_marker, (str(self.marker),)


def _sampler_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_rows = []
    manifest_rows = []
    participants = [
        ("n1", "negative", 3),
        ("n2", "negative", 1),
        ("n3", "negative", 1),
        ("p1", "positive", 4),
    ]
    for participant, label, recordings in participants:
        for index in range(recordings):
            recording = f"{participant}-r{index}"
            cache_rows.append(
                {
                    "dataset": "coswara",
                    "participant_key": f"coswara::{participant}",
                    "recording_key": f"coswara::{recording}",
                    "label_binary": label,
                    "modality": "cough",
                    "eligible": True,
                    "cache_path": f"/{recording}.npy",
                    "tensor_sha256": str(index) * 64,
                    "source_audio_sha256": hashlib.sha256(recording.encode()).hexdigest(),
                    "preprocessing_hash": "e" * 64,
                    "representation_id": "paper_logmel_224",
                }
            )
            manifest_rows.append(
                {
                    "fold": 1,
                    "training_seed": 1,
                    "protocol": "track_a",
                    "split": "train",
                    "dataset": "coswara",
                    "participant_key": f"coswara::{participant}",
                    "recording_key": f"coswara::{recording}",
                    "label_binary": label,
                    "modality": "cough",
                    "source_audio_sha256": hashlib.sha256(recording.encode()).hexdigest(),
                    "preprocessing_hash": "e" * 64,
                    "representation_id": "paper_logmel_224",
                }
            )
    return pd.DataFrame(cache_rows), pd.DataFrame(manifest_rows)


def test_participant_probabilities_do_not_weight_extra_recordings() -> None:
    from covid_audio_btp.hst_training import aggregate_recording_predictions

    participants = aggregate_recording_predictions(make_recording_predictions())
    value = participants.set_index("participant_key").loc["coswara::p1", "probability"]
    assert value == pytest.approx(0.8)


def test_hierarchical_sampler_balances_classes_not_recording_counts() -> None:
    from covid_audio_btp.hst_training import build_hierarchical_epoch_draw_plan

    cache, manifest = _sampler_inputs()
    plan = build_hierarchical_epoch_draw_plan(
        cache,
        manifest,
        fold=1,
        modality="cough",
        epoch=1,
        seed=1,
    )
    assert plan.groupby("label_binary").size().nunique() == 1
    assert plan["draw_id"].is_unique
    for _, class_draws in plan.groupby("label_binary"):
        per_person = class_draws.groupby("participant_key").size()
        assert per_person.max() - per_person.min() <= 1
    assert set(plan["split"]) == {"train"}


def test_hierarchical_sampler_is_deterministic_and_draws_recordings_uniformly() -> None:
    from covid_audio_btp.hst_training import build_hierarchical_epoch_draw_plan

    cache, manifest = _sampler_inputs()
    first = build_hierarchical_epoch_draw_plan(cache, manifest, fold=1, modality="cough", epoch=2, seed=52)
    second = build_hierarchical_epoch_draw_plan(cache, manifest, fold=1, modality="cough", epoch=2, seed=52)
    pd.testing.assert_frame_equal(first, second)
    assert first["augmentation_seed"].is_unique


def test_training_config_rejects_wrong_effective_batch_and_unfrozen_full_mode() -> None:
    from covid_audio_btp.hst_training import HSTTrainingConfig

    with pytest.raises(ValueError, match="effective batch"):
        HSTTrainingConfig(
            pilot_freeze_hash="pilot",
            data_contracts_freeze_hash="data",
            dependency_lock_hash="environment",
            accepted_environment_lock_hash="environment",
            physical_batch_size=4,
            gradient_accumulation=1,
            amp=False,
        )


def test_checkpoint_selection_contract_matches_primary_auroc_endpoint() -> None:
    from covid_audio_btp.hst_training import HSTTrainingConfig, validation_epoch_score

    config = HSTTrainingConfig(
        pilot_freeze_hash="pilot",
        data_contracts_freeze_hash="data",
        dependency_lock_hash="environment",
        accepted_environment_lock_hash="environment",
        physical_batch_size=4,
        gradient_accumulation=2,
        amp=False,
    )
    assert config.selection_metric == "participant_auroc"
    better_auroc = validation_epoch_score(
        {"auroc": 0.90, "auprc": 0.70, "nll": 0.8}, epoch=5
    )
    better_f1_but_lower_auroc = validation_epoch_score(
        {"auroc": 0.89, "auprc": 0.90, "nll": 0.2}, epoch=4
    )
    assert better_auroc > better_f1_but_lower_auroc

    same_metrics_earlier = validation_epoch_score(
        {"auroc": 0.90, "auprc": 0.70, "nll": 0.8}, epoch=4
    )
    assert same_metrics_earlier > better_auroc
    with pytest.raises(ValueError, match="100 epochs"):
        HSTTrainingConfig(
            pilot_freeze_hash="pilot",
            data_contracts_freeze_hash="data",
            dependency_lock_hash="environment",
            accepted_environment_lock_hash="environment",
            physical_batch_size=4,
            gradient_accumulation=2,
            amp=False,
            max_epochs=20,
            confirmatory=True,
        )


def test_prediction_context_rejects_placeholder_or_missing_provenance() -> None:
    from covid_audio_btp.hst_training import validate_prediction_context

    valid = {
        "run_id": "hst-abc123",
        "protocol": "repeated_participant_holdout",
        "model": "hst_base",
        "checkpoint_hash": "a" * 64,
        "representation": "paper_logmel_224",
        "architecture_sha256": "b" * 64,
        "executable_sha256": "c" * 64,
    }
    assert validate_prediction_context(valid) == valid

    for key in valid:
        invalid = dict(valid)
        invalid.pop(key)
        with pytest.raises(ValueError, match=key):
            validate_prediction_context(invalid)

    for key, value in (
        ("run_id", "pending_run"),
        ("protocol", "hst_pending_protocol"),
        ("checkpoint_hash", "pending_checkpoint"),
    ):
        invalid = dict(valid)
        invalid[key] = value
        with pytest.raises(ValueError, match=key):
            validate_prediction_context(invalid)


def test_training_fingerprint_binds_manifest_cache_and_scientific_config() -> None:
    from covid_audio_btp.hst_training import training_contract_fingerprint

    config = {
        "learning_rate": 1e-5,
        "max_epochs": 100,
        "physical_batch_size": 4,
    }
    first = training_contract_fingerprint(
        training_config=config,
        manifest_sha256="1" * 64,
        cache_index_sha256="2" * 64,
        source_checkpoint_sha256="3" * 64,
    )
    second = training_contract_fingerprint(
        training_config=config,
        manifest_sha256="1" * 64,
        cache_index_sha256="2" * 64,
        source_checkpoint_sha256="3" * 64,
    )
    assert first == second
    assert first != training_contract_fingerprint(
        training_config={**config, "learning_rate": 2e-5},
        manifest_sha256="1" * 64,
        cache_index_sha256="2" * 64,
        source_checkpoint_sha256="3" * 64,
    )

    with pytest.raises(ValueError, match="manifest_sha256"):
        training_contract_fingerprint(
            training_config=config,
            manifest_sha256="not-a-hash",
            cache_index_sha256="2" * 64,
            source_checkpoint_sha256="3" * 64,
        )


def test_resume_checkpoint_contract_mismatch_fails_closed() -> None:
    from covid_audio_btp.hst_training import validate_resume_checkpoint_contract

    payload = {"training_contract_fingerprint": "a" * 64}
    validate_resume_checkpoint_contract(payload, expected_fingerprint="a" * 64)
    with pytest.raises(ValueError, match="contract fingerprint"):
        validate_resume_checkpoint_contract(payload, expected_fingerprint="b" * 64)
    with pytest.raises(ValueError, match="contract fingerprint"):
        validate_resume_checkpoint_contract({}, expected_fingerprint="a" * 64)


def test_fold_result_has_frozen_test_predictions_field() -> None:
    from covid_audio_btp.hst_training import HSTFoldResult

    result = HSTFoldResult(
        last_epoch=2,
        best_epoch=1,
        resumed_from_epoch=None,
        validation_threshold=0.5,
    )
    assert isinstance(result.test_predictions, pd.DataFrame)


def _contract_inputs(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    participants = [
        ("train-n", "negative", "train"),
        ("train-p", "positive", "train"),
        ("val-n", "negative", "validation"),
        ("val-p", "positive", "validation"),
        ("test-n", "negative", "test"),
        ("test-p", "positive", "test"),
        ("ext-n", "negative", "external_test"),
        ("ext-p", "positive", "external_test"),
    ]
    for index, (participant, label, split) in enumerate(participants):
        array = np.full((224, 224), index / 10.0, dtype=np.float32)
        path = tmp_path / f"{participant}.npy"
        np.save(path, array, allow_pickle=False)
        tensor_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        row = {
            "fold": 1,
            "training_seed": 52,
            "protocol": "track_a",
            "split": split,
            "dataset": "coughvid" if split == "external_test" else "coswara",
            "participant_key": f"{'coughvid' if split == 'external_test' else 'coswara'}::{participant}",
            "recording_key": f"{'coughvid' if split == 'external_test' else 'coswara'}::{participant}-r0",
            "label_binary": label,
            "modality": "cough",
            "source_audio_sha256": hashlib.sha256(
                f"source::{participant}".encode()
            ).hexdigest(),
            "preprocessing_hash": "e" * 64,
            "representation_id": "paper_logmel_224",
        }
        manifest_rows.append(row)
        cache_rows.append(
            {
                **{key: value for key, value in row.items() if key not in {"fold", "split"}},
                "eligible": True,
                "cache_path": str(path),
                "tensor_sha256": tensor_hash,
            }
        )
    return pd.DataFrame(cache_rows), pd.DataFrame(manifest_rows)


def _scientific_training_claim() -> dict[str, object]:
    batch_hashes = ["4" * 64]
    batch_sequence_sha256 = hashlib.sha256(
        json.dumps(
            batch_hashes,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    epoch_schedule = {
        "schema_version": 1,
        "epoch": 1,
        "draw_plan_sha256": "1" * 64,
        "sample_count": 2,
        "batch_count": 1,
        "batch_identity_sha256": batch_hashes,
        "batch_sequence_sha256": batch_sequence_sha256,
        "optimizer_boundary_batch_indices": [1],
        "optimizer_boundary_count": 1,
        "physical_batch_size": 2,
        "gradient_accumulation": 2,
        "effective_batch_size": 4,
    }
    epoch_schedule["schedule_sha256"] = hashlib.sha256(
        json.dumps(
            epoch_schedule,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    schedule = {
        "schema_version": 1,
        "physical_batch_size": 2,
        "gradient_accumulation": 2,
        "effective_batch_size": 4,
        "epochs": [epoch_schedule],
    }
    schedule_sha256 = hashlib.sha256(
        json.dumps(
            schedule,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "data_contracts_freeze_hash": "data-contract-v1",
        "manifest_selection_sha256": "3" * 64,
        "training_schedule": schedule,
        "training_schedule_sha256": schedule_sha256,
    }


def _held_out_contract_row(
    tmp_path: Path,
    *,
    recording_id: str,
    label: str,
    split: str = "test",
) -> dict[str, object]:
    dataset = "coughvid" if split == "external_test" else "coswara"
    path = tmp_path / f"{recording_id}.npy"
    np.save(path, np.ones((224, 224), dtype=np.float32), allow_pickle=False)
    return {
        "dataset": dataset,
        "participant_key": f"{dataset}::{recording_id}",
        "recording_key": f"{dataset}::{recording_id}",
        "label_binary": label,
        "split": split,
        "fold": 1,
        "modality": "cough",
        "source_audio_sha256": hashlib.sha256(
            f"source::{recording_id}".encode()
        ).hexdigest(),
        "cache_path": str(path),
        "tensor_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "preprocessing_hash": "e" * 64,
        "representation_id": "paper_logmel_224",
    }


def _registry_test_case(
    tmp_path: Path,
) -> tuple[object, object, dict[str, object]]:
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from covid_audio_btp.hst_training import _collate, _model_architecture_sha256

    class TinyRegistryModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([0.0, 1.0]))

        def forward(self, values):
            return self.bias.unsqueeze(0).expand(values.shape[0], -1)

    row = _held_out_contract_row(
        tmp_path,
        recording_id="registry-r1",
        label="positive",
    )

    class FrozenRegistryDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.frame = pd.DataFrame([row])

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            if index != 0:
                raise IndexError(index)
            return torch.zeros((3, 4, 4)), torch.tensor(1), row

    model = TinyRegistryModel()
    loader = DataLoader(
        FrozenRegistryDataset(),
        batch_size=1,
        collate_fn=_collate,
    )
    context = {
        "run_id": "registry-source-run",
        "protocol": "track_a",
        "model": "tiny_registry",
        "checkpoint_hash": "a" * 64,
        "representation": "paper_logmel_224",
        "architecture_sha256": _model_architecture_sha256(model),
        "executable_sha256": "c" * 64,
    }
    return model, loader, context


def _run_registry_evaluation(
    model: object,
    loader: object,
    context: dict[str, object],
    *,
    tmp_path: Path,
    registry_root: Path,
    run_name: str,
) -> pd.DataFrame:
    from covid_audio_btp.hst_training import _evaluate_split_once

    return _evaluate_split_once(
        model,
        loader,
        split="test",
        fold=1,
        modality="cough",
        prediction_context={**context, "run_id": run_name},
        run_dir=tmp_path / run_name,
        project_registry_root=registry_root,
        training_contract="d" * 64,
        best_checkpoint_sha256="a" * 64,
        scientific_training_claim=_scientific_training_claim(),
    )


def test_manifest_cache_contract_is_exact_and_fail_closed(tmp_path: Path) -> None:
    from covid_audio_btp.hst_training import (
        build_hierarchical_epoch_draw_plan,
        validate_manifest_cache_contract,
    )

    cache, manifest = _contract_inputs(tmp_path)
    aligned = validate_manifest_cache_contract(cache, manifest, fold=1, modality="cough")
    assert len(aligned) == len(manifest)
    assert set(aligned["split"]) == {"train", "validation", "test", "external_test"}

    with pytest.raises(ValueError, match="exactly cover"):
        validate_manifest_cache_contract(cache.iloc[:-1], manifest, fold=1, modality="cough")

    extra = pd.concat(
        [
            cache,
            cache.iloc[[0]].assign(
                participant_key="coswara::extra",
                recording_key="coswara::extra-r0",
                source_audio_sha256="f" * 64,
                eligible=False,
            ),
        ],
        ignore_index=True,
    )
    aligned_with_global_extra = validate_manifest_cache_contract(
        extra,
        manifest,
        fold=1,
        modality="cough",
    )
    assert set(aligned_with_global_extra["recording_key"]) == set(
        manifest["recording_key"]
    )


def test_manifest_cache_contract_accepts_realistic_global_multidataset_cache(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_training import (
        build_hierarchical_epoch_draw_plan,
        validate_manifest_cache_contract,
    )

    cache, manifest = _contract_inputs(tmp_path)
    unrelated_rows: list[pd.Series] = []
    for index, (dataset, modality, eligible) in enumerate(
        [
            ("coswara", "breath", True),
            ("coswara", "speech", False),
            ("coughvid", "cough", False),
        ]
    ):
        row = cache.iloc[0].copy()
        row["dataset"] = dataset
        row["modality"] = modality
        row["eligible"] = eligible
        row["participant_key"] = f"{dataset}::global-extra-{index}"
        row["recording_key"] = f"{dataset}::global-extra-{index}-r0"
        row["source_audio_sha256"] = f"{index + 7:x}" * 64
        row["tensor_sha256"] = f"{index + 4:x}" * 64
        row["cache_path"] = str(tmp_path / f"global-extra-{index}.npy")
        unrelated_rows.append(row)
    global_cache = pd.concat(
        [cache, pd.DataFrame(unrelated_rows)],
        ignore_index=True,
    )

    aligned = validate_manifest_cache_contract(
        global_cache,
        manifest,
        fold=1,
        modality="cough",
    )

    assert len(aligned) == len(manifest)
    assert set(aligned["recording_key"]) == set(manifest["recording_key"])
    draw_plan = build_hierarchical_epoch_draw_plan(
        global_cache,
        manifest,
        fold=1,
        modality="cough",
        epoch=1,
        seed=52,
    )
    assert set(draw_plan["split"]) == {"train"}


def test_manifest_schema_validation_does_not_open_selected_tensor_files(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_training import validate_manifest_cache_contract

    cache, manifest = _contract_inputs(tmp_path)
    for path_value in cache["cache_path"]:
        Path(str(path_value)).unlink()

    aligned = validate_manifest_cache_contract(
        cache,
        manifest,
        fold=1,
        modality="cough",
    )

    assert len(aligned) == len(manifest)


def test_manifest_cache_contract_ignores_unrelated_duplicate_global_rows(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_training import (
        build_hierarchical_epoch_draw_plan,
        validate_manifest_cache_contract,
    )

    cache, manifest = _contract_inputs(tmp_path)
    unrelated = cache.iloc[[0]].assign(
        participant_key="coswara::unrelated",
        recording_key="coswara::unrelated-r0",
        label_binary="negative",
        modality="speech",
        source_audio_sha256="9" * 64,
    )
    global_cache = pd.concat([cache, unrelated, unrelated], ignore_index=True)

    aligned = validate_manifest_cache_contract(
        global_cache,
        manifest,
        fold=1,
        modality="cough",
    )

    assert set(aligned["recording_key"]) == set(manifest["recording_key"])
    draw_plan = build_hierarchical_epoch_draw_plan(
        global_cache,
        manifest,
        fold=1,
        modality="cough",
        epoch=1,
        seed=52,
    )
    assert set(draw_plan["split"]) == {"train"}


def test_manifest_cache_contract_ignores_unrelated_invalid_label_global_row(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_training import (
        build_hierarchical_epoch_draw_plan,
        validate_manifest_cache_contract,
    )

    cache, manifest = _contract_inputs(tmp_path)
    unrelated = cache.iloc[[0]].assign(
        participant_key="coughvid::unrelated",
        recording_key="coughvid::unrelated-r0",
        label_binary="not-a-study-label",
        modality="cough",
        eligible=False,
        source_audio_sha256="8" * 64,
    )

    global_cache = pd.concat([cache, unrelated], ignore_index=True)
    aligned = validate_manifest_cache_contract(
        global_cache,
        manifest,
        fold=1,
        modality="cough",
    )

    assert set(aligned["recording_key"]) == set(manifest["recording_key"])
    draw_plan = build_hierarchical_epoch_draw_plan(
        global_cache,
        manifest,
        fold=1,
        modality="cough",
        epoch=1,
        seed=52,
    )
    assert set(draw_plan["split"]) == {"train"}


@pytest.mark.parametrize("invalid", ["False", "yes", 2, None])
def test_manifest_cache_contract_rejects_ambiguous_eligibility(
    tmp_path: Path, invalid: object
) -> None:
    from covid_audio_btp.hst_training import validate_manifest_cache_contract

    cache, manifest = _contract_inputs(tmp_path)
    cache["eligible"] = cache["eligible"].astype(object)
    cache.loc[0, "eligible"] = invalid
    with pytest.raises(ValueError, match="eligible"):
        validate_manifest_cache_contract(cache, manifest, fold=1, modality="cough")


def test_manifest_cache_contract_rejects_label_or_split_leakage(tmp_path: Path) -> None:
    from covid_audio_btp.hst_training import validate_manifest_cache_contract

    cache, manifest = _contract_inputs(tmp_path)
    bad_label = manifest.copy()
    bad_label.loc[0, "label_binary"] = "unknown"
    with pytest.raises(ValueError, match="labels"):
        validate_manifest_cache_contract(cache, bad_label, fold=1, modality="cough")

    overlap = manifest.copy()
    changed_recordings = overlap.loc[
        overlap["participant_key"].eq("coswara::val-n"), "recording_key"
    ].tolist()
    overlap.loc[
        overlap["participant_key"].eq("coswara::val-n"), "participant_key"
    ] = "coswara::train-n"
    overlap_cache = cache.copy()
    overlap_cache.loc[
        overlap_cache["recording_key"].isin(changed_recordings), "participant_key"
    ] = "coswara::train-n"
    with pytest.raises(ValueError, match="participant overlap"):
        validate_manifest_cache_contract(overlap_cache, overlap, fold=1, modality="cough")


@pytest.mark.parametrize(
    "case",
    [
        "coughvid_development",
        "coswara_external",
        "external_non_cough",
        "unqualified_identity",
    ],
)
def test_manifest_cache_contract_enforces_dataset_roles_and_qualified_ids(
    tmp_path: Path,
    case: str,
) -> None:
    from covid_audio_btp.hst_training import validate_manifest_cache_contract

    cache, manifest = _contract_inputs(tmp_path)
    if case == "coughvid_development":
        mask = manifest["split"].eq("train")
        keys = manifest.loc[mask, "recording_key"].tolist()
        manifest.loc[mask, "dataset"] = "coughvid"
        manifest.loc[mask, "participant_key"] = manifest.loc[
            mask, "participant_key"
        ].str.replace("coswara::", "coughvid::", regex=False)
        manifest.loc[mask, "recording_key"] = manifest.loc[
            mask, "recording_key"
        ].str.replace("coswara::", "coughvid::", regex=False)
        cache_mask = cache["recording_key"].isin(keys)
        cache.loc[cache_mask, "dataset"] = "coughvid"
        cache.loc[cache_mask, "participant_key"] = cache.loc[
            cache_mask, "participant_key"
        ].str.replace("coswara::", "coughvid::", regex=False)
        cache.loc[cache_mask, "recording_key"] = cache.loc[
            cache_mask, "recording_key"
        ].str.replace("coswara::", "coughvid::", regex=False)
    elif case == "coswara_external":
        mask = manifest["split"].eq("external_test")
        keys = manifest.loc[mask, "recording_key"].tolist()
        manifest.loc[mask, "dataset"] = "coswara"
        manifest.loc[mask, "participant_key"] = manifest.loc[
            mask, "participant_key"
        ].str.replace("coughvid::", "coswara::", regex=False)
        manifest.loc[mask, "recording_key"] = manifest.loc[
            mask, "recording_key"
        ].str.replace("coughvid::", "coswara::", regex=False)
        cache_mask = cache["recording_key"].isin(keys)
        cache.loc[cache_mask, "dataset"] = "coswara"
        cache.loc[cache_mask, "participant_key"] = cache.loc[
            cache_mask, "participant_key"
        ].str.replace("coughvid::", "coswara::", regex=False)
        cache.loc[cache_mask, "recording_key"] = cache.loc[
            cache_mask, "recording_key"
        ].str.replace("coughvid::", "coswara::", regex=False)
    elif case == "external_non_cough":
        manifest["modality"] = "breath"
        cache["modality"] = "breath"
    else:
        manifest.loc[0, "participant_key"] = "participant-without-dataset"
        cache.loc[
            cache["recording_key"].eq(manifest.loc[0, "recording_key"]),
            "participant_key",
        ] = "participant-without-dataset"

    with pytest.raises(ValueError, match="Coswara|COUGHVID|cough|dataset-qualified"):
        validate_manifest_cache_contract(
            cache,
            manifest,
            fold=1,
            modality="breath" if case == "external_non_cough" else "cough",
        )


def test_cache_source_hash_alias_is_checked_not_trusted(tmp_path: Path) -> None:
    from covid_audio_btp.hst_training import validate_manifest_cache_contract

    cache, manifest = _contract_inputs(tmp_path)
    cache["source_sha256"] = cache["source_audio_sha256"]
    validate_manifest_cache_contract(cache, manifest, fold=1, modality="cough")
    cache.loc[0, "source_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="aliases disagree"):
        validate_manifest_cache_contract(cache, manifest, fold=1, modality="cough")


def test_tensor_sha_is_checked_before_loading(tmp_path: Path) -> None:
    from covid_audio_btp.hst_training import load_verified_cached_image

    path = tmp_path / "tensor.npy"
    np.save(path, np.ones((224, 224), dtype=np.float32), allow_pickle=False)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_verified_cached_image(path, expected).shape == (224, 224)
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_verified_cached_image(path, expected)


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.ones((224, 224), dtype=np.float64), "float32"),
        (np.ones((223, 224), dtype=np.float32), "224"),
        (np.full((224, 224), np.nan, dtype=np.float32), "finite"),
    ],
)
def test_cached_image_rejects_wrong_dtype_shape_or_finiteness(
    tmp_path: Path,
    array: np.ndarray,
    message: str,
) -> None:
    from covid_audio_btp.hst_training import load_verified_cached_image

    path = tmp_path / "invalid.npy"
    np.save(path, array, allow_pickle=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=message):
        load_verified_cached_image(path, digest)


def test_prediction_context_binds_architecture_and_executable() -> None:
    from covid_audio_btp.hst_training import validate_prediction_context

    context = {
        "run_id": "hst-abc123",
        "protocol": "track_a",
        "model": "hst_base",
        "checkpoint_hash": "a" * 64,
        "representation": "paper_logmel_224",
        "architecture_sha256": "b" * 64,
        "executable_sha256": "c" * 64,
    }
    assert validate_prediction_context(context) == context
    for field in ("architecture_sha256", "executable_sha256"):
        invalid = dict(context)
        invalid.pop(field)
        with pytest.raises(ValueError, match=field):
            validate_prediction_context(invalid)


def test_unsupported_training_settings_are_rejected() -> None:
    from covid_audio_btp.hst_training import HSTTrainingConfig

    base = dict(
        pilot_freeze_hash="pilot",
        data_contracts_freeze_hash="data",
        dependency_lock_hash="environment",
        accepted_environment_lock_hash="environment",
        physical_batch_size=4,
        gradient_accumulation=2,
        amp=False,
    )
    with pytest.raises(ValueError, match="class-balanced"):
        HSTTrainingConfig(**base, balance_training_classes=False)
    with pytest.raises(TypeError):
        HSTTrainingConfig(**base, early_stopping_patience=3)


def test_partial_or_incomplete_training_cannot_open_held_out_splits() -> None:
    from covid_audio_btp.hst_training import validate_evaluation_request

    validate_evaluation_request(
        training_complete=True,
        evaluate_test=False,
        evaluate_external=False,
        available_splits={"validation", "test"},
    )
    with pytest.raises(ValueError, match="complete"):
        validate_evaluation_request(
            training_complete=False,
            evaluate_test=True,
            evaluate_external=False,
            available_splits={"validation", "test"},
        )
    with pytest.raises(ValueError, match="external_test"):
        validate_evaluation_request(
            training_complete=True,
            evaluate_test=False,
            evaluate_external=True,
            available_splits={"validation", "test"},
        )


def test_artifact_hashes_are_recomputed_from_files(tmp_path: Path) -> None:
    from covid_audio_btp.hst_training import verify_training_artifact_hashes

    paths = {}
    for name, payload in (("manifest", b"manifest"), ("cache_index", b"cache"), ("source_checkpoint", b"weights")):
        path = tmp_path / name
        path.write_bytes(payload)
        paths[name] = path
    expected = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    assert verify_training_artifact_hashes(
        manifest_path=paths["manifest"],
        cache_index_path=paths["cache_index"],
        source_checkpoint_path=paths["source_checkpoint"],
        expected_manifest_sha256=expected["manifest"],
        expected_cache_index_sha256=expected["cache_index"],
        expected_source_checkpoint_sha256=expected["source_checkpoint"],
    ) == expected
    paths["manifest"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="manifest"):
        verify_training_artifact_hashes(
            manifest_path=paths["manifest"],
            cache_index_path=paths["cache_index"],
            source_checkpoint_path=paths["source_checkpoint"],
            expected_manifest_sha256=expected["manifest"],
            expected_cache_index_sha256=expected["cache_index"],
            expected_source_checkpoint_sha256=expected["source_checkpoint"],
        )


def test_checkpoint_sidecar_is_required_before_pickle_load(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from covid_audio_btp.hst_training import _atomic_torch_save, _load_verified_checkpoint

    path = tmp_path / "checkpoint.pt"
    _atomic_torch_save({"model_state_dict": {}, "epoch": 1}, path)
    pointer = json.loads(
        path.with_suffix(path.suffix + ".current.json").read_text(encoding="utf-8")
    )
    sidecar = path.parent / str(pointer["current"]["sidecar_path"])
    assert sidecar.is_file()
    assert _load_verified_checkpoint(path)["epoch"] == 1
    sidecar.write_text(json.dumps({"schema_version": 1, "sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        _load_verified_checkpoint(path)


def test_verified_checkpoint_loader_never_executes_forged_pickle_payload(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from covid_audio_btp.hst_training import _load_verified_checkpoint_pair

    checkpoint = tmp_path / "forged.pt"
    marker = tmp_path / "executed.txt"
    torch.save({"payload": _MaliciousCheckpointValue(marker)}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "writer": "covid_audio_btp.hst_training._atomic_torch_save",
                "filename": checkpoint.name,
                "size_bytes": checkpoint.stat().st_size,
                "sha256": digest,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="weights|Unsupported|checkpoint|pickle"):
        _load_verified_checkpoint_pair(checkpoint, sidecar)
    assert not marker.exists()


def test_predict_split_refuses_mislabeled_loader_rows() -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from covid_audio_btp.hst_training import _collate, predict_hst_split

    class TinyModel(torch.nn.Module):
        def forward(self, values):
            return torch.zeros((values.shape[0], 2), dtype=torch.float32)

    rows = [
        (
            torch.zeros((3, 4, 4), dtype=torch.float32),
            torch.tensor(0, dtype=torch.long),
            {
                "dataset": "coswara",
                "participant_key": "coswara::p1",
                "recording_key": "coswara::r1",
                "label_binary": "negative",
                "split": "test",
                "fold": 1,
                "modality": "cough",
            },
        )
    ]
    loader = DataLoader(rows, batch_size=1, collate_fn=_collate)
    context = {
        "run_id": "run",
        "protocol": "track_a",
        "model": "tiny",
        "checkpoint_hash": "a" * 64,
        "representation": "paper_logmel_224",
        "architecture_sha256": "b" * 64,
        "executable_sha256": "c" * 64,
    }
    with pytest.raises(ValueError, match="split"):
        predict_hst_split(
            TinyModel(),
            loader,
            split="validation",
            fold=1,
            modality="cough",
            prediction_context=context,
        )


def test_recorded_augmentation_seed_is_the_only_augmentation_driver() -> None:
    from covid_audio_btp.hst_training import _augment_image_with_exact_seed

    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    first = _augment_image_with_exact_seed(image, 1234)
    second = _augment_image_with_exact_seed(image, 1234)
    different = _augment_image_with_exact_seed(image, 1235)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    assert first.flags.writeable


def test_multiple_representations_require_explicit_selection(tmp_path: Path) -> None:
    from covid_audio_btp.hst_training import validate_manifest_cache_contract

    cache, manifest = _contract_inputs(tmp_path)
    second_cache = cache.copy()
    second_manifest = manifest.copy()
    second_cache["representation_id"] = "released_linear_specgram_224"
    second_manifest["representation_id"] = "released_linear_specgram_224"
    combined_cache = pd.concat([cache, second_cache], ignore_index=True)
    combined_manifest = pd.concat([manifest, second_manifest], ignore_index=True)
    with pytest.raises(ValueError, match="single frozen representation"):
        validate_manifest_cache_contract(
            combined_cache,
            combined_manifest,
            fold=1,
            modality="cough",
        )
    selected = validate_manifest_cache_contract(
        combined_cache,
        combined_manifest,
        fold=1,
        modality="cough",
        representation_id="paper_logmel_224",
    )
    assert set(selected["representation_id"]) == {"paper_logmel_224"}


def test_loader_frames_must_exactly_match_frozen_splits(tmp_path: Path) -> None:
    from covid_audio_btp.hst_training import (
        _verify_loader_split_contracts,
        validate_manifest_cache_contract,
    )

    cache, manifest = _contract_inputs(tmp_path)
    aligned = validate_manifest_cache_contract(cache, manifest, fold=1, modality="cough")
    loaders: dict[str, object] = {
        "cache_index": cache,
        "manifest": manifest,
        "fold": 1,
        "modality": "cough",
    }
    for split in ("validation", "test", "external_test"):
        frame = aligned.loc[aligned["split"].eq(split)].reset_index(drop=True)
        loaders[split] = SimpleNamespace(dataset=SimpleNamespace(frame=frame))
    _verify_loader_split_contracts(loaders)
    validation = loaders["validation"].dataset.frame.iloc[0:0].copy()
    loaders["validation"] = SimpleNamespace(dataset=SimpleNamespace(frame=validation))
    with pytest.raises(ValueError, match="validation loader rows"):
        _verify_loader_split_contracts(loaders)


def test_executable_fingerprint_is_stable_when_repo_is_relocated(tmp_path: Path) -> None:
    from covid_audio_btp.hst_training import _executable_files_sha256

    first = tmp_path / "first" / "src"
    second = tmp_path / "second" / "src"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    for root in (first, second):
        (root / "model.py").write_text("class Model: pass\n", encoding="utf-8")
        (root / "train.py").write_text("SEED = 52\n", encoding="utf-8")
    assert _executable_files_sha256([first / "model.py", first / "train.py"]) == (
        _executable_files_sha256([second / "model.py", second / "train.py"])
    )


def test_project_registry_slot_rejects_second_learned_model_state(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    model, loader, context = _registry_test_case(tmp_path)
    registry_root = tmp_path / "project-evaluation-registry"
    _run_registry_evaluation(
        model,
        loader,
        context,
        tmp_path=tmp_path,
        registry_root=registry_root,
        run_name="first-run",
    )
    with torch.no_grad():
        model.bias.copy_(torch.tensor([1.0, 0.0]))

    with pytest.raises(ValueError, match="learned model state|learned_model_state"):
        _run_registry_evaluation(
            model,
            loader,
            context,
            tmp_path=tmp_path,
            registry_root=registry_root,
            run_name="second-run",
        )
    assert len(list(registry_root.glob("*/claim.json"))) == 1


def test_project_registry_anchor_rejects_coherent_claim_and_receipt_rewrite(
    tmp_path: Path,
) -> None:
    model, loader, context = _registry_test_case(tmp_path)
    registry_root = tmp_path / "project-evaluation-registry"
    predictions = _run_registry_evaluation(
        model,
        loader,
        context,
        tmp_path=tmp_path,
        registry_root=registry_root,
        run_name="source-run",
    )
    anchor_path = next((registry_root / "_slot_anchors").glob("*.json"))
    anchor_before = anchor_path.read_bytes()
    assert anchor_path.stat().st_mode & stat.S_IWUSR == 0
    claim_path = next(registry_root.glob("*/claim.json"))
    receipt_path = claim_path.with_name("receipt.json")
    predictions_path = claim_path.with_name("predictions.csv")
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    forged_run_id = "coherently-forged-run"
    forged_predictions = predictions.copy()
    forged_predictions["run_id"] = forged_run_id
    forged_predictions.to_csv(predictions_path, index=False)
    claim["source_run_id"] = forged_run_id
    claim["source_record"]["source_run_id"] = forged_run_id
    claim["source_record_hash"] = hashlib.sha256(
        json.dumps(
            claim["source_record"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    claim.pop("record_hash")
    claim["record_hash"] = hashlib.sha256(
        json.dumps(
            claim,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    claim_path.write_text(json.dumps(claim, sort_keys=True), encoding="utf-8")
    receipt.update(
        {
            "source_run_id": forged_run_id,
            "claim_record_hash": claim["record_hash"],
            "source_record_hash": claim["source_record_hash"],
            "predictions_sha256": hashlib.sha256(
                predictions_path.read_bytes()
            ).hexdigest(),
        }
    )
    receipt.pop("record_hash")
    receipt["record_hash"] = hashlib.sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="slot anchor|anchored"):
        _run_registry_evaluation(
            model,
            loader,
            context,
            tmp_path=tmp_path,
            registry_root=registry_root,
            run_name="verification-run",
        )
    assert anchor_path.read_bytes() == anchor_before


def test_interrupted_held_out_evaluation_resumes_exact_anchored_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_training as training

    model, loader, context = _registry_test_case(tmp_path)
    registry_root = tmp_path / "project-evaluation-registry"
    real_predict = training.predict_hst_split

    def interrupted_predict(*args: object, **kwargs: object) -> pd.DataFrame:
        raise RuntimeError("simulated power loss after claim")

    monkeypatch.setattr(training, "predict_hst_split", interrupted_predict)
    with pytest.raises(RuntimeError, match="simulated power loss"):
        _run_registry_evaluation(
            model,
            loader,
            context,
            tmp_path=tmp_path,
            registry_root=registry_root,
            run_name="source-run",
        )
    monkeypatch.setattr(training, "predict_hst_split", real_predict)

    recovered = _run_registry_evaluation(
        model,
        loader,
        context,
        tmp_path=tmp_path,
        registry_root=registry_root,
        run_name="source-run",
    )
    assert len(recovered) == 1
    assert len(list(registry_root.glob("*/receipt.json"))) == 1


def test_interrupted_held_out_evaluation_rejects_wrong_learned_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    import covid_audio_btp.hst_training as training

    model, loader, context = _registry_test_case(tmp_path)
    registry_root = tmp_path / "project-evaluation-registry"

    def interrupted_predict(*args: object, **kwargs: object) -> pd.DataFrame:
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(training, "predict_hst_split", interrupted_predict)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _run_registry_evaluation(
            model,
            loader,
            context,
            tmp_path=tmp_path,
            registry_root=registry_root,
            run_name="source-run",
        )
    with torch.no_grad():
        model.bias.copy_(torch.tensor([1.0, 0.0]))

    with pytest.raises(ValueError, match="learned model state|learned_model_state"):
        _run_registry_evaluation(
            model,
            loader,
            context,
            tmp_path=tmp_path,
            registry_root=registry_root,
            run_name="source-run",
        )


def test_durable_held_out_evaluation_is_reused_not_recomputed(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from covid_audio_btp.hst_training import (
        _collate,
        _evaluate_split_once,
        _model_architecture_sha256,
    )

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([0.0, 1.0]))

        def forward(self, values):
            return self.bias.unsqueeze(0).expand(values.shape[0], -1)

    row = _held_out_contract_row(
        tmp_path,
        recording_id="r1",
        label="positive",
    )
    class FrozenCohortDataset(torch.utils.data.Dataset):
        def __init__(self, cohort_row: dict[str, object]) -> None:
            self.frame = pd.DataFrame([cohort_row])
            self.item = (torch.zeros((3, 4, 4)), torch.tensor(1), cohort_row)

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            if index != 0:
                raise IndexError(index)
            return self.item

    loader = DataLoader(
        FrozenCohortDataset(row),
        batch_size=1,
        collate_fn=_collate,
    )
    model = TinyModel()
    first_checkpoint = tmp_path / "first-checkpoint.pt"
    second_checkpoint = tmp_path / "second-checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "serialization_nonce": 1}, first_checkpoint)
    torch.save({"model_state_dict": model.state_dict(), "serialization_nonce": 2}, second_checkpoint)
    first_checkpoint_hash = hashlib.sha256(first_checkpoint.read_bytes()).hexdigest()
    second_checkpoint_hash = hashlib.sha256(second_checkpoint.read_bytes()).hexdigest()
    assert first_checkpoint_hash != second_checkpoint_hash
    context = {
        "run_id": "first-run",
        "protocol": "track_a",
        "model": "tiny",
        "checkpoint_hash": first_checkpoint_hash,
        "representation": "paper_logmel_224",
        "architecture_sha256": _model_architecture_sha256(model),
        "executable_sha256": "c" * 64,
    }
    registry_root = tmp_path / "project-evaluation-registry"
    first = _evaluate_split_once(
        model,
        loader,
        split="test",
        fold=1,
        modality="cough",
        prediction_context=context,
        run_dir=tmp_path / "first-run",
        project_registry_root=registry_root,
        training_contract="d" * 64,
        best_checkpoint_sha256=first_checkpoint_hash,
        scientific_training_claim=_scientific_training_claim(),
    )
    second_context = {
        **context,
        "run_id": "second-run",
        "checkpoint_hash": second_checkpoint_hash,
    }
    relocated_path = tmp_path / "relocated" / "r1.npy"
    relocated_path.parent.mkdir()
    relocated_path.write_bytes(Path(str(row["cache_path"])).read_bytes())
    relocated_row = {
        **row,
        "cache_path": str(relocated_path),
    }
    relocated_loader = DataLoader(
        FrozenCohortDataset(relocated_row),
        batch_size=1,
        collate_fn=_collate,
    )
    second = _evaluate_split_once(
        model,
        relocated_loader,
        split="test",
        fold=1,
        modality="cough",
        prediction_context=second_context,
        run_dir=tmp_path / "second-run",
        project_registry_root=registry_root,
        training_contract="e" * 64,
        best_checkpoint_sha256=second_checkpoint_hash,
        scientific_training_claim=_scientific_training_claim(),
    )
    assert second.loc[0, "probability"] == pytest.approx(first.loc[0, "probability"])
    assert second.loc[0, "run_id"] == "first-run"

    receipts = list(registry_root.glob("*/receipt.json"))
    assert len(receipts) == 1
    receipt_path = receipts[0]
    predictions_path = receipt_path.with_name("predictions.csv")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["training_schedule"]["physical_batch_size"] == 2
    assert receipt["training_schedule"]["gradient_accumulation"] == 2
    assert receipt["training_schedule"]["epochs"][0][
        "optimizer_boundary_batch_indices"
    ] == [1]
    tampered_receipt = {**receipt, "status": "rewritten"}
    tampered_receipt.pop("record_hash")
    tampered_receipt["record_hash"] = hashlib.sha256(
        json.dumps(
            tampered_receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    receipt_path.write_text(json.dumps(tampered_receipt, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable.*receipt|receipt.*status|slot anchor"):
        _evaluate_split_once(
            model,
            loader,
            split="test",
            fold=1,
            modality="cough",
            prediction_context={**context, "run_id": "tamper-check"},
            run_dir=tmp_path / "tamper-check",
            project_registry_root=registry_root,
            training_contract="d" * 64,
            best_checkpoint_sha256=first_checkpoint_hash,
            scientific_training_claim=_scientific_training_claim(),
        )
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    forged = first.copy()
    forged["run_id"] = "forged-source-run"
    forged.to_csv(predictions_path, index=False)
    forged_receipt = {
        **receipt,
        "source_run_id": "forged-source-run",
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "n_rows": len(forged),
    }
    forged_receipt.pop("record_hash")
    forged_receipt["record_hash"] = hashlib.sha256(
        json.dumps(
            forged_receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    receipt_path.write_text(
        json.dumps(forged_receipt, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source_run_id|source record|slot anchor"):
        _evaluate_split_once(
            model,
            loader,
            split="test",
            fold=1,
            modality="cough",
            prediction_context={**context, "run_id": "source-attack-check"},
            run_dir=tmp_path / "source-attack-check",
            project_registry_root=registry_root,
            training_contract="d" * 64,
            best_checkpoint_sha256=first_checkpoint_hash,
            scientific_training_claim=_scientific_training_claim(),
        )
    first.to_csv(predictions_path, index=False)
    receipt["predictions_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    receipt["n_rows"] = len(first)
    receipt.pop("record_hash")
    receipt["record_hash"] = hashlib.sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    duplicated = pd.concat([first, first], ignore_index=True)
    duplicated.to_csv(predictions_path, index=False)
    receipt["predictions_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    receipt["n_rows"] = 2
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="record hash|checksum|unique|cohort"):
        _evaluate_split_once(
            model,
            loader,
            split="test",
            fold=1,
            modality="cough",
            prediction_context={**context, "run_id": "third-run"},
            run_dir=tmp_path / "third-run",
            project_registry_root=registry_root,
            training_contract="d" * 64,
            best_checkpoint_sha256=first_checkpoint_hash,
            scientific_training_claim=_scientific_training_claim(),
        )


def test_durable_held_out_evaluation_keys_exact_loader_cohort(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from covid_audio_btp.hst_training import (
        _collate,
        _evaluate_split_once,
        _model_architecture_sha256,
    )

    class TinyModel(torch.nn.Module):
        def forward(self, values):
            return torch.zeros((values.shape[0], 2), dtype=torch.float32)

    class FrozenCohortDataset(torch.utils.data.Dataset):
        def __init__(self, recording_key: str) -> None:
            row = _held_out_contract_row(
                tmp_path,
                recording_id=recording_key,
                label="negative",
            )
            self.frame = pd.DataFrame([row])
            self.item = (torch.zeros((3, 4, 4)), torch.tensor(0), row)

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            if index != 0:
                raise IndexError(index)
            return self.item

    model = TinyModel()
    context = {
        "run_id": "run",
        "protocol": "track_a",
        "model": "tiny",
        "checkpoint_hash": "a" * 64,
        "representation": "paper_logmel_224",
        "architecture_sha256": _model_architecture_sha256(model),
        "executable_sha256": "c" * 64,
    }
    first_loader = DataLoader(
        FrozenCohortDataset("r1"), batch_size=1, collate_fn=_collate
    )
    registry_root = tmp_path / "project-evaluation-registry"
    _evaluate_split_once(
        model,
        first_loader,
        split="test",
        fold=1,
        modality="cough",
        prediction_context=context,
        run_dir=tmp_path / "first-run",
        project_registry_root=registry_root,
        training_contract="d" * 64,
        best_checkpoint_sha256="a" * 64,
        scientific_training_claim=_scientific_training_claim(),
    )
    changed_loader = DataLoader(
        FrozenCohortDataset("r2"), batch_size=1, collate_fn=_collate
    )
    _evaluate_split_once(
        model,
        changed_loader,
        split="test",
        fold=1,
        modality="cough",
        prediction_context={**context, "run_id": "second-run"},
        run_dir=tmp_path / "second-run",
        project_registry_root=registry_root,
        training_contract="d" * 64,
        best_checkpoint_sha256="a" * 64,
        scientific_training_claim=_scientific_training_claim(),
    )
    assert len(list(registry_root.glob("*/receipt.json"))) == 2


def test_project_evaluation_registry_fails_closed_on_incomplete_state(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from covid_audio_btp.hst_training import (
        _collate,
        _evaluate_split_once,
        _model_architecture_sha256,
    )

    class TinyModel(torch.nn.Module):
        def forward(self, values):
            return torch.zeros((values.shape[0], 2), dtype=torch.float32)

    row = _held_out_contract_row(
        tmp_path,
        recording_id="r1",
        label="negative",
    )

    class FrozenCohortDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.frame = pd.DataFrame([row])

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            return torch.zeros((3, 4, 4)), torch.tensor(0), row

    loader = DataLoader(FrozenCohortDataset(), batch_size=1, collate_fn=_collate)
    model = TinyModel()
    context = {
        "run_id": "first-run",
        "protocol": "track_a",
        "model": "tiny",
        "checkpoint_hash": "a" * 64,
        "representation": "paper_logmel_224",
        "architecture_sha256": _model_architecture_sha256(model),
        "executable_sha256": "c" * 64,
    }
    registry_root = tmp_path / "project-evaluation-registry"
    _evaluate_split_once(
        model,
        loader,
        split="test",
        fold=1,
        modality="cough",
        prediction_context=context,
        run_dir=tmp_path / "first-run",
        project_registry_root=registry_root,
        training_contract="d" * 64,
        best_checkpoint_sha256="a" * 64,
        scientific_training_claim=_scientific_training_claim(),
    )
    receipt_path = next(registry_root.glob("*/receipt.json"))
    receipt_path.unlink()

    with pytest.raises(ValueError, match="exact anchored source"):
        _evaluate_split_once(
            model,
            loader,
            split="test",
            fold=1,
            modality="cough",
            prediction_context={**context, "run_id": "second-run"},
            run_dir=tmp_path / "second-run",
            project_registry_root=registry_root,
            training_contract="d" * 64,
            best_checkpoint_sha256="a" * 64,
            scientific_training_claim=_scientific_training_claim(),
        )


def test_dataloaders_expose_external_split_and_do_not_persist_train_workers(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from covid_audio_btp.hst_training import make_hst_dataloaders

    cache, manifest = _contract_inputs(tmp_path)
    loaders = make_hst_dataloaders(
        cache,
        manifest,
        fold=1,
        modality="cough",
        physical_batch_size=2,
        num_workers=1,
        seed=52,
    )
    assert "external_test" in loaders
    train_loader = loaders["train_factory"](1)
    assert train_loader.persistent_workers is False


def test_held_out_tensor_access_occurs_only_inside_one_time_evaluation_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    import covid_audio_btp.hst_training as training

    cache, manifest = _contract_inputs(tmp_path)
    held_out_paths = {
        Path(str(path)).resolve()
        for path in cache.loc[
            cache["recording_key"].isin(
                manifest.loc[
                    manifest["split"].isin(["test", "external_test"]),
                    "recording_key",
                ]
            ),
            "cache_path",
        ]
    }
    test_paths = {
        Path(str(path)).resolve()
        for path in cache.loc[
            cache["recording_key"].isin(
                manifest.loc[manifest["split"].eq("test"), "recording_key"]
            ),
            "cache_path",
        ]
    }
    opened: list[Path] = []
    original_load = np.load

    def observed_load(path: object, *args: object, **kwargs: object) -> np.ndarray:
        opened.append(Path(str(path)).resolve())
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(training.np, "load", observed_load)
    loaders = training.make_hst_dataloaders(
        cache,
        manifest,
        fold=1,
        modality="cough",
        physical_batch_size=2,
        num_workers=0,
        seed=52,
    )
    training._verify_loader_split_contracts(loaders)

    assert held_out_paths.isdisjoint(opened)
    list(loaders["validation"])
    assert held_out_paths.isdisjoint(opened)

    class TinyModel(torch.nn.Module):
        def forward(self, values):
            pooled = values.mean(dim=(1, 2, 3))
            return torch.stack((-pooled, pooled), dim=1)

    context = {
        "run_id": "access-gate-run",
        "protocol": "track_a",
        "model": "tiny",
        "checkpoint_hash": "a" * 64,
        "representation": "paper_logmel_224",
        "architecture_sha256": training._model_architecture_sha256(TinyModel()),
        "executable_sha256": "c" * 64,
    }
    training._evaluate_split_once(
        TinyModel(),
        loaders["test"],
        split="test",
        fold=1,
        modality="cough",
        prediction_context=context,
        run_dir=tmp_path / "access-gate-run",
        project_registry_root=tmp_path / "evaluation-registry",
        training_contract="d" * 64,
        best_checkpoint_sha256="a" * 64,
        scientific_training_claim=_scientific_training_claim(),
    )

    assert test_paths.issubset(set(opened))
    assert (held_out_paths - test_paths).isdisjoint(opened)


def test_tiny_torch_training_is_validation_only_and_writes_verified_checkpoints(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    from covid_audio_btp.hst_training import (
        HSTTrainingConfig,
        _model_architecture_sha256,
        _model_state_sha256,
        make_hst_dataloaders,
        train_hst_fold,
        verify_executable_allowlist,
        verify_initial_model_load_audit,
    )

    cache, manifest = _contract_inputs(tmp_path)
    # Add one participant per class so OneCycleLR has two updates in the tiny epoch.
    additions_cache = []
    additions_manifest = []
    for suffix, label in (("n2", "negative"), ("p2", "positive")):
        path = tmp_path / f"train-{suffix}.npy"
        np.save(path, np.full((224, 224), 0.25, dtype=np.float32), allow_pickle=False)
        recording_key = f"coswara::train-{suffix}-r0"
        participant_key = f"coswara::train-{suffix}"
        additions_cache.append(
            {
                "dataset": "coswara",
                "participant_key": participant_key,
                "recording_key": recording_key,
                "label_binary": label,
                "modality": "cough",
                "eligible": True,
                "cache_path": str(path),
                "tensor_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_audio_sha256": hashlib.sha256(
                    f"source::train-{suffix}".encode()
                ).hexdigest(),
                "preprocessing_hash": "e" * 64,
                "representation_id": "paper_logmel_224",
            }
        )
        additions_manifest.append(
            {
                "fold": 1,
                "training_seed": 52,
                "protocol": "track_a",
                "split": "train",
                "dataset": "coswara",
                "participant_key": participant_key,
                "recording_key": recording_key,
                "label_binary": label,
                "modality": "cough",
                "source_audio_sha256": hashlib.sha256(
                    f"source::train-{suffix}".encode()
                ).hexdigest(),
                "preprocessing_hash": "e" * 64,
                "representation_id": "paper_logmel_224",
            }
        )
    cache = pd.concat([cache, pd.DataFrame(additions_cache)], ignore_index=True)
    manifest = pd.concat([manifest, pd.DataFrame(additions_manifest)], ignore_index=True)
    # Replace the small pure-contract tensors with model-sized tensors.
    for row_index, row in cache.iterrows():
        path = Path(str(row["cache_path"]))
        np.save(path, np.full((224, 224), row_index / 20.0, dtype=np.float32), allow_pickle=False)
        cache.loc[row_index, "tensor_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    manifest_path = tmp_path / "manifest.csv"
    cache_path = tmp_path / "cache.csv"
    source_path = tmp_path / "source.pt"
    manifest.to_csv(manifest_path, index=False)
    cache.to_csv(cache_path, index=False)

    class TinyHST(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.head = torch.nn.Linear(3, 2)

        def forward(self, values):
            return self.head(self.pool(values).flatten(1))

    model = TinyHST()
    torch.save(model.state_dict(), source_path)
    source_state = model.state_dict()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    initial_audit = {
        "source_commit": "1" * 40,
        "checkpoint_sha256": source_hash,
        "checkpoint_size_bytes": source_path.stat().st_size,
        "checkpoint_tensor_count": len(source_state),
        "checkpoint_element_count_without_head": sum(
            value.numel()
            for name, value in source_state.items()
            if not name.startswith("head.")
        ),
        "model_parameter_count": sum(value.numel() for value in model.parameters()),
        "backbone_parameter_count": sum(
            value.numel()
            for name, value in model.named_parameters()
            if not name.startswith("head.")
        ),
        "missing_keys": ["head.bias", "head.weight"],
        "unexpected_keys": [],
        "head_reinitialized": True,
        "head_initialization_seed": 52,
        "architecture": {"name": "tiny_hst"},
    }
    executable_paths = [Path(__file__)]
    executable_root = Path(__file__).parent
    executable_allowlist = {
        Path(__file__).resolve().relative_to(executable_root.resolve()).as_posix():
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }
    executable_hash = verify_executable_allowlist(
        executable_root=executable_root,
        executable_paths=executable_paths,
        frozen_allowlist=executable_allowlist,
    )
    initial_binding = verify_initial_model_load_audit(
        model,
        source_checkpoint_path=source_path,
        initial_model_audit=initial_audit,
        model_seed=52,
    )
    context = {
        "run_id": "tiny-run",
        "protocol": "track_a",
        "model": "tiny_hst",
        "checkpoint_hash": source_hash,
        "representation": "paper_logmel_224",
        "architecture_sha256": _model_architecture_sha256(model),
        "executable_sha256": executable_hash,
    }
    loaders = make_hst_dataloaders(
        cache,
        manifest,
        fold=1,
        modality="cough",
        physical_batch_size=1,
        num_workers=0,
        seed=52,
    )
    config = HSTTrainingConfig(
        pilot_freeze_hash="pilot",
        data_contracts_freeze_hash="data",
        dependency_lock_hash="environment",
        accepted_environment_lock_hash="environment",
        physical_batch_size=1,
        gradient_accumulation=1,
        effective_batch_size=1,
        amp=False,
        max_epochs=1,
    )
    result = train_hst_fold(
        model,
        loaders,
        config,
        tmp_path / "run",
        prediction_context=context,
        manifest_path=manifest_path,
        cache_index_path=cache_path,
        source_checkpoint_path=source_path,
        executable_root=executable_root,
        executable_paths=executable_paths,
        frozen_executable_allowlist=executable_allowlist,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        cache_index_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        source_checkpoint_sha256=source_hash,
        initial_model_state_sha256=_model_state_sha256(model),
        initial_model_audit=initial_audit,
        expected_initial_model_binding_sha256=initial_binding,
        evaluate_test=False,
        evaluate_external=False,
    )
    assert result.training_complete is True
    assert result.test_predictions.empty
    assert result.external_predictions.empty
    assert (tmp_path / "run" / "best.pt.current.json").is_file()
    assert not (tmp_path / "run" / "evaluation" / "test_receipt.json").exists()


def test_real_loader_factory_resolves_inferred_representation_without_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from covid_audio_btp.hst_training import make_hst_dataloaders

    class FakeGenerator:
        def manual_seed(self, seed: int) -> None:
            self.seed = seed

    class FakeDataLoader:
        def __init__(self, dataset, **options):
            self.dataset = dataset
            self.options = options
            self.persistent_workers = bool(options.get("persistent_workers", False))

        def __len__(self) -> int:
            return max(1, len(self.dataset))

    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.Generator = FakeGenerator
    fake_utils = ModuleType("torch.utils")
    fake_data = ModuleType("torch.utils.data")
    fake_data.DataLoader = FakeDataLoader
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", fake_data)

    cache, manifest = _contract_inputs(tmp_path)
    loaders = make_hst_dataloaders(
        cache,
        manifest,
        fold=1,
        modality="cough",
        physical_batch_size=2,
        num_workers=0,
        seed=52,
    )
    assert loaders["representation_id"] == "paper_logmel_224"
    assert loaders["train_factory"](1).dataset.frame["representation_id"].eq(
        "paper_logmel_224"
    ).all()


def test_draw_plan_hash_preserves_optimizer_sample_order() -> None:
    from covid_audio_btp.hst_training import (
        _ordered_frame_sha256,
        _unordered_frame_sha256,
    )

    frame = pd.DataFrame(
        {
            "draw_id": ["d0", "d1", "d2"],
            "recording_key": ["r0", "r1", "r2"],
            "augmentation_seed": [10, 11, 12],
        }
    )
    reordered = frame.iloc[[2, 0, 1]].reset_index(drop=True)
    assert _unordered_frame_sha256(frame) == _unordered_frame_sha256(reordered)
    assert _ordered_frame_sha256(frame) != _ordered_frame_sha256(reordered)


def test_training_loop_uses_ordered_draw_identity() -> None:
    from covid_audio_btp.hst_training import train_hst_fold

    source = inspect.getsource(train_hst_fold)
    assert "_ordered_frame_sha256(dataset_frame)" in source
    assert "draw_hash = _canonical_frame_sha256" not in source
    assert "worker state is not restorable" in source


def test_epoch_training_loss_is_weighted_by_sample_count() -> None:
    from covid_audio_btp.hst_training import _sample_weighted_epoch_loss

    assert _sample_weighted_epoch_loss(total_loss=5.0, sample_count=3) == pytest.approx(
        5.0 / 3.0
    )
    with pytest.raises(ValueError, match="sample count"):
        _sample_weighted_epoch_loss(total_loss=0.0, sample_count=0)


def test_training_loop_persists_weighted_loss_accumulators() -> None:
    from covid_audio_btp.hst_training import train_hst_fold

    source = inspect.getsource(train_hst_fold)
    assert '"epoch_loss_sum"' in source
    assert '"epoch_sample_count"' in source
    assert "np.mean(losses)" not in source


def test_draw_plan_verifies_identity_fields_instead_of_overwriting_them() -> None:
    from covid_audio_btp.hst_training import build_hierarchical_epoch_draw_plan

    source = inspect.getsource(build_hierarchical_epoch_draw_plan)
    assert '"split": "train",' not in source
    assert '"label_binary": label,' not in source
    assert '"modality": modality,' not in source


def test_signal_handler_only_requests_optimizer_boundary_stop() -> None:
    import signal

    from covid_audio_btp.hst_training import _optimizer_boundary_signal_guard

    with _optimizer_boundary_signal_guard(required=False) as request:
        request.request(signal.SIGINT, None)
        assert request.requested is True
        assert request.signal_number == signal.SIGINT


def test_duplicate_audio_content_across_splits_hard_fails(tmp_path: Path) -> None:
    from covid_audio_btp.hst_training import validate_manifest_cache_contract

    cache, manifest = _contract_inputs(tmp_path)
    train_hash = manifest.loc[manifest["split"].eq("train"), "source_audio_sha256"].iloc[0]
    test_index = manifest.index[manifest["split"].eq("test")][0]
    test_recording = manifest.loc[test_index, "recording_key"]
    manifest.loc[test_index, "source_audio_sha256"] = train_hash
    cache.loc[cache["recording_key"].eq(test_recording), "source_audio_sha256"] = train_hash
    with pytest.raises(ValueError, match="content-level.*split leakage"):
        validate_manifest_cache_contract(cache, manifest, fold=1, modality="cough")

    cache, manifest = _contract_inputs(tmp_path)
    train_indices = manifest.index[manifest["split"].eq("train")].tolist()
    duplicate_hash = manifest.loc[train_indices[0], "source_audio_sha256"]
    duplicate_recording = manifest.loc[train_indices[1], "recording_key"]
    manifest.loc[train_indices[1], "source_audio_sha256"] = duplicate_hash
    cache.loc[
        cache["recording_key"].eq(duplicate_recording), "source_audio_sha256"
    ] = duplicate_hash
    with pytest.raises(ValueError, match="duplicate source audio content"):
        validate_manifest_cache_contract(cache, manifest, fold=1, modality="cough")


@pytest.mark.parametrize("identity_column", ["cache_path", "tensor_sha256"])
def test_model_input_identity_cannot_cross_frozen_splits(
    tmp_path: Path,
    identity_column: str,
) -> None:
    from covid_audio_btp.hst_training import validate_manifest_cache_contract

    cache, manifest = _contract_inputs(tmp_path)
    train_index = cache.index[cache["recording_key"].eq("coswara::train-n-r0")][0]
    test_index = cache.index[cache["recording_key"].eq("coswara::test-n-r0")][0]
    cache.loc[test_index, identity_column] = cache.loc[train_index, identity_column]
    with pytest.raises(ValueError, match=f"{identity_column}.*cross-split leakage"):
        validate_manifest_cache_contract(cache, manifest, fold=1, modality="cough")


def _confirmatory_config(**overrides: object):
    from covid_audio_btp.hst_training import HSTTrainingConfig

    values: dict[str, object] = {
        "pilot_freeze_hash": "a" * 64,
        "resource_pilot_receipt_sha256": "b" * 64,
        "approved_resource_pairs": ((4, 2),),
        "data_contracts_freeze_hash": "c" * 64,
        "dependency_lock_hash": "d" * 64,
        "accepted_environment_lock_hash": "d" * 64,
        "physical_batch_size": 4,
        "gradient_accumulation": 2,
        "effective_batch_size": 8,
        "amp": True,
        "amp_max_skipped_updates": 0,
        "max_epochs": 100,
        "learning_rate": 1e-5,
        "weight_decay": 1e-8,
        "gradient_clip_norm": 0.1,
        "scheduler_pct_start": 0.3,
        "scheduler_div_factor": 25.0,
        "scheduler_final_div_factor": 10000.0,
        "scheduler_anneal_strategy": "cos",
        "selection_metric": "participant_auroc",
        "epoch_selection_threshold": 0.5,
        "random_seed": 52,
        "deterministic_algorithms": True,
        "confirmatory": True,
    }
    values.update(overrides)
    return HSTTrainingConfig(**values)


def test_confirmatory_training_contract_accepts_only_exact_frozen_settings() -> None:
    config = _confirmatory_config()
    assert (config.physical_batch_size, config.gradient_accumulation) == (4, 2)


def test_confirmatory_training_requires_project_evaluation_registry_root(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_training import train_hst_fold

    parameters = inspect.signature(train_hst_fold).parameters
    assert "evaluation_registry_root" in parameters
    assert "project_evaluation_registry_root" not in parameters

    with pytest.raises(ValueError, match="project evaluation registry root"):
        train_hst_fold(
            object(),
            {},
            _confirmatory_config(),
            tmp_path / "run",
            prediction_context={},
            manifest_path=tmp_path / "manifest.csv",
            cache_index_path=tmp_path / "cache.csv",
            source_checkpoint_path=tmp_path / "source.pt",
            executable_root=tmp_path,
            executable_paths=[],
            frozen_executable_allowlist={},
            manifest_sha256="a" * 64,
            cache_index_sha256="b" * 64,
            source_checkpoint_sha256="c" * 64,
            initial_model_state_sha256="d" * 64,
            initial_model_audit={},
            expected_initial_model_binding_sha256="e" * 64,
        )


def test_confirmatory_stage_adapter_passes_exact_evaluation_registry_keyword(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from covid_audio_btp import hst_stages

    captured: dict[str, object] = {}

    def fake_train(*args: object, evaluation_registry_root: Path, **kwargs: object) -> str:
        captured["args"] = args
        captured["evaluation_registry_root"] = evaluation_registry_root
        captured["kwargs"] = kwargs
        return "called"

    monkeypatch.setattr(hst_stages, "train_hst_fold", fake_train)
    pipeline = SimpleNamespace(
        run_id="run-1",
        run_root=tmp_path / "data" / "outputs" / "hst" / "run-1",
        config=SimpleNamespace(workspace_root=tmp_path),
    )

    assert hst_stages._call_train_hst_fold(
        pipeline,
        "model",
        confirmatory=True,
        marker="value",
    ) == "called"
    assert captured["evaluation_registry_root"] == (
        tmp_path / "data" / "outputs" / "hst" / "_evaluation_registry"
    ).resolve()
    assert captured["kwargs"] == {"marker": "value"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", 2e-5),
        ("weight_decay", 1e-7),
        ("gradient_clip_norm", 0.2),
        ("scheduler_pct_start", 0.2),
        ("scheduler_div_factor", 10.0),
        ("scheduler_final_div_factor", 1000.0),
        ("scheduler_anneal_strategy", "linear"),
        ("selection_metric", "participant_auprc"),
        ("epoch_selection_threshold", 0.4),
        ("max_epochs", 99),
        ("amp_max_skipped_updates", 1),
        ("effective_batch_size", 16),
        ("random_seed", 999),
    ],
)
def test_confirmatory_training_contract_rejects_each_drifted_setting(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="Confirmatory|frozen|prespecified"):
        _confirmatory_config(**{field: value})


def test_confirmatory_resource_pair_must_be_approved() -> None:
    with pytest.raises(ValueError, match="resource-pilot-approved"):
        _confirmatory_config(approved_resource_pairs=((8, 1),))

    with pytest.raises(ValueError, match="batch candidates|resource-pilot-approved"):
        _confirmatory_config(
            physical_batch_size=1,
            gradient_accumulation=8,
            approved_resource_pairs=((1, 8),),
        )


def _write_resource_pilot_artifacts(
    tmp_path: Path,
    *,
    candidate_update: tuple[str, object] | None = None,
    receipt_update: tuple[str, object] | None = None,
) -> tuple[Path, dict[str, object]]:
    from covid_audio_btp.hst_resource_pilot import select_base_resource_pilot
    from covid_audio_btp.hst_runtime import canonical_json_sha256

    rows = []
    for batch_size in (8, 4, 2):
        for precision in ("fp32", "amp"):
            rows.append(
                {
                    "physical_batch_size": batch_size,
                    "precision": precision,
                    "valid": batch_size != 8,
                    "optimizer_updates": 100,
                    "skipped_optimizer_updates": 0,
                    "seconds": 8.0 if (batch_size, precision) == (4, "amp") else 10.0,
                    "free_vram_bytes": 2 * 1024**3,
                    "total_vram_bytes": 8 * 1024**3,
                    "peak_allocated_vram_bytes": 3 * 1024**3,
                    "peak_reserved_vram_bytes": 4 * 1024**3,
                    "max_abs_probability_difference_from_fp32": (
                        0.0 if precision == "fp32" else 0.005
                    ),
                    "relative_loss_difference_from_fp32": (
                        0.0 if precision == "fp32" else 0.005
                    ),
                    "finite_loss": True,
                    "finite_gradients": True,
                    "finite_parameters": True,
                    "finite_predictions": True,
                    "evaluation_loss": 0.5,
                }
            )
    benchmark = pd.DataFrame(rows)
    selection = select_base_resource_pilot(
        benchmark.drop(columns=["evaluation_loss", "error"], errors="ignore"),
        total_vram_bytes=8 * 1024**3,
    )
    assert (selection["physical_batch_size"], selection["gradient_accumulation"]) == (4, 2)
    if candidate_update is not None:
        field, value = candidate_update
        selected = benchmark["physical_batch_size"].eq(4) & benchmark["precision"].eq("amp")
        benchmark[field] = benchmark[field].astype(object)
        benchmark.loc[selected, field] = value
        records = benchmark.drop(
            columns=["evaluation_loss", "error"], errors="ignore"
        ).sort_values(
            ["physical_batch_size", "precision"],
            ascending=[False, True],
        ).to_dict(orient="records")
        selection["benchmark_sha256"] = canonical_json_sha256(records)
    if receipt_update is not None:
        field, value = receipt_update
        selection[field] = value
    from covid_audio_btp.hst_resource_pilot import resource_pilot_freeze_payload

    selection["pilot_freeze_hash"] = canonical_json_sha256(
        resource_pilot_freeze_payload(selection)
    )
    benchmark_path = tmp_path / "base_resource_pilot_trials.csv"
    benchmark.to_csv(benchmark_path, index=False)
    receipt_path = tmp_path / "base_resource_pilot_freeze.json"
    receipt_path.write_text(json.dumps(selection, sort_keys=True), encoding="utf-8")
    return receipt_path, selection


def test_resource_pilot_receipt_is_recomputed_and_binds_actual_producer_artifacts(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_training import verify_resource_pilot_receipt

    path, selection = _write_resource_pilot_artifacts(tmp_path)
    config = _confirmatory_config(
        pilot_freeze_hash=selection["pilot_freeze_hash"],
        resource_pilot_receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert verify_resource_pilot_receipt(path, config) == selection["pilot_freeze_hash"]
    path.with_name("base_resource_pilot_trials.csv").unlink()
    with pytest.raises(FileNotFoundError, match="trials"):
        verify_resource_pilot_receipt(path, config)


@pytest.mark.parametrize(
    ("artifact", "field", "invalid"),
    [
        ("candidate", "optimizer_updates", 99),
        ("candidate", "optimizer_updates", 100.5),
        ("candidate", "skipped_optimizer_updates", 1),
        ("candidate", "valid", False),
        ("candidate", "finite_loss", False),
        ("candidate", "finite_gradients", False),
        ("candidate", "finite_parameters", False),
        ("candidate", "finite_predictions", False),
        ("candidate", "seconds", "nan"),
        ("candidate", "seconds", 0.0),
        ("candidate", "free_vram_bytes", "nan"),
        ("candidate", "free_vram_bytes", 1024),
        ("candidate", "peak_allocated_vram_bytes", "inf"),
        ("candidate", "peak_reserved_vram_bytes", -1),
        ("receipt", "model_metrics_used", 0),
        ("receipt", "probability_tolerance", 0.02),
        ("receipt", "relative_loss_tolerance", 0.02),
        ("candidate", "max_abs_probability_difference_from_fp32", 0.010001),
        ("candidate", "relative_loss_difference_from_fp32", 0.010001),
    ],
)
def test_resource_pilot_receipt_rejects_incomplete_or_unsafe_authorization(
    tmp_path: Path,
    artifact: str,
    field: str,
    invalid: object,
) -> None:
    from covid_audio_btp.hst_training import verify_resource_pilot_receipt

    kwargs = {
        f"{artifact}_update": (field, invalid),
    }
    path, selection = _write_resource_pilot_artifacts(tmp_path, **kwargs)
    config = _confirmatory_config(
        pilot_freeze_hash=selection["pilot_freeze_hash"],
        resource_pilot_receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="resource pilot|Resource pilot|AMP|model endpoints"):
        verify_resource_pilot_receipt(path, config)


def test_resource_pilot_receipt_rejects_tampered_trial_table(tmp_path: Path) -> None:
    from covid_audio_btp.hst_training import verify_resource_pilot_receipt

    path, selection = _write_resource_pilot_artifacts(tmp_path)
    config = _confirmatory_config(
        pilot_freeze_hash=selection["pilot_freeze_hash"],
        resource_pilot_receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    trials_path = path.with_name("base_resource_pilot_trials.csv")
    trials = pd.read_csv(trials_path)
    trials.loc[trials["physical_batch_size"].eq(4), "seconds"] += 1.0
    trials.to_csv(trials_path, index=False)
    with pytest.raises(ValueError, match="benchmark|trial"):
        verify_resource_pilot_receipt(path, config)


def test_resource_pilot_receipt_rejects_model_metrics_in_trial_table(tmp_path: Path) -> None:
    from covid_audio_btp.hst_runtime import canonical_json_sha256
    from covid_audio_btp.hst_training import verify_resource_pilot_receipt

    path, selection = _write_resource_pilot_artifacts(tmp_path)
    trials_path = path.with_name("base_resource_pilot_trials.csv")
    trials = pd.read_csv(trials_path)
    trials["auroc"] = 0.9
    trials.to_csv(trials_path, index=False)
    selection["benchmark_sha256"] = canonical_json_sha256(
        trials.drop(columns=["evaluation_loss", "error"], errors="ignore")
        .sort_values(["physical_batch_size", "precision"], ascending=[False, True])
        .to_dict(orient="records")
    )
    from covid_audio_btp.hst_resource_pilot import resource_pilot_freeze_payload

    selection["pilot_freeze_hash"] = canonical_json_sha256(
        resource_pilot_freeze_payload(selection)
    )
    path.write_text(json.dumps(selection, sort_keys=True), encoding="utf-8")
    config = _confirmatory_config(
        pilot_freeze_hash=selection["pilot_freeze_hash"],
        resource_pilot_receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="model metrics"):
        verify_resource_pilot_receipt(path, config)


def test_executable_allowlist_is_exact_and_content_verified(tmp_path: Path) -> None:
    from covid_audio_btp.hst_training import verify_executable_allowlist

    root = tmp_path / "repo"
    first = root / "src" / "model.py"
    second = root / "src" / "train.py"
    first.parent.mkdir(parents=True)
    first.write_text("MODEL = 1\n", encoding="utf-8")
    second.write_text("TRAIN = 1\n", encoding="utf-8")
    allowlist = {
        "src/model.py": hashlib.sha256(first.read_bytes()).hexdigest(),
        "src/train.py": hashlib.sha256(second.read_bytes()).hexdigest(),
    }
    digest = verify_executable_allowlist(
        executable_root=root,
        executable_paths=[first, second],
        frozen_allowlist=allowlist,
    )
    assert len(digest) == 64
    with pytest.raises(ValueError, match="exactly match"):
        verify_executable_allowlist(
            executable_root=root,
            executable_paths=[first],
            frozen_allowlist=allowlist,
        )
    second.write_text("TRAIN = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_executable_allowlist(
            executable_root=root,
            executable_paths=[first, second],
            frozen_allowlist=allowlist,
        )


def test_training_execution_identity_binds_fold_modality_representation_and_seed() -> None:
    from covid_audio_btp.hst_training import (
        HSTTrainingConfig,
        build_training_execution_identity,
    )

    config = HSTTrainingConfig(
        pilot_freeze_hash="pilot",
        data_contracts_freeze_hash="data",
        dependency_lock_hash="environment",
        accepted_environment_lock_hash="environment",
        physical_batch_size=1,
        gradient_accumulation=1,
        effective_batch_size=1,
        amp=False,
        max_epochs=1,
        random_seed=52,
    )
    loaders = {
        "fold": 3,
        "modality": "cough",
        "representation_id": "paper_logmel_224",
        "seed": 52,
        "manifest": pd.DataFrame(
            {
                "fold": [3],
                "modality": ["cough"],
                "representation_id": ["paper_logmel_224"],
                "training_seed": [52],
                "protocol": ["track_a"],
            }
        ),
    }
    identity = build_training_execution_identity(
        loaders,
        config,
        prediction_context={
            "representation": "paper_logmel_224",
            "protocol": "track_a",
        },
    )
    assert identity == {
        "schema_version": 2,
        "fold": 3,
        "modality": "cough",
        "representation_id": "paper_logmel_224",
        "model_seed": 52,
        "sampler_seed": 52,
        "manifest_training_seed": 52,
        "manifest_protocol": "track_a",
    }
    with pytest.raises(ValueError, match="loader.*seed|sampler.*seed"):
        build_training_execution_identity(
            {**loaders, "seed": 51},
            config,
            prediction_context={
                "representation": "paper_logmel_224",
                "protocol": "track_a",
            },
        )
    with pytest.raises(ValueError, match="representation"):
        build_training_execution_identity(
            loaders,
            config,
            prediction_context={"representation": "other", "protocol": "track_a"},
        )


def test_confirmatory_seed_is_validated_against_protocol_context() -> None:
    from covid_audio_btp.hst_training import build_training_execution_identity

    temporal_config = _confirmatory_config(random_seed=42)
    temporal_loaders = {
        "fold": 1,
        "modality": "cough",
        "representation_id": "paper_logmel_224",
        "seed": 42,
        "manifest": pd.DataFrame(
            {
                "fold": [1],
                "modality": ["cough"],
                "representation_id": ["paper_logmel_224"],
                "training_seed": [42],
                "protocol": ["hst_chronological_split_policy"],
            }
        ),
    }
    identity = build_training_execution_identity(
        temporal_loaders,
        temporal_config,
        prediction_context={
            "representation": "paper_logmel_224",
            "protocol": "hst_chronological_split_policy",
        },
    )
    assert identity["model_seed"] == 42
    with pytest.raises(ValueError, match="protocol"):
        build_training_execution_identity(
            temporal_loaders,
            temporal_config,
            prediction_context={
                "representation": "paper_logmel_224",
                "protocol": "hst_literature_aligned_repeated_holdout",
            },
        )


def test_confirmatory_track_a_fold_is_bound_to_manifest_training_seed() -> None:
    from covid_audio_btp.hst_training import build_training_execution_identity

    config = _confirmatory_config(random_seed=52)
    manifest = pd.DataFrame(
        {
            "fold": [6],
            "modality": ["cough"],
            "representation_id": ["paper_logmel_224"],
            "training_seed": [52],
            "protocol": ["hst_literature_aligned_repeated_holdout"],
        }
    )
    loaders = {
        "fold": 6,
        "modality": "cough",
        "representation_id": "paper_logmel_224",
        "seed": 52,
        "manifest": manifest,
    }
    identity = build_training_execution_identity(
        loaders,
        config,
        prediction_context={
            "representation": "paper_logmel_224",
            "protocol": "hst_literature_aligned_repeated_holdout",
        },
    )
    assert identity["manifest_training_seed"] == 52

    for broken in (
        {**loaders, "fold": 5, "manifest": manifest.assign(fold=5)},
        {**loaders, "manifest": manifest.assign(training_seed=40)},
    ):
        with pytest.raises(ValueError, match="fold.*seed|training_seed|manifest"):
            build_training_execution_identity(
                broken,
                config,
                prediction_context={
                    "representation": "paper_logmel_224",
                    "protocol": "hst_literature_aligned_repeated_holdout",
                },
            )


def test_confirmatory_temporal_manifest_requires_fold_one_and_seed_42() -> None:
    from covid_audio_btp.hst_training import build_training_execution_identity

    config = _confirmatory_config(random_seed=42)
    base_manifest = pd.DataFrame(
        {
            "fold": [1],
            "modality": ["cough"],
            "representation_id": ["paper_logmel_224"],
            "training_seed": [42],
            "protocol": ["hst_chronological_split_policy"],
        }
    )
    base = {
        "fold": 1,
        "modality": "cough",
        "representation_id": "paper_logmel_224",
        "seed": 42,
        "manifest": base_manifest,
    }
    for broken in (
        {**base, "fold": 2, "manifest": base_manifest.assign(fold=2)},
        {**base, "manifest": base_manifest.assign(training_seed=52)},
    ):
        with pytest.raises(ValueError, match="fold|training_seed|manifest"):
            build_training_execution_identity(
                broken,
                config,
                prediction_context={
                    "representation": "paper_logmel_224",
                    "protocol": "hst_chronological_split_policy",
                },
            )


def test_confirmatory_optimizer_covers_every_trainable_backbone_and_head_parameter_once() -> None:
    torch = pytest.importorskip("torch")
    from covid_audio_btp.hst_training import _verify_full_finetuning_optimizer

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.head = torch.nn.Linear(2, 2)

    model = TinyModel()
    optimizer = SimpleNamespace(param_groups=[{"params": list(model.parameters())}])
    audit_hash = _verify_full_finetuning_optimizer(model, optimizer)
    assert len(audit_hash) == 64
    from covid_audio_btp.hst_training import train_hst_fold

    assert "_verify_full_finetuning_optimizer(model, optimizer)" in inspect.getsource(
        train_hst_fold
    )

    model.backbone.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="requires_grad|frozen"):
        _verify_full_finetuning_optimizer(model, optimizer)
    model.backbone.weight.requires_grad_(True)

    missing = SimpleNamespace(param_groups=[{"params": list(model.parameters())[1:]}])
    with pytest.raises(ValueError, match="exactly once|missing"):
        _verify_full_finetuning_optimizer(model, missing)

    parameters = list(model.parameters())
    duplicated = SimpleNamespace(param_groups=[{"params": parameters + [parameters[0]]}])
    with pytest.raises(ValueError, match="exactly once|duplicate"):
        _verify_full_finetuning_optimizer(model, duplicated)


def test_instantiated_model_audit_is_bound_to_actual_source_backbone(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from covid_audio_btp.hst_training import verify_initial_model_load_audit

    class TinyLoadedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.head = torch.nn.Linear(2, 2)

    torch.manual_seed(52)
    model = TinyLoadedModel()
    source_path = tmp_path / "source.pt"
    torch.save(model.state_dict(), source_path)
    state = model.state_dict()
    audit = {
        "source_commit": "1" * 40,
        "checkpoint_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "checkpoint_size_bytes": source_path.stat().st_size,
        "checkpoint_tensor_count": len(state),
        "checkpoint_element_count_without_head": sum(
            value.numel() for name, value in state.items() if not name.startswith("head.")
        ),
        "model_parameter_count": sum(value.numel() for value in model.parameters()),
        "backbone_parameter_count": sum(
            value.numel()
            for name, value in model.named_parameters()
            if not name.startswith("head.")
        ),
        "missing_keys": ["head.bias", "head.weight"],
        "unexpected_keys": [],
        "head_reinitialized": True,
        "head_initialization_seed": 52,
        "architecture": {"name": "tiny"},
    }
    binding = verify_initial_model_load_audit(
        model,
        source_checkpoint_path=source_path,
        initial_model_audit=audit,
        model_seed=52,
    )
    assert len(binding) == 64
    with torch.no_grad():
        model.backbone.weight.add_(1.0)
    with pytest.raises(ValueError, match="backbone"):
        verify_initial_model_load_audit(
            model,
            source_checkpoint_path=source_path,
            initial_model_audit=audit,
            model_seed=52,
        )


@pytest.mark.parametrize("name", ["last.pt", "best.pt"])
def test_transactional_checkpoint_falls_back_to_previous_good_generation(
    tmp_path: Path,
    name: str,
) -> None:
    pytest.importorskip("torch")
    from covid_audio_btp.hst_training import _atomic_torch_save, _load_verified_checkpoint

    path = tmp_path / name
    _atomic_torch_save({"model_state_dict": {}, "epoch": 1}, path)
    _atomic_torch_save({"model_state_dict": {}, "epoch": 2}, path)
    pointer_path = path.with_suffix(path.suffix + ".current.json")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    current_path = path.parent / str(pointer["current"]["checkpoint_path"])
    current_path.write_bytes(b"corrupt-current-generation")
    assert _load_verified_checkpoint(path)["epoch"] == 1
    _atomic_torch_save({"model_state_dict": {}, "epoch": 3}, path)
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    current_path = path.parent / str(pointer["current"]["checkpoint_path"])
    current_path.write_bytes(b"corrupt-new-current-generation")
    assert _load_verified_checkpoint(path)["epoch"] == 1


def test_last_checkpoint_retains_generation_pinned_by_durable_progress(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from covid_audio_btp.hst_runtime import canonical_json_sha256
    from covid_audio_btp.hst_training import _atomic_torch_save

    path = tmp_path / "last.pt"
    pointer_path = path.with_suffix(path.suffix + ".current.json")
    _atomic_torch_save({"model_state_dict": {}, "epoch": 1}, path)
    first_pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    first_generation = str(first_pointer["current"]["generation"])
    progress = {
        "schema_version": 1,
        "receipt_type": "hst_training_progress",
        "status": "checkpointed",
        "checkpoint": dict(first_pointer["current"]),
    }
    progress["record_hash"] = canonical_json_sha256(progress)
    (tmp_path / "training_progress.json").write_text(
        json.dumps(progress), encoding="utf-8"
    )

    _atomic_torch_save({"model_state_dict": {}, "epoch": 2}, path)
    _atomic_torch_save({"model_state_dict": {}, "epoch": 3}, path)
    generation_root = tmp_path / ".last.pt.generations"
    assert (generation_root / first_generation).is_dir()

    latest_pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    progress["checkpoint"] = dict(latest_pointer["current"])
    progress.pop("record_hash")
    progress["record_hash"] = canonical_json_sha256(progress)
    (tmp_path / "training_progress.json").write_text(
        json.dumps(progress), encoding="utf-8"
    )
    _atomic_torch_save({"model_state_dict": {}, "epoch": 4}, path)
    assert not (generation_root / first_generation).exists()


def test_best_checkpoint_recovery_rejects_partial_next_epoch_state(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_training import _load_or_recover_best_checkpoint

    run_dir = tmp_path / "partial-next-epoch"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="epoch-end"):
        _load_or_recover_best_checkpoint(
            run_dir=run_dir,
            last_payload={
                "checkpoint_role": "last",
                "epoch": 5,
                "completed_epoch": 5,
                "best_epoch": 5,
                "resume_epoch": 6,
                "next_consumed_batch_index": 3,
                "epoch_loss_sum": 1.0,
                "epoch_sample_count": 6,
                "epoch_update_boundaries": 2,
                "checkpoint_reason": "wall_clock_interval",
            },
            expected_fingerprint="frozen-contract",
            prediction_context={},
            execution_identity={},
            completed_epoch=5,
            best_epoch=5,
        )
    assert not (run_dir / "best.pt.current.json").exists()


def test_confirmatory_resume_refuses_invalid_artifacts_and_nonempty_fresh_start(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_training import _resolve_resume_checkpoint, train_hst_fold

    assert "_resolve_resume_checkpoint(" in inspect.getsource(train_hst_fold)

    path = tmp_path / "last.pt"
    assert _resolve_resume_checkpoint(
        [path], resume_requested=True, confirmatory=True
    ) is None
    path.with_suffix(path.suffix + ".current.json").write_text(
        "{not-json", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="resume artifacts|valid checkpoint"):
        _resolve_resume_checkpoint(
            [path], resume_requested=True, confirmatory=True
        )
    with pytest.raises(ValueError, match="fresh start|resume artifacts"):
        _resolve_resume_checkpoint(
            [path], resume_requested=False, confirmatory=True
        )


@pytest.mark.parametrize("role", ["best", "other", None])
def test_resume_accepts_only_last_checkpoint_role(tmp_path: Path, role: object) -> None:
    from covid_audio_btp.hst_training import _atomic_torch_save, _resolve_resume_checkpoint

    path = tmp_path / "last.pt"
    payload = {"model_state_dict": {}, "epoch": 1}
    if role is not None:
        payload["checkpoint_role"] = role
    _atomic_torch_save(payload, path)

    with pytest.raises(ValueError, match="checkpoint_role.*last|role.*last"):
        _resolve_resume_checkpoint(
            [path],
            resume_requested=True,
            confirmatory=True,
        )


def test_optimizer_boundary_resume_matches_uninterrupted_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from covid_audio_btp.hst_training import (
        HSTTrainingConfig,
        _load_verified_checkpoint_pair,
        _model_architecture_sha256,
        _model_state_sha256,
        make_hst_dataloaders,
        train_hst_fold,
        verify_executable_allowlist,
        verify_initial_model_load_audit,
    )

    cache, manifest = _contract_inputs(tmp_path)
    additional_cache: list[dict[str, object]] = []
    additional_manifest: list[dict[str, object]] = []
    for label, prefix in (("negative", "n"), ("positive", "p")):
        for index in range(2, 5):
            participant_key = f"coswara::train-{prefix}{index}"
            recording_key = f"{participant_key}-r0"
            path = tmp_path / f"train-{prefix}{index}.npy"
            np.save(path, np.full((224, 224), index / 20.0, dtype=np.float32), allow_pickle=False)
            common = {
                "dataset": "coswara",
                "participant_key": participant_key,
                "recording_key": recording_key,
                "label_binary": label,
                "modality": "cough",
                "source_audio_sha256": hashlib.sha256(recording_key.encode()).hexdigest(),
                "preprocessing_hash": "e" * 64,
                "representation_id": "paper_logmel_224",
            }
            additional_cache.append(
                {
                    **common,
                    "training_seed": 52,
                    "protocol": "track_a",
                    "eligible": True,
                    "cache_path": str(path),
                    "tensor_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            additional_manifest.append(
                {
                    **common,
                    "fold": 1,
                    "training_seed": 52,
                    "protocol": "track_a",
                    "split": "train",
                }
            )
    cache = pd.concat([cache, pd.DataFrame(additional_cache)], ignore_index=True)
    manifest = pd.concat([manifest, pd.DataFrame(additional_manifest)], ignore_index=True)
    for row_index, row in cache.iterrows():
        path = Path(str(row["cache_path"]))
        np.save(path, np.full((224, 224), row_index / 20.0, dtype=np.float32), allow_pickle=False)
        cache.loc[row_index, "tensor_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.csv"
    cache_path = tmp_path / "cache.csv"
    manifest.to_csv(manifest_path, index=False)
    cache.to_csv(cache_path, index=False)

    class TinyResumeHST(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.backbone = torch.nn.Linear(3, 3)
            self.head = torch.nn.Linear(3, 2)

        def forward(self, values):
            return self.head(self.backbone(self.pool(values).flatten(1)))

    torch.manual_seed(52)
    source_model = TinyResumeHST()
    source_path = tmp_path / "source.pt"
    torch.save(source_model.state_dict(), source_path)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    audit = {
        "source_commit": "1" * 40,
        "checkpoint_sha256": source_hash,
        "checkpoint_size_bytes": source_path.stat().st_size,
        "checkpoint_tensor_count": len(source_model.state_dict()),
        "checkpoint_element_count_without_head": sum(
            value.numel()
            for name, value in source_model.state_dict().items()
            if not name.startswith("head.")
        ),
        "model_parameter_count": sum(value.numel() for value in source_model.parameters()),
        "backbone_parameter_count": sum(
            value.numel()
            for name, value in source_model.named_parameters()
            if not name.startswith("head.")
        ),
        "missing_keys": ["head.bias", "head.weight"],
        "unexpected_keys": [],
        "head_reinitialized": True,
        "head_initialization_seed": 52,
        "architecture": {"name": "tiny_resume"},
    }
    executable_root = Path(__file__).parent
    executable_paths = [Path(__file__)]
    relative = Path(__file__).resolve().relative_to(executable_root.resolve()).as_posix()
    allowlist = {relative: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    executable_hash = verify_executable_allowlist(
        executable_root=executable_root,
        executable_paths=executable_paths,
        frozen_allowlist=allowlist,
    )
    config = HSTTrainingConfig(
        pilot_freeze_hash="pilot",
        data_contracts_freeze_hash="data",
        dependency_lock_hash="environment",
        accepted_environment_lock_hash="environment",
        physical_batch_size=2,
        gradient_accumulation=2,
        effective_batch_size=4,
        amp=False,
        max_epochs=1,
        random_seed=52,
        wall_clock_checkpoint_interval_seconds=30.0,
    )

    def execute(
        run_dir: Path,
        *,
        stop_after_optimizer_updates=None,
        resume=True,
        monotonic_clock=None,
        training_config=None,
        reverse_yield_order=False,
        progress_seed=52,
    ):
        active_config = training_config or config
        torch.manual_seed(52)
        model = TinyResumeHST()
        model.load_state_dict(source_model.state_dict())
        binding = verify_initial_model_load_audit(
            model,
            source_checkpoint_path=source_path,
            initial_model_audit=audit,
            model_seed=52,
        )
        context = {
            "run_id": run_dir.name,
            "protocol": "track_a",
            "model": "tiny_hst",
            "checkpoint_hash": source_hash,
            "representation": "paper_logmel_224",
            "architecture_sha256": _model_architecture_sha256(model),
            "executable_sha256": executable_hash,
        }
        loaders = make_hst_dataloaders(
            cache,
            manifest,
            fold=1,
            modality="cough",
            physical_batch_size=active_config.physical_batch_size,
            num_workers=0,
            seed=52,
        )
        if reverse_yield_order:
            original_train_factory = loaders["train_factory"]

            def reversed_train_factory(epoch: int):
                base_loader = original_train_factory(epoch)

                class ReversedYieldLoader:
                    dataset = base_loader.dataset
                    num_workers = base_loader.num_workers

                    def __len__(self) -> int:
                        return len(base_loader)

                    def __iter__(self):
                        return iter(reversed(list(base_loader)))

                return ReversedYieldLoader()

            loaders["train_factory"] = reversed_train_factory
        result = train_hst_fold(
            model,
            loaders,
            active_config,
            run_dir,
            prediction_context=context,
            manifest_path=manifest_path,
            cache_index_path=cache_path,
            source_checkpoint_path=source_path,
            executable_root=executable_root,
            executable_paths=executable_paths,
            frozen_executable_allowlist=allowlist,
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            cache_index_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            source_checkpoint_sha256=source_hash,
            initial_model_state_sha256=_model_state_sha256(model),
            initial_model_audit=audit,
            expected_initial_model_binding_sha256=binding,
            progress_context={
                "run_id": run_dir.name,
                "stage": "internal_cv",
                "job_id": "internal-cough-fold-1",
                "job_spec_sha256": "9" * 64,
                "fold": 1,
                "seed": progress_seed,
                "modality": "cough",
                "protocol": "track_a",
            },
            resume=resume,
            stop_after_optimizer_updates=stop_after_optimizer_updates,
            monotonic_clock=monotonic_clock,
        )
        return model, result

    with pytest.raises(ValueError, match="progress context.*identity|seed"):
        execute(
            tmp_path / "wrong-progress-seed",
            resume=False,
            monotonic_clock=lambda: 0.0,
            progress_seed=999,
        )

    full_model, full_result = execute(tmp_path / "full", monotonic_clock=lambda: 0.0)

    with pytest.raises(ValueError, match="yielded batch identity|batch.*order"):
        execute(
            tmp_path / "reversed-yield",
            resume=False,
            monotonic_clock=lambda: 0.0,
            reverse_yield_order=True,
        )

    clock_values = iter([0.0, 31.0, 31.0])

    def timed_clock() -> float:
        return next(clock_values, 31.0)

    _, interrupted = execute(
        tmp_path / "resumed",
        stop_after_optimizer_updates=1,
        resume=False,
        monotonic_clock=timed_clock,
    )
    assert interrupted.interrupted is True
    progress_path = tmp_path / "resumed" / "training_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    unsigned_progress = dict(progress)
    progress_record_hash = unsigned_progress.pop("record_hash")
    assert progress["receipt_type"] == "hst_training_progress"
    assert progress["status"] == "checkpointed"
    assert progress["run_id"] == "resumed"
    assert progress["stage"] == "internal_cv"
    assert progress["job_id"] == "internal-cough-fold-1"
    assert progress["fold"] == 1
    assert progress["modality"] == "cough"
    assert progress["completed_epoch"] == 0
    assert progress["resume_epoch"] == 1
    assert progress["next_consumed_batch_index"] == 2
    assert progress["epoch_batch_count"] == 4
    assert progress["max_epochs"] == 1
    assert progress["checkpoint_reason"] == "test_stop"
    assert progress["checkpoint_resume_safe"] is True
    assert len(progress_record_hash) == 64
    from covid_audio_btp.hst_runtime import canonical_json_sha256

    assert canonical_json_sha256(unsigned_progress) == progress_record_hash
    pointer_path = progress_path.parent / progress["checkpoint_pointer_path"]
    assert pointer_path.is_file()
    assert hashlib.sha256(pointer_path.read_bytes()).hexdigest() == progress[
        "checkpoint_pointer_sha256"
    ]
    pointer = json.loads(
        (tmp_path / "resumed" / "last.pt.current.json").read_text(encoding="utf-8")
    )
    previous = pointer["previous"]
    periodic_payload = _load_verified_checkpoint_pair(
        tmp_path / "resumed" / previous["checkpoint_path"],
        tmp_path / "resumed" / previous["sidecar_path"],
    )
    assert periodic_payload["checkpoint_reason"] == "wall_clock_interval"
    assert periodic_payload["next_consumed_batch_index"] == 2
    assert periodic_payload["physical_batch_size"] == 2
    assert periodic_payload["gradient_accumulation"] == 2
    assert periodic_payload["epoch_optimizer_boundary_batch_indices"] == [2, 4]
    assert len(periodic_payload["epoch_batch_schedule"]["batch_identity_sha256"]) == 4

    incompatible_config = replace(
        config,
        physical_batch_size=4,
        gradient_accumulation=1,
    )
    with pytest.raises(ValueError, match="physical batch size|gradient accumulation"):
        execute(
            tmp_path / "resumed",
            resume=True,
            monotonic_clock=lambda: 0.0,
            training_config=incompatible_config,
        )

    resumed_model, resumed_result = execute(
        tmp_path / "resumed", resume=True, monotonic_clock=lambda: 0.0
    )
    assert resumed_result.interrupted is False
    assert _model_state_sha256(resumed_model) == _model_state_sha256(full_model)
    pd.testing.assert_frame_equal(
        resumed_result.history.reset_index(drop=True),
        full_result.history.reset_index(drop=True),
        check_exact=True,
    )

    import covid_audio_btp.hst_training as training_module

    real_progress_writer = training_module._write_training_progress
    failed_once = False

    def fail_first_epoch_end_progress(**kwargs):
        nonlocal failed_once
        payload = kwargs["checkpoint_payload"]
        if payload["checkpoint_reason"] == "epoch_end" and not failed_once:
            failed_once = True
            raise RuntimeError("simulated progress publication failure")
        return real_progress_writer(**kwargs)

    monkeypatch.setattr(
        training_module,
        "_write_training_progress",
        fail_first_epoch_end_progress,
    )
    recovery_root = tmp_path / "epoch-end-progress-failure"
    with pytest.raises(RuntimeError, match="simulated progress publication failure"):
        execute(recovery_root, resume=False, monotonic_clock=lambda: 0.0)
    assert (recovery_root / "last.pt.current.json").is_file()
    assert not (recovery_root / "best.pt.current.json").is_file()

    monkeypatch.setattr(
        training_module,
        "_write_training_progress",
        real_progress_writer,
    )
    recovered_model, recovered_result = execute(
        recovery_root,
        resume=True,
        monotonic_clock=lambda: 0.0,
    )
    assert recovered_result.interrupted is False
    assert (recovery_root / "best.pt.current.json").is_file()
    assert _model_state_sha256(recovered_model) == _model_state_sha256(full_model)


def test_wall_clock_checkpoint_interval_must_be_finite_and_positive() -> None:
    from covid_audio_btp.hst_training import HSTTrainingConfig

    base = {
        "pilot_freeze_hash": "pilot",
        "data_contracts_freeze_hash": "data",
        "dependency_lock_hash": "environment",
        "accepted_environment_lock_hash": "environment",
        "physical_batch_size": 1,
        "gradient_accumulation": 1,
        "effective_batch_size": 1,
        "amp": False,
        "max_epochs": 1,
    }
    assert HSTTrainingConfig(**base).wall_clock_checkpoint_interval_seconds == 1800.0
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="wall-clock checkpoint"):
            HSTTrainingConfig(
                **base,
                wall_clock_checkpoint_interval_seconds=invalid,
            )
