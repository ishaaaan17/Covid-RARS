from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


HST_SEEDS = (1, 2, 5, 12, 40, 52, 72, 2002, 4002, 6002)


def _digest(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pipeline(tmp_path: Path, *, mode: str = "full") -> SimpleNamespace:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "stage.py"
    source.write_text("SOURCE = 1\n", encoding="utf-8")
    dependency = tmp_path / "requirements.lock"
    dependency.write_text("torch==test\n", encoding="utf-8")
    config = SimpleNamespace(
        workspace_root=tmp_path,
        mode=mode,
        scientific_config={
            "paths": {
                "coswara_metadata": "coswara.csv",
                "coughvid_metadata": "coughvid.csv",
                "compare_is10_features": "features.csv",
                "checkpoint_directory": "checkpoints",
            },
            "datasets": {
                "coughvid": {
                    "primary_label_column": "status_SSL",
                    "release_id": "coughvid-test-derived-cohort",
                    "source_release_reference": "COUGHVID test release",
                    "metadata_input_level": "derived_processed_csv",
                    "raw_release_membership_reconstructed": False,
                    "identity_source": "recording_uuid",
                    "subject_linkage_available": False,
                    "primary_label_provenance": "semi_supervised_status",
                }
            },
            "experiment": {
                "primary_modalities": ["cough", "speech"],
                "secondary_modalities": ["breath"],
                "project_seeds": list(HST_SEEDS),
            },
            "checkpoints": {
                "hst_base_imagenet": {
                    "filename": "hst_base.pth",
                    "size_bytes": 4,
                    "sha256": "0" * 64,
                }
            },
        },
        accepted_hashes={
            "data_contracts_freeze": "a" * 64,
            "pilot_freeze": "b" * 64,
            "environment_lock": "c" * 64,
        },
        dependency_lock_path=dependency,
        source_root=source_root,
        source_paths=(source,),
        resume=True,
        device="cpu",
    )
    run_root = tmp_path / "run"
    run_root.mkdir()
    return SimpleNamespace(
        config=config,
        run_root=run_root,
        run_id="hst-test-run",
        initial_source_hash="d" * 64,
    )


def _manifest_rows(
    *,
    protocol: str,
    folds: tuple[int, ...],
    seeds: tuple[int, ...],
    modalities: tuple[str, ...],
    external: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold, seed in zip(folds, seeds, strict=True):
        for modality in modalities:
            for split, label in (
                ("train", "negative"),
                ("train", "positive"),
                ("validation", "negative"),
                ("validation", "positive"),
                ("test", "negative"),
                ("test", "positive"),
            ):
                suffix = f"{protocol}-{fold}-{modality}-{split}-{label}"
                rows.append(
                    {
                        "fold": fold,
                        "training_seed": seed,
                        "protocol": protocol,
                        "dataset": "coswara",
                        "participant_key": f"coswara::p-{suffix}",
                        "recording_key": f"coswara::r-{suffix}",
                        "modality": modality,
                        "split": split,
                        "label_binary": label,
                        "representation_id": "paper_logmel_224",
                        "tensor_sha256": "1" * 64,
                        "analysis_role": "primary" if protocol.endswith("holdout") else "secondary",
                        "analysis_scope": "internal_performance",
                        "estimand_id": "test-estimand",
                        "multiplicity_family": "test-family",
                        "confirmatory_protocol": True,
                    }
                )
            if external and modality == "cough":
                for label in ("negative", "positive"):
                    suffix = f"external-{fold}-{label}"
                    rows.append(
                        {
                            "fold": fold,
                            "training_seed": seed,
                            "protocol": protocol,
                            "dataset": "coughvid",
                            "participant_key": f"coughvid::p-{suffix}",
                            "recording_key": f"coughvid::r-{suffix}",
                            "modality": "cough",
                            "split": "external_test",
                            "label_binary": label,
                            "representation_id": "paper_logmel_224",
                            "tensor_sha256": "2" * 64,
                            "analysis_role": "secondary",
                            "analysis_scope": "reliability_evaluation",
                            "estimand_id": "external-transfer",
                            "multiplicity_family": "reliability",
                            "confirmatory_protocol": True,
                        }
                    )
    return pd.DataFrame(rows)


def _write_manifests(pipeline: SimpleNamespace, manifests: dict[str, pd.DataFrame]) -> None:
    root = pipeline.run_root / "manifests"
    root.mkdir(parents=True)
    index: dict[str, object] = {"schema_version": 1, "manifests": {}}
    for name, frame in manifests.items():
        path = root / f"{name}.csv"
        frame.to_csv(path, index=False)
        index["manifests"][name] = {
            "path": path.relative_to(pipeline.run_root).as_posix(),
            "sha256": _digest(path),
            "rows": len(frame),
        }
    (root / "manifest_index.json").write_text(
        json.dumps(index, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "recording_key": ["coswara::cache"],
            "eligible": [True],
            "tensor_sha256": ["3" * 64],
        }
    ).to_csv(root / "spectrogram_cache_index.csv", index=False)


def _write_external_manifest_bound_to_internal(
    pipeline: SimpleNamespace,
    stages: object,
    internal: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    _write_manifests(pipeline, {"internal": internal})
    root = pipeline.run_root / "manifests"
    internal_path = root / "internal.csv"
    internal_sha256 = _digest(internal_path)
    source = internal.loc[internal["modality"].eq("cough")].copy()
    source_identity = stages._external_source_rows_sha256(source)
    source["source_protocol"] = source["protocol"]
    source["protocol"] = "coswara_to_coughvid_hst_external"
    target = _manifest_rows(
        protocol="coswara_to_coughvid_hst_external",
        folds=tuple(range(1, 11)),
        seeds=HST_SEEDS,
        modalities=("cough",),
        external=True,
    )
    target = target.loc[target["dataset"].eq("coughvid")].copy()
    for frame in (source, target):
        frame["source_track_a_manifest_sha256"] = internal_sha256
        frame["source_track_a_cough_rows_sha256"] = source_identity
    external = pd.concat([source, target], ignore_index=True, sort=False)
    external_path = root / "external.csv"
    external.to_csv(external_path, index=False)
    index_path = root / "manifest_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["manifests"]["external"] = {
        "path": external_path.relative_to(pipeline.run_root).as_posix(),
        "sha256": _digest(external_path),
        "rows": len(external),
    }
    index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
    return external, internal_sha256


def test_coswara_contract_preserves_protocol_canonical_string_labels(tmp_path: Path) -> None:
    import covid_audio_btp.hst_stages as stages

    (tmp_path / "negative.wav").write_bytes(b"audio")
    (tmp_path / "positive.wav").write_bytes(b"audio")
    metadata_path = tmp_path / "coswara.csv"
    pd.DataFrame(
        {
            "participant_id": ["n", "p"],
            "recording_id": ["rn", "rp"],
            "modality": ["cough", "cough"],
            "audio_path": ["negative.wav", "positive.wav"],
            "label_binary": [0, "positive"],
        }
    ).to_csv(metadata_path, index=False)

    contract, _audit = stages._coswara_contract(metadata_path)

    assert contract["label_binary"].tolist() == ["negative", "positive"]
    assert contract["contract_eligible"].tolist() == [True, True]


def _mock_job_outputs(root: Path, job: dict[str, object]) -> dict[str, object]:
    job_root = root / str(job["job_id"])
    job_root.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            "run_id": ["hst-test-run"],
            "protocol": [job["protocol"]],
            "fold": [job["fold"]],
            "dataset": ["coswara"],
            "participant_key": [f"coswara::heldout-{job['job_id']}"],
            "split": ["test"],
            "modality": [job["modality"]],
            "model": ["hst_base"],
            "checkpoint_hash": ["4" * 64],
            "representation": ["paper_logmel_224"],
            "label_binary": ["negative"],
            "probability": [0.2],
            "n_recordings": [1],
        }
    )
    metrics = pd.DataFrame(
        {
            "stage": [job["stage"]],
            "protocol": [job["protocol"]],
            "fold": [job["fold"]],
            "modality": [job["modality"]],
            "model": ["hst_base"],
            "metric_split": ["test"],
            "threshold_source": ["validation_balanced_accuracy"],
            "threshold": [0.5],
            "n_participants": [1],
            "auroc": [0.5],
        }
    )
    prediction_path = job_root / "participant_predictions.csv"
    metrics_path = job_root / "metrics.csv"
    checkpoint_path = job_root / "best-generation.pt"
    predictions.to_csv(prediction_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    checkpoint_path.write_bytes(b"best")
    receipt_path = job_root / "job_receipt.json"
    stage_root = root.parent
    outputs = []
    for path in (prediction_path, metrics_path, checkpoint_path):
        outputs.append(
            {
                "path": path.relative_to(stage_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _digest(path),
            }
        )
    receipt_path.write_text(
        json.dumps(
            {
                "status": "success",
                "job_id": job["job_id"],
                "job_spec_sha256": job["job_spec_sha256"],
                "outputs": outputs,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "participant_predictions": predictions,
        "metrics": metrics,
        "receipt_path": receipt_path,
        "best_checkpoint_path": checkpoint_path,
        "best_checkpoint_sha256": _digest(checkpoint_path),
        "validation_threshold": 0.5,
        "training_contract_fingerprint": "6" * 64,
    }


def test_pilot_contract_includes_both_datasets_but_preprocessing_is_coswara_cough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path, mode="pilot")
    (tmp_path / "coswara.csv").write_text("source\ncoswara\n", encoding="utf-8")
    (tmp_path / "coughvid.csv").write_text("source\ncoughvid\n", encoding="utf-8")
    captured_sources: list[Path] = []
    coswara = pd.DataFrame(
        {
            "dataset": ["coswara"],
            "participant_id": ["p1"],
            "participant_key": ["coswara::p1"],
            "recording_id": ["r1"],
            "recording_key": ["coswara::r1"],
            "modality": ["cough"],
            "label_binary": [0],
            "contract_eligible": [True],
        }
    )
    coughvid = pd.DataFrame(
        {
            "dataset": ["coughvid"],
            "participant_id": ["p2"],
            "participant_key": ["coughvid::p2"],
            "recording_id": ["r2"],
            "recording_key": ["coughvid::r2"],
            "modality": ["cough"],
            "label_binary": ["negative"],
            "analysis_unit_type": ["recording_uuid"],
            "subject_linkage_available": [False],
            "metadata_source_level": ["derived_processed_csv"],
        }
    )
    monkeypatch.setattr(stages, "_coswara_contract", lambda _path: (coswara, pd.DataFrame({"n": [1]})))
    monkeypatch.setattr(stages, "build_audited_coughvid_index", lambda *_args, **_kwargs: coughvid)
    monkeypatch.setattr(stages, "audit_coughvid_labels", lambda frame: (frame, pd.DataFrame({"n": [1]})))

    def fake_freeze(*, source_paths: tuple[Path, ...], **_kwargs: object) -> str:
        captured_sources.extend(source_paths)
        return "a" * 64

    monkeypatch.setattr(stages, "freeze_data_contracts", fake_freeze)
    stages._data_contracts(pipeline, "data_contracts")

    assert {path.name for path in captured_sources} == {"coswara.csv", "coughvid.csv"}
    mixed = pd.DataFrame(
        {
            "dataset": ["coswara", "coswara", "coughvid"],
            "modality": ["cough", "speech", "cough"],
        }
    )
    selected = stages._metadata_for_spectrogram_stage(mixed, mode="pilot")
    assert selected[["dataset", "modality"]].to_dict(orient="records") == [
        {"dataset": "coswara", "modality": "cough"}
    ]


def test_frozen_audio_binding_rejects_bytes_changed_after_run_identity(
    tmp_path: Path,
) -> None:
    import covid_audio_btp.hst_stages as stages
    from covid_audio_btp.hst_reliability import audio_input_manifest_sha256

    pipeline = _pipeline(tmp_path, mode="pilot")
    coswara_audio = tmp_path / "coswara.wav"
    coughvid_audio = tmp_path / "coughvid.wav"
    coswara_audio.write_bytes(b"coswara-original")
    coughvid_audio.write_bytes(b"coughvid-original")
    pd.DataFrame(
        {
            "recording_key": ["coswara::r1"],
            "modality": ["cough"],
            "audio_path": [coswara_audio.as_posix()],
        }
    ).to_csv(tmp_path / "coswara.csv", index=False)
    pd.DataFrame(
        {
            "recording_key": ["coughvid::r2"],
            "modality": ["cough"],
            "audio_path": [coughvid_audio.as_posix()],
        }
    ).to_csv(tmp_path / "coughvid.csv", index=False)
    pipeline.config.input_hashes = {
        "coswara_audio_content": audio_input_manifest_sha256(
            tmp_path / "coswara.csv",
            project_root=tmp_path.parent,
            modality="cough",
        ),
        "coughvid_audio_content": audio_input_manifest_sha256(
            tmp_path / "coughvid.csv",
            project_root=tmp_path.parent,
            modality="cough",
        ),
    }
    selected = pd.DataFrame(
        {
            "dataset": ["coswara", "coughvid"],
            "recording_id": ["r1", "r2"],
            "recording_key": ["coswara::r1", "coughvid::r2"],
            "modality": ["cough", "cough"],
            "audio_path": [coswara_audio.as_posix(), coughvid_audio.as_posix()],
        }
    )
    coswara_audio.write_bytes(b"coswara-changed")

    with pytest.raises(ValueError, match="(?i)frozen run identity"):
        stages._bind_frozen_audio_sources(pipeline, selected)


def test_data_contract_materializes_honest_coughvid_source_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path, mode="pilot")
    pipeline.config.scientific_config["datasets"]["coughvid"].update(
        {
            "release_id": "coughvid-v3-derived-cohort-from-zenodo-7024894",
            "source_release_reference": "COUGHVID-v3 Zenodo 7024894",
            "metadata_input_level": "derived_processed_csv",
            "raw_release_membership_reconstructed": False,
            "identity_source": "recording_uuid",
            "subject_linkage_available": False,
        }
    )
    (tmp_path / "coswara.csv").write_text("source\ncoswara\n", encoding="utf-8")
    (tmp_path / "coughvid.csv").write_text("source\ncoughvid\n", encoding="utf-8")
    coswara = pd.DataFrame(
        {
            "dataset": ["coswara"],
            "participant_id": ["p1"],
            "participant_key": ["coswara::p1"],
            "recording_id": ["r1"],
            "recording_key": ["coswara::r1"],
            "modality": ["cough"],
            "label_binary": ["negative"],
            "contract_eligible": [True],
        }
    )
    coughvid = pd.DataFrame(
        {
            "dataset": ["coughvid"],
            "participant_id": ["uuid-1"],
            "participant_key": ["coughvid::uuid-1"],
            "recording_id": ["uuid-1"],
            "recording_key": ["coughvid::uuid-1"],
            "modality": ["cough"],
            "label_binary": ["positive"],
            "identity_source_column": ["uuid"],
            "analysis_unit_type": ["recording_uuid"],
            "participant_id_is_recording_proxy": [True],
            "subject_linkage_available": [False],
            "metadata_source_level": ["derived_processed_csv"],
        }
    )
    monkeypatch.setattr(
        stages, "_coswara_contract", lambda _path: (coswara, pd.DataFrame({"n": [1]}))
    )
    monkeypatch.setattr(
        stages, "build_audited_coughvid_index", lambda *_args, **_kwargs: coughvid
    )
    monkeypatch.setattr(
        stages,
        "audit_coughvid_labels",
        lambda frame, **_kwargs: (frame, pd.DataFrame({"n": [1]})),
    )
    captured_metadata: dict[str, object] = {}

    def fake_freeze(*, contract_metadata: dict[str, object], **_kwargs: object) -> str:
        captured_metadata.update(contract_metadata)
        return "a" * 64

    monkeypatch.setattr(stages, "freeze_data_contracts", fake_freeze)

    result = stages._data_contracts(pipeline, "data_contracts")

    provenance_path = pipeline.run_root / "contracts" / "coughvid_source_provenance.csv"
    provenance = pd.read_csv(provenance_path)
    assert provenance.to_dict(orient="records") == [
        {
            "dataset": "coughvid",
            "cohort_release_id": "coughvid-v3-derived-cohort-from-zenodo-7024894",
            "source_release_reference": "COUGHVID-v3 Zenodo 7024894",
            "metadata_input_level": "derived_processed_csv",
            "raw_release_membership_reconstructed": False,
            "identity_source": "recording_uuid",
            "analysis_unit_type": "recording_uuid",
            "subject_linkage_available": False,
            "primary_label_column": "status_SSL",
            "primary_label_provenance": "semi_supervised_status",
            "source_manifest_sha256": stages.stable_file_sha256(tmp_path / "coughvid.csv"),
        }
    ]
    assert captured_metadata["coughvid_metadata_input_level"] == "derived_processed_csv"
    assert captured_metadata["coughvid_analysis_unit_type"] == "recording_uuid"
    assert captured_metadata["coughvid_subject_linkage_available"] is False
    assert provenance_path in result["output_paths"]


def test_clinical_utility_scope_excludes_external_pseudo_label_endpoint() -> None:
    import covid_audio_btp.hst_stages as stages

    sources = {
        "internal_hst_platt": object(),
        "internal_hst_raw": object(),
        "external_hst_platt": object(),
        "external_hst_raw": object(),
    }
    estimands = {
        "internal_hst_platt": "internal",
        "internal_hst_raw": "internal",
        "external_hst_platt": "external",
        "external_hst_raw": "external",
    }

    scoped_sources, scoped_estimands, audit = stages._clinical_utility_scope(
        sources,
        estimands,
    )

    assert set(scoped_sources) == {"internal_hst_platt", "internal_hst_raw"}
    assert set(scoped_estimands) == set(scoped_sources)
    external = audit.loc[audit["series"].str.startswith("external_hst")]
    assert external["included_in_clinical_utility_outputs"].eq(False).all()  # noqa: E712
    assert external["reason"].eq("semi_supervised_external_pseudo_label").all()


def test_statistics_reporting_contract_is_bound_to_fingerprinted_configuration() -> None:
    import covid_audio_btp.hst_stages as stages

    reporting = {
        "bootstrap_replicates": 1000,
        "bootstrap_seed": 42,
        "confidence_level": 0.95,
        "ece_bins": 10,
        "fixed_sensitivity": 0.90,
        "decision_thresholds": [
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
        ],
        "probability_clip_epsilon": 0.000001,
        "primary_estimand_id": (
            "primary_hst_vs_comparator_uniform_cough_speech_auroc"
        ),
    }

    stages._assert_reporting_config_bound(reporting)

    changed = dict(reporting)
    changed["bootstrap_replicates"] = 999
    with pytest.raises(ValueError, match="reporting configuration"):
        stages._assert_reporting_config_bound(changed)


def test_data_contract_materializes_raw_status_sensitivity_without_entering_primary_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path, mode="pilot")
    pipeline.config.scientific_config["datasets"]["coughvid"].update(
        {
            "raw_status_sensitivity": {
                "label_column": "status",
                "label_provenance": "raw_self_report",
                "execution": "relabel_frozen_external_predictions",
                "selection_use": False,
            },
            "event_quality_sensitivity": {
                "representation": "cough_event_reconstruction",
                "execution": "deferred_missing_checksum_pinned_algorithm",
                "primary_blocking": False,
            },
        }
    )
    pipeline.config.scientific_config["preprocessing"] = {
        "released_code_sensitivity": {
            "representation": "released_linear_specgram_224",
            "execution": "deferred_optional_extension",
            "primary_blocking": False,
        }
    }
    (tmp_path / "coswara.csv").write_text("source\ncoswara\n", encoding="utf-8")
    (tmp_path / "coughvid.csv").write_text("source\ncoughvid\n", encoding="utf-8")
    coswara = pd.DataFrame(
        {
            "dataset": ["coswara"],
            "participant_id": ["p1"],
            "participant_key": ["coswara::p1"],
            "recording_id": ["r1"],
            "recording_key": ["coswara::r1"],
            "modality": ["cough"],
            "label_binary": ["negative"],
            "contract_eligible": [True],
        }
    )

    def coughvid_index(
        _path: Path, *, label_column: str, **_kwargs: object
    ) -> pd.DataFrame:
        label = "positive" if label_column == "status_SSL" else "negative"
        return pd.DataFrame(
            {
                "dataset": ["coughvid"],
                "participant_id": ["p2"],
                "participant_key": ["coughvid::p2"],
                "recording_id": ["r2"],
                "recording_key": ["coughvid::r2"],
                    "modality": ["cough"],
                    "label_binary": [label],
                    "label_source": [label_column],
                    "analysis_unit_type": ["recording_uuid"],
                    "subject_linkage_available": [False],
                    "metadata_source_level": ["derived_processed_csv"],
                }
            )

    monkeypatch.setattr(
        stages, "_coswara_contract", lambda _path: (coswara, pd.DataFrame({"n": [1]}))
    )
    monkeypatch.setattr(stages, "build_audited_coughvid_index", coughvid_index)
    monkeypatch.setattr(
        stages,
        "audit_coughvid_labels",
        lambda frame, **_kwargs: (frame, pd.DataFrame({"n": [1]})),
    )
    monkeypatch.setattr(stages, "freeze_data_contracts", lambda **_kwargs: "a" * 64)

    result = stages._data_contracts(pipeline, "data_contracts")

    primary = pd.read_csv(pipeline.run_root / "contracts" / "coughvid_index.csv")
    raw = pd.read_csv(
        pipeline.run_root / "contracts" / "coughvid_raw_status_sensitivity.csv"
    )
    registry = pd.read_csv(
        pipeline.run_root / "contracts" / "sensitivity_execution_registry.csv"
    )
    assert primary.loc[0, "label_binary"] == "positive"
    assert raw.loc[0, "label_binary"] == "negative"
    assert raw.loc[0, "label_provenance"] == "raw_self_report"
    assert not any(
        path.name == "coughvid_raw_status_sensitivity.csv"
        for path in (pipeline.run_root / "contracts").glob("*_index.csv")
    )
    assert set(registry["sensitivity_id"]) == {
        "coughvid_raw_status_label",
        "coughvid_event_quality",
        "released_code_representation",
    }
    assert result["row_counts"]["coughvid_raw_status_eligible"] == 1


def test_raw_status_sensitivity_relabels_frozen_predictions_without_selection() -> None:
    import covid_audio_btp.hst_stages as stages

    predictions = pd.DataFrame(
        {
            "run_id": ["run"] * 4,
            "protocol": ["primary"] * 4,
            "fold": [1, 1, 2, 2],
            "dataset": ["coughvid"] * 4,
            "participant_key": ["coughvid::n", "coughvid::p"] * 2,
            "split": ["external_test"] * 4,
            "modality": ["cough"] * 4,
            "model": ["hst_base"] * 4,
            "checkpoint_hash": ["a" * 64] * 4,
            "representation": ["paper_logmel_224"] * 4,
            "label_binary": ["positive", "negative"] * 2,
            "probability": [0.1, 0.9, 0.2, 0.8],
        }
    )
    raw_contract = pd.DataFrame(
        {
            "participant_key": ["coughvid::n", "coughvid::p"],
            "recording_key": ["coughvid::n", "coughvid::p"],
            "label_binary": ["negative", "positive"],
            "contract_eligible": [True, True],
            "label_source": ["status", "status"],
            "label_provenance": ["raw_self_report", "raw_self_report"],
        }
    )
    jobs = [
        {
            "fold": fold,
            "seed": fold,
            "source_validation_threshold": 0.5,
        }
        for fold in (1, 2)
    ]

    relabeled, metrics, audit = stages._build_raw_status_external_sensitivity(
        predictions,
        raw_contract,
        jobs=jobs,
    )

    assert relabeled["label_binary"].tolist() == [
        "negative",
        "positive",
        "negative",
        "positive",
    ]
    assert relabeled["primary_label_binary"].tolist() == [
        "positive",
        "negative",
        "positive",
        "negative",
    ]
    assert relabeled["protocol"].eq(
        "coswara_to_coughvid_hst_external_raw_status_sensitivity"
    ).all()
    assert relabeled["analysis_scope"].eq("sensitivity_analysis").all()
    assert relabeled["target_fit"].eq(False).all()
    assert relabeled["target_selection"].eq(False).all()
    assert metrics["auroc"].eq(1.0).all()
    assert metrics["threshold"].eq(0.5).all()
    assert audit.loc[0, "label_disagreement_participants"] == 2
    assert audit.loc[0, "raw_status_eligible_participants"] == 2


def test_raw_status_sensitivity_single_class_is_an_explicit_nonblocking_skip() -> None:
    import covid_audio_btp.hst_stages as stages

    predictions = pd.DataFrame(
        {
            "fold": [1, 2],
            "dataset": ["coughvid", "coughvid"],
            "participant_key": ["coughvid::n", "coughvid::n"],
            "split": ["external_test", "external_test"],
            "modality": ["cough", "cough"],
            "model": ["hst_base", "hst_base"],
            "checkpoint_hash": ["a" * 64, "a" * 64],
            "representation": ["paper_logmel_224", "paper_logmel_224"],
            "label_binary": ["positive", "positive"],
            "probability": [0.1, 0.2],
        }
    )
    raw_contract = pd.DataFrame(
        {
            "participant_key": ["coughvid::n"],
            "recording_key": ["coughvid::n"],
            "label_binary": ["negative"],
            "contract_eligible": [True],
            "label_source": ["status"],
            "label_provenance": ["raw_self_report"],
        }
    )
    jobs = [
        {
            "fold": fold,
            "seed": fold,
            "source_validation_threshold": 0.5,
        }
        for fold in (1, 2)
    ]

    relabeled, metrics, audit = stages._build_raw_status_external_sensitivity(
        predictions,
        raw_contract,
        jobs=jobs,
    )

    assert len(relabeled) == 2
    assert metrics["skipped"].eq(True).all()
    assert set(metrics["skip_reason"]) == {
        "raw_status_aligned_labels_do_not_contain_both_classes"
    }
    assert metrics["auroc"].isna().all()
    assert audit.loc[0, "skipped"]
    assert not audit.loc[0, "primary_blocking"]
    assert audit.loc[0, "skip_reason"] == (
        "raw_status_aligned_labels_do_not_contain_both_classes"
    )


def test_raw_status_sensitivity_without_eligible_labels_does_not_block_primary() -> None:
    import covid_audio_btp.hst_stages as stages

    predictions = pd.DataFrame(
        {
            "fold": [1],
            "dataset": ["coughvid"],
            "participant_key": ["coughvid::u"],
            "split": ["external_test"],
            "modality": ["cough"],
            "label_binary": ["negative"],
            "probability": [0.4],
        }
    )
    raw_contract = pd.DataFrame(
        {
            "participant_key": ["coughvid::u"],
            "label_binary": ["unknown"],
            "contract_eligible": [False],
            "label_source": ["status"],
            "label_provenance": ["raw_self_report"],
        }
    )

    relabeled, metrics, audit = stages._build_raw_status_external_sensitivity(
        predictions,
        raw_contract,
        jobs=[{"fold": 1, "seed": 1, "source_validation_threshold": 0.5}],
    )

    assert relabeled.empty
    assert metrics.loc[0, "skipped"]
    assert metrics.loc[0, "skip_reason"] == "raw_status_has_no_supervised_participants"
    assert audit.loc[0, "skipped"]
    assert not audit.loc[0, "primary_blocking"]


def test_spectrogram_stage_uses_shared_content_addressed_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path, mode="pilot")
    selection = pipeline.run_root / "audits" / "preprocess_worker_selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps({"workers": 1}), encoding="utf-8")
    metadata = pd.DataFrame(
        {
            "dataset": ["coswara", "coughvid"],
            "modality": ["cough", "cough"],
            "recording_key": ["coswara::r1", "coughvid::r2"],
        }
    )
    captured: dict[str, object] = {}

    def fake_build(frame: pd.DataFrame, *, output_dir: Path, **_kwargs: object) -> pd.DataFrame:
        captured["rows"] = frame.copy()
        captured["root"] = output_dir
        tensor = output_dir / ("e" * 64) / "tensors" / "tensor.npy"
        tensor.parent.mkdir(parents=True)
        tensor.write_bytes(b"tensor")
        return pd.DataFrame(
            {
                "recording_key": ["coswara::r1"],
                "eligible": [True],
                "source_sha256": ["1" * 64],
                "expected_source_sha256": ["1" * 64],
                "tensor_sha256": ["2" * 64],
                "cache_path": [tensor.as_posix()],
            }
        )

    monkeypatch.setattr(stages, "_primary_contract_metadata", lambda _pipeline: metadata)
    monkeypatch.setattr(
        stages,
        "_bind_frozen_audio_sources",
        lambda _pipeline, frame: frame.assign(
            expected_source_sha256="1" * 64,
            expected_source_size_bytes=1,
        ),
    )
    monkeypatch.setattr(stages, "parallel_build_spectrograms", fake_build)
    monkeypatch.setattr(stages, "load_verified_cached_image", lambda *_args, **_kwargs: object())
    result = stages._spectrogram_cache(pipeline, "spectrogram_cache")

    expected_root = tmp_path / "data" / "processed" / "hst_spectrogram_cache"
    assert captured["root"] == expected_root
    assert captured["rows"][["dataset", "modality"]].to_dict(orient="records") == [
        {"dataset": "coswara", "modality": "cough"}
    ]
    index = pd.read_csv(result["output_paths"][0])
    assert Path(index.loc[0, "cache_path"]).is_relative_to(expected_root)


def test_manifest_stage_keeps_coughvid_out_of_all_source_training_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_protocols as protocols
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path)
    cache_path = pipeline.run_root / "manifests" / "spectrogram_cache_index.csv"
    cache_path.parent.mkdir(parents=True)
    cache = pd.DataFrame(
        {
            "eligible": [True, True, True],
            "dataset": ["coswara", "coswara", "coughvid"],
            "participant_key": [
                "coswara::p-c",
                "coswara::p-s",
                "coughvid::p-c",
            ],
            "modality": ["cough", "speech", "cough"],
            "recording_key": ["coswara::c", "coswara::s", "coughvid::c"],
            "label_binary": ["negative", "positive", "negative"],
            "tensor_sha256": ["1" * 64, "2" * 64, "3" * 64],
            "source_audio_sha256": ["4" * 64, "5" * 64, "6" * 64],
            "preprocessing_hash": ["7" * 64] * 3,
            "representation_id": ["paper_logmel_224"] * 3,
        }
    )
    cache.to_csv(cache_path, index=False)
    protocol_inputs: list[pd.DataFrame] = []
    task_inputs: list[pd.DataFrame] = []
    temporal_inputs: list[pd.DataFrame] = []
    external_inputs: dict[str, pd.DataFrame] = {}

    def frozen(frame: pd.DataFrame, *, protocol: str) -> pd.DataFrame:
        result = frame[["dataset", "modality", "recording_key"]].copy()
        result["protocol"] = protocol
        return result

    def internal_builder(frame: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        protocol_inputs.append(frame.copy())
        return frozen(frame, protocol="internal")

    def task_builder(frame: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        task_inputs.append(frame.copy())
        return frozen(frame.loc[frame["modality"].eq("cough")], protocol="task2")

    def split_builder(frame: pd.DataFrame, **_kwargs: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        temporal_inputs.append(frame.copy())
        return frozen(frame, protocol="mixed"), frozen(frame, protocol="early")

    def common_builder(frame: pd.DataFrame, **_kwargs: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        temporal_inputs.append(frame.copy())
        return frozen(frame, protocol="common-mixed"), frozen(frame, protocol="common-early")

    def reverse_builder(frame: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        temporal_inputs.append(frame.copy())
        return frozen(frame, protocol="reverse")

    def external_builder(
        frame: pd.DataFrame, source: pd.DataFrame, **_kwargs: object
    ) -> pd.DataFrame:
        external_inputs["cache"] = frame.copy()
        external_inputs["source"] = source.copy()
        return frozen(frame.loc[frame["modality"].eq("cough")], protocol="external")

    def aligned_builder(components: dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows = []
        for index, (name, frame) in enumerate(components.items(), start=1):
            current = frame.copy()
            current["manifest_component"] = name
            current["source_manifest_sha256"] = f"{index:x}" * 64
            current["manifest_sha256"] = "9" * 64
            rows.append(current)
        return pd.concat(rows, ignore_index=True, sort=False)

    monkeypatch.setattr(stages, "scientific_configuration_fingerprint", lambda _config: "e" * 64)
    monkeypatch.setattr(
        protocols,
        "scientific_configuration_fingerprint",
        lambda _config: "e" * 64,
    )
    monkeypatch.setattr(stages, "build_protocol_matched_hst_manifest", internal_builder)
    monkeypatch.setattr(stages, "build_hst_task2_like_cough_manifest", task_builder)
    monkeypatch.setattr(stages, "build_split_policy_contrast_manifests", split_builder)
    monkeypatch.setattr(stages, "build_common_late_test_manifests", common_builder)
    monkeypatch.setattr(stages, "build_reverse_temporal_hst_manifest", reverse_builder)
    monkeypatch.setattr(stages, "build_external_hst_manifest", external_builder)
    monkeypatch.setattr(
        stages,
        "_bind_external_manifest_to_internal",
        lambda frame, _internal, **_kwargs: frame,
    )
    monkeypatch.setattr(
        stages,
        "_external_source_rows_sha256",
        lambda _frame: "7" * 64,
    )
    monkeypatch.setattr(stages, "_build_aligned_comparator_manifest", aligned_builder)
    monkeypatch.setattr(stages, "audit_hst_manifest", lambda frame: pd.DataFrame({"rows": [len(frame)]}))
    stages._manifests(pipeline, "manifests")

    assert len(protocol_inputs) == 1
    assert all(frame["dataset"].eq("coswara").all() for frame in protocol_inputs)
    assert all(frame["dataset"].eq("coswara").all() for frame in task_inputs)
    assert all(frame["dataset"].eq("coswara").all() for frame in temporal_inputs)
    assert set(external_inputs["cache"]["dataset"]) == {"coswara", "coughvid"}
    assert external_inputs["source"]["dataset"].eq("coswara").all()
    assert external_inputs["source"]["modality"].eq("cough").all()
    expected_external_source = frozen(
        protocol_inputs[0], protocol="internal"
    ).loc[lambda frame: frame["modality"].eq("cough")]
    pd.testing.assert_frame_equal(
        external_inputs["source"].reset_index(drop=True),
        expected_external_source.reset_index(drop=True),
    )


def test_capacity_manifest_stage_builds_only_internal_comparator_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_protocols as protocols
    import covid_audio_btp.hst_stages as stages
    from covid_audio_btp.hst_workloads import CAPACITY_INTERNAL_FUSION_PROFILE

    pipeline = _pipeline(tmp_path)
    pipeline.config.scientific_config["experiment"]["workload_profile"] = (
        CAPACITY_INTERNAL_FUSION_PROFILE
    )
    pipeline.config.scientific_config["experiment"]["secondary_modalities"] = []
    cache_path = pipeline.run_root / "manifests" / "spectrogram_cache_index.csv"
    cache_path.parent.mkdir(parents=True)
    cache = pd.DataFrame(
        {
            "eligible": [True, True, True],
            "dataset": ["coswara", "coswara", "coughvid"],
            "participant_key": ["coswara::p-c", "coswara::p-s", "coughvid::p-c"],
            "modality": ["cough", "speech", "cough"],
            "recording_key": ["coswara::c", "coswara::s", "coughvid::c"],
            "label_binary": ["negative", "positive", "negative"],
            "tensor_sha256": ["1" * 64, "2" * 64, "3" * 64],
            "source_audio_sha256": ["4" * 64, "5" * 64, "6" * 64],
            "preprocessing_hash": ["7" * 64] * 3,
            "representation_id": ["paper_logmel_224"] * 3,
        }
    )
    cache.to_csv(cache_path, index=False)
    aligned_components: list[tuple[str, ...]] = []

    def internal_builder(frame: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        result = frame[["dataset", "modality", "recording_key"]].copy()
        result["protocol"] = "internal"
        return result

    def forbidden_builder(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise AssertionError("capacity profile invoked an excluded manifest builder")

    def forbidden_pair(*_args: object, **_kwargs: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise AssertionError("capacity profile invoked an excluded manifest builder")

    def aligned_builder(components: dict[str, pd.DataFrame]) -> pd.DataFrame:
        aligned_components.append(tuple(components))
        result = components["internal"].copy()
        result["manifest_component"] = "internal"
        result["source_manifest_sha256"] = "8" * 64
        result["manifest_sha256"] = "9" * 64
        return result

    monkeypatch.setattr(stages, "scientific_configuration_fingerprint", lambda _config: "e" * 64)
    monkeypatch.setattr(
        protocols,
        "scientific_configuration_fingerprint",
        lambda _config: "e" * 64,
    )
    monkeypatch.setattr(stages, "build_protocol_matched_hst_manifest", internal_builder)
    monkeypatch.setattr(stages, "build_hst_task2_like_cough_manifest", forbidden_builder)
    monkeypatch.setattr(stages, "build_split_policy_contrast_manifests", forbidden_pair)
    monkeypatch.setattr(stages, "build_common_late_test_manifests", forbidden_pair)
    monkeypatch.setattr(stages, "build_reverse_temporal_hst_manifest", forbidden_builder)
    monkeypatch.setattr(stages, "build_external_hst_manifest", forbidden_builder)
    monkeypatch.setattr(stages, "_build_aligned_comparator_manifest", aligned_builder)
    monkeypatch.setattr(
        stages,
        "audit_hst_manifest",
        lambda frame: pd.DataFrame({"rows": [len(frame)]}),
    )

    result = stages._manifests(pipeline, "manifests")

    assert aligned_components == [("internal",)]
    assert result["row_counts"].keys() == {
        "representation_eligibility",
        "internal",
        "aligned_comparator",
    }
    index = json.loads(
        (pipeline.run_root / "manifests" / "manifest_index.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(index["manifests"]) == {"internal", "aligned_comparator"}
    assert set(
        index["manifests"]["aligned_comparator"]["component_manifest_sha256"]
    ) == {"internal"}


def test_aligned_comparator_manifest_unions_all_declared_contexts_once() -> None:
    import covid_audio_btp.hst_stages as stages
    from covid_audio_btp import hst_protocols

    protocols = {
        "internal": "hst_literature_aligned_repeated_holdout",
        "task2_like_cough": "hst_literature_aligned_repeated_holdout",
        "calendar_mixed": "hst_calendar_mixed_split_policy",
        "early_to_late": "hst_chronological_split_policy",
        "common_late_mixed": "hst_common_late_test_date_balanced_source",
        "common_late_chronological": "hst_common_late_test_chronological_source",
        "reverse_temporal": "hst_reverse_temporal_sensitivity",
        "external": "coswara_to_coughvid_hst_external",
    }
    components: dict[str, pd.DataFrame] = {}
    for index, (name, protocol) in enumerate(protocols.items(), start=1):
        frame = _manifest_rows(
            protocol=protocol,
            folds=(1,),
            seeds=(1 if name in {"internal", "task2_like_cough", "external"} else 42,),
            modalities=("cough",),
            external=name == "external",
        )
        if name == "task2_like_cough":
            frame["analysis_scope"] = "symptom_matched_cough"
            frame["analysis_role"] = "exploratory"
            frame["confirmatory_protocol"] = False
        frame["manifest_sha256"] = f"{index:x}" * 64
        frame["manifest_sha256"] = frame["manifest_sha256"].str[:64]
        components[name] = frame

    union = stages._build_aligned_comparator_manifest(components)

    assert set(union["manifest_component"]) == set(protocols)
    assert union["manifest_sha256"].nunique() == 1
    assert not union.duplicated(
        ["protocol", "fold", "recording_key", "modality"]
    ).any()
    for name, frame in components.items():
        observed = set(
            union.loc[
                union["manifest_component"].eq(name), "source_manifest_sha256"
            ]
        )
        assert observed == {frame["manifest_sha256"].iloc[0]}
    assert union["manifest_sha256"].iloc[0] == hst_protocols._manifest_digest(union)


def test_aligned_comparator_manifest_rejects_unresolved_context_collision() -> None:
    import covid_audio_btp.hst_stages as stages

    frame = _manifest_rows(
        protocol="duplicate-protocol",
        folds=(1,),
        seeds=(42,),
        modalities=("cough",),
    )
    frame["manifest_sha256"] = "1" * 64
    duplicate = frame.copy()
    duplicate["manifest_sha256"] = "2" * 64

    with pytest.raises(ValueError, match="context collision"):
        stages._build_aligned_comparator_manifest(
            {"calendar_mixed": frame, "early_to_late": duplicate}
        )


def test_smoke_training_is_real_two_epoch_hst_small_and_fails_without_inputs(
    tmp_path: Path,
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path, mode="smoke")
    handlers = stages.build_scientific_stage_handlers(pipeline.config)

    assert handlers["small_smoke"] is stages._small_smoke
    with pytest.raises(FileNotFoundError):
        handlers["small_smoke"](pipeline, "small_smoke")


def test_small_smoke_manifest_is_deterministic_in_pilot_and_full_modes() -> None:
    import covid_audio_btp.hst_stages as stages

    manifest = _manifest_rows(
        protocol="hst_literature_aligned_repeated_holdout",
        folds=tuple(range(1, 11)),
        seeds=HST_SEEDS,
        modalities=("cough", "breath", "speech"),
    )
    external = manifest.iloc[[0]].copy()
    external["dataset"] = "coughvid"
    external["split"] = "external_test"
    combined = pd.concat([manifest, external], ignore_index=True, sort=False)

    selected = stages._derive_small_smoke_manifest(combined)

    assert set(selected["modality"]) == {"cough"}
    assert set(selected["training_seed"]) == {52}
    assert selected["fold"].nunique() == 1
    assert "external_test" not in set(selected["split"])
    assert set(selected["dataset"]) == {"coswara"}
    assert selected["row_content_sha256"].str.len().eq(64).all()
    assert selected["manifest_sha256"].nunique() == 1


@pytest.mark.parametrize("mode", ["pilot", "full"])
def test_real_small_smoke_handler_traverses_pilot_and_full_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path, mode=mode)
    internal = _manifest_rows(
        protocol="hst_literature_aligned_repeated_holdout",
        folds=tuple(range(1, 11)),
        seeds=HST_SEEDS,
        modalities=("cough", "speech", "breath"),
    )
    _write_manifests(pipeline, {"internal": internal})
    checkpoint = tmp_path / "hst-small.pth"
    checkpoint.write_bytes(b"small")

    class FakeModel:
        def to(self, _device: str) -> "FakeModel":
            return self

    result = SimpleNamespace(
        last_epoch=2,
        training_complete=True,
        history=pd.DataFrame({"epoch": [1, 2], "validation_auroc": [0.5, 0.6]}),
        validation_predictions=pd.DataFrame(
            {
                "participant_key": ["coswara::validation"],
                "label_binary": ["negative"],
                "probability": [0.4],
            }
        ),
        best_epoch=2,
        validation_threshold=0.5,
        test_evaluated=False,
        best_checkpoint_sha256="a" * 64,
        training_contract_fingerprint="b" * 64,
    )
    monkeypatch.setattr(stages, "_checkpoint_path", lambda *_args: checkpoint)
    monkeypatch.setattr(stages, "_source_path", lambda *_args: tmp_path)
    monkeypatch.setattr(
        stages,
        "load_verified_hst_model",
        lambda **_kwargs: (FakeModel(), {"verified": True}),
    )
    monkeypatch.setattr(
        stages,
        "_executable_allowlist",
        lambda _config: ((pipeline.config.source_paths[0],), {"stage.py": "c" * 64}, "d" * 64),
    )
    monkeypatch.setattr(
        stages,
        "verify_initial_model_load_audit",
        lambda *_args, **_kwargs: "e" * 64,
    )
    monkeypatch.setattr(stages, "make_hst_dataloaders", lambda *_args, **_kwargs: {"loader": True})
    monkeypatch.setattr(stages, "_data_contract_freeze_hash", lambda _pipeline: "f" * 64)
    monkeypatch.setattr(stages, "_model_state_sha256", lambda _model: "1" * 64)
    monkeypatch.setattr(stages, "_training_context", lambda **_kwargs: {"context": True})
    monkeypatch.setattr(stages, "_call_train_hst_fold", lambda *_args, **_kwargs: result)

    output = stages._small_smoke(pipeline, "small_smoke")
    derived = pd.read_csv(pipeline.run_root / "manifests" / "small_smoke.csv")

    assert output["metadata"] == {
        "model": "hst_small",
        "epochs": 2,
        "test_labels_opened": False,
    }
    assert set(derived["training_seed"]) == {52}
    assert set(derived["modality"]) == {"cough"}
    assert "external_test" not in set(derived["split"])


def test_confirmatory_training_uses_project_level_test_once_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path)
    observed: dict[str, object] = {}

    def fake_train(
        model: object,
        loaders: object,
        config: object,
        run_dir: Path,
        *,
        prediction_context: object,
        evaluation_registry_root: Path,
    ) -> str:
        observed["model"] = model
        observed["loaders"] = loaders
        observed["config"] = config
        observed["run_dir"] = run_dir
        observed["prediction_context"] = prediction_context
        observed["evaluation_registry_root"] = evaluation_registry_root
        return "trained"

    monkeypatch.setattr(stages, "train_hst_fold", fake_train)
    result = stages._call_train_hst_fold(
        pipeline,
        object(),
        {},
        object(),
        pipeline.run_root / "job",
        confirmatory=True,
        prediction_context={},
    )

    assert result == "trained"
    assert observed["evaluation_registry_root"] == (
        tmp_path / "data" / "outputs" / "hst" / "_evaluation_registry"
    ).resolve()
    assert not Path(str(observed["evaluation_registry_root"])).is_relative_to(
        pipeline.run_root
    )


def test_internal_cv_enumerates_primary_and_secondary_hst_base_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path)
    internal = _manifest_rows(
        protocol="hst_literature_aligned_repeated_holdout",
        folds=tuple(range(1, 11)),
        seeds=HST_SEEDS,
        modalities=("cough", "speech", "breath"),
    )
    task2 = _manifest_rows(
        protocol="hst_literature_aligned_repeated_holdout",
        folds=tuple(range(1, 11)),
        seeds=HST_SEEDS,
        modalities=("cough",),
    )
    task2["analysis_role"] = "exploratory"
    task2["analysis_scope"] = "symptom_matched_cough"
    task2["estimand_id"] = "task2_like_cough_internal_performance"
    task2["multiplicity_family"] = "exploratory_task2_like"
    task2["confirmatory_protocol"] = False
    _write_manifests(
        pipeline,
        {"internal": internal, "task2_like_cough": task2},
    )
    monkeypatch.setattr(stages, "_load_confirmatory_bindings", lambda _pipeline: {"verified": True})
    observed: list[dict[str, object]] = []

    def fake_execute(_pipeline: object, job: dict[str, object], _bindings: object) -> dict[str, object]:
        observed.append(job)
        return _mock_job_outputs(pipeline.run_root / "scientific" / "internal_cv" / "jobs", job)

    monkeypatch.setattr(stages, "_execute_training_job", fake_execute)
    result = stages._internal_cv(pipeline, "internal_cv")

    assert len(observed) == 40
    assert {(job["fold"], job["seed"]) for job in observed} == set(zip(range(1, 11), HST_SEEDS))
    assert {job["model_name"] for job in observed} == {"hst_base"}
    assert sum(job["analysis_queue"] == "primary" for job in observed) == 20
    assert sum(job["analysis_queue"] == "secondary" for job in observed) == 10
    assert sum(job["analysis_queue"] == "exploratory" for job in observed) == 10
    task2_jobs = [job for job in observed if job["manifest_name"] == "task2_like_cough"]
    assert len(task2_jobs) == 10
    assert all(job["analysis_role"] == "exploratory" for job in task2_jobs)
    assert all(job["analysis_queue"] != "primary" for job in task2_jobs)
    predictions = pd.read_csv(pipeline.run_root / "scientific" / "internal_cv" / "participant_predictions.csv")
    metrics = pd.read_csv(pipeline.run_root / "scientific" / "internal_cv" / "metrics.csv")
    sources = pd.read_csv(pipeline.run_root / "scientific" / "internal_cv" / "source_checkpoints.csv")
    assert len(predictions) == len(metrics) == len(sources) == 40
    assert {"cough", "speech", "breath"} == set(metrics["modality"])
    assert {
        "manifest_sha256",
        "source_fold_rows_sha256",
        "training_rows_sha256",
        "training_contract_fingerprint",
    }.issubset(sources.columns)
    assert sources["source_fold_rows_sha256"].astype(str).str.len().eq(64).all()
    assert sources["training_rows_sha256"].astype(str).str.len().eq(64).all()
    assert all(Path(path).is_file() for path in result["output_paths"])


def test_split_and_reverse_jobs_bind_each_frozen_manifest_separately(tmp_path: Path) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path)
    manifests: dict[str, pd.DataFrame] = {}
    protocols = {
        "calendar_mixed": "hst_calendar_mixed_split_policy",
        "early_to_late": "hst_chronological_split_policy",
        "common_late_mixed": "hst_common_late_test_date_balanced_source",
        "common_late_chronological": "hst_common_late_test_chronological_source",
        "reverse_temporal": "hst_reverse_temporal_sensitivity",
    }
    for name, protocol in protocols.items():
        manifests[name] = _manifest_rows(
            protocol=protocol,
            folds=(1,),
            seeds=(42,),
            modalities=("cough", "speech"),
        )
        if name == "reverse_temporal":
            manifests[name]["analysis_scope"] = "sensitivity_analysis"
            manifests[name]["analysis_role"] = "sensitivity"
            manifests[name]["confirmatory_protocol"] = False
        else:
            manifests[name]["analysis_scope"] = "reliability_evaluation"
            manifests[name]["analysis_role"] = "secondary"
            manifests[name]["confirmatory_protocol"] = True
    _write_manifests(pipeline, manifests)

    split_jobs = stages._enumerate_training_jobs(
        pipeline,
        stage="split_policy_contrast",
        manifest_names=(
            "calendar_mixed",
            "early_to_late",
            "common_late_mixed",
            "common_late_chronological",
        ),
        modalities=("cough", "speech"),
        primary_modalities=("cough", "speech"),
    )
    reverse_jobs = stages._enumerate_training_jobs(
        pipeline,
        stage="reverse_temporal",
        manifest_names=("reverse_temporal",),
        modalities=("cough", "speech"),
        primary_modalities=(),
    )

    assert len(split_jobs) == 8
    assert len(reverse_jobs) == 2
    assert len({job["manifest_sha256"] for job in split_jobs}) == 4
    assert {job["protocol"] for job in reverse_jobs} == {"hst_reverse_temporal_sensitivity"}
    assert all(job["seed"] == 42 and job["fold"] == 1 for job in split_jobs + reverse_jobs)


def test_task2_like_manifest_cannot_be_promoted_into_primary_track_a(tmp_path: Path) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path)
    promoted = _manifest_rows(
        protocol="hst_literature_aligned_repeated_holdout",
        folds=tuple(range(1, 11)),
        seeds=HST_SEEDS,
        modalities=("cough",),
    )
    promoted["analysis_role"] = "primary"
    promoted["analysis_scope"] = "internal_performance"
    promoted["confirmatory_protocol"] = True
    _write_manifests(pipeline, {"task2_like_cough": promoted})

    with pytest.raises(ValueError, match="Task-2-like.*exploratory"):
        stages._enumerate_training_jobs(
            pipeline,
            stage="internal_cv",
            manifest_names=("task2_like_cough",),
            modalities=("cough",),
            primary_modalities=(),
        )


def test_external_manifest_source_rows_are_exact_frozen_internal_track_a_rows(
    tmp_path: Path,
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path)
    internal = _manifest_rows(
        protocol="hst_literature_aligned_repeated_holdout",
        folds=tuple(range(1, 11)),
        seeds=HST_SEEDS,
        modalities=("cough", "speech", "breath"),
    )
    external, internal_sha256 = _write_external_manifest_bound_to_internal(
        pipeline, stages, internal
    )

    stages._validate_external_source_binding(
        external,
        internal,
        internal_manifest_sha256=internal_sha256,
    )

    tampered = external.copy()
    source_index = tampered.index[tampered["dataset"].eq("coswara")][0]
    tampered.loc[source_index, "split"] = "validation"
    with pytest.raises(ValueError, match="exact frozen internal Track-A cough rows"):
        stages._validate_external_source_binding(
            tampered,
            internal,
            internal_manifest_sha256=internal_sha256,
        )


def test_external_transfer_reuses_all_internal_cough_checkpoints_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path)
    internal = _manifest_rows(
        protocol="hst_literature_aligned_repeated_holdout",
        folds=tuple(range(1, 11)),
        seeds=HST_SEEDS,
        modalities=("cough", "speech", "breath"),
    )
    external, internal_sha256 = _write_external_manifest_bound_to_internal(
        pipeline, stages, internal
    )
    external["analysis_scope"] = "reliability_evaluation"
    external["analysis_role"] = "secondary"
    external["confirmatory_protocol"] = True
    external_path = pipeline.run_root / "manifests" / "external.csv"
    external.to_csv(external_path, index=False)
    manifest_index_path = pipeline.run_root / "manifests" / "manifest_index.json"
    manifest_index = json.loads(manifest_index_path.read_text(encoding="utf-8"))
    manifest_index["manifests"]["external"].update(
        {"sha256": _digest(external_path), "rows": len(external)}
    )
    manifest_index_path.write_text(
        json.dumps(manifest_index, sort_keys=True), encoding="utf-8"
    )
    source_root = pipeline.run_root / "scientific" / "internal_cv"
    receipt_root = source_root / "jobs"
    source_rows = []
    for fold, seed in zip(range(1, 11), HST_SEEDS, strict=True):
        job_id = f"source-{fold}"
        job_root = receipt_root / job_id
        job_root.mkdir(parents=True)
        checkpoint = job_root / "best.pt"
        checkpoint.write_bytes(f"checkpoint-{fold}".encode("ascii"))
        receipt = job_root / "job_receipt.json"
        training_rows = internal.loc[
            internal["fold"].eq(fold)
            & internal["modality"].eq("cough")
            & internal["split"].isin(["train", "validation"])
        ]
        fold_rows = internal.loc[
            internal["fold"].eq(fold) & internal["modality"].eq("cough")
        ]
        training_job_spec_sha256 = f"{fold:x}" * 64
        training_job_spec_sha256 = training_job_spec_sha256[:64]
        receipt.write_text(
            json.dumps(
                {
                    "status": "success",
                    "job_id": job_id,
                    "job_spec_sha256": training_job_spec_sha256,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        source_rows.append(
            {
                "training_job_id": job_id,
                "training_job_spec_sha256": training_job_spec_sha256,
                "fold": fold,
                "seed": seed,
                "modality": "cough",
                "protocol": "hst_literature_aligned_repeated_holdout",
                "manifest_name": "internal",
                "manifest_sha256": internal_sha256,
                "source_fold_rows_sha256": stages._external_source_rows_sha256(
                    fold_rows
                ),
                "training_rows_sha256": stages._job_manifest_rows_sha256(
                    training_rows
                ),
                "training_contract_fingerprint": "6" * 64,
                "best_checkpoint_path": checkpoint.as_posix(),
                "best_checkpoint_sha256": _digest(checkpoint),
                "source_job_receipt_path": receipt.as_posix(),
                "source_job_receipt_sha256": _digest(receipt),
                "validation_threshold": 0.5,
            }
        )
    source_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows).to_csv(source_root / "source_checkpoints.csv", index=False)
    monkeypatch.setattr(stages, "_load_confirmatory_bindings", lambda _pipeline: {"verified": True})
    monkeypatch.setattr(
        stages,
        "train_hst_fold",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("external target was trained")),
    )
    observed: list[dict[str, object]] = []

    def fake_external(_pipeline: object, job: dict[str, object], _bindings: object) -> dict[str, object]:
        observed.append(job)
        root = pipeline.run_root / "scientific" / "external_transfer" / "jobs"
        result = _mock_job_outputs(root, {**job, "stage": "external_transfer", "modality": "cough"})
        result["participant_predictions"]["dataset"] = "coughvid"
        result["participant_predictions"]["split"] = "external_test"
        result["metrics"]["metric_split"] = "external_test"
        return result

    monkeypatch.setattr(stages, "_execute_external_job", fake_external)
    publication_paths: dict[str, Path] = {}
    for split in ("validation", "test"):
        path = source_root / f"publication_internal_{split}_predictions.csv"
        pd.DataFrame(
            {
                "fold": list(range(1, 11)),
                "modality": ["cough"] * 10,
                "split": [split] * 10,
            }
        ).to_csv(path, index=False)
        publication_paths[split] = path
    monkeypatch.setattr(
        stages,
        "_verified_stage_receipt",
        lambda *_args: (source_root / "stage.json", {"output_paths": []}, "7" * 64),
    )
    monkeypatch.setattr(
        stages,
        "_receipt_output_file",
        lambda _pipeline, _receipt, suffix: publication_paths[
            "validation" if "validation" in suffix else "test"
        ],
    )
    result = stages._external_transfer(pipeline, "external_transfer")

    assert len(observed) == 10
    assert {job["source_training_job_id"] for job in observed} == {
        f"source-{fold}" for fold in range(1, 11)
    }
    assert all(job["target_fit"] is False for job in observed)
    assert all(job["target_selection"] is False for job in observed)
    assert all(job["source_manifest_sha256"] == internal_sha256 for job in observed)
    assert all(len(str(job["source_fold_rows_sha256"])) == 64 for job in observed)
    assert all(len(str(job["source_training_rows_sha256"])) == 64 for job in observed)
    assert all(job["source_training_contract_fingerprint"] == "6" * 64 for job in observed)
    assert all(_digest(Path(job["source_checkpoint_path"])) == job["source_checkpoint_sha256"] for job in observed)
    predictions = pd.read_csv(pipeline.run_root / "scientific" / "external_transfer" / "participant_predictions.csv")
    assert predictions["dataset"].eq("coughvid").all()
    assert predictions["split"].eq("external_test").all()
    assert all(Path(path).is_file() for path in result["output_paths"])


def test_external_manifest_rejects_any_coughvid_training_or_selection_rows() -> None:
    import covid_audio_btp.hst_stages as stages

    frame = pd.DataFrame(
        {
            "dataset": ["coswara", "coughvid"],
            "split": ["train", "train"],
            "modality": ["cough", "cough"],
            "analysis_scope": ["reliability_evaluation"] * 2,
            "analysis_role": ["secondary"] * 2,
            "confirmatory_protocol": [True] * 2,
        }
    )
    with pytest.raises(ValueError, match="COUGHVID.*external_test"):
        stages._validate_external_manifest_scope(frame)


def test_external_job_is_inference_only_and_exports_participant_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path)
    source_checkpoint = pipeline.run_root / "source-best.pt"
    source_checkpoint.write_bytes(b"trained-source")
    initial_checkpoint = tmp_path / "initial-base.pt"
    initial_checkpoint.write_bytes(b"imagenet-base")
    manifest_path = pipeline.run_root / "manifests" / "external.csv"
    source_manifest_path = pipeline.run_root / "manifests" / "internal.csv"
    cache_path = pipeline.run_root / "manifests" / "spectrogram_cache_index.csv"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame({"placeholder": [1]}).to_csv(manifest_path, index=False)
    pd.DataFrame({"placeholder": [1]}).to_csv(cache_path, index=False)
    source_rows = _manifest_rows(
        protocol="hst_literature_aligned_repeated_holdout",
        folds=(1,),
        seeds=(1,),
        modalities=("cough",),
    )
    source_rows.to_csv(source_manifest_path, index=False)
    source_receipt = pipeline.run_root / "scientific" / "internal_cv" / "source.json"
    source_receipt.parent.mkdir(parents=True)
    source_receipt.write_text(
        json.dumps(
            {
                "status": "success",
                "job_id": "internal-source-1",
                "job_spec_sha256": "5" * 64,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    source_fold_rows_sha256 = stages._external_source_rows_sha256(source_rows)
    source_training_rows_sha256 = stages._job_manifest_rows_sha256(
        source_rows.loc[source_rows["split"].isin(["train", "validation"])]
    )
    job = {
        "stage": "external_transfer",
        "job_id": "external-job",
        "job_spec_sha256": "9" * 64,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": _digest(manifest_path),
        "cache_index_path": cache_path.as_posix(),
        "cache_index_sha256": _digest(cache_path),
        "protocol": "coswara_to_coughvid_hst_external",
        "source_protocol": "hst_literature_aligned_repeated_holdout",
        "fold": 1,
        "seed": 1,
        "modality": "cough",
        "source_training_job_id": "internal-source-1",
        "source_training_job_spec_sha256": "5" * 64,
        "source_manifest_path": source_manifest_path.as_posix(),
        "source_manifest_sha256": _digest(source_manifest_path),
        "source_fold_rows_sha256": source_fold_rows_sha256,
        "source_training_rows_sha256": source_training_rows_sha256,
        "source_training_contract_fingerprint": "6" * 64,
        "source_checkpoint_path": source_checkpoint.as_posix(),
        "source_checkpoint_sha256": _digest(source_checkpoint),
        "source_job_receipt_path": source_receipt.as_posix(),
        "source_job_receipt_sha256": _digest(source_receipt),
        "source_validation_threshold": 0.4,
        "target_fit": False,
        "target_selection": False,
        "analysis_queue": "secondary",
        "analysis_role": "secondary",
        "analysis_scope": "reliability_evaluation",
        "estimand_id": "coswara_to_coughvid_external_transfer",
        "multiplicity_family": "prespecified_reliability",
    }
    bindings = {
        "source_checkpoint_path": initial_checkpoint,
        "source_checkpoint_sha256": _digest(initial_checkpoint),
        "physical_batch_size": 2,
        "executable_sha256": "8" * 64,
        "executable_paths": list(pipeline.config.source_paths),
        "executable_allowlist": {"stage.py": _digest(pipeline.config.source_paths[0])},
    }

    class FakeModel:
        def to(self, _device: object) -> "FakeModel":
            return self

        def load_state_dict(self, state: object) -> None:
            assert state == {"weight": 1}

    monkeypatch.setattr(stages, "load_verified_hst_model", lambda **_kwargs: (FakeModel(), {}))
    monkeypatch.setattr(
        stages,
        "_load_verified_checkpoint_with_path",
        lambda path: (
            {
                "checkpoint_role": "best",
                "model_state_dict": {"weight": 1},
                "architecture_sha256": "7" * 64,
                "executable_sha256": "8" * 64,
                "prediction_context": {
                    "checkpoint_hash": _digest(initial_checkpoint),
                    "executable_sha256": "8" * 64,
                },
                "execution_identity": {
                    "fold": 1,
                    "modality": "cough",
                    "model_seed": 1,
                },
                "training_contract_fingerprint": "6" * 64,
            },
            Path(path),
        ),
    )
    monkeypatch.setattr(stages, "make_hst_dataloaders", lambda *_args, **_kwargs: {"external_test": object()})
    monkeypatch.setattr(stages, "_model_architecture_sha256", lambda _model: "7" * 64)
    monkeypatch.setattr(stages, "verify_executable_allowlist", lambda **_kwargs: "8" * 64)
    monkeypatch.setattr(
        stages,
        "train_hst_fold",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("target training called")),
    )

    def fake_predict(
        _model: object,
        _loader: object,
        *,
        prediction_context: dict[str, str],
        **_kwargs: object,
    ) -> pd.DataFrame:
        rows = []
        for index, (label, probability) in enumerate(
            (("negative", 0.2), ("positive", 0.8)), start=1
        ):
            rows.append(
                {
                    "run_id": prediction_context["run_id"],
                    "protocol": prediction_context["protocol"],
                    "fold": 1,
                    "dataset": "coughvid",
                    "participant_key": f"coughvid::p{index}",
                    "participant_id": f"p{index}",
                    "recording_key": f"coughvid::r{index}",
                    "split": "external_test",
                    "modality": "cough",
                    "model": "hst_base",
                    "checkpoint_hash": prediction_context["checkpoint_hash"],
                    "representation": "paper_logmel_224",
                    "label_binary": label,
                    "probability": probability,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(stages, "predict_hst_split", fake_predict)
    result = stages._execute_external_job(pipeline, job, bindings)

    assert result["participant_predictions"]["participant_key"].nunique() == 2
    assert result["metrics"].loc[0, "threshold_source"] == "source_validation_balanced_accuracy"
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["status"] == "success"
    assert receipt["target_fit"] is False
    assert receipt["target_selection"] is False


def test_external_checkpoint_identity_must_match_bound_internal_training_job() -> None:
    import covid_audio_btp.hst_stages as stages

    job = {
        "fold": 1,
        "seed": 1,
        "source_training_contract_fingerprint": "6" * 64,
    }
    payload = {
        "execution_identity": {
            "fold": 1,
            "modality": "cough",
            "model_seed": 1,
        },
        "training_contract_fingerprint": "6" * 64,
    }
    stages._validate_external_checkpoint_binding(payload, job)

    mismatched = {**payload, "training_contract_fingerprint": "7" * 64}
    with pytest.raises(ValueError, match="training contract"):
        stages._validate_external_checkpoint_binding(mismatched, job)


def test_checksum_valid_job_receipt_is_the_only_reusable_success(tmp_path: Path) -> None:
    import covid_audio_btp.hst_stages as stages

    output = tmp_path / "predictions.csv"
    output.write_text("probability\n0.5\n", encoding="utf-8")
    receipt = tmp_path / "job_receipt.json"
    job_spec_sha256 = "a" * 64
    stages._write_job_receipt(
        receipt,
        {
            "schema_version": 1,
            "status": "success",
            "run_id": "run-1",
            "job_id": "job-1",
            "job_spec_sha256": job_spec_sha256,
            "outputs": stages._output_records([output], root=tmp_path),
        },
    )

    assert stages._validated_reusable_job(
        receipt,
        job_spec_sha256=job_spec_sha256,
        run_id="run-1",
        job_id="job-1",
        root=tmp_path,
    )
    forged = json.loads(receipt.read_text(encoding="utf-8"))
    forged["run_id"] = "forged-run"
    receipt.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="identity|self-hash"):
        stages._validated_reusable_job(
            receipt,
            job_spec_sha256=job_spec_sha256,
            run_id="run-1",
            job_id="job-1",
            root=tmp_path,
        )
    stages._write_job_receipt(
        receipt,
        {
            "schema_version": 1,
            "status": "success",
            "run_id": "run-1",
            "job_id": "job-1",
            "job_spec_sha256": job_spec_sha256,
            "outputs": stages._output_records([output], root=tmp_path),
        },
    )
    output.write_text("probability\n0.9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        stages._validated_reusable_job(
            receipt,
            job_spec_sha256=job_spec_sha256,
            run_id="run-1",
            job_id="job-1",
            root=tmp_path,
        )


def test_confirmatory_bindings_require_exact_accepted_data_environment_and_pilot(
    tmp_path: Path,
) -> None:
    import covid_audio_btp.hst_stages as stages

    pipeline = _pipeline(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "hst_base.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"base")
    pipeline.config.scientific_config["checkpoints"]["hst_base_imagenet"].update(
        {"size_bytes": checkpoint.stat().st_size, "sha256": _digest(checkpoint)}
    )
    contracts = pipeline.run_root / "contracts" / "data_contracts_freeze.json"
    environment = pipeline.run_root / "audits" / "environment.json"
    pilot = pipeline.run_root / "audits" / "base_resource_pilot_freeze.json"
    contracts.parent.mkdir(parents=True)
    environment.parent.mkdir(parents=True)
    contracts.write_text(json.dumps({"manifest_sha256": "a" * 64}), encoding="utf-8")
    environment.write_text(json.dumps({"pip_freeze_sha256": "c" * 64}), encoding="utf-8")
    pilot.write_text(
        json.dumps(
            {
                "pilot_freeze_hash": "b" * 64,
                "physical_batch_size": 4,
                "gradient_accumulation": 2,
                "amp": False,
            }
        ),
        encoding="utf-8",
    )

    bindings = stages._load_confirmatory_bindings(pipeline)
    assert bindings["source_checkpoint_sha256"] == _digest(checkpoint)
    assert bindings["approved_resource_pairs"] == ((4, 2),)

    pipeline.config.accepted_hashes["data_contracts_freeze"] = "f" * 64
    with pytest.raises(ValueError, match="data_contracts_freeze"):
        stages._load_confirmatory_bindings(pipeline)


def test_runtime_projection_workload_is_modality_specific_and_source_only() -> None:
    import covid_audio_btp.hst_stages as stages

    metadata = pd.DataFrame(
        {
            "dataset": [
                "coswara",
                "coswara",
                "coswara",
                "coswara",
                "coswara",
                "coswara",
                "coswara",
                "coughvid",
            ],
            "participant_key": ["cn1", "cn1", "cn2", "cp1", "sn", "sp", "bn", "external"],
            "label_binary": [
                "negative",
                "negative",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
            ],
            "modality": ["cough", "cough", "cough", "cough", "speech", "speech", "breath", "cough"],
        }
    )
    metadata = pd.concat(
        [
            metadata,
            pd.DataFrame(
                {
                    "dataset": ["coswara"],
                    "participant_key": ["bp"],
                    "label_binary": ["positive"],
                    "modality": ["breath"],
                }
            ),
        ],
        ignore_index=True,
    )

    updates, jobs = stages._frozen_runtime_projection_workload(
        metadata,
        project_seeds=(1, 2, 5, 12, 40, 52, 72, 2002, 4002, 6002),
        primary_modalities=("cough", "speech"),
        secondary_modalities=("breath",),
        effective_batch_size=2,
    )

    assert updates == {"cough": 2, "speech": 1, "breath": 1}
    assert jobs == {"cough": 25, "speech": 15, "breath": 10}


def test_resource_pilot_uses_the_seed_bound_to_its_manifest_fold() -> None:
    import covid_audio_btp.hst_stages as stages

    manifest = _manifest_rows(
        protocol="hst_literature_aligned_repeated_holdout",
        folds=tuple(range(1, 11)),
        seeds=HST_SEEDS,
        modalities=("cough", "speech", "breath"),
    )

    assert stages._manifest_training_seed(manifest, fold=1, modality="cough") == 1
    assert stages._manifest_training_seed(manifest, fold=6, modality="cough") == 52
