from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tests.hst_test_helpers import make_recording_predictions


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("COVID-19", "positive"),
        ("covid positive", "positive"),
        ("positive", "positive"),
        ("healthy", "negative"),
        ("negative", "negative"),
        ("no covid", "negative"),
        ("covid negative", "negative"),
        ("symptomatic", "unknown"),
        ("unreviewed", "unknown"),
        ("possibly covid", "unknown"),
        (None, "unknown"),
    ],
)
def test_coughvid_labels_fail_closed(raw: object, expected: str) -> None:
    from covid_audio_btp.hst_data_contracts import normalize_coughvid_status

    assert normalize_coughvid_status(raw) == expected


def test_positive_class_index_is_frozen() -> None:
    from covid_audio_btp.hst_data_contracts import CLASS_TO_INDEX

    assert CLASS_TO_INDEX == {"negative": 0, "positive": 1}


def test_coughvid_label_source_must_be_explicit(tmp_path: Path) -> None:
    from covid_audio_btp.hst_data_contracts import build_audited_coughvid_index

    with pytest.raises(TypeError):
        build_audited_coughvid_index(tmp_path)  # type: ignore[call-arg]


def test_audited_coughvid_csv_uses_only_requested_label_source(tmp_path: Path) -> None:
    from covid_audio_btp.hst_data_contracts import build_audited_coughvid_index

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "uuid": ["a"],
            "status_SSL": ["healthy"],
            "status": ["COVID-19"],
        }
    ).to_csv(metadata, index=False)
    index = build_audited_coughvid_index(
        metadata,
        label_column="status_SSL",
        dataset_release_id="coughvid-v3-7024894",
        source_manifest_sha256="b" * 64,
    )
    assert index.loc[0, "label_binary"] == "negative"
    assert index.loc[0, "status"] == "COVID-19"
    assert index.loc[0, "participant_key"] == "coughvid::a"
    assert index.loc[0, "identity_source_column"] == "uuid"
    assert index.loc[0, "analysis_unit_type"] == "recording_uuid"
    assert bool(index.loc[0, "participant_id_is_recording_proxy"])
    assert not bool(index.loc[0, "subject_linkage_available"])
    assert index.loc[0, "metadata_source_level"] == "derived_csv"


def test_audited_coughvid_csv_preserves_valid_declared_audio_paths(tmp_path: Path) -> None:
    from covid_audio_btp.hst_data_contracts import build_audited_coughvid_index

    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    audio = audio_root / "recording.wav"
    audio.write_bytes(b"RIFF")
    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    metadata = metadata_root / "index.csv"
    pd.DataFrame(
        {
            "uuid": ["recording"],
            "status_SSL": ["COVID-19"],
            "audio_path": [audio.as_posix()],
        }
    ).to_csv(metadata, index=False)

    index = build_audited_coughvid_index(
        metadata,
        label_column="status_SSL",
        dataset_release_id="coughvid-v3-7024894",
        source_manifest_sha256="b" * 64,
    )
    assert index.loc[0, "audio_path"] == audio.as_posix()
    assert bool(index.loc[0, "audio_exists"])


def test_audited_coughvid_csv_records_declared_processed_source_level(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_data_contracts import build_audited_coughvid_index

    audio = tmp_path / "recording.wav"
    audio.write_bytes(b"RIFF")
    metadata = tmp_path / "index.csv"
    pd.DataFrame(
        {
            "uuid": ["recording"],
            "status_SSL": ["healthy"],
            "audio_path": [audio.as_posix()],
        }
    ).to_csv(metadata, index=False)

    index = build_audited_coughvid_index(
        metadata,
        label_column="status_SSL",
        dataset_release_id="coughvid-v3-7024894",
        source_manifest_sha256="b" * 64,
        metadata_source_level="derived_processed_csv",
    )

    assert index["metadata_source_level"].eq("derived_processed_csv").all()


def test_hst_contract_does_not_depend_on_legacy_coughvid_adapter() -> None:
    source = (
        Path(__file__).parents[1] / "src" / "covid_audio_btp" / "hst_data_contracts.py"
    ).read_text(encoding="utf-8")
    assert "external_datasets" not in source
    assert "normalize_coughvid_label" not in source
    assert "build_coughvid_index" not in source


def _contract_metadata(**changes: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "dataset_release_id": "coughvid-v3-7024894",
        "label_column": "status_SSL",
        "label_normalization_version": 1,
        "source_manifest_sha256": "a" * 64,
        "eligibility_policy_version": 1,
    }
    metadata.update(changes)
    return metadata


def _freeze(tmp_path: Path, output_name: str, **metadata_changes: object) -> str:
    from covid_audio_btp.hst_data_contracts import freeze_data_contracts

    return freeze_data_contracts(
        source_root=tmp_path,
        audit_root=tmp_path,
        source_paths=(tmp_path / "metadata.csv",),
        label_audits=(tmp_path / "label_audit.csv",),
        contract_metadata=_contract_metadata(**metadata_changes),
        output_path=tmp_path / output_name,
    )


def test_data_contract_hash_changes_when_source_bytes_change(tmp_path: Path) -> None:
    source = tmp_path / "metadata.csv"
    audit = tmp_path / "label_audit.csv"
    source.write_text("recording_id,status_SSL\na,healthy\n", encoding="utf-8")
    audit.write_text("raw,normalized\nhealthy,negative\n", encoding="utf-8")
    first = _freeze(tmp_path, "a.json")
    source.write_text("recording_id,status_SSL\na,COVID-19\n", encoding="utf-8")
    second = _freeze(tmp_path, "b.json")
    assert first != second


def test_data_contract_hash_changes_when_release_or_label_policy_changes(tmp_path: Path) -> None:
    (tmp_path / "metadata.csv").write_text("id,status_SSL\na,healthy\n", encoding="utf-8")
    (tmp_path / "label_audit.csv").write_text("raw,normalized\nhealthy,negative\n", encoding="utf-8")
    first = _freeze(tmp_path, "a.json", dataset_release_id="release-a")
    second = _freeze(tmp_path, "b.json", dataset_release_id="release-b")
    third = _freeze(tmp_path, "c.json", dataset_release_id="release-a", label_normalization_version=2)
    assert first != second
    assert first != third


def test_data_contract_is_location_independent_and_atomic(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (left, right):
        root.mkdir()
        (root / "metadata.csv").write_text("id,status_SSL\na,healthy\n", encoding="utf-8")
        (root / "label_audit.csv").write_text("raw,normalized\nhealthy,negative\n", encoding="utf-8")
    assert _freeze(left, "contract.json") == _freeze(right, "contract.json")
    payload = json.loads((left / "contract.json").read_text(encoding="utf-8"))
    assert payload["manifest_sha256"]
    assert not (left / "contract.json.tmp").exists()


def test_data_contract_rejects_missing_metadata_and_path_escape(tmp_path: Path) -> None:
    from covid_audio_btp.hst_data_contracts import freeze_data_contracts

    root = tmp_path / "root"
    root.mkdir()
    source = root / "metadata.csv"
    audit = root / "audit.csv"
    outside = tmp_path / "outside.csv"
    source.write_text("x\n", encoding="utf-8")
    audit.write_text("x\n", encoding="utf-8")
    outside.write_text("x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required"):
        freeze_data_contracts(
            source_root=root,
            audit_root=root,
            source_paths=(source,),
            label_audits=(audit,),
            contract_metadata={"dataset_release_id": "x"},
            output_path=root / "contract.json",
        )
    with pytest.raises(ValueError, match="escapes"):
        freeze_data_contracts(
            source_root=root,
            audit_root=root,
            source_paths=(outside,),
            label_audits=(audit,),
            contract_metadata=_contract_metadata(),
            output_path=root / "contract.json",
        )


def test_qualified_identifiers_prevent_cross_dataset_collision() -> None:
    from covid_audio_btp.hst_data_contracts import qualify_identifiers

    frame = pd.DataFrame(
        {
            "dataset": ["coswara", "coughvid"],
            "participant_id": ["same", "same"],
            "recording_id": ["same", "same"],
        }
    )
    qualified = qualify_identifiers(frame)
    assert qualified["participant_key"].is_unique
    assert qualified["recording_key"].is_unique


def test_participant_aggregation_equalizes_recording_weight() -> None:
    from covid_audio_btp.hst_data_contracts import aggregate_to_participant

    participant = aggregate_to_participant(make_recording_predictions())
    assert participant.shape[0] == 2
    p1 = participant.loc[participant["participant_id"] == "p1"].iloc[0]
    assert p1["probability"] == pytest.approx(0.8)
    assert p1["n_recordings"] == 2


def test_external_recording_uuid_analysis_unit_survives_probability_aggregation() -> None:
    from covid_audio_btp.hst_data_contracts import aggregate_to_participant

    frame = make_recording_predictions().assign(
        analysis_unit_type="recording_uuid",
        identity_source_column="uuid",
        participant_id_is_recording_proxy=True,
        subject_linkage_available=False,
        metadata_source_level="derived_csv",
    )

    aggregated = aggregate_to_participant(frame)

    assert aggregated["analysis_unit_type"].eq("recording_uuid").all()
    assert aggregated["identity_source_column"].eq("uuid").all()
    assert aggregated["participant_id_is_recording_proxy"].eq(True).all()  # noqa: E712
    assert aggregated["subject_linkage_available"].eq(False).all()  # noqa: E712
    assert aggregated["metadata_source_level"].eq("derived_csv").all()


def test_prediction_contract_rejects_missing_fold_and_cross_fold_pooling() -> None:
    from covid_audio_btp.hst_data_contracts import aggregate_to_participant, assert_prediction_key_contract

    frame = make_recording_predictions()
    without_fold = frame.drop(columns="fold")
    with pytest.raises(ValueError, match="fold"):
        assert_prediction_key_contract(without_fold, repeated=True)

    duplicate_fold = frame.copy()
    duplicate_fold["fold"] = 2
    mixed = pd.concat([frame, duplicate_fold], ignore_index=True)
    participant = aggregate_to_participant(mixed)
    assert participant.shape[0] == 4
    assert set(participant["fold"]) == {1, 2}


def test_audit_reports_prior_supervised_label_changes() -> None:
    from covid_audio_btp.hst_data_contracts import audit_coughvid_labels, qualify_identifiers

    frame = qualify_identifiers(
        pd.DataFrame(
            {
                "dataset": ["coughvid", "coughvid"],
                "participant_id": ["a", "b"],
                "recording_id": ["a", "b"],
                "status_SSL": ["COVID-19", "healthy"],
                "status": ["positive", "negative"],
                "label_source": ["status_SSL", "status_SSL"],
            }
        )
    )
    prior = frame[["recording_key"]].copy()
    prior["label_binary"] = ["negative", "negative"]
    normalized, audit = audit_coughvid_labels(frame, prior=prior)
    assert normalized["label_binary"].tolist() == ["positive", "negative"]
    prior_row = audit.loc[audit["audit_type"] == "prior_comparison"].iloc[0]
    assert prior_row["changed_supervised_labels"] == 1
    assert bool(prior_row["invalidates_prior_metrics"])


def test_coughvid_label_audit_includes_expert_diagnosis_columns() -> None:
    from covid_audio_btp.hst_data_contracts import audit_coughvid_labels

    frame = pd.DataFrame(
        {
            "status_SSL": ["COVID-19", "healthy"],
            "diagnosis_1": ["COVID-19", "healthy_cough"],
            "label_source": ["status_SSL", "status_SSL"],
        }
    )

    _normalized, audit = audit_coughvid_labels(frame)

    assert "diagnosis_1" in set(audit.get("label_source", pd.Series(dtype=str)).dropna())
    pairwise = audit.loc[audit["audit_type"].eq("pairwise")]
    assert (
        pairwise[["left_label_source", "right_label_source"]]
        .astype(str)
        .eq("diagnosis_1")
        .any(axis=1)
        .any()
    )
