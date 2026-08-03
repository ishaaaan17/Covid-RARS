from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    audio = tmp_path / "sample.webm"
    audio.write_bytes(b"audio")
    cohort = tmp_path / "cohort.csv"
    raw = tmp_path / "metadata_compiled.csv"
    output = tmp_path / "hst.csv"
    pd.DataFrame(
        {
            "recording_id": ["rec-a", "rec-b"],
            "participant_id": ["uuid-a", "uuid-b"],
            "audio_path": [str(audio), str(audio)],
            "label_binary": ["negative", "positive"],
            "dataset": ["coughvid", "coughvid"],
        }
    ).to_csv(cohort, index=False)
    pd.DataFrame(
        {
            "uuid": ["uuid-a", "uuid-b", "unused"],
            "status_SSL": ["healthy", "COVID-19", None],
            "status": ["healthy", "COVID-19", "symptomatic"],
            "diagnosis_1": ["healthy_cough", "COVID-19", None],
        }
    ).to_csv(raw, index=False)
    return cohort, raw, output


def test_build_hst_coughvid_metadata_binds_labels_and_upstream_hashes(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_coughvid_metadata import build_hst_coughvid_metadata

    cohort, raw, output = _inputs(tmp_path)
    result = build_hst_coughvid_metadata(
        cohort_path=cohort,
        raw_metadata_path=raw,
        output_path=output,
    )
    frame = pd.read_csv(output)

    assert result["row_count"] == 2
    assert result["status_ssl_negative"] == 1
    assert result["status_ssl_positive"] == 1
    assert result["legacy_label_disagreement_count"] == 0
    assert frame["uuid"].tolist() == ["uuid-a", "uuid-b"]
    assert frame["status_SSL"].tolist() == ["healthy", "COVID-19"]
    assert frame["legacy_label_binary"].tolist() == ["negative", "positive"]
    assert frame["cohort_source_sha256"].nunique() == 1
    assert frame["label_metadata_source_sha256"].nunique() == 1


def test_build_hst_coughvid_metadata_rejects_label_disagreement(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_coughvid_metadata import build_hst_coughvid_metadata

    cohort, raw, output = _inputs(tmp_path)
    frame = pd.read_csv(cohort)
    frame.loc[0, "label_binary"] = "positive"
    frame.to_csv(cohort, index=False)

    with pytest.raises(ValueError, match="disagree"):
        build_hst_coughvid_metadata(
            cohort_path=cohort,
            raw_metadata_path=raw,
            output_path=output,
        )


def test_build_hst_coughvid_metadata_rejects_missing_uuid(tmp_path: Path) -> None:
    from covid_audio_btp.hst_coughvid_metadata import build_hst_coughvid_metadata

    cohort, raw, output = _inputs(tmp_path)
    frame = pd.read_csv(raw).iloc[1:].copy()
    frame.to_csv(raw, index=False)

    with pytest.raises(ValueError, match="absent"):
        build_hst_coughvid_metadata(
            cohort_path=cohort,
            raw_metadata_path=raw,
            output_path=output,
        )
