import json
from pathlib import Path

import pandas as pd

from covid_rars.data_index import build_modality_availability, infer_modality


def test_infer_modality_from_common_coswara_names():
    assert infer_modality(Path("p1/heavy_cough.wav")) == ("cough", "heavy_cough")
    assert infer_modality(Path("p1/shallow_breath.wav")) == ("breath", "shallow_breath")
    assert infer_modality(Path("p1/counting_fast.wav")) == ("speech", "counting_fast")
    assert infer_modality(Path("p1/vowel_a.wav")) == ("speech", "vowel_a")


def test_build_modality_availability_marks_complete_case():
    metadata = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p1", "p2"],
            "modality": ["cough", "breath", "speech", "cough"],
        }
    )
    availability = build_modality_availability(metadata).set_index("participant_id")
    assert bool(availability.loc["p1", "complete_case"]) is True
    assert bool(availability.loc["p2", "complete_case"]) is False
    assert availability.loc["p1", "available_modalities"] == "breath,cough,speech"




def test_infer_participant_id_prefers_known_metadata_id(tmp_path):
    from covid_rars.data_index import infer_participant_id

    root = tmp_path / "coswara"
    audio = root / "20200413" / "abc123" / "audio" / "cough-heavy.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"")

    assert infer_participant_id(audio, root, known_participant_ids={"abc123"}) == "abc123"


def test_build_audio_index_attaches_manual_quality_from_combined_metadata(tmp_path):
    from covid_rars.data_index import build_audio_index

    root = tmp_path / "coswara"
    audio = root / "20200413" / "abc123" / "cough-heavy.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"fake wav bytes")
    pd.DataFrame(
        {
            "id": ["abc123"],
            "covid_status": ["healthy"],
            "cough_heavy_quality": [0],
        }
    ).to_csv(root / "combined_data.csv", index=False)

    index = build_audio_index(root)

    row = index.iloc[0]
    assert row["participant_id"] == "abc123"
    assert row["submodality"] == "heavy_cough"
    assert row["manual_quality_score"] == 0
    assert row["manual_quality_label"] == "bad"


def test_build_audio_index_uses_official_coswara_short_metadata_columns(tmp_path):
    from covid_rars.data_index import build_audio_index

    root = tmp_path / "coswara"
    audio = root / "20200413" / "abc123" / "cough-heavy.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"fake wav bytes")
    pd.DataFrame(
        {
            "id": ["abc123"],
            "a": [21],
            "covid_status": ["positive_mild"],
            "record_date": ["2020-04-13"],
            "g": ["male"],
            "l_c": ["India"],
            "cough": [True],
            "fever": [False],
            "asthma": [False],
            "testType": ["rtpcr"],
            "test_status": ["positive"],
        }
    ).to_csv(root / "combined_data.csv", index=False)

    index = build_audio_index(root)

    row = index.iloc[0]
    symptoms = json.loads(row["symptoms_json"])
    comorbidities = json.loads(row["comorbidities_json"])
    assert row["participant_id"] == "abc123"
    assert row["label_binary"] == "positive"
    assert row["age"] == 21
    assert row["gender"] == "male"
    assert row["country"] == "India"
    assert row["recording_date"] == "2020-04-13"
    assert symptoms["cough"] is True
    assert symptoms["fever"] is False
    assert comorbidities["asthma"] is False
    assert row["test_type"] == "rtpcr"
    assert row["test_status"] == "positive"



def test_discover_audio_files_ignores_appledouble_and_macosx_sidecars(tmp_path):
    from covid_rars.data_index import discover_audio_files

    root = tmp_path / "coswara"
    real = root / "20200413" / "abc123" / "cough-heavy.wav"
    sidecar = real.parent / "._cough-heavy.wav"
    macosx = root / "__MACOSX" / "20200413" / "abc123" / "cough-heavy.wav"
    real.parent.mkdir(parents=True)
    macosx.parent.mkdir(parents=True)
    real.write_bytes(b"fake wav bytes")
    sidecar.write_bytes(b"appledouble metadata")
    macosx.write_bytes(b"appledouble archive metadata")

    assert discover_audio_files(root) == [real]
