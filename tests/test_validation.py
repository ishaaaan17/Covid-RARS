import pandas as pd

from covid_rars.validation import validate_index, validate_metadata


def test_validate_index_flags_missing_labels():
    index = pd.DataFrame(
        {
            "participant_id": ["p1"],
            "recording_id": ["r1"],
            "dataset": ["coswara"],
            "modality": ["cough"],
            "submodality": ["heavy_cough"],
            "audio_path": ["/tmp/does-not-exist.wav"],
            "label_binary": ["unknown"],
        }
    )
    issues = validate_index(index)
    checks = {issue.check for issue in issues}
    assert "label_coverage" in checks


def test_validate_metadata_detects_leakage():
    metadata = pd.DataFrame(
        {
            "participant_id": ["p1", "p1"],
            "recording_id": ["r1", "r2"],
            "modality": ["cough", "breath"],
            "label_binary": ["positive", "positive"],
            "split": ["train", "test"],
            "audio_path": ["a.wav", "b.wav"],
        }
    )
    issues = validate_metadata(metadata)
    checks = {issue.check for issue in issues}
    assert "participant_leakage" in checks
