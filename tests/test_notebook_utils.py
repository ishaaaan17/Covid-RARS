import pandas as pd
import pytest

from covid_rars.notebook_utils import (
    assert_binary_labels_present,
    assert_no_participant_leakage,
    check_artifacts,
    stop_if_validation_errors,
)


def test_check_artifacts_reports_ok_missing_and_empty(tmp_path):
    ok = tmp_path / "ok.csv"
    empty = tmp_path / "empty.csv"
    ok.write_text("a\n1\n", encoding="utf-8")
    empty.write_text("", encoding="utf-8")

    status = check_artifacts([ok, empty, tmp_path / "missing.csv"]).set_index("path")

    assert status.loc[str(ok), "status"] == "ok"
    assert status.loc[str(empty), "status"] == "empty"
    assert status.loc[str(tmp_path / "missing.csv"), "status"] == "missing"


def test_assert_no_participant_leakage_rejects_split_overlap():
    metadata = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2"],
            "split": ["train", "test", "train"],
        }
    )

    with pytest.raises(AssertionError, match="Participant leakage"):
        assert_no_participant_leakage(metadata)


def test_assert_binary_labels_present_accepts_string_labels():
    metadata = pd.DataFrame({"label_binary": ["positive", "negative", "positive"]})

    assert_binary_labels_present(metadata)


def test_stop_if_validation_errors_rejects_error_severity():
    issues = pd.DataFrame({"severity": ["warning", "error"], "check": ["x", "y"]})

    with pytest.raises(AssertionError, match="validation gate failed"):
        stop_if_validation_errors(issues)
