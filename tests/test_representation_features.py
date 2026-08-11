from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covid_rars.representation_features import (
    read_feature_table,
    representation_feature_columns,
    validate_feature_table,
    write_feature_table,
)


def _feature_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recording_id": ["r1", "r2"],
            "participant_id": ["p1", "p2"],
            "dataset": ["coswara", "coswara"],
            "modality": ["cough", "cough"],
            "submodality": ["heavy_cough", "heavy_cough"],
            "label_binary": ["positive", "negative"],
            "split": ["train", "test"],
            "representation": ["opensmile_egemaps", "opensmile_egemaps"],
            "opensmile_egemaps_f0": [1.0, 2.0],
            "opensmile_egemaps_energy": [0.25, 0.5],
            "free_text_note": ["keep", "excluded"],
        }
    )


def test_representation_feature_columns_excludes_ids_and_non_numeric_metadata() -> None:
    df = _feature_table()

    assert representation_feature_columns(df) == [
        "opensmile_egemaps_f0",
        "opensmile_egemaps_energy",
    ]


def test_validate_feature_table_rejects_missing_required_columns() -> None:
    df = _feature_table().drop(columns=["label_binary"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_feature_table(df)


def test_validate_feature_table_rejects_nan_and_infinite_features() -> None:
    with_nan = _feature_table()
    with_nan.loc[0, "opensmile_egemaps_f0"] = np.nan
    with_inf = _feature_table()
    with_inf.loc[1, "opensmile_egemaps_energy"] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        validate_feature_table(with_nan)
    with pytest.raises(ValueError, match="non-finite"):
        validate_feature_table(with_inf)


def test_csv_feature_table_roundtrip_preserves_columns(tmp_path) -> None:
    df = _feature_table()
    output = tmp_path / "features.csv"

    write_feature_table(df, output)
    loaded = read_feature_table(output)

    assert list(loaded.columns) == list(df.columns)
    validate_feature_table(loaded)
