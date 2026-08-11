from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script_module():
    script_path = Path(__file__).parents[1] / "scripts" / "27_extract_opensmile_features.py"
    spec = importlib.util.spec_from_file_location("extract_opensmile_features_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_opensmile_cli_writes_validated_feature_table(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    metadata_path = tmp_path / "metadata.csv"
    output_path = tmp_path / "opensmile_features.csv"
    pd.DataFrame(
        {
            "recording_id": ["r1"],
            "participant_id": ["p1"],
            "dataset": ["coswara"],
            "modality": ["cough"],
            "submodality": ["heavy_cough"],
            "label_binary": ["positive"],
            "audio_path": ["dummy.wav"],
        }
    ).to_csv(metadata_path, index=False)

    def fake_extract(metadata, **kwargs):
        assert metadata["split"].tolist() == ["external"]
        assert kwargs["feature_set"] == "egemaps"
        assert kwargs["quality_mode"] == "all_samples"
        return pd.DataFrame(
            {
                "recording_id": ["r1"],
                "participant_id": ["p1"],
                "dataset": ["coswara"],
                "modality": ["cough"],
                "submodality": ["heavy_cough"],
                "label_binary": ["positive"],
                "split": ["train"],
                "representation": ["opensmile_egemaps"],
                "opensmile_egemaps_energy": [0.5],
            }
        )

    monkeypatch.setattr(module, "extract_opensmile_feature_table", fake_extract)
    module.main(
        [
            "--metadata",
            str(metadata_path),
            "--output",
            str(output_path),
            "--feature-set",
            "egemaps",
            "--quality-mode",
            "all_samples",
            "--split-name",
            "external",
        ]
    )

    written = pd.read_csv(output_path)
    assert written["representation"].tolist() == ["opensmile_egemaps"]
