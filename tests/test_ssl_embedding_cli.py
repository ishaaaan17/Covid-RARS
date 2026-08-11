from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


class _FakeExtractor:
    representation_name = "fake_ssl"
    sample_rate = 16_000


def _load_script_module():
    script_path = Path(__file__).parents[1] / "scripts" / "28_extract_ssl_embeddings.py"
    spec = importlib.util.spec_from_file_location("extract_ssl_embeddings_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ssl_embedding_cli_writes_validated_feature_table(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    metadata_path = tmp_path / "metadata.csv"
    output_path = tmp_path / "ssl_features.csv"
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

    monkeypatch.setattr(module, "create_ssl_extractor", lambda **kwargs: _FakeExtractor())
    monkeypatch.setattr(
        module,
        "extract_torch_embedding_feature_table",
        lambda metadata, **kwargs: (
            (_ for _ in ()).throw(AssertionError("split not external"))
            if metadata["split"].tolist() != ["external"]
            else pd.DataFrame(
            {
                "recording_id": ["r1"],
                "participant_id": ["p1"],
                "dataset": ["coswara"],
                "modality": ["cough"],
                "submodality": ["heavy_cough"],
                "label_binary": ["positive"],
                "split": ["train"],
                "representation": ["fake_ssl"],
                "fake_ssl_dim_0000": [0.25],
            }
        )
        ),
    )

    module.main(
        [
            "--metadata",
            str(metadata_path),
            "--output",
            str(output_path),
            "--backend",
            "wav2vec2",
            "--quality-mode",
            "all_samples",
            "--split-name",
            "external",
        ]
    )

    written = pd.read_csv(output_path)
    assert written["representation"].tolist() == ["fake_ssl"]
