from __future__ import annotations

import numpy as np
import pandas as pd
from covid_rars.representation_features import validate_feature_table
from covid_rars.torch_embedding_features import (
    extract_torch_embedding_feature_table,
    pool_time_sequence,
)


class _FakeBatchExtractor:
    representation_name = "fake_ssl"
    sample_rate = 16_000

    def embed_batch(self, waveforms: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                waveforms.mean(axis=1),
                waveforms.std(axis=1),
            ],
            axis=1,
        )


def test_pool_time_sequence_mean_pools_three_dimensional_embeddings() -> None:
    sequence = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ],
        dtype=np.float32,
    )

    pooled = pool_time_sequence(sequence)

    assert np.allclose(pooled, np.asarray([[2.0, 3.0], [4.0, 6.0]], dtype=np.float32))


def test_extract_torch_embedding_feature_table_batches_and_filters_quality(monkeypatch) -> None:
    metadata = pd.DataFrame(
        {
            "recording_id": ["r1", "r2", "r3"],
            "participant_id": ["p1", "p2", "p3"],
            "dataset": ["coswara", "coswara", "coswara"],
            "modality": ["cough", "cough", "breath"],
            "submodality": ["heavy_cough", "heavy_cough", "deep_breath"],
            "label_binary": ["positive", "negative", "negative"],
            "split": ["train", "test", "test"],
            "quality_flag": ["ok", "corrupt", "ok"],
            "audio_path": ["a.wav", "b.wav", "c.wav"],
        }
    )

    monkeypatch.setattr(
        "covid_rars.torch_embedding_features.load_audio",
        lambda path, config=None: (np.ones(config.sample_rate, dtype=np.float32), config.sample_rate, config.sample_rate),
    )
    monkeypatch.setattr(
        "covid_rars.torch_embedding_features.preprocess_for_features",
        lambda y, sample_rate, modality, config=None: (
            y.astype(np.float32, copy=False),
            {
                "event_start_sec": 0.0,
                "event_end_sec": 1.0,
                "event_duration_sec": 1.0,
                "active_audio_ratio": 1.0,
                "segmentation_method": "test",
            },
        ),
    )

    features = extract_torch_embedding_feature_table(
        metadata,
        extractor=_FakeBatchExtractor(),
        quality_mode="quality_ok_only",
        modality="cough",
        batch_size=2,
    )

    assert features["recording_id"].tolist() == ["r1"]
    assert features["representation"].tolist() == ["fake_ssl"]
    assert "fake_ssl_dim_0000" in features.columns
    assert "fake_ssl_dim_0001" in features.columns
    validate_feature_table(features)
