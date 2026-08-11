from __future__ import annotations

import numpy as np

from covid_rars.ssl_extractors import embedding_array_from_model_output, normalize_backend_name


def test_embedding_array_from_model_output_uses_last_torchaudio_layer() -> None:
    first = np.zeros((2, 3, 4), dtype=np.float32)
    last = np.ones((2, 3, 4), dtype=np.float32)

    extracted = embedding_array_from_model_output(([first, last], None))

    assert extracted.shape == (2, 3, 4)
    assert np.allclose(extracted, 1.0)


def test_embedding_array_from_model_output_uses_panns_embedding_key() -> None:
    embedding = np.ones((2, 2048), dtype=np.float32)
    output = {"clipwise_output": np.zeros((2, 527), dtype=np.float32), "embedding": embedding}

    extracted = embedding_array_from_model_output(output)

    assert extracted.shape == (2, 2048)
    assert np.allclose(extracted, embedding)


def test_normalize_backend_name_accepts_expected_aliases() -> None:
    assert normalize_backend_name("wav2vec2") == "wav2vec2_torchaudio"
    assert normalize_backend_name("beats") == "beats_official"
    assert normalize_backend_name("panns") == "panns_cnn14_official"
