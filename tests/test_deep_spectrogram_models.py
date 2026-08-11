from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def test_spectrogram_model_factory_supports_strong_architectures() -> None:
    from covid_rars.models_cnn import make_spectrogram_model

    x = torch.randn(2, 1, 64, 120)
    for architecture in ("compact_cnn", "residual_cnn", "cnn_bigru"):
        model = make_spectrogram_model(architecture)
        logits = model(x)
        assert logits.shape == (2,)


def _spectrogram_index(tmp_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(24):
        label = "positive" if idx % 4 in {1, 2} else "negative"
        split = "train" if idx % 6 < 4 else "validation" if idx % 6 == 4 else "test"
        base = 1.0 if label == "positive" else -1.0
        spec = np.full((1, 64, 48), base, dtype=np.float32)
        spec += np.random.default_rng(idx).normal(0.0, 0.03, size=spec.shape).astype(np.float32)
        path = tmp_path / f"spec_{idx}.npy"
        np.save(path, spec)
        rows.append(
            {
                "recording_id": f"rec_{idx}",
                "participant_id": f"p_{idx}",
                "dataset": "coswara",
                "modality": "cough",
                "submodality": "cough",
                "label_binary": label,
                "split": split,
                "spectrogram_path": path.as_posix(),
            }
        )
    return pd.DataFrame(rows)


def test_deep_spectrogram_trainer_reports_architecture_and_validation_threshold(tmp_path: Path) -> None:
    from covid_rars.train_cnn import train_cnn_for_modality

    index = _spectrogram_index(tmp_path)
    artifacts = train_cnn_for_modality(
        index,
        modality="cough",
        architecture="residual_cnn",
        max_epochs=2,
        patience=1,
        batch_size=8,
        learning_rate=1e-3,
        augment=False,
        device="cpu",
    )

    assert artifacts.metrics["model_name"] == "residual_cnn"
    assert artifacts.metrics["threshold_source"] == "validation_balanced_accuracy"
    assert "validation_auroc" in artifacts.metrics
    assert not artifacts.validation_predictions.empty
    assert not artifacts.test_predictions.empty
    assert {"validation_auroc", "validation_loss"}.issubset(artifacts.history.columns)


def test_deep_spectrogram_trainer_reports_external_predictions_when_available(tmp_path: Path) -> None:
    from covid_rars.train_cnn import train_cnn_for_modality

    index = _spectrogram_index(tmp_path)
    external_rows: list[dict[str, object]] = []
    for idx, label in enumerate(("negative", "positive", "negative", "positive")):
        base = 1.0 if label == "positive" else -1.0
        spec = np.full((1, 64, 48), base, dtype=np.float32)
        path = tmp_path / f"external_spec_{idx}.npy"
        np.save(path, spec)
        external_rows.append(
            {
                "recording_id": f"external_rec_{idx}",
                "participant_id": f"external_p_{idx}",
                "dataset": "coughvid",
                "modality": "cough",
                "submodality": "cough",
                "label_binary": label,
                "split": "external_test",
                "spectrogram_path": path.as_posix(),
            }
        )
    index = pd.concat([index, pd.DataFrame(external_rows)], ignore_index=True)

    artifacts = train_cnn_for_modality(
        index,
        modality="cough",
        architecture="compact_cnn",
        max_epochs=1,
        patience=1,
        batch_size=8,
        learning_rate=1e-3,
        augment=False,
        device="cpu",
    )

    assert artifacts.external_predictions is not None
    assert not artifacts.external_predictions.empty
    assert artifacts.external_predictions["split"].eq("external_test").all()
    assert artifacts.external_metrics is not None
    assert artifacts.external_metrics["metric_split"] == "external_test"
