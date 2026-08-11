from __future__ import annotations

import numpy as np
import pandas as pd


def _segment_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ("validation", "test"):
        for idx in range(12):
            label = "positive" if idx % 2 else "negative"
            base = 0.72 if label == "positive" else 0.28
            participant_id = f"{split}_p{idx:02d}"
            for model_name, offset in (("ssl_wavlm", 0.05), ("spectrogram_convnext", -0.02)):
                for seg in range(2):
                    rows.append(
                        {
                            "segment_id": f"{participant_id}_{model_name}_{seg}",
                            "recording_id": f"{participant_id}_rec",
                            "participant_id": participant_id,
                            "dataset": "coswara",
                            "modality": "cough",
                            "submodality": "cough",
                            "label_binary": label,
                            "split": split,
                            "model_name": model_name,
                            "analysis_family": "sota_ssl_branch" if model_name == "ssl_wavlm" else "sota_spectrogram_branch",
                            "evaluation_protocol": "sota_internal_protocol",
                            "probability": float(np.clip(base + offset + seg * 0.01, 0.001, 0.999)),
                        }
                    )
    return pd.DataFrame(rows)


def test_aggregate_sota_predictions_to_participants_and_evaluate_with_validation_threshold() -> None:
    from covid_rars.sota_predictions import (
        aggregate_sota_predictions,
        evaluate_sota_prediction_table,
    )

    segment = _segment_predictions()
    participant = aggregate_sota_predictions(segment, level="participant")
    metrics = evaluate_sota_prediction_table(participant)

    assert not participant.empty
    assert participant["participant_id"].nunique() == 24
    assert {"validation", "test"}.issubset(set(metrics["metric_split"]))
    assert metrics["threshold_source"].eq("validation_balanced_accuracy").all()
    assert metrics["auroc"].notna().all()


def test_fuse_sota_prediction_sources_reports_validation_and_test_metrics() -> None:
    from covid_rars.sota_predictions import (
        aggregate_sota_predictions,
        evaluate_sota_prediction_table,
        fuse_sota_prediction_sources,
    )

    participant = aggregate_sota_predictions(_segment_predictions(), level="participant")
    metrics = evaluate_sota_prediction_table(participant)
    fused_metrics, fused_predictions = fuse_sota_prediction_sources(
        metrics,
        participant,
        top_k=2,
        fusion_name="unit_test_stack",
    )

    assert not fused_metrics.empty
    assert not fused_predictions.empty
    assert set(fused_metrics["metric_split"]) == {"validation", "test"}
    assert fused_metrics["analysis_family"].eq("sota_prediction_fusion").all()
    assert fused_predictions["fusion_method"].isin(
        ["top_source_uniform_mean", "top_source_validation_weighted_auroc", "top_source_stacked_logistic_validation"]
    ).all()


def test_debug_sota_ssl_branch_keeps_external_test_predictions(tmp_path) -> None:
    import soundfile as sf

    from covid_rars.sota_ssl import train_sota_ssl_branch

    sample_rate = 16000
    rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test", "external_test"):
        for idx, label in enumerate(("negative", "positive")):
            y = np.sin(np.linspace(0, np.pi * (idx + 1), sample_rate, endpoint=False)).astype(np.float32) * 0.2
            path = tmp_path / f"{split}_{idx}.wav"
            sf.write(path, y, sample_rate)
            rows.append(
                {
                    "segment_id": f"{split}_{idx}_seg",
                    "recording_id": f"{split}_{idx}",
                    "participant_id": f"{split}_{idx}",
                    "dataset": "coughvid" if split == "external_test" else "coswara",
                    "modality": "cough",
                    "submodality": "cough",
                    "label_binary": label,
                    "split": split,
                    "audio_path": str(path),
                    "segment_start_sample": 0,
                    "segment_end_sample": sample_rate,
                    "is_augmented": False,
                    "augmentation_id": "original",
                    "augmentation_seed": 42,
                }
            )

    result = train_sota_ssl_branch(
        pd.DataFrame(rows),
        modality="cough",
        backend="debug_acoustic",
        target_samples=sample_rate,
    )

    assert "external_test" in set(result.predictions["split"])
    assert result.predictions[result.predictions["split"].eq("external_test")]["dataset"].eq("coughvid").all()
    assert "external_test" in set(result.metrics["metric_split"])
