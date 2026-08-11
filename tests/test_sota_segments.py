from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def _write_tone(path: Path, sample_rate: int = 16000, seconds: float = 2.2, amp: float = 0.2) -> None:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    y = np.zeros_like(t, dtype=np.float32)
    start = int(0.25 * sample_rate)
    end = int(1.75 * sample_rate)
    y[start:end] = amp * np.sin(2 * np.pi * 220 * t[start:end])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, sample_rate)


def _metadata(tmp_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    splits = ["train", "validation", "test"]
    labels = ["positive", "negative", "positive"]
    for idx, (split, label) in enumerate(zip(splits, labels)):
        for modality in ("cough", "breath"):
            audio_path = tmp_path / f"p{idx}_{modality}.wav"
            _write_tone(audio_path, amp=0.15 + idx * 0.03)
            rows.append(
                {
                    "recording_id": f"rec_{idx}_{modality}",
                    "participant_id": f"p{idx}",
                    "dataset": "coswara",
                    "modality": modality,
                    "submodality": modality,
                    "label_binary": label,
                    "split": split,
                    "quality_flag": "ok",
                    "audio_path": str(audio_path),
                }
            )
    return pd.DataFrame(rows)


def test_build_sota_segment_index_is_split_safe_and_train_augmented_only(tmp_path: Path) -> None:
    from covid_rars.sota_segments import (
        build_sota_segment_index,
        validate_segment_index_no_leakage,
    )

    index = build_sota_segment_index(
        _metadata(tmp_path),
        modalities=["cough", "breath"],
        quality_mode="quality_ok_only",
        window_sec=1.0,
        overlap=0.5,
        max_segments_per_recording=3,
        augment_train_copies=2,
        random_state=123,
    )

    assert not index.empty
    assert {"segment_id", "source_recording_id", "segment_start_sample", "segment_end_sample"}.issubset(index.columns)
    assert index["segment_id"].is_unique
    assert index["segment_end_sample"].gt(index["segment_start_sample"]).all()
    assert set(index["is_augmented"].dropna().unique()).issubset({False, True})
    assert index[index["is_augmented"].eq(True)]["split"].eq("train").all()
    assert not index[index["split"].isin(["validation", "test"])]["is_augmented"].any()

    audit = validate_segment_index_no_leakage(index)
    assert audit["severity"].eq("error").sum() == 0


def test_build_sota_segment_index_preserves_external_test_without_augmentation(tmp_path: Path) -> None:
    from covid_rars.sota_segments import build_sota_segment_index

    metadata = _metadata(tmp_path)
    external_audio = tmp_path / "external_cough.wav"
    _write_tone(external_audio, amp=0.22)
    metadata = pd.concat(
        [
            metadata,
            pd.DataFrame(
                [
                    {
                        "recording_id": "external_rec",
                        "participant_id": "external_p",
                        "dataset": "coughvid",
                        "modality": "cough",
                        "submodality": "cough",
                        "label_binary": "positive",
                        "split": "external_test",
                        "quality_flag": "ok",
                        "audio_path": str(external_audio),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    index = build_sota_segment_index(
        metadata,
        modalities=["cough"],
        window_sec=1.0,
        overlap=0.5,
        max_segments_per_recording=2,
        augment_train_copies=1,
    )

    external = index[index["split"].eq("external_test")]
    assert not external.empty
    assert external["dataset"].eq("coughvid").all()
    assert not external["is_augmented"].any()


def test_load_sota_segment_waveform_returns_fixed_length_normalized_audio(tmp_path: Path) -> None:
    from covid_rars.sota_segments import build_sota_segment_index, load_sota_segment_waveform

    index = build_sota_segment_index(
        _metadata(tmp_path),
        modalities=["cough"],
        window_sec=1.0,
        overlap=0.5,
        max_segments_per_recording=1,
    )
    row = index.iloc[0]

    y = load_sota_segment_waveform(row, target_samples=16000, rms_dbfs=-23.0)

    assert y.shape == (16000,)
    assert np.isfinite(y).all()
    assert float(np.max(np.abs(y))) <= 1.0
    assert float(np.sqrt(np.mean(np.square(y)))) > 0.0
