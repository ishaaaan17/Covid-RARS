from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest


def _claim_payload(recording_key: str = "coswara::r1") -> dict[str, object]:
    return {
        "recording_key": recording_key,
        "source_sha256": "a" * 64,
        "preprocessing_hash": "b" * 64,
    }


def _active_claim_record(*, identity: object, token: str = "old-token") -> dict[str, object]:
    return {
        "schema_version": 1,
        "claim_type": "hst_spectrogram_preprocessing",
        "status": "active",
        "token": token,
        "host": identity.host,
        "pid": identity.pid,
        "process_start_identity": identity.start_identity,
        "claimed_unix_time": 1.0,
        **_claim_payload(),
    }


def _tone(seconds: float = 3.0, frequency: float = 440.0, sr: int = 22050) -> np.ndarray:
    time = np.arange(int(sr * seconds), dtype=np.float64) / sr
    return (0.2 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def test_paper_logmel_is_deterministic_and_single_channel_cache() -> None:
    from covid_audio_btp.hst_spectrograms import HSTSpectrogramConfig, waveform_to_hst_image

    config = HSTSpectrogramConfig.paper_default()
    first = waveform_to_hst_image(_tone(), 22050, config)
    second = waveform_to_hst_image(_tone(), 22050, config)
    assert first.shape == (224, 224)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert float(first.std()) > 0.01
    assert first.dtype == np.float32


def test_post_trim_audio_must_be_strictly_longer_than_two_seconds() -> None:
    from covid_audio_btp.hst_spectrograms import HSTSpectrogramConfig, preprocess_recording

    config = HSTSpectrogramConfig.paper_default()
    for n_samples in (22050, 2 * 22050):
        result = preprocess_recording(np.ones(n_samples, dtype=np.float32), 22050, config)
        assert result.eligible is False
        assert result.reason == "post_trim_duration_not_above_2_seconds"


def test_waveform_shorter_than_trim_frame_is_excluded_without_padding() -> None:
    from covid_audio_btp.hst_spectrograms import HSTSpectrogramConfig, preprocess_recording

    config = HSTSpectrogramConfig.paper_default()
    waveform = np.ones(config.trim_frame_length - 1, dtype=np.float32)

    result = preprocess_recording(waveform, config.sample_rate, config)

    assert result.eligible is False
    assert result.reason == "post_trim_duration_not_above_2_seconds"
    assert result.original_duration_seconds == pytest.approx(
        waveform.size / config.sample_rate
    )
    assert result.trimmed_duration_seconds == pytest.approx(
        waveform.size / config.sample_rate
    )


def test_frequency_orientation_and_configuration_hash_are_frozen() -> None:
    from covid_audio_btp.hst_spectrograms import HSTSpectrogramConfig, preprocessing_hash, waveform_to_hst_image

    paper = HSTSpectrogramConfig.paper_default()
    image = waveform_to_hst_image(_tone(frequency=440.0), 22050, paper)
    strongest_row = int(np.argmax(image.mean(axis=1)))
    assert strongest_row > image.shape[0] // 2
    released = replace(paper, representation_id="released_linear_specgram_224")
    assert preprocessing_hash(paper) != preprocessing_hash(released)


def test_released_reference_spectrogram_is_a_distinct_rendered_branch() -> None:
    from covid_audio_btp.hst_spectrograms import (
        HSTSpectrogramConfig,
        waveform_to_hst_image,
    )

    waveform = _tone(frequency=880.0)
    paper = waveform_to_hst_image(
        waveform, 22050, HSTSpectrogramConfig.paper_default()
    )
    released_config = HSTSpectrogramConfig.released_reference()
    first = waveform_to_hst_image(waveform, 22050, released_config)
    second = waveform_to_hst_image(waveform, 22050, released_config)

    assert released_config.representation_id == "released_linear_specgram_224"
    assert np.array_equal(first, second)
    assert first.shape == (224, 224)
    assert float(np.mean(np.abs(first - paper))) > 0.05
    assert float(first[0, 0]) > 0.9


def test_cache_records_source_and_tensor_hashes(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    from covid_audio_btp.hst_spectrograms import HSTSpectrogramConfig, build_hst_spectrogram_cache

    audio = tmp_path / "tone.wav"
    soundfile.write(audio, _tone(), 22050)
    metadata = pd.DataFrame(
        {
            "dataset": ["coswara"],
            "participant_id": ["p1"],
            "participant_key": ["coswara::p1"],
            "recording_id": ["r1"],
            "recording_key": ["coswara::r1"],
            "modality": ["cough"],
            "label_binary": ["positive"],
            "audio_path": [audio.as_posix()],
            "recording_timestamp": ["2020-04-01T12:00:00Z"],
            "cough_symptom": ["yes"],
            "label_source": ["project_label"],
            "dataset_release_id": ["coswara-test"],
        }
    )
    output = build_hst_spectrogram_cache(
        metadata,
        output_dir=tmp_path / "cache",
        config=HSTSpectrogramConfig.paper_default(),
    )
    assert output.loc[0, "eligible"]
    assert len(output.loc[0, "source_sha256"]) == 64
    assert len(output.loc[0, "tensor_sha256"]) == 64
    assert output.loc[0, "representation_id"] == "paper_logmel_224"
    assert output.loc[0, "recording_timestamp"] == "2020-04-01T12:00:00Z"
    assert output.loc[0, "cough_symptom"] == "yes"
    assert output.loc[0, "label_source"] == "project_label"
    assert output.loc[0, "dataset_release_id"] == "coswara-test"
    array = np.load(output.loc[0, "cache_path"], allow_pickle=False)
    assert array.shape == (224, 224)
    assert array.dtype == np.float32
    cached_again = build_hst_spectrogram_cache(
        metadata,
        output_dir=tmp_path / "cache",
        config=HSTSpectrogramConfig.paper_default(),
    )
    assert cached_again.loc[0, "cache_status"] == "verified_hit"
    assert cached_again.loc[0, "representation_id"] == "paper_logmel_224"
    assert (
        cached_again.loc[0, "preprocessing_implementation_version"]
        == "hst-spectrogram-preprocessing-v2"
    )


def test_cache_rejects_audio_bytes_that_do_not_match_frozen_source_hash(
    tmp_path: Path,
) -> None:
    soundfile = pytest.importorskip("soundfile")
    from covid_audio_btp.hst_spectrograms import (
        HSTSpectrogramConfig,
        build_hst_spectrogram_cache,
    )

    audio = tmp_path / "tone.wav"
    soundfile.write(audio, _tone(), 22050)
    metadata = pd.DataFrame(
        {
            "dataset": ["coswara"],
            "participant_id": ["p1"],
            "participant_key": ["coswara::p1"],
            "recording_id": ["r1"],
            "recording_key": ["coswara::r1"],
            "modality": ["cough"],
            "label_binary": ["positive"],
            "audio_path": [audio.as_posix()],
            "expected_source_sha256": ["0" * 64],
            "expected_source_size_bytes": [audio.stat().st_size],
        }
    )

    with pytest.raises(ValueError, match="(?i)frozen source audio checksum changed"):
        build_hst_spectrogram_cache(
            metadata,
            output_dir=tmp_path / "cache",
            config=HSTSpectrogramConfig.paper_default(),
        )


def test_cache_hit_uses_current_contract_metadata_not_stale_fragment(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    from covid_audio_btp.hst_spectrograms import (
        HSTSpectrogramConfig,
        build_hst_spectrogram_cache,
    )

    audio = tmp_path / "tone.wav"
    soundfile.write(audio, _tone(), 22050)
    original = pd.DataFrame(
        {
            "dataset": ["coswara"],
            "participant_id": ["old-participant"],
            "participant_key": ["coswara::old-participant"],
            "recording_id": ["r1"],
            "recording_key": ["coswara::r1"],
            "modality": ["cough"],
            "label_binary": ["positive"],
            "audio_path": [audio.as_posix()],
            "label_source": ["old-contract"],
        }
    )
    current = original.assign(
        participant_id="current-participant",
        participant_key="coswara::current-participant",
        label_binary="negative",
        label_source="current-contract",
    )
    config = HSTSpectrogramConfig.paper_default()
    first = build_hst_spectrogram_cache(
        original,
        output_dir=tmp_path / "cache",
        config=config,
    )
    second = build_hst_spectrogram_cache(
        current,
        output_dir=tmp_path / "cache",
        config=config,
    )

    assert second.loc[0, "cache_status"] == "verified_hit"
    assert second.loc[0, "tensor_sha256"] == first.loc[0, "tensor_sha256"]
    assert second.loc[0, "participant_id"] == "current-participant"
    assert second.loc[0, "participant_key"] == "coswara::current-participant"
    assert second.loc[0, "label_binary"] == "negative"
    assert second.loc[0, "label_source"] == "current-contract"


def test_audio_source_snapshot_keeps_hashed_bytes_stable_after_path_replacement(
    tmp_path: Path,
) -> None:
    soundfile = pytest.importorskip("soundfile")
    from covid_audio_btp.hst_spectrograms import (
        HSTSpectrogramConfig,
        audio_source_snapshot,
        preprocess_audio_path,
    )

    source = tmp_path / "source.wav"
    soundfile.write(source, _tone(frequency=440.0), 22050)
    with audio_source_snapshot(source) as snapshot:
        expected_hash = snapshot.sha256
        expected_size = snapshot.size_bytes
        soundfile.write(source, _tone(frequency=880.0), 22050)
        result = preprocess_audio_path(snapshot.path, HSTSpectrogramConfig.paper_default())
        snapshot_bytes = snapshot.path.read_bytes()

    assert result.eligible
    assert len(snapshot_bytes) == expected_size
    import hashlib

    assert hashlib.sha256(snapshot_bytes).hexdigest() == expected_hash


def test_cache_index_verification_rejects_missing_or_corrupt_tensor(tmp_path: Path) -> None:
    from covid_audio_btp.hst_spectrograms import (
        _tensor_sha256,
        validate_hst_cache_index,
    )

    cache_root = tmp_path / "cache"
    tensor_path = cache_root / "config" / "tensors" / "one.npy"
    tensor_path.parent.mkdir(parents=True)
    image = np.ones((224, 224), dtype=np.float32)
    np.save(tensor_path, image, allow_pickle=False)
    index = pd.DataFrame(
        {
            "recording_key": ["coswara::r1"],
            "eligible": [True],
            "cache_path": [tensor_path.as_posix()],
            "tensor_sha256": [_tensor_sha256(image)],
        }
    )
    index_path = tmp_path / "cache_index.csv"
    index.to_csv(index_path, index=False)

    assert validate_hst_cache_index(index_path, cache_root=cache_root) == 1
    tensor_path.write_bytes(b"corrupt")
    with pytest.raises((ValueError, OSError), match="cache|tensor|load|pickle"):
        validate_hst_cache_index(index_path, cache_root=cache_root)


def test_three_channel_conversion_is_identical_and_normalized() -> None:
    torch = pytest.importorskip("torch")
    from covid_audio_btp.hst_spectrograms import HSTSpectrogramConfig, image_to_model_tensor

    image = np.full((224, 224), 0.75, dtype=np.float32)
    tensor = image_to_model_tensor(image, HSTSpectrogramConfig.paper_default())
    assert tuple(tensor.shape) == (3, 224, 224)
    assert torch.equal(tensor[0], tensor[1])
    assert torch.equal(tensor[1], tensor[2])
    assert float(tensor.mean()) == pytest.approx(0.5)


def test_three_channel_conversion_copies_read_only_cache_arrays() -> None:
    pytest.importorskip("torch")
    from covid_audio_btp.hst_spectrograms import HSTSpectrogramConfig, image_to_model_tensor

    image = np.full((224, 224), 0.75, dtype=np.float32)
    image.setflags(write=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tensor = image_to_model_tensor(image, HSTSpectrogramConfig.paper_default())

    assert tensor.is_contiguous()


def test_preprocessing_claim_binds_native_process_identity_and_records_completion(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_runtime import capture_process_identity
    from covid_audio_btp.hst_spectrograms import _acquire_preprocessing_claim

    identity = capture_process_identity()
    claim_path = tmp_path / "item.claim"
    claim = _acquire_preprocessing_claim(
        claim_path,
        _claim_payload(),
        identity=identity,
    )
    active = json.loads(claim_path.read_text(encoding="utf-8"))

    assert active["status"] == "active"
    assert active["host"] == identity.host
    assert active["pid"] == os.getpid()
    assert active["process_start_identity"] == identity.start_identity
    assert active["token"] == claim.token

    claim.release(status="completed")
    completed = json.loads(claim_path.read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert completed["token"] == claim.token
    assert completed["completed_unix_time"] >= active["claimed_unix_time"]


def test_preprocessing_claim_recovers_dead_owner_and_archives_failure_evidence(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_runtime import (
        ProcessIdentity,
        ProcessLiveness,
        capture_process_identity,
    )
    from covid_audio_btp.hst_spectrograms import _acquire_preprocessing_claim

    claim_path = tmp_path / "item.claim"
    dead_owner = ProcessIdentity(
        host=capture_process_identity().host,
        pid=999_999,
        start_identity="linux:test-boot:17",
    )
    prior = _active_claim_record(identity=dead_owner)
    claim_path.write_text(json.dumps(prior), encoding="utf-8")

    claim = _acquire_preprocessing_claim(
        claim_path,
        _claim_payload(),
        process_probe=lambda owner: ProcessLiveness.DEAD,
    )
    current = json.loads(claim_path.read_text(encoding="utf-8"))
    recovery_files = list(tmp_path.glob("item.claim.recovered.*.json"))

    assert current["status"] == "active"
    assert current["recovered_claim_token"] == "old-token"
    assert len(recovery_files) == 1
    assert json.loads(recovery_files[0].read_text(encoding="utf-8")) == prior
    claim.release(status="completed")


def test_preprocessing_claim_recovers_reused_pid_identity_mismatch(tmp_path: Path) -> None:
    from covid_audio_btp.hst_runtime import (
        ProcessIdentity,
        ProcessLiveness,
        capture_process_identity,
        process_identity_liveness,
    )
    from covid_audio_btp.hst_spectrograms import _acquire_preprocessing_claim

    current = capture_process_identity()
    prefix, value = current.start_identity.rsplit(":", 1)
    replaced_owner = ProcessIdentity(
        host=current.host,
        pid=current.pid,
        start_identity=f"{prefix}:{int(value) + 1}",
    )
    assert process_identity_liveness(replaced_owner) is ProcessLiveness.DEAD

    claim_path = tmp_path / "item.claim"
    claim_path.write_text(
        json.dumps(_active_claim_record(identity=replaced_owner)),
        encoding="utf-8",
    )
    claim = _acquire_preprocessing_claim(claim_path, _claim_payload())
    assert claim.recovered_claim_token == "old-token"
    claim.release(status="completed")


@pytest.mark.parametrize("owner_state", ["alive", "unknown"])
def test_preprocessing_claim_never_steals_non_dead_owner(
    tmp_path: Path,
    owner_state: str,
) -> None:
    from covid_audio_btp.hst_runtime import (
        ProcessIdentity,
        ProcessLiveness,
        capture_process_identity,
    )
    from covid_audio_btp.hst_spectrograms import _acquire_preprocessing_claim

    owner = ProcessIdentity(
        host=capture_process_identity().host,
        pid=1234,
        start_identity="linux:test-boot:17",
    )
    claim_path = tmp_path / "item.claim"
    prior = _active_claim_record(identity=owner)
    claim_path.write_text(json.dumps(prior), encoding="utf-8")

    with pytest.raises(BlockingIOError, match=f"owner is {owner_state}"):
        _acquire_preprocessing_claim(
            claim_path,
            _claim_payload(),
            process_probe=lambda candidate: ProcessLiveness(owner_state),
        )

    assert json.loads(claim_path.read_text(encoding="utf-8")) == prior


def test_preprocessing_claim_refuses_unverifiable_legacy_active_record(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_runtime import (
        ProcessLiveness,
        RuntimeStateError,
        capture_process_identity,
    )
    from covid_audio_btp.hst_spectrograms import _acquire_preprocessing_claim

    identity = capture_process_identity()
    claim_path = tmp_path / "item.claim"
    legacy = _active_claim_record(identity=identity)
    legacy.pop("claim_type")
    claim_path.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(RuntimeStateError, match="schema"):
        _acquire_preprocessing_claim(
            claim_path,
            _claim_payload(),
            process_probe=lambda owner: ProcessLiveness.DEAD,
        )

    assert json.loads(claim_path.read_text(encoding="utf-8")) == legacy


def test_preprocessing_claim_kernel_guard_prevents_same_process_double_acquire(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_spectrograms import _acquire_preprocessing_claim

    claim_path = tmp_path / "item.claim"
    first = _acquire_preprocessing_claim(claim_path, _claim_payload())
    try:
        with pytest.raises(BlockingIOError, match="already held"):
            _acquire_preprocessing_claim(claim_path, _claim_payload())
        active = json.loads(claim_path.read_text(encoding="utf-8"))
        assert active["token"] == first.token
        assert active["status"] == "active"
    finally:
        first.release(status="completed")


def test_preprocessing_claim_records_failure_without_deleting_evidence(tmp_path: Path) -> None:
    from covid_audio_btp.hst_spectrograms import _acquire_preprocessing_claim

    claim_path = tmp_path / "item.claim"
    claim = _acquire_preprocessing_claim(claim_path, _claim_payload())
    claim.release(status="failed", failure=ValueError("decode failed"))

    record = json.loads(claim_path.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["failure_type"] == "ValueError"
    assert record["failure_message"] == "decode failed"


def test_cache_records_downstream_blocking_error_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from covid_audio_btp import hst_spectrograms
    from covid_audio_btp.hst_spectrograms import (
        HSTSpectrogramConfig,
        build_hst_spectrogram_cache,
    )

    audio = tmp_path / "unread.wav"
    audio.write_bytes(b"source-fingerprint-only")
    metadata = pd.DataFrame(
        {
            "dataset": ["coswara"],
            "participant_id": ["p1"],
            "participant_key": ["coswara::p1"],
            "recording_id": ["r1"],
            "recording_key": ["coswara::r1"],
            "modality": ["cough"],
            "label_binary": ["positive"],
            "audio_path": [audio.as_posix()],
        }
    )

    def fail_decode(path: Path, config: HSTSpectrogramConfig) -> object:
        raise BlockingIOError("decoder busy")

    monkeypatch.setattr(hst_spectrograms, "preprocess_audio_path", fail_decode)
    output_dir = tmp_path / "cache"
    with pytest.raises(BlockingIOError, match="decoder busy"):
        build_hst_spectrogram_cache(
            metadata,
            output_dir=output_dir,
            config=HSTSpectrogramConfig.paper_default(),
        )

    claims = list(output_dir.rglob("*.claim"))
    assert len(claims) == 1
    evidence = json.loads(claims[0].read_text(encoding="utf-8"))
    assert evidence["status"] == "failed"
    assert evidence["failure_type"] == "BlockingIOError"
    assert evidence["failure_message"] == "decoder busy"
