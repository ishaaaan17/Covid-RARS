from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import pandas as pd


PREPROCESSING_IMPLEMENTATION_VERSION = "hst-spectrogram-preprocessing-v3"

from covid_audio_btp.audio_io import local_audio_path, split_archive_member_path
from covid_audio_btp.hst_runtime import (
    ProcessIdentity,
    ProcessLiveness,
    RuntimeStateError,
    capture_process_identity,
    process_identity_liveness,
)


_CACHE_METADATA_PASSTHROUGH = (
    "submodality",
    "quality_flag",
    "recording_timestamp_utc",
    "recording_timestamp",
    "recording_date",
    "date",
    "cough_symptom",
    "cough_symptoms",
    "symptoms",
    "label_source",
    "label_provenance",
    "dataset_release_id",
    "source_manifest_sha256",
    "preprocessing_variant",
    "analysis_unit_type",
    "identity_source_column",
    "participant_id_is_recording_proxy",
    "subject_linkage_available",
    "metadata_source_level",
    "status",
    "status_SSL",
    "expected_source_sha256",
    "expected_source_size_bytes",
)

_LOCAL_PREPROCESSING_CLAIMS: set[str] = set()
_LOCAL_PREPROCESSING_CLAIMS_LOCK = threading.Lock()


@dataclass(frozen=True)
class HSTSpectrogramConfig:
    representation_id: str
    sample_rate: int
    resample_type: str
    dtype: str
    trim_top_db: float
    trim_frame_length: int
    trim_hop_length: int
    minimum_duration_seconds: float
    n_fft: int
    win_length: int
    window: str
    hann_periodic: bool
    noverlap: int
    hop_length: int
    n_mels: int
    mel_htk: bool
    mel_norm: str
    fmin: float
    fmax: float
    power: float
    center: bool
    db_ref: str
    top_db: float
    image_size: int
    resize_interpolation: str
    resize_antialias: bool
    array_row_zero_frequency: str
    display_origin: str
    augment_before_normalize: bool
    normalization_mean: float
    normalization_std: float
    @classmethod
    def paper_default(cls) -> "HSTSpectrogramConfig":
        return cls(
            representation_id="paper_logmel_224",
            sample_rate=22050,
            resample_type="soxr_hq",
            dtype="float32",
            trim_top_db=60.0,
            trim_frame_length=2205,
            trim_hop_length=1102,
            minimum_duration_seconds=2.0,
            n_fft=2048,
            win_length=2048,
            window="hann",
            hann_periodic=True,
            noverlap=128,
            hop_length=1920,
            n_mels=224,
            mel_htk=False,
            mel_norm="slaney",
            fmin=0.0,
            fmax=11025.0,
            power=2.0,
            center=False,
            db_ref="max",
            top_db=80.0,
            image_size=224,
            resize_interpolation="bilinear",
            resize_antialias=True,
            array_row_zero_frequency="high",
            display_origin="upper",
            augment_before_normalize=True,
            normalization_mean=0.5,
            normalization_std=0.5,
        )

    @classmethod
    def released_reference(cls) -> "HSTSpectrogramConfig":
        values = asdict(cls.paper_default())
        values["representation_id"] = "released_linear_specgram_224"
        return cls(**values)


@dataclass(frozen=True)
class AudioSourceSnapshot:
    path: Path
    size_bytes: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class PreprocessResult:
    eligible: bool
    reason: str
    image: np.ndarray | None
    original_duration_seconds: float
    trimmed_duration_seconds: float


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def preprocessing_hash(config: HSTSpectrogramConfig) -> str:
    payload = {
        "implementation_version": PREPROCESSING_IMPLEMENTATION_VERSION,
        "config": asdict(config),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_config(config: HSTSpectrogramConfig) -> None:
    if config.dtype != "float32":
        raise ValueError("HST cache dtype must be float32")
    if config.hop_length != config.n_fft - config.noverlap:
        raise ValueError("hop_length must equal n_fft - noverlap")
    if config.window != "hann" or not config.hann_periodic:
        raise ValueError("The primary HST representation requires a periodic Hann window")
    if config.array_row_zero_frequency != "high" or config.display_origin != "upper":
        raise ValueError("The primary HST frequency orientation is high-frequency-first")
    if config.image_size <= 0 or config.n_mels <= 0:
        raise ValueError("Image and Mel dimensions must be positive")


def _prepare_waveform(y: np.ndarray, sr: int, config: HSTSpectrogramConfig) -> tuple[np.ndarray, float, float]:
    import librosa

    waveform = np.asarray(y, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError("Audio waveform must be one-dimensional after mono conversion")
    if sr <= 0:
        raise ValueError("Sample rate must be positive")
    original_duration = float(waveform.size / sr)
    if not np.isfinite(waveform).all():
        raise ValueError("Audio waveform contains non-finite values")
    if sr != config.sample_rate:
        waveform = librosa.resample(
            waveform,
            orig_sr=sr,
            target_sr=config.sample_rate,
            res_type=config.resample_type,
            fix=True,
            scale=False,
        ).astype(np.float32, copy=False)
    if waveform.size < config.trim_frame_length:
        # Such recordings are necessarily below the frozen two-second threshold.
        # Preserve their actual duration so the caller can exclude them without
        # padding or invoking librosa with an invalid frame size.
        trimmed = waveform
    else:
        trimmed, _ = librosa.effects.trim(
            waveform,
            top_db=config.trim_top_db,
            frame_length=config.trim_frame_length,
            hop_length=config.trim_hop_length,
        )
    trimmed = np.asarray(trimmed, dtype=np.float32)
    return trimmed, original_duration, float(trimmed.size / config.sample_rate)


def _released_reference_to_image(
    y: np.ndarray, config: HSTSpectrogramConfig
) -> np.ndarray:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image

    figure = Figure(figsize=(8.0, 8.0), dpi=100.0, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.specgram(
        y,
        NFFT=config.n_fft,
        Fs=config.sample_rate,
        Fc=0,
        noverlap=config.noverlap,
        cmap="gray",
        sides="default",
        mode="default",
        scale="dB",
    )
    axis.axis("off")
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    grayscale = rgba[:, :, 0].astype(np.float32) / 255.0
    image = Image.fromarray(grayscale, mode="F").resize(
        (config.image_size, config.image_size),
        resample=Image.Resampling.BILINEAR,
    )
    result = np.asarray(image, dtype=np.float32)
    if result.shape != (config.image_size, config.image_size):
        raise ValueError(f"Unexpected released HST image shape: {result.shape}")
    if not np.isfinite(result).all() or float(np.ptp(result)) <= 1e-8:
        raise ValueError("Released HST image is non-finite or constant")
    return result


def _prepared_waveform_to_image(y: np.ndarray, config: HSTSpectrogramConfig) -> np.ndarray:
    import librosa
    from PIL import Image

    if config.representation_id == "released_linear_specgram_224":
        return _released_reference_to_image(y, config)
    if config.representation_id != "paper_logmel_224":
        raise ValueError(f"Unsupported HST representation: {config.representation_id}")

    stft = librosa.stft(
        y,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=config.window,
        center=config.center,
    )
    power = np.abs(stft).astype(np.float32) ** config.power
    mel = librosa.feature.melspectrogram(
        S=power,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        htk=config.mel_htk,
        norm=config.mel_norm,
        power=config.power,
    )
    if config.db_ref != "max":
        raise ValueError(f"Unsupported dB reference: {config.db_ref}")
    decibels = librosa.power_to_db(mel, ref=np.max, top_db=config.top_db)
    image = np.clip((decibels + config.top_db) / config.top_db, 0.0, 1.0)
    if config.array_row_zero_frequency == "high":
        image = np.flipud(image)
    if config.resize_interpolation != "bilinear":
        raise ValueError(f"Unsupported resize interpolation: {config.resize_interpolation}")
    pil = Image.fromarray(np.ascontiguousarray(image, dtype=np.float32), mode="F")
    pil = pil.resize((config.image_size, config.image_size), resample=Image.Resampling.BILINEAR)
    result = np.asarray(pil, dtype=np.float32)
    if result.shape != (config.image_size, config.image_size):
        raise ValueError(f"Unexpected HST image shape: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("HST image contains non-finite values")
    if float(np.ptp(result)) <= 1e-8:
        raise ValueError("HST image is constant")
    return result


def waveform_to_hst_image(y: np.ndarray, sr: int, config: HSTSpectrogramConfig) -> np.ndarray:
    _validate_config(config)
    prepared, _, duration = _prepare_waveform(y, sr, config)
    if duration <= config.minimum_duration_seconds:
        raise ValueError("post_trim_duration_not_above_2_seconds")
    return _prepared_waveform_to_image(prepared, config)


def preprocess_recording(y: np.ndarray, sr: int, config: HSTSpectrogramConfig) -> PreprocessResult:
    _validate_config(config)
    try:
        prepared, original_duration, trimmed_duration = _prepare_waveform(y, sr, config)
    except (TypeError, ValueError) as exc:
        duration = float(np.asarray(y).size / sr) if sr > 0 else 0.0
        return PreprocessResult(False, f"invalid_waveform:{exc}", None, duration, 0.0)
    if trimmed_duration <= config.minimum_duration_seconds:
        return PreprocessResult(
            False,
            "post_trim_duration_not_above_2_seconds",
            None,
            original_duration,
            trimmed_duration,
        )
    try:
        image = _prepared_waveform_to_image(prepared, config)
    except (TypeError, ValueError) as exc:
        return PreprocessResult(False, f"invalid_spectrogram:{exc}", None, original_duration, trimmed_duration)
    return PreprocessResult(True, "", image, original_duration, trimmed_duration)


def preprocess_audio_path(path: Path, config: HSTSpectrogramConfig) -> PreprocessResult:
    import soundfile as sf

    try:
        with local_audio_path(path) as resolved:
            y, sr = sf.read(resolved, always_2d=False, dtype="float32")
            if np.asarray(y).ndim > 1:
                y = np.asarray(y, dtype=np.float32).mean(axis=1)
    except Exception as soundfile_error:
        try:
            import librosa

            with local_audio_path(path) as resolved:
                y, sr = librosa.load(resolved, sr=None, mono=True, dtype=np.float32)
        except Exception as librosa_error:
            return PreprocessResult(
                False,
                f"decode_error:soundfile={type(soundfile_error).__name__};librosa={type(librosa_error).__name__}",
                None,
                0.0,
                0.0,
            )
    return preprocess_recording(np.asarray(y, dtype=np.float32), int(sr), config)


@contextlib.contextmanager
def audio_source_snapshot(path: str | Path) -> Iterator[AudioSourceSnapshot]:
    """Yield an immutable local copy of exactly the bytes being fingerprinted."""

    archive_member = split_archive_member_path(path)
    if archive_member is not None:
        source_path, member = archive_member
        with source_path.open("rb") as source_handle:
            before = os.fstat(source_handle.fileno())
            with zipfile.ZipFile(source_handle) as archive:
                data = archive.read(member)
            after = os.fstat(source_handle.fileno())
        suffix = Path(member).suffix
        require_payload_size_match = False
    else:
        source_path = Path(path)
        with source_path.open("rb") as source_handle:
            before = os.fstat(source_handle.fileno())
            data = source_handle.read()
            after = os.fstat(source_handle.fileno())
        suffix = source_path.suffix
        require_payload_size_match = True
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or (require_payload_size_match and len(data) != after.st_size)
    ):
        raise RuntimeError("Audio source changed while creating an immutable snapshot")

    with tempfile.TemporaryDirectory(prefix="hst-audio-snapshot-") as temporary_dir:
        snapshot_path = Path(temporary_dir) / f"source{suffix}"
        with snapshot_path.open("xb") as snapshot_handle:
            snapshot_handle.write(data)
            snapshot_handle.flush()
            os.fsync(snapshot_handle.fileno())
        yield AudioSourceSnapshot(
            path=snapshot_path,
            size_bytes=len(data),
            mtime_ns=int(after.st_mtime_ns),
            sha256=_sha256_bytes(data),
        )


def _source_fingerprint(path: str | Path) -> tuple[int, int, str]:
    with audio_source_snapshot(path) as snapshot:
        return snapshot.size_bytes, snapshot.mtime_ns, snapshot.sha256


def _tensor_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return _sha256_bytes(contiguous.tobytes(order="C"))


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="ascii", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_array(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    expected_payload_hash = _tensor_sha256(array)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(array, dtype=np.float32), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        reloaded = np.load(temporary, allow_pickle=False)
        if reloaded.shape != array.shape or reloaded.dtype != np.float32 or not np.isfinite(reloaded).all():
            raise ValueError("Temporary HST cache tensor failed validation")
        if _tensor_sha256(reloaded) != expected_payload_hash:
            raise ValueError("Temporary HST cache tensor checksum mismatch")
        artifact_hash = _sha256_file(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return artifact_hash


class _PreprocessingClaim:
    def __init__(
        self,
        *,
        path: Path,
        guard_path: Path,
        descriptor: int,
        local_key: str,
        token: str,
        record: dict[str, object],
        recovered_claim_token: str | None,
    ) -> None:
        self.path = path
        self.guard_path = guard_path
        self._descriptor = descriptor
        self._local_key = local_key
        self.token = token
        self.record = record
        self.recovered_claim_token = recovered_claim_token
        self._released = False

    def release(
        self,
        *,
        status: str,
        failure: BaseException | None = None,
    ) -> None:
        if status not in {"completed", "failed"}:
            raise ValueError("Preprocessing claim release status must be completed or failed")
        if self._released:
            return
        try:
            current = _read_claim_record(self.path)
            if current.get("token") != self.token or current.get("status") != "active":
                raise RuntimeStateError(
                    f"Preprocessing claim ownership changed unexpectedly: {self.path}"
                )
            terminal = {
                **self.record,
                "status": status,
                "completed_unix_time": time.time(),
            }
            if failure is not None:
                terminal["failure_type"] = type(failure).__name__
                terminal["failure_message"] = str(failure)
            _atomic_json(self.path, terminal)
        finally:
            _unlock_claim_guard(self._descriptor)
            os.close(self._descriptor)
            with _LOCAL_PREPROCESSING_CLAIMS_LOCK:
                _LOCAL_PREPROCESSING_CLAIMS.discard(self._local_key)
            self._released = True


def _lock_claim_guard(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
    except (OSError, BlockingIOError) as exc:
        os.close(descriptor)
        raise BlockingIOError(f"Preprocessing claim guard is already held: {path}") from exc
    return descriptor


def _unlock_claim_guard(descriptor: int) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)
    else:
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        with contextlib.suppress(OSError):
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)


def _read_claim_record(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeStateError(f"Cannot safely inspect preprocessing claim: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeStateError(f"Preprocessing claim is not a JSON object: {path}")
    return value


def _claim_owner_state(
    record: Mapping[str, object],
    process_probe: Callable[[ProcessIdentity], ProcessLiveness],
) -> ProcessLiveness:
    if (
        record.get("schema_version") != 1
        or record.get("claim_type") != "hst_spectrogram_preprocessing"
        or not isinstance(record.get("token"), str)
        or not record.get("token")
    ):
        raise RuntimeStateError("Active preprocessing claim has an unverifiable schema")
    owner = ProcessIdentity.from_record(record)
    try:
        state = ProcessLiveness(process_probe(owner))
    except ValueError as exc:
        raise RuntimeStateError("Preprocessing claim process probe returned an invalid state") from exc
    return state


def _acquire_preprocessing_claim(
    path: Path,
    payload: Mapping[str, object],
    *,
    identity: ProcessIdentity | None = None,
    process_probe: Callable[[ProcessIdentity], ProcessLiveness] = process_identity_liveness,
) -> _PreprocessingClaim:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = identity or capture_process_identity()
    guard_path = path.with_suffix(path.suffix + ".lock")
    local_key = str(guard_path.resolve())
    with _LOCAL_PREPROCESSING_CLAIMS_LOCK:
        if local_key in _LOCAL_PREPROCESSING_CLAIMS:
            raise BlockingIOError(f"Preprocessing claim guard is already held: {guard_path}")
        _LOCAL_PREPROCESSING_CLAIMS.add(local_key)

    descriptor: int | None = None
    try:
        descriptor = _lock_claim_guard(guard_path)
        previous = _read_claim_record(path) if path.is_file() else None
        recovered_token: str | None = None
        if previous is not None and previous.get("status") == "active":
            state = _claim_owner_state(previous, process_probe)
            if state is not ProcessLiveness.DEAD:
                reason = "alive" if state is ProcessLiveness.ALIVE else "unknown"
                raise BlockingIOError(
                    f"Preprocessing claim is not recoverable because owner is {reason}: {path}"
                )
            recovered_token = str(previous.get("token", ""))
            if not recovered_token:
                raise RuntimeStateError(f"Active preprocessing claim has no token: {path}")
            evidence_path = path.with_name(
                f"{path.name}.recovered.{recovered_token}.{uuid.uuid4().hex}.json"
            )
            _atomic_json(evidence_path, previous)
        elif previous is not None and previous.get("status") not in {"completed", "failed"}:
            raise RuntimeStateError(f"Preprocessing claim has an invalid status: {path}")

        token = uuid.uuid4().hex
        record: dict[str, object] = {
            "schema_version": 1,
            "claim_type": "hst_spectrogram_preprocessing",
            "status": "active",
            "token": token,
            "host": identity.host,
            "pid": identity.pid,
            "process_start_identity": identity.start_identity,
            "claimed_unix_time": time.time(),
            **dict(payload),
        }
        if recovered_token is not None:
            record["recovered_claim_token"] = recovered_token
        _atomic_json(path, record)
        return _PreprocessingClaim(
            path=path,
            guard_path=guard_path,
            descriptor=descriptor,
            local_key=local_key,
            token=token,
            record=record,
            recovered_claim_token=recovered_token,
        )
    except Exception:
        if descriptor is not None:
            _unlock_claim_guard(descriptor)
            os.close(descriptor)
        with _LOCAL_PREPROCESSING_CLAIMS_LOCK:
            _LOCAL_PREPROCESSING_CLAIMS.discard(local_key)
        raise


def _required_cache_columns(metadata: pd.DataFrame) -> None:
    required = {
        "dataset",
        "participant_id",
        "participant_key",
        "recording_id",
        "recording_key",
        "modality",
        "label_binary",
        "audio_path",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"HST cache metadata missing columns: {missing}")
    if metadata["recording_key"].duplicated().any():
        raise ValueError("HST cache metadata contains duplicate recording_key values")


def _json_metadata_value(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _current_contract_metadata(
    row: Mapping[str, object], *, recording_key: str
) -> dict[str, object]:
    values = {
        column: _json_metadata_value(row[column])
        for column in _CACHE_METADATA_PASSTHROUGH
        if column in row
    }
    values.update(
        {
            column: _json_metadata_value(row[column])
            for column in (
                "dataset",
                "participant_id",
                "participant_key",
                "recording_id",
                "modality",
                "label_binary",
                "audio_path",
            )
        }
    )
    values["recording_key"] = recording_key
    return values


def build_hst_spectrogram_cache(
    metadata: pd.DataFrame,
    *,
    output_dir: Path,
    config: HSTSpectrogramConfig,
    force: bool = False,
) -> pd.DataFrame:
    _required_cache_columns(metadata)
    config_hash = preprocessing_hash(config)
    cache_root = Path(output_dir) / config_hash
    tensors_dir = cache_root / "tensors"
    fragments_dir = cache_root / "fragments"
    claims_dir = cache_root / "claims"
    for directory in (tensors_dir, fragments_dir, claims_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for row in metadata.sort_values("recording_key").to_dict(orient="records"):
        recording_key = str(row["recording_key"])
        cache_id = hashlib.sha256(f"{recording_key}\0{config_hash}".encode("utf-8")).hexdigest()
        tensor_path = tensors_dir / f"{cache_id}.npy"
        fragment_path = fragments_dir / f"{cache_id}.json"
        claim_path = claims_dir / f"{cache_id}.claim"
        with audio_source_snapshot(str(row["audio_path"])) as snapshot:
            source_size = snapshot.size_bytes
            source_mtime_ns = snapshot.mtime_ns
            source_hash = snapshot.sha256
            if "expected_source_sha256" in row:
                expected_hash = str(row["expected_source_sha256"])
                if len(expected_hash) != 64 or source_hash != expected_hash:
                    raise ValueError(
                        f"Frozen source audio checksum changed: {recording_key}"
                    )
            if "expected_source_size_bytes" in row:
                try:
                    expected_size = int(row["expected_source_size_bytes"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Frozen source audio size is invalid: {recording_key}"
                    ) from exc
                if source_size != expected_size:
                    raise ValueError(f"Frozen source audio size changed: {recording_key}")
            if not force and tensor_path.is_file() and fragment_path.is_file():
                fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
                cached = np.load(tensor_path, allow_pickle=False)
                valid = (
                    fragment.get("source_sha256") == source_hash
                    and fragment.get("preprocessing_hash") == config_hash
                    and fragment.get("preprocessing_implementation_version")
                    == PREPROCESSING_IMPLEMENTATION_VERSION
                    and fragment.get("representation_id") == config.representation_id
                    and cached.shape == (config.image_size, config.image_size)
                    and cached.dtype == np.float32
                    and np.isfinite(cached).all()
                    and fragment.get("tensor_sha256") == _sha256_file(tensor_path)
                    and fragment.get("tensor_payload_sha256")
                    == _tensor_sha256(cached)
                )
                if valid:
                    refreshed = {
                        **fragment,
                        **_current_contract_metadata(row, recording_key=recording_key),
                        "source_size_bytes": source_size,
                        "source_mtime_ns": source_mtime_ns,
                        "source_sha256": source_hash,
                        "cache_status": "verified_hit",
                    }
                    _atomic_json(fragment_path, refreshed)
                    rows.append(refreshed)
                    continue

            claim: _PreprocessingClaim | None = None
            try:
                claim = _acquire_preprocessing_claim(
                    claim_path,
                    {
                        "recording_key": recording_key,
                        "source_sha256": source_hash,
                        "preprocessing_hash": config_hash,
                    },
                )
            except BlockingIOError as exc:
                raise RuntimeError(f"HST cache item is already claimed: {recording_key}") from exc

            try:
                result = preprocess_audio_path(snapshot.path, config)
                tensor_hash = ""
                if result.eligible and result.image is not None:
                    tensor_hash = _atomic_array(tensor_path, result.image)
                fragment: dict[str, object] = {
                    **_current_contract_metadata(row, recording_key=recording_key),
                    "eligible": bool(result.eligible),
                    "reason": result.reason,
                    "original_duration_seconds": result.original_duration_seconds,
                    "trimmed_duration_seconds": result.trimmed_duration_seconds,
                    "source_size_bytes": source_size,
                    "source_mtime_ns": source_mtime_ns,
                    "source_sha256": source_hash,
                    "decode_attempt": 1,
                    "cache_path": tensor_path.as_posix() if result.eligible else "",
                    "tensor_sha256": tensor_hash,
                    "tensor_payload_sha256": (
                        _tensor_sha256(result.image)
                        if result.eligible and result.image is not None
                        else ""
                    ),
                    "preprocessing_hash": config_hash,
                    "preprocessing_implementation_version": (
                        PREPROCESSING_IMPLEMENTATION_VERSION
                    ),
                    "representation_id": config.representation_id,
                    "cache_status": "written" if result.eligible else "excluded",
                }
                _atomic_json(fragment_path, fragment)
                rows.append(fragment)
            except BaseException as exc:
                claim.release(status="failed", failure=exc)
                raise
            else:
                claim.release(status="completed")
    return pd.DataFrame(rows)


def validate_hst_cache_index(index_path: Path, *, cache_root: Path) -> int:
    """Verify every eligible tensor referenced by a frozen cache index."""

    index = pd.read_csv(index_path, low_memory=False)
    required = {"eligible", "cache_path", "tensor_sha256"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"HST cache index missing columns: {missing}")
    root = Path(cache_root).resolve()
    eligible = index.loc[index["eligible"].astype(bool)].copy()
    for row in eligible.itertuples(index=False):
        tensor_path = Path(str(row.cache_path)).resolve()
        try:
            tensor_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("HST cache tensor path escaped the shared cache root") from exc
        if not tensor_path.is_file():
            raise FileNotFoundError(f"HST cache tensor is missing: {tensor_path}")
        array = np.load(tensor_path, allow_pickle=False)
        if array.ndim != 2 or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"HST cache tensor failed shape/dtype/finite checks: {tensor_path}")
        if _sha256_file(tensor_path) != str(row.tensor_sha256):
            raise ValueError(f"HST cache tensor checksum mismatch: {tensor_path}")
    return len(eligible)


def image_to_model_tensor(image: np.ndarray, config: HSTSpectrogramConfig) -> object:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to create HST model tensors") from exc
    array = np.asarray(image, dtype=np.float32)
    if array.shape != (config.image_size, config.image_size):
        raise ValueError(f"Expected {(config.image_size, config.image_size)}, found {array.shape}")
    contiguous = np.array(array, dtype=np.float32, order="C", copy=True)
    tensor = torch.from_numpy(contiguous).unsqueeze(0).repeat(3, 1, 1)
    return (tensor - config.normalization_mean) / config.normalization_std


def deterministic_augmentation_seed(
    *,
    seed: int,
    fold: int,
    epoch: int,
    recording_key: str,
    draw_id: int,
) -> int:
    payload = f"{seed}\0{fold}\0{epoch}\0{recording_key}\0{draw_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def augment_training_image(
    image: np.ndarray,
    *,
    seed: int,
    fold: int,
    epoch: int,
    recording_key: str,
    draw_id: int,
) -> np.ndarray:
    from PIL import Image

    generator = np.random.default_rng(
        deterministic_augmentation_seed(
            seed=seed,
            fold=fold,
            epoch=epoch,
            recording_key=recording_key,
            draw_id=draw_id,
        )
    )
    angle = float(generator.uniform(-20.0, 20.0))
    flip = bool(generator.random() < 0.5)
    pil = Image.fromarray(np.asarray(image, dtype=np.float32), mode="F")
    pil = pil.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=0.0)
    if flip:
        pil = pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return np.asarray(pil, dtype=np.float32)


def estimate_coughvid_snr(y: np.ndarray, sr: int, *, pinned_algorithm: object) -> float:
    estimator = getattr(pinned_algorithm, "estimate_snr", None)
    if not callable(estimator):
        raise ValueError("pinned_algorithm must expose a versioned estimate_snr function")
    value = float(estimator(np.asarray(y, dtype=np.float32), int(sr)))
    if not math.isfinite(value):
        raise ValueError("Pinned SNR algorithm returned a non-finite value")
    return value


def segment_cough_events(y: np.ndarray, sr: int, *, pinned_algorithm: object) -> list[np.ndarray]:
    segmenter = getattr(pinned_algorithm, "segment_cough_events", None)
    if not callable(segmenter):
        raise ValueError("pinned_algorithm must expose a versioned segment_cough_events function")
    events = [np.asarray(event, dtype=np.float32) for event in segmenter(np.asarray(y, dtype=np.float32), int(sr))]
    if any(event.ndim != 1 or not np.isfinite(event).all() for event in events):
        raise ValueError("Pinned cough-event algorithm returned an invalid event")
    return events


def build_hst_event_cache(
    metadata: pd.DataFrame,
    *,
    output_dir: Path,
    config: HSTSpectrogramConfig,
    snr_minimum: float = 0.8,
) -> pd.DataFrame:
    raise RuntimeError(
        "Event cache requires a checksum-pinned segmentation implementation; "
        "supply it through the dedicated sensitivity runner"
    )
