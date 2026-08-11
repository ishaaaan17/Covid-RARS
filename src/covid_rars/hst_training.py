from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import signal
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from covid_rars.hst_data_contracts import aggregate_to_participant
from covid_rars.hst_runtime import (
    _KernelFileLock,
    canonical_json_sha256,
    stable_file_sha256,
)
from covid_rars.hst_resource_pilot import (
    canonical_resource_benchmark_records,
    resource_pilot_freeze_payload,
)
from covid_rars.hst_spectrograms import (
    HSTSpectrogramConfig,
    deterministic_augmentation_seed,
    image_to_model_tensor,
)
from covid_rars.metrics import best_threshold_by_balanced_accuracy, binary_metric_bundle, labels_to_binary


_CONFIRMATORY_BATCH_PAIRS = {(8, 1), (4, 2), (2, 4)}
_TRACK_A_TRAINING_SEEDS = (1, 2, 5, 12, 40, 52, 72, 2002, 4002, 6002)
_TRACK_A_PROTOCOLS = {
    "hst_literature_aligned_repeated_holdout",
    "coswara_to_coughvid_hst_external",
}
_TEMPORAL_PROTOCOLS = {
    "hst_calendar_mixed_split_policy",
    "hst_chronological_split_policy",
    "hst_common_late_test_date_balanced_source",
    "hst_common_late_test_chronological_source",
    "hst_reverse_temporal_sensitivity",
}
_VERIFIED_CACHE_FILES: set[tuple[str, str, int, int, int, int]] = set()
_VERIFIED_CACHE_FILES_LOCK = threading.Lock()
_CHECKPOINT_WRITER = "covid_rars.hst_training._atomic_torch_save"
_TRUSTED_CHECKPOINT_WRITERS = frozenset(
    {
        _CHECKPOINT_WRITER,
        "covid_audio_btp.hst_training._atomic_torch_save",
    }
)
_EVALUATION_WRITER = "covid_rars.hst_training._evaluate_split_once"
_TRUSTED_EVALUATION_WRITERS = frozenset(
    {
        _EVALUATION_WRITER,
        "covid_audio_btp.hst_training._evaluate_split_once",
    }
)
_SLOT_ANCHOR_WRITER = "covid_rars.hst_training.slot_anchor"
_TRUSTED_SLOT_ANCHOR_WRITERS = frozenset(
    {
        _SLOT_ANCHOR_WRITER,
        "covid_audio_btp.hst_training.slot_anchor",
    }
)


@dataclass(frozen=True)
class HSTTrainingConfig:
    pilot_freeze_hash: str | None
    data_contracts_freeze_hash: str
    dependency_lock_hash: str
    accepted_environment_lock_hash: str | None
    physical_batch_size: int
    gradient_accumulation: int
    amp: bool
    resource_pilot_receipt_sha256: str | None = None
    approved_resource_pairs: tuple[tuple[int, int], ...] = ()
    max_epochs: int = 100
    effective_batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 1e-8
    gradient_clip_norm: float = 0.1
    scheduler_pct_start: float = 0.3
    scheduler_div_factor: float = 25.0
    scheduler_final_div_factor: float = 10000.0
    scheduler_anneal_strategy: str = "cos"
    selection_metric: str = "participant_auroc"
    epoch_selection_threshold: float = 0.5
    balance_training_classes: bool = True
    amp_max_skipped_updates: int = 0
    random_seed: int = 52
    deterministic_algorithms: bool = True
    wall_clock_checkpoint_interval_seconds: float = 1800.0
    confirmatory: bool = False

    def __post_init__(self) -> None:
        if self.physical_batch_size <= 0 or self.gradient_accumulation <= 0:
            raise ValueError("Physical batch size and gradient accumulation must be positive")
        if self.physical_batch_size * self.gradient_accumulation != self.effective_batch_size:
            prefix = "Confirmatory " if self.confirmatory else ""
            raise ValueError(
                f"{prefix}physical batch size times accumulation must equal the effective batch size"
            )
        for name in ("data_contracts_freeze_hash", "dependency_lock_hash"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.accepted_environment_lock_hash is not None and (
            self.accepted_environment_lock_hash != self.dependency_lock_hash
        ):
            raise ValueError("Accepted environment lock does not match the actual dependency lock")
        if not self.balance_training_classes:
            raise ValueError("HST training is frozen to the class-balanced hierarchical sampler")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        numeric_settings = (
            self.gradient_clip_norm,
            self.learning_rate,
            self.weight_decay,
            self.scheduler_pct_start,
            self.scheduler_div_factor,
            self.scheduler_final_div_factor,
        )
        if not all(math.isfinite(value) for value in numeric_settings):
            raise ValueError("Optimizer and scheduler settings must be finite")
        if self.gradient_clip_norm <= 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Optimizer settings must be finite and valid")
        if self.scheduler_div_factor <= 0 or self.scheduler_final_div_factor <= 0:
            raise ValueError("Scheduler division factors must be positive")
        if self.amp_max_skipped_updates < 0:
            raise ValueError("amp_max_skipped_updates cannot be negative")
        if not self.amp and self.amp_max_skipped_updates != 0:
            raise ValueError("AMP skip tolerance is invalid when AMP is disabled")
        if self.random_seed < 0:
            raise ValueError("random_seed cannot be negative")
        if (
            not math.isfinite(self.wall_clock_checkpoint_interval_seconds)
            or self.wall_clock_checkpoint_interval_seconds <= 0
        ):
            raise ValueError("The wall-clock checkpoint interval must be finite and positive")
        if self.confirmatory:
            if self.pilot_freeze_hash is None:
                raise ValueError("Confirmatory training requires an accepted pilot freeze hash")
            if self.max_epochs != 100:
                raise ValueError("Confirmatory HST training must complete exactly 100 epochs")
            for field_name in (
                "pilot_freeze_hash",
                "resource_pilot_receipt_sha256",
                "data_contracts_freeze_hash",
                "dependency_lock_hash",
                "accepted_environment_lock_hash",
            ):
                _validate_sha256(getattr(self, field_name), field_name=field_name)
            approved_pairs: set[tuple[int, int]] = set()
            for pair in self.approved_resource_pairs:
                if (
                    len(pair) != 2
                    or not all(isinstance(value, int) and value > 0 for value in pair)
                    or pair[0] * pair[1] != 8
                ):
                    raise ValueError(
                        "Confirmatory resource-pilot-approved pairs must be positive and "
                        "produce effective batch size 8"
                    )
                approved_pairs.add((int(pair[0]), int(pair[1])))
            if not approved_pairs or not approved_pairs.issubset(_CONFIRMATORY_BATCH_PAIRS):
                raise ValueError(
                    "Confirmatory resource-pilot-approved batch candidates are restricted to "
                    "(8,1), (4,2), and (2,4)"
                )
            selected_pair = (self.physical_batch_size, self.gradient_accumulation)
            if selected_pair not in approved_pairs:
                raise ValueError(
                    "Confirmatory physical batch/accumulation pair is not resource-pilot-approved"
                )
            if self.accepted_environment_lock_hash is None:
                raise ValueError("Confirmatory training requires an accepted environment lock")
            frozen_settings = {
                "learning_rate": 1e-5,
                "weight_decay": 1e-8,
                "gradient_clip_norm": 0.1,
                "scheduler_pct_start": 0.3,
                "scheduler_div_factor": 25.0,
                "scheduler_final_div_factor": 10000.0,
                "scheduler_anneal_strategy": "cos",
                "selection_metric": "participant_auroc",
                "epoch_selection_threshold": 0.5,
                "max_epochs": 100,
                "amp_max_skipped_updates": 0,
                "effective_batch_size": 8,
                "wall_clock_checkpoint_interval_seconds": 1800.0,
            }
            drifted = [
                name for name, expected in frozen_settings.items()
                if getattr(self, name) != expected
            ]
            if drifted:
                raise ValueError(
                    "Confirmatory settings differ from the frozen protocol: "
                    + ", ".join(drifted)
                )
            if self.random_seed not in {1, 2, 5, 12, 40, 42, 52, 72, 2002, 4002, 6002}:
                raise ValueError("Confirmatory random seed is not in the prespecified seed list")
            if not self.deterministic_algorithms:
                raise ValueError("Confirmatory HST training requires deterministic algorithms")
        if not 0.0 < self.scheduler_pct_start < 1.0:
            raise ValueError("scheduler_pct_start must be in (0, 1)")
        if self.epoch_selection_threshold != 0.5:
            raise ValueError("Checkpoint selection threshold is frozen at 0.5")
        if self.selection_metric != "participant_auroc":
            raise ValueError("Confirmatory checkpoint selection is frozen to participant AUROC")


@dataclass
class HSTFoldResult:
    last_epoch: int
    best_epoch: int
    resumed_from_epoch: int | None
    validation_threshold: float
    validation_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    external_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    skipped_optimizer_updates: int = 0
    training_complete: bool = False
    test_evaluated: bool = False
    external_evaluated: bool = False
    best_checkpoint_sha256: str | None = None
    training_contract_fingerprint: str | None = None
    interrupted: bool = False


_PREDICTION_CONTEXT_FIELDS = (
    "run_id",
    "protocol",
    "model",
    "checkpoint_hash",
    "representation",
    "architecture_sha256",
    "executable_sha256",
)

_SHA256_CONTEXT_FIELDS = {
    "checkpoint_hash",
    "architecture_sha256",
    "executable_sha256",
}


def validate_prediction_context(context: Mapping[str, object]) -> dict[str, str]:
    """Validate immutable provenance attached to every exported prediction."""
    validated: dict[str, str] = {}
    for field_name in _PREDICTION_CONTEXT_FIELDS:
        value = str(context.get(field_name, "")).strip()
        if not value or "pending" in value.casefold() or "placeholder" in value.casefold():
            raise ValueError(f"Invalid or missing prediction context field: {field_name}")
        validated[field_name] = value
    for field_name in _SHA256_CONTEXT_FIELDS:
        digest = validated[field_name]
        if len(digest) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in digest
        ):
            raise ValueError(f"{field_name} must be a 64-character SHA-256 digest")
    return validated


def training_contract_fingerprint(
    *,
    training_config: Mapping[str, object],
    manifest_sha256: str,
    cache_index_sha256: str,
    source_checkpoint_sha256: str,
    initial_model_state_sha256: str | None = None,
    initial_model_binding_sha256: str | None = None,
    optimizer_parameter_sha256: str | None = None,
    execution_identity: Mapping[str, object] | None = None,
    prediction_context: Mapping[str, object] | None = None,
) -> str:
    """Bind resumable optimizer state to exact data and scientific inputs."""
    hashes = {
        "manifest_sha256": _validate_sha256(
            manifest_sha256,
            field_name="manifest_sha256",
        ),
        "cache_index_sha256": _validate_sha256(
            cache_index_sha256,
            field_name="cache_index_sha256",
        ),
        "source_checkpoint_sha256": _validate_sha256(
            source_checkpoint_sha256,
            field_name="source_checkpoint_sha256",
        ),
    }
    payload: dict[str, object] = {
        "schema_version": 2,
        "training_config": dict(training_config),
        **hashes,
    }
    if initial_model_state_sha256 is not None:
        payload["initial_model_state_sha256"] = _validate_sha256(
            initial_model_state_sha256,
            field_name="initial_model_state_sha256",
        )
    if initial_model_binding_sha256 is not None:
        payload["initial_model_binding_sha256"] = _validate_sha256(
            initial_model_binding_sha256,
            field_name="initial_model_binding_sha256",
        )
    if optimizer_parameter_sha256 is not None:
        payload["optimizer_parameter_sha256"] = _validate_sha256(
            optimizer_parameter_sha256,
            field_name="optimizer_parameter_sha256",
        )
    if execution_identity is not None:
        payload["execution_identity"] = dict(execution_identity)
    if prediction_context is not None:
        payload["prediction_context"] = validate_prediction_context(prediction_context)
    return canonical_json_sha256(payload)


def validate_resume_checkpoint_contract(
    payload: Mapping[str, object], *, expected_fingerprint: str
) -> None:
    actual = str(payload.get("training_contract_fingerprint", ""))
    if actual != expected_fingerprint:
        raise ValueError(
            "Resume checkpoint training contract fingerprint does not match the current run"
        )


def build_training_execution_identity(
    loaders: Mapping[str, object],
    config: HSTTrainingConfig,
    *,
    prediction_context: Mapping[str, object],
) -> dict[str, object]:
    """Freeze the exact scientific fold and sampler identity used by a run."""
    try:
        fold = int(loaders["fold"])
        sampler_seed = int(loaders["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Training loaders require an exact fold and sampler seed") from exc
    modality = str(loaders.get("modality", "")).strip()
    representation = str(loaders.get("representation_id", "")).strip()
    if fold < 0 or not modality or not representation:
        raise ValueError("Training identity requires fold, modality, and representation")
    if sampler_seed != config.random_seed:
        raise ValueError("Training loader sampler seed must equal the frozen model seed")
    context_representation = str(prediction_context.get("representation", "")).strip()
    if context_representation != representation:
        raise ValueError("Training loader representation disagrees with prediction context")
    manifest = loaders.get("manifest")
    if not isinstance(manifest, pd.DataFrame):
        raise ValueError("Training identity requires the checksum-pinned manifest frame")
    required_manifest_identity = {
        "fold",
        "modality",
        "representation_id",
        "training_seed",
        "protocol",
    }
    missing = sorted(required_manifest_identity - set(manifest.columns))
    if missing:
        raise ValueError(f"Training manifest identity is missing columns: {missing}")
    selected_manifest = manifest.loc[
        pd.to_numeric(manifest["fold"], errors="coerce").eq(fold)
        & manifest["modality"].astype(str).eq(modality)
        & manifest["representation_id"].astype(str).eq(representation)
    ]
    if selected_manifest.empty:
        raise ValueError("Training fold is absent from the checksum-pinned manifest")
    try:
        training_seed_values = pd.to_numeric(
            selected_manifest["training_seed"], errors="raise"
        ).astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("Manifest training_seed must contain exact integers") from exc
    if not training_seed_values.astype(str).eq(
        selected_manifest["training_seed"].astype(str).str.strip()
    ).all():
        raise ValueError("Manifest training_seed must contain exact integers")
    manifest_seeds = set(training_seed_values.tolist())
    if manifest_seeds != {config.random_seed}:
        raise ValueError("Manifest training_seed does not equal the frozen model seed")
    manifest_protocols = set(selected_manifest["protocol"].astype(str).str.strip())
    context_protocol = str(prediction_context.get("protocol", "")).strip()
    if manifest_protocols != {context_protocol}:
        raise ValueError("Manifest protocol does not equal the prediction context protocol")
    if config.confirmatory:
        protocol = context_protocol
        if protocol in _TRACK_A_PROTOCOLS:
            if fold < 1 or fold > len(_TRACK_A_TRAINING_SEEDS):
                raise ValueError("Confirmatory Track-A fold has no frozen training seed")
            expected_seed = _TRACK_A_TRAINING_SEEDS[fold - 1]
            if config.random_seed != expected_seed:
                raise ValueError("Confirmatory Track-A fold/seed mapping is invalid")
        elif protocol in _TEMPORAL_PROTOCOLS:
            if fold != 1 or config.random_seed != 42:
                raise ValueError("Confirmatory temporal fold/model seed must be fold 1 and seed 42")
        else:
            raise ValueError("Confirmatory protocol has no prespecified model-seed context")
    return {
        "schema_version": 2,
        "fold": fold,
        "modality": modality,
        "representation_id": representation,
        "model_seed": int(config.random_seed),
        "sampler_seed": sampler_seed,
        "manifest_training_seed": int(config.random_seed),
        "manifest_protocol": context_protocol,
    }


def _stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _validate_sha256(value: object, *, field_name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 digest")
    return digest


def _frame_for_hash(frame: pd.DataFrame) -> pd.DataFrame:
    columns = sorted(str(column) for column in frame.columns)
    canonical = frame.loc[:, columns].copy()
    for column in columns:
        canonical[column] = canonical[column].astype("string").fillna("<NA>")
    return canonical


def _unordered_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash a table as a scientific row set, independent of storage order."""
    canonical = _frame_for_hash(frame)
    columns = list(canonical.columns)
    canonical = canonical.sort_values(columns, kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(
        b"unordered-frame-v1\0"
        + canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _ordered_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash an optimizer draw sequence while preserving exact row order."""
    canonical = _frame_for_hash(frame).reset_index(drop=True)
    return hashlib.sha256(
        b"ordered-frame-v1\0"
        + canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _canonical_frame_sha256(frame: pd.DataFrame) -> str:
    """Backward-compatible name for order-insensitive artifact identity."""
    return _unordered_frame_sha256(frame)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported training artifact table: {path}")


def verify_training_artifact_hashes(
    *,
    manifest_path: Path,
    cache_index_path: Path,
    source_checkpoint_path: Path,
    expected_manifest_sha256: str,
    expected_cache_index_sha256: str,
    expected_source_checkpoint_sha256: str,
) -> dict[str, str]:
    """Recompute all frozen input hashes from the files that will be used."""
    paths = {
        "manifest": Path(manifest_path),
        "cache_index": Path(cache_index_path),
        "source_checkpoint": Path(source_checkpoint_path),
    }
    expected = {
        "manifest": _validate_sha256(expected_manifest_sha256, field_name="manifest_sha256"),
        "cache_index": _validate_sha256(
            expected_cache_index_sha256, field_name="cache_index_sha256"
        ),
        "source_checkpoint": _validate_sha256(
            expected_source_checkpoint_sha256,
            field_name="source_checkpoint_sha256",
        ),
    }
    actual: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Training {name} artifact does not exist: {path}")
        actual[name] = stable_file_sha256(path)
        if actual[name] != expected[name]:
            raise ValueError(f"Training {name} checksum does not match its frozen hash")
    return actual


def verify_resource_pilot_receipt(
    path: Path,
    config: HSTTrainingConfig,
) -> str:
    """Verify the frozen selection and its complete resource-trial evidence."""
    receipt_path = Path(path)
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Resource pilot receipt is missing: {receipt_path}")
    expected_file_hash = _validate_sha256(
        config.resource_pilot_receipt_sha256,
        field_name="resource_pilot_receipt_sha256",
    )
    if stable_file_sha256(receipt_path) != expected_file_hash:
        raise ValueError("Resource pilot receipt checksum does not match the frozen config")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Resource pilot receipt is not valid JSON") from exc
    if not isinstance(receipt, dict):
        raise ValueError("Resource pilot receipt must be a JSON object")
    frozen_hash = _validate_sha256(
        receipt.get("pilot_freeze_hash"),
        field_name="resource pilot freeze hash",
    )
    if canonical_json_sha256(resource_pilot_freeze_payload(receipt)) != frozen_hash:
        raise ValueError("Resource pilot receipt freeze hash cannot be reproduced")
    if frozen_hash != _validate_sha256(
        config.pilot_freeze_hash,
        field_name="pilot_freeze_hash",
    ):
        raise ValueError("Resource pilot receipt does not match the accepted pilot freeze")
    def exact_json_integer(field_name: str) -> int:
        value = receipt.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"Resource pilot {field_name} must be an exact integer")
        return value

    pair = (
        exact_json_integer("physical_batch_size"),
        exact_json_integer("gradient_accumulation"),
    )
    effective_batch = exact_json_integer("effective_batch_size")
    if pair != (config.physical_batch_size, config.gradient_accumulation):
        raise ValueError("Resource pilot receipt selected a different batch/accumulation pair")
    if (
        pair not in _CONFIRMATORY_BATCH_PAIRS
        or pair not in set(config.approved_resource_pairs)
        or pair[0] * pair[1] != effective_batch
    ):
        raise ValueError("Resource pilot receipt pair is not in the approved frozen set")
    if effective_batch != config.effective_batch_size:
        raise ValueError("Resource pilot receipt effective batch differs from training")
    if receipt.get("model_metrics_used") is not False:
        raise ValueError("Resource pilot receipt must be independent of model endpoints")
    minimum_updates = exact_json_integer("minimum_optimizer_updates")
    if minimum_updates < 100:
        raise ValueError("Resource pilot must require at least 100 optimizer updates")

    def finite_receipt_number(field_name: str, *, positive: bool = False) -> float:
        try:
            value = float(receipt[field_name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Resource pilot {field_name} is missing or invalid") from exc
        if not math.isfinite(value) or (positive and value <= 0) or (not positive and value < 0):
            raise ValueError(f"Resource pilot {field_name} is not finite and valid")
        return value

    required_headroom = finite_receipt_number("headroom_required_bytes", positive=True)
    measured_free_vram = finite_receipt_number("measured_free_vram_bytes", positive=True)
    probability_tolerance = finite_receipt_number("probability_tolerance")
    relative_loss_tolerance = finite_receipt_number("relative_loss_tolerance")
    if probability_tolerance != 0.01 or relative_loss_tolerance != 0.01:
        raise ValueError("Resource pilot AMP tolerances must both be frozen at 0.01")
    precision = str(receipt.get("precision", ""))
    if precision not in {"fp32", "amp"} or (precision == "amp") is not config.amp:
        raise ValueError("Resource pilot precision does not match the training AMP setting")
    if receipt.get("amp") is not config.amp:
        raise ValueError("Resource pilot AMP flag does not match training")
    if receipt.get("selection_basis") != (
        "first_safe_batch_then_amp_if_numerically_valid_else_fp32"
    ):
        raise ValueError("Resource pilot selection basis is not the frozen rule")

    trials_path = receipt_path.with_name("base_resource_pilot_trials.csv")
    if not trials_path.is_file():
        raise FileNotFoundError(f"Resource pilot trials are missing: {trials_path}")
    try:
        trials = pd.read_csv(trials_path, low_memory=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("Resource pilot trial table is unreadable") from exc
    forbidden_metrics = {
        "accuracy",
        "auprc",
        "auroc",
        "balanced_accuracy",
        "f1",
        "sensitivity",
        "specificity",
    }
    forbidden = sorted(
        column for column in trials.columns if str(column).casefold() in forbidden_metrics
    )
    if forbidden:
        raise ValueError(f"Resource pilot trials contain forbidden model metrics: {forbidden}")
    required_columns = {
        "physical_batch_size",
        "precision",
        "valid",
        "optimizer_updates",
        "skipped_optimizer_updates",
        "seconds",
        "free_vram_bytes",
        "total_vram_bytes",
        "peak_allocated_vram_bytes",
        "peak_reserved_vram_bytes",
        "max_abs_probability_difference_from_fp32",
        "relative_loss_difference_from_fp32",
        "finite_loss",
        "finite_gradients",
        "finite_parameters",
        "finite_predictions",
    }
    missing = sorted(required_columns - set(trials.columns))
    if missing:
        raise ValueError(f"Resource pilot trials are missing columns: {missing}")
    hash_frame = trials.drop(columns=["evaluation_loss", "error"], errors="ignore")
    try:
        benchmark_records = canonical_resource_benchmark_records(hash_frame)
        benchmark_hash = canonical_json_sha256(benchmark_records)
    except (TypeError, ValueError) as exc:
        raise ValueError("Resource pilot benchmark contains non-canonical values") from exc
    expected_benchmark_hash = _validate_sha256(
        receipt.get("benchmark_sha256"),
        field_name="resource pilot benchmark_sha256",
    )
    if benchmark_hash != expected_benchmark_hash:
        raise ValueError("Resource pilot benchmark hash does not match its frozen receipt")

    batches = pd.to_numeric(trials["physical_batch_size"], errors="coerce")
    precisions = trials["precision"].astype(str)
    expected_trials = {
        (batch, precision_name)
        for batch in (8, 4, 2)
        for precision_name in ("fp32", "amp")
    }
    actual_trials = set(zip(batches.tolist(), precisions.tolist(), strict=True))
    if actual_trials != expected_trials or len(trials) != len(expected_trials):
        raise ValueError(
            "Resource pilot trials must cover only batches 8/4/2 in fp32 and AMP exactly once"
        )
    selected_rows = trials.loc[
        batches.eq(pair[0]) & precisions.eq(precision)
    ]
    if len(selected_rows) != 1:
        raise ValueError("Resource pilot selected trial is missing or duplicated")
    selected = selected_rows.iloc[0]

    def explicit_trial_bool(field_name: str) -> bool:
        value = selected[field_name]
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str) and value in {"True", "False"}:
            return value == "True"
        raise ValueError(f"Resource pilot {field_name} must be an explicit boolean")

    if not explicit_trial_bool("valid"):
        raise ValueError("Resource pilot selected candidate must be explicitly valid")
    for field_name in (
        "finite_loss",
        "finite_gradients",
        "finite_parameters",
        "finite_predictions",
    ):
        if not explicit_trial_bool(field_name):
            raise ValueError(f"Resource pilot {field_name} must be explicitly true")

    def exact_trial_integer(field_name: str) -> int:
        value = selected[field_name]
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
            raise ValueError(f"Resource pilot {field_name} must be an exact integer")
        return int(value)

    optimizer_updates = exact_trial_integer("optimizer_updates")
    skipped_updates = exact_trial_integer("skipped_optimizer_updates")
    if optimizer_updates < minimum_updates:
        raise ValueError("Resource pilot selected trial has too few optimizer updates")
    if skipped_updates != 0:
        raise ValueError("Resource pilot requires zero skipped optimizer updates")

    def finite_trial_number(field_name: str, *, positive: bool = False) -> float:
        try:
            value = float(selected[field_name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Resource pilot {field_name} is missing or invalid") from exc
        if not math.isfinite(value) or (positive and value <= 0) or (not positive and value < 0):
            raise ValueError(f"Resource pilot {field_name} is not finite and valid")
        return value

    seconds = finite_trial_number("seconds", positive=True)
    free_vram = finite_trial_number("free_vram_bytes", positive=True)
    total_vram = finite_trial_number("total_vram_bytes", positive=True)
    peak_allocated = finite_trial_number("peak_allocated_vram_bytes")
    peak_reserved = finite_trial_number("peak_reserved_vram_bytes")
    if seconds <= 0 or free_vram < required_headroom:
        raise ValueError("Resource pilot runtime or free-memory headroom is unsafe")
    if free_vram != measured_free_vram:
        raise ValueError("Resource pilot measured free memory differs from selected trial")
    if total_vram != finite_receipt_number("total_vram_bytes", positive=True):
        raise ValueError("Resource pilot total memory differs from selected trial")
    if peak_allocated > peak_reserved or peak_reserved > total_vram or free_vram > total_vram:
        raise ValueError("Resource pilot memory measurements are internally inconsistent")
    probability_difference = finite_trial_number(
        "max_abs_probability_difference_from_fp32"
    )
    relative_loss_difference = finite_trial_number("relative_loss_difference_from_fp32")
    if config.amp and (
        probability_difference > probability_tolerance
        or relative_loss_difference > relative_loss_tolerance
    ):
        raise ValueError("Resource pilot AMP numerical agreement exceeds 0.01")
    return frozen_hash


def _strict_eligibility(values: pd.Series) -> pd.Series:
    parsed: list[bool] = []
    for value in values.tolist():
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
        elif isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value in (0, 1):
            parsed.append(bool(value))
        else:
            raise ValueError("eligible must contain only explicit booleans or integer 0/1")
    return pd.Series(parsed, index=values.index, dtype=bool)


def _validate_binary_labels(frame: pd.DataFrame, *, name: str) -> None:
    if "label_binary" not in frame:
        raise ValueError(f"{name} is missing label_binary")
    labels = frame["label_binary"]
    if any(not isinstance(value, str) for value in labels.tolist()):
        raise ValueError(f"{name} labels must be canonical strings")
    invalid = sorted(set(labels) - {"negative", "positive"})
    if invalid:
        raise ValueError(f"{name} contains invalid labels: {invalid}")


def _validate_dataset_roles(frame: pd.DataFrame, *, name: str) -> None:
    required = {"dataset", "participant_key", "recording_key", "split", "modality"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing dataset-role columns: {missing}")
    for row in frame.loc[:, sorted(required)].itertuples(index=False):
        values = row._asdict()
        dataset = str(values["dataset"]).strip().casefold()
        split = str(values["split"]).strip()
        modality = str(values["modality"]).strip().casefold()
        if dataset not in {"coswara", "coughvid"}:
            raise ValueError(f"{name} contains an unsupported dataset role: {dataset!r}")
        prefix = f"{dataset}::"
        for identity_name in ("participant_key", "recording_key"):
            identity = str(values[identity_name]).strip()
            if not identity.startswith(prefix) or identity == prefix:
                raise ValueError(
                    f"{name} {identity_name} must be an exact dataset-qualified identity"
                )
        if split in {"train", "validation", "test"} and dataset != "coswara":
            raise ValueError(f"{name} development rows must use the Coswara dataset")
        if split == "external_test" and (
            dataset != "coughvid" or modality != "cough"
        ):
            raise ValueError(
                f"{name} external_test rows must use COUGHVID cough recordings only"
            )


def _verify_selected_cache_file(path_value: object, digest_value: object) -> None:
    path = Path(str(path_value)).expanduser().resolve(strict=False)
    digest = _validate_sha256(digest_value, field_name="tensor_sha256")
    if not path.is_file():
        raise FileNotFoundError(f"Cached tensor does not exist: {path}")
    before = path.stat()
    identity = (
        os.path.normcase(str(path)),
        digest,
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
    )
    with _VERIFIED_CACHE_FILES_LOCK:
        if identity in _VERIFIED_CACHE_FILES:
            return
    load_verified_cached_image(path, digest)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"Cached tensor changed while it was being verified: {path}")
    with _VERIFIED_CACHE_FILES_LOCK:
        _VERIFIED_CACHE_FILES.add(identity)


def validate_manifest_cache_contract(
    cache_index: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    fold: int,
    modality: str,
    representation_id: str | None = None,
) -> pd.DataFrame:
    """Return an exact alignment without opening any cached tensor payload."""
    required_cache = {
        "dataset",
        "participant_key",
        "recording_key",
        "label_binary",
        "modality",
        "eligible",
        "cache_path",
        "tensor_sha256",
        "preprocessing_hash",
        "representation_id",
    }
    required_manifest = {
        "fold",
        "training_seed",
        "protocol",
        "split",
        "dataset",
        "participant_key",
        "recording_key",
        "label_binary",
        "modality",
        "source_audio_sha256",
        "preprocessing_hash",
        "representation_id",
    }
    normalized_cache = cache_index.copy()
    if "source_audio_sha256" not in normalized_cache and "source_sha256" in normalized_cache:
        normalized_cache["source_audio_sha256"] = normalized_cache["source_sha256"]
    required_cache.add("source_audio_sha256")
    for name, required, frame in (
        ("cache index", required_cache, normalized_cache),
        ("manifest", required_manifest, manifest),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"HST {name} missing columns: {missing}")

    selected_manifest = manifest.loc[
        manifest["fold"].eq(fold) & manifest["modality"].eq(modality)
    ].copy()
    manifest_representations = set(selected_manifest["representation_id"].astype(str))
    if representation_id is None:
        if len(manifest_representations) != 1:
            raise ValueError("A single frozen representation_id must be selected for training")
        representation_id = next(iter(manifest_representations))
    selected_manifest = selected_manifest.loc[
        selected_manifest["representation_id"].astype(str).eq(representation_id)
    ].copy()
    if selected_manifest.empty:
        raise ValueError(f"No manifest rows for fold={fold}, modality={modality}")
    _validate_binary_labels(selected_manifest, name="manifest")
    if selected_manifest["recording_key"].duplicated().any():
        raise ValueError("Manifest contains duplicate recording keys within a fold")

    identity_columns = ["dataset", "recording_key", "representation_id"]
    selected_identities = pd.MultiIndex.from_frame(
        selected_manifest.loc[:, identity_columns].astype(str)
    )
    cache_identities = pd.MultiIndex.from_frame(
        normalized_cache.loc[:, identity_columns].astype(str)
    )
    selected_cache = normalized_cache.loc[
        cache_identities.isin(selected_identities)
    ].copy()
    duplicate_cache_identity = selected_cache.duplicated(identity_columns, keep=False)
    if duplicate_cache_identity.any():
        raise ValueError(
            "Selected cache index contains duplicate dataset-qualified recording keys "
            "for a representation"
        )
    _validate_binary_labels(selected_cache, name="selected cache index")
    if "source_sha256" in selected_cache:
        if not selected_cache["source_audio_sha256"].astype(str).str.lower().eq(
            selected_cache["source_sha256"].astype(str).str.lower()
        ).all():
            raise ValueError("Cache source_sha256 aliases disagree")
    eligibility = _strict_eligibility(selected_cache["eligible"])
    if not eligibility.all():
        ineligible = selected_cache.loc[~eligibility, "recording_key"].astype(str).tolist()
        raise ValueError(f"Frozen manifest references ineligible cache rows: {ineligible[:5]}")

    manifest_keys = set(selected_manifest["recording_key"].astype(str))
    cache_keys = set(selected_cache["recording_key"].astype(str))
    if manifest_keys != cache_keys:
        missing_cache = sorted(manifest_keys - cache_keys)
        extra_cache = sorted(cache_keys - manifest_keys)
        raise ValueError(
            "Manifest and eligible cache must exactly cover the same recordings; "
            f"missing={missing_cache[:5]}, extra={extra_cache[:5]}"
        )

    identity_columns = [
        "recording_key",
        "dataset",
        "participant_key",
        "label_binary",
        "modality",
        "source_audio_sha256",
        "preprocessing_hash",
    ]
    identity_columns.append("representation_id")
    identity_set = set(identity_columns)
    overlapping_payload_columns = sorted(
        (
            set(selected_manifest.columns)
            & set(selected_cache.columns)
            & required_cache
        )
        - identity_set
    )
    if overlapping_payload_columns:
        manifest_overlap = selected_manifest[
            identity_columns + overlapping_payload_columns
        ].rename(
            columns={
                column: f"manifest::{column}"
                for column in overlapping_payload_columns
            }
        )
        cache_overlap = selected_cache[
            identity_columns + overlapping_payload_columns
        ].rename(
            columns={
                column: f"cache::{column}" for column in overlapping_payload_columns
            }
        )
        overlap = manifest_overlap.merge(
            cache_overlap,
            on=identity_columns,
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        if not overlap["_merge"].eq("both").all():
            raise ValueError("Manifest identity or labels disagree with the cache contract")
        mismatched_columns: list[str] = []
        for column in overlapping_payload_columns:
            manifest_values = overlap[f"manifest::{column}"]
            cache_values = overlap[f"cache::{column}"]
            equal = manifest_values.eq(cache_values) | (
                manifest_values.isna() & cache_values.isna()
            )
            if not equal.all():
                mismatched_columns.append(column)
        if mismatched_columns:
            raise ValueError(
                "Manifest cache provenance disagrees for overlapping columns: "
                f"{mismatched_columns}"
            )
    cache_payload_columns = [
        column
        for column in selected_cache.columns
        if column not in identity_set and column not in selected_manifest.columns
    ]
    aligned = selected_manifest.merge(
        selected_cache[identity_columns + cache_payload_columns],
        on=identity_columns,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Manifest identity or labels disagree with the cache contract")
    aligned = aligned.drop(columns="_merge")

    participant_labels = aligned.groupby("participant_key")["label_binary"].nunique()
    if (participant_labels != 1).any():
        raise ValueError("A participant has contradictory labels")
    participant_splits = aligned.groupby("participant_key")["split"].nunique()
    if (participant_splits != 1).any():
        raise ValueError("Detected participant overlap across train/validation/test splits")
    duplicated_audio = aligned.loc[
        aligned["source_audio_sha256"].astype(str).str.lower().duplicated(keep=False)
    ].copy()
    if not duplicated_audio.empty:
        duplicated_audio["source_audio_sha256"] = (
            duplicated_audio["source_audio_sha256"].astype(str).str.lower()
        )
        cross_split = duplicated_audio.groupby("source_audio_sha256")["split"].nunique()
        if (cross_split > 1).any():
            raise ValueError(
                "Detected content-level cross-split leakage: identical source audio bytes "
                "occur in multiple frozen splits"
            )
        raise ValueError(
            "Detected duplicate source audio content across participants or recordings"
        )
    normalized_cache_paths = aligned["cache_path"].map(
        lambda value: os.path.normcase(
            str(Path(str(value)).expanduser().resolve(strict=False))
        )
    )
    for identity_name, identities in (
        ("cache_path", normalized_cache_paths),
        ("tensor_sha256", aligned["tensor_sha256"].astype(str).str.lower()),
    ):
        duplicated = aligned.assign(_model_input_identity=identities).loc[
            identities.duplicated(keep=False)
        ]
        if not duplicated.empty and (
            duplicated.groupby("_model_input_identity")["split"].nunique() > 1
        ).any():
            raise ValueError(
                f"Detected {identity_name} cross-split leakage: the same model input "
                "occurs in multiple frozen splits"
            )
    valid_splits = {"train", "validation", "test", "external_test"}
    invalid_splits = sorted(set(aligned["split"].astype(str)) - valid_splits)
    if invalid_splits:
        raise ValueError(f"Manifest contains unsupported splits: {invalid_splits}")
    _validate_dataset_roles(aligned, name="frozen HST manifest/cache alignment")
    for digest in aligned["tensor_sha256"]:
        _validate_sha256(digest, field_name="tensor_sha256")
    for digest in aligned["source_audio_sha256"]:
        _validate_sha256(digest, field_name="source_audio_sha256")
    for digest in aligned["preprocessing_hash"]:
        _validate_sha256(digest, field_name="preprocessing_hash")
    return aligned.sort_values(["split", "recording_key"], kind="mergesort").reset_index(drop=True)


def load_verified_cached_image(path: Path | str, expected_sha256: str) -> np.ndarray:
    resolved = Path(path)
    expected = _validate_sha256(expected_sha256, field_name="tensor_sha256")
    if not resolved.is_file():
        raise FileNotFoundError(f"Cached tensor does not exist: {resolved}")
    actual = stable_file_sha256(resolved)
    if actual != expected:
        raise ValueError(f"Cached tensor checksum mismatch: {resolved}")
    image = np.load(resolved, allow_pickle=False)
    if image.shape != (224, 224):
        raise ValueError(f"Cached spectrogram must have shape (224, 224): {resolved}")
    if image.dtype != np.float32:
        raise ValueError(f"Cached spectrogram must have dtype float32: {resolved}")
    if not np.isfinite(image).all():
        raise ValueError(f"Cached spectrogram must contain only finite values: {resolved}")
    return image


def validate_evaluation_request(
    *,
    training_complete: bool,
    evaluate_test: bool,
    evaluate_external: bool,
    available_splits: set[str],
    confirmatory: bool = True,
) -> None:
    if (evaluate_test or evaluate_external) and not training_complete:
        raise ValueError("Held-out evaluation requires complete training")
    if (evaluate_test or evaluate_external) and not confirmatory:
        raise ValueError("Smoke and pilot runs may never evaluate held-out data")
    if evaluate_test and "test" not in available_splits:
        raise ValueError("Requested test evaluation but no test loader is available")
    if evaluate_external and "external_test" not in available_splits:
        raise ValueError("Requested external_test evaluation but no external_test loader is available")


def build_hierarchical_epoch_draw_plan(
    cache_index: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    fold: int,
    modality: str,
    epoch: int,
    seed: int,
    representation_id: str | None = None,
) -> pd.DataFrame:
    required_cache = {
        "dataset",
        "participant_key",
        "recording_key",
        "label_binary",
        "modality",
        "eligible",
        "cache_path",
        "tensor_sha256",
        "preprocessing_hash",
        "representation_id",
    }
    required_manifest = {
        "fold",
        "split",
        "dataset",
        "participant_key",
        "recording_key",
        "label_binary",
        "modality",
        "source_audio_sha256",
        "preprocessing_hash",
        "representation_id",
    }
    for name, required, frame in (
        ("cache index", required_cache, cache_index),
        ("manifest", required_manifest, manifest),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"HST {name} missing columns: {missing}")
    aligned = validate_manifest_cache_contract(
        cache_index,
        manifest,
        fold=fold,
        modality=modality,
        representation_id=representation_id,
    )
    training = aligned.loc[aligned["split"].eq("train")].copy()
    if training.empty:
        raise ValueError(f"No training rows for fold={fold}, modality={modality}")
    if set(training["label_binary"].unique()) != {"negative", "positive"}:
        raise ValueError("Hierarchical training sampler requires both frozen classes")
    if training["recording_key"].duplicated().any():
        raise ValueError("Training manifest contains duplicate recording keys")
    participants_by_class = {
        label: sorted(group["participant_key"].astype(str).unique())
        for label, group in training.groupby("label_binary", sort=True)
    }
    if set(participants_by_class) != {"negative", "positive"}:
        raise ValueError("Eligible training cache does not contain both classes")
    draws_per_class = max(len(values) for values in participants_by_class.values())
    rows: list[dict[str, object]] = []
    for label in ("negative", "positive"):
        participants = participants_by_class[label]
        participant_rng = np.random.default_rng(_stable_seed(seed, fold, modality, epoch, label, "participants"))
        order = list(np.asarray(participants)[participant_rng.permutation(len(participants))])
        for class_draw_index in range(draws_per_class):
            participant_key = str(order[class_draw_index % len(order)])
            candidates = training.loc[
                training["participant_key"].eq(participant_key)
                & training["label_binary"].eq(label)
            ].sort_values("recording_key")
            recording_rng = np.random.default_rng(
                _stable_seed(seed, fold, modality, epoch, label, participant_key, class_draw_index)
            )
            selected_position = int(recording_rng.integers(0, len(candidates)))
            selected = candidates.iloc[selected_position]
            if (
                str(selected["split"]) != "train"
                or str(selected["modality"]) != modality
                or str(selected["label_binary"]) != label
                or int(selected["fold"]) != int(fold)
            ):
                raise RuntimeError("Selected training draw violates its verified manifest identity")
            draw_id = f"fold{fold:02d}::epoch{epoch:03d}::{label}::{class_draw_index:06d}"
            rows.append(
                {
                    **selected.to_dict(),
                    "draw_id": draw_id,
                    "augmentation_seed": deterministic_augmentation_seed(
                        seed=seed,
                        fold=fold,
                        epoch=epoch,
                        recording_key=str(selected["recording_key"]),
                        draw_id=class_draw_index + (0 if label == "negative" else draws_per_class),
                    ),
                }
            )
    draw_plan = pd.DataFrame(rows)
    shuffle_rng = np.random.default_rng(_stable_seed(seed, fold, modality, epoch, "interleave"))
    draw_plan = draw_plan.iloc[shuffle_rng.permutation(len(draw_plan))].reset_index(drop=True)
    if draw_plan["draw_id"].duplicated().any() or draw_plan["augmentation_seed"].duplicated().any():
        raise RuntimeError("Hierarchical draw plan contains duplicate draw identities")
    return draw_plan


class _CachedImageDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        training: bool,
        seed: int,
        fold: int,
        epoch: int,
        spectrogram_config: HSTSpectrogramConfig,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.training = training
        self.seed = seed
        self.fold = fold
        self.epoch = epoch
        self.spectrogram_config = spectrogram_config

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[object, object, dict[str, object]]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for HST datasets") from exc
        row = self.frame.iloc[index].to_dict()
        label_binary = row.get("label_binary")
        if not isinstance(label_binary, str) or label_binary not in {"negative", "positive"}:
            raise ValueError("Cached HST rows require canonical negative/positive labels")
        image = load_verified_cached_image(
            str(row["cache_path"]),
            str(row["tensor_sha256"]),
        )
        if self.training:
            if "augmentation_seed" not in row:
                raise ValueError("Training draw is missing its frozen augmentation_seed")
            image = _augment_image_with_exact_seed(image, int(row["augmentation_seed"]))
        tensor = image_to_model_tensor(image, self.spectrogram_config)
        label = torch.tensor(1 if label_binary == "positive" else 0, dtype=torch.long)
        return tensor, label, row


def _augment_image_with_exact_seed(image: np.ndarray, augmentation_seed: int) -> np.ndarray:
    """Apply the frozen augmentation using the seed recorded in the draw plan."""
    from PIL import Image

    generator = np.random.default_rng(int(augmentation_seed))
    angle = float(generator.uniform(-20.0, 20.0))
    flip = bool(generator.random() < 0.5)
    pil = Image.fromarray(np.asarray(image, dtype=np.float32), mode="F")
    pil = pil.rotate(
        angle,
        resample=Image.Resampling.NEAREST,
        expand=False,
        fillcolor=0.0,
    )
    if flip:
        pil = pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return np.array(pil, dtype=np.float32, copy=True)


def _collate(batch: list[tuple[object, object, dict[str, object]]]) -> tuple[object, object, list[dict[str, object]]]:
    import torch

    images, labels, rows = zip(*batch)
    return torch.stack(list(images)), torch.stack(list(labels)), list(rows)


def _seed_dataloader_worker(worker_id: int) -> None:
    import torch

    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_hst_dataloaders(
    cache_index: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    fold: int,
    modality: str,
    physical_batch_size: int,
    num_workers: int,
    seed: int,
    representation_id: str | None = None,
) -> dict[str, object]:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to build HST DataLoaders") from exc
    if physical_batch_size <= 0 or num_workers < 0:
        raise ValueError("Invalid HST DataLoader sizing")
    config = HSTSpectrogramConfig.paper_default()
    aligned = validate_manifest_cache_contract(
        cache_index,
        manifest,
        fold=fold,
        modality=modality,
        representation_id=representation_id,
    )
    effective_representation = representation_id
    if effective_representation is None:
        representations = aligned["representation_id"].astype(str).unique().tolist()
        if len(representations) != 1:
            raise ValueError("HST loaders require exactly one frozen representation_id")
        effective_representation = representations[0]

    def shared_rows(split: str) -> pd.DataFrame:
        return (
            aligned.loc[aligned["split"].eq(split)]
            .sort_values("recording_key", kind="mergesort")
            .reset_index(drop=True)
        )

    def loader_for_frame(frame: pd.DataFrame, *, training: bool, epoch: int) -> object:
        dataset = _CachedImageDataset(
            frame,
            training=training,
            seed=seed,
            fold=fold,
            epoch=epoch,
            spectrogram_config=config,
        )
        options: dict[str, object] = {
            "batch_size": physical_batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "collate_fn": _collate,
            "worker_init_fn": _seed_dataloader_worker,
        }
        generator = torch.Generator()
        generator.manual_seed(_stable_seed(seed, fold, modality, epoch, "loader", training))
        options["generator"] = generator
        if num_workers > 0:
            options.update(
                {
                    "persistent_workers": not training,
                    "prefetch_factor": 2,
                    "multiprocessing_context": "spawn",
                }
            )
        return DataLoader(dataset, **options)

    validation_rows = shared_rows("validation")
    test_rows = shared_rows("test")
    external_rows = shared_rows("external_test")
    if validation_rows.empty or test_rows.empty:
        raise ValueError("HST fold requires non-empty validation and test rows")

    def train_factory(epoch: int) -> object:
        draw_plan = build_hierarchical_epoch_draw_plan(
            cache_index,
            manifest,
            fold=fold,
            modality=modality,
            epoch=epoch,
            seed=seed,
            representation_id=effective_representation,
        )
        return loader_for_frame(draw_plan, training=True, epoch=epoch)

    result = {
        "train_factory": train_factory,
        "validation": loader_for_frame(validation_rows, training=False, epoch=0),
        "test": loader_for_frame(test_rows, training=False, epoch=0),
        "fold": fold,
        "modality": modality,
        "seed": seed,
        "representation_id": effective_representation,
        "cache_index": cache_index,
        "manifest": manifest,
        "manifest_frame_sha256": _canonical_frame_sha256(manifest),
        "cache_index_frame_sha256": _canonical_frame_sha256(cache_index),
    }
    if not external_rows.empty:
        result["external_test"] = loader_for_frame(
            external_rows,
            training=False,
            epoch=0,
        )
    return result


def aggregate_recording_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return aggregate_to_participant(predictions)


def _model_device(model: object) -> object:
    try:
        return next(model.parameters()).device  # type: ignore[attr-defined]
    except StopIteration:
        import torch

        return torch.device("cpu")


def predict_hst_split(
    model: object,
    loader: object,
    *,
    split: str,
    fold: int,
    modality: str,
    prediction_context: Mapping[str, object],
) -> pd.DataFrame:
    import torch

    if split not in {"validation", "test", "external_test"}:
        raise ValueError(f"Unsupported HST prediction split: {split}")
    context = validate_prediction_context(prediction_context)
    device = _model_device(model)
    model.eval()  # type: ignore[attr-defined]
    rows_out: list[dict[str, object]] = []
    with torch.no_grad():
        for images, labels, rows in loader:  # type: ignore[union-attr]
            logits = model(images.to(device, non_blocking=True))  # type: ignore[operator]
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError("HST logits must use frozen class order [negative, positive]")
            probabilities = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            label_values = labels.cpu().numpy().astype(int)
            for metadata, label_value, probability in zip(rows, label_values, probabilities, strict=True):
                if str(metadata.get("split", "")) != split:
                    raise ValueError(
                        f"Loader row split {metadata.get('split')!r} does not match requested {split!r}"
                    )
                try:
                    metadata_fold = int(metadata.get("fold"))
                except (TypeError, ValueError) as exc:
                    raise ValueError("Loader row is missing a valid fold") from exc
                if metadata_fold != int(fold):
                    raise ValueError("Loader row fold does not match the requested fold")
                if str(metadata.get("modality", "")) != modality:
                    raise ValueError("Loader row modality does not match the requested modality")
                expected_label = "positive" if label_value == 1 else "negative"
                if metadata.get("label_binary") != expected_label:
                    raise ValueError("HST batch label order disagrees with metadata")
                participant_key = str(metadata.get("participant_key", ""))
                recording_key = str(metadata.get("recording_key", ""))
                dataset = str(metadata.get("dataset", ""))
                if not dataset or not participant_key.startswith(f"{dataset}::"):
                    raise ValueError("Loader row has invalid dataset-qualified participant identity")
                if not recording_key.startswith(f"{dataset}::"):
                    raise ValueError("Loader row has invalid dataset-qualified recording identity")
                rows_out.append(
                    {
                        **metadata,
                        "run_id": context["run_id"],
                        "protocol": context["protocol"],
                        "fold": int(fold),
                        "dataset": dataset,
                        "split": str(metadata["split"]),
                        "modality": modality,
                        "model": context["model"],
                        "checkpoint_hash": context["checkpoint_hash"],
                        "representation": context["representation"],
                        "architecture_sha256": context["architecture_sha256"],
                        "executable_sha256": context["executable_sha256"],
                        "probability": float(probability),
                    }
                )
    return pd.DataFrame(rows_out)


def _participant_selection_metrics(predictions: pd.DataFrame, threshold: float) -> dict[str, float]:
    participants = aggregate_recording_predictions(predictions)
    y_true = labels_to_binary(participants["label_binary"])
    probability = participants["probability"].to_numpy(dtype=float)
    return binary_metric_bundle(y_true, probability, threshold=threshold)


def validation_epoch_score(
    metrics: Mapping[str, object], *, epoch: int
) -> tuple[float, float, float, float]:
    """Lexicographic validation-only score for the AUROC primary endpoint."""
    required = ("auroc", "auprc", "nll")
    values = {name: float(metrics[name]) for name in required}
    if epoch <= 0 or not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Validation checkpoint metrics and epoch must be finite and valid")
    return (
        values["auroc"],
        values["auprc"],
        -values["nll"],
        -float(epoch),
    )


def _sample_weighted_epoch_loss(*, total_loss: float, sample_count: int) -> float:
    if sample_count <= 0:
        raise ValueError("Epoch loss sample count must be positive")
    if not math.isfinite(total_loss):
        raise ValueError("Epoch total loss must be finite")
    return float(total_loss / sample_count)


def _rng_state() -> dict[str, object]:
    import torch

    numpy_state = np.random.get_state()
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": torch.as_tensor(
                np.asarray(numpy_state[1], dtype=np.int64), dtype=torch.int64
            ),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, object]) -> None:
    import torch

    random.setstate(state["python"])  # type: ignore[arg-type]
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise ValueError("Checkpoint NumPy RNG state is malformed")
    numpy_values = numpy_state["state"]
    if not isinstance(numpy_values, torch.Tensor):
        raise ValueError("Checkpoint NumPy RNG values are not a restricted tensor")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_values.detach().cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.random.set_rng_state(state["torch_cpu"])  # type: ignore[arg-type]
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])  # type: ignore[arg-type]


def _configure_determinism(seed: int, device: object, *, enabled: bool) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_type = str(getattr(device, "type", device))
    if device_type == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                "CUDA deterministic training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                "before CUDA context initialization"
            )
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(enabled)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(enabled)


def _model_state_sha256(model: object) -> str:
    import torch

    digest = hashlib.sha256()
    state = model.state_dict()  # type: ignore[attr-defined]
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Model state {name!r} is not a tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _model_architecture_sha256(model: object) -> str:
    state = model.state_dict()  # type: ignore[attr-defined]
    source = ""
    try:
        source = inspect.getsource(type(model))
    except (OSError, TypeError):
        source = "source-unavailable"
    payload = {
        "class_module": type(model).__module__,
        "class_qualname": type(model).__qualname__,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "state_schema": [
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
            for name, value in sorted(state.items())
        ],
    }
    return canonical_json_sha256(payload)


def _verify_full_finetuning_optimizer(model: object, optimizer: object) -> str:
    """Require one optimizer reference for every unfrozen backbone/head parameter."""
    named_parameters = list(model.named_parameters())  # type: ignore[attr-defined]
    if not named_parameters:
        raise ValueError("Confirmatory HST model exposes no parameters")
    names = [str(name) for name, _ in named_parameters]
    if len(names) != len(set(names)):
        raise ValueError("Confirmatory HST model exposes duplicate parameter names")
    head_names = [name for name in names if name.startswith("head.")]
    backbone_names = [name for name in names if not name.startswith("head.")]
    if not head_names or not backbone_names:
        raise ValueError("Confirmatory HST requires both backbone and head parameters")
    frozen = [name for name, parameter in named_parameters if not parameter.requires_grad]
    if frozen:
        raise ValueError(
            "Confirmatory full-backbone fine-tuning rejects frozen requires_grad parameters: "
            + ", ".join(frozen[:10])
        )

    expected_by_id = {id(parameter): name for name, parameter in named_parameters}
    if len(expected_by_id) != len(named_parameters):
        raise ValueError("Confirmatory HST parameter aliases cannot be audited exactly once")
    observed_counts: dict[int, int] = {}
    unknown: list[int] = []
    param_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(param_groups, list) or not param_groups:
        raise ValueError("Confirmatory optimizer exposes no parameter groups")
    for group in param_groups:
        if not isinstance(group, Mapping) or "params" not in group:
            raise ValueError("Confirmatory optimizer has an invalid parameter group")
        for parameter in group["params"]:  # type: ignore[index]
            parameter_id = id(parameter)
            if parameter_id not in expected_by_id:
                unknown.append(parameter_id)
            observed_counts[parameter_id] = observed_counts.get(parameter_id, 0) + 1
    missing = [
        name for parameter_id, name in expected_by_id.items()
        if observed_counts.get(parameter_id, 0) == 0
    ]
    duplicated = [
        expected_by_id.get(parameter_id, "<unknown>")
        for parameter_id, count in observed_counts.items()
        if count != 1
    ]
    if missing or duplicated or unknown or set(observed_counts) != set(expected_by_id):
        raise ValueError(
            "Confirmatory optimizer must cover each model parameter exactly once; "
            f"missing={missing[:10]}, duplicate={duplicated[:10]}, unknown={len(unknown)}"
        )
    return canonical_json_sha256(
        {
            "schema_version": 1,
            "full_backbone_finetuning": True,
            "parameters": [
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "requires_grad": bool(parameter.requires_grad),
                    "optimizer_references": observed_counts[id(parameter)],
                }
                for name, parameter in named_parameters
            ],
        }
    )


def verify_initial_model_load_audit(
    model: object,
    *,
    source_checkpoint_path: Path,
    initial_model_audit: Mapping[str, object],
    model_seed: int,
) -> str:
    """Bind the instantiated model to the bytes and backbone reported by its loader."""
    import torch

    source_path = Path(source_checkpoint_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source checkpoint is missing: {source_path}")
    required = {
        "source_commit",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "checkpoint_tensor_count",
        "checkpoint_element_count_without_head",
        "model_parameter_count",
        "backbone_parameter_count",
        "missing_keys",
        "unexpected_keys",
        "head_reinitialized",
        "head_initialization_seed",
        "architecture",
    }
    missing = sorted(required - set(initial_model_audit))
    if missing:
        raise ValueError(f"Initial model load audit is missing fields: {missing}")
    source_commit = str(initial_model_audit["source_commit"]).strip().lower()
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("Initial model audit source_commit must be a full Git commit")
    actual_checkpoint_hash = stable_file_sha256(source_path)
    if _validate_sha256(
        initial_model_audit["checkpoint_sha256"],
        field_name="initial_model_audit.checkpoint_sha256",
    ) != actual_checkpoint_hash:
        raise ValueError("Initial model audit checkpoint checksum does not match source bytes")
    if int(initial_model_audit["checkpoint_size_bytes"]) != source_path.stat().st_size:
        raise ValueError("Initial model audit checkpoint size does not match source bytes")
    if int(initial_model_audit["head_initialization_seed"]) != int(model_seed):
        raise ValueError("Initial model audit head seed does not match the frozen model seed")
    if initial_model_audit["head_reinitialized"] is not True:
        raise ValueError("Initial model audit must confirm classification-head reinitialization")
    if sorted(str(value) for value in initial_model_audit["missing_keys"]) != [  # type: ignore[arg-type]
        "head.bias",
        "head.weight",
    ]:
        raise ValueError("Initial model audit must identify only the reinitialized head as missing")
    if list(initial_model_audit["unexpected_keys"]):  # type: ignore[arg-type]
        raise ValueError("Initial model audit contains unexpected source-checkpoint keys")
    architecture = initial_model_audit["architecture"]
    if not isinstance(architecture, Mapping) or not architecture:
        raise ValueError("Initial model audit requires a non-empty architecture mapping")

    raw = torch.load(source_path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError("Source checkpoint must contain a tensor state dictionary")
    if "state_dict" in raw:
        raw = raw["state_dict"]
    if not isinstance(raw, dict) or not raw or not all(isinstance(key, str) for key in raw):
        raise ValueError("Source checkpoint has an invalid state dictionary")
    if not all(torch.is_tensor(value) for value in raw.values()):
        raise ValueError("Source checkpoint state contains non-tensor values")
    source_state = dict(raw)
    head_keys = {"head.weight", "head.bias"}
    if not head_keys.issubset(source_state):
        raise ValueError("Source checkpoint is missing the classification head")
    if int(initial_model_audit["checkpoint_tensor_count"]) != len(source_state):
        raise ValueError("Initial model audit tensor count does not match source checkpoint")
    source_backbone = {
        name: value.detach().cpu()
        for name, value in source_state.items()
        if name not in head_keys
    }
    source_elements = sum(value.numel() for value in source_backbone.values())
    if int(initial_model_audit["checkpoint_element_count_without_head"]) != source_elements:
        raise ValueError("Initial model audit backbone element count is incorrect")

    model_state = model.state_dict()  # type: ignore[attr-defined]
    model_backbone = {
        name: value.detach().cpu()
        for name, value in model_state.items()
        if not name.startswith("head.")
    }
    if set(model_backbone) != set(source_backbone):
        raise ValueError("Instantiated model backbone schema differs from source checkpoint")
    for name in sorted(source_backbone):
        if not torch.equal(model_backbone[name], source_backbone[name]):
            raise ValueError(f"Instantiated model backbone tensor was not loaded from source: {name}")
    model_parameters = sum(parameter.numel() for parameter in model.parameters())  # type: ignore[attr-defined]
    backbone_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()  # type: ignore[attr-defined]
        if not name.startswith("head.")
    )
    if int(initial_model_audit["model_parameter_count"]) != model_parameters:
        raise ValueError("Initial model audit model-parameter count is incorrect")
    if int(initial_model_audit["backbone_parameter_count"]) != backbone_parameters:
        raise ValueError("Initial model audit backbone-parameter count is incorrect")

    normalized_audit = dict(initial_model_audit)
    normalized_audit["source_commit"] = source_commit
    normalized_audit["checkpoint_sha256"] = actual_checkpoint_hash
    return canonical_json_sha256(
        {
            "schema_version": 1,
            "verified_load_audit": normalized_audit,
            "instantiated_model_state_sha256": _model_state_sha256(model),
            "instantiated_architecture_sha256": _model_architecture_sha256(model),
            "source_checkpoint_sha256": actual_checkpoint_hash,
            "model_seed": int(model_seed),
        }
    )


def _executable_files_sha256(paths: list[Path] | tuple[Path, ...]) -> str:
    if not paths:
        raise ValueError("At least one executable source path must be frozen")
    resolved = sorted((Path(item).resolve() for item in paths), key=lambda item: item.as_posix())
    common_root = Path(os.path.commonpath([str(path.parent) for path in resolved]))
    records: list[dict[str, str]] = []
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"Frozen executable source is missing: {path}")
        records.append(
            {
                "path": path.relative_to(common_root).as_posix(),
                "sha256": stable_file_sha256(path),
            }
        )
    return canonical_json_sha256(records)


def verify_executable_allowlist(
    *,
    executable_root: Path,
    executable_paths: list[Path] | tuple[Path, ...],
    frozen_allowlist: Mapping[str, object],
) -> str:
    """Verify the exact executable source set and each file's frozen digest."""
    root = Path(executable_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Executable root does not exist: {root}")
    expected: dict[str, str] = {}
    for raw_name, raw_digest in frozen_allowlist.items():
        name = Path(str(raw_name)).as_posix()
        if Path(name).is_absolute() or name.startswith("../") or "/../" in name:
            raise ValueError(f"Frozen executable allow-list contains an unsafe path: {raw_name}")
        expected[name] = _validate_sha256(
            raw_digest,
            field_name=f"executable checksum for {name}",
        )
    actual_paths: dict[str, Path] = {}
    for raw_path in executable_paths:
        path = Path(raw_path).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Executable path is outside the frozen root: {path}") from exc
        if relative in actual_paths:
            raise ValueError(f"Duplicate executable path: {relative}")
        actual_paths[relative] = path
    if set(actual_paths) != set(expected):
        raise ValueError(
            "Executable paths must exactly match the frozen/discovered allow-list"
        )
    records: list[dict[str, str]] = []
    for relative in sorted(expected):
        path = actual_paths[relative]
        if not path.is_file():
            raise FileNotFoundError(f"Frozen executable source is missing: {path}")
        actual_digest = stable_file_sha256(path)
        if actual_digest != expected[relative]:
            raise ValueError(f"Executable checksum mismatch: {relative}")
        records.append({"path": relative, "sha256": actual_digest})
    if not records:
        raise ValueError("Frozen executable allow-list cannot be empty")
    return canonical_json_sha256(records)


def _atomic_json_write(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(
        str(Path(path)),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _checkpoint_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256.json")


def _checkpoint_pointer(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".current.json")


_TRAINING_PROGRESS_CONTEXT_FIELDS = (
    "run_id",
    "stage",
    "job_id",
    "job_spec_sha256",
    "fold",
    "seed",
    "modality",
    "protocol",
)


def _validated_training_progress_context(
    supplied: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if supplied is None:
        return None
    context = dict(supplied)
    missing = [name for name in _TRAINING_PROGRESS_CONTEXT_FIELDS if name not in context]
    if missing:
        raise ValueError(f"Training progress context is missing fields: {missing}")
    for name in ("run_id", "stage", "job_id", "modality", "protocol"):
        if not str(context[name]).strip():
            raise ValueError(f"Training progress context {name!r} must be non-empty")
    context["job_spec_sha256"] = _validate_sha256(
        context["job_spec_sha256"], field_name="training progress job specification"
    )
    for name in ("fold", "seed"):
        value = int(context[name])
        if value <= 0:
            raise ValueError(f"Training progress context {name!r} must be positive")
        context[name] = value
    return {name: context[name] for name in _TRAINING_PROGRESS_CONTEXT_FIELDS}


def _write_training_progress(
    *,
    run_dir: Path,
    progress_context: Mapping[str, object],
    config: HSTTrainingConfig,
    checkpoint_payload: Mapping[str, object],
) -> Path:
    """Publish resume coordinates only after a transactional checkpoint commits."""

    run_dir = Path(run_dir).resolve()
    context = _validated_training_progress_context(progress_context)
    if context is None:
        raise ValueError("Training progress context is required")
    pointer_path = _checkpoint_pointer(run_dir / "last.pt")
    if not pointer_path.is_file() or pointer_path.is_symlink():
        raise FileNotFoundError("Transactional last-checkpoint pointer is missing")
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Transactional last-checkpoint pointer is corrupt") from exc
    if pointer.get("writer") not in _TRUSTED_CHECKPOINT_WRITERS:
        raise ValueError("Training progress cannot bind an untrusted checkpoint pointer")
    current = pointer.get("current")
    if not isinstance(current, Mapping):
        raise ValueError("Transactional checkpoint pointer has no current generation")
    checkpoint_path = (run_dir / str(current.get("checkpoint_path", ""))).resolve()
    sidecar_path = (run_dir / str(current.get("sidecar_path", ""))).resolve()
    for candidate in (checkpoint_path, sidecar_path):
        try:
            candidate.relative_to(run_dir)
        except ValueError as exc:
            raise ValueError("Training checkpoint progress escaped its run directory") from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise FileNotFoundError("Training checkpoint generation is incomplete")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Training checkpoint checksum sidecar is corrupt") from exc
    if sidecar.get("writer") not in _TRUSTED_CHECKPOINT_WRITERS:
        raise ValueError("Training checkpoint sidecar has an untrusted writer")
    checkpoint_sha256 = _validate_sha256(
        current.get("sha256"), field_name="training progress checkpoint checksum"
    )
    if (
        sidecar.get("sha256") != checkpoint_sha256
        or int(sidecar.get("size_bytes", -1)) != int(current.get("size_bytes", -2))
        or checkpoint_path.stat().st_size != int(current.get("size_bytes", -1))
    ):
        raise ValueError("Training progress checkpoint metadata is inconsistent")
    schedule = checkpoint_payload.get("epoch_batch_schedule")
    if not isinstance(schedule, Mapping):
        raise ValueError("Training checkpoint has no audited epoch batch schedule")
    record: dict[str, object] = {
        "schema_version": 1,
        "receipt_type": "hst_training_progress",
        "status": "checkpointed",
        **context,
        "completed_epoch": int(checkpoint_payload["completed_epoch"]),
        "resume_epoch": int(checkpoint_payload["resume_epoch"]),
        "next_consumed_batch_index": int(
            checkpoint_payload["next_consumed_batch_index"]
        ),
        "epoch_batch_count": int(schedule["batch_count"]),
        "completed_optimizer_boundaries": int(
            checkpoint_payload["epoch_update_boundaries"]
        ),
        "epoch_optimizer_boundary_count": int(schedule["optimizer_boundary_count"]),
        "max_epochs": int(config.max_epochs),
        "successful_optimizer_updates": int(
            checkpoint_payload["successful_optimizer_updates"]
        ),
        "skipped_optimizer_updates": int(
            checkpoint_payload["skipped_optimizer_updates"]
        ),
        "checkpoint_reason": str(checkpoint_payload["checkpoint_reason"]),
        "checkpoint_resume_safe": (
            checkpoint_payload.get("resume_semantics") == "optimizer_boundary_v1"
        ),
        "checkpoint_pointer_path": pointer_path.relative_to(run_dir).as_posix(),
        "checkpoint_pointer_sha256": stable_file_sha256(pointer_path),
        "checkpoint": dict(current),
        "updated_at_unix": time.time(),
    }
    record["record_hash"] = canonical_json_sha256(record)
    progress_path = run_dir / "training_progress.json"
    _atomic_json_write(record, progress_path)
    return progress_path


def _load_verified_checkpoint_pair(path: Path, sidecar_path: Path) -> dict[str, Any]:
    import torch

    if not path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"Checkpoint or checksum sidecar is missing: {path}")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid checkpoint checksum sidecar: {sidecar_path}") from exc
    expected = _validate_sha256(sidecar.get("sha256"), field_name="checkpoint checksum")
    actual = stable_file_sha256(path)
    if actual != expected or int(sidecar.get("size_bytes", -1)) != path.stat().st_size:
        raise ValueError(f"Checkpoint checksum does not match its sidecar: {path}")
    if sidecar.get("writer") not in _TRUSTED_CHECKPOINT_WRITERS:
        raise ValueError("Checkpoint sidecar was not emitted by the trusted training writer")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Restricted checkpoint load failed: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Verified checkpoint payload is not a mapping")
    return payload


def _progress_pinned_checkpoint_generation(
    path: Path,
    generation_root: Path,
) -> str | None:
    if path.name != "last.pt":
        return None
    progress_path = path.parent / "training_progress.json"
    if not progress_path.is_file() or progress_path.is_symlink():
        return None
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if not isinstance(progress, dict):
            return None
        record_hash = progress.pop("record_hash", None)
        if (
            progress.get("receipt_type") != "hst_training_progress"
            or progress.get("status") != "checkpointed"
            or not isinstance(record_hash, str)
            or len(record_hash) != 64
            or canonical_json_sha256(progress) != record_hash
        ):
            return None
        declared = progress.get("checkpoint")
        if not isinstance(declared, Mapping):
            return None
        generation = str(declared.get("generation", ""))
        candidate_root = (generation_root / generation).resolve()
        candidate_root.relative_to(generation_root.resolve())
        if candidate_root.parent != generation_root.resolve() or not candidate_root.is_dir():
            return None
        checkpoint_path = candidate_root / "checkpoint.pt"
        sidecar_path = _checkpoint_sidecar(checkpoint_path)
        expected_checkpoint = checkpoint_path.relative_to(path.parent).as_posix()
        expected_sidecar = sidecar_path.relative_to(path.parent).as_posix()
        if (
            str(declared.get("checkpoint_path", "")) != expected_checkpoint
            or str(declared.get("sidecar_path", "")) != expected_sidecar
        ):
            return None
        _load_verified_checkpoint_pair(checkpoint_path, sidecar_path)
        if (
            stable_file_sha256(checkpoint_path) != declared.get("sha256")
            or checkpoint_path.stat().st_size != int(declared.get("size_bytes", -1))
        ):
            return None
        return generation
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _atomic_torch_save(payload: object, path: Path) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generation_root = path.parent / f".{path.name}.generations"
    generation_root.mkdir(parents=True, exist_ok=True)
    generation_id = uuid.uuid4().hex
    staging = generation_root / f".{generation_id}.tmp"
    final_generation = generation_root / generation_id
    staging.mkdir()
    checkpoint_path = staging / "checkpoint.pt"
    sidecar_path = _checkpoint_sidecar(checkpoint_path)
    try:
        with checkpoint_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = stable_file_sha256(checkpoint_path)
        size_bytes = checkpoint_path.stat().st_size
        _atomic_json_write(
            {
                "schema_version": 2,
                "writer": _CHECKPOINT_WRITER,
                "filename": checkpoint_path.name,
                "size_bytes": size_bytes,
                "sha256": digest,
            },
            sidecar_path,
        )
        reloaded = _load_verified_checkpoint_pair(checkpoint_path, sidecar_path)
        if not isinstance(reloaded, dict):
            raise ValueError("Checkpoint candidate failed transactional reload verification")
        os.replace(staging, final_generation)
        _fsync_directory(generation_root)

        relative_checkpoint = (final_generation / "checkpoint.pt").relative_to(
            path.parent
        ).as_posix()
        relative_sidecar = _checkpoint_sidecar(
            final_generation / "checkpoint.pt"
        ).relative_to(path.parent).as_posix()
        current_record = {
            "generation": generation_id,
            "checkpoint_path": relative_checkpoint,
            "sidecar_path": relative_sidecar,
            "sha256": digest,
            "size_bytes": size_bytes,
        }
        previous_record: Mapping[str, object] | None = None
        pointer_path = _checkpoint_pointer(path)
        if pointer_path.is_file():
            try:
                previous_pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                if (
                    previous_pointer.get("writer")
                    not in _TRUSTED_CHECKPOINT_WRITERS
                ):
                    raise ValueError("Untrusted previous checkpoint pointer")
                parent = path.parent.resolve()
                for role in ("current", "previous"):
                    candidate = previous_pointer.get(role)
                    if not isinstance(candidate, Mapping):
                        continue
                    try:
                        candidate_checkpoint = (
                            path.parent / str(candidate["checkpoint_path"])
                        ).resolve()
                        candidate_sidecar = (
                            path.parent / str(candidate["sidecar_path"])
                        ).resolve()
                        candidate_checkpoint.relative_to(parent)
                        candidate_sidecar.relative_to(parent)
                        _load_verified_checkpoint_pair(
                            candidate_checkpoint,
                            candidate_sidecar,
                        )
                        if stable_file_sha256(candidate_checkpoint) != _validate_sha256(
                            candidate.get("sha256"),
                            field_name="previous checkpoint checksum",
                        ):
                            continue
                        if candidate_checkpoint.stat().st_size != int(
                            candidate.get("size_bytes", -1)
                        ):
                            continue
                        previous_record = dict(candidate)
                        break
                    except Exception:
                        continue
            except (OSError, json.JSONDecodeError, ValueError):
                previous_record = None
        pointer_payload: dict[str, object] = {
            "schema_version": 2,
            "writer": _CHECKPOINT_WRITER,
            "logical_name": path.name,
            "current": current_record,
            "previous": previous_record,
        }
        _atomic_json_write(pointer_payload, pointer_path)

        retained = {generation_id}
        if previous_record is not None:
            previous_id = str(previous_record.get("generation", ""))
            if previous_id:
                retained.add(previous_id)
        pinned_generation = _progress_pinned_checkpoint_generation(path, generation_root)
        if pinned_generation is not None:
            retained.add(pinned_generation)
        for candidate in generation_root.iterdir():
            if candidate.is_dir() and candidate.name not in retained:
                for child in candidate.iterdir():
                    child.unlink()
                candidate.rmdir()
    finally:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()


def _load_verified_checkpoint_with_path(path: Path) -> tuple[dict[str, Any], Path]:
    path = Path(path)
    pointer_path = _checkpoint_pointer(path)
    if not pointer_path.is_file():
        payload = _load_verified_checkpoint_pair(path, _checkpoint_sidecar(path))
        return payload, path
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid checkpoint generation pointer: {pointer_path}") from exc
    if pointer.get("writer") not in _TRUSTED_CHECKPOINT_WRITERS:
        raise ValueError("Checkpoint pointer was not emitted by the trusted training writer")
    failures: list[str] = []
    parent = path.parent.resolve()
    for role in ("current", "previous"):
        record = pointer.get(role)
        if not isinstance(record, Mapping):
            continue
        try:
            checkpoint = (path.parent / str(record["checkpoint_path"])).resolve()
            sidecar = (path.parent / str(record["sidecar_path"])).resolve()
            checkpoint.relative_to(parent)
            sidecar.relative_to(parent)
            payload = _load_verified_checkpoint_pair(checkpoint, sidecar)
            if stable_file_sha256(checkpoint) != _validate_sha256(
                record.get("sha256"), field_name=f"{role} checkpoint checksum"
            ):
                raise ValueError(f"{role} pointer checksum disagrees with generation")
            if checkpoint.stat().st_size != int(record.get("size_bytes", -1)):
                raise ValueError(f"{role} pointer size disagrees with generation")
            return payload, checkpoint
        except Exception as exc:
            failures.append(f"{role}: {exc}")
    raise ValueError(
        "No valid checkpoint generation remains for "
        f"{path}: {'; '.join(failures)}"
    )


def _load_verified_checkpoint(path: Path) -> dict[str, Any]:
    return _load_verified_checkpoint_with_path(path)[0]


def _load_or_recover_best_checkpoint(
    *,
    run_dir: Path,
    last_payload: Mapping[str, object],
    expected_fingerprint: str,
    prediction_context: Mapping[str, object],
    execution_identity: Mapping[str, object],
    completed_epoch: int,
    best_epoch: int,
) -> dict[str, Any]:
    def validate(payload: Mapping[str, object]) -> None:
        validate_resume_checkpoint_contract(
            payload,
            expected_fingerprint=expected_fingerprint,
        )
        if payload.get("checkpoint_role") != "best":
            raise ValueError("Best checkpoint has the wrong immutable role")
        if payload.get("prediction_context") != prediction_context:
            raise ValueError("Best checkpoint prediction context does not match the run")
        if payload.get("execution_identity") != execution_identity:
            raise ValueError("Best checkpoint execution identity does not match the run")
        if int(payload.get("epoch", -1)) != best_epoch or int(
            payload.get("best_epoch", -1)
        ) != best_epoch:
            raise ValueError("Best checkpoint epoch does not match the resume state")

    best_path = run_dir / "best.pt"
    try:
        best_payload = _load_verified_checkpoint(best_path)
        validate(best_payload)
        return best_payload
    except (FileNotFoundError, ValueError) as exc:
        # Epoch-end publication writes the resume checkpoint first. If the process
        # stops before publishing best.pt, that verified last payload is also the
        # selected best model only when the just-completed epoch became the best.
        if best_epoch != completed_epoch:
            raise ValueError(
                "Best checkpoint is unavailable and cannot be recovered from a "
                "later non-best resume checkpoint"
            ) from exc
        if (
            last_payload.get("checkpoint_role") != "last"
            or int(last_payload.get("epoch", -1)) != best_epoch
            or int(last_payload.get("best_epoch", -1)) != best_epoch
        ):
            raise ValueError(
                "Resume checkpoint cannot safely reconstruct the selected best epoch"
            ) from exc
        if (
            last_payload.get("checkpoint_reason") != "epoch_end"
            or int(last_payload.get("completed_epoch", -1)) != completed_epoch
            or int(last_payload.get("resume_epoch", -1)) != completed_epoch + 1
            or int(last_payload.get("next_consumed_batch_index", -1)) != 0
            or float(last_payload.get("epoch_loss_sum", math.nan)) != 0.0
            or int(last_payload.get("epoch_sample_count", -1)) != 0
            or int(last_payload.get("epoch_update_boundaries", -1)) != 0
        ):
            raise ValueError(
                "Best checkpoint recovery requires an immutable epoch-end state, "
                "not a partial next-epoch checkpoint"
            ) from exc
        recovered_payload = {**dict(last_payload), "checkpoint_role": "best"}
        _atomic_torch_save(recovered_payload, best_path)
        best_payload = _load_verified_checkpoint(best_path)
        validate(best_payload)
        return best_payload


def _load_latest_valid_checkpoint(paths: list[Path]) -> tuple[dict[str, Any], Path] | None:
    for path in paths:
        if not path.is_file() and not _checkpoint_pointer(path).is_file():
            continue
        try:
            payload, resolved_path = _load_verified_checkpoint_with_path(path)
            if not isinstance(payload, dict) or "model_state_dict" not in payload or "epoch" not in payload:
                continue
            return payload, resolved_path
        except Exception:
            continue
    return None


def _checkpoint_artifacts_exist(path: Path) -> bool:
    path = Path(path)
    generation_root = path.parent / f".{path.name}.generations"
    return any(
        candidate.exists()
        for candidate in (
            path,
            _checkpoint_sidecar(path),
            _checkpoint_pointer(path),
            generation_root,
        )
    )


def _resolve_resume_checkpoint(
    paths: list[Path],
    *,
    resume_requested: bool,
    confirmatory: bool,
) -> tuple[dict[str, Any], Path] | None:
    """Distinguish a genuinely fresh run from corrupt or incomplete resume state."""
    artifacts_exist = any(_checkpoint_artifacts_exist(path) for path in paths)
    if not resume_requested:
        if confirmatory and artifacts_exist:
            raise ValueError(
                "Confirmatory fresh start is forbidden while resume artifacts exist"
            )
        return None
    loaded = _load_latest_valid_checkpoint(paths)
    if loaded is None and confirmatory and artifacts_exist:
        raise ValueError(
            "Confirmatory resume artifacts exist but no valid checkpoint generation remains"
        )
    if loaded is not None and loaded[0].get("checkpoint_role") != "last":
        raise ValueError("Resume checkpoint_role must be exactly 'last'")
    return loaded


def _updates_per_epoch(loaders: Mapping[str, object], config: HSTTrainingConfig) -> int:
    if callable(loaders.get("train_factory")):
        cache_index = loaders.get("cache_index")
        manifest = loaders.get("manifest")
        if not isinstance(cache_index, pd.DataFrame) or not isinstance(manifest, pd.DataFrame):
            raise ValueError("Dynamic training loaders require cache_index and manifest")
        draw_plan = build_hierarchical_epoch_draw_plan(
            cache_index,
            manifest,
            fold=int(loaders.get("fold", 0)),
            modality=str(loaders.get("modality", "")),
            epoch=1,
            seed=int(loaders.get("seed", 0)),
            representation_id=(
                str(loaders["representation_id"])
                if loaders.get("representation_id") is not None
                else None
            ),
        )
        batch_count = math.ceil(len(draw_plan) / config.physical_batch_size)
    else:
        loader = loaders.get("train")
        if loader is None:
            raise ValueError("HST loaders require train_factory or train")
        batch_count = len(loader)  # type: ignore[arg-type]
    updates = math.ceil(batch_count / config.gradient_accumulation)
    if updates <= 0:
        raise ValueError("HST training requires at least one optimizer update per epoch")
    return updates


def _scientific_batch_identity_sha256(frame: pd.DataFrame) -> str:
    if frame.empty:
        raise ValueError("Training batch identity cannot be empty")
    scientific = frame.drop(
        columns=[column for column in ("cache_path",) if column in frame]
    )
    return _ordered_frame_sha256(scientific)


def _epoch_batch_schedule_audit(
    draw_plan: pd.DataFrame,
    *,
    epoch: int,
    physical_batch_size: int,
    gradient_accumulation: int,
) -> dict[str, object]:
    if draw_plan.empty or epoch <= 0:
        raise ValueError("Epoch batch schedule requires a non-empty positive epoch")
    if physical_batch_size <= 0 or gradient_accumulation <= 0:
        raise ValueError("Epoch batch schedule requires positive batch settings")
    batch_hashes = [
        _scientific_batch_identity_sha256(
            draw_plan.iloc[start : start + physical_batch_size]
        )
        for start in range(0, len(draw_plan), physical_batch_size)
    ]
    boundaries = [
        batch_index
        for batch_index in range(1, len(batch_hashes) + 1)
        if batch_index % gradient_accumulation == 0
        or batch_index == len(batch_hashes)
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "epoch": int(epoch),
        "draw_plan_sha256": _scientific_batch_identity_sha256(draw_plan),
        "sample_count": int(len(draw_plan)),
        "batch_count": int(len(batch_hashes)),
        "batch_identity_sha256": batch_hashes,
        "batch_sequence_sha256": canonical_json_sha256(batch_hashes),
        "optimizer_boundary_batch_indices": boundaries,
        "optimizer_boundary_count": int(len(boundaries)),
        "physical_batch_size": int(physical_batch_size),
        "gradient_accumulation": int(gradient_accumulation),
        "effective_batch_size": int(physical_batch_size * gradient_accumulation),
    }
    payload["schedule_sha256"] = canonical_json_sha256(payload)
    return payload


def _validate_epoch_batch_schedule_audit(
    value: object,
    *,
    config: HSTTrainingConfig,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Resume checkpoint is missing its epoch batch schedule")
    schedule = dict(value)
    claimed_hash = _validate_sha256(
        schedule.pop("schedule_sha256", None),
        field_name="epoch schedule_sha256",
    )
    if canonical_json_sha256(schedule) != claimed_hash:
        raise ValueError("Resume checkpoint epoch batch schedule checksum mismatch")
    schedule["schedule_sha256"] = claimed_hash
    expected_batch_settings = (
        config.physical_batch_size,
        config.gradient_accumulation,
        config.effective_batch_size,
    )
    try:
        observed_batch_settings = (
            int(schedule["physical_batch_size"]),
            int(schedule["gradient_accumulation"]),
            int(schedule["effective_batch_size"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Resume checkpoint epoch batch settings are invalid") from exc
    if observed_batch_settings != expected_batch_settings:
        raise ValueError(
            "Resume checkpoint physical batch size or gradient accumulation is incompatible"
        )
    batch_hashes = schedule.get("batch_identity_sha256")
    boundaries = schedule.get("optimizer_boundary_batch_indices")
    if not isinstance(batch_hashes, list) or not batch_hashes:
        raise ValueError("Resume checkpoint has no batch identity sequence")
    for digest in batch_hashes:
        _validate_sha256(digest, field_name="batch identity SHA-256")
    if canonical_json_sha256(batch_hashes) != schedule.get("batch_sequence_sha256"):
        raise ValueError("Resume checkpoint batch identity sequence checksum mismatch")
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("Resume checkpoint has no optimizer boundary sequence")
    try:
        normalized_boundaries = [int(value) for value in boundaries]
        batch_count = int(schedule["batch_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Resume checkpoint optimizer boundaries are invalid") from exc
    if (
        batch_count != len(batch_hashes)
        or normalized_boundaries != sorted(set(normalized_boundaries))
        or normalized_boundaries[0] <= 0
        or normalized_boundaries[-1] != batch_count
        or int(schedule.get("optimizer_boundary_count", -1))
        != len(normalized_boundaries)
    ):
        raise ValueError("Resume checkpoint optimizer boundaries are inconsistent")
    return schedule


def _completed_training_schedule(
    config: HSTTrainingConfig,
    epoch_schedules: list[dict[str, object]],
) -> tuple[dict[str, object], str]:
    schedule: dict[str, object] = {
        "schema_version": 1,
        "physical_batch_size": int(config.physical_batch_size),
        "gradient_accumulation": int(config.gradient_accumulation),
        "effective_batch_size": int(config.effective_batch_size),
        "epochs": [dict(epoch_schedule) for epoch_schedule in epoch_schedules],
    }
    return schedule, canonical_json_sha256(schedule)


def _verify_loaded_frames_match_artifacts(
    loaders: Mapping[str, object],
    *,
    manifest_path: Path,
    cache_index_path: Path,
) -> None:
    manifest = loaders.get("manifest")
    cache_index = loaders.get("cache_index")
    if not isinstance(manifest, pd.DataFrame) or not isinstance(cache_index, pd.DataFrame):
        raise ValueError("Training loaders must retain their manifest and cache-index frames")
    disk_manifest = _read_table(Path(manifest_path))
    disk_cache = _read_table(Path(cache_index_path))
    if _canonical_frame_sha256(manifest) != _canonical_frame_sha256(disk_manifest):
        raise ValueError("In-memory manifest differs from the checksum-pinned manifest artifact")
    if _canonical_frame_sha256(cache_index) != _canonical_frame_sha256(disk_cache):
        raise ValueError("In-memory cache index differs from the checksum-pinned cache artifact")


def _verify_loader_split_contracts(loaders: Mapping[str, object]) -> pd.DataFrame:
    manifest = loaders.get("manifest")
    cache_index = loaders.get("cache_index")
    if not isinstance(manifest, pd.DataFrame) or not isinstance(cache_index, pd.DataFrame):
        raise ValueError("HST loaders are missing frozen manifest/cache frames")
    fold = int(loaders.get("fold", -1))
    modality = str(loaders.get("modality", ""))
    representation = loaders.get("representation_id")
    aligned = validate_manifest_cache_contract(
        cache_index,
        manifest,
        fold=fold,
        modality=modality,
        representation_id=str(representation) if representation is not None else None,
    )
    for split in ("validation", "test", "external_test"):
        expected = aligned.loc[aligned["split"].eq(split)].reset_index(drop=True)
        loader = loaders.get(split)
        if expected.empty:
            if loader is not None:
                raise ValueError(f"Unexpected {split} loader for an empty frozen split")
            continue
        if loader is None:
            raise ValueError(f"Missing {split} loader required by the frozen manifest")
        actual = getattr(getattr(loader, "dataset", None), "frame", None)
        if not isinstance(actual, pd.DataFrame):
            raise ValueError(f"{split} loader does not expose its immutable dataset frame")
        if _canonical_frame_sha256(expected) != _canonical_frame_sha256(actual):
            raise ValueError(f"{split} loader rows differ from the frozen manifest/cache alignment")
    return aligned


def _scientific_manifest_selection_sha256(aligned: pd.DataFrame) -> str:
    scientific_columns = [
        "fold",
        "training_seed",
        "protocol",
        "split",
        "dataset",
        "participant_key",
        "recording_key",
        "label_binary",
        "modality",
        "source_audio_sha256",
        "preprocessing_hash",
        "representation_id",
        "tensor_sha256",
    ]
    if "submodality" in aligned:
        scientific_columns.append("submodality")
    projection: dict[str, pd.Series] = {}
    missing: list[str] = []
    for column in scientific_columns:
        candidates = [column, f"{column}_x"]
        selected = next((candidate for candidate in candidates if candidate in aligned), None)
        if selected is None:
            missing.append(column)
        else:
            projection[column] = aligned[selected]
    if missing:
        raise ValueError(f"Scientific manifest selection is missing columns: {missing}")
    return _unordered_frame_sha256(pd.DataFrame(projection))


def _atomic_csv_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, lineterminator="\n", float_format="%.17g")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


_HELD_OUT_COHORT_BASE_COLUMNS = (
    "dataset",
    "participant_key",
    "recording_key",
    "label_binary",
    "split",
    "fold",
    "modality",
)
_HELD_OUT_COHORT_OPTIONAL_COLUMNS = (
    "source_audio_sha256",
    "cache_path",
    "tensor_sha256",
    "preprocessing_hash",
    "representation_id",
    "submodality",
)


def _validate_scientific_training_claim(
    claim: Mapping[str, object],
) -> dict[str, object]:
    if int(claim.get("schema_version", -1)) != 1:
        raise ValueError("Scientific training claim has an unsupported schema")
    data_contract = str(claim.get("data_contracts_freeze_hash", "")).strip()
    if not data_contract:
        raise ValueError("Scientific training claim is missing its data contract")
    manifest_selection_sha256 = _validate_sha256(
        claim.get("manifest_selection_sha256"),
        field_name="manifest_selection_sha256",
    )
    schedule = claim.get("training_schedule")
    if not isinstance(schedule, Mapping):
        raise ValueError("Scientific training claim is missing its training schedule")
    normalized_schedule = dict(schedule)
    schedule_sha256 = _validate_sha256(
        claim.get("training_schedule_sha256"),
        field_name="training_schedule_sha256",
    )
    if canonical_json_sha256(normalized_schedule) != schedule_sha256:
        raise ValueError("Scientific training schedule checksum mismatch")
    if int(normalized_schedule.get("schema_version", -1)) != 1:
        raise ValueError("Scientific training schedule has an unsupported schema")
    try:
        physical_batch_size = int(normalized_schedule["physical_batch_size"])
        gradient_accumulation = int(normalized_schedule["gradient_accumulation"])
        effective_batch_size = int(normalized_schedule["effective_batch_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Scientific training schedule has invalid batch settings") from exc
    if (
        physical_batch_size <= 0
        or gradient_accumulation <= 0
        or physical_batch_size * gradient_accumulation != effective_batch_size
    ):
        raise ValueError("Scientific training schedule batch settings are inconsistent")
    epochs = normalized_schedule.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise ValueError("Scientific training schedule must contain completed epochs")
    normalized_epochs: list[dict[str, object]] = []
    previous_epoch = 0
    for raw_epoch in epochs:
        if not isinstance(raw_epoch, Mapping):
            raise ValueError("Scientific training schedule epoch is invalid")
        normalized_epoch = dict(raw_epoch)
        claimed_epoch_hash = _validate_sha256(
            normalized_epoch.pop("schedule_sha256", None),
            field_name="scientific epoch schedule_sha256",
        )
        if canonical_json_sha256(normalized_epoch) != claimed_epoch_hash:
            raise ValueError("Scientific epoch batch schedule checksum mismatch")
        try:
            epoch = int(normalized_epoch["epoch"])
            epoch_physical_batch = int(normalized_epoch["physical_batch_size"])
            epoch_accumulation = int(normalized_epoch["gradient_accumulation"])
            epoch_effective_batch = int(normalized_epoch["effective_batch_size"])
            sample_count = int(normalized_epoch["sample_count"])
            batch_count = int(normalized_epoch["batch_count"])
            boundary_count = int(normalized_epoch["optimizer_boundary_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Scientific training schedule epoch is invalid") from exc
        if epoch <= previous_epoch:
            raise ValueError("Scientific training schedule epochs must be strictly ordered")
        previous_epoch = epoch
        if (
            (epoch_physical_batch, epoch_accumulation, epoch_effective_batch)
            != (physical_batch_size, gradient_accumulation, effective_batch_size)
            or sample_count <= 0
            or batch_count <= 0
        ):
            raise ValueError("Scientific epoch batch settings are inconsistent")
        _validate_sha256(
            normalized_epoch.get("draw_plan_sha256"), field_name="draw_plan_sha256"
        )
        batch_hashes = normalized_epoch.get("batch_identity_sha256")
        if not isinstance(batch_hashes, list) or len(batch_hashes) != batch_count:
            raise ValueError("Scientific training schedule batch identities are invalid")
        for digest in batch_hashes:
            _validate_sha256(digest, field_name="batch identity SHA-256")
        if canonical_json_sha256(batch_hashes) != normalized_epoch.get(
            "batch_sequence_sha256"
        ):
            raise ValueError("Scientific training schedule batch sequence mismatch")
        boundaries_raw = normalized_epoch.get("optimizer_boundary_batch_indices")
        if not isinstance(boundaries_raw, list) or not boundaries_raw:
            raise ValueError("Scientific training schedule has no optimizer boundaries")
        try:
            boundaries = [int(value) for value in boundaries_raw]
        except (TypeError, ValueError) as exc:
            raise ValueError("Scientific training schedule optimizer boundaries are invalid") from exc
        if (
            boundaries != sorted(set(boundaries))
            or boundaries[0] <= 0
            or boundaries[-1] != batch_count
            or boundary_count != len(boundaries)
        ):
            raise ValueError("Scientific training schedule optimizer boundaries are invalid")
        normalized_epoch["epoch"] = epoch
        normalized_epoch["sample_count"] = sample_count
        normalized_epoch["batch_count"] = batch_count
        normalized_epoch["optimizer_boundary_count"] = boundary_count
        normalized_epoch["optimizer_boundary_batch_indices"] = boundaries
        normalized_epoch["schedule_sha256"] = claimed_epoch_hash
        normalized_epochs.append(normalized_epoch)
    normalized_schedule["epochs"] = normalized_epochs
    return {
        "schema_version": 1,
        "data_contracts_freeze_hash": data_contract,
        "manifest_selection_sha256": manifest_selection_sha256,
        "training_schedule": normalized_schedule,
        "training_schedule_sha256": schedule_sha256,
    }


def _verify_held_out_tensor_files(loader: object) -> None:
    frame = getattr(getattr(loader, "dataset", None), "frame", None)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Held-out loader does not expose tensor cache rows")
    required = {"cache_path", "tensor_sha256"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Held-out tensor verification is missing columns: {missing}")
    for cache_path, tensor_sha256 in frame.loc[
        :, ["cache_path", "tensor_sha256"]
    ].itertuples(index=False, name=None):
        _verify_selected_cache_file(cache_path, tensor_sha256)


def _normalized_cohort_projection(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Held-out cohort is missing identity columns: {missing}")
    result = frame.loc[:, list(columns)].copy()
    result["fold"] = pd.to_numeric(result["fold"], errors="raise").astype(int)
    for column in columns:
        if column != "fold":
            result[column] = result[column].astype(str)
    return result.sort_values(
        ["dataset", "recording_key"], kind="mergesort"
    ).reset_index(drop=True)


def _expected_loader_cohort(
    loader: object,
    *,
    split: str,
    fold: int,
    modality: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frame = getattr(getattr(loader, "dataset", None), "frame", None)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Held-out loader must expose its non-empty frozen cohort frame")
    columns = _HELD_OUT_COHORT_BASE_COLUMNS + tuple(
        column for column in _HELD_OUT_COHORT_OPTIONAL_COLUMNS if column in frame.columns
    )
    cohort = _normalized_cohort_projection(frame, columns)
    _validate_binary_labels(cohort, name=f"{split} frozen cohort")
    _validate_dataset_roles(cohort, name=f"{split} frozen cohort")
    if not cohort["split"].eq(split).all():
        raise ValueError(f"Frozen loader cohort contains rows outside {split}")
    if not cohort["fold"].eq(int(fold)).all():
        raise ValueError("Frozen loader cohort contains another fold")
    if not cohort["modality"].eq(modality).all():
        raise ValueError("Frozen loader cohort contains another modality")
    if cohort.duplicated(["dataset", "recording_key"]).any():
        raise ValueError("Frozen held-out cohort recording identities must be unique")
    return cohort, columns


def _validate_persisted_predictions(
    frame: pd.DataFrame,
    *,
    split: str,
    fold: int,
    modality: str,
    prediction_context: Mapping[str, str],
    expected_cohort: pd.DataFrame,
    cohort_columns: tuple[str, ...],
) -> None:
    required = {
        "dataset",
        "participant_key",
        "recording_key",
        "label_binary",
        "split",
        "fold",
        "modality",
        "run_id",
        "protocol",
        "model",
        "checkpoint_hash",
        "representation",
        "architecture_sha256",
        "executable_sha256",
        "probability",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"Persisted {split} predictions have an invalid schema: {missing}")
    _validate_binary_labels(frame, name=f"{split} predictions")
    if not frame["split"].astype(str).eq(split).all():
        raise ValueError(f"Persisted {split} predictions contain another split")
    if not pd.to_numeric(frame["fold"], errors="coerce").eq(int(fold)).all():
        raise ValueError(f"Persisted {split} predictions contain another fold")
    if not frame["modality"].astype(str).eq(modality).all():
        raise ValueError(f"Persisted {split} predictions contain another modality")
    if frame.duplicated(["dataset", "recording_key"]).any():
        raise ValueError(f"Persisted {split} recording predictions must be unique")
    for field_name, expected in prediction_context.items():
        if not frame[field_name].astype(str).eq(expected).all():
            raise ValueError(f"Persisted {split} predictions have inconsistent {field_name}")
    probability = pd.to_numeric(frame["probability"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError(f"Persisted {split} probabilities are invalid")
    observed_cohort = _normalized_cohort_projection(frame, cohort_columns)
    if _unordered_frame_sha256(observed_cohort) != _unordered_frame_sha256(expected_cohort):
        raise ValueError(
            f"Persisted {split} predictions do not exactly cover the frozen loader cohort"
        )


def _load_registry_record(path: Path, *, name: str) -> dict[str, object]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid project-level evaluation {name}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"Invalid project-level evaluation {name}")
    claimed_hash = record.get("record_hash")
    unsigned = {key: value for key, value in record.items() if key != "record_hash"}
    if claimed_hash != canonical_json_sha256(unsigned):
        raise ValueError(f"Project-level evaluation {name} record hash mismatch")
    return record


def _exclusive_read_only_json_write(payload: Mapping[str, object], path: Path) -> None:
    """Atomically publish a complete no-replace anchor using an O_EXCL temp inode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            temporary.unlink()
        if path.exists():
            os.chmod(path, 0o444)


def _evaluate_split_once(
    model: object,
    loader: object,
    *,
    split: str,
    fold: int,
    modality: str,
    prediction_context: Mapping[str, object],
    run_dir: Path,
    project_registry_root: Path,
    training_contract: str,
    best_checkpoint_sha256: str,
    scientific_training_claim: Mapping[str, object],
) -> pd.DataFrame:
    run_root = Path(run_dir).resolve()
    registry_root = Path(project_registry_root).resolve()
    try:
        registry_root.relative_to(run_root)
    except ValueError:
        pass
    else:
        raise ValueError("Project evaluation registry must be independent of the run directory")
    validated_context = validate_prediction_context(prediction_context)
    persistent_context = {
        key: value for key, value in validated_context.items() if key != "run_id"
    }
    verified_best_hash = _validate_sha256(
        best_checkpoint_sha256,
        field_name="best_checkpoint_sha256",
    )
    verified_training_contract = _validate_sha256(
        training_contract,
        field_name="training_contract_fingerprint",
    )
    verified_scientific_claim = _validate_scientific_training_claim(
        scientific_training_claim
    )
    if validated_context["checkpoint_hash"] != verified_best_hash:
        raise ValueError("Held-out prediction context does not identify the best checkpoint")
    expected_cohort, cohort_columns = _expected_loader_cohort(
        loader,
        split=split,
        fold=fold,
        modality=modality,
    )
    scientific_cohort_columns = tuple(
        column for column in cohort_columns if column != "cache_path"
    )
    scientific_cohort = expected_cohort.loc[:, list(scientific_cohort_columns)]
    expected_cohort_sha256 = _unordered_frame_sha256(scientific_cohort)
    learned_model_state_sha256 = _model_state_sha256(model)
    scientific_context = {
        key: value
        for key, value in persistent_context.items()
        if key != "checkpoint_hash"
    }
    submodalities = (
        sorted(set(scientific_cohort["submodality"].astype(str)))
        if "submodality" in scientific_cohort
        else []
    )
    slot_identity = {
        "schema_version": 1,
        "split": split,
        "fold": int(fold),
        "modality": modality,
        "submodalities": submodalities,
        "representation": validated_context["representation"],
        "protocol": validated_context["protocol"],
        "architecture_sha256": validated_context["architecture_sha256"],
        "executable_sha256": validated_context["executable_sha256"],
        "scientific_training_claim": verified_scientific_claim,
        "expected_cohort_sha256": expected_cohort_sha256,
        "cohort_columns": list(scientific_cohort_columns),
        "persistent_prediction_context": scientific_context,
    }
    slot_id = canonical_json_sha256(slot_identity)
    registry_root.mkdir(parents=True, exist_ok=True)
    entry_dir = registry_root / slot_id
    anchor_path = registry_root / "_slot_anchors" / f"{slot_id}.json"
    lock = _KernelFileLock(registry_root / f"{slot_id}.lock")
    lock.acquire()
    try:
        claim_path = entry_dir / "claim.json"
        predictions_path = entry_dir / "predictions.csv"
        receipt_path = entry_dir / "receipt.json"
        expected_writer = _EVALUATION_WRITER
        current_source_record = {
            "schema_version": 1,
            "slot_id": slot_id,
            "source_run_id": validated_context["run_id"],
            "source_training_contract_fingerprint": verified_training_contract,
            "source_checkpoint_sha256": verified_best_hash,
            "learned_model_state_sha256": learned_model_state_sha256,
        }
        current_source_record_hash = canonical_json_sha256(current_source_record)
        candidate_claim: dict[str, object] = {
            "schema_version": 2,
            "writer": _EVALUATION_WRITER,
            "status": "claimed",
            "slot_id": slot_id,
            "slot_identity": slot_identity,
            "source_run_id": validated_context["run_id"],
            "source_checkpoint_sha256": verified_best_hash,
            "learned_model_state_sha256": learned_model_state_sha256,
            "source_record": current_source_record,
            "source_record_hash": current_source_record_hash,
        }
        candidate_claim["record_hash"] = canonical_json_sha256(candidate_claim)
        candidate_anchor: dict[str, object] = {
            "schema_version": 1,
            "writer": _SLOT_ANCHOR_WRITER,
            "status": "anchored",
            "slot_id": slot_id,
            "slot_identity": slot_identity,
            "initial_claim_record_hash": candidate_claim["record_hash"],
            "source_record_hash": current_source_record_hash,
            "learned_model_state_sha256": learned_model_state_sha256,
        }
        candidate_anchor["record_hash"] = canonical_json_sha256(candidate_anchor)

        if anchor_path.exists():
            anchor = _load_registry_record(anchor_path, name="slot anchor")
        else:
            if entry_dir.exists():
                raise ValueError("Project evaluation entry exists without its trusted slot anchor")
            try:
                _exclusive_read_only_json_write(candidate_anchor, anchor_path)
                anchor = candidate_anchor
            except FileExistsError:
                anchor = _load_registry_record(anchor_path, name="slot anchor")
        if (
            anchor.get("schema_version") != 1
            or anchor.get("writer") not in _TRUSTED_SLOT_ANCHOR_WRITERS
            or anchor.get("status") != "anchored"
            or anchor.get("slot_id") != slot_id
            or anchor.get("slot_identity") != slot_identity
        ):
            raise ValueError("Project evaluation slot anchor is invalid")
        if anchor.get("learned_model_state_sha256") != learned_model_state_sha256:
            raise ValueError(
                "Project evaluation slot is already bound to a different learned model state"
            )

        if not entry_dir.exists():
            if anchor.get("source_record_hash") != current_source_record_hash:
                raise ValueError("Interrupted slot recovery does not match the anchored source")
            if anchor.get("initial_claim_record_hash") != candidate_claim["record_hash"]:
                raise ValueError("Interrupted slot recovery does not match the anchored claim")
            entry_dir.mkdir()
            _atomic_json_write(candidate_claim, claim_path)
        elif not claim_path.exists():
            unexpected_entries = [
                path for path in entry_dir.iterdir() if path.name != claim_path.name
            ]
            if unexpected_entries:
                raise ValueError("Unclaimed project evaluation slot contains unexpected artifacts")
            if (
                anchor.get("source_record_hash") != current_source_record_hash
                or anchor.get("initial_claim_record_hash")
                != candidate_claim["record_hash"]
            ):
                raise ValueError("Interrupted slot claim recovery does not match its anchor")
            _atomic_json_write(candidate_claim, claim_path)
        if not claim_path.is_file():
            raise ValueError("Project evaluation slot is missing its anchored claim")
        claim = _load_registry_record(claim_path, name="claim")
        if (
            claim.get("schema_version") != 2
            or claim.get("writer") not in _TRUSTED_EVALUATION_WRITERS
            or claim.get("status") != "claimed"
            or claim.get("slot_id") != slot_id
            or claim.get("slot_identity") != slot_identity
            or claim.get("record_hash") != anchor.get("initial_claim_record_hash")
            or claim.get("source_record_hash") != anchor.get("source_record_hash")
            or claim.get("learned_model_state_sha256")
            != anchor.get("learned_model_state_sha256")
        ):
            raise ValueError("Project evaluation claim disagrees with its trusted slot anchor")
        source_record = claim.get("source_record")
        if (
            not isinstance(source_record, dict)
            or canonical_json_sha256(source_record) != claim.get("source_record_hash")
            or source_record.get("slot_id") != slot_id
            or source_record.get("source_run_id") != claim.get("source_run_id")
            or source_record.get("learned_model_state_sha256")
            != learned_model_state_sha256
        ):
            raise ValueError("Project evaluation anchored source record is inconsistent")

        predictions_exist = predictions_path.is_file()
        receipt_exists = receipt_path.is_file()
        if receipt_exists and not predictions_exist:
            raise ValueError("Complete evaluation receipt exists without durable predictions")
        if receipt_exists:
            receipt = _load_registry_record(receipt_path, name="receipt")
            if (
                receipt.get("schema_version") != 4
                or receipt.get("writer") not in _TRUSTED_EVALUATION_WRITERS
                or receipt.get("status") != "complete"
                or receipt.get("slot_id") != slot_id
                or receipt.get("slot_identity") != slot_identity
                or receipt.get("claim_record_hash") != claim.get("record_hash")
                or receipt.get("source_record_hash") != claim.get("source_record_hash")
                or receipt.get("source_run_id") != claim.get("source_run_id")
                or receipt.get("learned_model_state_sha256")
                != learned_model_state_sha256
            ):
                raise ValueError("Project evaluation receipt disagrees with its slot anchor")
            expected = _validate_sha256(
                receipt.get("predictions_sha256"),
                field_name=f"{split} predictions_sha256",
            )
            if stable_file_sha256(predictions_path) != expected:
                raise ValueError(f"Durable {split} predictions checksum mismatch")
            persisted = pd.read_csv(predictions_path, low_memory=False)
            if int(receipt.get("n_rows", -1)) != len(persisted):
                raise ValueError(f"Durable {split} prediction count disagrees with its receipt")
            _validate_persisted_predictions(
                persisted,
                split=split,
                fold=fold,
                modality=modality,
                prediction_context={
                    **scientific_context,
                    "checkpoint_hash": str(claim.get("source_checkpoint_sha256", "")),
                    "run_id": str(claim.get("source_run_id", "")),
                },
                expected_cohort=scientific_cohort,
                cohort_columns=scientific_cohort_columns,
            )
            return persisted

        if current_source_record_hash != anchor.get("source_record_hash"):
            raise ValueError("Incomplete evaluation recovery requires the exact anchored source")
        _verify_held_out_tensor_files(loader)
        if predictions_exist:
            predictions = pd.read_csv(predictions_path, low_memory=False)
        else:
            predictions = predict_hst_split(
                model,
                loader,
                split=split,
                fold=fold,
                modality=modality,
                prediction_context=prediction_context,
            )
            if predictions.empty:
                raise ValueError(f"Held-out {split} evaluation produced no predictions")
        _validate_persisted_predictions(
            predictions,
            split=split,
            fold=fold,
            modality=modality,
            prediction_context=validated_context,
            expected_cohort=scientific_cohort,
            cohort_columns=scientific_cohort_columns,
        )
        if not predictions_exist:
            _atomic_csv_write(predictions, predictions_path)
        receipt = {
            "schema_version": 4,
            "writer": expected_writer,
            "status": "complete",
            "slot_id": slot_id,
            "slot_identity": slot_identity,
            "source_run_id": claim["source_run_id"],
            "claim_record_hash": claim["record_hash"],
            "source_record_hash": claim["source_record_hash"],
            "learned_model_state_sha256": learned_model_state_sha256,
            "source_training_contract_fingerprint": source_record[
                "source_training_contract_fingerprint"
            ],
            "source_checkpoint_sha256": source_record["source_checkpoint_sha256"],
            "training_schedule": verified_scientific_claim["training_schedule"],
            "training_schedule_sha256": verified_scientific_claim[
                "training_schedule_sha256"
            ],
            "predictions_path": predictions_path.name,
            "predictions_sha256": stable_file_sha256(predictions_path),
            "n_rows": int(len(predictions)),
        }
        receipt["record_hash"] = canonical_json_sha256(receipt)
        _atomic_json_write(receipt, receipt_path)
        return predictions
    finally:
        lock.release()


class _OptimizerBoundaryStopRequest:
    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None

    def request(self, signal_number: int, _frame: object) -> None:
        self.requested = True
        self.signal_number = int(signal_number)


@contextmanager
def _optimizer_boundary_signal_guard(*, required: bool):
    """Convert termination signals into a checkpoint request outside the handler."""
    controller = _OptimizerBoundaryStopRequest()
    if threading.current_thread() is not threading.main_thread():
        if required:
            raise RuntimeError(
                "Confirmatory resumable training must run on the main thread for signal-safe checkpoints"
            )
        yield controller
        return
    previous: dict[int, object] = {}
    supported = [signal.SIGINT]
    if hasattr(signal, "SIGTERM"):
        supported.append(signal.SIGTERM)
    try:
        for signal_number in supported:
            previous[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, controller.request)
        yield controller
    finally:
        for signal_number, handler in previous.items():
            signal.signal(signal_number, handler)


def train_hst_fold(
    model: object,
    loaders: dict[str, object],
    config: HSTTrainingConfig,
    run_dir: Path,
    *,
    prediction_context: Mapping[str, object],
    manifest_path: Path,
    cache_index_path: Path,
    source_checkpoint_path: Path,
    executable_root: Path,
    executable_paths: list[Path] | tuple[Path, ...],
    frozen_executable_allowlist: Mapping[str, object],
    manifest_sha256: str,
    cache_index_sha256: str,
    source_checkpoint_sha256: str,
    initial_model_state_sha256: str,
    initial_model_audit: Mapping[str, object],
    expected_initial_model_binding_sha256: str,
    progress_context: Mapping[str, object] | None = None,
    evaluation_registry_root: Path | None = None,
    resource_pilot_receipt_path: Path | None = None,
    resume: bool = True,
    stop_after_epoch: int | None = None,
    stop_after_optimizer_updates: int | None = None,
    evaluate_test: bool = False,
    evaluate_external: bool = False,
    monotonic_clock: Callable[[], float] | None = None,
) -> HSTFoldResult:
    import torch

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if config.confirmatory and evaluation_registry_root is None:
        raise ValueError("Confirmatory training requires a project evaluation registry root")
    context = validate_prediction_context(prediction_context)
    training_progress_context = _validated_training_progress_context(progress_context)
    if training_progress_context is not None and (
        str(training_progress_context["run_id"]) != str(context["run_id"])
        or int(training_progress_context["fold"]) != int(loaders.get("fold", -1))
        or int(training_progress_context["seed"]) != int(config.random_seed)
        or str(training_progress_context["modality"])
        != str(loaders.get("modality", ""))
        or str(training_progress_context["protocol"]) != str(context["protocol"])
    ):
        raise ValueError("Training progress context differs from the frozen training identity")
    execution_identity = build_training_execution_identity(
        loaders,
        config,
        prediction_context=context,
    )
    if config.confirmatory:
        if resource_pilot_receipt_path is None:
            raise ValueError("Confirmatory training requires the frozen resource pilot receipt")
        verify_resource_pilot_receipt(resource_pilot_receipt_path, config)
    if stop_after_epoch is not None:
        if stop_after_epoch <= 0 or stop_after_epoch > config.max_epochs:
            raise ValueError("stop_after_epoch must be within the configured epoch budget")
        if config.confirmatory:
            raise ValueError("Confirmatory execution cannot use the test-only stop_after_epoch hook")
    if stop_after_optimizer_updates is not None:
        if stop_after_optimizer_updates <= 0:
            raise ValueError("stop_after_optimizer_updates must be positive")
        if config.confirmatory:
            raise ValueError(
                "Confirmatory execution cannot use the test-only optimizer-stop hook"
            )
    actual_hashes = verify_training_artifact_hashes(
        manifest_path=manifest_path,
        cache_index_path=cache_index_path,
        source_checkpoint_path=source_checkpoint_path,
        expected_manifest_sha256=manifest_sha256,
        expected_cache_index_sha256=cache_index_sha256,
        expected_source_checkpoint_sha256=source_checkpoint_sha256,
    )
    _verify_loaded_frames_match_artifacts(
        loaders,
        manifest_path=manifest_path,
        cache_index_path=cache_index_path,
    )
    aligned_manifest_cache = _verify_loader_split_contracts(loaders)
    if not callable(loaders.get("train_factory")):
        raise ValueError("HST training requires the frozen epoch-specific train_factory")
    if context["checkpoint_hash"] != actual_hashes["source_checkpoint"]:
        raise ValueError("Prediction context checkpoint does not match the source checkpoint bytes")
    loader_representation = loaders.get("representation_id")
    if loader_representation is not None and str(loader_representation) != context["representation"]:
        raise ValueError("Prediction context representation does not match the loaded cache")
    actual_executable_hash = verify_executable_allowlist(
        executable_root=executable_root,
        executable_paths=executable_paths,
        frozen_allowlist=frozen_executable_allowlist,
    )
    if context["executable_sha256"] != actual_executable_hash:
        raise ValueError("Prediction context executable hash does not match the training sources")
    actual_architecture_hash = _model_architecture_sha256(model)
    if context["architecture_sha256"] != actual_architecture_hash:
        raise ValueError("Prediction context architecture hash does not match the instantiated model")
    expected_initial_state = _validate_sha256(
        initial_model_state_sha256,
        field_name="initial_model_state_sha256",
    )
    actual_initial_state = _model_state_sha256(model)
    if actual_initial_state != expected_initial_state:
        raise ValueError("Instantiated model state does not match the frozen initial model state")
    actual_initial_binding = verify_initial_model_load_audit(
        model,
        source_checkpoint_path=source_checkpoint_path,
        initial_model_audit=initial_model_audit,
        model_seed=config.random_seed,
    )
    expected_initial_binding = _validate_sha256(
        expected_initial_model_binding_sha256,
        field_name="expected_initial_model_binding_sha256",
    )
    if actual_initial_binding != expected_initial_binding:
        raise ValueError("Instantiated model load audit differs from the frozen binding")
    device = _model_device(model)
    device_type = str(getattr(device, "type", device))
    if config.amp and (device_type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("AMP is permitted only when the HST model is on an available CUDA device")
    _configure_determinism(
        config.random_seed,
        device,
        enabled=config.deterministic_algorithms,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),  # type: ignore[attr-defined]
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer_parameter_sha256 = (
        _verify_full_finetuning_optimizer(model, optimizer)
        if config.confirmatory
        else None
    )
    contract_fingerprint = training_contract_fingerprint(
        training_config=asdict(config),
        manifest_sha256=actual_hashes["manifest"],
        cache_index_sha256=actual_hashes["cache_index"],
        source_checkpoint_sha256=actual_hashes["source_checkpoint"],
        initial_model_state_sha256=actual_initial_state,
        initial_model_binding_sha256=actual_initial_binding,
        optimizer_parameter_sha256=optimizer_parameter_sha256,
        execution_identity=execution_identity,
        prediction_context=context,
    )
    updates_per_epoch = _updates_per_epoch(loaders, config)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=updates_per_epoch * config.max_epochs,
        pct_start=config.scheduler_pct_start,
        anneal_strategy=config.scheduler_anneal_strategy,
        div_factor=config.scheduler_div_factor,
        final_div_factor=config.scheduler_final_div_factor,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    start_epoch = 1
    resumed_from: int | None = None
    best_epoch = 0
    best_score = (-math.inf, -math.inf, -math.inf, -math.inf)
    history_rows: list[dict[str, object]] = []
    skipped_updates = 0
    successful_updates = 0
    resume_consumed_batches = 0
    resume_epoch_loss_sum = 0.0
    resume_epoch_sample_count = 0
    resume_update_boundaries = 0
    resume_draw_plan_hash: str | None = None
    resume_epoch_schedule: dict[str, object] | None = None
    completed_epoch_schedules: list[dict[str, object]] = []
    clock = monotonic_clock or time.monotonic
    wall_checkpoint_at = float(clock())
    if not math.isfinite(wall_checkpoint_at):
        raise ValueError("Wall-clock checkpoint clock returned a non-finite value")

    loaded = _resolve_resume_checkpoint(
        [run_dir / "last.pt"],
        resume_requested=resume,
        confirmatory=config.confirmatory,
    )
    if loaded is not None:
        payload, _ = loaded
        resume_epoch_schedule = _validate_epoch_batch_schedule_audit(
            payload.get("epoch_batch_schedule"),
            config=config,
        )
        validate_resume_checkpoint_contract(
            payload,
            expected_fingerprint=contract_fingerprint,
        )
        if payload.get("resume_semantics") != "optimizer_boundary_v1":
            raise ValueError("Resume checkpoint does not declare optimizer-boundary semantics")
        if payload.get("prediction_context") != context:
            raise ValueError("Resume checkpoint prediction context does not match the current run")
        if payload.get("execution_identity") != execution_identity:
            raise ValueError("Resume checkpoint execution identity does not match the current run")
        if payload.get("initial_model_binding_sha256") != actual_initial_binding:
            raise ValueError("Resume checkpoint initial-model binding does not match the run")
        model.load_state_dict(payload["model_state_dict"])  # type: ignore[attr-defined]
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        scaler.load_state_dict(payload["scaler_state_dict"])
        _restore_rng_state(payload["rng_state"])
        completed_epoch = int(payload.get("completed_epoch", -1))
        start_epoch = int(payload.get("resume_epoch", -1))
        resumed_from = completed_epoch
        if completed_epoch < 0 or start_epoch < 1 or start_epoch > config.max_epochs + 1:
            raise ValueError("Resume checkpoint has invalid epoch coordinates")
        best_epoch = int(payload.get("best_epoch", 0))
        best_score = tuple(payload.get("best_score", best_score))  # type: ignore[assignment]
        history_rows = list(payload.get("history_rows", []))
        raw_completed_schedules = payload.get("completed_epoch_schedules", [])
        if not isinstance(raw_completed_schedules, list):
            raise ValueError("Resume checkpoint completed schedule history is invalid")
        completed_epoch_schedules = []
        for raw_schedule in raw_completed_schedules:
            completed_epoch_schedules.append(
                _validate_epoch_batch_schedule_audit(raw_schedule, config=config)
            )
        skipped_updates = int(payload.get("skipped_optimizer_updates", 0))
        successful_updates = int(payload.get("successful_optimizer_updates", -1))
        if successful_updates < 0:
            raise ValueError("Resume checkpoint is missing successful optimizer-update count")
        scheduler_steps = int(scheduler.state_dict().get("last_epoch", -1))
        if scheduler_steps != successful_updates:
            raise ValueError("Resume scheduler position disagrees with successful optimizer updates")
        resume_consumed_batches = int(payload.get("next_consumed_batch_index", -1))
        if resume_consumed_batches < 0:
            raise ValueError("Resume checkpoint has an invalid consumed-batch position")
        resume_epoch_loss_sum = float(payload.get("epoch_loss_sum", math.nan))
        resume_epoch_sample_count = int(payload.get("epoch_sample_count", -1))
        if (
            not math.isfinite(resume_epoch_loss_sum)
            or resume_epoch_loss_sum < 0
            or resume_epoch_sample_count < 0
        ):
            raise ValueError("Resume checkpoint has invalid weighted-loss accumulators")
        resume_update_boundaries = int(payload.get("epoch_update_boundaries", 0))
        resume_draw_plan_hash = str(payload.get("epoch_draw_plan_hash", "")) or None
        if resume_consumed_batches == 0 and (
            resume_epoch_loss_sum != 0.0
            or resume_epoch_sample_count != 0
            or resume_update_boundaries != 0
        ):
            raise ValueError("Completed-epoch resume contains partial-epoch accumulators")
        if resume_consumed_batches > 0 and resume_epoch_sample_count <= 0:
            raise ValueError("Partial-epoch resume is missing weighted-loss samples")
        if best_epoch > 0:
            _load_or_recover_best_checkpoint(
                run_dir=run_dir,
                last_payload=payload,
                expected_fingerprint=contract_fingerprint,
                prediction_context=context,
                execution_identity=execution_identity,
                completed_epoch=completed_epoch,
                best_epoch=best_epoch,
            )

    last_epoch = start_epoch - 1
    with _optimizer_boundary_signal_guard(required=config.confirmatory) as stop_request:
        for epoch in range(start_epoch, config.max_epochs + 1):
            expected_draw_plan = build_hierarchical_epoch_draw_plan(
                loaders["cache_index"],  # type: ignore[arg-type]
                loaders["manifest"],  # type: ignore[arg-type]
                fold=int(execution_identity["fold"]),
                modality=str(execution_identity["modality"]),
                epoch=epoch,
                seed=int(execution_identity["sampler_seed"]),
                representation_id=str(execution_identity["representation_id"]),
            )
            loader = loaders["train_factory"](epoch)  # type: ignore[operator]
            if int(getattr(loader, "num_workers", 0)) != 0:
                raise ValueError(
                    "Optimizer-boundary resume equivalence requires a zero-worker training loader; "
                    "worker state is not restorable"
                )
            dataset = getattr(loader, "dataset", None)
            if (
                int(getattr(dataset, "seed", -1)) != int(execution_identity["sampler_seed"])
                or int(getattr(dataset, "fold", -1)) != int(execution_identity["fold"])
                or int(getattr(dataset, "epoch", -1)) != epoch
            ):
                raise ValueError("Training loader sampler identity differs from the frozen run")
            actual_batch_count = len(loader)  # type: ignore[arg-type]
            actual_update_budget = math.ceil(actual_batch_count / config.gradient_accumulation)
            if actual_update_budget != updates_per_epoch:
                raise ValueError(
                    "Actual training loader update count differs from the frozen OneCycleLR budget"
                )
            dataset_frame = getattr(dataset, "frame", None)
            if not isinstance(dataset_frame, pd.DataFrame) or dataset_frame.empty:
                raise ValueError("Training loader does not expose its frozen draw plan")
            if _ordered_frame_sha256(dataset_frame) != _ordered_frame_sha256(expected_draw_plan):
                raise ValueError("Training loader draw plan differs from the frozen epoch plan")
            draw_hash = _ordered_frame_sha256(dataset_frame)
            epoch_batch_schedule = _epoch_batch_schedule_audit(
                dataset_frame,
                epoch=epoch,
                physical_batch_size=config.physical_batch_size,
                gradient_accumulation=config.gradient_accumulation,
            )
            if int(epoch_batch_schedule["batch_count"]) != actual_batch_count:
                raise ValueError("Training loader batch count differs from its audited schedule")
            consumed_before = resume_consumed_batches if epoch == start_epoch else 0
            if consumed_before > actual_batch_count:
                raise ValueError("Resume checkpoint consumed beyond the reconstructed epoch loader")
            audited_boundaries = list(
                epoch_batch_schedule["optimizer_boundary_batch_indices"]  # type: ignore[arg-type]
            )
            if consumed_before and consumed_before not in audited_boundaries:
                raise ValueError("Resume checkpoint is not at an optimizer boundary")
            if consumed_before and resume_draw_plan_hash != draw_hash:
                raise ValueError("Resume draw-plan sequence differs from the checkpoint")
            if consumed_before and resume_epoch_schedule != epoch_batch_schedule:
                raise ValueError(
                    "Resume sampler/batch identity sequence or optimizer boundaries are incompatible"
                )
            expected_completed_boundaries = sum(
                boundary <= consumed_before for boundary in audited_boundaries
            )
            if consumed_before and resume_update_boundaries != expected_completed_boundaries:
                raise ValueError("Resume optimizer-boundary count is inconsistent")
            model.train()  # type: ignore[attr-defined]
            optimizer.zero_grad(set_to_none=True)
            accumulated_samples = 0
            epoch_loss_sum = resume_epoch_loss_sum if epoch == start_epoch else 0.0
            epoch_sample_count = resume_epoch_sample_count if epoch == start_epoch else 0
            update_boundaries = resume_update_boundaries if epoch == start_epoch else 0

            def checkpoint_payload(
                *,
                completed_epoch: int,
                resume_at_epoch: int,
                consumed_batches: int,
                epoch_loss_sum: float,
                epoch_sample_count: int,
                epoch_boundaries: int,
                checkpoint_reason: str,
            ) -> dict[str, object]:
                completed_schedule, completed_schedule_sha256 = (
                    _completed_training_schedule(config, completed_epoch_schedules)
                )
                return {
                    "model_state_dict": model.state_dict(),  # type: ignore[attr-defined]
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "epoch": completed_epoch,
                    "completed_epoch": completed_epoch,
                    "resume_epoch": resume_at_epoch,
                    "next_consumed_batch_index": consumed_batches,
                    "epoch_loss_sum": float(epoch_loss_sum),
                    "epoch_sample_count": int(epoch_sample_count),
                    "epoch_update_boundaries": epoch_boundaries,
                    "epoch_draw_plan_hash": draw_hash,
                    "physical_batch_size": config.physical_batch_size,
                    "gradient_accumulation": config.gradient_accumulation,
                    "effective_batch_size": config.effective_batch_size,
                    "epoch_batch_schedule": epoch_batch_schedule,
                    "epoch_batch_sequence_sha256": epoch_batch_schedule[
                        "batch_sequence_sha256"
                    ],
                    "epoch_optimizer_boundary_batch_indices": epoch_batch_schedule[
                        "optimizer_boundary_batch_indices"
                    ],
                    "completed_epoch_schedules": [
                        dict(schedule) for schedule in completed_epoch_schedules
                    ],
                    "completed_training_schedule": completed_schedule,
                    "completed_training_schedule_sha256": completed_schedule_sha256,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "history_rows": list(history_rows),
                    "skipped_optimizer_updates": skipped_updates,
                    "successful_optimizer_updates": successful_updates,
                    "rng_state": _rng_state(),
                    "training_config": asdict(config),
                    "training_contract_fingerprint": contract_fingerprint,
                    "prediction_context": context,
                    "execution_identity": execution_identity,
                    "resume_semantics": "optimizer_boundary_v1",
                    "checkpoint_reason": checkpoint_reason,
                    "initial_model_state_sha256": actual_initial_state,
                    "initial_model_binding_sha256": actual_initial_binding,
                    "architecture_sha256": actual_architecture_hash,
                    "executable_sha256": actual_executable_hash,
                }

            def persist_last_checkpoint(payload: Mapping[str, object]) -> None:
                _atomic_torch_save(
                    {**dict(payload), "checkpoint_role": "last"},
                    run_dir / "last.pt",
                )
                if training_progress_context is not None:
                    _write_training_progress(
                        run_dir=run_dir,
                        progress_context=training_progress_context,
                        config=config,
                        checkpoint_payload=payload,
                    )

            interrupted = False
            expected_batch_hashes = list(
                epoch_batch_schedule["batch_identity_sha256"]  # type: ignore[arg-type]
            )
            for batch_index, (images, labels, batch_rows) in enumerate(  # type: ignore[union-attr]
                loader,
                start=1,
            ):
                if batch_index > len(expected_batch_hashes):
                    raise ValueError("Training loader yielded more batches than prespecified")
                if not isinstance(batch_rows, list) or len(batch_rows) != int(labels.shape[0]):
                    raise ValueError("Training loader yielded invalid batch metadata")
                yielded_batch_hash = _scientific_batch_identity_sha256(
                    pd.DataFrame(batch_rows)
                )
                if yielded_batch_hash != expected_batch_hashes[batch_index - 1]:
                    raise ValueError(
                        "Training yielded batch identity/order differs from the prespecified schedule"
                    )
                if batch_index <= consumed_before:
                    continue
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                batch_samples = int(labels.shape[0])
                with torch.amp.autocast(device_type=device_type, enabled=config.amp):
                    logits = model(images)  # type: ignore[operator]
                    if logits.ndim != 2 or logits.shape[1] != 2:
                        raise ValueError("HST training logits violate frozen two-class mapping")
                    mean_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
                    summed_loss = mean_loss * batch_samples
                if not bool(torch.isfinite(mean_loss)) and not config.amp:
                    raise FloatingPointError("Non-finite HST loss")
                scaler.scale(summed_loss).backward()
                accumulated_samples += batch_samples
                epoch_loss_sum += float(summed_loss.detach().cpu())
                epoch_sample_count += batch_samples
                boundary = (
                    batch_index % config.gradient_accumulation == 0
                    or batch_index == actual_batch_count
                )
                if not boundary:
                    continue
                if (
                    update_boundaries >= len(audited_boundaries)
                    or batch_index != audited_boundaries[update_boundaries]
                ):
                    raise RuntimeError(
                        "Observed optimizer boundary differs from the audited batch schedule"
                    )
                update_boundaries += 1
                scaler.unscale_(optimizer)
                finite = True
                for parameter in model.parameters():  # type: ignore[attr-defined]
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_samples)
                        finite = finite and bool(torch.isfinite(parameter.grad).all())
                if not finite and not config.amp:
                    raise FloatingPointError("Non-finite HST gradient")
                if finite:
                    torch.nn.utils.clip_grad_norm_(  # type: ignore[attr-defined]
                        model.parameters(), config.gradient_clip_norm
                    )
                old_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                new_scale = float(scaler.get_scale())
                update_skipped = config.amp and (not finite or new_scale < old_scale)
                if update_skipped:
                    skipped_updates += 1
                    if skipped_updates > config.amp_max_skipped_updates:
                        raise FloatingPointError(
                            "AMP skipped more optimizer updates than the frozen pilot permits"
                        )
                else:
                    scheduler.step()
                    successful_updates += 1
                optimizer.zero_grad(set_to_none=True)
                accumulated_samples = 0
                current_wall_time = float(clock())
                if not math.isfinite(current_wall_time) or current_wall_time < wall_checkpoint_at:
                    raise ValueError("Wall-clock checkpoint clock must be finite and monotonic")
                if (
                    current_wall_time - wall_checkpoint_at
                    >= config.wall_clock_checkpoint_interval_seconds
                ):
                    periodic_payload = checkpoint_payload(
                        completed_epoch=epoch - 1,
                        resume_at_epoch=epoch,
                        consumed_batches=batch_index,
                        epoch_loss_sum=epoch_loss_sum,
                        epoch_sample_count=epoch_sample_count,
                        epoch_boundaries=update_boundaries,
                        checkpoint_reason="wall_clock_interval",
                    )
                    persist_last_checkpoint(periodic_payload)
                    wall_checkpoint_at = current_wall_time
                should_stop = stop_request.requested or (
                    stop_after_optimizer_updates is not None
                    and successful_updates + skipped_updates >= stop_after_optimizer_updates
                )
                if should_stop:
                    payload = checkpoint_payload(
                        completed_epoch=epoch - 1,
                        resume_at_epoch=epoch,
                        consumed_batches=batch_index,
                        epoch_loss_sum=epoch_loss_sum,
                        epoch_sample_count=epoch_sample_count,
                        epoch_boundaries=update_boundaries,
                        checkpoint_reason=(
                            "signal_request" if stop_request.requested else "test_stop"
                        ),
                    )
                    persist_last_checkpoint(payload)
                    interrupted = True
                    break
            if interrupted:
                return HSTFoldResult(
                    last_epoch=epoch - 1,
                    best_epoch=best_epoch,
                    resumed_from_epoch=resumed_from,
                    validation_threshold=math.nan,
                    history=pd.DataFrame(history_rows),
                    skipped_optimizer_updates=skipped_updates,
                    training_complete=False,
                    training_contract_fingerprint=contract_fingerprint,
                    interrupted=True,
                )
            if update_boundaries != updates_per_epoch:
                raise RuntimeError(
                    "Observed optimizer-update boundaries differ from the OneCycleLR budget"
                )

            validation_predictions = predict_hst_split(
                model,
                loaders["validation"],
                split="validation",
                fold=int(execution_identity["fold"]),
                modality=str(execution_identity["modality"]),
                prediction_context=context,
            )
            selection = _participant_selection_metrics(
                validation_predictions, config.epoch_selection_threshold
            )
            score = validation_epoch_score(selection, epoch=epoch)
            improved = score > best_score
            if improved:
                best_score = score
                best_epoch = epoch
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": _sample_weighted_epoch_loss(
                        total_loss=epoch_loss_sum,
                        sample_count=epoch_sample_count,
                    ),
                    "validation_f1_at_0.5": selection["f1"],
                    "validation_nll": selection["nll"],
                    "validation_auroc": selection["auroc"],
                    "validation_auprc": selection["auprc"],
                    "learning_rate": scheduler.get_last_lr()[0],
                    "draw_plan_sha256": draw_hash,
                    "batch_sequence_sha256": epoch_batch_schedule[
                        "batch_sequence_sha256"
                    ],
                    "optimizer_boundary_count": epoch_batch_schedule[
                        "optimizer_boundary_count"
                    ],
                }
            )
            completed_epoch_schedules.append(epoch_batch_schedule)
            common_payload = checkpoint_payload(
                completed_epoch=epoch,
                resume_at_epoch=epoch + 1,
                consumed_batches=0,
                epoch_loss_sum=0.0,
                epoch_sample_count=0,
                epoch_boundaries=0,
                checkpoint_reason="epoch_end",
            )
            persist_last_checkpoint(common_payload)
            if improved:
                _atomic_torch_save(
                    {**common_payload, "checkpoint_role": "best"},
                    run_dir / "best.pt",
                )
            last_epoch = epoch
            if stop_request.requested:
                return HSTFoldResult(
                    last_epoch=last_epoch,
                    best_epoch=best_epoch,
                    resumed_from_epoch=resumed_from,
                    validation_threshold=math.nan,
                    history=pd.DataFrame(history_rows),
                    skipped_optimizer_updates=skipped_updates,
                    training_complete=False,
                    training_contract_fingerprint=contract_fingerprint,
                    interrupted=True,
                )
            resume_consumed_batches = 0
            resume_epoch_loss_sum = 0.0
            resume_epoch_sample_count = 0
            resume_update_boundaries = 0
            resume_draw_plan_hash = None
            resume_epoch_schedule = None
            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                break

    best_path = run_dir / "best.pt"
    best_loaded = _load_latest_valid_checkpoint([best_path])
    if best_loaded is None:
        raise RuntimeError("HST training did not produce a valid best checkpoint")
    validate_resume_checkpoint_contract(
        best_loaded[0],
        expected_fingerprint=contract_fingerprint,
    )
    if best_loaded[0].get("checkpoint_role") != "best":
        raise ValueError("Best checkpoint has the wrong immutable role")
    if int(best_loaded[0].get("epoch", -1)) != int(best_epoch) or int(
        best_loaded[0].get("best_epoch", -1)
    ) != int(best_epoch):
        raise ValueError("Best checkpoint identity does not match the selected validation epoch")
    model.load_state_dict(best_loaded[0]["model_state_dict"])  # type: ignore[attr-defined]
    best_checkpoint_sha256 = stable_file_sha256(best_loaded[1])
    full_training_schedule, full_training_schedule_sha256 = (
        _completed_training_schedule(config, completed_epoch_schedules)
    )
    scientific_training_claim = _validate_scientific_training_claim(
        {
            "schema_version": 1,
            "data_contracts_freeze_hash": config.data_contracts_freeze_hash,
            "manifest_selection_sha256": _scientific_manifest_selection_sha256(
                aligned_manifest_cache
            ),
            "training_schedule": full_training_schedule,
            "training_schedule_sha256": full_training_schedule_sha256,
        }
    )
    final_context = {
        **context,
        "checkpoint_hash": best_checkpoint_sha256,
    }
    validation_predictions = predict_hst_split(
        model,
        loaders["validation"],
        split="validation",
        fold=int(loaders.get("fold", 0)),
        modality=str(loaders.get("modality", "unknown")),
        prediction_context=final_context,
    )
    participants = aggregate_recording_predictions(validation_predictions)
    threshold = best_threshold_by_balanced_accuracy(
        labels_to_binary(participants["label_binary"]),
        participants["probability"].to_numpy(dtype=float),
    )
    training_complete = last_epoch == config.max_epochs
    available_splits = {
        split for split in ("validation", "test", "external_test") if split in loaders
    }
    validate_evaluation_request(
        training_complete=training_complete,
        evaluate_test=evaluate_test,
        evaluate_external=evaluate_external,
        available_splits=available_splits,
        confirmatory=config.confirmatory,
    )
    fold = int(loaders.get("fold", 0))
    modality = str(loaders.get("modality", "unknown"))
    test_predictions = pd.DataFrame()
    external_predictions = pd.DataFrame()
    if evaluate_test:
        assert evaluation_registry_root is not None
        test_predictions = _evaluate_split_once(
            model,
            loaders["test"],
            split="test",
            fold=fold,
            modality=modality,
            prediction_context=final_context,
            run_dir=run_dir,
            project_registry_root=evaluation_registry_root,
            training_contract=contract_fingerprint,
            best_checkpoint_sha256=best_checkpoint_sha256,
            scientific_training_claim=scientific_training_claim,
        )
    if evaluate_external:
        assert evaluation_registry_root is not None
        external_predictions = _evaluate_split_once(
            model,
            loaders["external_test"],
            split="external_test",
            fold=fold,
            modality=modality,
            prediction_context=final_context,
            run_dir=run_dir,
            project_registry_root=evaluation_registry_root,
            training_contract=contract_fingerprint,
            best_checkpoint_sha256=best_checkpoint_sha256,
            scientific_training_claim=scientific_training_claim,
        )
    return HSTFoldResult(
        last_epoch=last_epoch,
        best_epoch=best_epoch,
        resumed_from_epoch=resumed_from,
        validation_threshold=float(threshold),
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        external_predictions=external_predictions,
        history=pd.DataFrame(history_rows),
        skipped_optimizer_updates=skipped_updates,
        training_complete=training_complete,
        test_evaluated=evaluate_test,
        external_evaluated=evaluate_external,
        best_checkpoint_sha256=best_checkpoint_sha256,
        training_contract_fingerprint=contract_fingerprint,
        interrupted=False,
    )
