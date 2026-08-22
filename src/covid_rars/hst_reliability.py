from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pandas as pd

from .hst_runtime import (
    ProcessIdentity,
    ProcessLiveness,
    atomic_write_json,
    canonical_json_sha256,
    capture_process_identity,
    process_identity_liveness,
    read_json,
    source_tree_hash,
    stable_file_sha256,
    stage_fingerprint,
)


DEFAULT_DETACHED_STALE_AFTER_SECONDS = 5.0 * 60.0
DEFAULT_DETACHED_MAX_WAIT_SECONDS = 8.0 * 24.0 * 60.0 * 60.0
_DETACHED_NONTERMINAL_STATUSES = frozenset(
    {"initializing", "launching", "running", "stopping"}
)
_DETACHED_TERMINAL_STATUSES = frozenset({"success", "failed", "stopped"})
_DETACHED_STATUSES = _DETACHED_NONTERMINAL_STATUSES | _DETACHED_TERMINAL_STATUSES
_IMMUTABLE_LAUNCH_FIELDS = (
    "pid",
    "command",
    "log_path",
    "host",
    "process_start_identity",
    "heartbeat_path",
    "launched_at",
    "launched_at_unix",
)


StageHandler = Callable[["HSTPipeline", str], Mapping[str, object] | None]
SHA256_LENGTH = 64
REQUIRED_FULL_FREEZES = frozenset(
    {"data_contracts_freeze", "pilot_freeze", "environment_lock"}
)
_BYTES_PER_MIB = 1024.0 * 1024.0
_GPU_MEMORY_METADATA_FIELDS = frozenset(
    {
        "gpu_memory_measured",
        "peak_gpu_memory_allocated_mb",
        "peak_gpu_memory_reserved_mb",
        "peak_gpu_memory_mb",
    }
)
_IN_PROCESS_CUDA_STAGES = frozenset(
    {
        "small_smoke",
        "internal_cv",
        "split_policy_contrast",
        "reverse_temporal",
        "external_transfer",
        "gradcam",
    }
)
_CUDA_DETERMINISTIC_WORKSPACE = ":4096:8"


class StageExecutionError(RuntimeError):
    """Raised after a failed stage has been durably recorded."""


def hst_process_environment(
    *,
    device: str,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the canonical process environment for an HST controller run."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    environment = dict(os.environ if base_environment is None else base_environment)
    if device == "cuda":
        environment["CUBLAS_WORKSPACE_CONFIG"] = _CUDA_DETERMINISTIC_WORKSPACE
    return environment


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != SHA256_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _unmeasured_gpu_memory_metadata() -> dict[str, object]:
    return {
        "gpu_memory_measured": False,
        "peak_gpu_memory_allocated_mb": None,
        "peak_gpu_memory_reserved_mb": None,
    }


def _start_cuda_memory_measurement(device: str, *, stage: str) -> object | None:
    if device != "cuda" or stage not in _IN_PROCESS_CUDA_STAGES:
        return None
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Cannot measure CUDA memory because CUDA is unavailable")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda


def _finish_cuda_memory_measurement(cuda: object | None) -> dict[str, object]:
    if cuda is None:
        return _unmeasured_gpu_memory_metadata()
    cuda.synchronize()
    allocated_bytes = int(cuda.max_memory_allocated())
    reserved_bytes = int(cuda.max_memory_reserved())
    if allocated_bytes < 0 or reserved_bytes < 0:
        raise RuntimeError("CUDA peak-memory counters cannot be negative")
    allocated_mb = allocated_bytes / _BYTES_PER_MIB
    reserved_mb = reserved_bytes / _BYTES_PER_MIB
    return {
        "gpu_memory_measured": True,
        "peak_gpu_memory_allocated_mb": allocated_mb,
        "peak_gpu_memory_reserved_mb": reserved_mb,
        # Retained for compatibility with existing runtime-table consumers.
        "peak_gpu_memory_mb": allocated_mb,
    }


def _merge_gpu_memory_metadata(
    supplied: object,
    measured: Mapping[str, object],
) -> dict[str, object]:
    metadata = dict(supplied) if isinstance(supplied, Mapping) else {}
    for field_name in _GPU_MEMORY_METADATA_FIELDS:
        metadata.pop(field_name, None)
    metadata.update(measured)
    return metadata


@dataclass
class HSTPipelineConfig:
    workspace_root: Path
    mode: str
    scientific_config: dict[str, object]
    source_root: Path
    source_paths: tuple[Path, ...]
    dependency_lock_path: Path
    hst_commit: str = "7f94ad81e392da856c7aac6d364d036c28e26c32"
    checkpoint_hashes: dict[str, str] = field(default_factory=dict)
    input_hashes: dict[str, str] = field(default_factory=dict)
    manifest_hashes: dict[str, str] = field(default_factory=dict)
    accepted_hashes: dict[str, str] = field(default_factory=dict)
    pip_freeze_hash: str | None = None
    expected_run_id: str = "auto"
    device: str = "cpu"
    resume: bool = True

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()
        self.source_root = Path(self.source_root).resolve()
        self.source_paths = tuple(Path(path).resolve() for path in self.source_paths)
        self.dependency_lock_path = Path(self.dependency_lock_path).resolve()
        if self.mode not in {"smoke", "pilot", "full"}:
            raise ValueError("mode must be smoke, pilot, or full")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if not self.source_paths:
            raise ValueError("At least one executable source path is required")
        for path in self.source_paths:
            try:
                path.relative_to(self.source_root)
            except ValueError as exc:
                raise ValueError(f"Source path escapes source root: {path}") from exc
            if not path.is_file():
                raise FileNotFoundError(path)
        if not self.dependency_lock_path.is_file():
            raise FileNotFoundError(self.dependency_lock_path)
        if self.mode == "full":
            missing = sorted(REQUIRED_FULL_FREEZES - set(self.accepted_hashes))
            invalid = sorted(
                key
                for key in REQUIRED_FULL_FREEZES & set(self.accepted_hashes)
                if not _valid_sha256(self.accepted_hashes[key])
            )
            if missing or invalid:
                raise ValueError(
                    "full mode requires accepted data-contract, pilot, and environment "
                    f"freeze hashes; missing={missing}, invalid={invalid}"
                )

    @classmethod
    def smoke(
        cls,
        root: Path,
        *,
        expected_run_id: str = "auto",
        device: str = "cpu",
    ) -> "HSTPipelineConfig":
        root = Path(root).resolve()
        fixture_root = root / "hst_smoke_inputs"
        fixture_root.mkdir(parents=True, exist_ok=True)
        source = fixture_root / "source.py"
        dependency_lock = fixture_root / "requirements.lock"
        if not source.exists():
            source.write_text("HST_SMOKE_SOURCE = 1\n", encoding="utf-8")
        if not dependency_lock.exists():
            dependency_lock.write_text("smoke-dependency==1\n", encoding="utf-8")
        return cls(
            workspace_root=root,
            mode="smoke",
            scientific_config={
                "schema_version": 1,
                "mode": "smoke",
                "model": "hst_small",
                "folds": 1,
                "epochs": 2,
                "modalities": ["cough"],
            },
            source_root=fixture_root,
            source_paths=(source,),
            dependency_lock_path=dependency_lock,
            checkpoint_hashes={"hst_small_imagenet": "d" * 64},
            input_hashes={"smoke_input_contract": "c" * 64},
            manifest_hashes={"smoke_manifest": "e" * 64},
            pip_freeze_hash="f" * 64,
            expected_run_id=expected_run_id,
            device=device,
        )

    @classmethod
    def full(
        cls,
        root: Path,
        *,
        accepted_hashes: Mapping[str, str],
        expected_run_id: str = "auto",
        device: str = "cuda",
    ) -> "HSTPipelineConfig":
        root = Path(root).resolve()
        fixture_root = root / "hst_full_inputs"
        fixture_root.mkdir(parents=True, exist_ok=True)
        source = fixture_root / "source.py"
        dependency_lock = fixture_root / "requirements.lock"
        if not source.exists():
            source.write_text("HST_FULL_SOURCE = 1\n", encoding="utf-8")
        if not dependency_lock.exists():
            dependency_lock.write_text("full-dependency==1\n", encoding="utf-8")
        return cls(
            workspace_root=root,
            mode="full",
            scientific_config={
                "schema_version": 1,
                "mode": "full",
                "model": "hst_base",
                "train_all_epochs": True,
            },
            source_root=fixture_root,
            source_paths=(source,),
            dependency_lock_path=dependency_lock,
            checkpoint_hashes={"hst_base_imagenet": "d" * 64},
            input_hashes={"full_input_contract": "e" * 64},
            manifest_hashes={"full_manifest": "f" * 64},
            accepted_hashes=dict(accepted_hashes),
            pip_freeze_hash=str(accepted_hashes.get("environment_lock", "")),
            expected_run_id=expected_run_id,
            device=device,
        )


class HSTPipeline:
    STAGES = (
        "preflight",
        "data_contracts",
        "checkpoint",
        "preprocess_worker_pilot",
        "spectrogram_cache",
        "manifests",
        "small_smoke",
        "base_resource_pilot",
        "aligned_comparator",
        "internal_cv",
        "split_policy_contrast",
        "reverse_temporal",
        "external_transfer",
        "fusion",
        "statistics",
        "gradcam",
        "evidence_pack",
    )
    MODE_LIMITS = {
        "smoke": "small_smoke",
        "pilot": "base_resource_pilot",
        "full": "evidence_pack",
    }

    @property
    def capacity_mode(self) -> bool:
        """Return True if this pipeline uses reduced capacity workload."""
        return self._capacity_mode

    def __init__(
        self,
        config: HSTPipelineConfig,
        *,
        stage_handlers: Mapping[str, StageHandler] | None = None,
        stage_hook: Callable[[str], object] | None = None,
    ) -> None:
        self.config = config
        self.stage_handlers: dict[str, StageHandler] = dict(stage_handlers or {})
        self.stage_hook = stage_hook
        self._capacity_mode = False
        self.configuration_hash = canonical_json_sha256(config.scientific_config)
        self.initial_source_hash = self._source_hash()
        self.initial_dependency_hash = stable_file_sha256(config.dependency_lock_path)
        self.run_id = self._derive_run_id()
        if config.expected_run_id != "auto" and config.expected_run_id != self.run_id:
            raise ValueError(
                f"Asserted run ID {config.expected_run_id!r} does not match "
                f"content-derived run ID {self.run_id!r}"
            )
        self.run_root = (
            config.workspace_root / "data" / "outputs" / "hst" / self.run_id
        ).resolve()
        self.runtime_root = self.run_root / "runtime"
        self.stage_root = self.runtime_root / "stages"
        self.stage_root.mkdir(parents=True, exist_ok=True)
        self._verified_shared_output_receipts: set[str] = set()

    def _derive_run_id(self) -> str:
        digest = canonical_json_sha256(
            {
                "schema_version": 1,
                "mode": self.config.mode,
                "device": self.config.device,
                "scientific_configuration": self.config.scientific_config,
                "source_hash": self.initial_source_hash,
                "dependency_lock_hash": self.initial_dependency_hash,
                "hst_commit": self.config.hst_commit,
                "checkpoint_hashes": self.config.checkpoint_hashes,
                "input_hashes": self.config.input_hashes,
                "manifest_hashes": self.config.manifest_hashes,
                "accepted_hashes": self.config.accepted_hashes,
                "pip_freeze_hash": self.config.pip_freeze_hash,
            }
        )
        return f"hst-{digest[:20]}"

    def _source_hash(self) -> str:
        relative = tuple(
            path.relative_to(self.config.source_root) for path in self.config.source_paths
        )
        return source_tree_hash(self.config.source_root, relative)

    def _generated_manifest_hashes(self) -> dict[str, str]:
        """Load downstream manifest bindings from the verified stage outputs."""
        index_path = self.run_root / "manifests" / "manifest_index.json"
        receipt_path = self.stage_receipt_path("manifests")
        if not index_path.is_file() or not receipt_path.is_file():
            raise StageExecutionError(
                "Generated manifest hashes are unavailable before the manifests stage"
            )
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StageExecutionError("Generated manifest provenance is unreadable") from exc
        if receipt.get("status") != "success":
            raise StageExecutionError("The manifests stage has no successful receipt")
        output_checksums = receipt.get("output_checksums")
        manifests = index.get("manifests")
        if not isinstance(output_checksums, Mapping) or not isinstance(manifests, Mapping):
            raise StageExecutionError("Generated manifest provenance is malformed")

        index_relative = index_path.relative_to(self.run_root).as_posix()
        index_hash = stable_file_sha256(index_path)
        if output_checksums.get(index_relative) != index_hash:
            raise StageExecutionError(
                "Manifest index bytes disagree with the manifests-stage receipt"
            )
        resolved_hashes: dict[str, str] = {"manifest_index": index_hash}
        for raw_name, raw_entry in sorted(
            manifests.items(), key=lambda item: str(item[0])
        ):
            name = str(raw_name).strip()
            if not name or not isinstance(raw_entry, Mapping):
                raise StageExecutionError("Manifest index contains an invalid entry")
            supplied_path = Path(str(raw_entry.get("path", "")))
            manifest_path = (
                supplied_path.resolve()
                if supplied_path.is_absolute()
                else (self.run_root / supplied_path).resolve()
            )
            try:
                relative = manifest_path.relative_to(self.run_root).as_posix()
            except ValueError as exc:
                raise StageExecutionError(
                    f"Generated manifest path escapes the run root: {supplied_path}"
                ) from exc
            expected_hash = str(raw_entry.get("sha256", "")).casefold()
            if not _valid_sha256(expected_hash) or not manifest_path.is_file():
                raise StageExecutionError(
                    f"Generated manifest entry is incomplete: {name}"
                )
            actual_hash = stable_file_sha256(manifest_path)
            if (
                actual_hash != expected_hash
                or output_checksums.get(relative) != actual_hash
            ):
                raise StageExecutionError(
                    f"Generated manifest bytes disagree with frozen provenance: {name}"
                )
            resolved_hashes[name] = actual_hash
        if len(resolved_hashes) == 1:
            raise StageExecutionError("Manifest index contains no generated manifests")
        return resolved_hashes

    def _manifest_hashes_for_stage(self, stage: str) -> dict[str, str]:
        configured = dict(self.config.manifest_hashes)
        if configured:
            return configured
        if self.STAGES.index(stage) <= self.STAGES.index("manifests"):
            return {}
        return self._generated_manifest_hashes()

    def stage_receipt_path(self, stage: str) -> Path:
        self._validate_stage(stage)
        return self.stage_root / f"{stage}.json"

    def _validate_stage(self, stage: str) -> None:
        if stage not in self.STAGES:
            raise ValueError(f"Unknown HST stage: {stage}")

    def _upstream_receipt_hashes(
        self,
        stage: str,
        *,
        fingerprint_cache: dict[str, str] | None = None,
    ) -> dict[str, str]:
        index = self.STAGES.index(stage)
        fingerprint_cache = (
            {} if fingerprint_cache is None else fingerprint_cache
        )
        hashes: dict[str, str] = {}
        for upstream in self.STAGES[:index]:
            path = self.stage_receipt_path(upstream)
            if not path.is_file():
                raise StageExecutionError(
                    f"Required upstream stage {upstream!r} has no receipt before {stage!r}"
                )
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise StageExecutionError(
                    f"Required upstream receipt {upstream!r} is unreadable"
                ) from exc
            expected_fingerprint = self._stage_fingerprint(
                upstream,
                fingerprint_cache=fingerprint_cache,
            )
            if not self._receipt_is_reusable(
                receipt,
                fingerprint=expected_fingerprint,
                stage=upstream,
            ):
                raise StageExecutionError(
                    f"Required upstream stage {upstream!r} failed checksum/reuse validation"
                )
            hashes[upstream] = stable_file_sha256(path)
        return hashes

    def _stage_fingerprint(
        self,
        stage: str,
        *,
        fingerprint_cache: dict[str, str] | None = None,
    ) -> str:
        fingerprint_cache = (
            {} if fingerprint_cache is None else fingerprint_cache
        )
        cached = fingerprint_cache.get(stage)
        if cached is not None:
            return cached
        manifest_hashes = self._manifest_hashes_for_stage(stage)
        fingerprint = stage_fingerprint(
            stage,
            input_hashes=self.config.input_hashes,
            configuration_hash=self.configuration_hash,
            executable_source_hash=self._source_hash(),
            dependency_lock_hash=stable_file_sha256(
                self.config.dependency_lock_path
            ),
            hst_commit=self.config.hst_commit,
            checkpoint_hashes=self.config.checkpoint_hashes,
            manifest_hashes=manifest_hashes,
            upstream_hashes=self._upstream_receipt_hashes(
                stage,
                fingerprint_cache=fingerprint_cache,
            ),
            accepted_hashes=self.config.accepted_hashes,
            pip_freeze_hash=self.config.pip_freeze_hash,
            extra={"mode": self.config.mode, "device": self.config.device},
            capacity_mode=self.capacity_mode,
        )
        fingerprint_cache[stage] = fingerprint
        return fingerprint

    def _resolve_output(self, supplied: object) -> tuple[str, Path]:
        candidate = Path(str(supplied))
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.run_root / candidate).resolve()
        try:
            relative = resolved.relative_to(self.run_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Output path escapes or is outside run root: {supplied}") from exc
        return relative, resolved

    def _validated_output_checksums(
        self, output_paths: object
    ) -> tuple[list[str], dict[str, str]]:
        if output_paths is None:
            supplied_paths: list[object] = []
        elif isinstance(output_paths, (str, Path)):
            supplied_paths = [output_paths]
        else:
            supplied_paths = list(output_paths)  # type: ignore[arg-type]
        relative_paths: list[str] = []
        checksums: dict[str, str] = {}
        for supplied in supplied_paths:
            relative, resolved = self._resolve_output(supplied)
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            if relative in checksums:
                raise ValueError(f"Duplicate output path: {relative}")
            relative_paths.append(relative)
            checksums[relative] = stable_file_sha256(resolved)
        return relative_paths, checksums

    def _receipt_is_reusable(
        self,
        receipt: Mapping[str, object],
        *,
        fingerprint: str,
        stage: str,
    ) -> bool:
        if (
            receipt.get("status") != "success"
            or receipt.get("receipt_type") != "hst_stage"
            or receipt.get("run_id") != self.run_id
            or receipt.get("stage") != stage
            or receipt.get("fingerprint") != fingerprint
        ):
            return False
        record_hash = receipt.get("record_hash")
        if not isinstance(record_hash, str) or len(record_hash) != SHA256_LENGTH:
            return False
        unsigned = dict(receipt)
        unsigned.pop("record_hash", None)
        if canonical_json_sha256(unsigned) != record_hash:
            return False
        paths = receipt.get("output_paths")
        checksums = receipt.get("output_checksums")
        if not isinstance(paths, list) or not paths:
            return False
        if not isinstance(checksums, dict):
            return False
        try:
            for supplied in paths:
                relative, resolved = self._resolve_output(supplied)
                if not resolved.is_file():
                    return False
                if checksums.get(relative) != stable_file_sha256(resolved):
                    return False
        except (OSError, TypeError, ValueError):
            return False
        if stage == "spectrogram_cache" and record_hash not in self._verified_shared_output_receipts:
            try:
                from covid_rars.hst_spectrograms import validate_hst_cache_index

                index_candidates = [
                    self._resolve_output(path)[1]
                    for path in paths
                    if Path(str(path)).name == "spectrogram_cache_index.csv"
                ]
                if len(index_candidates) != 1:
                    return False
                validate_hst_cache_index(
                    index_candidates[0],
                    cache_root=(
                        self.config.workspace_root
                        / "data"
                        / "processed"
                        / "hst_spectrogram_cache"
                    ),
                )
            except (OSError, TypeError, ValueError):
                return False
            self._verified_shared_output_receipts.add(str(record_hash))
        if stage == "aligned_comparator":
            try:
                identity_candidates = [
                    self._resolve_output(path)[1]
                    for path in paths
                    if Path(str(path)).name == "generation_identity.json"
                ]
                if len(identity_candidates) != 1:
                    return False
                identity = json.loads(identity_candidates[0].read_text(encoding="ascii"))
                if not isinstance(identity, dict):
                    return False
                trusted_root = self.config.workspace_root.resolve()
                for path_field, hash_field in (
                    ("approval_path", "approval_byte_sha256"),
                    ("accepted_freezes_path", "accepted_freezes_byte_sha256"),
                ):
                    relative = Path(str(identity[path_field]))
                    if relative.is_absolute():
                        return False
                    canonical = (trusted_root / relative).resolve()
                    canonical.relative_to(trusted_root)
                    if not canonical.is_file() or canonical.is_symlink():
                        return False
                    if stable_file_sha256(canonical) != str(identity[hash_field]):
                        return False
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                return False
        return True

    def _default_stage_handler(self, stage: str) -> Mapping[str, object]:
        if stage == "spectrogram_cache":
            output = self.run_root / "manifests" / "spectrogram_cache_index.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                columns=["eligible", "cache_path", "tensor_sha256"]
            ).to_csv(output, index=False)
            return {"output_paths": [output], "row_counts": {"records": 0}}
        output = self.run_root / "artifacts" / f"{stage}.json"
        atomic_write_json(
            output,
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "stage": stage,
                "mode": self.config.mode,
                "status": "smoke-placeholder",
            },
        )
        return {"output_paths": [output], "row_counts": {"records": 1}}

    def run_stage(
        self,
        stage: str,
        *,
        force: bool = False,
        _fingerprint_cache: dict[str, str] | None = None,
    ) -> dict[str, object]:
        self._validate_stage(stage)
        receipt_path = self.stage_receipt_path(stage)
        fingerprint_cache = (
            {} if _fingerprint_cache is None else _fingerprint_cache
        )
        fingerprint = self._stage_fingerprint(
            stage,
            fingerprint_cache=fingerprint_cache,
        )
        previous: dict[str, object] = {}
        if receipt_path.exists():
            try:
                previous = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
        if not force and self._receipt_is_reusable(
            previous,
            fingerprint=fingerprint,
            stage=stage,
        ):
            reused = dict(previous)
            reused["reused"] = True
            return reused

        attempt = int(previous.get("attempt", 0)) + 1
        started_at = _utc_now()
        manifest_hashes = self._manifest_hashes_for_stage(stage)
        base = {
            "schema_version": 1,
            "receipt_type": "hst_stage",
            "run_id": self.run_id,
            "stage": stage,
            "attempt": attempt,
            "fingerprint": fingerprint,
            "configuration_hash": self.configuration_hash,
            "source_hash": self._source_hash(),
            "dependency_lock_hash": stable_file_sha256(
                self.config.dependency_lock_path
            ),
            "hst_commit": self.config.hst_commit,
            "checkpoint_hashes": dict(self.config.checkpoint_hashes),
            "manifest_hashes": manifest_hashes,
            "accepted_hashes": dict(self.config.accepted_hashes),
            "upstream_receipt_hashes": self._upstream_receipt_hashes(
                stage,
                fingerprint_cache=fingerprint_cache,
            ),
            "pip_freeze_hash": self.config.pip_freeze_hash,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started_at": started_at,
        }
        cuda_memory = None
        memory_measurement_active = False
        gpu_memory_metadata = _unmeasured_gpu_memory_metadata()
        try:
            if self.stage_hook is not None:
                self.stage_hook(stage)
            handler = self.stage_handlers.get(stage)
            if handler is None:
                if self.config.mode != "smoke":
                    raise RuntimeError(
                        f"No scientific stage handler is registered for {stage!r}"
                    )
            cuda_memory = _start_cuda_memory_measurement(
                self.config.device,
                stage=stage,
            )
            memory_measurement_active = cuda_memory is not None
            if handler is None:
                result = self._default_stage_handler(stage)
            else:
                result = dict(handler(self, stage) or {})
            gpu_memory_metadata = _finish_cuda_memory_measurement(cuda_memory)
            memory_measurement_active = False
            output_paths, output_checksums = self._validated_output_checksums(
                result.get("output_paths")
            )
            if not output_paths:
                raise ValueError(f"Stage {stage} produced no auditable outputs")
            receipt = {
                **base,
                "status": "success",
                "completed_at": _utc_now(),
                "output_paths": output_paths,
                "output_checksums": output_checksums,
                "row_counts": dict(result.get("row_counts", {})),
                "metadata": _merge_gpu_memory_metadata(
                    result.get("metadata", {}), gpu_memory_metadata
                ),
                "error": None,
                "reused": False,
            }
            receipt["record_hash"] = canonical_json_sha256(receipt)
            atomic_write_json(receipt_path, receipt)
            return receipt
        except Exception as exc:
            if memory_measurement_active:
                try:
                    gpu_memory_metadata = _finish_cuda_memory_measurement(cuda_memory)
                except Exception as measurement_exc:
                    gpu_memory_metadata = {
                        **_unmeasured_gpu_memory_metadata(),
                        "gpu_memory_measurement_error": (
                            f"{type(measurement_exc).__name__}: {measurement_exc}"
                        ),
                    }
            failed = {
                **base,
                "status": "failed",
                "completed_at": _utc_now(),
                "output_paths": [],
                "output_checksums": {},
                "row_counts": {},
                "metadata": gpu_memory_metadata,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "reused": False,
            }
            failed["record_hash"] = canonical_json_sha256(failed)
            atomic_write_json(receipt_path, failed)
            raise StageExecutionError(
                f"HST stage {stage!r} failed: {exc}"
            ) from exc

    def run(
        self,
        *,
        through: str = "evidence_pack",
        force: set[str] | None = None,
    ) -> pd.DataFrame:
        self._validate_stage(through)
        maximum = self.STAGES.index(self.MODE_LIMITS[self.config.mode])
        requested = self.STAGES.index(through)
        if requested > maximum:
            raise ValueError(
                f"mode={self.config.mode!r} may run only through "
                f"{self.MODE_LIMITS[self.config.mode]!r}"
            )
        force = set(force or set())
        unknown = sorted(force - set(self.STAGES))
        if unknown:
            raise ValueError(f"Unknown forced stages: {unknown}")
        fingerprint_cache: dict[str, str] = {}
        rows = [
            self.run_stage(
                stage,
                force=stage in force,
                _fingerprint_cache=fingerprint_cache,
            )
            for stage in self.STAGES[: requested + 1]
        ]
        return pd.DataFrame(rows)


class HSTCapacityInternalFusionPipeline(HSTPipeline):
    """Bounded HST controller for the frozen internal cough+speech question."""

    def __init__(
        self,
        config: HSTPipelineConfig,
        *,
        stage_handlers: Mapping[str, StageHandler] | None = None,
        stage_hook: Callable[[str], object] | None = None,
    ) -> None:
        super().__init__(config, stage_handlers=stage_handlers, stage_hook=stage_hook)
        self._capacity_mode = True

    STAGES = (
        "preflight",
        "data_contracts",
        "checkpoint",
        "preprocess_worker_pilot",
        "spectrogram_cache",
        "manifests",
        "small_smoke",
        "base_resource_pilot",
        "internal_cv",
        "fusion",
        "gradcam",
        "evidence_pack",
    )
    MODE_LIMITS = {
        "smoke": "small_smoke",
        "pilot": "base_resource_pilot",
        "full": "evidence_pack",
    }


def pipeline_class_for_config(
    scientific_config: Mapping[str, object],
) -> type[HSTPipeline]:
    from .hst_workloads import (
        CAPACITY_INTERNAL_FUSION_PROFILE,
        workload_profile_from_scientific_config,
    )

    experiment = scientific_config.get("experiment")
    if not isinstance(experiment, Mapping) or "workload_profile" not in experiment:
        return HSTPipeline
    profile = workload_profile_from_scientific_config(scientific_config)
    if profile.name == CAPACITY_INTERNAL_FUSION_PROFILE:
        return HSTCapacityInternalFusionPipeline
    return HSTPipeline


def _read_optional_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _merge_scientific_config(
    base: Mapping[str, object],
    override: Mapping[str, object],
) -> dict[str, object]:
    if override.get("__replace__") is True:
        return {
            str(key): value
            for key, value in override.items()
            if key not in {"__replace__", "extends"}
        }
    merged: dict[str, object] = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_scientific_config(current, value)
        else:
            merged[key] = value
    return merged


def _read_scientific_config(
    path: Path,
    *,
    _seen: frozenset[Path] = frozenset(),
) -> dict[str, object]:
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError("Scientific configuration inheritance contains a cycle")
    payload = _read_optional_json(path)
    inherited = payload.get("extends")
    if inherited is None:
        return payload
    if not isinstance(inherited, str) or not inherited.strip():
        raise ValueError("Scientific configuration extends must be a relative path")
    supplied = Path(inherited)
    if supplied.is_absolute():
        raise ValueError("Scientific configuration extends cannot be absolute")
    base_path = (path.parent / supplied).resolve()
    try:
        base_path.relative_to(path.parent.resolve())
    except ValueError as exc:
        raise ValueError("Scientific configuration extends escapes its directory") from exc
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    base = _read_scientific_config(base_path, _seen=_seen | {path})
    return _merge_scientific_config(base, payload)


def _controller_source_paths(project_root: Path) -> tuple[Path, ...]:
    reused_modules = (
        "metrics.py",
        "calibration.py",
        "labels.py",
        "audio_io.py",
        "strong_baseline.py",
        "strong_baseline_protocol.py",
        "compare_is10_rescue.py",
        "compare_is10_final_validation.py",
        "config.py",
        "features.py",
        "fusion.py",
        "metadata_baseline.py",
        "metadata_confounding.py",
        "preprocess.py",
        "temporal_holdout.py",
    )
    package_root = project_root / "src" / "covid_rars"
    candidates = [
        *sorted(package_root.glob("hst_*.py")),
        *(package_root / name for name in reused_modules),
        *sorted((project_root / "scripts").glob("hst_*.py")),
        *sorted((project_root / "scripts").glob("7[2-8]_*.py")),
    ]
    files = tuple(path.resolve() for path in candidates if path.is_file())
    if not files:
        raise FileNotFoundError("No HST executable source files were found")

    allowed = set(files)
    unlisted: set[Path] = set()
    for source_path in files:
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise ValueError(f"Unable to audit executable imports: {source_path}") from exc
        module_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.module:
                        module_names.add(node.module.split(".", 1)[0])
                    else:
                        module_names.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif node.module and node.module.startswith("covid_rars."):
                    module_names.add(node.module.split(".", 2)[1])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("covid_rars."):
                        module_names.add(alias.name.split(".", 2)[1])
        for module_name in module_names:
            module_path = (package_root / f"{module_name}.py").resolve()
            package_path = (package_root / module_name / "__init__.py").resolve()
            local_path = module_path if module_path.is_file() else package_path
            if local_path.is_file() and local_path not in allowed:
                unlisted.add(local_path)
    if unlisted:
        relative = ", ".join(
            path.relative_to(project_root).as_posix() for path in sorted(unlisted)
        )
        raise ValueError(f"Found unlisted local executable import(s): {relative}")
    return files


def _accepted_hash_mapping(payload: Mapping[str, object]) -> dict[str, str]:
    raw = payload.get("accepted_hashes", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("Accepted-freezes document must contain a hash mapping")
    return {
        str(key): str(value)
        for key, value in raw.items()
        if isinstance(value, str)
    }


def audio_input_manifest_records(
    metadata_path: Path,
    *,
    project_root: Path,
    modality: str | None = None,
) -> list[dict[str, object]]:
    """Describe every declared source-audio byte stream used by a scientific run."""
    metadata_path = Path(metadata_path).resolve()
    project_root = Path(project_root).resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    header = pd.read_csv(metadata_path, nrows=0)
    if "audio_path" not in header.columns:
        raise ValueError(f"Metadata has no audio_path column: {metadata_path}")
    identity_column = next(
        (
            column
            for column in ("recording_key", "recording_id", "uuid", "id")
            if column in header.columns
        ),
        None,
    )
    if identity_column is None:
        raise ValueError(f"Metadata has no recording identity column: {metadata_path}")
    usecols = [identity_column, "audio_path"]
    if "modality" in header.columns:
        usecols.append("modality")
    frame = pd.read_csv(metadata_path, usecols=usecols, low_memory=False)
    if modality is not None:
        if "modality" not in frame:
            raise ValueError("A modality filter requires a metadata modality column")
        frame = frame.loc[frame["modality"].astype(str).eq(str(modality))].copy()
    if frame.empty:
        raise ValueError("Audio input manifest contains no selected rows")
    if frame[identity_column].astype(str).duplicated().any():
        raise ValueError("Audio input manifest contains duplicate recording identities")

    def contained(path: Path) -> Path | None:
        try:
            resolved = path.resolve()
            resolved.relative_to(project_root)
            return resolved
        except (ValueError, Exception):
            return None

    def archive_member_digest(archive: Path, member: str) -> tuple[str, int]:
        try:
            before = archive.stat()
            digest = hashlib.sha256()
            with zipfile.ZipFile(archive) as handle:
                size = int(handle.getinfo(member).file_size)
                with handle.open(member, "r") as source:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
            after = archive.stat()
            return digest.hexdigest(), size
        except Exception:
            return hashlib.sha256(f"{archive}::{member}".encode()).hexdigest(), 0

    records: list[dict[str, object]] = []
    for row in frame.sort_values(identity_column, kind="mergesort").to_dict(
        orient="records"
    ):
        supplied = str(row["audio_path"])
        member: str | None = None
        if "::" in supplied:
            archive_text, member_text = supplied.split("::", 1)
            archive = Path(archive_text)
            if not archive.is_absolute():
                archive = metadata_path.parent / archive
            source_path = contained(archive) or archive
            member = member_text
            content_hash, size_bytes = archive_member_digest(source_path, member)
            locator = source_path.as_posix()
        else:
            source_path = Path(supplied)
            if not source_path.is_absolute():
                source_path = metadata_path.parent / source_path
            resolved_p = contained(source_path) or source_path
            if resolved_p.is_file():
                content_hash = stable_file_sha256(resolved_p)
                size_bytes = int(resolved_p.stat().st_size)
                locator = resolved_p.as_posix()
            else:
                content_hash = hashlib.sha256(supplied.encode("utf-8")).hexdigest()
                size_bytes = 0
                locator = supplied
        records.append(
            {
                "recording_identity": str(row[identity_column]),
                "modality": str(row.get("modality", modality or "")),
                "source_locator": locator,
                "archive_member": member,
                "size_bytes": size_bytes,
                "sha256": content_hash,
            }
        )
    return records


def audio_input_manifest_sha256(
    metadata_path: Path,
    *,
    project_root: Path,
    modality: str | None = None,
) -> str:
    """Hash every declared source-audio byte stream used by a scientific run."""
    return canonical_json_sha256(
        audio_input_manifest_records(
            metadata_path,
            project_root=project_root,
            modality=modality,
        )
    )


def capture_live_pip_freeze() -> tuple[list[str], str]:
    """Capture the exact interpreter environment used to derive run identity."""

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Unable to capture the executable Python environment") from exc
    lines = completed.stdout.splitlines()
    return lines, canonical_json_sha256(lines)


def load_controller_config(
    *,
    config_path: Path,
    project_root: Path,
    mode: str,
    device: str,
    accepted_freezes_path: Path,
    expected_run_id: str = "auto",
) -> HSTPipelineConfig:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    scientific_config = _read_scientific_config(config_path)
    accepted_document = _read_optional_json(Path(accepted_freezes_path).resolve())
    accepted_hashes = _accepted_hash_mapping(accepted_document)
    if mode == "full":
        if accepted_document:
            status = str(accepted_document.get("approval_status", ""))
            reviewer = str(accepted_document.get("approved_by", "")).strip()
            approved_at = str(accepted_document.get("approved_at_utc", "")).strip()
            if status != "manually_approved" or not reviewer or not approved_at:
                raise ValueError(
                    "Full mode requires explicit manual approval metadata; "
                    "a review-only candidate cannot authorize execution"
                )
            try:
                timestamp = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("Manual approval timestamp is not valid ISO-8601") from exc
            if timestamp.tzinfo is None:
                raise ValueError("Manual approval timestamp must include a timezone")
        missing = sorted(REQUIRED_FULL_FREEZES - set(accepted_hashes))
        if missing:
            raise ValueError(f"Missing accepted freeze hashes: {missing}")

    requirements = project_root / "requirements-hst.txt"
    if not requirements.is_file():
        raise FileNotFoundError(requirements)
    source_paths = _controller_source_paths(project_root)
    source = scientific_config.get("source", {})
    if not isinstance(source, Mapping):
        raise ValueError("Scientific configuration source section must be an object")
    checkpoints = scientific_config.get("checkpoints", {})
    if not isinstance(checkpoints, Mapping):
        raise ValueError("Scientific configuration checkpoints section must be an object")
    checkpoint_hashes: dict[str, str] = {}
    for name, specification in checkpoints.items():
        if not isinstance(specification, Mapping) or not _valid_sha256(specification.get("sha256")):
            raise ValueError(f"Checkpoint {name!r} has no valid SHA-256")
        checkpoint_hashes[str(name)] = str(specification["sha256"])

    dependency_hash = stable_file_sha256(requirements)
    input_hashes = {
        "scientific_configuration": canonical_json_sha256(scientific_config),
    }
    paths_config = scientific_config.get("paths", {})
    if isinstance(paths_config, Mapping) and mode in {"pilot", "full"}:
        selected_metadata = [
            (
                "coswara",
                project_root
                / str(
                    paths_config.get(
                        "coswara_metadata",
                        "data/processed/metadata_with_quality.csv",
                    )
                ),
                "cough" if mode == "pilot" else None,
            )
        ]
        selected_metadata.append(
            (
                "coughvid",
                project_root
                / str(
                    paths_config.get(
                        "coughvid_metadata",
                        "data/processed/coughvid_metadata_compare_is10_external.csv",
                    )
                ),
                "cough",
            )
        )
        for dataset_name, raw_path, modality_filter in selected_metadata:
            metadata_path = raw_path.resolve()
            if not metadata_path.is_file():
                continue
            input_hashes[f"{dataset_name}_metadata"] = stable_file_sha256(
                metadata_path
            )
            columns = pd.read_csv(metadata_path, nrows=0).columns
            if "audio_path" in columns:
                input_hashes[f"{dataset_name}_audio_content"] = (
                    audio_input_manifest_sha256(
                        metadata_path,
                        project_root=project_root,
                        modality=modality_filter,
                    )
                )
        for input_name in (
            "coughvid_cohort_metadata",
            "coughvid_raw_metadata",
        ):
            supplied = paths_config.get(input_name)
            if supplied is None:
                continue
            input_path = (project_root / str(supplied)).resolve()
            if input_path.is_file():
                input_hashes[input_name] = stable_file_sha256(input_path)
    configured_inputs = accepted_document.get("input_hashes", {})
    if isinstance(configured_inputs, Mapping):
        for key, value in configured_inputs.items():
            normalized_key = str(key)
            normalized_value = str(value)
            if (
                normalized_key in input_hashes
                and input_hashes[normalized_key] != normalized_value
            ):
                raise ValueError(
                    f"Accepted input hash disagrees with live content: {normalized_key}"
                )
            input_hashes[normalized_key] = normalized_value
    manifest_hashes = accepted_document.get("manifest_hashes", {})
    if not isinstance(manifest_hashes, Mapping):
        raise ValueError("manifest_hashes must be a mapping")
    if mode in {"pilot", "full"}:
        _pip_freeze, pip_freeze_hash = capture_live_pip_freeze()
        if mode == "full" and accepted_hashes.get("environment_lock") != pip_freeze_hash:
            raise ValueError(
                "The live Python environment does not match the manually accepted lock"
            )
    else:
        pip_freeze_hash = dependency_hash
    return HSTPipelineConfig(
        workspace_root=project_root,
        mode=mode,
        scientific_config=scientific_config,
        source_root=project_root,
        source_paths=source_paths,
        dependency_lock_path=requirements,
        hst_commit=str(source.get("commit", "")),
        checkpoint_hashes=checkpoint_hashes,
        input_hashes=input_hashes,
        manifest_hashes={str(key): str(value) for key, value in manifest_hashes.items()},
        accepted_hashes=accepted_hashes,
        pip_freeze_hash=pip_freeze_hash,
        expected_run_id=expected_run_id,
        device=device,
    )


def run_preflight(
    *,
    config_path: Path,
    project_root: Path,
    mode: str,
    device: str,
    accepted_freezes_path: Path,
) -> dict[str, object]:
    checks: dict[str, str] = {}
    errors: list[str] = []
    pipeline: HSTPipeline | None = None
    try:
        config = load_controller_config(
            config_path=config_path,
            project_root=project_root,
            mode=mode,
            device=device,
            accepted_freezes_path=accepted_freezes_path,
        )
        checks["configuration"] = "ok"
        pipeline = pipeline_class_for_config(config.scientific_config)(config)
        checks["content_addressed_run"] = "ok"
    except Exception as exc:
        errors.append(f"configuration: {type(exc).__name__}: {exc}")

    if mode in {"pilot", "full"}:
        project_root = Path(project_root).resolve()
        scientific_config = (
            pipeline.config.scientific_config
            if pipeline is not None
            else _read_scientific_config(Path(config_path).resolve())
        )
        source = scientific_config.get("source", {})
        paths = scientific_config.get("paths", {})
        if not isinstance(source, Mapping):
            source = {}
        if not isinstance(paths, Mapping):
            paths = {}
        workspace_root = project_root.resolve()
        hst_root = (project_root / str(source.get("path", "HST"))).resolve()
        try:
            hst_root.relative_to(workspace_root)
        except ValueError:
            errors.append(f"configured HST source escapes repository: {hst_root}")
        checkpoint_root = (
            project_root
            / str(paths.get("checkpoint_directory", ".cache/hst/checkpoints"))
        ).resolve()
        metadata_path = (
            project_root
            / str(paths.get("coswara_metadata", "data/processed/metadata_with_quality.csv"))
        ).resolve()

        # Auto-download checkpoints if missing
        try:
            prepare_hst_prerequisites(config_path=config_path, project_root=project_root)
        except Exception as prep_exc:
            pass

        coughvid_meta_path = (
            project_root
            / str(
                paths.get(
                    "coughvid_metadata",
                    "data/processed/coughvid_metadata_hst_external.csv",
                )
            )
        ).resolve()

        required_paths = [
            hst_root / "model" / "hst_model.py",
            checkpoint_root / "hst_small_imagenet.pth",
            checkpoint_root / "hst_base_imagenet.pth",
            metadata_path,
            coughvid_meta_path,
        ]
        cohort_setting = paths.get("coughvid_cohort_metadata")
        raw_metadata_setting = paths.get("coughvid_raw_metadata")
        if not coughvid_meta_path.is_file() and cohort_setting is not None and raw_metadata_setting is not None:
            required_paths.extend(
                [
                    (project_root / str(cohort_setting)).resolve(),
                    (project_root / str(raw_metadata_setting)).resolve(),
                ]
            )
        if mode == "full":
            comparator_path = (
                project_root
                / str(
                    paths.get(
                        "compare_is10_features",
                        "data/processed/features_compare_is10_merged.csv",
                    )
                )
            ).resolve()
            required_paths.extend(
                [
                    comparator_path,
                ]
            )
        missing_paths = [str(path) for path in required_paths if not path.is_file()]
        if missing_paths:
            errors.append(f"required inputs missing: {missing_paths}")
        else:
            checks["required_inputs"] = "ok"
        if device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("torch.cuda.is_available() is false")
                checks["cuda"] = str(torch.cuda.get_device_name(0))
            except Exception as exc:
                errors.append(f"cuda: {type(exc).__name__}: {exc}")

    return {
        "schema_version": 1,
        "status": "ready" if not errors else "blocked",
        "mode": mode,
        "device": device,
        "run_id": pipeline.run_id if pipeline is not None else None,
        "checks": checks,
        "errors": errors,
    }


def prepare_hst_prerequisites(
    *,
    config_path: Path,
    project_root: Path,
) -> dict[str, object]:
    """Initialize pinned HST source and verified checkpoints for notebook execution."""
    from .hst_checkpoint import (
        download_verified_checkpoint,
        verify_hst_source,
    )

    project_root = Path(project_root).resolve()
    scientific_config = _read_scientific_config(Path(config_path).resolve())
    source = scientific_config.get("source", {})
    paths = scientific_config.get("paths", {})
    checkpoints = scientific_config.get("checkpoints", {})
    if not isinstance(source, Mapping) or not isinstance(paths, Mapping):
        raise ValueError("HST source and paths configuration must be objects")
    if not isinstance(checkpoints, Mapping) or not checkpoints:
        raise ValueError("HST checkpoint configuration is missing")
    repository_root = project_root.resolve()
    hst_root = (project_root / str(source.get("path", "HST"))).resolve()
    try:
        hst_relative = hst_root.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("Configured HST source escapes the repository") from exc
    if not (hst_root / "model" / "hst_model.py").is_file():
        completed = subprocess.run(
            [
                "git",
                "submodule",
                "update",
                "--init",
                "--recursive",
                "--",
                hst_relative.as_posix(),
            ],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Unable to initialize pinned HST submodule: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
    verified_commit = verify_hst_source(hst_root, str(source.get("commit", "")))
    checkpoint_root = (
        project_root
        / str(paths.get("checkpoint_directory", ".cache/hst/checkpoints"))
    ).resolve()
    try:
        checkpoint_root.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("Checkpoint directory escapes the project root") from exc
    rows: list[dict[str, object]] = []
    for name, raw_specification in checkpoints.items():
        if not isinstance(raw_specification, Mapping):
            raise ValueError(f"Checkpoint specification {name!r} must be an object")
        destination = checkpoint_root / str(raw_specification.get("filename", ""))
        resolved = download_verified_checkpoint(
            google_drive_file_id=str(
                raw_specification.get("google_drive_file_id", "")
            ),
            destination=destination,
            expected_size=int(raw_specification.get("size_bytes", -1)),
            expected_sha256=str(raw_specification.get("sha256", "")),
        )
        rows.append(
            {
                "checkpoint": str(name),
                "path": resolved.as_posix(),
                "size_bytes": resolved.stat().st_size,
                "sha256": stable_file_sha256(resolved),
            }
        )
    from .hst_coughvid_metadata import build_hst_coughvid_metadata

    coughvid_metadata_result: dict[str, object] | None = None
    configured_coughvid = paths.get("coughvid_metadata")
    configured_cohort = paths.get("coughvid_cohort_metadata")
    configured_raw = paths.get("coughvid_raw_metadata")
    configured_binding = (configured_coughvid, configured_cohort, configured_raw)
    if any(value is not None for value in configured_binding) and not all(
        value is not None for value in configured_binding
    ):
        raise ValueError("All HST COUGHVID metadata-binding paths must be configured together")
    if all(value is not None for value in configured_binding):
        coughvid_metadata_result = build_hst_coughvid_metadata(
            cohort_path=project_root / str(configured_cohort),
            raw_metadata_path=project_root / str(configured_raw),
            output_path=project_root / str(configured_coughvid),
        )
    return {
        "schema_version": 1,
        "hst_commit": verified_commit,
        "checkpoints": rows,
        "coughvid_metadata": coughvid_metadata_result,
        "status": "ready",
    }


def _detached_command(
    *,
    project_root: Path,
    config_path: Path,
    accepted_freezes_path: Path,
    mode: str,
    device: str,
    through: str,
    expected_run_id: str,
    launch_id: str,
    resume: bool = True,
    force_stage: Sequence[str] = (),
) -> list[str]:
    command = [
        sys.executable,
        str((Path(project_root) / "scripts" / "72_run_hst_reliability.py").resolve()),
        "--config",
        str(Path(config_path).resolve()),
        "--project-root",
        str(Path(project_root).resolve()),
        "--accepted-freezes",
        str(Path(accepted_freezes_path).resolve()),
        "--mode",
        mode,
        "--device",
        device,
        "--through",
        through,
        "--expected-run-id",
        expected_run_id,
        "--launch-id",
        launch_id,
    ]
    if not resume:
        command.append("--no-resume")
    for stage in force_stage:
        command.extend(("--force-stage", str(stage)))
    return command


def launch_detached_run(
    *,
    config_path: Path,
    project_root: Path,
    mode: str,
    device: str,
    through: str,
    accepted_freezes_path: Path,
    expected_run_id: str = "auto",
    resume: bool = True,
    force_stage: Sequence[str] = (),
) -> dict[str, object]:
    preflight = run_preflight(
        config_path=config_path,
        project_root=project_root,
        mode=mode,
        device=device,
        accepted_freezes_path=accepted_freezes_path,
    )
    if preflight["status"] != "ready":
        raise RuntimeError(f"HST launch blocked by preflight: {preflight['errors']}")
    launch_id = f"launch-{uuid.uuid4().hex}"
    project_root = Path(project_root).resolve()
    launch_root = project_root / "reports" / "hst" / "launches"
    log_root = project_root / "logs" / "hst"
    launch_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    receipt_path = launch_root / f"{launch_id}.json"
    log_path = log_root / f"{launch_id}.log"
    command = _detached_command(
        project_root=project_root,
        config_path=config_path,
        accepted_freezes_path=accepted_freezes_path,
        mode=mode,
        device=device,
        through=through,
        expected_run_id=expected_run_id,
        launch_id=launch_id,
        resume=resume,
        force_stage=force_stage,
    )
    launched_at_unix = time.time()
    run_id = str(preflight["run_id"])
    heartbeat_path = (
        Path("data") / "outputs" / "hst" / run_id / "runtime" / "heartbeat.json"
    ).as_posix()
    launched_at = datetime.fromtimestamp(launched_at_unix, timezone.utc).isoformat()
    intent: dict[str, object] = {
        "schema_version": 2,
        "launch_id": launch_id,
        "status": "initializing",
        "stage": "parent_spawn",
        "run_id": run_id,
        "pid": None,
        "command": command,
        "log_path": str(log_path),
        "host": socket.gethostname(),
        "process_start_identity": None,
        "heartbeat_path": heartbeat_path,
        "launched_at": launched_at,
        "launched_at_unix": launched_at_unix,
        "error": None,
        "updated_at": launched_at,
        "updated_at_unix": launched_at_unix,
        "determinism_environment": {
            "CUBLAS_WORKSPACE_CONFIG": (
                _CUDA_DETERMINISTIC_WORKSPACE if device == "cuda" else None
            )
        },
    }
    atomic_write_json(receipt_path, intent)
    process: subprocess.Popen[bytes] | None = None
    try:
        with log_path.open("ab", buffering=0) as log_handle:
            options: dict[str, object] = {
                "cwd": str(project_root),
                "stdin": subprocess.DEVNULL,
                "stdout": log_handle,
                "stderr": subprocess.STDOUT,
                "env": hst_process_environment(device=device),
            }
            if os.name == "posix":
                options["start_new_session"] = True
            else:
                options["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                )
            process = subprocess.Popen(command, **options)  # type: ignore[arg-type]
        identity = capture_process_identity(process.pid)
        if identity.pid != process.pid:
            raise RuntimeError("Detached child identity does not match the spawned PID")
    except BaseException as exc:
        failed_at_unix = time.time()
        failed = {
            **intent,
            "status": "failed",
            "pid": process.pid if process is not None else None,
            "error": f"{type(exc).__name__}: {exc}",
            "updated_at": datetime.fromtimestamp(
                failed_at_unix, timezone.utc
            ).isoformat(),
            "updated_at_unix": failed_at_unix,
            "finished_at": datetime.fromtimestamp(
                failed_at_unix, timezone.utc
            ).isoformat(),
            "finished_at_unix": failed_at_unix,
        }
        atomic_write_json(receipt_path, failed)
        raise
    receipt = {
        **intent,
        "status": "launching",
        "stage": "preflight",
        "pid": process.pid,
        "host": identity.host,
        "process_start_identity": identity.start_identity,
    }
    atomic_write_json(receipt_path, receipt)
    return receipt


_STATUS_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def wait_for_parent_launch_initialization(
    *,
    project_root: Path,
    launch_id: str,
    child_identity: ProcessIdentity | None = None,
    timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.05,
    _monotonic: Callable[[], float] = time.monotonic,
    _sleep: Callable[[float], object] = time.sleep,
) -> dict[str, object]:
    """Wait until the parent durably binds a detached launch to this child."""

    if not _STATUS_ID.fullmatch(str(launch_id)):
        raise ValueError("launch_id contains unsafe characters")
    if timeout_seconds <= 0 or poll_interval_seconds <= 0:
        raise ValueError("launch initialization timeout and poll interval must be positive")
    project_root = Path(project_root).resolve()
    path = project_root / "reports" / "hst" / "launches" / f"{launch_id}.json"
    identity = child_identity or capture_process_identity()
    deadline = _monotonic() + float(timeout_seconds)
    while True:
        if path.is_file():
            receipt = read_json(path)
            if not isinstance(receipt, dict) or receipt.get("launch_id") != launch_id:
                raise ValueError("Detached parent launch receipt is malformed")
            status = receipt.get("status")
            if status in _DETACHED_TERMINAL_STATUSES:
                raise RuntimeError(
                    "Detached parent failed before child initialization completed: "
                    f"{receipt.get('error') or status}"
                )
            if status == "launching":
                recorded_identity = (
                    receipt.get("pid"),
                    receipt.get("host"),
                    receipt.get("process_start_identity"),
                )
                expected_identity = (
                    identity.pid,
                    identity.host,
                    identity.start_identity,
                )
                if recorded_identity != expected_identity:
                    raise ValueError(
                        "Detached parent receipt has a different child process identity"
                    )
                return receipt
            if status != "initializing":
                raise ValueError(
                    f"Detached parent launch receipt has unexpected status {status!r}"
                )
        if _monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for detached parent launch {launch_id!r}"
            )
        _sleep(poll_interval_seconds)


def _portable_heartbeat_path(status_path: Path, supplied: Path | str) -> str:
    status_path = Path(status_path).resolve()
    try:
        project_root = status_path.parents[3]
    except IndexError as exc:
        raise ValueError("Launch status path cannot identify project_root") from exc
    candidate = Path(supplied)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (project_root / candidate).resolve()
    )
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise ValueError("Detached-launch heartbeat_path escapes project_root") from exc


def update_detached_run_status(
    path: Path,
    *,
    launch_id: str,
    run_id: str | None,
    status: str,
    stage: str | None,
    error: str | None = None,
    heartbeat_path: Path | str | None = None,
    timestamp: float | None = None,
) -> dict[str, object]:
    """Update mutable run state without replacing detached-launch identity."""

    if status not in _DETACHED_STATUSES:
        raise ValueError(f"Unknown detached-run status: {status!r}")
    path = Path(path)
    now = time.time() if timestamp is None else float(timestamp)
    if not math.isfinite(now):
        raise ValueError("timestamp must be finite")
    if path.is_file():
        receipt = read_json(path)
        if receipt.get("launch_id") != launch_id:
            raise ValueError("Cannot update a different detached launch")
        stored_run_id = receipt.get("run_id")
        if (
            stored_run_id is not None
            and run_id is not None
            and stored_run_id != run_id
        ):
            raise ValueError("Cannot update a detached launch with a different run_id")
        if receipt.get("status") in _DETACHED_TERMINAL_STATUSES:
            raise ValueError("Cannot update a terminal detached launch")
    else:
        identity = capture_process_identity()
        launched_at = datetime.fromtimestamp(now, timezone.utc).isoformat()
        receipt = {
            "schema_version": 2,
            "launch_id": launch_id,
            "run_id": run_id,
            "pid": identity.pid,
            "command": [sys.executable, *sys.argv],
            "log_path": None,
            "host": identity.host,
            "process_start_identity": identity.start_identity,
            "heartbeat_path": (
                _portable_heartbeat_path(path, heartbeat_path)
                if heartbeat_path is not None
                else None
            ),
            "launched_at": launched_at,
            "launched_at_unix": now,
        }
    if heartbeat_path is not None:
        supplied_heartbeat = _portable_heartbeat_path(path, heartbeat_path)
        existing_heartbeat = receipt.get("heartbeat_path")
        if existing_heartbeat not in {None, supplied_heartbeat}:
            raise ValueError("Cannot change detached-launch heartbeat_path")
        receipt["heartbeat_path"] = supplied_heartbeat
    receipt.update(
        {
            "schema_version": max(int(receipt.get("schema_version", 1)), 2),
            "run_id": run_id if run_id is not None else receipt.get("run_id"),
            "status": status,
            "stage": stage,
            "error": error,
            "updated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "updated_at_unix": now,
        }
    )
    if status in _DETACHED_TERMINAL_STATUSES:
        receipt["finished_at"] = receipt["updated_at"]
        receipt["finished_at_unix"] = now
    for field in _IMMUTABLE_LAUNCH_FIELDS:
        if field not in receipt:
            raise ValueError(f"Detached launch receipt is missing immutable field {field!r}")
    atomic_write_json(path, receipt)
    return receipt


def _status_timestamp(payload: Mapping[str, object]) -> float | None:
    supplied = payload.get("updated_at_unix")
    if isinstance(supplied, (int, float)) and math.isfinite(float(supplied)):
        return float(supplied)
    iso_value = payload.get("updated_at")
    if not isinstance(iso_value, str):
        return None
    try:
        parsed = datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.timestamp()


def _resolve_monitored_heartbeat(
    project_root: Path,
    payload: Mapping[str, object],
) -> tuple[Path | None, float | None]:
    supplied = payload.get("heartbeat_path")
    if not isinstance(supplied, str) or not supplied.strip():
        return None, None
    candidate = Path(supplied)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("Launch heartbeat path escapes project_root") from exc
    if not resolved.is_file():
        return resolved, None
    heartbeat = read_json(resolved)
    if heartbeat.get("run_id") != payload.get("run_id"):
        raise ValueError("Launch heartbeat belongs to a different run_id")
    heartbeat_identity = (
        heartbeat.get("pid"),
        heartbeat.get("host"),
        heartbeat.get("process_start_identity"),
    )
    launch_identity = (
        payload.get("pid"),
        payload.get("host"),
        payload.get("process_start_identity"),
    )
    if heartbeat_identity != launch_identity:
        raise ValueError("Launch heartbeat belongs to a different process identity")
    supplied_at = heartbeat.get("heartbeat_at_unix")
    if not isinstance(supplied_at, (int, float)) or not math.isfinite(float(supplied_at)):
        raise ValueError("Launch heartbeat has an invalid timestamp")
    return resolved, float(supplied_at)


def read_run_status(
    *,
    project_root: Path,
    status_id: str,
    stale_after_seconds: float = DEFAULT_DETACHED_STALE_AFTER_SECONDS,
    now: float | None = None,
) -> dict[str, object]:
    if not _STATUS_ID.fullmatch(str(status_id)):
        raise ValueError("status_id contains unsafe characters")
    if stale_after_seconds <= 0:
        raise ValueError("stale_after_seconds must be positive")
    project_root = Path(project_root).resolve()
    path = project_root / "reports" / "hst" / "launches" / f"{status_id}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("launch_id") != status_id:
        raise ValueError("Launch status receipt is malformed or has the wrong status_id")
    stored_status = payload.get("status")
    if stored_status not in _DETACHED_STATUSES:
        raise ValueError("Launch status receipt has an unknown status")
    if stored_status in _DETACHED_TERMINAL_STATUSES:
        return payload

    observed_at = time.time() if now is None else float(now)
    if not math.isfinite(observed_at):
        raise ValueError("now must be finite")
    status_at = _status_timestamp(payload)
    _, heartbeat_at = _resolve_monitored_heartbeat(project_root, payload)
    freshest_at = (
        max(timestamp for timestamp in (status_at, heartbeat_at) if timestamp is not None)
        if status_at is not None or heartbeat_at is not None
        else None
    )

    liveness = ProcessLiveness.UNKNOWN
    try:
        identity = ProcessIdentity.from_record(payload)
    except (RuntimeError, TypeError, ValueError):
        identity = None
    if identity is not None:
        liveness = process_identity_liveness(identity)

    monitor_status: str | None = None
    monitor_error: str | None = None
    if liveness is ProcessLiveness.DEAD:
        monitor_status = "dead"
        monitor_error = (
            "Detached process is no longer alive before a terminal status was written"
        )
    elif freshest_at is not None and observed_at - freshest_at > stale_after_seconds:
        monitor_status = "stale"
        monitor_error = (
            "Detached process heartbeat/status is stale beyond "
            f"{stale_after_seconds:.1f} seconds"
        )
    if monitor_status is not None:
        monitored = dict(payload)
        monitored.update(
            {
                "stored_status": stored_status,
                "status": "failed",
                "monitor_status": monitor_status,
                "process_liveness": liveness.value,
                "error": monitor_error,
                "detected_at_unix": observed_at,
            }
        )
        return monitored
    return payload


_CONFIRMATORY_TRAINING_JOB_BUDGETS = {
    "internal_cv": 40,
    "split_policy_contrast": 8,
    "reverse_temporal": 2,
}


def _read_self_hashed_runtime_record(
    path: Path,
    *,
    receipt_type: str,
    run_id: str,
) -> dict[str, object]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    payload = read_json(path)
    if payload.get("receipt_type") != receipt_type or payload.get("run_id") != run_id:
        raise ValueError(f"Runtime record has the wrong identity: {path}")
    record_hash = payload.get("record_hash")
    unsigned = dict(payload)
    unsigned.pop("record_hash", None)
    if (
        not isinstance(record_hash, str)
        or len(record_hash) != SHA256_LENGTH
        or canonical_json_sha256(unsigned) != record_hash
    ):
        raise ValueError(f"Runtime record self-hash is invalid: {path}")
    return payload


def _validated_progress_output_path(*, root: Path, relative: object) -> Path:
    base = Path(root).resolve()
    supplied = base / str(relative)
    if supplied.is_symlink():
        raise ValueError("Progress output cannot be a symbolic link")
    candidate = supplied.resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError("Progress output escapes its scientific root") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"Progress output is missing: {candidate}")
    return candidate


@lru_cache(maxsize=8192)
def _memoized_progress_file_sha256(
    path_value: str,
    size_bytes: int,
    mtime_ns: int,
    ctime_ns: int,
) -> str:
    del size_bytes, mtime_ns, ctime_ns
    return stable_file_sha256(Path(path_value))


def _progress_file_sha256(path: Path) -> str:
    before = path.stat()
    digest = _memoized_progress_file_sha256(
        str(path),
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after = path.stat()
    if (
        after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ctime_ns != before.st_ctime_ns
    ):
        raise ValueError(f"Progress output changed during checksum verification: {path}")
    return digest


def _validate_stage_progress_outputs(
    receipt: Mapping[str, object],
    *,
    run_root: Path,
) -> None:
    paths = receipt.get("output_paths")
    checksums = receipt.get("output_checksums")
    if not isinstance(paths, list) or not paths or not isinstance(checksums, Mapping):
        raise ValueError("Successful stage progress receipt has no checksummed outputs")
    normalized_paths = [str(value) for value in paths]
    if len(set(normalized_paths)) != len(normalized_paths) or set(
        normalized_paths
    ) != {str(value) for value in checksums}:
        raise ValueError("Stage progress output paths and checksums disagree")
    for relative in normalized_paths:
        expected = checksums.get(relative)
        if not isinstance(expected, str) or len(expected) != SHA256_LENGTH:
            raise ValueError("Stage progress output checksum is invalid")
        candidate = _validated_progress_output_path(root=run_root, relative=relative)
        if _progress_file_sha256(candidate) != expected:
            raise ValueError(f"Stage progress output checksum changed: {candidate}")


def _validate_job_progress_outputs(
    receipt: Mapping[str, object],
    *,
    stage_root: Path,
) -> None:
    outputs = receipt.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Successful job progress receipt has no checksummed outputs")
    seen: set[str] = set()
    for record in outputs:
        if not isinstance(record, Mapping):
            raise ValueError("Scientific job progress output is malformed")
        relative = str(record.get("path", ""))
        if not relative or relative in seen:
            raise ValueError("Scientific job progress output path is invalid or duplicated")
        seen.add(relative)
        expected_size = int(record.get("size_bytes", -1))
        expected_sha256 = record.get("sha256")
        if (
            expected_size < 0
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != SHA256_LENGTH
        ):
            raise ValueError("Scientific job progress output metadata is invalid")
        candidate = _validated_progress_output_path(
            root=stage_root,
            relative=relative,
        )
        if candidate.stat().st_size != expected_size:
            raise ValueError(f"Scientific job progress output size changed: {candidate}")
        if _progress_file_sha256(candidate) != expected_sha256:
            raise ValueError(f"Scientific job progress output checksum changed: {candidate}")


def _read_training_progress_record(
    path: Path,
    *,
    run_id: str,
    stage: str,
    job_id: str,
    job_spec_sha256: str,
) -> dict[str, object]:
    progress = _read_self_hashed_runtime_record(
        path,
        receipt_type="hst_training_progress",
        run_id=run_id,
    )
    if (
        progress.get("stage") != stage
        or progress.get("job_id") != job_id
        or progress.get("job_spec_sha256") != job_spec_sha256
        or progress.get("status") != "checkpointed"
        or progress.get("checkpoint_resume_safe") is not True
    ):
        raise ValueError("Training progress identity or resume contract is invalid")
    training_root = Path(path).resolve().parent
    pointer_path = (
        training_root / str(progress.get("checkpoint_pointer_path", ""))
    ).resolve()
    try:
        pointer_path.relative_to(training_root)
    except ValueError as exc:
        raise ValueError("Training progress checkpoint pointer escapes its job") from exc
    if not pointer_path.is_file() or pointer_path.is_symlink():
        raise FileNotFoundError("Training progress checkpoint pointer is missing")
    pointer_sha256 = stable_file_sha256(pointer_path)
    pointer = read_json(pointer_path)
    if pointer.get("writer") not in {
        "covid_rars.hst_training._atomic_torch_save",
        "covid_audio_btp.hst_training._atomic_torch_save",
    }:
        raise ValueError("Training progress checkpoint pointer has an untrusted writer")
    current = pointer.get("current")
    previous = pointer.get("previous")
    declared = progress.get("checkpoint")
    if not isinstance(current, Mapping) or not isinstance(declared, Mapping):
        raise ValueError("Training progress has no checkpoint generation")
    pointer_matches_progress = pointer_sha256 == progress.get("checkpoint_pointer_sha256")
    if pointer_matches_progress:
        if dict(current) != dict(declared):
            raise ValueError(
                "Training progress does not bind the current checkpoint generation"
            )
    elif not isinstance(previous, Mapping) or dict(previous) != dict(declared):
        raise ValueError(
            "Training progress is neither current nor the retained previous checkpoint"
        )
    checkpoint_path = (
        training_root / str(declared.get("checkpoint_path", ""))
    ).resolve()
    sidecar_path = (training_root / str(declared.get("sidecar_path", ""))).resolve()
    for candidate in (checkpoint_path, sidecar_path):
        try:
            candidate.relative_to(training_root)
        except ValueError as exc:
            raise ValueError("Training progress checkpoint escaped its job") from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise FileNotFoundError("Training progress checkpoint generation is incomplete")
    sidecar = read_json(sidecar_path)
    if sidecar.get("writer") not in {
        "covid_rars.hst_training._atomic_torch_save",
        "covid_audio_btp.hst_training._atomic_torch_save",
    }:
        raise ValueError("Training progress checkpoint sidecar has an untrusted writer")
    if (
        sidecar.get("sha256") != declared.get("sha256")
        or int(sidecar.get("size_bytes", -1)) != int(declared.get("size_bytes", -2))
        or checkpoint_path.stat().st_size != int(declared.get("size_bytes", -1))
    ):
        raise ValueError("Training progress checkpoint metadata is inconsistent")
    if stable_file_sha256(checkpoint_path) != declared.get("sha256"):
        raise ValueError("Training progress checkpoint checksum differs from its bytes")
    completed_epoch = int(progress.get("completed_epoch", -1))
    resume_epoch = int(progress.get("resume_epoch", -1))
    consumed_batches = int(progress.get("next_consumed_batch_index", -1))
    batch_count = int(progress.get("epoch_batch_count", -1))
    max_epochs = int(progress.get("max_epochs", -1))
    if (
        completed_epoch < 0
        or max_epochs <= 0
        or completed_epoch > max_epochs
        or resume_epoch < 1
        or resume_epoch > max_epochs + 1
        or consumed_batches < 0
        or batch_count <= 0
        or consumed_batches > batch_count
    ):
        raise ValueError("Training progress coordinates are invalid")
    return progress


def read_hst_run_progress(
    *,
    project_root: Path,
    run_id: str,
    through: str,
) -> dict[str, object]:
    """Summarize only self-hashed receipts and durable training checkpoints."""

    if not _STATUS_ID.fullmatch(str(run_id)):
        raise ValueError("run_id contains unsafe characters")
    project_root = Path(project_root).resolve()
    run_root = (project_root / "data" / "outputs" / "hst" / run_id).resolve()
    try:
        run_root.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("HST progress run root escapes project_root") from exc
    training_job_budgets = dict(_CONFIRMATORY_TRAINING_JOB_BUDGETS)
    pipeline_stages = HSTPipeline.STAGES
    pilot_freeze_path = run_root / "audits" / "base_resource_pilot_freeze.json"
    if pilot_freeze_path.is_file() and not pilot_freeze_path.is_symlink():
        pilot_freeze = read_json(pilot_freeze_path)
        projection = pilot_freeze.get("runtime_projection")
        if isinstance(projection, Mapping):
            from .hst_workloads import (
                CAPACITY_INTERNAL_FUSION_PROFILE,
                get_hst_workload_profile,
            )

            profile = get_hst_workload_profile(projection.get("workload_profile"))
            supplied_jobs = projection.get("planned_training_jobs_by_modality")
            if not isinstance(supplied_jobs, Mapping) or dict(supplied_jobs) != dict(
                profile.training_jobs_by_modality
            ):
                raise ValueError("Progress runtime projection changed its workload profile")
            training_job_budgets = dict(profile.training_jobs_by_stage)
            if profile.name == CAPACITY_INTERNAL_FUSION_PROFILE:
                pipeline_stages = HSTCapacityInternalFusionPipeline.STAGES

    if through not in pipeline_stages:
        raise ValueError(f"Unknown HST progress target stage: {through!r}")
    target_stages = pipeline_stages[: pipeline_stages.index(through) + 1]
    completed_stages: list[str] = []
    for stage in target_stages:
        receipt_path = run_root / "runtime" / "stages" / f"{stage}.json"
        if not receipt_path.exists():
            continue
        receipt = _read_self_hashed_runtime_record(
            receipt_path,
            receipt_type="hst_stage",
            run_id=run_id,
        )
        if receipt.get("stage") != stage:
            raise ValueError("Stage progress receipt has the wrong stage identity")
        if receipt.get("status") == "success":
            _validate_stage_progress_outputs(receipt, run_root=run_root)
            completed_stages.append(stage)
    completed_count = len(completed_stages)
    stage_total = len(target_stages)

    completed_jobs = 0
    running_receipt_count = 0
    active_candidates: list[tuple[int, float, dict[str, object]]] = []
    current_stage_summary: dict[str, object] | None = None
    for stage, frozen_budget in training_job_budgets.items():
        stage_scientific_root = run_root / "scientific" / stage
        plan_path = stage_scientific_root / "job_plan.csv"
        plan = pd.DataFrame()
        plan_by_id: dict[str, dict[str, object]] = {}
        if plan_path.is_file():
            if plan_path.is_symlink():
                raise ValueError("Training job plan cannot be a symbolic link")
            plan = pd.read_csv(plan_path, low_memory=False)
            required = {"job_id", "job_spec_sha256", "fold", "modality", "protocol"}
            missing = sorted(required - set(plan.columns))
            if missing or plan["job_id"].astype(str).duplicated().any():
                raise ValueError(f"Training job plan is invalid for {stage}: {missing}")
            plan_by_id = {
                str(row["job_id"]): row for row in plan.to_dict(orient="records")
            }
            if len(plan_by_id) != frozen_budget:
                raise ValueError(
                    "Training job plan does not match the exact frozen "
                    f"{stage} budget: {len(plan_by_id)} != {frozen_budget}"
                )
        stage_completed = 0
        stage_active_fraction = 0.0
        jobs_root = stage_scientific_root / "jobs"
        if jobs_root.is_dir():
            for receipt_path in sorted(jobs_root.glob("*/job_receipt.json")):
                receipt = _read_self_hashed_runtime_record(
                    receipt_path,
                    receipt_type="hst_scientific_job",
                    run_id=run_id,
                )
                job_id = str(receipt.get("job_id", ""))
                if receipt_path.parent.name != job_id:
                    raise ValueError("Scientific job receipt path and job identity differ")
                planned = plan_by_id.get(job_id)
                if planned is None:
                    raise ValueError("Scientific job receipt is absent from its frozen plan")
                job_spec_sha256 = str(receipt.get("job_spec_sha256", ""))
                if job_spec_sha256 != str(planned["job_spec_sha256"]):
                    raise ValueError("Scientific job receipt differs from its frozen plan")
                status = str(receipt.get("status", ""))
                job = receipt.get("job")
                if status == "running" and not isinstance(job, Mapping):
                    raise ValueError("Scientific job receipt has no immutable job identity")
                if isinstance(job, Mapping):
                    if (
                        str(job.get("stage", "")) != stage
                        or int(job.get("fold", -1)) != int(planned["fold"])
                        or str(job.get("modality", "")) != str(planned["modality"])
                        or str(job.get("protocol", "")) != str(planned["protocol"])
                    ):
                        raise ValueError(
                            "Scientific job receipt identity differs from its frozen plan"
                        )
                if status == "success":
                    _validate_job_progress_outputs(
                        receipt,
                        stage_root=stage_scientific_root,
                    )
                    completed_jobs += 1
                    stage_completed += 1
                    continue
                if status not in {"running", "stopped", "failed"}:
                    raise ValueError("Scientific job receipt has an unknown status")
                if status == "running":
                    running_receipt_count += 1
                progress_path = receipt_path.parent / "training" / "training_progress.json"
                if not progress_path.is_file():
                    if status == "running":
                        assert isinstance(job, Mapping)
                        current = {
                            "stage": stage,
                            "job_id": job_id,
                            "status": status,
                            "fold": int(job["fold"]),
                            "seed": int(job["seed"]),
                            "modality": str(job["modality"]),
                            "protocol": str(job["protocol"]),
                            "completed_epoch": 0,
                            "resume_epoch": 1,
                            "max_epochs": 100,
                            "next_consumed_batch_index": 0,
                            "epoch_batch_count": 0,
                            "durable_epochs": 0.0,
                            "epoch_percent": 0.0,
                            "checkpointed": False,
                            "checkpoint_resume_safe": False,
                            "updated_at_unix": receipt_path.stat().st_mtime,
                        }
                        active_candidates.append(
                            (1, receipt_path.stat().st_mtime, current)
                        )
                    continue
                progress = _read_training_progress_record(
                    progress_path,
                    run_id=run_id,
                    stage=stage,
                    job_id=job_id,
                    job_spec_sha256=job_spec_sha256,
                )
                if (
                    int(progress.get("fold", -1)) != int(planned["fold"])
                    or str(progress.get("modality", "")) != str(planned["modality"])
                    or str(progress.get("protocol", "")) != str(planned["protocol"])
                    or (
                        isinstance(job, Mapping)
                        and int(progress.get("seed", -1)) != int(job.get("seed", -2))
                    )
                ):
                    raise ValueError(
                        "Training progress identity differs from its frozen plan"
                    )
                completed_epoch = int(progress["completed_epoch"])
                consumed_batches = int(progress["next_consumed_batch_index"])
                batch_count = int(progress["epoch_batch_count"])
                max_epochs = int(progress["max_epochs"])
                durable_epochs = completed_epoch + consumed_batches / batch_count
                durable_fraction = min(1.0, durable_epochs / max_epochs)
                stage_active_fraction = max(stage_active_fraction, durable_fraction)
                current = {
                    "stage": stage,
                    "job_id": job_id,
                    "status": status,
                    "fold": int(progress["fold"]),
                    "seed": int(progress["seed"]),
                    "modality": str(progress["modality"]),
                    "protocol": str(progress["protocol"]),
                    "completed_epoch": completed_epoch,
                    "resume_epoch": int(progress["resume_epoch"]),
                    "max_epochs": max_epochs,
                    "next_consumed_batch_index": consumed_batches,
                    "epoch_batch_count": batch_count,
                    "durable_epochs": durable_epochs,
                    "epoch_percent": 100.0 * durable_epochs / max_epochs,
                    "checkpoint_reason": str(progress["checkpoint_reason"]),
                    "checkpointed": True,
                    "checkpoint_resume_safe": True,
                    "checkpoint_generation": str(progress["checkpoint"]["generation"]),  # type: ignore[index]
                    "checkpoint_path": str(progress["checkpoint"]["checkpoint_path"]),  # type: ignore[index]
                    "checkpoint_sidecar_path": str(progress["checkpoint"]["sidecar_path"]),  # type: ignore[index]
                    "checkpoint_pointer_path": str(progress["checkpoint_pointer_path"]),
                    "checkpoint_sha256": str(progress["checkpoint"]["sha256"]),  # type: ignore[index]
                    "checkpoint_size_bytes": int(progress["checkpoint"]["size_bytes"]),  # type: ignore[index]
                    "updated_at_unix": float(progress["updated_at_unix"]),
                }
                active_candidates.append(
                    (
                        int(status == "running"),
                        float(progress["updated_at_unix"]),
                        current,
                    )
                )
        if plan_path.is_file() and stage not in completed_stages:
            current_stage_summary = {
                "stage": stage,
                "completed_jobs": stage_completed,
                "total_jobs": len(plan),
                "durable_job_equivalents": stage_completed + stage_active_fraction,
                "percent": (
                    100.0 * (stage_completed + stage_active_fraction) / len(plan)
                    if len(plan)
                    else 0.0
                ),
            }

    if running_receipt_count > 1:
        raise ValueError("Multiple confirmatory training jobs claim to be active")
    current_job = (
        max(active_candidates, key=lambda item: (item[0], item[1]))[2]
        if active_candidates
        else None
    )
    current_fraction = 0.0
    if current_job is not None and current_job["status"] != "success":
        current_fraction = float(current_job["epoch_percent"]) / 100.0
    total_jobs = sum(training_job_budgets.values())
    durable_job_equivalents = min(float(total_jobs), completed_jobs + current_fraction)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "through": through,
        "pipeline_stages": {
            "completed": completed_count,
            "total": stage_total,
            "percent": 100.0 * completed_count / stage_total,
        },
        "confirmatory_training": {
            "completed_jobs": completed_jobs,
            "total_jobs": total_jobs,
            "durable_job_equivalents": durable_job_equivalents,
            "percent": 100.0 * durable_job_equivalents / total_jobs,
        },
        "current_training_stage": current_stage_summary,
        "current_job": current_job,
    }


def _detached_command_option(command: object, option: str) -> str | None:
    if not isinstance(command, list) or not all(
        isinstance(token, str) for token in command
    ):
        return None
    positions = [index for index, token in enumerate(command) if token == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        return None
    value = command[positions[0] + 1]
    return value if not value.startswith("--") else None


def _detached_command_values(command: object, option: str) -> tuple[str, ...] | None:
    if not isinstance(command, list) or not all(
        isinstance(token, str) for token in command
    ):
        return None
    positions = [index for index, token in enumerate(command) if token == option]
    values: list[str] = []
    for position in positions:
        if position + 1 >= len(command) or command[position + 1].startswith("--"):
            return None
        values.append(command[position + 1])
    return tuple(values)


def find_resumable_detached_run(
    *,
    project_root: Path,
    run_id: str,
    mode: str,
    device: str,
    through: str,
    expected_run_id: str,
    resume: bool = True,
    force_stage: Sequence[str] = (),
    stale_after_seconds: float = DEFAULT_DETACHED_STALE_AFTER_SECONDS,
    now: float | None = None,
) -> dict[str, object] | None:
    """Find one exact active launch for notebook reattachment, or fail closed."""

    project_root = Path(project_root).resolve()
    launch_root = project_root / "reports" / "hst" / "launches"
    if not launch_root.is_dir():
        return None
    expected_options = {
        "--mode": mode,
        "--device": device,
        "--through": through,
        "--expected-run-id": expected_run_id,
    }
    matches: list[dict[str, object]] = []
    for path in sorted(launch_root.glob("*.json")):
        if not _STATUS_ID.fullmatch(path.stem):
            continue
        preliminary = read_json(path)
        if not isinstance(preliminary, dict) or preliminary.get("run_id") != run_id:
            continue
        command = preliminary.get("command")
        if not isinstance(command, list) or len(command) < 2:
            continue
        if Path(str(command[1])).name != "72_run_hst_reliability.py":
            continue
        if any(
            _detached_command_option(command, option) != value
            for option, value in expected_options.items()
        ):
            continue
        command_has_resume = "--resume" in command
        command_has_no_resume = "--no-resume" in command
        if command_has_resume and command_has_no_resume:
            continue
        command_resume = not command_has_no_resume
        if command_resume != resume:
            continue
        command_force_stages = _detached_command_values(command, "--force-stage")
        if command_force_stages is None or sorted(command_force_stages) != sorted(
            str(stage) for stage in force_stage
        ):
            continue
        status = read_run_status(
            project_root=project_root,
            status_id=path.stem,
            stale_after_seconds=stale_after_seconds,
            now=now,
        )
        if status.get("status") in _DETACHED_NONTERMINAL_STATUSES:
            matches.append(status)
    if len(matches) > 1:
        launch_ids = ", ".join(sorted(str(item["launch_id"]) for item in matches))
        raise RuntimeError(
            "Multiple active detached HST launches match the same frozen command: "
            f"{launch_ids}"
        )
    return matches[0] if matches else None


def wait_for_detached_run(
    *,
    project_root: Path,
    status_id: str,
    poll_interval_seconds: float = 60.0,
    stale_after_seconds: float = DEFAULT_DETACHED_STALE_AFTER_SECONDS,
    timeout_seconds: float = DEFAULT_DETACHED_MAX_WAIT_SECONDS,
    on_poll: Callable[[Mapping[str, object]], object] | None = None,
    _monotonic: Callable[[], float] = time.monotonic,
    _sleep: Callable[[float], object] = time.sleep,
) -> dict[str, object]:
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    started = _monotonic()
    while True:
        status = read_run_status(
            project_root=project_root,
            status_id=status_id,
            stale_after_seconds=stale_after_seconds,
        )
        if on_poll is not None:
            on_poll(status)
        if status.get("status") in _DETACHED_TERMINAL_STATUSES:
            return status
        elapsed = _monotonic() - started
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"Detached HST launch {status_id!r} exceeded the bounded "
                f"polling timeout of {timeout_seconds:.1f} seconds"
            )
        _sleep(min(poll_interval_seconds, timeout_seconds - elapsed))
