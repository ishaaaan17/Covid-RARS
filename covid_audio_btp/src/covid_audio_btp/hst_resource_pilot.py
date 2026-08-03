from __future__ import annotations

import math
import json
import numbers
import hashlib
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import pandas as pd

from covid_audio_btp.hst_runtime import canonical_json_sha256


_REQUIRED_COLUMNS = {
    "physical_batch_size",
    "precision",
    "valid",
    "optimizer_updates",
    "skipped_optimizer_updates",
    "seconds",
    "free_vram_bytes",
    "max_abs_probability_difference_from_fp32",
    "relative_loss_difference_from_fp32",
    "finite_loss",
    "finite_gradients",
    "finite_parameters",
    "finite_predictions",
}
_FORBIDDEN_MODEL_METRICS = {
    "accuracy",
    "auprc",
    "auroc",
    "balanced_accuracy",
    "f1",
    "sensitivity",
    "specificity",
}
_PILOT_FREEZE_FIELDS = (
    "schema_version",
    "physical_batch_size",
    "gradient_accumulation",
    "effective_batch_size",
    "precision",
    "amp",
    "minimum_optimizer_updates",
    "headroom_required_bytes",
    "probability_tolerance",
    "relative_loss_tolerance",
    "selection_basis",
    "model_metrics_used",
    "total_vram_bytes",
    "freeze_context",
)
_RUNTIME_ESTIMATE_BASIS = (
    "selected_cough_fold_optimizer_throughput_times_modality_specific_"
    "all_participant_upper_bound_and_frozen_job_epoch_plan"
)


def runtime_projection_policy_payload(
    *,
    optimizer_updates_per_epoch_by_modality: Mapping[str, int],
    planned_training_jobs_by_modality: Mapping[str, int],
    confirmatory_epochs: int,
    end_to_end_overhead_multiplier: float,
    maximum_approved_runtime_hours: float,
) -> dict[str, object]:
    """Freeze the deterministic workload policy without measured wall-clock time."""
    modalities = set(optimizer_updates_per_epoch_by_modality)
    if not modalities or modalities != set(planned_training_jobs_by_modality):
        raise ValueError(
            "Runtime projection modality workload mappings must be nonempty and aligned"
        )
    if not isinstance(confirmatory_epochs, int) or isinstance(confirmatory_epochs, bool):
        raise ValueError("Runtime projection confirmatory epochs must be a positive integer")
    if confirmatory_epochs <= 0:
        raise ValueError("Runtime projection confirmatory epochs must be positive")
    for name, mapping in (
        ("optimizer updates", optimizer_updates_per_epoch_by_modality),
        ("planned jobs", planned_training_jobs_by_modality),
    ):
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in mapping.values()
        ):
            raise ValueError(f"Runtime projection {name} must contain positive integers")
    for name, value in (
        ("end-to-end overhead multiplier", end_to_end_overhead_multiplier),
        ("maximum approved runtime", maximum_approved_runtime_hours),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"Runtime projection {name} must be finite and positive")
    return {
        "schema_version": 1,
        "estimate_basis": _RUNTIME_ESTIMATE_BASIS,
        "optimizer_updates_per_epoch_by_modality": {
            modality: int(optimizer_updates_per_epoch_by_modality[modality])
            for modality in sorted(modalities)
        },
        "planned_training_jobs_by_modality": {
            modality: int(planned_training_jobs_by_modality[modality])
            for modality in sorted(modalities)
        },
        "planned_training_jobs": int(sum(planned_training_jobs_by_modality.values())),
        "confirmatory_epochs": int(confirmatory_epochs),
        "end_to_end_overhead_multiplier": float(end_to_end_overhead_multiplier),
        "maximum_approved_runtime_hours": float(maximum_approved_runtime_hours),
        "gpu_concurrency": 1,
        "training_time_equation": (
            "sum_m(updates_per_epoch_m*jobs_m)*epochs*seconds_per_update/3600"
        ),
        "end_to_end_equation": "training_only_serial_gpu_hours*overhead_multiplier",
    }


def _canonical_diagnostic_value(value: object) -> object:
    if value is None or value is pd.NA:
        return "__nonfinite__:nan"
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if math.isnan(numeric):
            return "__nonfinite__:nan"
        if math.isinf(numeric):
            return "__nonfinite__:posinf" if numeric > 0 else "__nonfinite__:neginf"
        return numeric
    return value


def canonical_resource_benchmark_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Encode all pilot outcomes, including failed non-finite diagnostics."""
    return [
        {
            str(key): _canonical_diagnostic_value(value)
            for key, value in row.items()
        }
        for row in frame.sort_values(
            ["physical_batch_size", "precision"],
            ascending=[False, True],
        ).to_dict(orient="records")
    ]


def resource_pilot_freeze_payload(selection: Mapping[str, object]) -> dict[str, object]:
    """Return only deterministic fields that authorize confirmatory training."""
    missing = sorted(set(_PILOT_FREEZE_FIELDS) - set(selection))
    if missing:
        raise ValueError(f"Resource-pilot freeze payload is missing fields: {missing}")
    context = selection.get("freeze_context")
    if not isinstance(context, Mapping):
        raise ValueError("Resource-pilot freeze context must be a mapping")
    return {
        field: (
            {str(key): value for key, value in sorted(context.items())}
            if field == "freeze_context"
            else selection[field]
        )
        for field in _PILOT_FREEZE_FIELDS
    }


def project_full_training_runtime(
    *,
    selected_trial_seconds: float,
    selected_trial_optimizer_updates: int,
    optimizer_updates_per_epoch_by_modality: Mapping[str, int],
    planned_training_jobs_by_modality: Mapping[str, int],
    confirmatory_epochs: int,
    end_to_end_overhead_multiplier: float,
    maximum_approved_runtime_hours: float,
) -> dict[str, object]:
    """Project serial GPU training time from the measured pilot throughput."""
    for name, value in (
        ("selected_trial_optimizer_updates", selected_trial_optimizer_updates),
        ("confirmatory_epochs", confirmatory_epochs),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"Runtime projection {name} must be a positive integer")
        if value <= 0:
            raise ValueError(f"Runtime projection {name} must be a positive integer")
    if not math.isfinite(float(selected_trial_seconds)) or selected_trial_seconds <= 0:
        raise ValueError("Runtime projection inputs must be finite and positive")
    policy = runtime_projection_policy_payload(
        optimizer_updates_per_epoch_by_modality=optimizer_updates_per_epoch_by_modality,
        planned_training_jobs_by_modality=planned_training_jobs_by_modality,
        confirmatory_epochs=confirmatory_epochs,
        end_to_end_overhead_multiplier=end_to_end_overhead_multiplier,
        maximum_approved_runtime_hours=maximum_approved_runtime_hours,
    )
    modalities = set(optimizer_updates_per_epoch_by_modality)
    updates_per_epoch = sum(
        int(optimizer_updates_per_epoch_by_modality[modality])
        * int(planned_training_jobs_by_modality[modality])
        for modality in sorted(modalities)
    )
    estimated_updates = updates_per_epoch * int(confirmatory_epochs)
    seconds_per_update = float(selected_trial_seconds) / int(
        selected_trial_optimizer_updates
    )
    training_only_hours = estimated_updates * seconds_per_update / 3600.0
    estimated_hours = training_only_hours * float(end_to_end_overhead_multiplier)
    return {
        "schema_version": 1,
        "estimate_basis": _RUNTIME_ESTIMATE_BASIS,
        "selected_trial_seconds": float(selected_trial_seconds),
        "selected_trial_optimizer_updates": int(selected_trial_optimizer_updates),
        "optimizer_updates_per_epoch_by_modality": {
            modality: int(optimizer_updates_per_epoch_by_modality[modality])
            for modality in sorted(modalities)
        },
        "planned_training_jobs_by_modality": {
            modality: int(planned_training_jobs_by_modality[modality])
            for modality in sorted(modalities)
        },
        "planned_training_jobs": int(sum(planned_training_jobs_by_modality.values())),
        "confirmatory_epochs": int(confirmatory_epochs),
        "estimated_optimizer_updates": int(estimated_updates),
        "training_only_serial_gpu_hours": float(training_only_hours),
        "end_to_end_overhead_multiplier": float(end_to_end_overhead_multiplier),
        "estimated_serial_gpu_hours": float(estimated_hours),
        "maximum_approved_runtime_hours": float(maximum_approved_runtime_hours),
        "within_approved_runtime_ceiling": bool(
            estimated_hours <= maximum_approved_runtime_hours
        ),
        "gpu_concurrency": 1,
        "estimate_is_completion_guarantee": False,
        "estimate_limitations": (
            "Per-update throughput is measured on one Coswara cough fold. The workload "
            "uses all contract-eligible participants as a conservative upper bound per "
            "modality, and the frozen overhead multiplier covers evaluation, checkpoint, "
            "and orchestration cost; thermal variation may still change completion time."
        ),
        "runtime_projection_policy": policy,
        "runtime_projection_policy_sha256": canonical_json_sha256(policy),
    }


def conservative_balanced_optimizer_updates_per_epoch(
    metadata: pd.DataFrame,
    *,
    modalities: tuple[str, ...],
    effective_batch_size: int,
) -> dict[str, int]:
    """Upper-bound balanced-sampler updates using all eligible participants."""
    required = {"participant_key", "label_binary", "modality"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Runtime workload metadata is missing columns: {missing}")
    if effective_batch_size <= 0 or not modalities or len(set(modalities)) != len(modalities):
        raise ValueError("Runtime workload modalities and effective batch must be valid")
    result: dict[str, int] = {}
    for modality in modalities:
        selected = metadata.loc[metadata["modality"].astype(str).eq(modality)].copy()
        if selected.empty:
            raise ValueError(f"Runtime workload has no participants for modality={modality!r}")
        participant_labels = selected.groupby("participant_key")["label_binary"].nunique()
        if participant_labels.ne(1).any():
            raise ValueError("Runtime workload contains conflicting participant labels")
        participants = selected[["participant_key", "label_binary"]].drop_duplicates()
        counts = participants.groupby("label_binary")["participant_key"].nunique()
        if set(counts.index.astype(str)) != {"negative", "positive"}:
            raise ValueError(f"Runtime workload modality={modality!r} requires both classes")
        draws = 2 * int(counts.max())
        result[modality] = int(math.ceil(draws / effective_batch_size))
    return result


def _read_worker_trial_result(
    *,
    result_path: Path,
    completed: object,
    job: Mapping[str, object],
) -> dict[str, object]:
    """Preserve child failure evidence even when the worker exits nonzero."""
    returncode = int(getattr(completed, "returncode", -1))
    stderr = str(getattr(completed, "stderr", "") or "")
    try:
        payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("resource worker result is not an object")
        mismatched = [
            key
            for key, expected in job.items()
            if key in payload and payload[key] != expected
        ]
        if mismatched:
            raise ValueError(f"resource worker result changed job identity: {mismatched}")
        result: dict[str, object] = {**job, **payload}
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        result = {
            **job,
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": "",
        }
    for field in ("physical_batch_size", "precision"):
        result.setdefault(field, job.get(field))
    if returncode != 0:
        result["valid"] = False
        result.setdefault("error", f"WorkerExitError: worker exited {returncode}")
    error = str(result.get("error", ""))
    traceback_text = str(result.pop("traceback", "") or "")
    result["failure_type"] = error.split(":", 1)[0].strip() if error else ""
    result["traceback_sha256"] = (
        hashlib.sha256(traceback_text.encode("utf-8")).hexdigest()
        if traceback_text
        else ""
    )
    result["traceback_tail"] = traceback_text[-4000:]
    result["worker_returncode"] = returncode
    result["worker_stderr_sha256"] = hashlib.sha256(stderr.encode("utf-8")).hexdigest()
    result["worker_stderr_tail"] = stderr[-4000:]
    defaults: dict[str, object] = {
        "valid": False,
        "finite_loss": False,
        "finite_gradients": False,
        "finite_parameters": False,
        "finite_predictions": False,
        "skipped_optimizer_updates": 0,
        "seconds": 0.0,
        "free_vram_bytes": 0,
        "total_vram_bytes": 0,
        "peak_allocated_vram_bytes": 0,
        "peak_reserved_vram_bytes": 0,
        "completed_optimizer_updates": 0,
        "optimizer_updates_per_epoch": 0,
        "evaluation_loss": math.nan,
        "evaluation_probabilities": [],
    }
    for key, value in defaults.items():
        result.setdefault(key, value)
    return result


def _worker_stream_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _worker_exception_result(
    exc: Exception,
    *,
    job: Mapping[str, object],
) -> dict[str, object]:
    """Preserve timeout/subprocess streams while failing a pilot trial closed."""
    traceback_text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    stdout = _worker_stream_text(getattr(exc, "stdout", ""))
    stderr = _worker_stream_text(getattr(exc, "stderr", ""))
    return {
        **job,
        "valid": False,
        "error": f"{type(exc).__name__}: {exc}",
        "failure_type": type(exc).__name__,
        "traceback_sha256": hashlib.sha256(traceback_text.encode("utf-8")).hexdigest(),
        "traceback_tail": traceback_text[-4000:],
        "worker_returncode": -1,
        "worker_stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "worker_stdout_tail": stdout[-4000:],
        "worker_stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "worker_stderr_tail": stderr[-4000:],
        "finite_loss": False,
        "finite_gradients": False,
        "finite_parameters": False,
        "finite_predictions": False,
        "skipped_optimizer_updates": 0,
        "seconds": math.inf,
        "free_vram_bytes": 0,
        "total_vram_bytes": 0,
        "peak_allocated_vram_bytes": 0,
        "peak_reserved_vram_bytes": 0,
        "completed_optimizer_updates": 0,
        "optimizer_updates_per_epoch": 0,
        "evaluation_loss": math.nan,
        "evaluation_probabilities": [],
    }


def select_base_resource_pilot(
    benchmark: pd.DataFrame,
    *,
    total_vram_bytes: int,
    effective_batch_size: int = 8,
    minimum_updates: int = 100,
    probability_tolerance: float = 0.01,
    relative_loss_tolerance: float = 0.01,
    freeze_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Select throughput configuration without consulting any model endpoint."""
    forbidden = sorted(
        column for column in benchmark.columns if str(column).casefold() in _FORBIDDEN_MODEL_METRICS
    )
    if forbidden:
        raise ValueError(f"Resource pilot cannot contain model metrics: {forbidden}")
    missing = sorted(_REQUIRED_COLUMNS - set(benchmark.columns))
    if missing:
        raise ValueError(f"Resource pilot table is missing columns: {missing}")
    if total_vram_bytes <= 0 or effective_batch_size <= 0 or minimum_updates <= 0:
        raise ValueError("Resource pilot sizing arguments must be positive")
    if probability_tolerance < 0 or relative_loss_tolerance < 0:
        raise ValueError("AMP numerical tolerances cannot be negative")

    rows = benchmark.copy()
    rows["physical_batch_size"] = pd.to_numeric(
        rows["physical_batch_size"], errors="raise"
    ).astype(int)
    if rows.duplicated(["physical_batch_size", "precision"]).any():
        raise ValueError("Resource pilot contains duplicate batch/precision trials")
    if not rows["precision"].isin(["fp32", "amp"]).all():
        raise ValueError("Resource pilot precision must be fp32 or amp")
    headroom_required = max(1024**3, int(math.ceil(total_vram_bytes * 0.15)))

    selected_row: pd.Series | None = None
    for batch_size in (8, 4, 2):
        if effective_batch_size % batch_size:
            continue
        candidates = rows.loc[rows["physical_batch_size"].eq(batch_size)].copy()
        if candidates.empty:
            continue
        common_valid = (
            candidates["valid"].astype(bool)
            & pd.to_numeric(candidates["optimizer_updates"], errors="coerce").ge(minimum_updates)
            & pd.to_numeric(candidates["skipped_optimizer_updates"], errors="coerce").eq(0)
            & pd.to_numeric(candidates["free_vram_bytes"], errors="coerce").ge(headroom_required)
            & candidates[
                ["finite_loss", "finite_gradients", "finite_parameters", "finite_predictions"]
            ].astype(bool).all(axis=1)
            & pd.to_numeric(candidates["seconds"], errors="coerce").gt(0)
        )
        amp_valid = (
            candidates["precision"].eq("amp")
            & pd.to_numeric(
                candidates["max_abs_probability_difference_from_fp32"],
                errors="coerce",
            ).le(probability_tolerance)
            & pd.to_numeric(
                candidates["relative_loss_difference_from_fp32"],
                errors="coerce",
            ).le(relative_loss_tolerance)
        )
        precision_valid = candidates["precision"].eq("fp32") | amp_valid
        viable = candidates.loc[common_valid & precision_valid].copy()
        if viable.empty:
            continue
        # Precision selection must not depend on noisy wall-clock timing. AMP is
        # preferred only after it passes the frozen FP32-agreement checks.
        viable["precision_order"] = viable["precision"].map({"amp": 0, "fp32": 1})
        selected_row = viable.sort_values("precision_order", kind="mergesort").iloc[0]
        break

    if selected_row is None:
        raise RuntimeError(
            "No HST-Base resource configuration passed update, finiteness, AMP, and VRAM gates"
        )
    batch_size = int(selected_row["physical_batch_size"])
    selection: dict[str, object] = {
        "schema_version": 2,
        "physical_batch_size": batch_size,
        "gradient_accumulation": effective_batch_size // batch_size,
        "effective_batch_size": effective_batch_size,
        "precision": str(selected_row["precision"]),
        "amp": str(selected_row["precision"]) == "amp",
        "minimum_optimizer_updates": minimum_updates,
        "headroom_required_bytes": headroom_required,
        "measured_free_vram_bytes": int(selected_row["free_vram_bytes"]),
        "selected_trial_seconds": float(selected_row["seconds"]),
        "probability_tolerance": probability_tolerance,
        "relative_loss_tolerance": relative_loss_tolerance,
        "selection_basis": "first_safe_batch_then_amp_if_numerically_valid_else_fp32",
        "model_metrics_used": False,
        "total_vram_bytes": int(total_vram_bytes),
        "freeze_context": {
            str(key): value for key, value in sorted((freeze_context or {}).items())
        },
        "benchmark_sha256": canonical_json_sha256(
            canonical_resource_benchmark_records(rows)
        ),
    }
    selection["pilot_freeze_hash"] = canonical_json_sha256(
        resource_pilot_freeze_payload(selection)
    )
    return selection


def run_base_resource_pilot_trials(
    *,
    cache_index_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    hst_repo: Path,
    worker_script: Path,
    fold: int,
    modality: str,
    seed: int,
    freeze_context: Mapping[str, object] | None = None,
    timeout_seconds: float = 7200.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run every resource trial in a fresh interpreter and freeze the choice."""
    supplied_paths = {
        "cache_index_path": Path(cache_index_path),
        "manifest_path": Path(manifest_path),
        "checkpoint_path": Path(checkpoint_path),
        "hst_repo": Path(hst_repo),
        "worker_script": Path(worker_script),
    }
    for name, path in supplied_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    if timeout_seconds <= 0:
        raise ValueError("Resource-pilot timeout must be positive")

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="hst-base-resource-pilot-") as directory:
        root = Path(directory)
        for batch_size in (8, 4, 2):
            batch_rows: dict[str, dict[str, object]] = {}
            for precision in ("fp32", "amp"):
                job = {
                    "schema_version": 1,
                    "cache_index_path": str(supplied_paths["cache_index_path"].resolve()),
                    "manifest_path": str(supplied_paths["manifest_path"].resolve()),
                    "checkpoint_path": str(supplied_paths["checkpoint_path"].resolve()),
                    "hst_repo": str(supplied_paths["hst_repo"].resolve()),
                    "model_name": "hst_base",
                    "fold": int(fold),
                    "modality": modality,
                    "seed": int(seed),
                    "physical_batch_size": batch_size,
                    "gradient_accumulation": 8 // batch_size,
                    "precision": precision,
                    "optimizer_updates": 100,
                }
                job_path = root / f"batch-{batch_size}-{precision}.job.json"
                result_path = root / f"batch-{batch_size}-{precision}.result.json"
                job_path.write_text(json.dumps(job, sort_keys=True), encoding="utf-8")
                command = [
                    sys.executable,
                    str(supplied_paths["worker_script"].resolve()),
                    "--job-json",
                    str(job_path),
                    "--result-json",
                    str(result_path),
                ]
                try:
                    completed = subprocess.run(
                        command,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=timeout_seconds,
                        cwd=str(supplied_paths["worker_script"].resolve().parents[1]),
                    )
                    result = _read_worker_trial_result(
                        result_path=result_path,
                        completed=completed,
                        job=job,
                    )
                except Exception as exc:
                    result = _worker_exception_result(exc, job=job)
                batch_rows[precision] = result

            fp32 = batch_rows["fp32"]
            fp32_probability = list(fp32.get("evaluation_probabilities", []))
            fp32_loss = float(fp32.get("evaluation_loss", math.nan))
            for precision in ("fp32", "amp"):
                result = batch_rows[precision]
                probability = list(result.pop("evaluation_probabilities", []))
                loss = float(result.get("evaluation_loss", math.nan))
                if precision == "fp32":
                    probability_difference = 0.0
                    relative_loss_difference = 0.0
                elif len(probability) == len(fp32_probability) and probability:
                    probability_difference = max(
                        abs(float(left) - float(right))
                        for left, right in zip(probability, fp32_probability, strict=True)
                    )
                    relative_loss_difference = abs(loss - fp32_loss) / max(abs(fp32_loss), 1e-12)
                else:
                    probability_difference = math.inf
                    relative_loss_difference = math.inf
                result["optimizer_updates"] = int(
                    result.pop("completed_optimizer_updates", result.get("optimizer_updates", 0))
                )
                result["max_abs_probability_difference_from_fp32"] = probability_difference
                result["relative_loss_difference_from_fp32"] = relative_loss_difference
                rows.append(result)

    benchmark = pd.DataFrame(rows)
    total_vram_values = pd.to_numeric(
        benchmark.get("total_vram_bytes", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    if total_vram_values.empty or int(total_vram_values.max()) <= 0:
        raise RuntimeError("Resource pilot did not report a valid CUDA total-memory value")
    selection = select_base_resource_pilot(
        benchmark.drop(columns=["evaluation_loss", "error"], errors="ignore"),
        total_vram_bytes=int(total_vram_values.max()),
        freeze_context=freeze_context,
    )
    return benchmark, selection
