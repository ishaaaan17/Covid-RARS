#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import ExitStack
import json
import os
import sys
import traceback
import uuid
from pathlib import Path
from types import TracebackType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from covid_rars.hst_reliability import (
    HSTPipeline,
    hst_process_environment,
    launch_detached_run,
    load_controller_config,
    pipeline_class_for_config,
    read_run_status,
    run_preflight,
    update_detached_run_status,
    wait_for_parent_launch_initialization,
)
from covid_rars.hst_runtime import (
    HeartbeatEmitter,
    acquire_gpu_execution_lease,
    acquire_run_lock,
    capture_process_identity,
    default_gpu_lease_root,
    default_run_lock_root,
)
from covid_rars.hst_stages import build_scientific_stage_handlers
from covid_rars.hst_evidence import publish_hst_latest
from covid_rars.hst_workloads import (
    CAPACITY_INTERNAL_FUSION_PROFILE,
    workload_profile_from_scientific_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the content-addressed HST reliability pipeline."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/hst_reliability.json"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--accepted-freezes",
        type=Path,
        default=Path("reports/hst/accepted_freezes.json"),
    )
    parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--through", choices=HSTPipeline.STAGES, default="evidence_pack")
    parser.add_argument("--expected-run-id", default="auto")
    parser.add_argument("--force-stage", action="append", default=[])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status-id")
    parser.add_argument("--launch-id")
    return parser.parse_args()


def _absolute(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _write_status(
    path: Path,
    *,
    launch_id: str,
    run_id: str | None,
    status: str,
    stage: str | None,
    error: str | None = None,
    heartbeat_path: Path | None = None,
) -> None:
    update_detached_run_status(
        path,
        launch_id=launch_id,
        run_id=run_id,
        status=status,
        stage=stage,
        error=error,
        heartbeat_path=heartbeat_path,
    )


def _cuda_device_selector(environment: Mapping[str, str]) -> str:
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return "0"
    first = visible.split(",", 1)[0].strip()
    if not first or first == "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not expose a CUDA device")
    return first


def _cuda_uuid() -> str:
    import subprocess

    selector = _cuda_device_selector(os.environ)
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader",
            f"--id={selector}",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if (
        completed.returncode != 0
        or len(values) != 1
        or not values[0].startswith(("GPU-", "MIG-"))
    ):
        raise RuntimeError(
            "Unable to resolve the exclusive CUDA device UUID: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    return values[0]


def main() -> None:
    args = parse_args()
    os.environ.update(hst_process_environment(device=args.device))
    project_root = args.project_root.resolve()
    config_path = _absolute(project_root, args.config)
    accepted_path = _absolute(project_root, args.accepted_freezes)
    if args.status_id:
        print(json.dumps(read_run_status(project_root=project_root, status_id=args.status_id), indent=2))
        return
    if args.detach:
        result = launch_detached_run(
            config_path=config_path,
            project_root=project_root,
            mode=args.mode,
            device=args.device,
            through=args.through,
            accepted_freezes_path=accepted_path,
            expected_run_id=args.expected_run_id,
            resume=bool(args.resume),
            force_stage=tuple(args.force_stage),
        )
        print(json.dumps(result, indent=2))
        return
    if args.launch_id is not None:
        wait_for_parent_launch_initialization(
            project_root=project_root,
            launch_id=args.launch_id,
            child_identity=capture_process_identity(),
        )

    preflight = run_preflight(
        config_path=config_path,
        project_root=project_root,
        mode=args.mode,
        device=args.device,
        accepted_freezes_path=accepted_path,
    )
    if preflight["status"] != "ready":
        raise RuntimeError(f"Preflight blocked the run: {preflight['errors']}")
    config = load_controller_config(
        config_path=config_path,
        project_root=project_root,
        mode=args.mode,
        device=args.device,
        accepted_freezes_path=accepted_path,
        expected_run_id=args.expected_run_id,
    )
    config.resume = bool(args.resume)
    scientific_config = getattr(config, "scientific_config", {})
    pipeline_class = (
        pipeline_class_for_config(scientific_config)
        if isinstance(scientific_config, Mapping) and scientific_config
        else HSTPipeline
    )
    pipeline = pipeline_class(
        config,
        stage_handlers=build_scientific_stage_handlers(config),
    )
    launch_id = args.launch_id or f"launch-{uuid.uuid4().hex}"
    launch_root = project_root / "reports" / "hst" / "launches"
    launch_root.mkdir(parents=True, exist_ok=True)
    status_path = launch_root / f"{launch_id}.json"
    heartbeat_path = pipeline.runtime_root / "heartbeat.json"
    heartbeat = HeartbeatEmitter(
        heartbeat_path,
        run_id=pipeline.run_id,
        interval_seconds=60.0,
    )

    def stage_hook(stage: str) -> None:
        _write_status(
            status_path,
            launch_id=launch_id,
            run_id=pipeline.run_id,
            status="running",
            stage=stage,
            heartbeat_path=heartbeat_path,
        )

    pipeline.stage_hook = stage_hook
    summary = None
    execution_error: BaseException | None = None
    execution_traceback: TracebackType | None = None
    error_text: str | None = None
    heartbeat_started = False
    try:
        _write_status(
            status_path,
            launch_id=launch_id,
            run_id=pipeline.run_id,
            status="running",
            stage="preflight",
            heartbeat_path=heartbeat_path,
        )
        heartbeat_started = True
        heartbeat.start()
        with ExitStack() as stack:
            stack.enter_context(
                acquire_run_lock(
                    pipeline.runtime_root,
                    run_lock_root=default_run_lock_root(),
                    run_id=pipeline.run_id,
                    config_hash=pipeline.configuration_hash,
                )
            )
            if config.device == "cuda":
                stack.enter_context(
                    acquire_gpu_execution_lease(
                        default_gpu_lease_root(),
                        gpu_uuid=_cuda_uuid(),
                        run_id=pipeline.run_id,
                    )
                )
            summary = pipeline.run(
                through=args.through,
                force=set(args.force_stage),
            )
    except BaseException as exc:
        execution_error = exc
        execution_traceback = exc.__traceback__
        error_text = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        heartbeat_failures: list[str] = []
        if heartbeat_started:
            try:
                heartbeat.stop()
            except BaseException as exc:
                heartbeat_failures.append(
                    f"heartbeat.stop failed: {type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
            try:
                heartbeat.raise_if_failed()
            except BaseException as exc:
                heartbeat_failures.append(
                    f"heartbeat.raise_if_failed failed: {type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                if execution_error is None:
                    execution_error = exc
                    execution_traceback = exc.__traceback__
            if heartbeat_failures:
                heartbeat_error = "\n".join(heartbeat_failures)
                error_text = (
                    f"{error_text}\n{heartbeat_error}" if error_text else heartbeat_error
                )
                if execution_error is None:
                    execution_error = RuntimeError("Heartbeat shutdown failed")
                    execution_traceback = execution_error.__traceback__

    if execution_error is None:
        if args.through == "evidence_pack":
            try:
                experiment = (
                    scientific_config.get("experiment")
                    if isinstance(scientific_config, Mapping)
                    else None
                )
                latest_name = "latest.json"
                if isinstance(experiment, Mapping):
                    workload_profile = workload_profile_from_scientific_config(
                        scientific_config
                    )
                    if workload_profile.name == CAPACITY_INTERNAL_FUSION_PROFILE:
                        latest_name = "latest_capacity_internal_fusion.json"
                publish_hst_latest(
                    run_root=pipeline.run_root,
                    evidence_manifest_path=(
                        pipeline.run_root / "evidence" / "hst_evidence_manifest.json"
                    ),
                    latest_path=project_root / "reports" / "hst" / latest_name,
                )
            except BaseException as exc:
                execution_error = exc
                execution_traceback = exc.__traceback__
                error_text = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    if execution_error is not None:
        _write_status(
            status_path,
            launch_id=launch_id,
            run_id=pipeline.run_id,
            status="failed",
            stage=None,
            error=error_text or f"{type(execution_error).__name__}: {execution_error}",
            heartbeat_path=heartbeat_path,
        )
        raise execution_error.with_traceback(execution_traceback)

    if summary is None:
        raise RuntimeError("HST pipeline completed without a summary")
    _write_status(
        status_path,
        launch_id=launch_id,
        run_id=pipeline.run_id,
        status="success",
        stage=str(summary.iloc[-1]["stage"]),
        heartbeat_path=heartbeat_path,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
