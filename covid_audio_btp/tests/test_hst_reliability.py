from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


EXPECTED_STAGES = (
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


def _pipeline(tmp_path: Path, **kwargs):
    from covid_audio_btp.hst_reliability import HSTPipeline, HSTPipelineConfig

    return HSTPipeline(HSTPipelineConfig.smoke(tmp_path, **kwargs))


def test_stage_order_is_frozen() -> None:
    from covid_audio_btp.hst_reliability import HSTPipeline

    assert HSTPipeline.STAGES == EXPECTED_STAGES


def test_completed_stage_with_matching_hash_is_reused(tmp_path: Path) -> None:
    calls: list[str] = []
    pipeline = _pipeline(tmp_path)
    pipeline.stage_hook = calls.append

    first = pipeline.run_stage("preflight")
    second = pipeline.run_stage("preflight")

    assert calls == ["preflight"]
    assert first["reused"] is False
    assert second["reused"] is True
    assert first["fingerprint"] == second["fingerprint"]


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [("run_id", "forged-run"), ("stage", "forged-stage"), ("record_hash", "0" * 64)],
)
def test_forged_stage_receipt_identity_or_self_hash_is_never_reused(
    tmp_path: Path,
    field: str,
    forged_value: str,
) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.run_stage("preflight")
    receipt_path = pipeline.stage_receipt_path("preflight")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[field] = forged_value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    rerun = pipeline.run_stage("preflight")

    assert rerun["reused"] is False
    assert rerun["run_id"] == pipeline.run_id
    assert rerun["stage"] == "preflight"


def test_cpu_stage_records_gpu_memory_as_unmeasured_not_zero(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)

    receipt = pipeline.run_stage("preflight")

    metadata = receipt["metadata"]
    assert metadata["gpu_memory_measured"] is False
    assert metadata["peak_gpu_memory_allocated_mb"] is None
    assert metadata["peak_gpu_memory_reserved_mb"] is None
    assert "peak_gpu_memory_mb" not in metadata


def test_cuda_stage_captures_peak_allocated_and_reserved_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize() -> None:
            calls.append("synchronize")

        @staticmethod
        def reset_peak_memory_stats() -> None:
            calls.append("reset")

        @staticmethod
        def max_memory_allocated() -> int:
            calls.append("allocated")
            return 2 * 1024 * 1024

        @staticmethod
        def max_memory_reserved() -> int:
            calls.append("reserved")
            return 3 * 1024 * 1024

    import covid_audio_btp.hst_reliability as reliability

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda()))
    monkeypatch.setattr(reliability, "_IN_PROCESS_CUDA_STAGES", frozenset({"preflight"}))
    pipeline = _pipeline(tmp_path, device="cuda")

    def handler(pipeline_arg, stage):
        calls.append("handler")
        return pipeline_arg._default_stage_handler(stage)

    pipeline.stage_handlers["preflight"] = handler

    receipt = pipeline.run_stage("preflight")

    assert calls == [
        "synchronize",
        "reset",
        "handler",
        "synchronize",
        "allocated",
        "reserved",
    ]
    metadata = receipt["metadata"]
    assert metadata["gpu_memory_measured"] is True
    assert metadata["peak_gpu_memory_allocated_mb"] == pytest.approx(2.0)
    assert metadata["peak_gpu_memory_reserved_mb"] == pytest.approx(3.0)
    assert metadata["peak_gpu_memory_mb"] == pytest.approx(2.0)


def test_cuda_config_does_not_label_cpu_only_stage_as_gpu_measured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            calls.append("availability")
            return True

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda()))
    pipeline = _pipeline(tmp_path, device="cuda")

    receipt = pipeline.run_stage("preflight")

    assert calls == []
    assert receipt["metadata"]["gpu_memory_measured"] is False
    assert receipt["metadata"]["peak_gpu_memory_allocated_mb"] is None


def test_corrupt_output_invalidates_completed_stage(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    first = pipeline.run_stage("preflight")
    output = pipeline.run_root / first["output_paths"][0]
    output.write_text("corrupt", encoding="utf-8")

    second = pipeline.run_stage("preflight")

    assert second["reused"] is False
    assert second["attempt"] == 2


def test_corrupt_upstream_output_blocks_direct_downstream_reuse(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.run(through="small_smoke")
    upstream = pipeline.run_stage("manifests")
    output = pipeline.run_root / upstream["output_paths"][0]
    output.write_text("corrupt", encoding="utf-8")

    with pytest.raises(Exception, match="checksum|upstream|reuse"):
        pipeline.run_stage("small_smoke")


def test_source_and_dependency_changes_invalidate_stage(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.run_stage("preflight")
    pipeline.config.source_paths[0].write_text("changed\n", encoding="utf-8")
    assert pipeline.run_stage("preflight")["reused"] is False

    pipeline.config.dependency_lock_path.write_text("changed-lock\n", encoding="utf-8")
    assert pipeline.run_stage("preflight")["reused"] is False


def test_forced_upstream_rerun_invalidates_downstream(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.run(through="small_smoke")
    assert pipeline.run_stage("small_smoke")["reused"] is True

    pipeline.run_stage("manifests", force=True)

    assert pipeline.run_stage("small_smoke")["reused"] is False


def test_run_through_executes_only_ordered_prefix(tmp_path: Path) -> None:
    calls: list[str] = []
    pipeline = _pipeline(tmp_path)
    pipeline.stage_hook = calls.append

    summary = pipeline.run(through="checkpoint")

    assert calls == list(EXPECTED_STAGES[:3])
    assert summary["stage"].tolist() == list(EXPECTED_STAGES[:3])
    assert summary["status"].eq("success").all()


def test_failed_stage_is_recorded_and_never_reused(tmp_path: Path) -> None:
    from covid_audio_btp.hst_reliability import StageExecutionError

    pipeline = _pipeline(tmp_path)

    def fail(_pipeline, _stage):
        raise RuntimeError("intentional failure")

    pipeline.stage_handlers["checkpoint"] = fail
    with pytest.raises(StageExecutionError, match="intentional failure"):
        pipeline.run(through="checkpoint")

    receipt = json.loads(pipeline.stage_receipt_path("checkpoint").read_text())
    assert receipt["status"] == "failed"
    assert "intentional failure" in receipt["error"]


def test_full_mode_requires_accepted_freezes(tmp_path: Path) -> None:
    from covid_audio_btp.hst_reliability import HSTPipelineConfig

    with pytest.raises(ValueError, match="accepted"):
        HSTPipelineConfig.full(tmp_path, accepted_hashes={})

    config = HSTPipelineConfig.full(
        tmp_path,
        accepted_hashes={
            "data_contracts_freeze": "a" * 64,
            "pilot_freeze": "b" * 64,
            "environment_lock": "c" * 64,
        },
    )
    assert config.mode == "full"


def test_scientific_modes_fail_closed_without_real_stage_handlers(tmp_path: Path) -> None:
    from covid_audio_btp.hst_reliability import (
        HSTPipeline,
        HSTPipelineConfig,
        StageExecutionError,
    )

    config = HSTPipelineConfig.full(
        tmp_path,
        accepted_hashes={
            "data_contracts_freeze": "a" * 64,
            "pilot_freeze": "b" * 64,
            "environment_lock": "c" * 64,
        },
    )
    pipeline = HSTPipeline(config)
    with pytest.raises(StageExecutionError, match="handler"):
        pipeline.run_stage("preflight")


def test_asserted_run_id_must_match_content_address(tmp_path: Path) -> None:
    from covid_audio_btp.hst_reliability import HSTPipeline, HSTPipelineConfig

    config = HSTPipelineConfig.smoke(tmp_path, expected_run_id="wrong")
    with pytest.raises(ValueError, match="run ID"):
        HSTPipeline(config)


def test_output_paths_cannot_escape_run_root(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)

    def escape(_pipeline, _stage):
        return {"output_paths": [Path("..") / "escaped.txt"]}

    pipeline.stage_handlers["preflight"] = escape
    with pytest.raises(Exception, match="escape|outside"):
        pipeline.run_stage("preflight")
