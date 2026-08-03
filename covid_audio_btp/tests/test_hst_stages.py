from __future__ import annotations

from pathlib import Path
import inspect


def test_scientific_stage_registry_covers_every_pipeline_stage(tmp_path: Path) -> None:
    from covid_audio_btp.hst_reliability import HSTPipeline, HSTPipelineConfig
    from covid_audio_btp.hst_stages import build_scientific_stage_handlers

    accepted = {
        "data_contracts_freeze": "a" * 64,
        "pilot_freeze": "b" * 64,
        "environment_lock": "c" * 64,
    }
    config = HSTPipelineConfig.full(tmp_path, accepted_hashes=accepted, device="cpu")
    handlers = build_scientific_stage_handlers(config)

    assert tuple(handlers) == HSTPipeline.STAGES
    assert all(callable(handler) for handler in handlers.values())
    assert all(
        getattr(handler, "scientific_stage_handler", False)
        for handler in handlers.values()
    )


def test_smoke_mode_uses_real_scientific_handlers_through_small_smoke(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_reliability import HSTPipeline, HSTPipelineConfig
    from covid_audio_btp.hst_stages import build_scientific_stage_handlers

    config = HSTPipelineConfig.smoke(tmp_path)
    handlers = build_scientific_stage_handlers(config)
    expected = HSTPipeline.STAGES[: HSTPipeline.STAGES.index("small_smoke") + 1]

    assert tuple(handlers) == expected
    assert all(callable(handler) for handler in handlers.values())
    assert all(
        getattr(handler, "scientific_stage_handler", False)
        for handler in handlers.values()
    )


def test_controller_injects_scientific_handlers_before_pipeline_execution() -> None:
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "72_run_hst_reliability.py"
    specification = importlib.util.spec_from_file_location("hst_controller_script", script_path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    source = inspect.getsource(module)
    assert "build_scientific_stage_handlers" in source
    assert "stage_handlers=build_scientific_stage_handlers(config)" in source
