from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _prediction_source(name: str, strength: float, n: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(42)
    for split in ("validation", "test"):
        for idx in range(n):
            label = "positive" if idx % 2 else "negative"
            y = 1.0 if label == "positive" else 0.0
            probability = 0.5 + strength * (y - 0.5) + float(rng.normal(0.0, 0.02))
            rows.append(
                {
                    "recording_id": f"{split}_{idx}_{name}",
                    "participant_id": f"{split}_p{idx:03d}",
                    "dataset": "coswara",
                    "label_binary": label,
                    "split": split,
                    "evaluation_protocol": "unit_protocol",
                    "analysis_family": "unit_family",
                    "model_name": name,
                    "modality": "multimodal",
                    "submodality": "",
                    "modality_combination": "unit",
                    "fusion_method": "",
                    "probability": float(np.clip(probability, 0.001, 0.999)),
                    "source_run": name,
                }
            )
    predictions = pd.DataFrame(rows)
    metrics_rows = []
    for split in ("validation", "test"):
        metrics_rows.append(
            {
                "evaluation_protocol": "unit_protocol",
                "analysis_family": "unit_family",
                "model_name": name,
                "modality": "multimodal",
                "submodality": "",
                "modality_combination": "unit",
                "fusion_method": "",
                "metric_split": split,
                "auroc": 0.95 if name == "strong" else 0.65,
                "auprc": 0.95 if name == "strong" else 0.65,
                "skipped": False,
                "source_run": name,
            }
        )
    return pd.DataFrame(metrics_rows), predictions


def test_gated_prediction_stack_rejects_weak_sources_and_reports_test_metrics() -> None:
    from covid_rars.sota_prediction_stack import run_gated_prediction_stack

    strong_metrics, strong_predictions = _prediction_source("strong", strength=0.8)
    weak_metrics, weak_predictions = _prediction_source("weak_wavlm", strength=0.2)
    result = run_gated_prediction_stack(
        pd.concat([strong_metrics, weak_metrics], ignore_index=True),
        pd.concat([strong_predictions, weak_predictions], ignore_index=True),
        top_k=4,
        max_validation_drop=0.05,
        min_sources=1,
    )

    assert not result.metrics.empty
    assert not result.predictions.empty
    assert {"validation", "test"}.issubset(set(result.metrics["metric_split"]))
    assert result.candidates[result.candidates["model_name"].eq("weak_wavlm")]["selected"].eq(False).all()
    assert result.candidates[result.candidates["model_name"].eq("strong")]["selected"].eq(True).all()
    assert result.metrics["analysis_family"].eq("sota_gated_prediction_stack").all()


def test_gated_prediction_stack_rejects_metric_sources_missing_predictions() -> None:
    from covid_rars.sota_prediction_stack import run_gated_prediction_stack

    strong_metrics, strong_predictions = _prediction_source("strong", strength=0.8)
    missing_metrics, _ = _prediction_source("missing_predictions", strength=0.7)
    result = run_gated_prediction_stack(
        pd.concat([strong_metrics, missing_metrics], ignore_index=True),
        strong_predictions,
        top_k=4,
        max_validation_drop=0.50,
        min_sources=1,
    )

    missing = result.candidates[result.candidates["model_name"].eq("missing_predictions")]
    assert not result.metrics.empty
    assert not result.predictions.empty
    assert missing["selected"].eq(False).all()
    assert missing["reject_reason"].eq("missing_predictions").all()


def test_gated_prediction_stack_selects_next_available_source_when_top_candidate_has_no_predictions() -> None:
    from covid_rars.sota_prediction_stack import run_gated_prediction_stack

    strong_metrics, strong_predictions = _prediction_source("strong", strength=0.8)
    missing_metrics, _ = _prediction_source("missing_predictions", strength=0.7)
    missing_metrics["auroc"] = 0.99
    result = run_gated_prediction_stack(
        pd.concat([missing_metrics, strong_metrics], ignore_index=True),
        strong_predictions,
        top_k=1,
        max_validation_drop=0.50,
        min_sources=1,
    )

    strong = result.candidates[result.candidates["model_name"].eq("strong")]
    missing = result.candidates[result.candidates["model_name"].eq("missing_predictions")]
    assert not result.metrics.empty
    assert not result.predictions.empty
    assert strong["selected"].eq(True).all()
    assert strong["available_validation_rank"].eq(1).all()
    assert missing["selected"].eq(False).all()
    assert missing["reject_reason"].eq("missing_predictions").all()


def test_gated_prediction_stack_matches_recording_submodalities_to_participant_metric_source() -> None:
    from covid_rars.sota_prediction_stack import run_gated_prediction_stack

    metrics, predictions = _prediction_source("strong", strength=0.8)
    predictions.loc[predictions.index[::2], "submodality"] = "heavy_cough"
    predictions.loc[predictions.index[1::2], "submodality"] = "shallow_cough"
    metrics["submodality"] = ""

    result = run_gated_prediction_stack(metrics, predictions, top_k=1, max_validation_drop=0.01, min_sources=1)

    assert not result.metrics.empty
    assert not result.predictions.empty
    assert result.candidates["has_predictions"].eq(True).all()
    assert result.candidates["selected"].eq(True).all()


def test_sota_prediction_stack_cli_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    strong_metrics, strong_predictions = _prediction_source("strong", strength=0.8)
    weak_metrics, weak_predictions = _prediction_source("weak_wavlm", strength=0.2)
    strong_metric_path = tmp_path / "strong_metrics.csv"
    strong_prediction_path = tmp_path / "strong_predictions.csv"
    weak_metric_path = tmp_path / "weak_metrics.csv"
    weak_prediction_path = tmp_path / "weak_predictions.csv"
    strong_metrics.drop(columns=["source_run"]).to_csv(strong_metric_path, index=False)
    strong_predictions.drop(columns=["source_run"]).to_csv(strong_prediction_path, index=False)
    weak_metrics.drop(columns=["source_run"]).to_csv(weak_metric_path, index=False)
    weak_predictions.drop(columns=["source_run"]).to_csv(weak_prediction_path, index=False)

    outputs = {
        "--metrics-output": tmp_path / "stack_metrics.csv",
        "--predictions-output": tmp_path / "stack_predictions.csv",
        "--candidates-output": tmp_path / "stack_candidates.csv",
    }
    script = Path(__file__).parents[1] / "scripts" / "55_run_sota_prediction_stack.py"
    spec = importlib.util.spec_from_file_location("sota_prediction_stack_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv = [
        "55_run_sota_prediction_stack.py",
        "--sources",
        f"strong={strong_metric_path}:{strong_prediction_path}",
        f"weak={weak_metric_path}:{weak_prediction_path}",
        "--max-validation-drop",
        "0.05",
        "--min-sources",
        "1",
    ]
    for flag, path in outputs.items():
        argv.extend([flag, str(path)])
    monkeypatch.setattr(sys, "argv", argv)
    module.main()

    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0
    candidates = pd.read_csv(outputs["--candidates-output"])
    assert candidates[candidates["source_run"].eq("weak")]["selected"].eq(False).all()
