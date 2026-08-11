from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _feature_frame(n_participants: int = 54) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    for idx in range(n_participants):
        split = "train" if idx % 6 < 4 else "validation" if idx % 6 == 4 else "test"
        label = "positive" if idx % 4 in {1, 2} else "negative"
        y = 1.0 if label == "positive" else -1.0
        rows.append(
            {
                "recording_id": f"r{idx:03d}",
                "participant_id": f"p{idx:03d}",
                "dataset": "coswara",
                "modality": "cough",
                "submodality": "cough",
                "label_binary": label,
                "split": split,
                "signal_main": y + float(rng.normal(0.0, 0.05)),
                "signal_aux": 0.5 * y + float(rng.normal(0.0, 0.05)),
                "noise_0": float(rng.normal()),
                "noise_1": float(rng.normal()),
                "noise_2": float(rng.normal()),
                "noise_3": float(rng.normal()),
            }
        )
    return pd.DataFrame(rows)


def test_binary_pso_feature_search_selects_signal_and_reports_metrics() -> None:
    from covid_rars.swarm_feature_search import run_swarm_feature_search

    result = run_swarm_feature_search(
        _feature_frame(),
        modalities=["cough"],
        classifier="logistic",
        particles=5,
        iterations=4,
        max_candidate_features=6,
        random_state=0,
    )

    assert not result.metrics.empty
    assert not result.predictions.empty
    assert not result.selection.empty
    selected = result.selection.iloc[0]["selected_features"]
    assert "signal" in selected
    assert {"validation", "test"}.issubset(set(result.metrics["metric_split"]))
    assert result.metrics["analysis_family"].eq("sota_swarm_feature_search").all()


def test_swarm_feature_search_cli_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    features = _feature_frame()
    features_path = tmp_path / "features.csv"
    features.to_csv(features_path, index=False)
    outputs = {
        "--metrics-output": tmp_path / "metrics.csv",
        "--predictions-output": tmp_path / "predictions.csv",
        "--selection-output": tmp_path / "selection.csv",
    }
    script = Path(__file__).parents[1] / "scripts" / "54_run_swarm_feature_search.py"
    spec = importlib.util.spec_from_file_location("swarm_feature_search_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv = [
        "54_run_swarm_feature_search.py",
        "--features",
        str(features_path),
        "--modalities",
        "cough",
        "--classifier",
        "logistic",
        "--particles",
        "5",
        "--iterations",
        "4",
        "--max-candidate-features",
        "6",
    ]
    for flag, path in outputs.items():
        argv.extend([flag, str(path)])
    monkeypatch.setattr(sys, "argv", argv)
    module.main()

    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0
    metrics = pd.read_csv(outputs["--metrics-output"])
    assert {"validation", "test"}.issubset(set(metrics["metric_split"]))
