from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covid_rars.dndf_fusion import run_dndf_comparator_hybrid_fusion, run_dndf_multimodal_fusion


def test_dndf_multimodal_fusion():
    rng = np.random.RandomState(42)
    rows = []
    for i in range(30):
        p_id = f"p_{i}"
        lbl = "positive" if i % 2 == 0 else "negative"
        split = "train" if i < 15 else ("val" if i < 22 else "test")
        prob_c = 0.8 if lbl == "positive" else 0.2
        prob_s = 0.7 if lbl == "positive" else 0.3

        rows.append({
            "participant_id": p_id,
            "split": split,
            "modality": "cough",
            "model_name": "dndf_test",
            "label_binary": lbl,
            "probability": prob_c + rng.randn() * 0.05,
        })
        rows.append({
            "participant_id": p_id,
            "split": split,
            "modality": "speech",
            "model_name": "dndf_test",
            "label_binary": lbl,
            "probability": prob_s + rng.randn() * 0.05,
        })

    preds_df = pd.DataFrame(rows)
    m_df, p_df = run_dndf_multimodal_fusion(preds_df, modalities=["cough", "speech"])

    assert not m_df.empty
    assert not p_df.empty
    assert "uniform_mean" in m_df["fusion_method"].values
    assert "modality_combination" in m_df.columns
    assert set(p_df["split"].unique()).issubset({"train", "val", "test"})


def test_dndf_comparator_hybrid_fusion():
    dndf_df = pd.DataFrame({
        "participant_id": ["p1", "p2", "p3", "p4"],
        "split": ["val", "val", "test", "test"],
        "modality": ["cough", "cough", "cough", "cough"],
        "label_binary": ["positive", "negative", "positive", "negative"],
        "probability": [0.8, 0.2, 0.85, 0.15],
    })

    comp_df = pd.DataFrame({
        "participant_id": ["p1", "p2", "p3", "p4"],
        "split": ["val", "val", "test", "test"],
        "modality": ["cough", "cough", "cough", "cough"],
        "label_binary": ["positive", "negative", "positive", "negative"],
        "probability": [0.7, 0.3, 0.75, 0.25],
    })

    m_df, p_df = run_dndf_comparator_hybrid_fusion(dndf_df, comp_df, modality="cough")
    assert not m_df.empty
    assert not p_df.empty
    assert "prob_dndf" in p_df.columns
    assert "prob_comparator" in p_df.columns
    assert pytest.approx(p_df.iloc[0]["probability"], 1e-5) == 0.75
