from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covid_rars.dndf_training import participant_average_predictions, train_dndf_modality_models


def test_participant_average_predictions():
    df = pd.DataFrame({
        "participant_id": ["p1", "p1", "p2", "p2", "p3"],
        "recording_id": ["r1", "r2", "r3", "r4", "r5"],
        "modality": ["cough", "cough", "cough", "cough", "cough"],
        "split": ["train", "train", "test", "test", "test"],
        "label_binary": ["positive", "positive", "negative", "negative", "positive"],
        "probability": [0.8, 0.6, 0.2, 0.4, 0.9],
        "threshold": [0.5, 0.5, 0.5, 0.5, 0.5],
    })

    part_df = participant_average_predictions(df)
    assert len(part_df) == 3
    p1_row = part_df[part_df["participant_id"] == "p1"].iloc[0]
    assert pytest.approx(p1_row["probability"], 1e-5) == 0.7
    p2_row = part_df[part_df["participant_id"] == "p2"].iloc[0]
    assert pytest.approx(p2_row["probability"], 1e-5) == 0.3


def test_train_dndf_modality_models_synthetic():
    rng = np.random.RandomState(42)
    records = []
    for i in range(40):
        p_id = f"p_{i}"
        lbl = "positive" if i % 2 == 0 else "negative"
        split = "train" if i < 24 else ("val" if i < 32 else "test")
        rec = {
            "participant_id": p_id,
            "recording_id": f"{p_id}_rec",
            "modality": "cough",
            "split": split,
            "label_binary": lbl,
        }
        for f in range(10):
            rec[f"feature_{f}"] = rng.randn() + (0.5 if lbl == "positive" else -0.5)
        records.append(rec)

    feat_df = pd.DataFrame(records)

    m_df, p_df, sel_df = train_dndf_modality_models(
        feat_df,
        modalities=["cough"],
        model_types=["dndf"],
        num_trees=2,
        depth=2,
        max_epochs=3,
        batch_size=8,
        use_smote=False,
        random_state=42,
        device="cpu",
    )

    assert not m_df.empty
    assert not p_df.empty
    assert not sel_df.empty
    assert "auroc" in m_df.columns
    assert "probability" in p_df.columns
