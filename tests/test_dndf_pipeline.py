from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covid_rars.dndf_stages import run_dndf_reliability_pipeline


def test_dndf_pipeline_end_to_end(tmp_path):
    rng = np.random.RandomState(42)
    records = []
    for i in range(40):
        p_id = f"p_{i:03d}"
        lbl = "positive" if i % 2 == 0 else "negative"
        for mod in ["cough", "speech"]:
            rec = {
                "participant_id": p_id,
                "recording_id": f"{p_id}_{mod}",
                "modality": mod,
                "label_binary": lbl,
            }
            for f in range(8):
                rec[f"feat_{f}"] = rng.randn() + (0.6 if lbl == "positive" else -0.6)
            records.append(rec)

    feat_df = pd.DataFrame(records)

    # Synthetic external dataset
    ext_records = []
    for i in range(20):
        p_id = f"ext_p_{i:03d}"
        lbl = "positive" if i % 3 == 0 else "negative"
        rec = {
            "participant_id": p_id,
            "recording_id": f"{p_id}_cough",
            "modality": "cough",
            "label_binary": lbl,
        }
        for f in range(8):
            rec[f"feat_{f}"] = rng.randn()
        ext_records.append(rec)
    ext_df = pd.DataFrame(ext_records)

    artifacts = run_dndf_reliability_pipeline(
        features_df=feat_df,
        external_features_df=ext_df,
        modalities=["cough", "speech"],
        seeds=[1, 2],
        num_trees=2,
        depth=2,
        max_epochs=2,
        patience=2,
        use_smote=False,
        device="cpu",
        output_dir=tmp_path / "dndf_test_out",
    )

    assert not artifacts.final_summary_table.empty
    assert not artifacts.track_a_metrics.empty
    assert not artifacts.calibration_summary.empty
    assert not artifacts.operating_points.empty
    assert not artifacts.dca_summary.empty
    assert (tmp_path / "dndf_test_out/dndf_final_validation_summary.csv").exists()
