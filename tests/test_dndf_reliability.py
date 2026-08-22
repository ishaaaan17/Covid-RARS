from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covid_rars.dndf_reliability import (
    compute_dndf_calibration_summary,
    compute_dndf_decision_curve_analysis,
    compute_dndf_fixed_sensitivity_operating_points,
    run_dndf_bootstrap_uncertainty,
)


def test_dndf_reliability_audits():
    rng = np.random.RandomState(42)
    n = 50
    lbls = ["positive" if i % 2 == 0 else "negative" for i in range(n)]
    probs = [0.8 if l == "positive" else 0.2 for l in lbls] + rng.randn(n) * 0.05
    probs = np.clip(probs, 0.01, 0.99)

    preds_df = pd.DataFrame({
        "participant_id": [f"p_{i}" for i in range(n)],
        "split": ["test"] * n,
        "modality": ["cough"] * n,
        "model_name": ["dndf_test"] * n,
        "label_binary": lbls,
        "probability": probs,
    })

    # 1. Calibration summary
    cal_df = compute_dndf_calibration_summary(preds_df)
    assert not cal_df.empty
    assert "ece" in cal_df.columns
    assert "brier_score" in cal_df.columns

    # 2. Operating points
    op_df = compute_dndf_fixed_sensitivity_operating_points(preds_df, min_sensitivity=0.85)
    assert not op_df.empty
    assert "achieved_sensitivity" in op_df.columns
    assert "achieved_specificity" in op_df.columns

    # 3. Decision curve analysis
    dca_df = compute_dndf_decision_curve_analysis(preds_df)
    assert not dca_df.empty
    assert "net_benefit_model" in dca_df.columns
    assert "net_benefit_all" in dca_df.columns

    # 4. Bootstrap CI
    boot_df = run_dndf_bootstrap_uncertainty(preds_df, n_bootstraps=20)
    assert not boot_df.empty
    assert "auroc_ci_low" in boot_df.columns
    assert "auroc_ci_high" in boot_df.columns
