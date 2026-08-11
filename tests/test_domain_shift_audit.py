from __future__ import annotations

import numpy as np
import pandas as pd


def _domain_feature_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    source_rows = []
    external_rows = []
    for idx in range(40):
        source_rows.append(
            {
                "recording_id": f"s{idx}",
                "participant_id": f"ps{idx}",
                "dataset": "coswara",
                "modality": "cough",
                "label_binary": "positive" if idx % 5 == 0 else "negative",
                "split": "train" if idx < 30 else "test",
                "feat_shifted": float(idx % 5) * 0.01,
                "feat_shared": float(idx % 7),
                "constant": 1.0,
            }
        )
        external_rows.append(
            {
                "recording_id": f"e{idx}",
                "participant_id": f"pe{idx}",
                "dataset": "coughvid",
                "modality": "cough",
                "label_binary": "positive" if idx % 9 == 0 else "negative",
                "split": "external",
                "feat_shifted": 4.0 + float(idx % 5) * 0.01,
                "feat_shared": float(idx % 7),
                "constant": 1.0,
            }
        )
    return pd.DataFrame(source_rows), pd.DataFrame(external_rows)


def test_domain_shift_audit_detects_dataset_separability() -> None:
    from covid_rars.domain_shift_audit import run_domain_shift_audit

    source, external = _domain_feature_frames()

    result = run_domain_shift_audit(
        source,
        external,
        representation="toy",
        modality="cough",
        test_size=0.35,
        random_state=7,
    )

    assert result.metrics.loc[0, "representation"] == "toy"
    assert result.metrics.loc[0, "domain_auroc"] > 0.95
    assert result.metrics.loc[0, "n_source"] == 40
    assert result.metrics.loc[0, "n_external"] == 40
    assert result.metrics.loc[0, "n_features"] == 2
    assert {"source_domain", "probability_external"}.issubset(result.predictions.columns)
    assert result.feature_importance.iloc[0]["feature"] == "feat_shifted"
    assert np.isfinite(result.feature_importance.iloc[0]["importance_abs"])
