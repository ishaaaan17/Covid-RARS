from __future__ import annotations

import numpy as np
import pandas as pd

from covid_rars.metrics import evaluate_predictions


DEFAULT_ACCEPTED_QUALITY = {"ok", "not_audited", "unknown", ""}


def apply_abstention(
    predictions: pd.DataFrame,
    probability_column: str = "probability",
    quality_column: str = "quality_flag",
    uncertainty_low: float = 0.4,
    uncertainty_high: float = 0.6,
    accepted_quality_flags: set[str] | None = None,
) -> pd.DataFrame:
    accepted_quality_flags = accepted_quality_flags or DEFAULT_ACCEPTED_QUALITY
    out = predictions.copy()
    prob = out[probability_column].astype(float)
    quality = out.get(quality_column, pd.Series(["unknown"] * len(out), index=out.index)).fillna("unknown").astype(str)
    low_quality = ~quality.str.lower().isin({q.lower() for q in accepted_quality_flags})
    uncertain = prob.between(uncertainty_low, uncertainty_high, inclusive="both")
    out["predicted_label"] = np.where(prob >= 0.5, "positive", "negative")
    out["confidence"] = np.maximum(prob, 1.0 - prob)
    out["accepted"] = ~(low_quality | uncertain)
    out["abstention_reason"] = "accepted"
    out.loc[uncertain, "abstention_reason"] = "uncertain_probability"
    out.loc[low_quality, "abstention_reason"] = "low_quality"
    return out


def coverage_curve(
    predictions: pd.DataFrame,
    probability_column: str = "probability",
    label_column: str = "label_binary",
    quality_column: str = "quality_flag",
    uncertainty_half_widths: list[float] | None = None,
) -> pd.DataFrame:
    widths = uncertainty_half_widths or [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    rows: list[dict[str, object]] = []
    for width in widths:
        low = 0.5 - width
        high = 0.5 + width
        abstained = apply_abstention(
            predictions,
            probability_column=probability_column,
            quality_column=quality_column,
            uncertainty_low=low,
            uncertainty_high=high,
        )
        kept = abstained[abstained["accepted"]].copy()
        row: dict[str, object] = {
            "uncertainty_half_width": width,
            "uncertainty_low": low,
            "uncertainty_high": high,
            "coverage": len(kept) / max(len(predictions), 1),
            "n_accepted": len(kept),
        }
        if len(kept) and kept[label_column].isin(["positive", "negative"]).all():
            metrics = evaluate_predictions(kept, probability_column=probability_column, label_column=label_column)
            if not metrics.empty:
                row.update(metrics.iloc[0].to_dict())
        rows.append(row)
    return pd.DataFrame(rows)
