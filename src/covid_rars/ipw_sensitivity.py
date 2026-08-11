from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from covid_rars.confounding_controlled_eval import (
    DEFAULT_CONFOUNDERS,
    _effective_sample_size,
    _fit_ipw_weights,
    _iter_groups,
    _unit_id_column,
    balance_diagnostics,
    merge_predictions_with_confounders,
    weighted_binary_metric_bundle,
)


@dataclass
class IPWSensitivityResult:
    metrics: pd.DataFrame
    balance: pd.DataFrame
    weights: pd.DataFrame


def _balance_summary(balance: pd.DataFrame) -> dict[str, float]:
    if balance.empty:
        return {
            "mean_abs_smd_before": float("nan"),
            "mean_abs_smd_after": float("nan"),
            "max_abs_smd_before": float("nan"),
            "max_abs_smd_after": float("nan"),
        }
    return {
        "mean_abs_smd_before": float(pd.to_numeric(balance["before_abs_smd"], errors="coerce").mean()),
        "mean_abs_smd_after": float(pd.to_numeric(balance["after_abs_smd"], errors="coerce").mean()),
        "max_abs_smd_before": float(pd.to_numeric(balance["before_abs_smd"], errors="coerce").max()),
        "max_abs_smd_after": float(pd.to_numeric(balance["after_abs_smd"], errors="coerce").max()),
    }


def _append_metric_rows(
    rows: list[dict[str, object]],
    merged: pd.DataFrame,
    group_columns: list[str],
    weights_column: str | None,
    control_method: str,
    weight_config: str,
    weight_cap: float | None,
    clip_quantile: float | None,
    balance_summary: dict[str, float],
    threshold: float,
) -> None:
    for group_values, group in _iter_groups(merged, group_columns):
        weights = group[weights_column] if weights_column else None
        bundle = weighted_binary_metric_bundle(
            group["label_binary"],
            group["probability"],
            weights=weights,
            threshold=threshold,
        )
        bundle.update(group_values)
        bundle.update(balance_summary)
        bundle["control_method"] = control_method
        bundle["weight_config"] = weight_config
        bundle["weight_cap"] = weight_cap
        bundle["clip_quantile"] = clip_quantile
        if weights_column:
            values = group[weights_column].astype(float).to_numpy()
            bundle["min_weight"] = float(np.min(values))
            bundle["max_weight"] = float(np.max(values))
            bundle["mean_weight"] = float(np.mean(values))
        else:
            bundle["min_weight"] = 1.0
            bundle["max_weight"] = 1.0
            bundle["mean_weight"] = 1.0
            bundle["effective_sample_size"] = float(len(group))
        rows.append(bundle)


def run_ipw_sensitivity_analysis(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    covariates: list[str] | None = None,
    group_columns: list[str] | None = None,
    split: str | None = "test",
    threshold: float = 0.5,
    weight_caps: list[float] | None = None,
    clip_quantiles: list[float] | None = None,
    random_state: int = 42,
) -> IPWSensitivityResult:
    covariates = covariates or DEFAULT_CONFOUNDERS
    group_columns = [col for col in (group_columns or []) if col in predictions.columns]
    weight_caps = weight_caps or [2.0, 5.0, 10.0, 20.0]
    clip_quantiles = clip_quantiles or [0.95, 0.99, 1.0]

    merged = merge_predictions_with_confounders(predictions, metadata)
    merged = merged[merged["label_binary"].isin(["positive", "negative"])].copy()
    if split is not None and "split" in merged.columns:
        merged = merged[merged["split"].astype(str).eq(str(split))].copy()
    if merged.empty:
        raise ValueError("No labeled prediction rows available for IPW sensitivity analysis")

    unit_col = _unit_id_column(merged)
    unit_frame = merged.drop_duplicates(unit_col, keep="first").copy()
    metric_rows: list[dict[str, object]] = []
    balance_frames: list[pd.DataFrame] = []
    weight_frames: list[pd.DataFrame] = []

    base_weights = pd.Series(np.ones(len(unit_frame), dtype=float), index=unit_frame.index)
    _, _, base_x = _fit_ipw_weights(unit_frame, covariates=covariates, max_weight=1.0, clip_quantile=1.0, random_state=random_state)
    base_balance = balance_diagnostics(unit_frame, base_x, base_weights) if not base_x.empty else pd.DataFrame()
    if not base_balance.empty:
        base_balance["weight_config"] = "unweighted"
        base_balance["control_method"] = "unweighted"
        balance_frames.append(base_balance)
    _append_metric_rows(
        metric_rows,
        merged,
        group_columns,
        weights_column=None,
        control_method="unweighted",
        weight_config="unweighted",
        weight_cap=None,
        clip_quantile=None,
        balance_summary=_balance_summary(base_balance),
        threshold=threshold,
    )

    for cap in weight_caps:
        for quantile in clip_quantiles:
            weights, propensity, x = _fit_ipw_weights(
                unit_frame,
                covariates=covariates,
                max_weight=float(cap),
                clip_quantile=float(quantile),
                random_state=random_state,
            )
            config = f"ipw_cap_{float(cap):g}_q_{float(quantile):g}"
            unit_weights = unit_frame[[unit_col, "label_binary"]].copy()
            unit_weights["weight_config"] = config
            unit_weights["control_method"] = "ipw_label_propensity"
            unit_weights["weight_cap"] = float(cap)
            unit_weights["clip_quantile"] = float(quantile)
            unit_weights["propensity_score"] = propensity.to_numpy(dtype=float)
            unit_weights["ipw_weight"] = weights.to_numpy(dtype=float)
            unit_weights["effective_sample_size"] = _effective_sample_size(unit_weights["ipw_weight"].to_numpy(dtype=float))
            weight_frames.append(unit_weights)

            weighted = merged.merge(unit_weights[[unit_col, "ipw_weight"]], on=unit_col, how="left")
            balance = balance_diagnostics(unit_frame, x, weights) if not x.empty else pd.DataFrame()
            if not balance.empty:
                balance["weight_config"] = config
                balance["control_method"] = "ipw_label_propensity"
                balance["weight_cap"] = float(cap)
                balance["clip_quantile"] = float(quantile)
                balance_frames.append(balance)
            _append_metric_rows(
                metric_rows,
                weighted,
                group_columns,
                weights_column="ipw_weight",
                control_method="ipw_label_propensity",
                weight_config=config,
                weight_cap=float(cap),
                clip_quantile=float(quantile),
                balance_summary=_balance_summary(balance),
                threshold=threshold,
            )

    return IPWSensitivityResult(
        metrics=pd.DataFrame(metric_rows),
        balance=pd.concat(balance_frames, ignore_index=True, sort=False) if balance_frames else pd.DataFrame(),
        weights=pd.concat(weight_frames, ignore_index=True, sort=False) if weight_frames else pd.DataFrame(),
    )
