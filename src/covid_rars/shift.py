from __future__ import annotations

import numpy as np
import pandas as pd

from covid_rars.cross_dataset import numeric_feature_columns


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    values = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, values, side="right") / a.size
    cdf_b = np.searchsorted(b, values, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _standardized_mean_difference(source: np.ndarray, external: np.ndarray) -> float:
    source = np.asarray(source, dtype=float)
    external = np.asarray(external, dtype=float)
    source = source[np.isfinite(source)]
    external = external[np.isfinite(external)]
    if source.size == 0 or external.size == 0:
        return float("nan")
    pooled = np.sqrt((float(np.var(source, ddof=1)) + float(np.var(external, ddof=1))) / 2.0)
    diff = float(np.mean(external) - np.mean(source))
    if not np.isfinite(pooled) or pooled == 0.0:
        return 0.0 if diff == 0.0 else float("inf")
    return diff / pooled


def feature_shift_report(
    source_features: pd.DataFrame,
    external_features: pd.DataFrame,
    id_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Rank numeric features by source-vs-external distribution shift."""
    source_cols = set(numeric_feature_columns(source_features, id_columns=id_columns))
    external_cols = set(numeric_feature_columns(external_features, id_columns=id_columns))
    columns = sorted(source_cols & external_cols)
    rows: list[dict[str, object]] = []
    for col in columns:
        source = pd.to_numeric(source_features[col], errors="coerce").to_numpy(dtype=float)
        external = pd.to_numeric(external_features[col], errors="coerce").to_numpy(dtype=float)
        smd = _standardized_mean_difference(source, external)
        rows.append(
            {
                "feature": col,
                "source_mean": float(np.nanmean(source)) if np.isfinite(source).any() else float("nan"),
                "external_mean": float(np.nanmean(external)) if np.isfinite(external).any() else float("nan"),
                "source_std": float(np.nanstd(source)) if np.isfinite(source).any() else float("nan"),
                "external_std": float(np.nanstd(external)) if np.isfinite(external).any() else float("nan"),
                "standardized_mean_difference": float(smd),
                "abs_standardized_mean_difference": float(abs(smd)) if np.isfinite(smd) else smd,
                "ks_statistic": _ks_statistic(source, external),
                "n_source": int(np.isfinite(source).sum()),
                "n_external": int(np.isfinite(external).sum()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["abs_standardized_mean_difference", "ks_statistic"], ascending=[False, False]
    ).reset_index(drop=True)


def shift_summary(report: pd.DataFrame, smd_threshold: float = 0.5) -> dict[str, object]:
    if report is None or report.empty:
        return {"n_features": 0, "n_high_shift_features": 0, "max_abs_standardized_mean_difference": float("nan")}
    high = report[report["abs_standardized_mean_difference"].astype(float) >= smd_threshold]
    return {
        "n_features": int(len(report)),
        "n_high_shift_features": int(len(high)),
        "max_abs_standardized_mean_difference": float(report["abs_standardized_mean_difference"].max()),
        "smd_threshold": float(smd_threshold),
    }
