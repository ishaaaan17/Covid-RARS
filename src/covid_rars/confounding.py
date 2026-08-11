from __future__ import annotations

import numpy as np
import pandas as pd


def age_bucket(value: object) -> str:
    try:
        age = float(value)
    except Exception:
        return "unknown"
    if age < 30:
        return "<30"
    if age < 45:
        return "30-44"
    if age < 60:
        return "45-59"
    return "60+"


def add_coarsened_covariates(metadata: pd.DataFrame) -> pd.DataFrame:
    out = metadata.copy()
    if "age" in out.columns and "age_bucket" not in out.columns:
        out["age_bucket"] = out["age"].map(age_bucket)
    return out


def _set_id(row: pd.Series, covariates: list[str]) -> str:
    return "|".join(f"{col}={row[col]}" for col in covariates)


def coarsened_exact_match(
    metadata: pd.DataFrame,
    covariates: list[str] | None = None,
    label_column: str = "label_binary",
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a positive/negative matched subset within coarsened covariate cells."""
    covariates = covariates or ["age_bucket", "gender"]
    df = add_coarsened_covariates(metadata)
    missing = set(covariates + [label_column]) - set(df.columns)
    if missing:
        raise KeyError(f"Missing matching columns: {sorted(missing)}")
    df = df[df[label_column].isin(["positive", "negative"])].copy()
    rng = np.random.default_rng(random_state)
    frames: list[pd.DataFrame] = []
    for _, group in df.groupby(covariates, dropna=False):
        positives = group[group[label_column] == "positive"]
        negatives = group[group[label_column] == "negative"]
        n = min(len(positives), len(negatives))
        if n == 0:
            continue
        pos_idx = rng.choice(positives.index.to_numpy(), size=n, replace=False)
        neg_idx = rng.choice(negatives.index.to_numpy(), size=n, replace=False)
        matched = df.loc[np.concatenate([pos_idx, neg_idx])].copy()
        matched["matched_set_id"] = matched.apply(lambda row: _set_id(row, covariates), axis=1)
        matched["matching_method"] = "coarsened_exact"
        frames.append(matched)
    if not frames:
        return pd.DataFrame(columns=list(df.columns) + ["matched_set_id", "matching_method"])
    return pd.concat(frames, ignore_index=True, sort=False)


def _standardized_mean_difference(pos: pd.Series, neg: pd.Series) -> float:
    pos_num = pd.to_numeric(pos, errors="coerce").dropna()
    neg_num = pd.to_numeric(neg, errors="coerce").dropna()
    if pos_num.empty or neg_num.empty:
        return float("nan")
    pooled = np.sqrt((float(pos_num.var(ddof=1)) + float(neg_num.var(ddof=1))) / 2.0)
    if not np.isfinite(pooled) or pooled == 0.0:
        return 0.0 if float(pos_num.mean()) == float(neg_num.mean()) else float("inf")
    return float((float(pos_num.mean()) - float(neg_num.mean())) / pooled)


def balance_table(
    metadata: pd.DataFrame,
    covariates: list[str] | None = None,
    label_column: str = "label_binary",
) -> pd.DataFrame:
    """Summarize positive/negative balance for matched or full metadata."""
    covariates = covariates or ["age_bucket", "gender"]
    df = add_coarsened_covariates(metadata)
    df = df[df[label_column].isin(["positive", "negative"])].copy()
    rows: list[dict[str, object]] = []
    for covariate in covariates:
        if covariate not in df.columns:
            continue
        pos = df[df[label_column] == "positive"][covariate]
        neg = df[df[label_column] == "negative"][covariate]
        if pd.api.types.is_numeric_dtype(df[covariate]):
            smd = _standardized_mean_difference(pos, neg)
            rows.append(
                {
                    "covariate": covariate,
                    "type": "numeric",
                    "n_positive": int(pos.notna().sum()),
                    "n_negative": int(neg.notna().sum()),
                    "max_abs_standardized_difference": abs(smd) if np.isfinite(smd) else smd,
                    "details_json": "{}",
                }
            )
        else:
            levels = sorted(set(pos.dropna().astype(str)) | set(neg.dropna().astype(str)))
            diffs: dict[str, float] = {}
            for level in levels:
                p_pos = float((pos.astype(str) == level).mean()) if len(pos) else 0.0
                p_neg = float((neg.astype(str) == level).mean()) if len(neg) else 0.0
                diffs[level] = p_pos - p_neg
            max_abs = max((abs(v) for v in diffs.values()), default=0.0)
            rows.append(
                {
                    "covariate": covariate,
                    "type": "categorical",
                    "n_positive": int(len(pos)),
                    "n_negative": int(len(neg)),
                    "max_abs_standardized_difference": float(max_abs),
                    "details_json": pd.Series(diffs).to_json(),
                }
            )
    return pd.DataFrame(rows)
