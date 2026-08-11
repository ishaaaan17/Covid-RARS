from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from covid_rars.features import feature_columns
from covid_rars.metrics import labels_to_binary
from covid_rars.strong_baseline import _safe_f_classif


MERGE_ID_COLUMNS = [
    "recording_id",
    "participant_id",
    "dataset",
    "modality",
    "submodality",
    "label_binary",
    "split",
    "quality_flag",
    "manual_quality_label",
]


@dataclass(frozen=True)
class TopKFeatureSelectionResult:
    tables: dict[int, pd.DataFrame]
    importance: pd.DataFrame
    summary: pd.DataFrame


def _normalise_source_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name).strip().lower()).strip("_") or "source"


def _canonical_feature_table(table: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "recording_id" not in table.columns:
        raise ValueError(f"{source_name} feature table is missing recording_id")
    if "is_augmented" in table.columns:
        augmented = table["is_augmented"].fillna(False).astype(bool)
        table = table.loc[~augmented].copy()
    else:
        table = table.copy()

    if "source_recording_id" in table.columns:
        table["_merge_recording_id"] = table["source_recording_id"].fillna(table["recording_id"]).astype(str)
    else:
        table["_merge_recording_id"] = table["recording_id"].astype(str)

    duplicated = table["_merge_recording_id"].duplicated(keep=False)
    if duplicated.any():
        examples = sorted(table.loc[duplicated, "_merge_recording_id"].astype(str).unique())[:5]
        raise ValueError(
            f"{source_name} feature table has duplicate recording ids after augmentation filtering: {examples}"
        )
    return table


def _prefixed_feature_block(table: pd.DataFrame, source_name: str) -> pd.DataFrame:
    prefix = _normalise_source_name(source_name)
    cols = feature_columns(table)
    if not cols:
        raise ValueError(f"{source_name} feature table has no numeric feature columns")
    block = table[["_merge_recording_id", *cols]].copy()
    rename = {col: f"{prefix}__{col}" for col in cols}
    return block.rename(columns=rename)


def merge_feature_tables(feature_tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if not feature_tables:
        raise ValueError("No feature tables were provided")

    canonical: dict[str, pd.DataFrame] = {
        str(name): _canonical_feature_table(table, str(name))
        for name, table in feature_tables.items()
    }
    first_name = next(iter(canonical))
    first = canonical[first_name]
    id_cols = [col for col in MERGE_ID_COLUMNS if col in first.columns]
    merged = first[["_merge_recording_id", *id_cols]].copy()
    for name, table in canonical.items():
        block = _prefixed_feature_block(table, name)
        merged = merged.merge(block, on="_merge_recording_id", how="inner", validate="one_to_one")
    merged = merged.drop(columns=["_merge_recording_id"])
    cols = feature_columns(merged)
    if not cols:
        raise ValueError("Merged feature table has no numeric feature columns")
    merged[cols] = merged[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return merged


def _has_two_classes(frame: pd.DataFrame) -> bool:
    return frame["label_binary"].isin(["positive", "negative"]).all() and frame["label_binary"].nunique() == 2


def _rank_with_univariate(x: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    scores, _ = _safe_f_classif(x.to_numpy(dtype=float), y)
    return np.asarray(scores, dtype=float)


def _rank_with_extra_trees(x: pd.DataFrame, y: np.ndarray, random_state: int) -> np.ndarray:
    from sklearn.ensemble import ExtraTreesClassifier

    model = ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(x, y)
    return np.asarray(model.feature_importances_, dtype=float)


def _rank_with_lightgbm(x: pd.DataFrame, y: np.ndarray, random_state: int) -> np.ndarray:
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("lightgbm is required for --ranker lightgbm") from exc

    min_child_samples = max(5, min(20, int(max(1, len(x) // 20))))
    model = LGBMClassifier(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=min_child_samples,
        subsample=0.9,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        objective="binary",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x, y)
    booster = getattr(model, "booster_", None)
    if booster is not None:
        importance = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
        if np.any(importance > 0):
            return importance
    return np.asarray(model.feature_importances_, dtype=float)


def _rank_features_for_frame(
    frame: pd.DataFrame,
    cols: list[str],
    *,
    ranker: str,
    random_state: int,
) -> np.ndarray:
    if not _has_two_classes(frame):
        return np.zeros(len(cols), dtype=float)
    x = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = labels_to_binary(frame["label_binary"])
    if ranker == "univariate":
        return _rank_with_univariate(x, y)
    if ranker == "extra_trees":
        return _rank_with_extra_trees(x, y, random_state=random_state)
    if ranker == "lightgbm":
        return _rank_with_lightgbm(x, y, random_state=random_state)
    if ranker == "auto":
        try:
            return _rank_with_lightgbm(x, y, random_state=random_state)
        except Exception:
            return _rank_with_extra_trees(x, y, random_state=random_state)
    raise ValueError(f"Unsupported feature ranker: {ranker}")


def rank_train_features(
    features: pd.DataFrame,
    *,
    ranker: str = "lightgbm",
    selection_scope: str = "per_modality_mean",
    random_state: int = 42,
) -> pd.DataFrame:
    cols = feature_columns(features)
    if not cols:
        raise ValueError("No numeric feature columns are available for feature selection")
    train = features[
        features["split"].astype(str).eq("train")
        & features["label_binary"].isin(["positive", "negative"])
    ].copy()
    if train.empty:
        raise ValueError("No supervised training rows are available for feature selection")

    scopes: list[tuple[str, pd.DataFrame]]
    if selection_scope == "global":
        scopes = [("global", train)]
    elif selection_scope == "per_modality_mean":
        scopes = [
            (str(modality), group.copy())
            for modality, group in train.groupby("modality", dropna=False)
            if _has_two_classes(group)
        ]
        if not scopes:
            scopes = [("global", train)]
    else:
        raise ValueError(f"Unsupported selection_scope: {selection_scope}")

    per_scope: list[np.ndarray] = []
    for offset, (scope_name, scope_frame) in enumerate(scopes):
        importance = _rank_features_for_frame(
            scope_frame,
            cols,
            ranker=ranker,
            random_state=random_state + offset,
        )
        importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
        scale = float(np.max(importance)) if np.any(importance > 0) else 1.0
        per_scope.append(importance / scale)
    aggregate = np.mean(np.vstack(per_scope), axis=0)
    ranking = pd.DataFrame(
        {
            "feature": cols,
            "importance": aggregate,
            "ranker": ranker,
            "selection_split": "train",
            "selection_scope": selection_scope,
            "n_selection_scopes": float(len(scopes)),
            "n_train_rows": float(len(train)),
            "n_train_positive": float((train["label_binary"] == "positive").sum()),
            "n_train_negative": float((train["label_binary"] == "negative").sum()),
        }
    )
    ranking = ranking.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1, dtype=int)
    return ranking


def select_top_k_feature_tables(
    features: pd.DataFrame,
    *,
    k_values: list[int],
    ranker: str = "lightgbm",
    selection_scope: str = "per_modality_mean",
    random_state: int = 42,
) -> TopKFeatureSelectionResult:
    if not k_values:
        raise ValueError("At least one top-k value is required")
    ranking = rank_train_features(
        features,
        ranker=ranker,
        selection_scope=selection_scope,
        random_state=random_state,
    )
    id_cols = [col for col in MERGE_ID_COLUMNS if col in features.columns]
    tables: dict[int, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []
    for raw_k in k_values:
        k = int(raw_k)
        if k <= 0:
            raise ValueError(f"top-k values must be positive, got {raw_k}")
        selected = ranking.head(min(k, len(ranking)))["feature"].astype(str).tolist()
        table = features[[*id_cols, *selected]].copy()
        tables[k] = table
        summary_rows.append(
            {
                "k": k,
                "n_selected_features": float(len(selected)),
                "ranker": ranker,
                "selection_split": "train",
                "selection_scope": selection_scope,
                "top_feature": selected[0] if selected else "",
                "selected_features": ";".join(selected),
            }
        )
    return TopKFeatureSelectionResult(
        tables=tables,
        importance=ranking,
        summary=pd.DataFrame(summary_rows),
    )


def read_feature_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)
