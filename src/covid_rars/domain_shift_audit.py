from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from covid_rars.cross_dataset import numeric_feature_columns
from covid_rars.metrics import binary_metric_bundle


@dataclass
class DomainShiftAuditResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame


def _filter_modality(frame: pd.DataFrame, modality: str | None) -> pd.DataFrame:
    if modality is None or "modality" not in frame.columns:
        return frame.copy()
    return frame[frame["modality"].astype(str).eq(str(modality))].copy()


def _common_varying_numeric_columns(source: pd.DataFrame, external: pd.DataFrame) -> list[str]:
    source_cols = set(numeric_feature_columns(source))
    external_cols = set(numeric_feature_columns(external))
    common = sorted(source_cols & external_cols)
    combined = pd.concat(
        [
            source.reindex(columns=common, fill_value=0.0),
            external.reindex(columns=common, fill_value=0.0),
        ],
        ignore_index=True,
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return [col for col in common if combined[col].nunique(dropna=False) > 1]


def _domain_frame(source: pd.DataFrame, external: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    source_part = source.copy()
    external_part = external.copy()
    source_part["source_domain"] = "source"
    external_part["source_domain"] = "external"
    source_part["domain_label"] = 0
    external_part["domain_label"] = 1
    return pd.concat([source_part, external_part], ignore_index=True, sort=False).replace([np.inf, -np.inf], np.nan).fillna(
        {col: 0.0 for col in columns}
    )


def run_domain_shift_audit(
    source_features: pd.DataFrame,
    external_features: pd.DataFrame,
    representation: str,
    modality: str | None = "cough",
    test_size: float = 0.30,
    random_state: int = 42,
) -> DomainShiftAuditResult:
    source = _filter_modality(source_features, modality)
    external = _filter_modality(external_features, modality)
    if source.empty or external.empty:
        raise ValueError("Need non-empty source and external rows for domain-shift audit")

    columns = _common_varying_numeric_columns(source, external)
    if not columns:
        raise ValueError("No common varying numeric feature columns are available for domain-shift audit")

    combined = _domain_frame(source, external, columns)
    train_idx, test_idx = train_test_split(
        combined.index.to_numpy(),
        test_size=float(test_size),
        random_state=random_state,
        stratify=combined["domain_label"].to_numpy(dtype=int),
    )
    train = combined.loc[train_idx].copy()
    test = combined.loc[test_idx].copy()
    x_train = train[columns].astype(float)
    x_test = test[columns].astype(float)
    y_train = train["domain_label"].to_numpy(dtype=int)
    y_test = test["domain_label"].to_numpy(dtype=int)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=3000, random_state=random_state),
            ),
        ]
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    bundle = binary_metric_bundle(y_test, probabilities, threshold=0.5)
    metrics = {
        "representation": representation,
        "modality": modality if modality is not None else "all",
        "audit_model": "domain_logistic_regression",
        "domain_auroc": bundle["auroc"],
        "domain_auprc": bundle["auprc"],
        "balanced_accuracy": bundle["balanced_accuracy"],
        "f1": bundle["f1"],
        "accuracy": float(accuracy_score(y_test, probabilities >= 0.5)),
        "brier": bundle["brier"],
        "ece": bundle["ece"],
        "nll": bundle["nll"],
        "threshold": 0.5,
        "n_samples": float(len(test)),
        "n_source": int(len(source)),
        "n_external": int(len(external)),
        "test_source_rows": int(np.sum(y_test == 0)),
        "test_external_rows": int(np.sum(y_test == 1)),
        "n_features": int(len(columns)),
    }

    id_defaults = pd.Series([""] * len(test), index=test.index)
    predictions = pd.DataFrame(
        {
            "recording_id": test.get("recording_id", id_defaults).to_numpy(),
            "participant_id": test.get("participant_id", id_defaults).to_numpy(),
            "dataset": test.get("dataset", id_defaults).to_numpy(),
            "modality": test.get("modality", pd.Series([modality or ""] * len(test), index=test.index)).to_numpy(),
            "source_domain": test["source_domain"].to_numpy(),
            "domain_label": y_test,
            "representation": representation,
            "probability_external": probabilities,
        }
    )

    coefs = model.named_steps["classifier"].coef_[0]
    importance = pd.DataFrame(
        {
            "representation": representation,
            "modality": modality if modality is not None else "all",
            "feature": columns,
            "coefficient": coefs,
        }
    )
    importance["importance_abs"] = importance["coefficient"].abs()
    importance["direction"] = np.where(importance["coefficient"] >= 0, "external_domain", "source_domain")
    importance = importance.sort_values(["importance_abs", "feature"], ascending=[False, True]).reset_index(drop=True)

    return DomainShiftAuditResult(
        metrics=pd.DataFrame([metrics]),
        predictions=predictions,
        feature_importance=importance,
    )
