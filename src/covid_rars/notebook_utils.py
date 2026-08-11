from __future__ import annotations

from pathlib import Path

import pandas as pd

try:  # Notebook display when available; harmless fallback for tests/scripts.
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(obj: object) -> None:
        print(obj)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


def project_path(path: str | Path) -> Path:
    """Resolve a path against the project root unless it is already absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def artifact(path: str | Path, required: bool = True) -> Path:
    """Resolve an artifact path and optionally require it to exist."""
    p = project_path(path)
    if required and not p.exists():
        raise FileNotFoundError(f"Missing artifact: {p}")
    return p


def read_csv(path: str | Path, n: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(artifact(path))
    print(f"{path}: {df.shape[0]} rows x {df.shape[1]} columns")
    if n is not None:
        display(df.head(n))
    return df


def read_optional_csv(path: str | Path, n: int | None = None) -> pd.DataFrame | None:
    p = artifact(path, required=False)
    if not p.exists() or p.stat().st_size == 0:
        print(f"Missing optional artifact: {path}")
        return None
    return read_csv(path, n=n)


def check_artifacts(paths: list[str | Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        p = project_path(path)
        size = p.stat().st_size if p.exists() else 0
        if not p.exists():
            status = "missing"
        elif size == 0:
            status = "empty"
        else:
            status = "ok"
        rows.append({"path": str(path), "exists": p.exists(), "size_bytes": size, "status": status})
    return pd.DataFrame(rows)


def require_artifacts(paths: list[str | Path]) -> pd.DataFrame:
    status = check_artifacts(paths)
    missing = status[status["status"] != "ok"]
    if not missing.empty:
        display(status)
        missing_paths = ", ".join(missing["path"].tolist())
        raise FileNotFoundError(f"Required artifacts are missing or empty: {missing_paths}")
    return status


def count_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.groupby(columns, dropna=False).size().reset_index(name="n").sort_values(columns)


def value_counts_frame(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame({"column": [column], "error": ["missing"]})
    out = df[column].value_counts(dropna=False).rename_axis(column).reset_index(name="n")
    out["percent"] = out["n"] / max(len(df), 1)
    return out


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    p = project_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"saved {path}: {df.shape[0]} rows x {df.shape[1]} columns")
    return p


def _label_set(series: pd.Series) -> set[str]:
    labels: set[str] = set()
    for value in series.dropna().tolist():
        text = str(value).strip().lower()
        if text in {"positive", "pos", "1", "true", "yes"}:
            labels.add("positive")
        elif text in {"negative", "neg", "0", "false", "no"}:
            labels.add("negative")
        elif text and text != "unknown":
            labels.add(text)
    return labels


def assert_no_participant_leakage(
    df: pd.DataFrame,
    participant_col: str = "participant_id",
    split_col: str = "split",
) -> None:
    required = {participant_col, split_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing leakage-check columns: {sorted(missing)}")
    split_counts = df.groupby(participant_col)[split_col].nunique(dropna=False)
    leaked = split_counts[split_counts > 1]
    if not leaked.empty:
        raise AssertionError(f"Participant leakage detected for {len(leaked)} participants")
    print(f"leakage gate passed: {split_counts.shape[0]} participants appear in one split each")


def assert_binary_labels_present(df: pd.DataFrame, label_col: str = "label_binary") -> None:
    if label_col not in df.columns:
        raise KeyError(f"Missing label column: {label_col}")
    labels = _label_set(df[label_col])
    if not {"positive", "negative"}.issubset(labels):
        raise AssertionError(f"Expected positive and negative labels, found {sorted(labels)}")
    print("label gate passed: both positive and negative classes are present")


def stop_if_validation_errors(issues: pd.DataFrame) -> None:
    if issues is None or issues.empty:
        print("validation gate passed: no issues")
        return
    if "severity" in issues.columns and (issues["severity"].astype(str).str.lower() == "error").any():
        display(issues)
        raise AssertionError("validation gate failed: fix error-level issues before continuing")
    print("validation gate passed with warnings only")
