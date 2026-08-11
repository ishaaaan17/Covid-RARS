from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import sys
from typing import Iterable


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(packages: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        if package == "python":
            versions["python"] = sys.version.split()[0]
            continue
        try:
            versions[package] = importlib.metadata.version(package)
        except Exception:
            versions[package] = "unavailable"
    return versions


def artifact_record(path: str | Path) -> dict[str, object]:
    p = Path(path)
    record: dict[str, object] = {
        "path": p.as_posix(),
        "exists": p.exists(),
        "size_bytes": int(p.stat().st_size) if p.exists() else 0,
    }
    record["sha256"] = sha256_file(p) if p.exists() and p.is_file() else ""
    return record


def build_experiment_manifest(
    config: dict[str, object] | None = None,
    artifact_paths: list[str | Path] | None = None,
    include_packages: list[str] | None = None,
) -> dict[str, object]:
    include_packages = include_packages or [
        "python",
        "numpy",
        "pandas",
        "scikit-learn",
        "librosa",
        "soundfile",
        "torch",
        "torchaudio",
        "xgboost",
        "lightgbm",
        "catboost",
        "opensmile",
        "shap",
    ]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "config": config or {},
        "packages": package_versions(include_packages),
        "artifacts": [artifact_record(path) for path in (artifact_paths or [])],
    }
