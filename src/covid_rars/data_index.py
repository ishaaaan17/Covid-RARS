from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from covid_rars.labels import normalize_label
from covid_rars.schemas import INDEX_COLUMNS, MODALITY_COLUMNS


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".webm", ".m4a", ".aac"}

COUGH_HINTS = {"cough", "heavycough", "shallowcough", "cough-heavy", "cough-shallow"}
BREATH_HINTS = {"breath", "breathing", "deepbreath", "shallowbreath", "breath-deep", "breath-shallow"}
SPEECH_HINTS = {"vowel", "count", "counting", "a", "e", "o", "speech", "voice"}


def discover_audio_files(raw_dir: Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
    files = [
        p
        for p in raw_dir.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower() in AUDIO_EXTENSIONS
            and not p.name.startswith("._")
            and "__MACOSX" not in p.parts
        )
    ]
    return sorted(files)


def stable_recording_id(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]
    return f"rec_{digest}"


def infer_participant_id(
    path: Path,
    root: Path,
    known_participant_ids: set[str] | None = None,
) -> str:
    rel_parts = path.relative_to(root).parts
    known = {str(pid).strip() for pid in known_participant_ids or set() if str(pid).strip()}
    for part in rel_parts[:-1]:
        if part in known:
            return part
    if len(rel_parts) >= 2:
        parent = path.parent.name.strip()
        if parent and parent.lower() not in {"audio", "wav", "recordings"}:
            return parent
    stem = path.stem
    token = re.split(r"[_\-\s]+", stem)[0]
    return token or stable_recording_id(path, root)


def _normalized_tokens(path: Path) -> set[str]:
    text = path.as_posix().lower().replace("_", " ").replace("-", " ")
    return set(re.split(r"[^a-z0-9]+", text))


def infer_modality(path: Path) -> tuple[str, str]:
    text = path.as_posix().lower()
    tokens = _normalized_tokens(path)
    compact = text.replace("_", "").replace("-", "").replace(" ", "")

    if "cough" in tokens or any(hint in compact for hint in COUGH_HINTS):
        if "heavy" in tokens or "heavycough" in compact:
            return "cough", "heavy_cough"
        if "shallow" in tokens or "shallowcough" in compact:
            return "cough", "shallow_cough"
        return "cough", "cough"

    if "breath" in tokens or "breathing" in tokens or any(hint in compact for hint in BREATH_HINTS):
        if "deep" in tokens or "deepbreath" in compact:
            return "breath", "deep_breath"
        if "shallow" in tokens or "shallowbreath" in compact:
            return "breath", "shallow_breath"
        return "breath", "breath"

    if "count" in tokens or "counting" in tokens:
        if "fast" in tokens:
            return "speech", "counting_fast"
        return "speech", "counting_normal"

    if "vowel" in tokens:
        for vowel in ("a", "e", "o"):
            if vowel in tokens or f"vowel{vowel}" in compact:
                return "speech", f"vowel_{vowel}"
        return "speech", "vowel"

    stem = path.stem.lower()
    if stem in {"a", "e", "o"}:
        return "speech", f"vowel_{stem}"

    return "unknown", "unknown"


def load_metadata_candidates(raw_dir: Path) -> pd.DataFrame:
    """Load lightweight metadata files when present.

    The Coswara repository/mirrors have appeared in multiple layouts. This function
    scans common CSV/JSON metadata files and returns a best-effort participant table.
    It never raises if no metadata is found.
    """
    raw_dir = Path(raw_dir)
    frames: list[pd.DataFrame] = []
    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if path.suffix.lower() == ".csv" and any(k in lower_name for k in ("metadata", "combined", "csv", "data")):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            df["_metadata_source"] = path.as_posix()
            frames.append(df)
        elif path.suffix.lower() == ".json" and "metadata" in lower_name:
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(obj, list):
                frames.append(pd.DataFrame(obj))
            elif isinstance(obj, dict):
                frames.append(pd.DataFrame([obj]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _pick_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {str(c).lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    for col in columns:
        lowered = str(col).lower()
        if any(candidate.lower() in lowered for candidate in candidates):
            return col
    return None


def manual_quality_label(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "unknown", "na"}:
        return "unknown"
    try:
        numeric = float(text)
    except ValueError:
        if text in {"good", "ok", "acceptable", "valid"}:
            return "ok"
        if text in {"bad", "poor", "invalid", "reject"}:
            return "bad"
        return "unknown"
    if numeric <= 0:
        return "bad"
    return "ok"


def manual_quality_for_submodality(meta: dict[str, object], submodality: str) -> tuple[object, str]:
    terms = [t for t in re.split(r"[^a-z0-9]+", str(submodality).lower()) if t]
    for key, value in meta.items():
        key_terms = set(t for t in re.split(r"[^a-z0-9]+", str(key).lower()) if t)
        if "quality" not in key_terms:
            continue
        if all(term in key_terms for term in terms):
            return value, manual_quality_label(value)
    return "", "unknown"


def build_participant_metadata(raw_dir: Path) -> pd.DataFrame:
    meta = load_metadata_candidates(raw_dir)
    if meta.empty:
        return pd.DataFrame(columns=["participant_id", "label_raw", "label_binary"])

    participant_col = _pick_column(meta.columns, ["id", "participant_id", "subject_id", "user_id", "uid"])
    label_col = _pick_column(
        meta.columns,
        ["covid_status", "covid19_status", "status", "label", "test_status", "covid"],
    )
    age_col = _pick_column(meta.columns, ["age", "a"])
    gender_col = _pick_column(meta.columns, ["gender", "sex", "g"])
    country_col = _pick_column(meta.columns, ["country", "region", "location", "l_c"])
    date_col = _pick_column(meta.columns, ["recording_date", "record_date", "date", "created_at", "timestamp"])
    test_status_col = _pick_column(meta.columns, ["test_status", "test status"])
    test_type_col = _pick_column(meta.columns, ["testType", "test_type", "test type"])

    if participant_col is None:
        return pd.DataFrame(columns=["participant_id", "label_raw", "label_binary"])

    out = pd.DataFrame()
    out["participant_id"] = meta[participant_col].astype(str)
    out["label_raw"] = meta[label_col].astype(str) if label_col else "unknown"
    out["label_binary"] = out["label_raw"].map(normalize_label)
    out["age"] = meta[age_col] if age_col else ""
    out["gender"] = meta[gender_col] if gender_col else ""
    out["country"] = meta[country_col] if country_col else ""
    out["recording_date"] = meta[date_col] if date_col else ""
    symptom_cols = [
        col for col in ["cough", "cold", "fever", "diarrhoea", "loss_of_smell", "bd", "st", "ftg", "mp", "others_resp"]
        if col in meta.columns
    ]
    comorbidity_cols = [
        col for col in ["asthma", "ht", "diabetes", "ihd", "cld", "pneumonia", "smoker", "others_preexist"]
        if col in meta.columns
    ]
    out["symptoms_json"] = meta[symptom_cols].to_dict(orient="records") if symptom_cols else json.dumps({})
    out["symptoms_json"] = out["symptoms_json"].map(json.dumps) if symptom_cols else out["symptoms_json"]
    out["comorbidities_json"] = meta[comorbidity_cols].to_dict(orient="records") if comorbidity_cols else json.dumps({})
    out["comorbidities_json"] = out["comorbidities_json"].map(json.dumps) if comorbidity_cols else out["comorbidities_json"]
    out["test_status"] = meta[test_status_col] if test_status_col else ""
    out["test_type"] = meta[test_type_col] if test_type_col else ""
    quality_cols = [col for col in meta.columns if "quality" in str(col).lower()]
    for col in quality_cols:
        out[str(col)] = meta[col]
    out = out.drop_duplicates(subset=["participant_id"], keep="first")
    return out


def build_audio_index(raw_dir: Path, dataset: str = "coswara") -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    files = discover_audio_files(raw_dir)
    participant_meta = build_participant_metadata(raw_dir)
    meta_by_id = (
        participant_meta.set_index("participant_id").to_dict(orient="index")
        if not participant_meta.empty
        else {}
    )

    known_participant_ids = set(participant_meta["participant_id"].astype(str)) if not participant_meta.empty else set()

    rows = []
    for path in files:
        participant_id = infer_participant_id(path, raw_dir, known_participant_ids=known_participant_ids)
        modality, submodality = infer_modality(path)
        meta = meta_by_id.get(participant_id, {})
        label_raw = meta.get("label_raw", "unknown")
        manual_quality_score, manual_quality_label_value = manual_quality_for_submodality(meta, submodality)
        rows.append(
            {
                "participant_id": participant_id,
                "recording_id": stable_recording_id(path, raw_dir),
                "dataset": dataset,
                "modality": modality,
                "submodality": submodality,
                "audio_path": path.as_posix(),
                "label_raw": label_raw,
                "label_binary": normalize_label(label_raw),
                "recording_date": meta.get("recording_date", ""),
                "age": meta.get("age", ""),
                "gender": meta.get("gender", ""),
                "country": meta.get("country", ""),
                "symptoms_json": meta.get("symptoms_json", json.dumps({})),
                "comorbidities_json": meta.get("comorbidities_json", json.dumps({})),
                "test_status": meta.get("test_status", ""),
                "test_type": meta.get("test_type", ""),
                "manual_quality_score": manual_quality_score,
                "manual_quality_label": manual_quality_label_value,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=INDEX_COLUMNS)
    return df


def build_modality_availability(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for participant_id, group in metadata.groupby("participant_id"):
        counts = group["modality"].value_counts().to_dict()
        available = sorted(m for m in ("cough", "breath", "speech") if counts.get(m, 0) > 0)
        rows.append(
            {
                "participant_id": participant_id,
                "has_cough": counts.get("cough", 0) > 0,
                "has_breath": counts.get("breath", 0) > 0,
                "has_speech": counts.get("speech", 0) > 0,
                "n_cough": int(counts.get("cough", 0)),
                "n_breath": int(counts.get("breath", 0)),
                "n_speech": int(counts.get("speech", 0)),
                "complete_case": all(counts.get(m, 0) > 0 for m in ("cough", "breath", "speech")),
                "available_modalities": ",".join(available),
            }
        )
    return pd.DataFrame(rows, columns=MODALITY_COLUMNS)
