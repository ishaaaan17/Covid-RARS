from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from covid_audio_btp.metrics import binary_metric_bundle, labels_to_binary


CONTEXT_COLUMNS = ("run_id", "protocol", "fold", "dataset")
COHORT_COLUMNS = (*CONTEXT_COLUMNS, "participant_key", "split")
SHA256_COLUMNS = (
    "audio_content_sha256",
    "manifest_sha256",
    "recording_intersection_sha256",
    "feature_artifact_sha256",
    "preprocessing_sha256",
    "checkpoint_hash",
)
BRANCH_IDENTITY_COLUMNS = (
    "source_family",
    "model",
    "checkpoint_hash",
    "representation",
    "feature_artifact_sha256",
    "feature_approval_id",
    "preprocessing_sha256",
)
REQUIRED_PREDICTION_COLUMNS = {
    *COHORT_COLUMNS,
    *BRANCH_IDENTITY_COLUMNS,
    "recording_key",
    "audio_content_sha256",
    "eligible",
    "manifest_sha256",
    "recording_intersection_sha256",
    "modality",
    "label_binary",
    "probability",
}
FOUR_BRANCH_COLUMNS = (
    "hst_cough",
    "hst_speech",
    "comparator_cough",
    "comparator_speech",
)
HYBRID_SOURCE_ARTIFACT_COLUMNS = (
    "hst_prediction_artifact_hash",
    "comparator_prediction_artifact_hash",
)
HYBRID_BRANCH_PROVENANCE_FIELDS = (
    "model",
    "checkpoint_hash",
    "representation",
    "feature_artifact_sha256",
    "feature_approval_id",
    "preprocessing_sha256",
    "branch_provenance_hash",
)
PRIMARY_MODALITIES = ("cough", "speech")
ANALYSIS_HIERARCHY_COLUMNS = (
    "analysis_scope",
    "analysis_role",
    "estimand_id",
    "multiplicity_family",
)
PRIMARY_ESTIMAND_ID = "primary_hst_vs_comparator_uniform_cough_speech_auroc"
PAIRED_DELTA_COLUMNS = (
    "run_id",
    "protocol",
    "fold",
    "dataset",
    "split",
    "candidate_family",
    "reference_family",
    "metric",
    "hybrid_value",
    "reference_value",
    "delta",
    "paired_participants",
    "comparison_binding_hash",
    "authenticated_registry_receipt_sha256",
    "authenticated_context_binding_sha256",
    *ANALYSIS_HIERARCHY_COLUMNS,
)
PREDICTION_ARTIFACT_COLUMNS = (
    *COHORT_COLUMNS,
    "recording_key",
    "audio_content_sha256",
    "eligible",
    "modality",
    "label_binary",
    "probability",
    "manifest_sha256",
    "recording_intersection_sha256",
    *BRANCH_IDENTITY_COLUMNS,
    "branch_provenance_hash",
)
FUSION_TABLE_FILENAMES = (
    ("predictions", "predictions.csv"),
    ("metrics", "metrics.csv"),
    ("weights", "weights.csv"),
    ("stacker_parameters", "stacker_parameters.csv"),
    ("complete_case_counts", "complete_case_counts.csv"),
    ("paired_deltas", "paired_deltas.csv"),
)
FUSION_EXPORT_SCHEMA_VERSION = 1
AUTHENTICATED_BINDING_SCHEMA_VERSION = 1
_AUTHENTICATED_BINDING_TOKEN = object()


def _is_lower_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_normalized_weight_vector(
    values: Iterable[float],
    *,
    name: str,
) -> np.ndarray:
    vector = np.asarray([float(value) for value in values], dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must contain at least one weight")
    if (~np.isfinite(vector)).any() or (vector < 0).any():
        raise ValueError(f"{name} weights must be finite and non-negative")
    try:
        total = math.fsum(float(value) for value in vector)
    except OverflowError as exc:
        raise ValueError(f"{name} weight sum overflowed") from exc
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"{name} weight sum must be finite and positive")
    if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{name} normalized weights must sum to one")
    normalized = vector / total
    if (~np.isfinite(normalized)).any():
        raise ValueError(f"{name} normalization produced nonfinite weights")
    return normalized


class _ImmutableMappingView(Mapping[str, float]):
    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, float]) -> None:
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))

    def __getitem__(self, key: str) -> float:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __setitem__(self, _key: str, _value: float) -> None:
        raise TypeError("Validation fusion weight provenance is immutable")


class _ImmutableStringMappingView(Mapping[str, str]):
    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, str]) -> None:
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))

    def __getitem__(self, key: str) -> str:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __setitem__(self, _key: str, _value: str) -> None:
        raise TypeError("Validation branch provenance is immutable")


class ValidationWeightMap(Mapping[str, float]):
    """Immutable weights whose fold-local validation provenance is explicit."""

    __slots__ = (
        "_normalized",
        "_raw_weights",
        "_branch_provenance_hashes",
        "run_id",
        "protocol",
        "fold",
        "dataset",
        "reference",
        "floor",
        "source_split",
        "_locked",
    )

    def __init__(
        self,
        normalized: dict[str, float],
        *,
        raw_weights: dict[str, float],
        run_id: str,
        protocol: str,
        fold: int,
        dataset: str,
        reference: float,
        floor: float,
        branch_provenance_hashes: Mapping[str, str] | None = None,
    ) -> None:
        normalized_copy = {str(name): float(value) for name, value in normalized.items()}
        raw_copy = {str(name): float(value) for name, value in raw_weights.items()}
        if not normalized_copy or set(normalized_copy) != set(raw_copy):
            raise ValueError("Normalized and raw validation weights must have identical keys")
        if any(not name or name != name.strip() for name in normalized_copy):
            raise ValueError("Validation weight branch names must be canonical strings")
        _validated_normalized_weight_vector(
            normalized_copy.values(),
            name="Validation fusion",
        )
        raw_values = np.asarray(list(raw_copy.values()), dtype=float)
        if (~np.isfinite(raw_values)).any() or (raw_values < 0).any():
            raise ValueError("Raw validation weights must be finite and non-negative")
        provenance_copy = dict(branch_provenance_hashes or {})
        if provenance_copy and set(provenance_copy) != set(normalized_copy):
            raise ValueError(
                "Validation branch provenance must match the exact weight branches"
            )
        if any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in provenance_copy.values()
        ):
            raise ValueError("Validation branch provenance hashes must be SHA-256 digests")
        object.__setattr__(self, "_normalized", MappingProxyType(normalized_copy))
        object.__setattr__(self, "_raw_weights", _ImmutableMappingView(raw_copy))
        object.__setattr__(
            self,
            "_branch_provenance_hashes",
            _ImmutableStringMappingView(provenance_copy),
        )
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "fold", fold)
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "reference", float(reference))
        object.__setattr__(self, "floor", float(floor))
        object.__setattr__(self, "source_split", "validation")
        object.__setattr__(self, "_locked", True)

    @property
    def raw_weights(self) -> Mapping[str, float]:
        return self._raw_weights

    @property
    def branch_provenance_hashes(self) -> Mapping[str, str]:
        return self._branch_provenance_hashes

    def __getitem__(self, key: str) -> float:
        return self._normalized[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._normalized)

    def __len__(self) -> int:
        return len(self._normalized)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items())

    def __repr__(self) -> str:
        return f"ValidationWeightMap({dict(self.items())!r})"

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise TypeError("Validation fusion weight provenance is immutable")
        object.__setattr__(self, name, value)


@dataclass(frozen=True)
class _FrozenStackerState:
    coef: tuple[tuple[float, ...], ...]
    coef_dtype: str
    intercept: tuple[float, ...]
    intercept_dtype: str
    classes: tuple[int, ...]
    classes_dtype: str
    n_features_in: int
    C: float
    class_weight: str
    l1_ratio: float
    penalty: str
    max_iter: int
    random_state: int


@dataclass(frozen=True)
class _PredictionProvenance:
    artifact_hash: str
    branch_hashes: tuple[tuple[str, str], ...]
    manifest_sha256: str
    recording_intersection_sha256: str
    upstream_artifact_hashes: tuple[tuple[str, str], ...] = ()
    upstream_branch_hashes: tuple[tuple[str, str], ...] = ()

    def branch_hash_map(self) -> dict[str, str]:
        return dict(self.branch_hashes)

    def upstream_artifact_hash_map(self) -> dict[str, str]:
        return dict(self.upstream_artifact_hashes)

    def upstream_branch_hash_map(self) -> dict[str, str]:
        return dict(self.upstream_branch_hashes)


@dataclass(frozen=True, init=False)
class AuthenticatedFusionBinding:
    """Frozen trust-boundary receipt supplied by an upstream artifact registry."""

    _receipt_json: str = field(repr=False)
    receipt_sha256: str

    def __init__(
        self,
        receipt_json: str,
        receipt_sha256: str,
        *,
        _token: object,
    ) -> None:
        if _token is not _AUTHENTICATED_BINDING_TOKEN:
            raise TypeError(
                "AuthenticatedFusionBinding must be created from a trusted registry receipt"
            )
        object.__setattr__(self, "_receipt_json", receipt_json)
        object.__setattr__(self, "receipt_sha256", receipt_sha256)

    @classmethod
    def from_registry_receipt(
        cls,
        receipt: Mapping[str, object],
        *,
        trusted_receipt_sha256: str,
    ) -> AuthenticatedFusionBinding:
        if not isinstance(receipt, Mapping):
            raise TypeError("Authenticated registry receipt must be a mapping")
        if not _is_lower_sha256(trusted_receipt_sha256):
            raise ValueError("Trusted registry receipt hash must be a lowercase SHA-256 digest")
        receipt_json = _canonical_json_bytes(dict(receipt)).decode("utf-8")
        computed = hashlib.sha256(receipt_json.encode("utf-8")).hexdigest()
        if computed != trusted_receipt_sha256:
            raise ValueError("Authenticated registry receipt hash does not match trusted digest")
        parsed = json.loads(receipt_json)
        _validate_authenticated_registry_receipt(parsed)
        return cls(
            receipt_json,
            trusted_receipt_sha256,
            _token=_AUTHENTICATED_BINDING_TOKEN,
        )

    @property
    def receipt_json(self) -> str:
        return self._receipt_json

    def verified_receipt(self) -> dict[str, object]:
        try:
            parsed = json.loads(self._receipt_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Authenticated registry receipt tampering detected") from exc
        canonical = _canonical_json_bytes(parsed).decode("utf-8")
        if canonical != self._receipt_json:
            raise ValueError("Authenticated registry receipt is not canonical or was tampered")
        computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if computed != self.receipt_sha256:
            raise ValueError("Authenticated registry receipt hash tampering detected")
        _validate_authenticated_registry_receipt(parsed)
        return parsed


def _validate_authenticated_registry_receipt(receipt: object) -> None:
    if not isinstance(receipt, dict):
        raise ValueError("Authenticated registry receipt must decode to an object")
    required_top = {
        "schema_version",
        "receipt_type",
        "registry_authority",
        "receipt_id",
        "contexts",
    }
    if set(receipt) != required_top:
        raise ValueError("Authenticated registry receipt has an invalid top-level schema")
    if receipt["schema_version"] != AUTHENTICATED_BINDING_SCHEMA_VERSION:
        raise ValueError("Authenticated registry receipt schema version is unsupported")
    if receipt["receipt_type"] != "hst_fusion_authenticated_registry":
        raise ValueError("Authenticated registry receipt type is invalid")
    for column in ("registry_authority", "receipt_id"):
        value = receipt[column]
        if type(value) is not str or not value or value != value.strip():
            raise ValueError(f"Authenticated registry receipt {column} is invalid")
    contexts = receipt["contexts"]
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("Authenticated registry receipt must contain context bindings")

    required_context = {
        *CONTEXT_COLUMNS,
        "manifest_receipt",
        "hst",
        "comparator",
    }
    seen_contexts: set[tuple[object, ...]] = set()
    for context in contexts:
        if not isinstance(context, dict) or set(context) != required_context:
            raise ValueError("Authenticated registry context binding has an invalid schema")
        for column in ("run_id", "protocol", "dataset"):
            value = context[column]
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"Authenticated registry context {column} is invalid")
        fold = context["fold"]
        if isinstance(fold, bool) or not isinstance(fold, int):
            raise ValueError("Authenticated registry context fold must be a canonical integer")
        context_key = tuple(context[column] for column in CONTEXT_COLUMNS)
        if context_key in seen_contexts:
            raise ValueError("Authenticated registry receipt contains duplicate contexts")
        seen_contexts.add(context_key)

        manifest = context["manifest_receipt"]
        required_manifest = {
            "receipt_id",
            "receipt_sha256",
            "manifest_sha256",
            "recording_intersection_sha256",
        }
        if not isinstance(manifest, dict) or set(manifest) != required_manifest:
            raise ValueError("Authenticated manifest receipt has an invalid schema")
        if type(manifest["receipt_id"]) is not str or not manifest["receipt_id"]:
            raise ValueError("Authenticated manifest receipt_id is invalid")
        for column in (
            "receipt_sha256",
            "manifest_sha256",
            "recording_intersection_sha256",
        ):
            if not _is_lower_sha256(manifest[column]):
                raise ValueError(f"Authenticated manifest {column} must be SHA-256")

        for family in ("hst", "comparator"):
            source = context[family]
            required_source = {"prediction_artifact_sha256", "branches"}
            if family == "comparator":
                required_source |= {"generation_id", "generation_receipt_sha256"}
            if not isinstance(source, dict) or set(source) != required_source:
                raise ValueError(
                    f"Authenticated {family} registry identity has an invalid schema"
                )
            if not _is_lower_sha256(source["prediction_artifact_sha256"]):
                raise ValueError(
                    f"Authenticated {family} prediction artifact must be SHA-256"
                )
            if family == "comparator":
                if type(source["generation_id"]) is not str or not source["generation_id"]:
                    raise ValueError("Approved comparator generation_id is invalid")
                if not _is_lower_sha256(source["generation_receipt_sha256"]):
                    raise ValueError("Approved comparator generation receipt must be SHA-256")
            branches = source["branches"]
            if not isinstance(branches, dict) or set(branches) != set(PRIMARY_MODALITIES):
                raise ValueError(
                    f"Authenticated {family} branches must be exact cough+speech identities"
                )
            required_branch = {*BRANCH_IDENTITY_COLUMNS, "branch_provenance_hash"}
            for modality, branch in branches.items():
                if not isinstance(branch, dict) or set(branch) != required_branch:
                    raise ValueError(
                        f"Authenticated {family} {modality} branch identity is incomplete"
                    )
                for column, value in branch.items():
                    if type(value) is not str or not value:
                        raise ValueError(
                            f"Authenticated {family} {modality} {column} is invalid"
                        )
                for column in (
                    "checkpoint_hash",
                    "feature_artifact_sha256",
                    "preprocessing_sha256",
                    "branch_provenance_hash",
                ):
                    if not _is_lower_sha256(branch[column]):
                        raise ValueError(
                            f"Authenticated {family} {modality} {column} must be SHA-256"
                        )


@dataclass(frozen=True)
class ValidationLogisticStacker:
    estimator: LogisticRegression
    feature_names: tuple[str, ...]
    frozen_state: _FrozenStackerState
    fitted_state_hash: str
    run_id: str
    protocol: str
    fold: int
    dataset: str
    branch_provenance_hashes: tuple[tuple[str, str], ...]
    source_split: str = "validation"


def _capture_stacker_state(estimator: LogisticRegression) -> _FrozenStackerState:
    coefficients = np.asarray(estimator.coef_)
    intercept = np.asarray(estimator.intercept_)
    classes = np.asarray(estimator.classes_)
    return _FrozenStackerState(
        coef=tuple(
            tuple(float(value) for value in row)
            for row in coefficients
        ),
        coef_dtype=coefficients.dtype.str,
        intercept=tuple(float(value) for value in intercept),
        intercept_dtype=intercept.dtype.str,
        classes=tuple(int(value) for value in classes),
        classes_dtype=classes.dtype.str,
        n_features_in=int(estimator.n_features_in_),
        C=float(estimator.C),
        class_weight=str(estimator.class_weight),
        l1_ratio=float(estimator.l1_ratio),
        penalty="l2" if float(estimator.l1_ratio) == 0.0 else "elasticnet",
        max_iter=int(estimator.max_iter),
        random_state=int(estimator.random_state),
    )


def _stacker_state_hash(state: _FrozenStackerState) -> str:
    return hashlib.sha256(repr(state).encode("utf-8")).hexdigest()


def _verify_stacker_state(model: ValidationLogisticStacker) -> _FrozenStackerState:
    frozen_hash = _stacker_state_hash(model.frozen_state)
    current_hash = _stacker_state_hash(_capture_stacker_state(model.estimator))
    if frozen_hash != model.fitted_state_hash or current_hash != model.fitted_state_hash:
        raise ValueError("Logistic stacker fitted-state tampering detected")
    return model.frozen_state


def _binary_logistic_probability(
    features: np.ndarray,
    state: _FrozenStackerState,
) -> np.ndarray:
    coefficients = np.asarray(state.coef, dtype=float)
    intercept = np.asarray(state.intercept, dtype=float)
    if (
        coefficients.shape != (1, state.n_features_in)
        or intercept.shape != (1,)
        or state.classes != (0, 1)
    ):
        raise ValueError("Logistic stacker frozen binary-state schema is invalid")
    logits = features @ coefficients[0] + intercept[0]
    probabilities = np.empty_like(logits, dtype=float)
    nonnegative = logits >= 0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exponential = np.exp(logits[~nonnegative])
    probabilities[~nonnegative] = exponential / (1.0 + exponential)
    return probabilities


@dataclass(frozen=True)
class HSTFusionResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    weights: pd.DataFrame
    stacker_parameters: pd.DataFrame
    complete_case_counts: pd.DataFrame
    paired_deltas: pd.DataFrame
    weights_content_hash: str = field(init=False)
    stacker_parameters_content_hash: str = field(init=False)
    table_content_hashes: Mapping[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name, _filename in FUSION_TABLE_FILENAMES:
            _validate_analysis_hierarchy_table(getattr(self, name), name=name)
        table_hashes = {
            name: _dataframe_content_hash(getattr(self, name))
            for name, _filename in FUSION_TABLE_FILENAMES
        }
        object.__setattr__(
            self,
            "table_content_hashes",
            MappingProxyType(table_hashes),
        )
        object.__setattr__(self, "weights_content_hash", table_hashes["weights"])
        object.__setattr__(
            self,
            "stacker_parameters_content_hash",
            table_hashes["stacker_parameters"],
        )

    def save_weights(self, path: Path) -> None:
        if _dataframe_content_hash(self.weights) != self.weights_content_hash:
            raise ValueError("Fusion weight table mutation/tampering detected")
        _atomic_write_csv(self.weights, Path(path))

    def save_stacker_parameters(self, path: Path) -> None:
        if (
            _dataframe_content_hash(self.stacker_parameters)
            != self.stacker_parameters_content_hash
        ):
            raise ValueError("Fusion stacker parameter mutation/tampering detected")
        _atomic_write_csv(self.stacker_parameters, Path(path))

    def save_generation(self, output_root: Path) -> dict[str, object]:
        return _save_fusion_generation(self, Path(output_root))


def _dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _dataframe_content_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(_dataframe_csv_bytes(frame)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _branch_identity_hash(identity: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(dict(identity))).hexdigest()


def _prediction_artifact_hash(frame: pd.DataFrame) -> str:
    optional_provenance = [
        column
        for column in (
            "source_prediction_artifact_hash",
            "upstream_branch_provenance_hash",
            "upstream_recording_intersection_sha256",
            "comparison_binding_hash",
        )
        if column in frame.columns
    ]
    ordered = frame[[*PREDICTION_ARTIFACT_COLUMNS, *optional_provenance]].sort_values(
        [*CONTEXT_COLUMNS, "split", "participant_key", "recording_key", "modality"],
        kind="stable",
    )
    return _dataframe_content_hash(ordered.reset_index(drop=True))


def _single_sha256(frame: pd.DataFrame, column: str, *, name: str) -> str:
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"{name} must contain exactly one {column}")
    value = values[0]
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} {column} must be a lowercase SHA-256 digest")
    return value


def _recording_intersection_hash(frame: pd.DataFrame) -> str:
    modalities = set(frame["modality"].astype(str))
    contract = (
        frame.loc[frame["modality"].isin(PRIMARY_MODALITIES)]
        if set(PRIMARY_MODALITIES).issubset(modalities)
        else frame
    )
    records = (
        contract[["split", "recording_key", "audio_content_sha256", "modality"]]
        .drop_duplicates()
        .sort_values(
            ["split", "recording_key", "audio_content_sha256", "modality"],
            kind="stable",
        )
        .to_dict(orient="records")
    )
    return hashlib.sha256(_canonical_json_bytes(records)).hexdigest()


def _prediction_provenance(
    frame: pd.DataFrame,
    modalities: tuple[str, ...],
) -> _PredictionProvenance:
    selected = frame.loc[frame["modality"].isin(modalities)].copy()
    branch_hashes: list[tuple[str, str]] = []
    upstream_artifact_hashes: list[tuple[str, str]] = []
    upstream_branch_hashes: list[tuple[str, str]] = []
    for modality in modalities:
        branch = selected.loc[selected["modality"].eq(modality)]
        values = branch["branch_provenance_hash"].drop_duplicates().tolist()
        if len(values) != 1:
            raise ValueError(f"Branch provenance is not unique for modality {modality!r}")
        branch_hashes.append((modality, str(values[0])))
        if "source_prediction_artifact_hash" in branch.columns:
            source_hashes = branch["source_prediction_artifact_hash"].drop_duplicates().tolist()
            if len(source_hashes) != 1:
                raise ValueError(
                    f"Upstream prediction artifact is not unique for modality {modality!r}"
                )
            upstream_artifact_hashes.append((modality, str(source_hashes[0])))
        if "upstream_branch_provenance_hash" in branch.columns:
            source_branch_hashes = branch[
                "upstream_branch_provenance_hash"
            ].drop_duplicates().tolist()
            if len(source_branch_hashes) != 1:
                raise ValueError(
                    f"Upstream branch provenance is not unique for modality {modality!r}"
                )
            upstream_branch_hashes.append((modality, str(source_branch_hashes[0])))
    return _PredictionProvenance(
        artifact_hash=_prediction_artifact_hash(selected),
        branch_hashes=tuple(branch_hashes),
        manifest_sha256=_single_sha256(
            selected,
            "manifest_sha256",
            name="prediction provenance",
        ),
        recording_intersection_sha256=_single_sha256(
            selected,
            "recording_intersection_sha256",
            name="prediction provenance",
        ),
        upstream_artifact_hashes=tuple(upstream_artifact_hashes),
        upstream_branch_hashes=tuple(upstream_branch_hashes),
    )


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _dataframe_csv_bytes(frame)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"Refusing to overwrite different existing fusion artifact: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() == payload:
                return
            raise FileExistsError(
                f"Refusing to overwrite different existing fusion artifact: {path}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _write_bytes_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_bytes_fsync(temporary, payload)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _verified_fusion_table_payloads(
    result: HSTFusionResult,
) -> tuple[dict[str, bytes], dict[str, dict[str, object]]]:
    payloads: dict[str, bytes] = {}
    artifacts: dict[str, dict[str, object]] = {}
    for name, filename in FUSION_TABLE_FILENAMES:
        frame = getattr(result, name)
        payload = _dataframe_csv_bytes(frame)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != result.table_content_hashes[name]:
            raise ValueError(f"Fusion result table {name!r} mutation/tampering detected")
        payloads[name] = payload
        artifacts[name] = {
            "relative_path": filename,
            "sha256": digest,
            "rows": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
        }
    return payloads, artifacts


def _verify_existing_fusion_generation(
    generation_path: Path,
    *,
    payloads: Mapping[str, bytes],
    receipt_payload: bytes,
) -> None:
    expected_names = {
        *(filename for _name, filename in FUSION_TABLE_FILENAMES),
        "checksums.json",
    }
    if not generation_path.is_dir():
        raise FileExistsError(
            f"Fusion generation path exists but is not a directory: {generation_path}"
        )
    actual_names = {path.name for path in generation_path.iterdir()}
    if actual_names != expected_names:
        raise FileExistsError("Existing fusion generation has a different file set")
    for name, filename in FUSION_TABLE_FILENAMES:
        if (generation_path / filename).read_bytes() != payloads[name]:
            raise FileExistsError(
                f"Existing fusion generation artifact differs: {filename}"
            )
    if (generation_path / "checksums.json").read_bytes() != receipt_payload:
        raise FileExistsError("Existing fusion generation receipt differs")


def _save_fusion_generation(
    result: HSTFusionResult,
    output_root: Path,
) -> dict[str, object]:
    payloads, artifacts = _verified_fusion_table_payloads(result)
    generation_id = hashlib.sha256(
        _canonical_json_bytes(
            {
                "schema_version": FUSION_EXPORT_SCHEMA_VERSION,
                "artifacts": artifacts,
            }
        )
    ).hexdigest()
    generation_relative = f"generations/{generation_id}"
    receipt: dict[str, object] = {
        "schema_version": FUSION_EXPORT_SCHEMA_VERSION,
        "receipt_type": "hst_fusion_generation",
        "status": "success",
        "generation_id": generation_id,
        "generation_path": generation_relative,
        "artifacts": artifacts,
    }
    receipt["record_hash"] = hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest()
    receipt_payload = (
        json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    ).encode("utf-8")

    output_root.mkdir(parents=True, exist_ok=True)
    generations_root = output_root / "generations"
    generations_root.mkdir(parents=True, exist_ok=True)
    final_generation = output_root / Path(generation_relative)
    staging = output_root / f".hst-fusion-staging-{uuid.uuid4().hex}"
    try:
        if final_generation.exists():
            _verify_existing_fusion_generation(
                final_generation,
                payloads=payloads,
                receipt_payload=receipt_payload,
            )
        else:
            staging.mkdir()
            for name, filename in FUSION_TABLE_FILENAMES:
                _write_bytes_fsync(staging / filename, payloads[name])
            _write_bytes_fsync(staging / "checksums.json", receipt_payload)
            _fsync_directory(staging)
            try:
                os.replace(staging, final_generation)
            except OSError:
                if not final_generation.exists():
                    raise
                _verify_existing_fusion_generation(
                    final_generation,
                    payloads=payloads,
                    receipt_payload=receipt_payload,
                )
            _fsync_directory(generations_root)
        _atomic_replace_bytes(output_root / "current.json", receipt_payload)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _ordered_modalities(values: Iterable[object]) -> tuple[str, ...]:
    modalities = {str(value) for value in values}
    if modalities == set(FOUR_BRANCH_COLUMNS):
        return FOUR_BRANCH_COLUMNS
    conventional = [name for name in ("cough", "breath", "speech") if name in modalities]
    extras = sorted(modalities - set(conventional))
    return tuple([*conventional, *extras])


def _normalize_context_schema(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    normalized = frame.copy()
    for column in ("run_id", "protocol", "dataset"):
        values = normalized[column].tolist()
        if any(type(value) is not str for value in values):
            raise TypeError(f"{name} {column} values must be canonical strings")
        if any(not value or value != value.strip() for value in values):
            raise ValueError(f"{name} {column} values must be non-empty canonical strings")

    canonical_folds: list[int] = []
    for value in normalized["fold"].tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} fold values must be canonical integers")
        canonical_folds.append(int(value))
    normalized["fold"] = canonical_folds
    return normalized


def _require_participant_label_invariant(frame: pd.DataFrame, *, name: str) -> None:
    identity = ["run_id", "protocol", "dataset", "participant_key"]
    label_counts = frame.groupby(identity, dropna=False)["label_binary"].nunique()
    if label_counts.gt(1).any():
        raise ValueError(
            f"{name} contains contradictory participant ground-truth labels "
            "across folds or splits"
        )


def _validate_predictions(predictions: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise ValueError(f"{name} must be a non-empty participant prediction table")
    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

    frame = _normalize_context_schema(predictions, name=name).reset_index(drop=True)
    for column in (
        "participant_key",
        "recording_key",
        "audio_content_sha256",
        "split",
        "modality",
        "manifest_sha256",
        "recording_intersection_sha256",
        *BRANCH_IDENTITY_COLUMNS,
    ):
        values = frame[column].tolist()
        if any(type(value) is not str for value in values):
            raise TypeError(f"{name} {column} values must be canonical strings")
        if any(not value or value != value.strip() for value in values):
            raise ValueError(f"{name} {column} values must be non-empty canonical strings")
    label_values = frame["label_binary"].tolist()
    if any(type(value) is not str for value in label_values):
        raise TypeError(f"{name} label_binary values must be canonical strings")
    if any(value not in {"negative", "positive"} for value in label_values):
        raise ValueError(
            f"{name} label_binary values must use canonical negative/positive strings"
        )
    eligible_values = frame["eligible"].tolist()
    if any(type(value) is not bool for value in eligible_values):
        raise TypeError(f"{name} eligible values must be canonical booleans")
    if not all(eligible_values):
        raise ValueError(f"{name} contains ineligible recording predictions")
    for column in (*COHORT_COLUMNS, "recording_key", "modality", "label_binary"):
        invalid = frame[column].isna() | frame[column].astype(str).str.strip().eq("")
        if invalid.any():
            raise ValueError(f"{name} contains empty {column} values")
    frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce")
    if (~np.isfinite(frame["probability"])).any() or not frame["probability"].between(0.0, 1.0).all():
        raise ValueError(f"{name} probabilities must be finite and within [0, 1]")

    branch_key = [*COHORT_COLUMNS, "recording_key", "modality"]
    if frame.duplicated(branch_key).any():
        raise ValueError(f"{name} contains duplicate recording/fold/modality rows")
    for identity_column, identity_name in (
        ("recording_key", "recording_key"),
        ("audio_content_sha256", "audio content identity"),
    ):
        identity_splits = frame.groupby(
            [*CONTEXT_COLUMNS, identity_column],
            dropna=False,
        )["split"].agg(lambda values: frozenset(values))
        reused = identity_splits.loc[identity_splits.map(len).gt(1)]
        if not reused.empty:
            split_names = sorted(
                {split for splits in reused for split in splits},
                key=lambda split: (split != "validation", split),
            )
            raise ValueError(
                f"{name} reuses {identity_name} across "
                f"{'/'.join(split_names)} splits"
            )
    recording_owners = frame.groupby(
        [*CONTEXT_COLUMNS, "recording_key", "modality"],
        dropna=False,
    )["participant_key"].nunique()
    if recording_owners.gt(1).any():
        raise ValueError(f"{name} maps one recording branch to multiple participants")
    label_counts = frame.groupby(list(COHORT_COLUMNS), dropna=False)["label_binary"].nunique()
    if label_counts.gt(1).any():
        raise ValueError(f"{name} contains conflicting participant labels")
    _require_participant_label_invariant(frame, name=name)

    for _context, context_frame in frame.groupby(
        list(CONTEXT_COLUMNS),
        sort=False,
        dropna=False,
    ):
        _single_sha256(context_frame, "manifest_sha256", name=name)
        supplied_intersection = _single_sha256(
            context_frame,
            "recording_intersection_sha256",
            name=name,
        )
        actual_intersection = _recording_intersection_hash(context_frame)
        has_complete_split_contract = context_frame["split"].nunique() > 1
        modalities = set(context_frame["modality"].astype(str))
        can_reconstruct_contract = (
            set(PRIMARY_MODALITIES).issubset(modalities)
            or set(FOUR_BRANCH_COLUMNS).issubset(modalities)
        )
        if (
            has_complete_split_contract
            and can_reconstruct_contract
            and supplied_intersection != actual_intersection
        ):
            raise ValueError(
                f"{name} recording intersection hash does not match eligible recording keys"
            )

    branch_group = [*CONTEXT_COLUMNS, "modality"]
    optional_branch_identity_columns = [
        column
        for column in (
            "source_prediction_artifact_hash",
            "upstream_branch_provenance_hash",
            "upstream_recording_intersection_sha256",
            "comparison_binding_hash",
        )
        if column in frame.columns
    ]
    identity_counts = frame.groupby(branch_group, dropna=False)[
        [
            *BRANCH_IDENTITY_COLUMNS,
            "manifest_sha256",
            "recording_intersection_sha256",
            *optional_branch_identity_columns,
        ]
    ].nunique(dropna=False)
    if identity_counts.gt(1).any(axis=None):
        raise ValueError(
            f"{name} branch identity changes across validation/test; "
            "model/checkpoint_hash/representation substitution is forbidden"
        )
    frame["branch_provenance_hash"] = ""
    for branch_key, group in frame.groupby(branch_group, sort=False, dropna=False):
        exemplar = group.iloc[0]
        identity = {
            **dict(zip(branch_group, branch_key)),
            **{
                column: exemplar[column]
                for column in (
                    *BRANCH_IDENTITY_COLUMNS,
                    "manifest_sha256",
                    "recording_intersection_sha256",
                    *optional_branch_identity_columns,
                )
            },
        }
        frame.loc[group.index, "branch_provenance_hash"] = _branch_identity_hash(identity)

    for column in SHA256_COLUMNS:
        invalid_digest = ~frame[column].astype(str).str.fullmatch(r"[0-9a-f]{64}")
        if invalid_digest.any():
            raise ValueError(f"{name} {column} must contain lowercase SHA-256 digests")
    for column in optional_branch_identity_columns:
        invalid_digest = ~frame[column].astype(str).str.fullmatch(r"[0-9a-f]{64}")
        if invalid_digest.any():
            raise ValueError(f"{name} {column} must contain lowercase SHA-256 digests")

    participant_splits = frame.groupby(
        [*CONTEXT_COLUMNS, "participant_key"],
        dropna=False,
    )["split"].agg(lambda values: frozenset(values))
    overlap = participant_splits.loc[participant_splits.map(len).gt(1)]
    if not overlap.empty:
        split_names = sorted({str(split) for splits in overlap for split in splits})
        raise ValueError(
            f"{name} participant overlap across validation/test or other splits: {split_names}"
        )
    return frame


def _single_context(frame: pd.DataFrame, *, purpose: str) -> dict[str, object]:
    values: dict[str, object] = {}
    for column in CONTEXT_COLUMNS:
        unique = frame[column].drop_duplicates().tolist()
        if len(unique) != 1:
            qualifier = "fold" if column == "fold" else column
            raise ValueError(f"{purpose} must be {qualifier}-local; found {len(unique)} values")
        values[column] = unique[0]
    return values


def _aggregate_recording_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = [*COHORT_COLUMNS, "modality"]
    carry_columns = [
        "eligible",
        "manifest_sha256",
        "recording_intersection_sha256",
        *BRANCH_IDENTITY_COLUMNS,
        "branch_provenance_hash",
        *[
            column
            for column in (
                "source_prediction_artifact_hash",
                "upstream_branch_provenance_hash",
                "upstream_recording_intersection_sha256",
            )
            if column in frame.columns
        ],
    ]
    rows: list[dict[str, object]] = []
    for group_key, group in frame.groupby(group_columns, sort=True, dropna=False):
        labels = group["label_binary"].drop_duplicates().tolist()
        if len(labels) != 1:
            raise ValueError("Recording aggregation found conflicting participant labels")
        for column in carry_columns:
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(
                    f"Recording aggregation found conflicting upstream identity {column!r}"
                )
        recording_keys = sorted(group["recording_key"].tolist())
        audio_content_hashes = sorted(group["audio_content_sha256"].tolist())
        rows.append(
            {
                **dict(zip(group_columns, group_key)),
                "recording_key": "participant-aggregate::"
                + hashlib.sha256(_canonical_json_bytes(recording_keys)).hexdigest(),
                "audio_content_sha256": hashlib.sha256(
                    _canonical_json_bytes(audio_content_hashes)
                ).hexdigest(),
                "label_binary": labels[0],
                "probability": float(group["probability"].mean()),
                **{column: group[column].iloc[0] for column in carry_columns},
                "n_recordings_aggregated": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _pivot_predictions(
    predictions: pd.DataFrame,
    *,
    modalities: tuple[str, ...] | None = None,
    name: str,
) -> tuple[pd.DataFrame, tuple[str, ...], _PredictionProvenance]:
    frame = _validate_predictions(predictions, name=name)
    _single_context(frame, purpose=name)
    selected_modalities = modalities or _ordered_modalities(frame["modality"].unique())
    if not selected_modalities:
        raise ValueError(f"{name} has no fusion modalities")
    missing = sorted(set(selected_modalities) - set(frame["modality"].astype(str)))
    if missing:
        raise ValueError(f"{name} is missing required modalities: {missing}")
    frame = frame.loc[frame["modality"].astype(str).isin(selected_modalities)].copy()
    provenance = _prediction_provenance(frame, selected_modalities)
    frame = _aggregate_recording_predictions(frame)
    index_columns = [*COHORT_COLUMNS, "label_binary"]
    matrix = frame.pivot(
        index=index_columns,
        columns="modality",
        values="probability",
    ).reset_index()
    matrix.columns.name = None
    for modality in selected_modalities:
        if modality not in matrix:
            matrix[modality] = np.nan
    return (
        matrix[[*index_columns, *selected_modalities]],
        selected_modalities,
        provenance,
    )


def _source_family(predictions: pd.DataFrame) -> str:
    if "source_family" not in predictions:
        return "unspecified"
    values = predictions["source_family"].dropna().astype(str).unique().tolist()
    if len(values) > 1:
        raise ValueError("Fusion input mixes source families")
    return values[0] if values else "unspecified"


def _require_source_family(frame: pd.DataFrame, *, expected: str, name: str) -> None:
    values = frame["source_family"].drop_duplicates().tolist()
    if values != [expected]:
        raise ValueError(
            f"{name} source_family must be exactly {expected!r}; found {values!r}"
        )


def _branch_identity_values(frame: pd.DataFrame, modality: str) -> dict[str, str]:
    branch = frame.loc[frame["modality"].eq(modality)]
    if branch.empty:
        raise ValueError(f"Missing branch provenance for modality {modality!r}")
    values: dict[str, str] = {}
    for column in (*BRANCH_IDENTITY_COLUMNS, "branch_provenance_hash"):
        unique = branch[column].drop_duplicates().tolist()
        if len(unique) != 1:
            raise ValueError(
                f"Branch provenance column {column!r} is not unique for {modality!r}"
            )
        values[column] = str(unique[0])
    return values


def _primary_recording_binding_hash(
    hst: pd.DataFrame,
    comparator: pd.DataFrame,
    *,
    authenticated_registry_receipt_sha256: str | None = None,
    authenticated_context_binding_sha256: str | None = None,
) -> str:
    hst_primary = hst.loc[hst["modality"].isin(PRIMARY_MODALITIES)].copy()
    comparator_primary = comparator.loc[
        comparator["modality"].isin(PRIMARY_MODALITIES)
    ].copy()
    key_columns = [
        "split",
        "recording_key",
        "audio_content_sha256",
        "modality",
        "participant_key",
        "label_binary",
    ]
    hst_keys = hst_primary[key_columns].sort_values(key_columns, kind="stable").reset_index(
        drop=True
    )
    comparator_keys = comparator_primary[key_columns].sort_values(
        key_columns,
        kind="stable",
    ).reset_index(drop=True)
    if not hst_keys.equals(comparator_keys):
        raise ValueError(
            "Primary HST/comparator recording intersection differs before participant aggregation"
        )

    hst_manifest = _single_sha256(hst_primary, "manifest_sha256", name="primary HST")
    comparator_manifest = _single_sha256(
        comparator_primary,
        "manifest_sha256",
        name="primary comparator",
    )
    if hst_manifest != comparator_manifest:
        raise ValueError("Primary HST/comparator manifest identity differs")
    hst_intersection = _single_sha256(
        hst_primary,
        "recording_intersection_sha256",
        name="primary HST",
    )
    comparator_intersection = _single_sha256(
        comparator_primary,
        "recording_intersection_sha256",
        name="primary comparator",
    )
    actual_intersection = _recording_intersection_hash(hst_primary)
    if (
        hst_intersection != comparator_intersection
        or hst_intersection != actual_intersection
    ):
        raise ValueError("Primary HST/comparator recording intersection hash differs")

    source_identities: dict[str, object] = {}
    for source_name, source in (("hst", hst_primary), ("comparator", comparator_primary)):
        branch_identities = {
            modality: {
                column: source.loc[source["modality"].eq(modality), column].iloc[0]
                for column in BRANCH_IDENTITY_COLUMNS
            }
            for modality in PRIMARY_MODALITIES
        }
        source_identities[source_name] = {
            "prediction_artifact_sha256": _prediction_artifact_hash(source),
            "branches": branch_identities,
        }
    payload = {
        "estimand_id": PRIMARY_ESTIMAND_ID,
        "manifest_sha256": hst_manifest,
        "recording_intersection_sha256": actual_intersection,
        "eligible_recording_keys": hst_keys.to_dict(orient="records"),
        "source_identities": source_identities,
        "authenticated_registry_receipt_sha256": (
            authenticated_registry_receipt_sha256 or "not_applicable"
        ),
        "authenticated_context_binding_sha256": (
            authenticated_context_binding_sha256 or "not_applicable"
        ),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _verify_authenticated_context_binding(
    entry: Mapping[str, object],
    hst: pd.DataFrame,
    comparator: pd.DataFrame,
) -> str:
    context = _single_context(hst, purpose="authenticated HST context")
    comparator_context = _single_context(
        comparator,
        purpose="authenticated comparator context",
    )
    if context != comparator_context:
        raise ValueError("Authenticated HST/comparator contexts differ")
    for column in CONTEXT_COLUMNS:
        if entry[column] != context[column]:
            raise ValueError(f"Authenticated registry {column} does not match fusion input")

    hst_primary = hst.loc[hst["modality"].isin(PRIMARY_MODALITIES)].copy()
    comparator_primary = comparator.loc[
        comparator["modality"].isin(PRIMARY_MODALITIES)
    ].copy()
    manifest = entry["manifest_receipt"]
    if not isinstance(manifest, Mapping):
        raise ValueError("Authenticated manifest receipt is invalid")
    actual_manifest = _single_sha256(
        hst_primary,
        "manifest_sha256",
        name="authenticated HST input",
    )
    comparator_manifest = _single_sha256(
        comparator_primary,
        "manifest_sha256",
        name="authenticated comparator input",
    )
    actual_intersection = _single_sha256(
        hst_primary,
        "recording_intersection_sha256",
        name="authenticated HST input",
    )
    comparator_intersection = _single_sha256(
        comparator_primary,
        "recording_intersection_sha256",
        name="authenticated comparator input",
    )
    if not (
        manifest["manifest_sha256"]
        == actual_manifest
        == comparator_manifest
        and manifest["recording_intersection_sha256"]
        == actual_intersection
        == comparator_intersection
    ):
        raise ValueError("Authenticated manifest receipt does not match fusion inputs")

    for family, frame in (("hst", hst_primary), ("comparator", comparator_primary)):
        approved = entry[family]
        if not isinstance(approved, Mapping):
            raise ValueError(f"Authenticated {family} registry identity is invalid")
        actual_artifact = _prediction_artifact_hash(frame)
        if approved["prediction_artifact_sha256"] != actual_artifact:
            raise ValueError(
                f"Authenticated {family} prediction artifact identity does not match"
            )
        approved_branches = approved["branches"]
        if not isinstance(approved_branches, Mapping):
            raise ValueError(f"Authenticated {family} branch registry is invalid")
        for modality in PRIMARY_MODALITIES:
            branch = frame.loc[frame["modality"].eq(modality)]
            actual_identity = {
                **{
                    column: branch[column].iloc[0]
                    for column in BRANCH_IDENTITY_COLUMNS
                },
                "branch_provenance_hash": branch["branch_provenance_hash"].iloc[0],
            }
            if dict(approved_branches[modality]) != actual_identity:
                raise ValueError(
                    f"Authenticated {family} {modality} branch identity is not approved"
                )
    return hashlib.sha256(_canonical_json_bytes(dict(entry))).hexdigest()


def _fuse_matrix(
    matrix: pd.DataFrame,
    modalities: tuple[str, ...],
    weights: np.ndarray,
    *,
    fusion_method: str,
    source_family: str,
    complete_case_only: bool,
    provenance: _PredictionProvenance,
) -> pd.DataFrame:
    weights = _validated_normalized_weight_vector(weights, name="Fusion")
    probabilities = matrix[list(modalities)].to_numpy(dtype=float)
    available = np.isfinite(probabilities)
    complete = available.all(axis=1)
    selected = complete if complete_case_only else available.any(axis=1)
    matrix = matrix.loc[selected].copy()
    probabilities = probabilities[selected]
    available = available[selected]
    complete = complete[selected]

    weighted = np.where(available, probabilities * weights, 0.0)
    denominator = np.where(available, weights, 0.0).sum(axis=1)
    if (~np.isfinite(denominator)).any() or (denominator <= 0).any():
        raise ValueError("Fusion has a nonfinite or zero denominator for an included row")
    fused_probability = np.divide(
        weighted.sum(axis=1),
        denominator,
        out=np.full(len(matrix), np.nan, dtype=float),
        where=denominator > 0,
    )
    if (
        (~np.isfinite(fused_probability)).any()
        or not pd.Series(fused_probability).between(0.0, 1.0).all()
    ):
        raise ValueError("Fusion must produce finite probabilities within [0, 1]")
    available_names = [
        ",".join(sorted(modality for modality, present in zip(modalities, row) if present))
        for row in available
    ]
    output = matrix[list(COHORT_COLUMNS) + ["label_binary"]].copy()
    output["probability"] = fused_probability
    output["modality"] = "multimodal"
    output["modality_combination"] = "+".join(modalities)
    output["fusion_method"] = fusion_method
    output["source_family"] = source_family
    output["available_modalities"] = available_names
    output["n_modalities"] = available.sum(axis=1).astype(int)
    output["complete_case"] = complete.astype(bool)
    output["model"] = fusion_method
    branch_hashes = provenance.branch_hash_map()
    upstream_artifact_hashes: dict[str, str] = {}
    for branch, artifact_hash in provenance.upstream_artifact_hashes:
        source = (
            "hst"
            if branch.startswith("hst_")
            else "comparator"
            if branch.startswith("comparator_")
            else branch
        )
        previous = upstream_artifact_hashes.setdefault(source, artifact_hash)
        if previous != artifact_hash:
            raise ValueError(f"Hybrid source {source!r} has conflicting artifact hashes")
    upstream_branch_hashes = provenance.upstream_branch_hash_map()
    serialized_branch_hashes = _canonical_json_bytes(branch_hashes).decode("utf-8")
    serialized_upstream_artifacts = _canonical_json_bytes(
        upstream_artifact_hashes
    ).decode("utf-8")
    serialized_upstream_branches = _canonical_json_bytes(
        upstream_branch_hashes
    ).decode("utf-8")
    output["manifest_sha256"] = provenance.manifest_sha256
    output["recording_intersection_sha256"] = provenance.recording_intersection_sha256
    output["source_prediction_artifact_hash"] = provenance.artifact_hash
    output["source_branch_provenance_hashes"] = serialized_branch_hashes
    output["upstream_source_prediction_artifact_hashes"] = (
        serialized_upstream_artifacts
    )
    output["upstream_branch_provenance_hashes"] = serialized_upstream_branches
    output["checkpoint_hash"] = hashlib.sha256(
        _canonical_json_bytes(
            {
                "fusion_method": fusion_method,
                "source_prediction_artifact_hash": provenance.artifact_hash,
                "source_branch_provenance_hashes": branch_hashes,
                "upstream_source_prediction_artifact_hashes": upstream_artifact_hashes,
                "upstream_branch_provenance_hashes": upstream_branch_hashes,
                "manifest_sha256": provenance.manifest_sha256,
                "recording_intersection_sha256": (
                    provenance.recording_intersection_sha256
                ),
                "weights": [float(value) for value in weights],
            }
        )
    ).hexdigest()
    output["representation"] = "participant_probability_fusion"
    if fusion_method == "available_modalities_validation_weighted_auprc":
        designation = _analysis_designation(
            role="sensitivity",
            estimand_id="sensitivity_standalone_available_modality_fusion",
            multiplicity_family="sensitivity_standalone_fusion",
        )
    elif fusion_method in {
        "legacy_validation_weighted_auprc",
        "stacked_logistic_validation",
    }:
        designation = _analysis_designation(
            role="secondary",
            estimand_id=f"secondary_standalone_{fusion_method}",
            multiplicity_family="secondary_standalone_fusion",
        )
    else:
        designation = _analysis_designation(
            role="exploratory",
            estimand_id=f"exploratory_standalone_{fusion_method}",
            multiplicity_family="exploratory_standalone_fusion",
        )
    for column in ANALYSIS_HIERARCHY_COLUMNS:
        output[column] = designation[column]
    validation_rows = output["split"].eq("validation")
    if validation_rows.any():
        selection = _selection_designation(
            estimand_id=f"selection_standalone_{fusion_method}"
        )
        for column in ANALYSIS_HIERARCHY_COLUMNS:
            output.loc[validation_rows, column] = selection[column]
    output["comparison_binding_hash"] = "not_applicable"
    return output.sort_values(list(COHORT_COLUMNS), kind="stable").reset_index(drop=True)


def legacy_validation_auprc_weights(
    validation_metrics: pd.DataFrame,
    *,
    floor: float = 0.01,
    reference: float = 0.5,
) -> ValidationWeightMap:
    """Freeze one fold's weights from validation AUPRC and ignore test rows."""
    if not isinstance(validation_metrics, pd.DataFrame) or validation_metrics.empty:
        raise ValueError("validation metrics must be a non-empty table")
    split_column = "split" if "split" in validation_metrics else "metric_split"
    required = {split_column, *CONTEXT_COLUMNS, "modality", "auprc"}
    missing = sorted(required - set(validation_metrics.columns))
    if missing:
        raise ValueError(f"validation metrics are missing columns: {missing}")
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("floor must be finite and positive")
    if not np.isfinite(reference):
        raise ValueError("reference must be finite")

    normalized_metrics = _normalize_context_schema(
        validation_metrics,
        name="validation metrics",
    )
    if any(type(value) is not str for value in normalized_metrics[split_column].tolist()):
        raise TypeError("validation metrics split values must be canonical strings")
    if any(type(value) is not str for value in normalized_metrics["modality"].tolist()):
        raise TypeError("validation metrics modality values must be canonical strings")
    selected = normalized_metrics.loc[
        normalized_metrics[split_column].eq("validation")
    ].copy()
    if selected.empty:
        raise ValueError("No validation rows are available for fusion-weight estimation")
    context = _single_context(selected, purpose="validation AUPRC weighting")
    if selected["modality"].astype(str).duplicated().any():
        raise ValueError("Validation metrics contain duplicate modality rows within one fold")
    selected["auprc"] = pd.to_numeric(selected["auprc"], errors="coerce")
    if (~np.isfinite(selected["auprc"])).any() or not selected["auprc"].between(0.0, 1.0).all():
        raise ValueError("Validation AUPRC values must be finite and within [0, 1]")

    modality_names = [str(value) for value in selected["modality"]]
    auprc_values = selected["auprc"].to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        raw_values = np.maximum(auprc_values - float(reference), float(floor))
    if (~np.isfinite(raw_values)).any() or (raw_values <= 0).any():
        raise ValueError("Validation AUPRC weighting produced nonfinite/overflow raw weights")

    scale = float(np.max(raw_values))
    scaled = raw_values / scale
    scaled_total = float(np.sum(scaled, dtype=np.float64))
    if (
        not np.isfinite(scale)
        or scale <= 0
        or not np.isfinite(scaled_total)
        or scaled_total <= 0
    ):
        raise ValueError("Validation AUPRC weighting cannot be normalized safely")
    normalized_values = scaled / scaled_total
    if (
        (~np.isfinite(normalized_values)).any()
        or (normalized_values < 0).any()
        or not np.isclose(
            float(np.sum(normalized_values, dtype=np.float64)),
            1.0,
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        raise ValueError("Validation AUPRC normalized weights must be finite and sum to one")

    raw = dict(zip(modality_names, raw_values.tolist()))
    normalized = dict(zip(modality_names, normalized_values.tolist()))
    branch_provenance_hashes: dict[str, str] = {}
    if "branch_provenance_hash" in selected.columns:
        for modality, group in selected.groupby("modality", sort=False, dropna=False):
            hashes = group["branch_provenance_hash"].drop_duplicates().tolist()
            if len(hashes) != 1:
                raise ValueError(
                    f"Validation branch provenance is not unique for {modality!r}"
                )
            branch_provenance_hashes[str(modality)] = str(hashes[0])
    return ValidationWeightMap(
        normalized,
        raw_weights=raw,
        run_id=context["run_id"],
        protocol=context["protocol"],
        fold=context["fold"],
        dataset=context["dataset"],
        reference=reference,
        floor=floor,
        branch_provenance_hashes=branch_provenance_hashes,
    )


def _validated_fixed_weights(
    weights: ValidationWeightMap,
    *,
    modalities: tuple[str, ...],
    context: dict[str, object],
    provenance: _PredictionProvenance,
) -> np.ndarray:
    if not isinstance(weights, ValidationWeightMap) or weights.source_split != "validation":
        raise ValueError("Fixed fusion weights must be validation-derived and provenance-frozen")
    expected_context = {
        "run_id": weights.run_id,
        "protocol": weights.protocol,
        "fold": weights.fold,
        "dataset": weights.dataset,
    }
    for column, expected in expected_context.items():
        if context[column] != expected:
            raise ValueError(f"Fixed fusion weight {column} mismatch")
    if set(weights) != set(modalities):
        raise ValueError("Fixed fusion weights must match the exact modality set")
    if not weights.branch_provenance_hashes:
        raise ValueError("Fixed fusion weights have no validation branch provenance")
    if dict(weights.branch_provenance_hashes) != provenance.branch_hash_map():
        raise ValueError(
            "Fixed fusion branch identity/provenance mismatch prevents substitution"
        )
    return _validated_normalized_weight_vector(
        (weights[modality] for modality in modalities),
        name="Fixed fusion",
    )


def fuse_uniform_complete_case(predictions: pd.DataFrame) -> pd.DataFrame:
    matrix, modalities, provenance = _pivot_predictions(
        predictions,
        name="uniform fusion predictions",
    )
    weights = np.full(len(modalities), 1.0 / len(modalities), dtype=float)
    return _fuse_matrix(
        matrix,
        modalities,
        weights,
        fusion_method="uniform_mean",
        source_family=_source_family(predictions),
        complete_case_only=True,
        provenance=provenance,
    )


def fuse_with_fixed_weights(
    predictions: pd.DataFrame,
    weights: ValidationWeightMap,
) -> pd.DataFrame:
    modalities = _ordered_modalities(weights.keys())
    matrix, modalities, provenance = _pivot_predictions(
        predictions,
        modalities=modalities,
        name="fixed-weight fusion predictions",
    )
    context = _single_context(matrix, purpose="fixed-weight fusion")
    normalized = _validated_fixed_weights(
        weights,
        modalities=modalities,
        context=context,
        provenance=provenance,
    )
    return _fuse_matrix(
        matrix,
        modalities,
        normalized,
        fusion_method="legacy_validation_weighted_auprc",
        source_family=_source_family(predictions),
        complete_case_only=True,
        provenance=provenance,
    )


def fuse_available_modalities_sensitivity(
    predictions: pd.DataFrame,
    weights: ValidationWeightMap,
) -> pd.DataFrame:
    modalities = _ordered_modalities(weights.keys())
    matrix, modalities, provenance = _pivot_predictions(
        predictions,
        modalities=modalities,
        name="available-modality sensitivity predictions",
    )
    context = _single_context(matrix, purpose="available-modality sensitivity")
    normalized = _validated_fixed_weights(
        weights,
        modalities=modalities,
        context=context,
        provenance=provenance,
    )
    return _fuse_matrix(
        matrix,
        modalities,
        normalized,
        fusion_method="available_modalities_validation_weighted_auprc",
        source_family=_source_family(predictions),
        complete_case_only=False,
        provenance=provenance,
    )


def fit_validation_logistic_stacker(
    validation_predictions: pd.DataFrame,
    *,
    random_state: int,
) -> ValidationLogisticStacker:
    frame = _validate_predictions(validation_predictions, name="stacker training predictions")
    if not frame["split"].eq("validation").all():
        raise ValueError("Logistic stacking must be fitted from validation rows only")
    context = _single_context(frame, purpose="logistic stacker fitting")
    modalities = _ordered_modalities(frame["modality"].unique())
    matrix, modalities, provenance = _pivot_predictions(
        frame,
        modalities=modalities,
        name="stacker training predictions",
    )
    matrix = matrix.dropna(subset=list(modalities)).copy()
    if matrix.empty:
        raise ValueError("No complete-case validation rows are available for stacking")
    y_true = labels_to_binary(matrix["label_binary"])
    if len(np.unique(y_true)) != 2:
        raise ValueError("Logistic stacking requires both validation classes")
    estimator = LogisticRegression(
        l1_ratio=0.0,
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=int(random_state),
    )
    estimator.fit(matrix[list(modalities)].to_numpy(dtype=float), y_true)
    frozen_state = _capture_stacker_state(estimator)
    return ValidationLogisticStacker(
        estimator=estimator,
        feature_names=modalities,
        frozen_state=frozen_state,
        fitted_state_hash=_stacker_state_hash(frozen_state),
        run_id=context["run_id"],
        protocol=context["protocol"],
        fold=context["fold"],
        dataset=context["dataset"],
        branch_provenance_hashes=provenance.branch_hashes,
    )


def apply_validation_logistic_stacker(
    model: object,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(model, ValidationLogisticStacker) or model.source_split != "validation":
        raise ValueError("Logistic stacker must carry frozen validation provenance")
    frozen_state = _verify_stacker_state(model)
    frame = _validate_predictions(predictions, name="stacker application predictions")
    context = _single_context(frame, purpose="logistic stacker application")
    expected = {
        "run_id": model.run_id,
        "protocol": model.protocol,
        "fold": model.fold,
        "dataset": model.dataset,
    }
    for column, expected_value in expected.items():
        if context[column] != expected_value:
            raise ValueError(f"Logistic stacker {column} mismatch prevents cross-fold pooling")
    matrix, modalities, provenance = _pivot_predictions(
        frame,
        modalities=model.feature_names,
        name="stacker application predictions",
    )
    if provenance.branch_hashes != model.branch_provenance_hashes:
        raise ValueError(
            "Logistic stacker branch identity mismatch prevents validation/test substitution"
        )
    matrix = matrix.dropna(subset=list(modalities)).copy()
    features = matrix[list(modalities)].to_numpy(dtype=float)
    probabilities = _binary_logistic_probability(features, frozen_state)
    output = matrix[list(COHORT_COLUMNS) + ["label_binary"]].copy()
    output["probability"] = probabilities
    output["modality"] = "multimodal"
    output["modality_combination"] = "+".join(modalities)
    output["fusion_method"] = "stacked_logistic_validation"
    output["source_family"] = _source_family(predictions)
    output["available_modalities"] = ",".join(sorted(modalities))
    output["n_modalities"] = len(modalities)
    output["complete_case"] = True
    output["model"] = "stacked_logistic_validation"
    branch_hashes = provenance.branch_hash_map()
    upstream_artifact_hashes: dict[str, str] = {}
    for branch, artifact_hash in provenance.upstream_artifact_hashes:
        source = (
            "hst"
            if branch.startswith("hst_")
            else "comparator"
            if branch.startswith("comparator_")
            else branch
        )
        previous = upstream_artifact_hashes.setdefault(source, artifact_hash)
        if previous != artifact_hash:
            raise ValueError(f"Hybrid source {source!r} has conflicting artifact hashes")
    upstream_branch_hashes = provenance.upstream_branch_hash_map()
    output["manifest_sha256"] = provenance.manifest_sha256
    output["recording_intersection_sha256"] = provenance.recording_intersection_sha256
    output["source_prediction_artifact_hash"] = provenance.artifact_hash
    output["source_branch_provenance_hashes"] = _canonical_json_bytes(
        branch_hashes
    ).decode("utf-8")
    output["upstream_source_prediction_artifact_hashes"] = _canonical_json_bytes(
        upstream_artifact_hashes
    ).decode("utf-8")
    output["upstream_branch_provenance_hashes"] = _canonical_json_bytes(
        upstream_branch_hashes
    ).decode("utf-8")
    output["checkpoint_hash"] = hashlib.sha256(
        _canonical_json_bytes(
            {
                "fitted_state_hash": model.fitted_state_hash,
                "fusion_method": "stacked_logistic_validation",
                "source_prediction_artifact_hash": provenance.artifact_hash,
                "source_branch_provenance_hashes": branch_hashes,
                "upstream_source_prediction_artifact_hashes": upstream_artifact_hashes,
                "upstream_branch_provenance_hashes": upstream_branch_hashes,
                "manifest_sha256": provenance.manifest_sha256,
                "recording_intersection_sha256": (
                    provenance.recording_intersection_sha256
                ),
            }
        )
    ).hexdigest()
    output["representation"] = "participant_probability_fusion"
    designation = _analysis_designation(
        role="secondary",
        estimand_id="secondary_standalone_stacked_logistic_validation",
        multiplicity_family="secondary_standalone_fusion",
    )
    for column in ANALYSIS_HIERARCHY_COLUMNS:
        output[column] = designation[column]
    validation_rows = output["split"].eq("validation")
    if validation_rows.any():
        selection = _selection_designation(
            estimand_id="selection_standalone_stacked_logistic_validation"
        )
        for column in ANALYSIS_HIERARCHY_COLUMNS:
            output.loc[validation_rows, column] = selection[column]
    output["comparison_binding_hash"] = "not_applicable"
    return output.sort_values(list(COHORT_COLUMNS), kind="stable").reset_index(drop=True)


def build_four_branch_hybrid_inputs(
    hst_predictions: pd.DataFrame,
    compare_predictions: pd.DataFrame,
    *,
    authenticated_registry_receipt_sha256: str | None = None,
    authenticated_context_binding_sha256: str | None = None,
) -> pd.DataFrame:
    hst = _validate_predictions(hst_predictions, name="HST hybrid predictions")
    comparator = _validate_predictions(compare_predictions, name="comparator hybrid predictions")
    _require_source_family(hst, expected="hst", name="HST hybrid predictions")
    _require_source_family(
        comparator,
        expected="comparator",
        name="comparator hybrid predictions",
    )
    hst_context = _single_context(hst, purpose="HST hybrid input")
    comparator_context = _single_context(comparator, purpose="comparator hybrid input")
    for column in CONTEXT_COLUMNS:
        if hst_context[column] != comparator_context[column]:
            raise ValueError(f"Hybrid {column} mismatch prevents fold/cohort alignment")
    comparison_binding_hash = _primary_recording_binding_hash(
        hst,
        comparator,
        authenticated_registry_receipt_sha256=(
            authenticated_registry_receipt_sha256
        ),
        authenticated_context_binding_sha256=authenticated_context_binding_sha256,
    )

    hst_matrix, _, hst_provenance = _pivot_predictions(
        hst,
        modalities=PRIMARY_MODALITIES,
        name="HST hybrid predictions",
    )
    comparator_matrix, _, comparator_provenance = _pivot_predictions(
        comparator,
        modalities=PRIMARY_MODALITIES,
        name="comparator hybrid predictions",
    )
    if hst_matrix[list(PRIMARY_MODALITIES)].isna().any(axis=None):
        raise ValueError("HST hybrid cohort is not complete for cough and speech")
    if comparator_matrix[list(PRIMARY_MODALITIES)].isna().any(axis=None):
        raise ValueError("Comparator hybrid cohort is not complete for cough and speech")

    identity = list(COHORT_COLUMNS)
    hst_keys = set(map(tuple, hst_matrix[identity].itertuples(index=False, name=None)))
    comparator_keys = set(
        map(tuple, comparator_matrix[identity].itertuples(index=False, name=None))
    )
    if hst_keys != comparator_keys:
        raise ValueError("HST and comparator hybrid cohorts are not identical")

    hst_wide = hst_matrix.rename(
        columns={"cough": "hst_cough", "speech": "hst_speech", "label_binary": "hst_label"}
    )
    comparator_wide = comparator_matrix.rename(
        columns={
            "cough": "comparator_cough",
            "speech": "comparator_speech",
            "label_binary": "comparator_label",
        }
    )
    merged = hst_wide.merge(
        comparator_wide,
        on=identity,
        how="inner",
        validate="one_to_one",
    )
    if not merged["hst_label"].equals(merged["comparator_label"]):
        raise ValueError("HST and comparator labels disagree on the hybrid cohort")
    merged["label_binary"] = merged["hst_label"]

    merged["hst_prediction_artifact_hash"] = hst_provenance.artifact_hash
    merged["comparator_prediction_artifact_hash"] = comparator_provenance.artifact_hash
    merged["manifest_sha256"] = hst_provenance.manifest_sha256
    merged["upstream_recording_intersection_sha256"] = (
        hst_provenance.recording_intersection_sha256
    )
    merged["comparison_binding_hash"] = comparison_binding_hash
    provenance_columns = [
        *HYBRID_SOURCE_ARTIFACT_COLUMNS,
        "manifest_sha256",
        "upstream_recording_intersection_sha256",
        "comparison_binding_hash",
    ]
    for prefix, source in (("hst", hst), ("comparator", comparator)):
        for modality in PRIMARY_MODALITIES:
            branch = f"{prefix}_{modality}"
            branch_identity = _branch_identity_values(source, modality)
            for field_name in HYBRID_BRANCH_PROVENANCE_FIELDS:
                column = f"{branch}_{field_name}"
                merged[column] = branch_identity[field_name]
                provenance_columns.append(column)

    return merged[
        [
            *identity,
            "label_binary",
            *provenance_columns,
            *FOUR_BRANCH_COLUMNS,
        ]
    ].sort_values(identity, kind="stable").reset_index(drop=True)


def _complete_keys(frame: pd.DataFrame, modalities: tuple[str, ...]) -> set[tuple[object, ...]]:
    selected = frame.loc[frame["modality"].isin(modalities)].copy()
    counts = selected.groupby(list(COHORT_COLUMNS), dropna=False)["modality"].nunique()
    return {tuple(index) for index, count in counts.items() if int(count) == len(modalities)}


def _filter_cohort(
    frame: pd.DataFrame,
    keys: set[tuple[object, ...]],
) -> pd.DataFrame:
    index = pd.MultiIndex.from_frame(frame[list(COHORT_COLUMNS)])
    mask = index.isin(keys)
    return frame.loc[mask].copy()


def _branch_validation_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    participant_frame = _aggregate_recording_predictions(frame)
    validation = participant_frame.loc[
        participant_frame["split"].eq("validation")
    ].copy()
    if validation.empty:
        raise ValueError("Fusion requires fold-local validation predictions")
    rows: list[dict[str, object]] = []
    for modality, group in validation.groupby("modality", sort=True, dropna=False):
        y_true = labels_to_binary(group["label_binary"])
        if len(np.unique(y_true)) != 2:
            raise ValueError("Fusion-weight estimation requires both validation classes")
        branch_hashes = group["branch_provenance_hash"].drop_duplicates().tolist()
        if len(branch_hashes) != 1:
            raise ValueError("Validation branch provenance must be unique per modality")
        rows.append(
            {
                **{column: group[column].iloc[0] for column in CONTEXT_COLUMNS},
                "split": "validation",
                "modality": str(modality),
                "auprc": float(average_precision_score(y_true, group["probability"])),
                "branch_provenance_hash": str(branch_hashes[0]),
            }
        )
    return pd.DataFrame(rows)


def _weight_rows(
    context: dict[str, object],
    *,
    source_family: str,
    modality_combination: str,
    fusion_method: str,
    normalized: Mapping[str, float],
    raw: Mapping[str, float],
    validation_auprc: Mapping[str, float],
    branch_provenance_hashes: Mapping[str, str],
    source_split: str,
    designation: Mapping[str, str],
) -> list[dict[str, object]]:
    if source_split not in {"prespecified", "validation"}:
        raise ValueError(f"Unsupported fusion-weight provenance: {source_split}")
    if source_split == "validation":
        row_designation = _selection_designation(
            estimand_id=f"selection_{source_family}_{fusion_method}_weights"
        )
    elif designation["analysis_role"] == "primary":
        row_designation = _analysis_designation(
            role="secondary",
            estimand_id=f"secondary_prespecified_{source_family}_{fusion_method}_weights",
            multiplicity_family="secondary_prespecified_fusion_rules",
        )
    else:
        row_designation = dict(designation)
    return [
        {
            **context,
            "source_family": source_family,
            "modality_combination": modality_combination,
            "fusion_method": fusion_method,
            "branch": branch,
            "raw_weight": float(raw[branch]),
            "normalized_weight": float(normalized[branch]),
            "validation_auprc": float(validation_auprc[branch]),
            "branch_provenance_hash": branch_provenance_hashes[branch],
            "source_split": source_split,
            **row_designation,
        }
        for branch in normalized
    ]


def _stacker_parameter_rows(
    model: ValidationLogisticStacker,
    *,
    source_family: str,
    modality_combination: str,
    fusion_method: str,
    designation: Mapping[str, str] | None = None,
) -> list[dict[str, object]]:
    frozen_state = _verify_stacker_state(model)
    designation = _selection_designation(
        estimand_id=f"selection_{source_family}_{fusion_method}_parameters"
    )
    context = {
        "run_id": model.run_id,
        "protocol": model.protocol,
        "fold": model.fold,
        "dataset": model.dataset,
    }
    intercept = frozen_state.intercept[0]
    return [
        {
            **context,
            "source_family": source_family,
            "modality_combination": modality_combination,
            "fusion_method": fusion_method,
            "branch": branch,
            "coefficient": float(coefficient),
            "intercept": intercept,
            "C": frozen_state.C,
            "class_weight": frozen_state.class_weight,
            "penalty": frozen_state.penalty,
            "max_iter": frozen_state.max_iter,
            "random_state": frozen_state.random_state,
            "source_split": model.source_split,
            **designation,
        }
        for branch, coefficient in zip(model.feature_names, frozen_state.coef[0])
    ]


def _complete_case_count_rows(
    frame: pd.DataFrame,
    modalities: tuple[str, ...],
    *,
    source_family: str,
    analysis_cohort: str,
    designation: Mapping[str, str],
) -> list[dict[str, object]]:
    context = _single_context(frame, purpose="complete-case counting")
    rows: list[dict[str, object]] = []
    for split, group in frame.groupby("split", sort=True, dropna=False):
        total = int(group["participant_key"].nunique())
        complete = len(
            {
                key
                for key in _complete_keys(group, modalities)
            }
        )
        if split == "validation":
            row_designation = _selection_designation(
                estimand_id=f"selection_{source_family}_{analysis_cohort}_count"
            )
        elif designation["analysis_role"] == "primary":
            row_designation = _analysis_designation(
                role="secondary",
                estimand_id=f"secondary_{source_family}_{analysis_cohort}_count",
                multiplicity_family="secondary_cohort_accounting",
            )
        else:
            row_designation = dict(designation)
        rows.append(
            {
                **context,
                "split": split,
                "source_family": source_family,
                "analysis_cohort": analysis_cohort,
                "modality_combination": "+".join(modalities),
                "required_modalities": ",".join(modalities),
                "n_participants_input": total,
                "n_participants_complete_case": complete,
                "n_participants_excluded": total - complete,
                **row_designation,
            }
        )
    return rows


def _analysis_designation(
    *,
    role: str,
    estimand_id: str,
    multiplicity_family: str,
    scope: str | None = None,
) -> dict[str, str]:
    allowed = {"primary", "secondary", "sensitivity", "exploratory"}
    if role not in allowed:
        raise ValueError(f"Unsupported analysis role: {role!r}")
    if not estimand_id or not multiplicity_family:
        raise ValueError("Analysis hierarchy identifiers must be non-empty")
    resolved_scope = scope or ("confirmatory" if role == "primary" else role)
    allowed_scopes_by_role = {
        "primary": {"confirmatory"},
        "secondary": {"secondary", "selection"},
        "sensitivity": {"sensitivity"},
        "exploratory": {"exploratory"},
    }
    if resolved_scope not in allowed_scopes_by_role[role]:
        raise ValueError(
            f"Analysis role {role!r} cannot use analysis scope {resolved_scope!r}"
        )
    return {
        "analysis_scope": resolved_scope,
        "analysis_role": role,
        "estimand_id": estimand_id,
        "multiplicity_family": multiplicity_family,
    }


def _selection_designation(*, estimand_id: str) -> dict[str, str]:
    return _analysis_designation(
        role="secondary",
        scope="selection",
        estimand_id=estimand_id,
        multiplicity_family="validation_model_selection",
    )


def _validate_analysis_hierarchy_table(frame: pd.DataFrame, *, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Fusion export table {name!r} must be a pandas DataFrame")
    missing = sorted(set(ANALYSIS_HIERARCHY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(
            f"Fusion export table {name!r} is missing analysis hierarchy columns: {missing}"
        )
    if frame.empty:
        return

    for column in ANALYSIS_HIERARCHY_COLUMNS:
        values = frame[column].tolist()
        if any(type(value) is not str or not value for value in values):
            raise ValueError(
                f"Fusion export table {name!r} has invalid analysis hierarchy "
                f"values in {column}"
            )
    allowed_roles = {"primary", "secondary", "sensitivity", "exploratory"}
    allowed_scopes = {
        "confirmatory",
        "selection",
        "secondary",
        "sensitivity",
        "exploratory",
    }
    if not set(frame["analysis_role"]).issubset(allowed_roles):
        raise ValueError(f"Fusion export table {name!r} has an invalid analysis_role")
    if not set(frame["analysis_scope"]).issubset(allowed_scopes):
        raise ValueError(f"Fusion export table {name!r} has an invalid analysis_scope")

    valid_scopes_by_role = {
        "primary": {"confirmatory"},
        "secondary": {"secondary", "selection"},
        "sensitivity": {"sensitivity"},
        "exploratory": {"exploratory"},
    }
    invalid_role_scope = [
        (role, scope)
        for role, scope in frame[["analysis_role", "analysis_scope"]].itertuples(
            index=False,
            name=None,
        )
        if scope not in valid_scopes_by_role[role]
    ]
    if invalid_role_scope:
        raise ValueError(
            f"Fusion export table {name!r} has an invalid role/scope mapping: "
            f"{invalid_role_scope[0]}"
        )

    primary = frame["analysis_role"].eq("primary")
    if primary.any():
        if not frame.loc[primary, "analysis_scope"].eq("confirmatory").all():
            raise ValueError("Primary fusion analyses must have confirmatory analysis_scope")
        if not frame.loc[primary, "estimand_id"].eq(PRIMARY_ESTIMAND_ID).all():
            raise ValueError("Primary fusion analyses must use the prespecified estimand_id")
        if not frame.loc[primary, "multiplicity_family"].eq(
            "confirmatory_primary_single"
        ).all():
            raise ValueError("Primary fusion analyses must use the single confirmatory family")
        if "split" not in frame or not frame.loc[primary, "split"].eq("test").all():
            raise ValueError("Primary confirmatory fusion evidence must be held-out test rows")
        if "source_family" in frame and not frame.loc[
            primary, "source_family"
        ].isin({"hst", "comparator"}).all():
            raise ValueError(
                "Primary fusion source_family must be HST or the approved comparator"
            )
        if "fusion_method" in frame and not frame.loc[
            primary, "fusion_method"
        ].eq("uniform_mean").all():
            raise ValueError("Primary fusion method must be uniform_mean")
        if "modality_combination" in frame and not frame.loc[
            primary, "modality_combination"
        ].eq("cough+speech").all():
            raise ValueError("Primary fusion modality combination must be cough+speech")
        if "complete_case" in frame and not frame.loc[primary, "complete_case"].eq(
            True
        ).all():
            raise ValueError("Primary fusion predictions must use the complete-case cohort")
        if "comparison_binding_hash" not in frame or not frame.loc[
            primary, "comparison_binding_hash"
        ].map(_is_lower_sha256).all():
            raise ValueError("Primary fusion comparison binding must be a SHA-256 digest")
        for column in (
            "authenticated_registry_receipt_sha256",
            "authenticated_context_binding_sha256",
        ):
            if column not in frame or not frame.loc[primary, column].map(
                _is_lower_sha256
            ).all():
                raise ValueError(f"Primary fusion {column} must be an authenticated digest")
        if "approved_comparator_generation_id" in frame:
            values = frame.loc[primary, "approved_comparator_generation_id"].tolist()
            if any(type(value) is not str or not value for value in values):
                raise ValueError("Primary fusion comparator generation identity is invalid")
        if "approved_comparator_generation_receipt_sha256" in frame and not frame.loc[
            primary, "approved_comparator_generation_receipt_sha256"
        ].map(_is_lower_sha256).all():
            raise ValueError("Primary fusion comparator generation receipt is invalid")
        if "candidate_family" in frame and not (
            frame.loc[primary, "candidate_family"].eq("hst").all()
            and frame.loc[primary, "reference_family"].eq("comparator").all()
        ):
            raise ValueError(
                "Primary paired delta must compare HST against the approved comparator"
            )
        if "metric" in frame and not frame.loc[primary, "metric"].eq("auroc").all():
            raise ValueError("Primary paired delta metric must be AUROC")


def _annotate_predictions(
    frame: pd.DataFrame,
    *,
    source_family: str,
    method: str | None = None,
    designation: Mapping[str, str],
    comparison_binding_hash: str | None,
    authenticated_registry_receipt_sha256: str | None = None,
    authenticated_context_binding: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    output = frame.copy()
    output["source_family"] = source_family
    if method is not None:
        output["fusion_method"] = method
        output["model"] = method
    for column in ANALYSIS_HIERARCHY_COLUMNS:
        output[column] = designation[column]
    validation = output["split"].eq("validation")
    if validation.any():
        selection = _selection_designation(
            estimand_id=f"selection_{source_family}_{method or output['fusion_method'].iloc[0]}"
        )
        for column in ANALYSIS_HIERARCHY_COLUMNS:
            output.loc[validation, column] = selection[column]
    output["comparison_binding_hash"] = comparison_binding_hash or "not_applicable"
    output["authenticated_registry_receipt_sha256"] = (
        authenticated_registry_receipt_sha256 or "not_applicable"
    )
    output["authenticated_context_binding_sha256"] = (
        hashlib.sha256(_canonical_json_bytes(dict(authenticated_context_binding))).hexdigest()
        if authenticated_context_binding is not None
        else "not_applicable"
    )
    comparator_identity = (
        authenticated_context_binding.get("comparator")
        if authenticated_context_binding is not None
        else None
    )
    output["approved_comparator_generation_id"] = (
        comparator_identity["generation_id"]
        if isinstance(comparator_identity, Mapping)
        else "not_applicable"
    )
    output["approved_comparator_generation_receipt_sha256"] = (
        comparator_identity["generation_receipt_sha256"]
        if isinstance(comparator_identity, Mapping)
        else "not_applicable"
    )
    return output


def _run_family_bank(
    primary_frame: pd.DataFrame,
    sensitivity_frame: pd.DataFrame,
    *,
    source_family: str,
    random_state: int,
    modalities: tuple[str, ...],
    primary_analysis: bool,
    comparison_binding_hash: str | None,
    authenticated_registry_receipt_sha256: str | None,
    authenticated_context_binding: Mapping[str, object] | None,
) -> tuple[
    list[pd.DataFrame],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    context = _single_context(primary_frame, purpose=f"{source_family} fusion bank")
    branch_provenance_hashes = _prediction_provenance(
        primary_frame,
        modalities,
    ).branch_hash_map()
    branch_metrics = _branch_validation_metrics(primary_frame)
    validation_auprc = branch_metrics.set_index("modality")["auprc"].to_dict()
    weights = legacy_validation_auprc_weights(branch_metrics)
    combination = "+".join(modalities)
    modality_sensitivity = modalities == ("cough", "breath")
    if primary_analysis:
        if comparison_binding_hash is None:
            raise ValueError("Primary fusion requires a frozen comparison binding hash")
        uniform_designation = _analysis_designation(
            role="primary",
            estimand_id=PRIMARY_ESTIMAND_ID,
            multiplicity_family="confirmatory_primary_single",
        )
    elif modality_sensitivity:
        uniform_designation = _analysis_designation(
            role="sensitivity",
            estimand_id=f"sensitivity_{source_family}_{combination}_uniform",
            multiplicity_family="sensitivity_modality_definitions",
        )
    else:
        uniform_designation = _analysis_designation(
            role="secondary",
            estimand_id=f"secondary_unpaired_{source_family}_{combination}_uniform",
            multiplicity_family="secondary_unpaired_fusion",
        )
    learned_role = "sensitivity" if modality_sensitivity else "secondary"
    learned_family = (
        "sensitivity_modality_definitions"
        if modality_sensitivity
        else "secondary_fusion_rules"
    )
    weighted_designation = _analysis_designation(
        role=learned_role,
        estimand_id=f"{learned_role}_{source_family}_{combination}_validation_weighted",
        multiplicity_family=learned_family,
    )
    stacker_designation = _analysis_designation(
        role=learned_role,
        estimand_id=f"{learned_role}_{source_family}_{combination}_validation_stacker",
        multiplicity_family=learned_family,
    )
    available_designation = _analysis_designation(
        role="sensitivity",
        estimand_id=f"sensitivity_{source_family}_{combination}_available_modalities",
        multiplicity_family="sensitivity_missing_modality",
    )

    uniform = _annotate_predictions(
        fuse_uniform_complete_case(primary_frame),
        source_family=source_family,
        designation=uniform_designation,
        comparison_binding_hash=comparison_binding_hash,
        authenticated_registry_receipt_sha256=(
            authenticated_registry_receipt_sha256
        ),
        authenticated_context_binding=authenticated_context_binding,
    )
    weighted = _annotate_predictions(
        fuse_with_fixed_weights(primary_frame, weights),
        source_family=source_family,
        designation=weighted_designation,
        comparison_binding_hash=comparison_binding_hash,
        authenticated_registry_receipt_sha256=(
            authenticated_registry_receipt_sha256
        ),
        authenticated_context_binding=authenticated_context_binding,
    )
    available = _annotate_predictions(
        fuse_available_modalities_sensitivity(sensitivity_frame, weights),
        source_family=source_family,
        designation=available_designation,
        comparison_binding_hash=comparison_binding_hash,
        authenticated_registry_receipt_sha256=(
            authenticated_registry_receipt_sha256
        ),
        authenticated_context_binding=authenticated_context_binding,
    )
    validation = primary_frame.loc[primary_frame["split"].eq("validation")]
    stacker = fit_validation_logistic_stacker(validation, random_state=random_state)
    stacked = _annotate_predictions(
        apply_validation_logistic_stacker(stacker, primary_frame),
        source_family=source_family,
        designation=stacker_designation,
        comparison_binding_hash=comparison_binding_hash,
        authenticated_registry_receipt_sha256=(
            authenticated_registry_receipt_sha256
        ),
        authenticated_context_binding=authenticated_context_binding,
    )

    uniform_weights = {modality: 1.0 / len(modalities) for modality in modalities}
    weight_rows = _weight_rows(
        context,
        source_family=source_family,
        modality_combination=combination,
        fusion_method="uniform_mean",
        normalized=uniform_weights,
        raw={modality: 1.0 for modality in modalities},
        validation_auprc=validation_auprc,
        branch_provenance_hashes=branch_provenance_hashes,
        source_split="prespecified",
        designation=uniform_designation,
    )
    weight_rows.extend(
        _weight_rows(
            context,
            source_family=source_family,
            modality_combination=combination,
            fusion_method="legacy_validation_weighted_auprc",
            normalized=dict(weights),
            raw=weights.raw_weights,
            validation_auprc=validation_auprc,
            branch_provenance_hashes=branch_provenance_hashes,
            source_split="validation",
            designation=weighted_designation,
        )
    )
    weight_rows.extend(
        _weight_rows(
            context,
            source_family=source_family,
            modality_combination=combination,
            fusion_method="available_modalities_validation_weighted_auprc",
            normalized=weights,
            raw=weights.raw_weights,
            validation_auprc=validation_auprc,
            branch_provenance_hashes=branch_provenance_hashes,
            source_split="validation",
            designation=available_designation,
        )
    )
    stacker_rows = _stacker_parameter_rows(
        stacker,
        source_family=source_family,
        modality_combination=combination,
        fusion_method="stacked_logistic_validation",
        designation=stacker_designation,
    )
    count_rows = _complete_case_count_rows(
        primary_frame,
        modalities,
        source_family=source_family,
        analysis_cohort="primary_aligned_complete_case"
        if primary_analysis
        else "prespecified_complete_case",
        designation=uniform_designation,
    )
    count_rows.extend(_complete_case_count_rows(
        sensitivity_frame,
        modalities,
        source_family=source_family,
        analysis_cohort="family_available",
        designation=available_designation,
    ))
    return [uniform, weighted, available, stacked], weight_rows, stacker_rows, count_rows


def _hybrid_to_long(hybrid: pd.DataFrame) -> pd.DataFrame:
    branch_provenance_columns = [
        f"{branch}_{field_name}"
        for branch in FOUR_BRANCH_COLUMNS
        for field_name in HYBRID_BRANCH_PROVENANCE_FIELDS
    ]
    required_provenance = {
        *HYBRID_SOURCE_ARTIFACT_COLUMNS,
        "manifest_sha256",
        "upstream_recording_intersection_sha256",
        "comparison_binding_hash",
        *branch_provenance_columns,
    }
    missing = sorted(required_provenance - set(hybrid.columns))
    if missing:
        raise ValueError(f"Hybrid inputs are missing source provenance columns: {missing}")
    id_columns = [
        *COHORT_COLUMNS,
        "label_binary",
        *HYBRID_SOURCE_ARTIFACT_COLUMNS,
        "manifest_sha256",
        "upstream_recording_intersection_sha256",
        "comparison_binding_hash",
        *branch_provenance_columns,
    ]
    long = hybrid.melt(
        id_vars=id_columns,
        value_vars=list(FOUR_BRANCH_COLUMNS),
        var_name="modality",
        value_name="probability",
    )
    long["source_family"] = "hybrid"
    long["model"] = ""
    long["checkpoint_hash"] = ""
    long["representation"] = ""
    long["feature_artifact_sha256"] = ""
    long["feature_approval_id"] = ""
    long["preprocessing_sha256"] = ""
    long["upstream_branch_provenance_hash"] = ""
    long["source_prediction_artifact_hash"] = ""
    long["recording_key"] = (
        long["participant_key"].astype(str)
        + "::hybrid::"
        + long["modality"].astype(str)
    )
    long["audio_content_sha256"] = ""
    long["eligible"] = True
    for branch in FOUR_BRANCH_COLUMNS:
        selected = long["modality"].eq(branch)
        for field_name in (
            "model",
            "checkpoint_hash",
            "representation",
            "feature_artifact_sha256",
            "feature_approval_id",
            "preprocessing_sha256",
        ):
            long.loc[selected, field_name] = long.loc[
                selected, f"{branch}_{field_name}"
            ]
        long.loc[selected, "upstream_branch_provenance_hash"] = long.loc[
            selected, f"{branch}_branch_provenance_hash"
        ]
        artifact_column = (
            "hst_prediction_artifact_hash"
            if branch.startswith("hst_")
            else "comparator_prediction_artifact_hash"
        )
        long.loc[selected, "source_prediction_artifact_hash"] = long.loc[
            selected, artifact_column
        ]
        for index in long.index[selected]:
            long.loc[index, "audio_content_sha256"] = hashlib.sha256(
                _canonical_json_bytes(
                    {
                        "participant_key": long.loc[index, "participant_key"],
                        "hybrid_branch": branch,
                        "source_prediction_artifact_hash": long.loc[
                            index, "source_prediction_artifact_hash"
                        ],
                        "upstream_branch_provenance_hash": long.loc[
                            index, "upstream_branch_provenance_hash"
                        ],
                    }
                )
            ).hexdigest()
    long["recording_intersection_sha256"] = _recording_intersection_hash(long)
    return long.drop(
        columns=[*HYBRID_SOURCE_ARTIFACT_COLUMNS, *branch_provenance_columns]
    )


def _run_hybrid_bank(
    hybrid: pd.DataFrame,
    *,
    random_state: int,
    authenticated_registry_receipt_sha256: str | None,
    authenticated_context_binding: Mapping[str, object] | None,
) -> tuple[
    list[pd.DataFrame],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    long = _validate_predictions(
        _hybrid_to_long(hybrid),
        name="hybrid branch predictions",
    )
    context = _single_context(long, purpose="hybrid fusion bank")
    branch_provenance_hashes = _prediction_provenance(
        long,
        FOUR_BRANCH_COLUMNS,
    ).branch_hash_map()
    branch_metrics = _branch_validation_metrics(long)
    validation_auprc = branch_metrics.set_index("modality")["auprc"].to_dict()
    weights = legacy_validation_auprc_weights(branch_metrics)
    combination = "+".join(FOUR_BRANCH_COLUMNS)
    comparison_binding_hash = _single_sha256(
        long,
        "comparison_binding_hash",
        name="hybrid fusion bank",
    )
    uniform_designation = _analysis_designation(
        role="secondary",
        estimand_id="secondary_hybrid_uniform_four_branch",
        multiplicity_family="secondary_hybrid_fusion",
    )
    weighted_designation = _analysis_designation(
        role="secondary",
        estimand_id="secondary_hybrid_validation_weighted_auprc",
        multiplicity_family="secondary_hybrid_fusion",
    )
    stacker_designation = _analysis_designation(
        role="secondary",
        estimand_id="secondary_hybrid_validation_stacker",
        multiplicity_family="secondary_hybrid_fusion",
    )

    uniform = _annotate_predictions(
        fuse_uniform_complete_case(long),
        source_family="hybrid",
        method="hybrid_uniform_four_branch",
        designation=uniform_designation,
        comparison_binding_hash=comparison_binding_hash,
        authenticated_registry_receipt_sha256=(
            authenticated_registry_receipt_sha256
        ),
        authenticated_context_binding=authenticated_context_binding,
    )
    weighted = _annotate_predictions(
        fuse_with_fixed_weights(long, weights),
        source_family="hybrid",
        method="hybrid_legacy_validation_weighted_auprc",
        designation=weighted_designation,
        comparison_binding_hash=comparison_binding_hash,
        authenticated_registry_receipt_sha256=(
            authenticated_registry_receipt_sha256
        ),
        authenticated_context_binding=authenticated_context_binding,
    )
    validation = long.loc[long["split"].eq("validation")]
    stacker = fit_validation_logistic_stacker(validation, random_state=random_state)
    stacked = _annotate_predictions(
        apply_validation_logistic_stacker(stacker, long),
        source_family="hybrid",
        method="hybrid_stacked_logistic_validation",
        designation=stacker_designation,
        comparison_binding_hash=comparison_binding_hash,
        authenticated_registry_receipt_sha256=(
            authenticated_registry_receipt_sha256
        ),
        authenticated_context_binding=authenticated_context_binding,
    )

    uniform_weights = {branch: 0.25 for branch in FOUR_BRANCH_COLUMNS}
    weight_rows = _weight_rows(
        context,
        source_family="hybrid",
        modality_combination=combination,
        fusion_method="hybrid_uniform_four_branch",
        normalized=uniform_weights,
        raw={branch: 1.0 for branch in FOUR_BRANCH_COLUMNS},
        validation_auprc=validation_auprc,
        branch_provenance_hashes=branch_provenance_hashes,
        source_split="prespecified",
        designation=uniform_designation,
    )
    weight_rows.extend(
        _weight_rows(
            context,
            source_family="hybrid",
            modality_combination=combination,
            fusion_method="hybrid_legacy_validation_weighted_auprc",
            normalized=dict(weights),
            raw=weights.raw_weights,
            validation_auprc=validation_auprc,
            branch_provenance_hashes=branch_provenance_hashes,
            source_split="validation",
            designation=weighted_designation,
        )
    )
    stacker_rows = _stacker_parameter_rows(
        stacker,
        source_family="hybrid",
        modality_combination=combination,
        fusion_method="hybrid_stacked_logistic_validation",
        designation=stacker_designation,
    )
    count_rows = _complete_case_count_rows(
        long,
        FOUR_BRANCH_COLUMNS,
        source_family="hybrid",
        analysis_cohort="joint_four_branch_complete_case",
        designation=uniform_designation,
    )
    return [uniform, weighted, stacked], weight_rows, stacker_rows, count_rows


def _metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        *CONTEXT_COLUMNS,
        "source_family",
        "modality_combination",
        "fusion_method",
        "analysis_role",
        "analysis_scope",
        "estimand_id",
        "multiplicity_family",
        "comparison_binding_hash",
        "authenticated_registry_receipt_sha256",
        "authenticated_context_binding_sha256",
        "approved_comparator_generation_id",
        "approved_comparator_generation_receipt_sha256",
        "split",
    ]
    rows: list[dict[str, object]] = []
    for group_key, group in predictions.groupby(group_columns, sort=True, dropna=False):
        y_true = labels_to_binary(group["label_binary"])
        metrics = binary_metric_bundle(
            y_true,
            group["probability"].to_numpy(dtype=float),
            threshold=0.5,
        )
        metrics.update(dict(zip(group_columns, group_key)))
        metrics["threshold_source"] = "fixed_0.5"
        metrics["n_participants"] = int(group["participant_key"].nunique())
        rows.append(metrics)
    return pd.DataFrame(rows)


def _paired_delta_rows(predictions: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    context_and_split = [*CONTEXT_COLUMNS, "split"]

    def append_deltas(
        *,
        context_key: tuple[object, ...],
        candidate: pd.DataFrame,
        reference: pd.DataFrame,
        candidate_family: str,
        reference_family: str,
        metrics: tuple[str, ...],
        designation_for_metric: Mapping[str, Mapping[str, str]],
    ) -> None:
        candidate_keys = set(candidate["participant_key"])
        reference_keys = set(reference["participant_key"])
        if candidate_keys != reference_keys:
            raise ValueError("Paired fusion delta cohorts are not identical")
        aligned = candidate[["participant_key", "label_binary", "probability"]].merge(
            reference[["participant_key", "label_binary", "probability"]],
            on="participant_key",
            suffixes=("_candidate", "_reference"),
            validate="one_to_one",
        )
        if not aligned["label_binary_candidate"].equals(
            aligned["label_binary_reference"]
        ):
            raise ValueError("Paired fusion delta labels disagree")
        binding_values = pd.concat(
            [candidate["comparison_binding_hash"], reference["comparison_binding_hash"]],
            ignore_index=True,
        ).drop_duplicates().tolist()
        if len(binding_values) != 1:
            raise ValueError("Paired fusion delta comparison binding differs")
        binding = binding_values[0]
        if (
            type(binding) is not str
            or len(binding) != 64
            or any(character not in "0123456789abcdef" for character in binding)
        ):
            raise ValueError(
                "Paired fusion delta comparison binding must be a lowercase SHA-256 digest"
            )
        registry_values = pd.concat(
            [
                candidate["authenticated_registry_receipt_sha256"],
                reference["authenticated_registry_receipt_sha256"],
            ],
            ignore_index=True,
        ).drop_duplicates().tolist()
        context_binding_values = pd.concat(
            [
                candidate["authenticated_context_binding_sha256"],
                reference["authenticated_context_binding_sha256"],
            ],
            ignore_index=True,
        ).drop_duplicates().tolist()
        if len(registry_values) != 1 or len(context_binding_values) != 1:
            raise ValueError("Paired fusion delta authenticated registry binding differs")
        y_true = labels_to_binary(aligned["label_binary_candidate"])
        metric_functions = {
            "auroc": roc_auc_score,
            "auprc": average_precision_score,
        }
        for metric_name in metrics:
            metric_function = metric_functions[metric_name]
            candidate_value = float(
                metric_function(y_true, aligned["probability_candidate"])
            )
            reference_value = float(
                metric_function(y_true, aligned["probability_reference"])
            )
            rows.append(
                {
                    **dict(zip(context_and_split, context_key)),
                    "candidate_family": candidate_family,
                    "reference_family": reference_family,
                    "metric": metric_name,
                    "hybrid_value": candidate_value,
                    "reference_value": reference_value,
                    "delta": candidate_value - reference_value,
                    "paired_participants": int(len(aligned)),
                    "comparison_binding_hash": binding,
                    "authenticated_registry_receipt_sha256": registry_values[0],
                    "authenticated_context_binding_sha256": context_binding_values[0],
                    **designation_for_metric[metric_name],
                }
            )

    for key, group in predictions.groupby(context_and_split, sort=True, dropna=False):
        hst = group.loc[
            group["source_family"].eq("hst")
            & group["fusion_method"].eq("uniform_mean")
            & group["analysis_role"].eq("primary")
            & group["modality_combination"].eq("cough+speech")
        ].copy()
        comparator = group.loc[
            group["source_family"].eq("comparator")
            & group["fusion_method"].eq("uniform_mean")
            & group["analysis_role"].eq("primary")
            & group["modality_combination"].eq("cough+speech")
        ].copy()
        if not hst.empty or not comparator.empty:
            if hst.empty or comparator.empty:
                raise ValueError("Primary paired estimand is missing one model family")
            append_deltas(
                context_key=key,
                candidate=hst,
                reference=comparator,
                candidate_family="hst",
                reference_family="comparator",
                metrics=("auroc", "auprc"),
                designation_for_metric={
                    "auroc": _analysis_designation(
                        role="primary",
                        estimand_id=PRIMARY_ESTIMAND_ID,
                        multiplicity_family="confirmatory_primary_single",
                    ),
                    "auprc": _analysis_designation(
                        role="secondary",
                        estimand_id=(
                            "secondary_hst_vs_comparator_uniform_cough_speech_auprc"
                        ),
                        multiplicity_family="secondary_discrimination_metrics",
                    ),
                },
            )

        hybrid = group.loc[
            group["source_family"].eq("hybrid")
            & group["fusion_method"].eq("hybrid_uniform_four_branch")
            & group["analysis_role"].eq("secondary")
        ].copy()
        if hybrid.empty:
            continue
        for reference_family in ("hst", "comparator"):
            reference = group.loc[
                group["source_family"].eq(reference_family)
                & group["fusion_method"].eq("uniform_mean")
                & group["analysis_role"].eq("primary")
                & group["modality_combination"].eq("cough+speech")
            ].copy()
            if reference.empty:
                continue
            designation = {
                metric: _analysis_designation(
                    role="secondary",
                    estimand_id=f"secondary_hybrid_vs_{reference_family}_{metric}",
                    multiplicity_family="secondary_hybrid_deltas",
                )
                for metric in ("auroc", "auprc")
            }
            append_deltas(
                context_key=key,
                candidate=hybrid,
                reference=reference,
                candidate_family="hybrid",
                reference_family=reference_family,
                metrics=("auroc", "auprc"),
                designation_for_metric=designation,
            )
    return rows


def run_hst_fusion_bank(
    hst_predictions: pd.DataFrame,
    compare_predictions: pd.DataFrame | None = None,
    *,
    analysis_mode: str | None = None,
    authenticated_binding: object | None = None,
) -> HSTFusionResult:
    if analysis_mode not in {"confirmatory", "exploratory"}:
        raise ValueError(
            "Fusion bank analysis_mode must be explicitly 'confirmatory' or 'exploratory'"
        )
    if analysis_mode == "confirmatory":
        if compare_predictions is None or authenticated_binding is None:
            raise ValueError(
                "Confirmatory fusion requires a comparator and authenticated upstream binding"
            )
    elif authenticated_binding is not None:
        raise ValueError("Exploratory fusion must not carry a confirmatory authenticated binding")

    hst = _validate_predictions(hst_predictions, name="HST fusion-bank predictions")
    _require_source_family(hst, expected="hst", name="HST fusion-bank predictions")
    comparator = None
    if compare_predictions is not None:
        comparator = _validate_predictions(
            compare_predictions,
            name="comparator fusion-bank predictions",
        )
        _require_source_family(
            comparator,
            expected="comparator",
            name="comparator fusion-bank predictions",
        )

    hst_contexts = {
        tuple(row)
        for row in hst[list(CONTEXT_COLUMNS)].drop_duplicates().itertuples(index=False, name=None)
    }
    comparator_contexts: set[tuple[object, ...]] = set()
    if comparator is not None:
        comparator_contexts = {
            tuple(row)
            for row in comparator[list(CONTEXT_COLUMNS)]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }
        if hst_contexts != comparator_contexts:
            raise ValueError("HST and comparator fold contexts are not identical")
        combined_labels = pd.concat(
            [
                hst[["run_id", "protocol", "dataset", "participant_key", "label_binary"]],
                comparator[
                    ["run_id", "protocol", "dataset", "participant_key", "label_binary"]
                ],
            ],
            ignore_index=True,
        )
        _require_participant_label_invariant(
            combined_labels,
            name="fusion-bank inputs",
        )

    authenticated_entries: dict[tuple[object, ...], Mapping[str, object]] = {}
    authenticated_context_hashes: dict[tuple[object, ...], str] = {}
    authenticated_registry_sha256: str | None = None
    if analysis_mode == "confirmatory":
        if type(authenticated_binding) is not AuthenticatedFusionBinding:
            raise ValueError(
                "Confirmatory fusion requires a frozen authenticated registry binding"
            )
        receipt = authenticated_binding.verified_receipt()
        authenticated_registry_sha256 = authenticated_binding.receipt_sha256
        for entry in receipt["contexts"]:
            context_key = tuple(entry[column] for column in CONTEXT_COLUMNS)
            authenticated_entries[context_key] = entry
        if set(authenticated_entries) != hst_contexts:
            raise ValueError(
                "Authenticated registry contexts do not exactly match fusion-bank contexts"
            )
        if comparator is None:
            raise ValueError("Confirmatory fusion requires approved comparator predictions")
        for context_key in sorted(hst_contexts):
            hst_mask = np.logical_and.reduce(
                [hst[column].eq(value) for column, value in zip(CONTEXT_COLUMNS, context_key)]
            )
            comparator_mask = np.logical_and.reduce(
                [
                    comparator[column].eq(value)
                    for column, value in zip(CONTEXT_COLUMNS, context_key)
                ]
            )
            authenticated_context_hashes[context_key] = (
                _verify_authenticated_context_binding(
                    authenticated_entries[context_key],
                    hst.loc[hst_mask],
                    comparator.loc[comparator_mask],
                )
            )

    prediction_frames: list[pd.DataFrame] = []
    weight_rows: list[dict[str, object]] = []
    stacker_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    for context_key in sorted(hst_contexts):
        authenticated_entry = authenticated_entries.get(context_key)
        authenticated_context_hash = authenticated_context_hashes.get(context_key)
        hst_mask = np.logical_and.reduce(
            [hst[column].eq(value) for column, value in zip(CONTEXT_COLUMNS, context_key)]
        )
        hst_all_context = hst.loc[hst_mask].copy()
        hst_context = hst_all_context.loc[
            hst_all_context["modality"].isin(PRIMARY_MODALITIES)
        ].copy()
        hst_complete = _complete_keys(hst_context, PRIMARY_MODALITIES)
        if not hst_complete:
            raise ValueError("HST fold has no complete cough+speech participant cohort")

        comparator_context: pd.DataFrame | None = None
        comparison_binding_hash: str | None = None
        joint_keys = hst_complete
        if comparator is not None:
            comparator_mask = np.logical_and.reduce(
                [
                    comparator[column].eq(value)
                    for column, value in zip(CONTEXT_COLUMNS, context_key)
                ]
            )
            comparator_context = comparator.loc[
                comparator_mask & comparator["modality"].isin(PRIMARY_MODALITIES)
            ].copy()
            comparison_binding_hash = _primary_recording_binding_hash(
                hst_context,
                comparator_context,
                authenticated_registry_receipt_sha256=(
                    authenticated_registry_sha256
                ),
                authenticated_context_binding_sha256=authenticated_context_hash,
            )
            comparator_complete = _complete_keys(comparator_context, PRIMARY_MODALITIES)
            if hst_complete != comparator_complete:
                hst_only = len(hst_complete - comparator_complete)
                comparator_only = len(comparator_complete - hst_complete)
                raise ValueError(
                    "primary HST/comparator complete-case cohorts are unequal "
                    f"(HST-only={hst_only}, comparator-only={comparator_only})"
                )
            joint_keys = hst_complete

        hst_primary = _filter_cohort(hst_context, joint_keys)
        family_outputs = _run_family_bank(
            hst_primary,
            hst_context,
            source_family="hst",
            random_state=42,
            modalities=PRIMARY_MODALITIES,
            primary_analysis=(
                analysis_mode == "confirmatory" and comparator_context is not None
            ),
            comparison_binding_hash=comparison_binding_hash,
            authenticated_registry_receipt_sha256=authenticated_registry_sha256,
            authenticated_context_binding=authenticated_entry,
        )
        prediction_frames.extend(family_outputs[0])
        weight_rows.extend(family_outputs[1])
        stacker_rows.extend(family_outputs[2])
        count_rows.extend(family_outputs[3])

        cough_breath_modalities = ("cough", "breath")
        if set(cough_breath_modalities).issubset(
            set(hst_all_context["modality"].astype(str))
        ):
            hst_cough_breath = hst_all_context.loc[
                hst_all_context["modality"].isin(cough_breath_modalities)
            ].copy()
            cough_breath_keys = _complete_keys(
                hst_cough_breath,
                cough_breath_modalities,
            )
            if cough_breath_keys:
                cough_breath_primary = _filter_cohort(
                    hst_cough_breath,
                    cough_breath_keys,
                )
                sensitivity_outputs = _run_family_bank(
                    cough_breath_primary,
                    hst_cough_breath,
                    source_family="hst",
                    random_state=42,
                    modalities=cough_breath_modalities,
                    primary_analysis=False,
                    comparison_binding_hash=comparison_binding_hash,
                    authenticated_registry_receipt_sha256=(
                        authenticated_registry_sha256
                    ),
                    authenticated_context_binding=authenticated_entry,
                )
                prediction_frames.extend(sensitivity_outputs[0])
                weight_rows.extend(sensitivity_outputs[1])
                stacker_rows.extend(sensitivity_outputs[2])
                count_rows.extend(sensitivity_outputs[3])

        if comparator_context is not None:
            comparator_primary = _filter_cohort(comparator_context, joint_keys)
            family_outputs = _run_family_bank(
                comparator_primary,
                comparator_context,
                source_family="comparator",
                random_state=42,
                modalities=PRIMARY_MODALITIES,
                primary_analysis=analysis_mode == "confirmatory",
                comparison_binding_hash=comparison_binding_hash,
                authenticated_registry_receipt_sha256=authenticated_registry_sha256,
                authenticated_context_binding=authenticated_entry,
            )
            prediction_frames.extend(family_outputs[0])
            weight_rows.extend(family_outputs[1])
            stacker_rows.extend(family_outputs[2])
            count_rows.extend(family_outputs[3])

            hybrid = build_four_branch_hybrid_inputs(
                hst_primary,
                comparator_primary,
                authenticated_registry_receipt_sha256=(
                    authenticated_registry_sha256
                ),
                authenticated_context_binding_sha256=authenticated_context_hash,
            )
            hybrid_outputs = _run_hybrid_bank(
                hybrid,
                random_state=42,
                authenticated_registry_receipt_sha256=(
                    authenticated_registry_sha256
                ),
                authenticated_context_binding=authenticated_entry,
            )
            prediction_frames.extend(hybrid_outputs[0])
            weight_rows.extend(hybrid_outputs[1])
            stacker_rows.extend(hybrid_outputs[2])
            count_rows.extend(hybrid_outputs[3])

    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    predictions = predictions.sort_values(
        [*CONTEXT_COLUMNS, "split", "participant_key", "source_family", "fusion_method"],
        kind="stable",
    ).reset_index(drop=True)
    metrics = _metric_table(predictions)
    weights = pd.DataFrame(weight_rows).sort_values(
        [*CONTEXT_COLUMNS, "source_family", "fusion_method", "branch"],
        kind="stable",
    ).reset_index(drop=True)
    stacker_parameters = pd.DataFrame(stacker_rows).sort_values(
        [*CONTEXT_COLUMNS, "source_family", "fusion_method", "branch"],
        kind="stable",
    ).reset_index(drop=True)
    complete_case_counts = pd.DataFrame(count_rows).sort_values(
        [*CONTEXT_COLUMNS, "source_family", "split"],
        kind="stable",
    ).reset_index(drop=True)
    paired_deltas = pd.DataFrame(
        _paired_delta_rows(predictions),
        columns=PAIRED_DELTA_COLUMNS,
    )
    return HSTFusionResult(
        predictions=predictions,
        metrics=metrics,
        weights=weights,
        stacker_parameters=stacker_parameters,
        complete_case_counts=complete_case_counts,
        paired_deltas=paired_deltas,
    )
