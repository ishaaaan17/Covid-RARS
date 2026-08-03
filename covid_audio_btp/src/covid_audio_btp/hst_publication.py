from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from covid_audio_btp import hst_reporting
from covid_audio_btp.hst_runtime import canonical_json_sha256, stable_file_sha256
from covid_audio_btp.metrics import labels_to_binary


PRIMARY_ESTIMAND_ID = "primary_hst_vs_comparator_uniform_cough_speech_auroc"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_RECEIPT_AUTHENTICATOR = object()
_TEST_AUTHENTICATOR = object()


@dataclass(frozen=True)
class AnalysisScope:
    estimand_id: str
    role: str
    scope: str
    family: str
    metric: str
    design: str
    confirmatory: bool


_REGISTERED_SCOPES = {
    PRIMARY_ESTIMAND_ID: AnalysisScope(
        estimand_id=PRIMARY_ESTIMAND_ID,
        role="primary",
        scope="confirmatory",
        family="confirmatory_primary_single",
        metric="auroc",
        design="paired_model",
        confirmatory=True,
    ),
    "split_policy_temporal_contrast": AnalysisScope(
        estimand_id="split_policy_temporal_contrast",
        role="secondary",
        scope="reliability_evaluation",
        family="prespecified_reliability",
        metric="auroc",
        design="split_policy",
        confirmatory=True,
    ),
    "common_late_temporal_contrast": AnalysisScope(
        estimand_id="common_late_temporal_contrast",
        role="secondary",
        scope="reliability_evaluation",
        family="prespecified_reliability",
        metric="auroc",
        design="split_policy",
        confirmatory=True,
    ),
    "coswara_to_coughvid_external_transfer": AnalysisScope(
        estimand_id="coswara_to_coughvid_external_transfer",
        role="secondary",
        scope="reliability_evaluation",
        family="prespecified_reliability",
        metric="auroc",
        design="external_independent",
        confirmatory=True,
    ),
    "secondary_hybrid_vs_hst_auroc": AnalysisScope(
        estimand_id="secondary_hybrid_vs_hst_auroc",
        role="secondary",
        scope="secondary",
        family="secondary_hybrid_deltas",
        metric="auroc",
        design="paired_model",
        confirmatory=False,
    ),
    "secondary_hybrid_vs_comparator_auroc": AnalysisScope(
        estimand_id="secondary_hybrid_vs_comparator_auroc",
        role="secondary",
        scope="secondary",
        family="secondary_hybrid_deltas",
        metric="auroc",
        design="paired_model",
        confirmatory=False,
    ),
    "secondary_hst_vs_comparator_uniform_cough_speech_auprc": AnalysisScope(
        estimand_id="secondary_hst_vs_comparator_uniform_cough_speech_auprc",
        role="secondary",
        scope="secondary",
        family="secondary_discrimination_metrics",
        metric="auprc",
        design="paired_model",
        confirmatory=False,
    ),
    "secondary_fusion_vs_best_constituent_auroc": AnalysisScope(
        estimand_id="secondary_fusion_vs_best_constituent_auroc",
        role="secondary",
        scope="secondary",
        family="secondary_fusion_deltas",
        metric="auroc",
        design="paired_model",
        confirmatory=False,
    ),
}
ANALYSIS_SCOPE_REGISTRY: Mapping[str, AnalysisScope] = MappingProxyType(
    _REGISTERED_SCOPES
)


@dataclass(frozen=True)
class AuthenticatedTable:
    frame: pd.DataFrame
    source_name: str
    table_sha256: str
    manifest_sha256: str
    artifact_sha256: str | None = None
    receipt_sha256: str | None = None
    receipt_record_hash: str | None = None
    run_root: str | None = None
    relative_path: str | None = None
    stage: str | None = None
    provenance_verified: bool = False
    test_mode: bool = False
    derivation_sha256: str | None = None
    source_receipt_sha256s: tuple[str, ...] = ()
    _authenticator: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class AnalysisPlanBinding:
    frame: pd.DataFrame
    plan_sha256: str
    source_table_sha256: str
    source_artifact_sha256: str | None
    source_stage: str | None
    source_relative_path: str | None
    receipt_sha256: str | None
    receipt_record_hash: str | None
    provenance_verified: bool
    test_mode: bool


@dataclass(frozen=True)
class PublicationComparison:
    estimand_id: str
    left: AuthenticatedTable
    right: AuthenticatedTable
    common_test: bool | None = None
    ensemble_right: AuthenticatedTable | None = None


_PLAN_COLUMNS = {
    "estimand_id",
    "analysis_role",
    "analysis_scope",
    "multiplicity_family",
    "metric",
    "comparison_design",
}


def dataframe_sha256(frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Publication tables must be pandas DataFrames")
    payload = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        date_format="%Y-%m-%dT%H:%M:%S.%f",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def authenticate_table(
    frame: pd.DataFrame,
    *,
    source_name: str,
    manifest_sha256: str,
    expected_table_sha256: str | None = None,
    test_mode: bool = False,
) -> AuthenticatedTable:
    if test_mode is not True:
        raise ValueError(
            "Ad-hoc publication tables are permitted only with explicit test_mode=True"
        )
    if not source_name.strip():
        raise ValueError("source_name is required")
    if not _SHA256_RE.fullmatch(str(manifest_sha256)):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
    snapshot = frame.copy(deep=True)
    actual = dataframe_sha256(snapshot)
    if expected_table_sha256 is not None and actual != expected_table_sha256:
        raise ValueError("Publication table checksum does not match the authenticated value")
    return AuthenticatedTable(
        frame=snapshot,
        source_name=source_name,
        table_sha256=actual,
        manifest_sha256=manifest_sha256,
        provenance_verified=False,
        test_mode=True,
        _authenticator=_TEST_AUTHENTICATOR,
    )


def load_receipted_table(
    *,
    run_root: str | Path,
    stage: str,
    relative_path: str | Path,
    expected_receipt_sha256: str,
) -> AuthenticatedTable:
    if not _STAGE_NAME_RE.fullmatch(stage):
        raise ValueError("stage name is not a valid HST stage identity")
    if not _SHA256_RE.fullmatch(str(expected_receipt_sha256)):
        raise ValueError("expected_receipt_sha256 must be a lowercase SHA-256 digest")
    root_candidate = Path(run_root)
    if root_candidate.is_symlink():
        raise ValueError("Trusted HST run root must not be a symlink")
    root = root_candidate.resolve(strict=True)
    supplied = Path(relative_path)
    if supplied.is_absolute() or ".." in supplied.parts:
        raise ValueError("Receipted table path must be relative to the trusted run root")
    relative = supplied.as_posix()
    table_path = (root / supplied).resolve(strict=True)
    try:
        table_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Receipted table escapes the trusted run root") from exc
    if table_path.is_symlink() or not table_path.is_file():
        raise ValueError("Receipted table must be a regular non-symlink file")

    receipt_path = root / "runtime" / "stages" / f"{stage}.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("Successful HST stage receipt is missing or unsafe")
    actual_receipt_sha256 = stable_file_sha256(receipt_path)
    if actual_receipt_sha256 != expected_receipt_sha256:
        raise ValueError("HST stage receipt does not match the independently trusted checksum")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("HST stage receipt is not canonical ASCII JSON") from exc
    if not isinstance(receipt, dict):
        raise ValueError("HST stage receipt must be a JSON object")
    if receipt.get("receipt_type") != "hst_stage":
        raise ValueError("HST stage receipt has an invalid receipt_type")
    claimed_record_hash = receipt.get("record_hash")
    unsigned = {key: value for key, value in receipt.items() if key != "record_hash"}
    if claimed_record_hash != canonical_json_sha256(unsigned):
        raise ValueError("HST stage receipt record checksum is invalid")
    if receipt.get("status") != "success" or receipt.get("stage") != stage:
        raise ValueError("HST table requires its matching successful stage receipt")
    output_paths = receipt.get("output_paths")
    checksums = receipt.get("output_checksums")
    if not isinstance(output_paths, list) or relative not in output_paths:
        raise ValueError("Table is not declared by the successful HST stage receipt")
    if not isinstance(checksums, Mapping) or relative not in checksums:
        raise ValueError("Table lacks a checksum in the successful HST stage receipt")
    actual_table_sha256 = stable_file_sha256(table_path)
    if str(checksums[relative]) != actual_table_sha256:
        raise ValueError("Receipted publication table checksum does not match its stage receipt")
    frame = pd.read_csv(table_path, low_memory=False)
    manifest_sha256 = str(claimed_record_hash)
    if "manifest_sha256" in frame:
        values = frame["manifest_sha256"].dropna().astype(str).unique().tolist()
        if len(values) != 1 or not _SHA256_RE.fullmatch(values[0]):
            raise ValueError("Receipted table has an ambiguous or invalid manifest_sha256")
        manifest_sha256 = values[0]
    return AuthenticatedTable(
        frame=frame.copy(deep=True),
        source_name=relative,
        table_sha256=dataframe_sha256(frame),
        manifest_sha256=manifest_sha256,
        artifact_sha256=actual_table_sha256,
        receipt_sha256=actual_receipt_sha256,
        receipt_record_hash=str(claimed_record_hash),
        run_root=root.as_posix(),
        relative_path=relative,
        stage=stage,
        provenance_verified=True,
        test_mode=False,
        source_receipt_sha256s=(actual_receipt_sha256,),
        _authenticator=_RECEIPT_AUTHENTICATOR,
    )


def derive_authenticated_table(
    frame: pd.DataFrame,
    *,
    source_name: str,
    sources: Sequence[AuthenticatedTable],
    analysis_plan: AnalysisPlanBinding,
    test_mode: bool = False,
) -> AuthenticatedTable:
    if not source_name.strip():
        raise ValueError("source_name is required")
    if not sources:
        raise ValueError("A derived table requires at least one authenticated source")
    _require_plan_estimand(
        analysis_plan,
        PRIMARY_ESTIMAND_ID,
        test_mode=test_mode,
    )
    for source in sources:
        _verified_frame(source, require_confirmatory=not test_mode)
        if test_mode and not source.test_mode:
            raise ValueError("test_mode derivation requires explicit test-mode source tables")
    snapshot = frame.copy(deep=True)
    table_sha256 = dataframe_sha256(snapshot)
    source_receipts = tuple(
        sorted(
            {
                str(source.receipt_sha256)
                for source in sources
                if source.receipt_sha256 is not None
            }
        )
    )
    source_manifests = sorted({source.manifest_sha256 for source in sources})
    derivation_sha256 = canonical_json_sha256(
        {
            "source_name": source_name,
            "derived_table_sha256": table_sha256,
            "source_table_sha256s": sorted(source.table_sha256 for source in sources),
            "source_artifact_sha256s": sorted(
                str(source.artifact_sha256)
                for source in sources
                if source.artifact_sha256 is not None
            ),
            "source_receipt_sha256s": list(source_receipts),
            "analysis_plan_sha256": analysis_plan.plan_sha256,
            "analysis_plan_receipt_sha256": analysis_plan.receipt_sha256 or "",
        }
    )
    run_roots = {str(source.run_root) for source in sources if source.run_root is not None}
    if not test_mode and len(run_roots) != 1:
        raise ValueError("Confirmatory derived sources must share one trusted run root")
    return AuthenticatedTable(
        frame=snapshot,
        source_name=source_name,
        table_sha256=table_sha256,
        manifest_sha256=canonical_json_sha256(source_manifests),
        artifact_sha256=table_sha256,
        receipt_sha256=(derivation_sha256 if not test_mode else None),
        receipt_record_hash=(derivation_sha256 if not test_mode else None),
        run_root=(next(iter(run_roots)) if run_roots else None),
        relative_path=None,
        stage="derived_publication",
        provenance_verified=not test_mode,
        test_mode=test_mode,
        derivation_sha256=derivation_sha256,
        source_receipt_sha256s=source_receipts,
        _authenticator=(
            _TEST_AUTHENTICATOR if test_mode else _RECEIPT_AUTHENTICATOR
        ),
    )


def _verified_frame(
    table: AuthenticatedTable,
    *,
    require_confirmatory: bool = False,
) -> pd.DataFrame:
    if not isinstance(table, AuthenticatedTable):
        raise TypeError("An AuthenticatedTable is required")
    if not _SHA256_RE.fullmatch(table.table_sha256):
        raise ValueError("Authenticated table has an invalid SHA-256 digest")
    if not _SHA256_RE.fullmatch(table.manifest_sha256):
        raise ValueError("Authenticated table has an invalid manifest digest")
    expected_authenticator = (
        _TEST_AUTHENTICATOR if table.test_mode else _RECEIPT_AUTHENTICATOR
    )
    if table._authenticator is not expected_authenticator:
        raise ValueError(
            "Authenticated table was not minted by an independently verified receipt "
            "authenticator or the explicit test-mode authenticator"
        )
    if dataframe_sha256(table.frame) != table.table_sha256:
        raise ValueError(f"Authenticated table {table.source_name!r} was mutated")
    if require_confirmatory and (
        not table.provenance_verified
        or table.test_mode
        or not table.receipt_sha256
        or not table.receipt_record_hash
    ):
        raise ValueError(
            "Confirmatory publication evidence requires an independently verified stage receipt; "
            "an ad-hoc test-mode table cannot cross this boundary"
        )
    if table.artifact_sha256 is not None and not _SHA256_RE.fullmatch(
        table.artifact_sha256
    ):
        raise ValueError("Authenticated table has an invalid artifact digest")
    return table.frame.copy(deep=True)


def freeze_analysis_plan(plan: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(_PLAN_COLUMNS - set(plan.columns))
    if missing:
        raise ValueError(f"Analysis plan missing columns: {missing}")
    if plan.empty:
        raise ValueError("Analysis plan is empty")
    if plan["estimand_id"].duplicated().any():
        raise ValueError("Analysis plan contains duplicate estimand_id values")

    for row in plan.to_dict(orient="records"):
        estimand_id = str(row["estimand_id"])
        registered = ANALYSIS_SCOPE_REGISTRY.get(estimand_id)
        if registered is None:
            role = str(row["analysis_role"])
            scope = str(row["analysis_scope"])
            if role not in {"sensitivity", "exploratory"}:
                raise ValueError(
                    f"Unregistered estimand {estimand_id!r} cannot be primary or secondary"
                )
            if scope != role:
                raise ValueError(
                    f"Unregistered {role} estimand must use matching analysis_scope"
                )
            continue
        expected = {
            "analysis_role": registered.role,
            "analysis_scope": registered.scope,
            "multiplicity_family": registered.family,
            "metric": registered.metric,
            "comparison_design": registered.design,
        }
        for column, value in expected.items():
            if str(row[column]) != value:
                raise ValueError(
                    f"Estimand {estimand_id!r} contradicts frozen {column}={value!r}"
                )

    primary = plan.loc[plan["analysis_role"].eq("primary")]
    if len(primary) != 1 or primary["estimand_id"].item() != PRIMARY_ESTIMAND_ID:
        raise ValueError("Analysis plan must contain exactly one registered primary estimand")
    row = primary.iloc[0]
    exact_primary = {
        "candidate_family": "hst",
        "reference_family": "comparator",
        "split": "test",
        "fusion_method": "uniform_mean",
        "modality_combination": "cough+speech",
    }
    required_primary = set(exact_primary) | {"complete_case"}
    missing_primary = sorted(required_primary - set(plan.columns))
    if missing_primary:
        raise ValueError(f"Primary analysis contract missing columns: {missing_primary}")
    for column, expected in exact_primary.items():
        if str(row[column]) != expected:
            raise ValueError(f"Primary analysis requires {column}={expected!r}")
    if not isinstance(row["complete_case"], (bool, np.bool_)):
        raise ValueError("Primary complete_case must be a boolean")
    if not bool(row["complete_case"]):
        raise ValueError("Primary analysis requires complete_case=True")
    return plan.sort_values("estimand_id", kind="mergesort").reset_index(drop=True).copy()


def bind_analysis_plan(
    table: AuthenticatedTable,
    *,
    test_mode: bool = False,
) -> AnalysisPlanBinding:
    frame = _verified_frame(table, require_confirmatory=not test_mode)
    if test_mode and not table.test_mode:
        raise ValueError("A test analysis-plan binding requires an explicit test-mode table")
    frozen = freeze_analysis_plan(frame)
    return AnalysisPlanBinding(
        frame=frozen,
        plan_sha256=dataframe_sha256(frozen),
        source_table_sha256=table.table_sha256,
        source_artifact_sha256=table.artifact_sha256,
        source_stage=table.stage,
        source_relative_path=table.relative_path,
        receipt_sha256=table.receipt_sha256,
        receipt_record_hash=table.receipt_record_hash,
        provenance_verified=table.provenance_verified,
        test_mode=table.test_mode,
    )


def _require_plan_estimand(
    analysis_plan: AnalysisPlanBinding,
    estimand_id: str,
    *,
    test_mode: bool,
) -> pd.Series:
    if not isinstance(analysis_plan, AnalysisPlanBinding):
        raise TypeError("Evidence generation requires an authenticated frozen analysis plan")
    if test_mode:
        if not analysis_plan.test_mode:
            raise ValueError("test_mode evidence requires a test-mode analysis-plan binding")
    elif (
        not analysis_plan.provenance_verified
        or analysis_plan.test_mode
        or not analysis_plan.receipt_sha256
        or not analysis_plan.receipt_record_hash
    ):
        raise ValueError(
            "Confirmatory evidence requires a receipt-backed frozen analysis plan"
        )
    matches = analysis_plan.frame.loc[
        analysis_plan.frame["estimand_id"].astype(str).eq(estimand_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Estimand {estimand_id!r} is absent from the frozen analysis plan"
        )
    return matches.iloc[0]


def _plan_provenance(analysis_plan: AnalysisPlanBinding) -> dict[str, object]:
    return {
        "analysis_plan_sha256": analysis_plan.plan_sha256,
        "analysis_plan_source_table_sha256": analysis_plan.source_table_sha256,
        "analysis_plan_source_artifact_sha256": (
            analysis_plan.source_artifact_sha256 or ""
        ),
        "analysis_plan_source_stage": analysis_plan.source_stage or "",
        "analysis_plan_source_relative_path": (
            analysis_plan.source_relative_path or ""
        ),
        "analysis_plan_receipt_sha256": analysis_plan.receipt_sha256 or "",
        "analysis_plan_receipt_record_hash": analysis_plan.receipt_record_hash or "",
    }


def _source_provenance(
    table: AuthenticatedTable,
    *,
    prefix: str = "source",
) -> dict[str, object]:
    table_key = "source_table_sha256" if prefix == "source" else f"{prefix}_source_sha256"
    artifact_key = (
        "source_artifact_sha256"
        if prefix == "source"
        else f"{prefix}_source_artifact_sha256"
    )
    manifest_key = (
        "source_manifest_sha256"
        if prefix == "source"
        else f"{prefix}_source_manifest_sha256"
    )
    receipt_key = (
        "source_receipt_sha256"
        if prefix == "source"
        else f"{prefix}_source_receipt_sha256"
    )
    record_key = (
        "source_receipt_record_hash"
        if prefix == "source"
        else f"{prefix}_source_receipt_record_hash"
    )
    stage_key = "source_stage" if prefix == "source" else f"{prefix}_source_stage"
    path_key = (
        "source_relative_path"
        if prefix == "source"
        else f"{prefix}_source_relative_path"
    )
    derivation_key = (
        "source_derivation_sha256"
        if prefix == "source"
        else f"{prefix}_source_derivation_sha256"
    )
    return {
        table_key: table.table_sha256,
        artifact_key: table.artifact_sha256 or "",
        manifest_key: table.manifest_sha256,
        receipt_key: table.receipt_sha256 or "",
        record_key: table.receipt_record_hash or "",
        stage_key: table.stage or "",
        path_key: table.relative_path or "",
        derivation_key: table.derivation_sha256 or "",
    }


def _scope(estimand_id: str) -> AnalysisScope:
    try:
        return ANALYSIS_SCOPE_REGISTRY[estimand_id]
    except KeyError as exc:
        raise ValueError(
            f"Bootstrap evidence requires a registered estimand: {estimand_id!r}"
        ) from exc


def _validate_primary_comparison_contract(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> None:
    required_values = {
        "analysis_role": "primary",
        "analysis_scope": "confirmatory",
        "estimand_id": PRIMARY_ESTIMAND_ID,
        "multiplicity_family": "confirmatory_primary_single",
        "split": "test",
        "fusion_method": "uniform_mean",
        "modality_combination": "cough+speech",
    }
    digest_columns = (
        "comparison_binding_hash",
        "authenticated_registry_receipt_sha256",
        "authenticated_context_binding_sha256",
    )
    for table_name, frame, source_family in (
        ("candidate", left, "hst"),
        ("reference", right, "comparator"),
    ):
        required = set(required_values) | {"complete_case", "source_family"} | set(
            digest_columns
        )
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                f"Primary {table_name} table lacks authenticated contract columns: {missing}"
            )
        for column, expected in required_values.items():
            if not frame[column].eq(expected).all():
                raise ValueError(f"Primary comparison requires {column}={expected!r}")
        if not frame["source_family"].eq(source_family).all():
            raise ValueError(
                f"Primary {table_name} source_family must be {source_family!r}"
            )
        if not frame["complete_case"].map(
            lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
        ).all():
            raise ValueError("Primary comparison requires complete_case=True")
        for column in digest_columns:
            if not frame[column].map(
                lambda value: isinstance(value, str)
                and _SHA256_RE.fullmatch(value) is not None
            ).all():
                raise ValueError(f"Primary comparison has invalid {column}")
    for column in digest_columns:
        values = pd.concat([left[column], right[column]], ignore_index=True).unique()
        if len(values) != 1:
            raise ValueError(f"Primary candidate and comparator disagree on {column}")


def build_bootstrap_evidence(
    comparisons: Sequence[PublicationComparison],
    *,
    analysis_plan: AnalysisPlanBinding,
    test_mode: bool = False,
) -> pd.DataFrame:
    if not comparisons:
        raise ValueError("At least one publication comparison is required")
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    contract = hst_reporting.REPORTING_CONTRACT
    n_bootstrap = int(contract["bootstrap_replicates"])
    seed = int(contract["bootstrap_seed"])
    for comparison in comparisons:
        if comparison.estimand_id in seen:
            raise ValueError("Publication comparisons contain a duplicate estimand")
        seen.add(comparison.estimand_id)
        scope = _scope(comparison.estimand_id)
        plan_row = _require_plan_estimand(
            analysis_plan,
            comparison.estimand_id,
            test_mode=test_mode,
        )
        if str(plan_row["comparison_design"]) != scope.design:
            raise ValueError("Frozen analysis plan contradicts the registered comparison design")
        left = _verified_frame(comparison.left, require_confirmatory=not test_mode)
        right = _verified_frame(comparison.right, require_confirmatory=not test_mode)
        if test_mode and (not comparison.left.test_mode or not comparison.right.test_mode):
            raise ValueError("test_mode bootstrap evidence requires explicit test-mode tables")
        if comparison.ensemble_right is not None and scope.design != "external_independent":
            raise ValueError("A separate ensemble table is valid only for external comparisons")
        if scope.role == "primary":
            _validate_primary_comparison_contract(left, right)
        endpoint = "comparison_delta"
        extra_rows: list[dict[str, object]] = []
        if scope.design == "paired_model":
            result = hst_reporting.paired_model_cluster_delta(
                left,
                right,
                metric=scope.metric,
                n_bootstrap=n_bootstrap,
                seed=seed,
                allow_model_input_context_difference=(
                    scope.estimand_id
                    == "secondary_fusion_vs_best_constituent_auroc"
                ),
            )
            test_method = "paired_participant_cluster_bootstrap_ci_only"
        elif scope.design == "external_independent":
            if ("fold" in left) != ("fold" in right):
                raise ValueError("Repeated-fold external comparison requires folds on both tables")
            if "fold" in left:
                if left["fold"].nunique() != 10 or right["fold"].nunique() != 10:
                    raise ValueError("Confirmatory external fold analysis requires all ten source folds")
                result = hst_reporting.external_repeated_fold_delta(
                    left,
                    right,
                    metric=scope.metric,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
                endpoint = "mean_source_fold_vs_mean_external_fold_delta"
                ensemble_table = comparison.ensemble_right or comparison.right
                ensemble_right = _verified_frame(
                    ensemble_table,
                    require_confirmatory=not test_mode,
                )
                if test_mode and not ensemble_table.test_mode:
                    raise ValueError(
                        "test_mode external ensemble requires a test-mode table"
                    )
                identity_columns = ["fold", "participant_key", "label_binary"]
                right_identity = right[identity_columns].sort_values(
                    identity_columns,
                    kind="mergesort",
                ).reset_index(drop=True)
                ensemble_identity = ensemble_right[identity_columns].sort_values(
                    identity_columns,
                    kind="mergesort",
                ).reset_index(drop=True)
                if not right_identity.equals(ensemble_identity):
                    raise ValueError(
                        "External ensemble probabilities must preserve the exact target cohort"
                    )
                probability_scale = "raw"
                if comparison.ensemble_right is not None:
                    scales = ensemble_right.get("probability_scale")
                    if scales is None or not scales.astype(str).eq(
                        "source_validation_platt"
                    ).all():
                        raise ValueError(
                            "External ensemble table must use source-validation Platt probabilities"
                        )
                    probability_scale = "source_validation_platt"
                ensemble = hst_reporting.equal_fold_probability_ensemble_ci(
                    ensemble_right,
                    metric=scope.metric,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
                extra_rows.append(
                    {
                        "estimand_id": scope.estimand_id,
                        "analysis_role": scope.role,
                        "analysis_scope": scope.scope,
                        "multiplicity_family": scope.family,
                        "confirmatory": bool(scope.confirmatory and not test_mode),
                        "execution_class": (
                            "exploratory_test_only" if test_mode else "confirmatory"
                        ),
                        "test_method": "descriptive_participant_cluster_bootstrap_ci",
                        "endpoint": "equal_source_fold_probability_ensemble",
                        **_source_provenance(comparison.left, prefix="left"),
                        **_source_provenance(comparison.right, prefix="right"),
                        **_source_provenance(ensemble_table, prefix="ensemble"),
                        "probability_scale": probability_scale,
                        "bootstrap_replicates": n_bootstrap,
                        "bootstrap_seed": seed,
                        **_plan_provenance(analysis_plan),
                        **ensemble,
                    }
                )
                test_method = (
                    "independent_repeated_fold_participant_cluster_bootstrap_ci_only"
                )
            else:
                result = hst_reporting.external_transfer_delta(
                    left,
                    right,
                    metric=scope.metric,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
                test_method = "independent_participant_bootstrap_ci_only"
        elif scope.design == "split_policy":
            if comparison.common_test is None:
                raise ValueError("Split-policy comparisons must declare common_test")
            result = hst_reporting.split_policy_delta(
                (left, right),
                common_test=comparison.common_test,
                metric=scope.metric,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            test_method = "split_policy_participant_bootstrap_ci_only"
        else:
            raise ValueError(f"Unsupported frozen comparison design: {scope.design}")
        rows.append(
            {
                "estimand_id": scope.estimand_id,
                "analysis_role": scope.role,
                "analysis_scope": scope.scope,
                "multiplicity_family": scope.family,
                "confirmatory": bool(scope.confirmatory and not test_mode),
                "execution_class": (
                    "exploratory_test_only" if test_mode else "confirmatory"
                ),
                "test_method": test_method,
                "endpoint": endpoint,
                **_source_provenance(comparison.left, prefix="left"),
                **_source_provenance(comparison.right, prefix="right"),
                "bootstrap_replicates": n_bootstrap,
                "bootstrap_seed": seed,
                **_plan_provenance(analysis_plan),
                **result,
            }
        )
        rows.extend(extra_rows)
    return pd.DataFrame(rows).sort_values("estimand_id", kind="mergesort").reset_index(drop=True)


def _validation_discrimination(frame: pd.DataFrame) -> tuple[float, float, int]:
    required = {"participant_key", "label_binary", "probability", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Constituent validation table missing columns: {missing}")
    if frame.empty or not frame["split"].astype(str).eq("validation").all():
        raise ValueError("Constituent selection is permitted only on validation predictions")
    identity = ["participant_key"] + (["fold"] if "fold" in frame else [])
    if frame.duplicated(identity).any():
        raise ValueError("Constituent validation predictions contain duplicate participants")
    probabilities = pd.to_numeric(frame["probability"], errors="coerce")
    if probabilities.isna().any() or (~probabilities.between(0.0, 1.0)).any():
        raise ValueError("Constituent validation probabilities must be finite and in [0, 1]")
    groups = frame.groupby("fold", sort=True) if "fold" in frame else [(None, frame)]
    aurocs: list[float] = []
    auprcs: list[float] = []
    for _fold, group in groups:
        labels = labels_to_binary(group["label_binary"])
        if np.unique(labels).size != 2:
            raise ValueError("Every constituent validation fold must contain both classes")
        scores = pd.to_numeric(group["probability"], errors="raise").to_numpy(dtype=float)
        aurocs.append(float(roc_auc_score(labels, scores)))
        auprcs.append(float(average_precision_score(labels, scores)))
    return (
        float(np.mean(aurocs)),
        float(np.mean(auprcs)),
        int(frame["participant_key"].nunique()),
    )


def _assert_same_prediction_cohort(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    split: str,
) -> None:
    for name, frame in (("left", left), ("right", right)):
        if "split" not in frame or not frame["split"].astype(str).eq(split).all():
            raise ValueError(f"{name} constituent comparison table must contain only {split}")
    if ("fold" in left) != ("fold" in right):
        raise ValueError("Constituent comparison tables disagree on fold structure")
    keys = ["participant_key"] + (["fold"] if "fold" in left else [])
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("Constituent comparison identities must be unique")
    left_ordered = left.sort_values(keys, kind="mergesort").reset_index(drop=True)
    right_ordered = right.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if not left_ordered[keys].equals(right_ordered[keys]) or not left_ordered[
        "label_binary"
    ].astype("string").equals(right_ordered["label_binary"].astype("string")):
        raise ValueError("Constituent comparison requires an exact paired participant cohort")


def build_fusion_vs_best_constituent_evidence(
    fusion_test: AuthenticatedTable,
    constituent_validation: Mapping[str, AuthenticatedTable],
    constituent_test: Mapping[str, AuthenticatedTable],
    *,
    analysis_plan: AnalysisPlanBinding,
    test_mode: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimand_id = "secondary_fusion_vs_best_constituent_auroc"
    _require_plan_estimand(analysis_plan, estimand_id, test_mode=test_mode)
    if not constituent_validation or set(constituent_validation) != set(constituent_test):
        raise ValueError(
            "Validation and test constituent candidate names must match exactly"
        )
    fusion_frame = _verified_frame(
        fusion_test, require_confirmatory=not test_mode
    )
    if test_mode and not fusion_test.test_mode:
        raise ValueError("test_mode fusion evidence requires explicit test-mode tables")
    rows: list[dict[str, object]] = []
    reference_validation: pd.DataFrame | None = None
    reference_test: pd.DataFrame | None = None
    test_frames: dict[str, pd.DataFrame] = {}
    for name in sorted(constituent_validation):
        validation_table = constituent_validation[name]
        test_table = constituent_test[name]
        validation = _verified_frame(
            validation_table, require_confirmatory=not test_mode
        )
        test = _verified_frame(test_table, require_confirmatory=not test_mode)
        if test_mode and (not validation_table.test_mode or not test_table.test_mode):
            raise ValueError("test_mode constituent evidence requires test-mode tables")
        validation_groups = (
            validation.groupby("fold", sort=True)
            if "fold" in validation
            else [(None, validation)]
        )
        if reference_validation is None:
            reference_validation = validation
            reference_test = test
        else:
            _assert_same_prediction_cohort(
                reference_validation,
                validation,
                split="validation",
            )
            _assert_same_prediction_cohort(reference_test, test, split="test")
        _assert_same_prediction_cohort(fusion_frame, test, split="test")
        test_frames[str(name)] = test
        for fold, validation_group in validation_groups:
            auroc, auprc, n_validation = _validation_discrimination(
                validation_group
            )
            rows.append(
                {
                    "fold": fold,
                    "constituent": str(name),
                    "validation_auroc": auroc,
                    "validation_auprc": auprc,
                    "n_validation_participants": n_validation,
                    **_source_provenance(validation_table, prefix="validation"),
                    **_source_provenance(test_table, prefix="test"),
                    **_plan_provenance(analysis_plan),
                }
            )
    selection = pd.DataFrame(rows).sort_values(
        ["fold", "validation_auroc", "validation_auprc", "constituent"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    selection["selection_rank"] = (
        selection.groupby("fold", dropna=False, sort=False).cumcount() + 1
    )
    selection["selected"] = selection["selection_rank"].eq(1)
    selection["selection_split"] = "validation"
    selection["selection_primary_metric"] = "auroc"
    selection["selection_tiebreak_metric"] = "auprc"
    selection["selection_final_tiebreak"] = "constituent_name_ascending"
    selection["estimand_id"] = estimand_id
    selected_rows = selection.loc[selection["selected"]].copy()
    repeated = selected_rows["fold"].notna().any()
    if repeated:
        selected_test_parts = [
            test_frames[str(row.constituent)].loc[
                test_frames[str(row.constituent)]["fold"].eq(row.fold)
            ]
            for row in selected_rows.itertuples(index=False)
        ]
        selected_test_frame = pd.concat(
            selected_test_parts,
            ignore_index=True,
            sort=False,
        ).sort_values(["fold", "participant_key"], kind="mergesort").reset_index(
            drop=True
        )
        selected_test_table = derive_authenticated_table(
            selected_test_frame,
            source_name="fold_local_validation_selected_constituent_test",
            sources=[
                *constituent_validation.values(),
                *constituent_test.values(),
            ],
            analysis_plan=analysis_plan,
            test_mode=test_mode,
        )
        selected_name = "fold_local"
        selection_policy = "fold_local_validation"
    else:
        selected_name = str(selected_rows.iloc[0]["constituent"])
        selected_test_table = constituent_test[selected_name]
        selection_policy = "single_validation_cohort"
    evidence = build_bootstrap_evidence(
        [
            PublicationComparison(
                estimand_id=estimand_id,
                left=fusion_test,
                right=selected_test_table,
            )
        ],
        analysis_plan=analysis_plan,
        test_mode=test_mode,
    )
    evidence["selected_constituent"] = selected_name
    evidence["selection_policy"] = selection_policy
    evidence["selected_constituents_by_fold_json"] = json.dumps(
        {
            str(row.fold): str(row.constituent)
            for row in selected_rows.itertuples(index=False)
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    evidence["selected_constituent_validation_auroc"] = float(
        selected_rows["validation_auroc"].mean()
    )
    evidence["selected_constituent_validation_auprc"] = float(
        selected_rows["validation_auprc"].mean()
    )
    evidence["selection_split"] = "validation"
    evidence["selection_primary_metric"] = "auroc"
    evidence["selection_tiebreak_metric"] = "auprc"
    evidence["selection_final_tiebreak"] = "constituent_name_ascending"
    return evidence, selection


def build_repeated_fold_evidence(
    predictions: AuthenticatedTable,
    *,
    estimand_id: str,
    metric: str,
    analysis_plan: AnalysisPlanBinding,
    test_mode: bool = False,
) -> dict[str, object]:
    _require_plan_estimand(analysis_plan, estimand_id, test_mode=test_mode)
    frame = _verified_frame(predictions, require_confirmatory=not test_mode)
    if test_mode and not predictions.test_mode:
        raise ValueError("test_mode repeated-fold evidence requires a test-mode table")
    if "fold" not in frame:
        raise ValueError("Repeated-fold evidence requires a fold column")
    result = hst_reporting.repeated_holdout_cluster_ci(
        frame,
        metric=metric,
        n_bootstrap=int(hst_reporting.REPORTING_CONTRACT["bootstrap_replicates"]),
        seed=int(hst_reporting.REPORTING_CONTRACT["bootstrap_seed"]),
    )
    return {
        "estimand_id": estimand_id,
        "analysis_role": "exploratory",
        "analysis_scope": "exploratory",
        "multiplicity_family": "exploratory_repeated_holdout",
        **_source_provenance(predictions),
        "confirmatory": False,
        "execution_class": "exploratory_test_only" if test_mode else "confirmatory",
        **_plan_provenance(analysis_plan),
        **result,
    }


def build_paired_delong_evidence(
    comparison: PublicationComparison,
    *,
    analysis_plan: AnalysisPlanBinding,
    test_mode: bool = False,
) -> pd.DataFrame:
    scope = _scope(comparison.estimand_id)
    _require_plan_estimand(analysis_plan, comparison.estimand_id, test_mode=test_mode)
    if scope.design != "paired_model" or scope.metric != "auroc":
        raise ValueError("Paired DeLong is registered only for paired AUROC estimands")
    left = _verified_frame(comparison.left, require_confirmatory=not test_mode)
    right = _verified_frame(comparison.right, require_confirmatory=not test_mode)
    if test_mode and (not comparison.left.test_mode or not comparison.right.test_mode):
        raise ValueError("test_mode DeLong evidence requires explicit test-mode tables")
    if scope.role == "primary":
        _validate_primary_comparison_contract(left, right)
    if "fold" in left or "fold" in right:
        if "fold" not in left or "fold" not in right:
            raise ValueError("Repeated-holdout DeLong audit requires fold on both tables")
        if left["fold"].isna().any() or right["fold"].isna().any():
            raise ValueError("Repeated-holdout DeLong audit does not allow missing folds")
        keys = ["participant_key", "fold"]
        if left.duplicated(keys).any() or right.duplicated(keys).any():
            raise ValueError(
                "Repeated-holdout DeLong audit requires unique participant-fold rows"
            )
        left_ordered = left.sort_values(keys, kind="mergesort").reset_index(drop=True)
        right_ordered = right.sort_values(keys, kind="mergesort").reset_index(drop=True)
        if not left_ordered[keys].equals(right_ordered[keys]) or not left_ordered[
            "label_binary"
        ].astype("string").equals(right_ordered["label_binary"].astype("string")):
            raise ValueError(
                "Repeated-holdout DeLong audit requires exact paired participant-fold rows"
            )
        left_folds = set(left_ordered["fold"].tolist())
        right_folds = set(right_ordered["fold"].tolist())
        if left_folds != right_folds:
            raise ValueError("Repeated-holdout DeLong audit requires identical fold identities")
        return pd.DataFrame(
            [
                {
                    "estimand_id": scope.estimand_id,
                    "analysis_role": "descriptive",
                    "declared_analysis_role": scope.role,
                    "analysis_scope": "audit",
                    "multiplicity_family": "not_applicable_descriptive_skip",
                    "confirmatory": False,
                    "execution_class": (
                        "exploratory_test_only" if test_mode else "confirmatory_run_audit"
                    ),
                    "test_method": "not_run_repeated_holdout",
                    "skipped": True,
                    "skip_reason": (
                        "Paired DeLong requires one exact paired test set; repeated "
                        "holdouts were not pooled"
                    ),
                    "fold_count": len(left_folds),
                    "n": len(left_ordered),
                    "n_unique_participants": int(
                        left_ordered["participant_key"].nunique()
                    ),
                    "pooled_repeated_rows": False,
                    "p_value": np.nan,
                    "p_value_holm": np.nan,
                    **_source_provenance(comparison.left, prefix="left"),
                    **_source_provenance(comparison.right, prefix="right"),
                    **_plan_provenance(analysis_plan),
                }
            ]
        )
    result = hst_reporting.paired_delong_auc_test(left, right)
    return pd.DataFrame(
        [
            {
                "estimand_id": scope.estimand_id,
                "analysis_role": scope.role,
                "analysis_scope": scope.scope,
                "multiplicity_family": scope.family,
                "confirmatory": bool(scope.confirmatory and not test_mode),
                "execution_class": (
                    "exploratory_test_only" if test_mode else "confirmatory"
                ),
                "test_method": "paired_delong",
                "skipped": False,
                "pooled_repeated_rows": False,
                **_source_provenance(comparison.left, prefix="left"),
                **_source_provenance(comparison.right, prefix="right"),
                **_plan_provenance(analysis_plan),
                **result,
            }
        ]
    )


def adjust_secondary_holm(
    evidence: pd.DataFrame,
    *,
    analysis_plan: AnalysisPlanBinding,
    test_mode: bool = False,
) -> pd.DataFrame:
    required = {"estimand_id", "analysis_role", "multiplicity_family", "p_value"}
    missing = sorted(required - set(evidence.columns))
    if missing:
        raise ValueError(f"Holm adjustment table missing columns: {missing}")
    result = evidence.copy()
    result["p_value_holm"] = np.nan
    secondary = result["analysis_role"].eq("secondary")
    for row in result.loc[secondary].itertuples(index=False):
        planned = _require_plan_estimand(
            analysis_plan,
            str(row.estimand_id),
            test_mode=test_mode,
        )
        if str(planned["analysis_role"]) != "secondary" or str(
            planned["multiplicity_family"]
        ) != str(row.multiplicity_family):
            raise ValueError("Secondary evidence contradicts its frozen multiplicity family")
    p_values = pd.to_numeric(result.loc[secondary, "p_value"], errors="coerce")
    tested = secondary.copy()
    tested.loc[secondary] = p_values.notna().to_numpy()
    finite = pd.to_numeric(result.loc[tested, "p_value"], errors="coerce")
    if (~finite.between(0.0, 1.0)).any():
        raise ValueError("Secondary p-values must be finite and in [0, 1]")
    for _, indices in result.loc[tested].groupby(
        "multiplicity_family", sort=True
    ).groups.items():
        ordered = sorted(indices, key=lambda index: (float(result.at[index, "p_value"]), index))
        running = 0.0
        total = len(ordered)
        for rank, index in enumerate(ordered):
            candidate = min(1.0, float(result.at[index, "p_value"]) * (total - rank))
            running = max(running, candidate)
            result.at[index, "p_value_holm"] = running
    return result


def _fold_groups(frame: pd.DataFrame):
    if "fold" not in frame:
        yield None, frame
        return
    if frame["fold"].isna().any():
        raise ValueError("Publication evidence requires a non-null fold identity")
    for fold, group in frame.groupby("fold", sort=True):
        yield fold, group.reset_index(drop=True)


_EVALUATION_SPLITS = {"test", "temporal_test", "external_test"}
_EVALUATION_IDENTITY_COLUMNS = (
    "checkpoint_hash",
    "fold",
    "modality",
    "source_protocol",
    "source_manifest_sha256",
)


def _validate_evaluation_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    required = {"split", "protocol", *_EVALUATION_IDENTITY_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} lacks evaluation provenance columns: {missing}")
    splits = set(frame["split"].astype(str))
    if len(splits) != 1 or not splits <= _EVALUATION_SPLITS:
        raise ValueError(
            f"{name} is evaluation-only and must contain test, temporal_test, or external_test"
        )
    external = "external_test" in splits or (
        "dataset" in frame and frame["dataset"].astype(str).eq("coughvid").any()
    )
    if external:
        provenance_columns = {"analysis_unit_type", "subject_linkage_available"}
        if not provenance_columns.issubset(frame.columns):
            raise ValueError(
                f"{name} lacks required external analysis-unit provenance"
            )
        units = frame["analysis_unit_type"].astype(str).unique().tolist()
        linkages = frame["subject_linkage_available"].unique().tolist()
        if units != ["recording_uuid"] or len(linkages) != 1 or not isinstance(
            linkages[0], (bool, np.bool_)
        ) or bool(linkages[0]):
            raise ValueError(
                f"{name} has invalid external analysis-unit provenance"
            )
    for column in ("checkpoint_hash", "source_manifest_sha256"):
        if not frame[column].map(
            lambda value: isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None
        ).all():
            raise ValueError(f"{name} has invalid {column}")
    for fold, group in frame.groupby("fold", dropna=False, sort=False):
        for column in ("protocol", *(_EVALUATION_IDENTITY_COLUMNS)):
            if column == "fold":
                continue
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(
                    f"{name} mixes multiple {column} identities within fold {fold!r}"
                )
    return frame


def _validate_validation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "protocol", *_EVALUATION_IDENTITY_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Source validation lacks provenance columns: {missing}")
    if not frame["split"].astype(str).eq("validation").all():
        raise ValueError("Operating-point fitting is permitted only on source validation")
    if not frame["protocol"].astype(str).equals(frame["source_protocol"].astype(str)):
        raise ValueError("Source validation protocol must equal its frozen source_protocol")
    for column in ("checkpoint_hash", "source_manifest_sha256"):
        if not frame[column].map(
            lambda value: isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None
        ).all():
            raise ValueError(f"Source validation has invalid {column}")
    for fold, group in frame.groupby("fold", dropna=False, sort=False):
        for column in ("protocol", *(_EVALUATION_IDENTITY_COLUMNS)):
            if column == "fold":
                continue
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(
                    f"Source validation mixes multiple {column} identities within fold {fold!r}"
                )
    return frame


def derive_source_platt_calibrated_pair(
    validation_predictions: AuthenticatedTable,
    evaluation_predictions: AuthenticatedTable,
    *,
    source_name: str,
    analysis_plan: AnalysisPlanBinding,
    test_mode: bool = False,
) -> tuple[AuthenticatedTable, AuthenticatedTable, pd.DataFrame]:
    """Derive source-validation Platt probabilities with authenticated lineage."""

    validation = _validate_validation_frame(
        _verified_frame(
            validation_predictions,
            require_confirmatory=not test_mode,
        )
    )
    evaluation = _validate_evaluation_frame(
        _verified_frame(
            evaluation_predictions,
            require_confirmatory=not test_mode,
        ),
        name=f"Platt evaluation {source_name!r}",
    )
    if test_mode and (
        not validation_predictions.test_mode or not evaluation_predictions.test_mode
    ):
        raise ValueError("test_mode calibration derivation requires test-mode sources")
    validation_groups = dict(_fold_groups(validation))
    evaluation_groups = dict(_fold_groups(evaluation))
    if set(validation_groups) != set(evaluation_groups):
        raise ValueError("Platt validation and evaluation folds must match exactly")
    for fold in validation_groups:
        validation_group = validation_groups[fold]
        evaluation_group = evaluation_groups[fold]
        for column in _EVALUATION_IDENTITY_COLUMNS:
            left_values = validation_group[column].astype(str).unique().tolist()
            right_values = evaluation_group[column].astype(str).unique().tolist()
            if len(left_values) != 1 or left_values != right_values:
                raise ValueError(
                    f"Platt validation and evaluation provenance differ for {column}"
                )

    calibrated_validation, calibrated_evaluation, audit = (
        hst_reporting.apply_source_platt_calibration(validation, evaluation)
    )
    sources = [validation_predictions, evaluation_predictions]
    derived_validation = derive_authenticated_table(
        calibrated_validation,
        source_name=f"{source_name}_validation",
        sources=sources,
        analysis_plan=analysis_plan,
        test_mode=test_mode,
    )
    derived_evaluation = derive_authenticated_table(
        calibrated_evaluation,
        source_name=f"{source_name}_evaluation",
        sources=sources,
        analysis_plan=analysis_plan,
        test_mode=test_mode,
    )
    audit = audit.copy()
    audit.insert(0, "series", source_name)
    audit["validation_source"] = validation_predictions.source_name
    audit["evaluation_source"] = evaluation_predictions.source_name
    audit["validation_table_sha256"] = validation_predictions.table_sha256
    audit["evaluation_table_sha256"] = evaluation_predictions.table_sha256
    audit["calibrated_validation_sha256"] = derived_validation.table_sha256
    audit["calibrated_evaluation_sha256"] = derived_evaluation.table_sha256
    audit["analysis_plan_sha256"] = analysis_plan.plan_sha256
    audit["target_labels_used_for_fit"] = False
    return derived_validation, derived_evaluation, audit


def _validate_series_plan(
    predictions: Mapping[str, AuthenticatedTable],
    evidence_estimand_ids: Mapping[str, str],
    analysis_plan: AnalysisPlanBinding,
    *,
    test_mode: bool,
) -> None:
    if set(predictions) != set(evidence_estimand_ids):
        raise ValueError("Evidence series and frozen estimand mapping must match exactly")
    for series in predictions:
        _require_plan_estimand(
            analysis_plan,
            str(evidence_estimand_ids[series]),
            test_mode=test_mode,
        )


def build_calibration_evidence(
    predictions: Mapping[str, AuthenticatedTable],
    *,
    analysis_plan: AnalysisPlanBinding,
    evidence_estimand_ids: Mapping[str, str],
    test_mode: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_series_plan(
        predictions,
        evidence_estimand_ids,
        analysis_plan,
        test_mode=test_mode,
    )
    bin_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for series, table in sorted(predictions.items()):
        frame = _validate_evaluation_frame(
            _verified_frame(table, require_confirmatory=not test_mode),
            name=f"Calibration series {series!r}",
        )
        if test_mode and not table.test_mode:
            raise ValueError("test_mode calibration requires explicit test-mode tables")
        for fold, group in _fold_groups(frame):
            bins, summary = hst_reporting.build_calibration_report(
                group,
                n_bins=int(hst_reporting.REPORTING_CONTRACT["ece_bins"]),
            )
            bins.insert(0, "series", series)
            bins.insert(1, "fold", fold)
            for key, value in _source_provenance(table).items():
                bins[key] = value
            bins["estimand_id"] = evidence_estimand_ids[series]
            for key, value in _plan_provenance(analysis_plan).items():
                bins[key] = value
            bin_rows.append(bins)
            summaries.append(
                {
                    "series": series,
                    "fold": fold,
                    "n_participants": len(group),
                    **_source_provenance(table),
                    "estimand_id": evidence_estimand_ids[series],
                    **_plan_provenance(analysis_plan),
                    **summary,
                }
            )
    if not summaries:
        raise ValueError("Calibration evidence requires at least one prediction table")
    return pd.concat(bin_rows, ignore_index=True), pd.DataFrame(summaries)


def build_fixed_sensitivity_evidence(
    validation_predictions: AuthenticatedTable,
    evaluations: Mapping[str, AuthenticatedTable],
    *,
    analysis_plan: AnalysisPlanBinding,
    evidence_estimand_ids: Mapping[str, str],
    test_mode: bool = False,
) -> pd.DataFrame:
    _validate_series_plan(
        evaluations,
        evidence_estimand_ids,
        analysis_plan,
        test_mode=test_mode,
    )
    validation = _validate_validation_frame(
        _verified_frame(validation_predictions, require_confirmatory=not test_mode)
    )
    if test_mode and not validation_predictions.test_mode:
        raise ValueError("test_mode operating-point evidence requires a test-mode validation table")
    validation_groups = dict(_fold_groups(validation))
    target = float(hst_reporting.REPORTING_CONTRACT["fixed_sensitivity"])
    rows: list[dict[str, object]] = []
    for series, table in sorted(evaluations.items()):
        evaluation = _validate_evaluation_frame(
            _verified_frame(table, require_confirmatory=not test_mode),
            name=f"Operating-point series {series!r}",
        )
        if test_mode and not table.test_mode:
            raise ValueError("test_mode operating-point evidence requires test-mode tables")
        evaluation_groups = dict(_fold_groups(evaluation))
        if set(validation_groups) != set(evaluation_groups):
            raise ValueError("Validation and evaluation folds must match exactly")
        for fold in sorted(validation_groups, key=lambda value: (-1 if value is None else value)):
            validation_group = validation_groups[fold]
            evaluation_group = evaluation_groups[fold]
            for column in _EVALUATION_IDENTITY_COLUMNS:
                left_values = validation_group[column].astype(str).unique().tolist()
                right_values = evaluation_group[column].astype(str).unique().tolist()
                if len(left_values) != 1 or left_values != right_values:
                    raise ValueError(
                        f"Validation and evaluation provenance differ for {column}"
                    )
            operating = hst_reporting.fit_screening_operating_point(
                validation_group,
                target_sensitivity=target,
            )
            metrics = hst_reporting.apply_screening_operating_point(
                evaluation_group, operating
            )
            rows.append(
                {
                    "series": series,
                    "fold": fold,
                    "target_sensitivity": target,
                    "validation_sensitivity": operating.validation_sensitivity,
                    "validation_specificity": operating.validation_specificity,
                    "threshold": operating.threshold,
                    **_source_provenance(
                        validation_predictions, prefix="validation"
                    ),
                    **_source_provenance(table, prefix="evaluation"),
                    "estimand_id": evidence_estimand_ids[series],
                    **_plan_provenance(analysis_plan),
                    **metrics,
                }
            )
    if not rows:
        raise ValueError("Fixed-sensitivity evidence requires evaluation predictions")
    return pd.DataFrame(rows)


def build_decision_curve_evidence(
    predictions: Mapping[str, AuthenticatedTable],
    *,
    analysis_plan: AnalysisPlanBinding,
    evidence_estimand_ids: Mapping[str, str],
    test_mode: bool = False,
) -> pd.DataFrame:
    _validate_series_plan(
        predictions,
        evidence_estimand_ids,
        analysis_plan,
        test_mode=test_mode,
    )
    thresholds = list(hst_reporting.REPORTING_CONTRACT["decision_thresholds"])
    rows: list[pd.DataFrame] = []
    for series, table in sorted(predictions.items()):
        frame = _validate_evaluation_frame(
            _verified_frame(table, require_confirmatory=not test_mode),
            name=f"Decision-curve series {series!r}",
        )
        if test_mode and not table.test_mode:
            raise ValueError("test_mode decision curves require explicit test-mode tables")
        for fold, group in _fold_groups(frame):
            curve = hst_reporting.build_decision_curve(group, thresholds=thresholds)
            curve.insert(0, "series", series)
            curve.insert(1, "fold", fold)
            for key, value in _source_provenance(table).items():
                curve[key] = value
            curve["estimand_id"] = evidence_estimand_ids[series]
            for key, value in _plan_provenance(analysis_plan).items():
                curve[key] = value
            rows.append(curve)
    if not rows:
        raise ValueError("Decision-curve evidence requires predictions")
    return pd.concat(rows, ignore_index=True)


def audit_aligned_comparator(
    hst_predictions: AuthenticatedTable,
    comparator_predictions: AuthenticatedTable,
) -> pd.DataFrame:
    left = _verified_frame(hst_predictions)
    right = _verified_frame(comparator_predictions)
    required = {"participant_key", "label_binary", "probability"}
    for name, frame in (("HST", left), ("comparator", right)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} prediction table missing columns: {missing}")
    if ("fold" in left) != ("fold" in right):
        raise ValueError("Aligned predictions must both contain fold or both omit it")
    keys = ["participant_key"] + (["fold"] if "fold" in left else [])
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("Aligned prediction identities must be unique")
    left = left.sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = right.sort_values(keys, kind="mergesort").reset_index(drop=True)
    identical_keys = left[keys].equals(right[keys])
    if not identical_keys:
        raise ValueError("Aligned comparator requires identical participant keys")
    identical_labels = left["label_binary"].astype("string").equals(
        right["label_binary"].astype("string")
    )
    if not identical_labels:
        raise ValueError("Aligned comparator predictions disagree on labels")
    context_columns = [
        column
        for column in ("dataset", "split", "protocol", "modality", "manifest_sha256")
        if column in left or column in right
    ]
    for column in context_columns:
        if column not in left or column not in right:
            raise ValueError(f"Aligned comparator context differs for {column}")
        if not left[column].astype("string").equals(right[column].astype("string")):
            raise ValueError(f"Aligned comparator context differs for {column}")
    return pd.DataFrame(
        [
            {
                "identical_participants": True,
                "identical_labels": True,
                "identical_context": True,
                "n_aligned_participants": int(left["participant_key"].nunique()),
                "n_aligned_rows": len(left),
                "fold_clustered": "fold" in left,
                **_source_provenance(hst_predictions, prefix="hst"),
                **_source_provenance(
                    comparator_predictions, prefix="comparator"
                ),
            }
        ]
    )


def _gpu_memory_measured_flags(frame: pd.DataFrame) -> pd.Series:
    if "gpu_memory_measured" not in frame:
        allocated = pd.to_numeric(
            frame.get("peak_gpu_memory_allocated_mb", frame.get("peak_gpu_memory_mb")),
            errors="coerce",
        )
        return (allocated.notna() & np.isfinite(allocated) & (allocated > 0)).astype(bool)

    def parse(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
            return bool(value)
        if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
            return value.strip().casefold() == "true"
        raise ValueError("Runtime gpu_memory_measured must contain booleans")

    return frame["gpu_memory_measured"].map(parse).astype(bool)


def _normalize_runtime_gpu_memory(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if not {
        "peak_gpu_memory_mb",
        "peak_gpu_memory_allocated_mb",
    } & set(result.columns):
        raise ValueError("Runtime table has no GPU-memory measurement column")
    fallback = pd.to_numeric(result.get("peak_gpu_memory_mb"), errors="coerce")
    allocated = pd.to_numeric(
        result.get("peak_gpu_memory_allocated_mb", fallback), errors="coerce"
    )
    reserved = (
        pd.to_numeric(result["peak_gpu_memory_reserved_mb"], errors="coerce")
        if "peak_gpu_memory_reserved_mb" in result
        else pd.Series(np.nan, index=result.index, dtype=float)
    )
    measured = _gpu_memory_measured_flags(result)
    invalid_measured = measured & (
        allocated.isna()
        | ~np.isfinite(allocated)
        | allocated.lt(0)
        | (
            reserved.notna()
            & (~np.isfinite(reserved) | reserved.lt(0))
        )
    )
    if invalid_measured.any():
        raise ValueError(
            "Measured GPU-memory rows require a finite, non-negative allocated peak "
            "and any supplied reserved peak must also be finite and non-negative"
        )
    result["gpu_memory_measured"] = measured
    result["peak_gpu_memory_allocated_mb"] = allocated.where(measured, np.nan)
    result["peak_gpu_memory_reserved_mb"] = reserved.where(measured, np.nan)
    result["peak_gpu_memory_mb"] = allocated.where(measured, np.nan)
    return result


def build_runtime_gpu_tables(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "run_id",
        "stage",
        "elapsed_seconds",
        "gpu_uuid",
        "status",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Runtime table missing columns: {missing}")
    if events.empty:
        raise ValueError("Runtime table is empty")
    result = _normalize_runtime_gpu_memory(events)
    result["elapsed_seconds"] = pd.to_numeric(
        result["elapsed_seconds"], errors="coerce"
    )
    if (
        result["elapsed_seconds"].isna().any()
        or (~np.isfinite(result["elapsed_seconds"])).any()
        or (result["elapsed_seconds"] < 0).any()
    ):
        raise ValueError("Runtime elapsed_seconds must be finite and non-negative")
    if result.duplicated(["run_id", "stage"]).any():
        raise ValueError("Runtime stages must be unique within a run")
    result = result.sort_values(["run_id", "stage"], kind="mergesort").reset_index(drop=True)
    summaries: list[dict[str, object]] = []
    for run_id, group in result.groupby("run_id", sort=True):
        measured_group = group.loc[group["gpu_memory_measured"]]
        summaries.append(
            {
                "run_id": run_id,
                "n_stages": len(group),
                "total_elapsed_seconds": float(group["elapsed_seconds"].sum()),
                "gpu_memory_measured_stages": len(measured_group),
                "peak_gpu_memory_allocated_mb": float(
                    measured_group["peak_gpu_memory_allocated_mb"].max()
                ),
                "peak_gpu_memory_reserved_mb": float(
                    measured_group["peak_gpu_memory_reserved_mb"].max()
                ),
                "peak_gpu_memory_mb": float(
                    measured_group["peak_gpu_memory_allocated_mb"].max()
                ),
                "gpu_uuids": ";".join(
                    sorted(
                        value
                        for value in set(measured_group["gpu_uuid"].astype(str))
                        if value.strip()
                    )
                ),
                "all_stages_successful": bool(group["status"].eq("success").all()),
            }
        )
    return result, pd.DataFrame(summaries)


_FIGURE_SPECS = MappingProxyType(
    {
        "branch_fusion_performance": "branch_fusion_performance",
        "paired_comparison": "hst_vs_aligned_comparator",
        "validation_ladder": "validation_ladder",
        "calibration": "calibration",
        "decision_curve": "decision_curve",
        "runtime_gpu": "runtime_gpu",
    }
)
_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def _require_columns(frame: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Figure source {name!r} missing columns: {missing}")


def _errorbars(frame: pd.DataFrame, value: str) -> np.ndarray | None:
    if not {"ci_low", "ci_high"} <= set(frame.columns):
        return None
    center = frame[value].to_numpy(dtype=float)
    low = frame["ci_low"].to_numpy(dtype=float)
    high = frame["ci_high"].to_numpy(dtype=float)
    if np.any(low > center) or np.any(high < center):
        raise ValueError("Figure confidence intervals do not contain point estimates")
    return np.vstack((center - low, high - center))


def _plot_branch_fusion(frame: pd.DataFrame):
    _require_columns(frame, {"label", "auroc", "kind"}, name="branch_fusion_performance")
    frame = frame.sort_values(["kind", "label"], kind="mergesort")
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    colors = [_COLORS[2] if value == "fusion" else _COLORS[0] for value in frame["kind"]]
    bars = ax.bar(frame["label"], frame["auroc"], color=colors, width=0.62)
    errors = _errorbars(frame, "auroc")
    if errors is not None:
        ax.errorbar(frame["label"], frame["auroc"], yerr=errors, fmt="none", color="#202020", capsize=4)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, min(1.02, max(1.0, float(frame["auroc"].max()) + 0.08)))
    ax.set_title("Branch and multimodal fusion performance")
    return fig


def _plot_paired(frame: pd.DataFrame):
    _require_columns(frame, {"label", "hst_auroc", "comparator_auroc"}, name="paired_comparison")
    frame = frame.sort_values("label", kind="mergesort")
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for index, row in enumerate(frame.itertuples(index=False)):
        ax.plot([row.comparator_auroc, row.hst_auroc], [index, index], color="#8A8A8A", linewidth=1.5)
    ax.scatter(frame["comparator_auroc"], y, color=_COLORS[1], label="Aligned comparator", zorder=3)
    ax.scatter(frame["hst_auroc"], y, color=_COLORS[0], label="HST", zorder=3)
    ax.set_yticks(y, frame["label"])
    ax.set_xlabel("AUROC")
    ax.set_xlim(0.45, 1.0)
    ax.set_title("Paired HST versus aligned comparator")
    ax.legend(frameon=False, ncols=2, loc="lower right")
    return fig


def _plot_ladder(frame: pd.DataFrame):
    _require_columns(frame, {"stage", "auroc"}, name="validation_ladder")
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    x = np.arange(len(frame))
    ax.errorbar(
        x,
        frame["auroc"],
        yerr=_errorbars(frame, "auroc"),
        marker="o",
        markersize=7,
        linewidth=2.2,
        capsize=4,
        color=_COLORS[0],
    )
    ax.set_xticks(x, frame["stage"])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Validation ladder under increasing distribution shift")
    return fig


def _plot_calibration(frame: pd.DataFrame):
    _require_columns(
        frame,
        {"series", "mean_probability", "observed_prevalence"},
        name="calibration",
    )
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", label="Ideal")
    for index, (series, group) in enumerate(frame.groupby("series", sort=True)):
        ordered = group.sort_values("mean_probability")
        ax.plot(
            ordered["mean_probability"],
            ordered["observed_prevalence"],
            marker="o",
            linewidth=2,
            color=_COLORS[index % len(_COLORS)],
            label=str(series),
        )
    ax.set(xlabel="Mean predicted probability", ylabel="Observed prevalence", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Calibration across validation settings")
    ax.legend(frameon=False)
    return fig


def _plot_dca(frame: pd.DataFrame):
    _require_columns(
        frame,
        {"series", "threshold", "model_net_benefit", "treat_all_net_benefit", "treat_none_net_benefit"},
        name="decision_curve",
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for index, (series, group) in enumerate(frame.groupby("series", sort=True)):
        ordered = group.sort_values("threshold")
        ax.plot(
            ordered["threshold"],
            ordered["model_net_benefit"],
            linewidth=2.2,
            color=_COLORS[index % len(_COLORS)],
            label=str(series),
        )
    reference = frame.sort_values("threshold").drop_duplicates("threshold")
    ax.plot(reference["threshold"], reference["treat_all_net_benefit"], linestyle="--", color="#666666", label="Treat all")
    ax.plot(reference["threshold"], reference["treat_none_net_benefit"], linestyle=":", color="#111111", label="Treat none")
    ax.axhline(0.0, color="#BBBBBB", linewidth=0.8)
    ax.set(xlabel="Threshold probability", ylabel="Net benefit")
    ax.set_title("Decision-curve analysis")
    ax.legend(frameon=False, ncols=2)
    return fig


def _plot_runtime(frame: pd.DataFrame):
    _require_columns(frame, {"stage", "elapsed_seconds"}, name="runtime_gpu")
    frame = _normalize_runtime_gpu_memory(frame)
    frame = frame.sort_values("stage", kind="mergesort").reset_index(drop=True)
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.bar(x, frame["elapsed_seconds"] / 60.0, color=_COLORS[0], alpha=0.85, label="Runtime")
    ax.set_xticks(x, frame["stage"])
    ax.set_ylabel("Runtime (minutes)", color=_COLORS[0])
    ax.tick_params(axis="y", labelcolor=_COLORS[0])
    memory = ax.twinx()
    measured = frame["gpu_memory_measured"].to_numpy(dtype=bool)
    if measured.any():
        memory.plot(
            x[measured],
            frame.loc[measured, "peak_gpu_memory_allocated_mb"] / 1024.0,
            marker="o",
            color=_COLORS[1],
            linewidth=2,
            label="Peak allocated",
        )
        reserved_measured = measured & frame[
            "peak_gpu_memory_reserved_mb"
        ].notna().to_numpy(dtype=bool)
        if reserved_measured.any():
            memory.plot(
                x[reserved_measured],
                frame.loc[
                    reserved_measured, "peak_gpu_memory_reserved_mb"
                ] / 1024.0,
                marker="s",
                color=_COLORS[2],
                linewidth=2,
                label="Peak reserved",
            )
        memory.legend(frameon=False, loc="upper right")
    memory.set_ylabel("Peak GPU memory (GiB; measured stages only)", color=_COLORS[1])
    memory.tick_params(axis="y", labelcolor=_COLORS[1])
    ax.set_title("Stage runtime and measured GPU-memory profile")
    return fig


_PLOTTERS = {
    "branch_fusion_performance": _plot_branch_fusion,
    "paired_comparison": _plot_paired,
    "validation_ladder": _plot_ladder,
    "calibration": _plot_calibration,
    "decision_curve": _plot_dca,
    "runtime_gpu": _plot_runtime,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_publication_figures(
    output_dir: Path,
    source_tables: Mapping[str, AuthenticatedTable],
    *,
    analysis_plan: AnalysisPlanBinding,
    test_mode: bool = False,
) -> pd.DataFrame:
    _require_plan_estimand(
        analysis_plan,
        PRIMARY_ESTIMAND_ID,
        test_mode=test_mode,
    )
    missing = sorted(set(_FIGURE_SPECS) - set(source_tables))
    extra = sorted(set(source_tables) - set(_FIGURE_SPECS))
    if missing or extra:
        raise ValueError(f"Figure source tables differ from contract; missing={missing}, extra={extra}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.22,
        "svg.hashsalt": "covid-rars-hst-publication-v1",
    }
    with mpl.rc_context(style):
        for source_name, figure_id in _FIGURE_SPECS.items():
            table = source_tables[source_name]
            frame = _verified_frame(table, require_confirmatory=not test_mode)
            if test_mode and not table.test_mode:
                raise ValueError("test_mode figures require explicit test-mode source tables")
            figure = _PLOTTERS[source_name](frame)
            for suffix in ("svg", "png"):
                path = output_dir / f"{figure_id}.{suffix}"
                if suffix == "svg":
                    figure.savefig(
                        path,
                        format="svg",
                        metadata={"Date": None, "Creator": "COVID-RARS publication pipeline"},
                    )
                else:
                    figure.savefig(
                        path,
                        format="png",
                        dpi=180,
                        metadata={"Software": "COVID-RARS publication pipeline"},
                    )
                rows.append(
                    {
                        "figure_id": figure_id,
                        "format": suffix,
                        "path": str(path.resolve()),
                        "sha256": _sha256_file(path),
                        "size_bytes": path.stat().st_size,
                        "source_table_name": table.source_name,
                        **_source_provenance(table),
                        **_plan_provenance(analysis_plan),
                    }
                )
            plt.close(figure)
    manifest = pd.DataFrame(rows).sort_values(
        ["figure_id", "format"], kind="mergesort"
    ).reset_index(drop=True)
    logical_manifest = manifest.drop(columns="path").copy()
    logical_manifest.insert(
        2,
        "filename",
        [Path(value).name for value in manifest["path"]],
    )
    manifest["figure_manifest_sha256"] = dataframe_sha256(logical_manifest)
    return manifest
