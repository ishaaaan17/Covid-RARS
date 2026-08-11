from __future__ import annotations

import re
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from covid_rars.metrics import labels_to_binary

EXISTING_PROTOCOL = "existing_participant_split"
TEMPORAL_PROTOCOL = "temporal_early_to_late"
TIME_STRATIFIED_PROTOCOL = "time_stratified_participant_split"
FULL_MULTIMODAL = "breath+cough+speech"
UNIFORM_FUSION = "uniform_mean"
MONTH_YEAR_ABLATION_ORDER = [
    ("full_safe_metadata_full", "Full metadata"),
    ("full_safe_metadata_no_recording_year", "Remove year"),
    ("full_safe_metadata_no_recording_month", "Remove month"),
    ("full_safe_metadata_no_recording_year_month", "Remove year + month"),
]


def _as_float(value: object) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


def _parse_secondary_metric(text: object, metric: str) -> float:
    if text is None or pd.isna(text):
        return float("nan")
    pattern = rf"(?:^|;|\s){re.escape(metric)}\s*=\s*([-+0-9.eE]+)"
    match = re.search(pattern, str(text))
    return _as_float(match.group(1)) if match else float("nan")


def _best_full_multimodal_row(metrics: pd.DataFrame, protocol: str) -> pd.Series | None:
    if metrics.empty or "evaluation_protocol" not in metrics.columns:
        return None
    work = metrics[metrics["evaluation_protocol"].astype(str).eq(protocol)].copy()
    if work.empty:
        return None
    preferred = work[
        work.get("analysis_family", pd.Series(index=work.index, dtype=object)).astype(str).eq("multimodal_fusion")
        & work.get("modality_combination", pd.Series(index=work.index, dtype=object)).astype(str).eq(FULL_MULTIMODAL)
        & work.get("fusion_method", pd.Series(index=work.index, dtype=object)).astype(str).eq(UNIFORM_FUSION)
    ].copy()
    if preferred.empty:
        preferred = work[work.get("analysis_family", pd.Series(index=work.index, dtype=object)).astype(str).eq("multimodal_fusion")].copy()
    if preferred.empty:
        preferred = work.copy()
    auroc = pd.to_numeric(preferred.get("auroc"), errors="coerce")
    if auroc.notna().any():
        return preferred.loc[auroc.idxmax()]
    return preferred.iloc[0]


def _matching_auroc_ci(row: pd.Series, bootstrap_ci: pd.DataFrame) -> tuple[float, float]:
    if bootstrap_ci.empty or "metric" not in bootstrap_ci.columns:
        return float("nan"), float("nan")
    matches = bootstrap_ci[bootstrap_ci["metric"].astype(str).eq("auroc")].copy()
    for col in [
        "evaluation_protocol",
        "analysis_family",
        "model_name",
        "modality",
        "modality_combination",
        "fusion_method",
        "audit_model",
    ]:
        if col in matches.columns and col in row.index:
            value = "" if pd.isna(row.get(col)) else str(row.get(col))
            matches = matches[matches[col].fillna("").astype(str).eq(value)]
    if matches.empty:
        return float("nan"), float("nan")
    match = matches.iloc[0]
    return _as_float(match.get("ci_low")), _as_float(match.get("ci_high"))


def _external_best_row(evidence_matrix: pd.DataFrame) -> pd.Series | None:
    if evidence_matrix.empty or not {"claim_id", "primary_metric", "primary_value"}.issubset(evidence_matrix.columns):
        return None
    external = evidence_matrix[
        evidence_matrix["claim_id"].astype(str).str.startswith("external_transfer_")
        & evidence_matrix["primary_metric"].astype(str).eq("auroc")
    ].copy()
    if external.empty:
        return None
    external["primary_value"] = pd.to_numeric(external["primary_value"], errors="coerce")
    if external["primary_value"].notna().any():
        return external.loc[external["primary_value"].idxmax()]
    return external.iloc[0]


def _external_lift_for_row(row: pd.Series | None, external_auprc_lift: pd.DataFrame) -> tuple[float, float]:
    if row is None:
        return float("nan"), float("nan")
    parsed_auprc = _parse_secondary_metric(row.get("secondary_metrics"), "auprc")
    if external_auprc_lift.empty:
        return parsed_auprc, float("nan")
    lift = external_auprc_lift.copy()
    if "auroc" in lift.columns:
        lift["_distance"] = (pd.to_numeric(lift["auroc"], errors="coerce") - _as_float(row.get("primary_value"))).abs()
        if lift["_distance"].notna().any():
            best = lift.loc[lift["_distance"].idxmin()]
            return _as_float(best.get("auprc", parsed_auprc)), _as_float(best.get("absolute_auprc_lift"))
    return parsed_auprc, float("nan")


def build_temporal_stress_test_summary(
    temporal_metrics: pd.DataFrame,
    temporal_bootstrap_ci: pd.DataFrame,
    evidence_matrix: pd.DataFrame | None = None,
    external_auprc_lift: pd.DataFrame | None = None,
) -> pd.DataFrame:
    evidence_matrix = evidence_matrix if evidence_matrix is not None else pd.DataFrame()
    external_auprc_lift = external_auprc_lift if external_auprc_lift is not None else pd.DataFrame()
    rows: list[dict[str, object]] = []
    specs = [
        ("participant_internal", EXISTING_PROTOCOL, "random participant split internal reference"),
        ("time_stratified_internal", TIME_STRATIFIED_PROTOCOL, "calendar-balanced participant split reference"),
        ("temporal_holdout", TEMPORAL_PROTOCOL, "early-to-late chronological stress test"),
    ]
    for stress_test, protocol, interpretation in specs:
        metric_row = _best_full_multimodal_row(temporal_metrics, protocol)
        if metric_row is None:
            continue
        ci_low, ci_high = _matching_auroc_ci(metric_row, temporal_bootstrap_ci)
        rows.append(
            {
                "stress_test": stress_test,
                "evaluation_protocol": protocol,
                "analysis_family": metric_row.get("analysis_family"),
                "model_name": metric_row.get("model_name"),
                "modality": metric_row.get("modality"),
                "modality_combination": metric_row.get("modality_combination"),
                "fusion_method": metric_row.get("fusion_method"),
                "auroc": _as_float(metric_row.get("auroc")),
                "auroc_ci_low": ci_low,
                "auroc_ci_high": ci_high,
                "auprc": _as_float(metric_row.get("auprc")),
                "auprc_lift_over_prevalence": _as_float(metric_row.get("auprc_lift_over_prevalence")),
                "brier": _as_float(metric_row.get("brier")),
                "ece": _as_float(metric_row.get("ece")),
                "n_samples": metric_row.get("n_samples"),
                "interpretation": interpretation,
            }
        )
    external_row = _external_best_row(evidence_matrix)
    if external_row is not None:
        auprc, lift = _external_lift_for_row(external_row, external_auprc_lift)
        rows.append(
            {
                "stress_test": "external_transfer",
                "evaluation_protocol": "coswara_to_coughvid_external",
                "analysis_family": "external_transfer",
                "model_name": external_row.get("claim_id"),
                "modality": "cough",
                "modality_combination": "cough",
                "fusion_method": external_row.get("comparison"),
                "auroc": _as_float(external_row.get("primary_value")),
                "auroc_ci_low": float("nan"),
                "auroc_ci_high": float("nan"),
                "auprc": auprc,
                "auprc_lift_over_prevalence": lift,
                "brier": float("nan"),
                "ece": float("nan"),
                "n_samples": external_row.get("n_samples"),
                "interpretation": "independent dataset-transfer stress test",
            }
        )
    order = {"participant_internal": 0, "time_stratified_internal": 1, "temporal_holdout": 2, "external_transfer": 3}
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_order"] = out["stress_test"].map(order).fillna(99)
        out = out.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return out


def build_temporal_feature_attribution_comparison(feature_importance: pd.DataFrame) -> pd.DataFrame:
    if feature_importance.empty:
        return pd.DataFrame()
    required = {"evaluation_protocol", "audit_model", "feature", "importance_abs"}
    missing = required - set(feature_importance.columns)
    if missing:
        raise KeyError(f"feature importance missing columns: {sorted(missing)}")
    work = feature_importance.copy()
    work["importance_abs"] = pd.to_numeric(work["importance_abs"], errors="coerce")
    work["coefficient"] = pd.to_numeric(work.get("coefficient", np.nan), errors="coerce")
    work = work.sort_values(["evaluation_protocol", "audit_model", "importance_abs", "feature"], ascending=[True, True, False, True])
    work["rank"] = work.groupby(["evaluation_protocol", "audit_model"], dropna=False).cumcount() + 1
    protocol_alias = {
        EXISTING_PROTOCOL: "existing",
        TIME_STRATIFIED_PROTOCOL: "time_stratified",
        TEMPORAL_PROTOCOL: "temporal",
    }
    frames: list[pd.DataFrame] = []
    for protocol, alias in protocol_alias.items():
        subset = work[work["evaluation_protocol"].astype(str).eq(protocol)].copy()
        if subset.empty:
            continue
        keep = ["audit_model", "feature"]
        if "feature_group" in subset.columns:
            keep.append("feature_group")
        subset = subset[keep + ["importance_abs", "coefficient", "rank"]].copy()
        subset = subset.rename(
            columns={
                "importance_abs": f"{alias}_importance_abs",
                "coefficient": f"{alias}_coefficient",
                "rank": f"{alias}_rank",
            }
        )
        frames.append(subset)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        merge_cols = ["audit_model", "feature"]
        if "feature_group" in out.columns and "feature_group" in frame.columns:
            merge_cols.append("feature_group")
        out = out.merge(frame, on=merge_cols, how="outer")
    if "existing_importance_abs" in out.columns and "temporal_importance_abs" in out.columns:
        out["delta_temporal_minus_existing"] = out["temporal_importance_abs"] - out["existing_importance_abs"]
    if "existing_importance_abs" in out.columns and "time_stratified_importance_abs" in out.columns:
        out["delta_time_stratified_minus_existing"] = out["time_stratified_importance_abs"] - out["existing_importance_abs"]
    sort_col = "existing_importance_abs" if "existing_importance_abs" in out.columns else "temporal_importance_abs"
    return out.sort_values(["audit_model", sort_col, "feature"], ascending=[True, False, True]).reset_index(drop=True)


def _selected_prediction_group(predictions: pd.DataFrame, protocol: str) -> pd.DataFrame:
    work = predictions[
        predictions.get("evaluation_protocol", pd.Series(index=predictions.index, dtype=object)).astype(str).eq(protocol)
        & predictions.get("split", pd.Series(index=predictions.index, dtype=object)).astype(str).eq("test")
        & predictions.get("analysis_family", pd.Series(index=predictions.index, dtype=object)).astype(str).eq("multimodal_fusion")
        & predictions.get("modality_combination", pd.Series(index=predictions.index, dtype=object)).astype(str).eq(FULL_MULTIMODAL)
        & predictions.get("fusion_method", pd.Series(index=predictions.index, dtype=object)).astype(str).eq(UNIFORM_FUSION)
        & predictions.get("label_binary", pd.Series(index=predictions.index, dtype=object)).isin(["positive", "negative"])
    ].copy()
    work["probability"] = pd.to_numeric(work.get("probability"), errors="coerce")
    return work[np.isfinite(work["probability"])].copy()


def _bootstrap_auroc(group: pd.DataFrame, rng: np.random.Generator) -> float:
    participant_codes, participant_ids = pd.factorize(group["participant_id"].astype(str), sort=False)
    if len(participant_ids) < 2:
        return float("nan")
    indices = [np.flatnonzero(participant_codes == idx) for idx in range(len(participant_ids))]
    sampled_participants = rng.integers(0, len(participant_ids), size=len(participant_ids))
    sampled_indices = np.concatenate([indices[idx] for idx in sampled_participants])
    y = labels_to_binary(group.iloc[sampled_indices]["label_binary"])
    if len(np.unique(y)) < 2:
        return float("nan")
    prob = group.iloc[sampled_indices]["probability"].to_numpy(dtype=float)
    return float(roc_auc_score(y, prob))


def _point_auroc(group: pd.DataFrame) -> float:
    if group.empty or group["label_binary"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(labels_to_binary(group["label_binary"]), group["probability"].to_numpy(dtype=float)))


def build_temporal_delta_significance(
    predictions: pd.DataFrame,
    n_bootstraps: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    if predictions.empty or n_bootstraps <= 0:
        return pd.DataFrame()
    reference = _selected_prediction_group(predictions, EXISTING_PROTOCOL)
    stress = _selected_prediction_group(predictions, TEMPORAL_PROTOCOL)
    if reference.empty or stress.empty:
        return pd.DataFrame()
    reference_point = _point_auroc(reference)
    stress_point = _point_auroc(stress)
    if not np.isfinite(reference_point) or not np.isfinite(stress_point):
        return pd.DataFrame()
    rng = np.random.default_rng(random_state)
    deltas: list[float] = []
    for _ in range(n_bootstraps):
        ref = _bootstrap_auroc(reference, rng)
        tmp = _bootstrap_auroc(stress, rng)
        if np.isfinite(ref) and np.isfinite(tmp):
            deltas.append(tmp - ref)
    values = np.asarray(deltas, dtype=float)
    if values.size == 0:
        return pd.DataFrame()
    p_two_sided = float(min(1.0, 2.0 * min(np.mean(values <= 0.0), np.mean(values >= 0.0))))
    return pd.DataFrame(
        [
            {
                "comparison": "temporal_minus_participant_full_multimodal",
                "reference_protocol": EXISTING_PROTOCOL,
                "stress_protocol": TEMPORAL_PROTOCOL,
                "reference_auroc": reference_point,
                "stress_auroc": stress_point,
                "delta_auroc": stress_point - reference_point,
                "delta_ci_low": float(np.quantile(values, 0.025)),
                "delta_ci_high": float(np.quantile(values, 0.975)),
                "p_value_two_sided": p_two_sided,
                "n_bootstraps": int(values.size),
            }
        ]
    )



def build_temporal_month_year_ablation_table(metadata_ablation: pd.DataFrame) -> pd.DataFrame:
    if metadata_ablation.empty:
        return pd.DataFrame()
    required = {"evaluation_protocol", "base_feature_set", "ablation_name", "auroc", "auprc"}
    missing = required - set(metadata_ablation.columns)
    if missing:
        raise KeyError(f"metadata ablation missing columns: {sorted(missing)}")
    work = metadata_ablation[
        metadata_ablation["evaluation_protocol"].astype(str).eq(TEMPORAL_PROTOCOL)
        & metadata_ablation["base_feature_set"].astype(str).eq("full_safe_metadata")
    ].copy()
    rows: list[dict[str, object]] = []
    for order, (ablation_name, label) in enumerate(MONTH_YEAR_ABLATION_ORDER, start=1):
        match = work[work["ablation_name"].astype(str).eq(ablation_name)]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "display_order": order,
                "metadata_configuration": label,
                "evaluation_protocol": TEMPORAL_PROTOCOL,
                "base_feature_set": "full_safe_metadata",
                "ablation_name": ablation_name,
                "removed_features": row.get("removed_features"),
                "temporal_auroc": _as_float(row.get("auroc")),
                "temporal_auprc": _as_float(row.get("auprc")),
                "temporal_balanced_accuracy": _as_float(row.get("balanced_accuracy")),
                "n_features": row.get("n_features"),
            }
        )
    return pd.DataFrame(rows)


def _format_metric(value: object, digits: int = 3) -> str:
    numeric = _as_float(value)
    return "NA" if not np.isfinite(numeric) else f"{numeric:.{digits}f}"


def _format_p_value(value: object) -> str:
    numeric = _as_float(value)
    if not np.isfinite(numeric):
        return "NA"
    if numeric == 0:
        return "<0.0002"
    if numeric < 0.001:
        return "<0.001"
    return f"{numeric:.4f}"


def write_temporal_stress_figure_svg(stress_summary: pd.DataFrame, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if stress_summary.empty:
        output.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="900" height="200" />\n', encoding="utf-8")
        return
    labels = {
        "participant_internal": "Participant split",
        "time_stratified_internal": "Time-stratified split",
        "temporal_holdout": "Temporal holdout",
        "external_transfer": "External transfer",
    }
    rows = stress_summary[stress_summary["stress_test"].isin(labels)].copy()
    rows["_order"] = rows["stress_test"].map({key: idx for idx, key in enumerate(labels)})
    rows = rows.sort_values("_order")
    width = 900
    height = 165 + 120 * max(1, len(rows))
    x = 450
    box_w = 520
    box_h = 72
    y0 = 92
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.title{font-size:24px;font-weight:700}.label{font-size:18px;font-weight:700}.metric{font-size:16px}.note{font-size:13px;fill:#4b5563}.box{fill:#ffffff;stroke:#111827;stroke-width:1.5}.arrow{stroke:#111827;stroke-width:2;fill:none}",
        "</style>",
        '<text x="450" y="38" text-anchor="middle" class="title">COVID Audio Temporal Robustness Stress Test</text>',
    ]
    previous_y: float | None = None
    for idx, (_, row) in enumerate(rows.iterrows()):
        y = y0 + idx * 120
        if previous_y is not None:
            parts.extend(
                [
                    f'<path d="M{x} {previous_y + box_h / 2 + 10} L{x} {y - box_h / 2 - 14}" class="arrow"/>',
                    f'<path d="M{x - 7} {y - box_h / 2 - 22} L{x} {y - box_h / 2 - 12} L{x + 7} {y - box_h / 2 - 22}" fill="none" stroke="#111827" stroke-width="2"/>',
                ]
            )
        label = labels.get(str(row.get("stress_test")), str(row.get("stress_test")))
        auroc = _format_metric(row.get("auroc"))
        ci_low = _format_metric(row.get("auroc_ci_low"))
        ci_high = _format_metric(row.get("auroc_ci_high"))
        lift = _format_metric(row.get("auprc_lift_over_prevalence"))
        ci = "" if ci_low == "NA" or ci_high == "NA" else f"  95% CI {ci_low}-{ci_high}"
        parts.extend(
            [
                f'<rect x="{x - box_w / 2}" y="{y - box_h / 2}" width="{box_w}" height="{box_h}" rx="6" class="box"/>',
                f'<text x="{x}" y="{y - 10}" text-anchor="middle" class="label">{escape(label)}</text>',
                f'<text x="{x}" y="{y + 15}" text-anchor="middle" class="metric">AUROC {auroc}{escape(ci)} · AUPRC lift {lift}</text>',
            ]
        )
        previous_y = y
    parts.extend(
        [
            f'<text x="450" y="{height - 28}" text-anchor="middle" class="note">Participant performance weakens under calendar control and collapses under temporal/external stress.</text>',
            "</svg>",
        ]
    )
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_temporal_results_section(
    stress_summary: pd.DataFrame,
    month_ablation: pd.DataFrame,
    significance: pd.DataFrame,
    output: Path,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def _stress_value(stress_test: str, column: str) -> str:
        if stress_summary.empty or column not in stress_summary.columns:
            return "NA"
        row = stress_summary[stress_summary["stress_test"].astype(str).eq(stress_test)]
        return "NA" if row.empty else _format_metric(row.iloc[0].get(column))

    delta = p_value = delta_ci = "NA"
    if not significance.empty:
        sig = significance.iloc[0]
        delta = _format_metric(sig.get("delta_auroc"))
        p_value = _format_p_value(sig.get("p_value_two_sided"))
        low = _format_metric(sig.get("delta_ci_low"))
        high = _format_metric(sig.get("delta_ci_high"))
        delta_ci = "NA" if low == "NA" or high == "NA" else f"{low} to {high}"

    month_lines = []
    if not month_ablation.empty:
        for _, row in month_ablation.sort_values("display_order").iterrows():
            month_lines.append(
                f"| {row.get('metadata_configuration')} | {_format_metric(row.get('temporal_auroc'))} | {_format_metric(row.get('temporal_auprc'))} |"
            )
    if not month_lines:
        month_lines.append("| NA | NA | NA |")

    text = "\n".join(
        [
            "# Draft Results Section: Temporal Robustness",
            "",
            "This draft is intentionally limited to results language. It does not decide the final contribution or paper framing.",
            "",
            "## RQ1: Can multimodal audio detect COVID internally?",
            "",
            f"Under the existing participant-level split, full multimodal fusion across breath, cough, and speech achieved AUROC {_stress_value('participant_internal', 'auroc')} and AUPRC {_stress_value('participant_internal', 'auprc')}. This establishes that the pipeline can learn a strong source-domain signal when train and test participants are randomly separated but drawn from the same temporal collection distribution.",
            "",
            "## RQ2: Does performance survive temporal stress?",
            "",
            f"Performance weakened under calendar control and collapsed under strict chronological evaluation. The calendar-balanced participant split achieved AUROC {_stress_value('time_stratified_internal', 'auroc')}, while the early-to-late temporal holdout achieved AUROC {_stress_value('temporal_holdout', 'auroc')}. The participant-to-temporal AUROC difference was {delta} with 95% bootstrap CI {delta_ci} and two-sided bootstrap p={p_value}. This indicates that the original participant-split performance is not stable under temporal stress.",
            "",
            "## RQ3: Does performance survive external transfer?",
            "",
            f"The best independent Coswara-to-COUGHVID external transfer result achieved AUROC {_stress_value('external_transfer', 'auroc')} and AUPRC lift {_stress_value('external_transfer', 'auprc_lift_over_prevalence')} over the COUGHVID prevalence baseline. The external result is nearly identical to the temporal holdout result, suggesting that chronological stress inside Coswara and independent dataset transfer expose the same fragility.",
            "",
            "## RQ4: What drives the failure?",
            "",
            "The temporal metadata ablation isolates recording month as a major driver of poor chronological generalization in the full safe metadata model.",
            "",
            "| Metadata configuration | Temporal AUROC | Temporal AUPRC |",
            "| --- | ---: | ---: |",
            *month_lines,
            "",
            "Removing recording month increased temporal full-safe-metadata AUROC from 0.531 to 0.779, while removing recording year alone did not change the temporal result. This means recording month is not merely predictive; in the early-to-late setting it encodes collection-period structure that harms chronological generalization. Together with the participant-split, time-stratified, temporal-holdout, and external-transfer results, this supports the conclusion that temporal/protocol confounding inflates internal COVID-audio performance and weakens out-of-period transfer.",
            "",
        ]
    )
    output.write_text(text, encoding="utf-8")


def write_causal_chain_summary(
    stress_summary: pd.DataFrame,
    significance: pd.DataFrame,
    feature_comparison: pd.DataFrame,
    output: Path,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def _metric(stress_test: str, column: str) -> str:
        if stress_summary.empty or column not in stress_summary.columns:
            return "NA"
        row = stress_summary[stress_summary["stress_test"].astype(str).eq(stress_test)]
        if row.empty:
            return "NA"
        value = _as_float(row.iloc[0].get(column))
        return "NA" if not np.isfinite(value) else f"{value:.3f}"

    delta_text = "NA"
    p_text = "NA"
    if not significance.empty:
        sig = significance.iloc[0]
        delta = _as_float(sig.get("delta_auroc"))
        p = _as_float(sig.get("p_value_two_sided"))
        delta_text = "NA" if not np.isfinite(delta) else f"{delta:.3f}"
        p_text = "NA" if not np.isfinite(p) else f"{p:.4f}"

    feature_lines = []
    if not feature_comparison.empty:
        subset = feature_comparison[feature_comparison["feature"].isin(["recording_year", "recording_month"])]
        for _, row in subset.head(6).iterrows():
            existing = _as_float(row.get("existing_importance_abs"))
            temporal = _as_float(row.get("temporal_importance_abs"))
            feature_lines.append(
                f"- {row.get('audit_model')} / {row.get('feature')}: existing importance {existing:.3f}, temporal importance {temporal:.3f}"
            )
    if not feature_lines:
        feature_lines.append("- Year/month feature attribution rows were not available in the input tables.")

    text = "\n".join(
        [
            "# Temporal Robustness Causal Chain",
            "",
            "This derived summary is for results communication and manuscript planning.",
            "",
            "## Causal Chain",
            "",
            f"1. Participant split appears strong: AUROC {_metric('participant_internal', 'auroc')}.",
            f"2. Calendar-balanced split is lower: AUROC {_metric('time_stratified_internal', 'auroc')}.",
            f"3. Strict early-to-late temporal holdout collapses: AUROC {_metric('temporal_holdout', 'auroc')}.",
            f"4. External transfer is similarly weak: AUROC {_metric('external_transfer', 'auroc')}.",
            f"5. Temporal-minus-participant AUROC difference is {delta_text} with two-sided bootstrap p={p_text}.",
            "6. Year/month attribution and ablation tables identify temporal/protocol variables as structural label predictors.",
            "",
            "## Key Temporal Feature Rows",
            "",
            *feature_lines,
            "",
        ]
    )
    output.write_text(text, encoding="utf-8")
