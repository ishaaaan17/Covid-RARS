#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile


BUNDLE_NAME = "BTP_Research_Closure_Bundle_2026-06-15.zip"


BUNDLE_ARTIFACTS = [
    Path("reports/final/COVID_RARS_PHASED_RESULTS_BRIEF_2026-06-15.md"),
    Path("reports/final/COVID_RARS_PUBLICATION_RESULTS_REPORT.md"),
    Path("reports/final/COVID_RARS_PUBLICATION_RESULTS_SUMMARY.md"),
    Path("reports/final/MANUSCRIPT_SUPPORT_ANALYSES.md"),
    Path("reports/experiment_manifest.json"),
    Path("reports/tables/publication_evidence_matrix.csv"),
    Path("reports/tables/publication_evidence_matrix.md"),
    Path("reports/tables/related_paper_comparison.csv"),
    Path("reports/tables/related_paper_comparison.md"),
    Path("reports/tables/paper_metric_table.csv"),
    Path("reports/tables/paper_metric_table_raw.csv"),
    Path("reports/tables/clinical_operating_points.csv"),
    Path("reports/tables/calibration_under_shift_summary.csv"),
    Path("reports/tables/calibration_under_shift_bins.csv"),
    Path("reports/tables/metadata_confounding_feature_importance.csv"),
    Path("reports/tables/metadata_confounding_group_summary.csv"),
    Path("reports/tables/confounding_controlled_balance.csv"),
    Path("reports/tables/ipw_sensitivity_balance.csv"),
    Path("reports/tables/domain_shift_feature_importance.csv"),
    Path("reports/tables/domain_adaptation_mmd.csv"),
    Path("reports/tables/external_prevalence_recalibration.csv"),
    Path("reports/tables/paired_bootstrap_comparisons.csv"),
    Path("reports/tables/manuscript_demographic_protocol_linear_shap.csv"),
    Path("reports/tables/manuscript_ipw_residual_smd.csv"),
    Path("reports/tables/manuscript_external_auprc_lift.csv"),
    Path("reports/tables/manuscript_unknown_label_summary.csv"),
    Path("reports/tables/manuscript_unknown_label_balance.csv"),
    Path("data/outputs/metrics/quality_weighted_fusion_metrics.csv"),
    Path("data/outputs/metrics/quality_weighted_fusion_bootstrap_ci.csv"),
    Path("data/outputs/metrics/external_model_grid_metrics.csv"),
    Path("data/outputs/metrics/external_model_grid_bootstrap_ci.csv"),
    Path("data/outputs/metrics/external_model_grid_opensmile_egemaps_metrics.csv"),
    Path("data/outputs/metrics/external_model_grid_opensmile_egemaps_bootstrap_ci.csv"),
    Path("data/outputs/metrics/external_model_grid_beats_metrics.csv"),
    Path("data/outputs/metrics/external_model_grid_beats_bootstrap_ci.csv"),
    Path("data/outputs/metrics/external_model_grid_panns_metrics.csv"),
    Path("data/outputs/metrics/external_model_grid_panns_bootstrap_ci.csv"),
    Path("data/outputs/metrics/coughvid_internal_metrics.csv"),
    Path("data/outputs/metrics/coughvid_internal_bootstrap_ci.csv"),
    Path("data/outputs/metrics/coughvid_internal_opensmile_egemaps_metrics.csv"),
    Path("data/outputs/metrics/coughvid_internal_opensmile_egemaps_bootstrap_ci.csv"),
    Path("data/outputs/metrics/coughvid_internal_beats_metrics.csv"),
    Path("data/outputs/metrics/coughvid_internal_beats_bootstrap_ci.csv"),
    Path("data/outputs/metrics/coughvid_internal_panns_metrics.csv"),
    Path("data/outputs/metrics/coughvid_internal_panns_bootstrap_ci.csv"),
    Path("data/outputs/metrics/metadata_confounding_metrics.csv"),
    Path("data/outputs/metrics/confounding_controlled_audio_metrics.csv"),
    Path("data/outputs/metrics/confounding_controlled_audio_bootstrap_ci.csv"),
    Path("data/outputs/metrics/domain_shift_audit_metrics.csv"),
    Path("data/outputs/metrics/ipw_sensitivity_metrics.csv"),
    Path("data/outputs/metrics/domain_adaptation_baseline_metrics.csv"),
]

BUNDLE_ARTIFACTS = [PurePosixPath(path.as_posix()) for path in BUNDLE_ARTIFACTS]


def default_bundle_artifacts() -> list[Path]:
    return list(BUNDLE_ARTIFACTS)


def existing_artifacts(project_root: Path, artifacts: list[Path]) -> tuple[list[Path], list[Path]]:
    available: list[Path] = []
    missing: list[Path] = []
    for rel_path in artifacts:
        full_path = project_root / rel_path
        if full_path.exists():
            available.append(rel_path)
        else:
            missing.append(rel_path)
    return available, missing


def write_bundle(project_root: Path, output: Path, artifacts: list[Path] | None = None) -> tuple[list[Path], list[Path]]:
    artifacts = artifacts or default_bundle_artifacts()
    available, missing = existing_artifacts(project_root, artifacts)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as bundle:
        manifest_lines = [
            "Covid-RARS Research Closure Bundle 2026-06-15",
            "",
            "Included artifacts:",
        ]
        for rel_path in available:
            bundle.write(project_root / rel_path, arcname=str(rel_path))
            manifest_lines.append(f"- {rel_path}")
        manifest_lines.extend(["", "Missing artifacts:"])
        if missing:
            manifest_lines.extend(f"- {rel_path}" for rel_path in missing)
        else:
            manifest_lines.append("- none")
        bundle.writestr("MANIFEST.txt", "\n".join(manifest_lines) + "\n")
    return available, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a compact final research closure bundle without prediction-level CSVs.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(BUNDLE_NAME))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available, missing = write_bundle(args.project_root, args.output)
    print(f"Wrote closure bundle: {args.output}")
    print(f"Included artifacts: {len(available)}")
    print(f"Missing optional artifacts: {len(missing)}")
    if missing:
        print("Missing paths:")
        for rel_path in missing:
            print(f"  - {rel_path}")


if __name__ == "__main__":
    main()
