from __future__ import annotations

import pandas as pd


def _evidence_row(
    claim_id: str,
    evidence_type: str,
    primary_metric: str,
    primary_value: float,
    evidence_direction: str,
    claim: str,
    n_samples: int = 100,
    secondary_metrics: str = "",
    comparison: str = "comparison",
    artifact: str = "artifact.csv",
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "claim": claim,
        "evidence_type": evidence_type,
        "artifact": artifact,
        "comparison": comparison,
        "primary_metric": primary_metric,
        "primary_value": primary_value,
        "secondary_metrics": secondary_metrics,
        "n_samples": n_samples,
        "evidence_direction": evidence_direction,
        "paper_use": "Use this row in the manuscript.",
    }


def _toy_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _evidence_row(
                "external_transfer_beats_best",
                "external_transfer",
                "auroc",
                0.553,
                "cautionary",
                "Best BEATs external transfer remains weak under Coswara-to-COUGHVID shift.",
                n_samples=8331,
                secondary_metrics="auprc=0.039; ece=0.201",
                comparison="BEATs / logistic_regression / drop_high_shift",
                artifact="data/outputs/metrics/external_model_grid_beats_metrics.csv",
            ),
            _evidence_row(
                "coughvid_internal_mfcc_best",
                "internal_baseline",
                "auroc",
                0.781,
                "context",
                "MFCC is stronger in COUGHVID internal validation than in external transfer.",
                n_samples=1667,
                secondary_metrics="auprc=0.178",
            ),
            _evidence_row(
                "metadata_confounding_full_safe_metadata",
                "metadata_confounding",
                "auroc",
                0.964,
                "cautionary",
                "Non-audio metadata alone predicts COVID label.",
                n_samples=2862,
                secondary_metrics="auprc=0.928",
            ),
            _evidence_row(
                "confounding_controlled_audio_ipw",
                "confounding_controlled_audio",
                "auroc",
                0.780,
                "qualified_supportive",
                "The audio signal persists after inverse-propensity confounder control.",
                n_samples=318,
                secondary_metrics="auprc=0.537; effective_sample_size=130.439",
            ),
            _evidence_row(
                "clinical_fusion_specificity_0_900",
                "clinical_operating_point",
                "sensitivity",
                0.699,
                "operational_context",
                "At specificity>=0.900, quality-weighted fusion has a concrete operating-point tradeoff.",
                n_samples=318,
                secondary_metrics="specificity=0.907; precision=0.783",
            ),
            _evidence_row(
                "domain_shift_beats_max",
                "domain_shift",
                "domain_auroc",
                0.990,
                "cautionary",
                "BEATs features strongly separate source and external datasets.",
                n_samples=100,
                secondary_metrics="domain_auprc=0.980; n_features=128",
            ),
            _evidence_row(
                "domain_adaptation_coral_best",
                "domain_adaptation",
                "auroc",
                0.570,
                "adaptation_context",
                "CORAL tests whether second-order alignment can close the external transfer gap.",
                n_samples=8331,
                secondary_metrics="mmd_reduction=0.220; source_only_auroc=0.550",
            ),
            _evidence_row(
                "ipw_sensitivity_cap_2",
                "ipw_sensitivity",
                "auroc",
                0.760,
                "qualified_supportive",
                "IPW audio performance remains visible under stricter weight caps.",
                n_samples=318,
                secondary_metrics="effective_sample_size=190.000; max_abs_smd_after=0.300",
            ),
            _evidence_row(
                "external_prevalence_recalibration_best",
                "prevalence_recalibration",
                "ece_reduction",
                0.230,
                "reliability_context",
                "Target-prevalence intercept correction improves calibration but not discrimination.",
                n_samples=8331,
                secondary_metrics="corrected_ece=0.050; auroc=0.550",
            ),
            _evidence_row(
                "paired_bootstrap_external_best_vs_baseline",
                "paired_bootstrap_comparison",
                "auroc_difference",
                0.020,
                "comparison_context",
                "Best external model only modestly differs from logistic all-feature baseline.",
                n_samples=8331,
                secondary_metrics="ci_low=-0.010; ci_high=0.050",
            ),
            _evidence_row(
                "calibration_external_transfer_worst",
                "calibration_under_shift",
                "ece",
                0.286,
                "cautionary",
                "External transfer probabilities are strongly over-confident.",
                n_samples=8331,
                secondary_metrics="observed_prevalence=0.034; mean_probability=0.321",
            ),
        ]
    )


def test_final_report_contains_evidence_driven_sections() -> None:
    from covid_rars.final_report import build_final_report

    report = build_final_report(_toy_evidence())

    assert "# Covid-RARS Final Results Report" in report
    assert "## Executive Position" in report
    assert "## Pipeline Architecture" in report
    assert "## Decision Log" in report
    assert "## Quantitative Evidence Matrix" in report
    assert "## Tier-2 Strengthening Analyses" in report
    assert "## Novelty" in report
    assert "## Publication Readiness" in report
    assert "not a clinically deployable diagnostic model" in report
    assert "external_transfer_beats_best" in report
    assert "0.553" in report
    assert "metadata_confounding_full_safe_metadata" in report
    assert "confounding_controlled_audio_ipw" in report
    assert "calibration_external_transfer_worst" in report
    assert "domain_shift_beats_max" in report
    assert "domain_adaptation_coral_best" in report
    assert "external_prevalence_recalibration_best" in report


def test_final_report_includes_related_paper_table_when_provided() -> None:
    from covid_rars.final_report import build_final_report

    related = "# Related-Paper Comparison\n\n| paper_id | title |\n| --- | --- |\n| P1 | Base paper |\n"

    report = build_final_report(_toy_evidence(), related_paper_markdown=related)

    assert "## Related-Paper Comparison" in report
    assert "| P1 | Base paper |" in report


def test_summary_report_is_shorter_and_preserves_main_claims() -> None:
    from covid_rars.final_report import build_final_report, build_summary_report

    evidence = _toy_evidence()
    report = build_final_report(evidence)
    summary = build_summary_report(evidence)

    assert "# Covid-RARS Results Summary" in summary
    assert "Best external transfer" in summary
    assert "IPW-controlled" in summary
    assert "metadata confounding" in summary
    assert len(summary) < len(report)


def test_final_report_requires_evidence_rows() -> None:
    from covid_rars.final_report import build_final_report

    try:
        build_final_report(pd.DataFrame())
    except ValueError as exc:
        assert "evidence matrix" in str(exc)
    else:
        raise AssertionError("Expected empty evidence matrix to fail")
