from __future__ import annotations

from pathlib import Path


def test_phase_results_brief_has_three_research_phases() -> None:
    path = (
        Path(__file__).parents[1]
        / "reports"
        / "final"
        / "COVID_RARS_PHASED_RESULTS_BRIEF_2026-06-15.md"
    )

    text = path.read_text(encoding="utf-8")

    assert "# Covid-RARS Phase-Wise Results Brief" in text
    assert "## Phase 1" in text
    assert "## Phase 2" in text
    assert "## Phase 3" in text
    assert text.index("## Phase 1") < text.index("## Phase 2") < text.index("## Phase 3")


def test_phase_three_contains_full_research_closure_evidence() -> None:
    path = (
        Path(__file__).parents[1]
        / "reports"
        / "final"
        / "COVID_RARS_PHASED_RESULTS_BRIEF_2026-06-15.md"
    )

    text = path.read_text(encoding="utf-8")
    phase_three = text.split("## Phase 3", maxsplit=1)[1]
    required_terms = [
        "metadata",
        "AUROC 0.964",
        "demographic/protocol",
        "AUROC 0.914",
        "recording_year",
        "recording_month",
        "IPW",
        "ESS = 130.4",
        "SMD = 0.724",
        "AUPRC lift",
        "domain AUROC 0.966",
        "CORAL",
        "MMD 0.055",
        "0.004",
        "ECE 0.286",
        "ECE 0.001",
        "paired bootstrap",
        "p = 0.304",
        "unknown-label",
        "not a clinical diagnostic",
    ]

    for term in required_terms:
        assert term in phase_three
