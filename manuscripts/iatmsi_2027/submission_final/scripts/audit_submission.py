"""Fail-fast audit for the final IATMSI conference manuscript."""

from __future__ import annotations

import re
import subprocess
import sys
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "main.tex"
BIB = ROOT / "references.bib"
PDF = ROOT / "main.pdf"
LOG = ROOT / "main.log"


def fail(message: str) -> None:
    raise SystemExit(f"AUDIT FAILED: {message}")


def citation_keys(tex: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", tex):
        keys.update(key.strip() for key in match.group(1).split(","))
    return keys


def bibliography_keys(bib: str) -> list[str]:
    return re.findall(r"@[A-Za-z]+\s*\{\s*([^,\s]+)", bib)


def pdf_pages(path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        text=True,
        capture_output=True,
    )
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    if not match:
        fail("pdfinfo did not report a page count")
    return int(match.group(1))


def main() -> None:
    required = [
        TEX,
        BIB,
        ROOT / "manuscript_values.tex",
        ROOT / "figures" / "study_design.pdf",
        ROOT / "figures" / "selection_and_results.pdf",
        ROOT / "tables" / "claim_evidence_ledger.csv",
        ROOT / "tables" / "paired_fusion_branch_comparisons.csv",
        ROOT / "tables" / "heldout_auroc_intervals.csv",
        ROOT / "tables" / "selected_auprc_bootstrap_record.csv",
        ROOT / "tables" / "feature_level_fusion_comparator.csv",
        ROOT / "tables" / "multiseed_uniform_fusion.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        fail(f"missing required files: {', '.join(missing)}")

    tex = TEX.read_text(encoding="utf-8")
    bib = BIB.read_text(encoding="utf-8")
    values = (ROOT / "manuscript_values.tex").read_text(encoding="utf-8")

    cited = citation_keys(tex)
    bib_list = bibliography_keys(bib)
    bib_keys = set(bib_list)
    if len(bib_list) != len(bib_keys):
        fail("duplicate bibliography keys")
    if cited - bib_keys:
        fail(f"unresolved bibliography keys: {sorted(cited - bib_keys)}")
    if bib_keys - cited:
        fail(f"uncited bibliography entries: {sorted(bib_keys - cited)}")
    if len(cited) < 25:
        fail(f"literature coverage is unexpectedly thin: only {len(cited)} cited sources")

    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.DOTALL
    )
    if not abstract_match:
        fail("abstract is missing")
    abstract = abstract_match.group(1)
    if re.search(r"\\(?:cite|ref|eqref|begin\{equation)", abstract):
        fail("abstract must be self-contained and cannot contain citations or equations")
    abstract_for_count = re.sub(r"\\[A-Za-z]+(?:\{[^{}]*\})?", " ", abstract)
    abstract_words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", abstract_for_count)
    if not 150 <= len(abstract_words) <= 250:
        fail(f"abstract has {len(abstract_words)} words; expected 150--250")
    if ";" in tex:
        fail("semicolon found in manuscript prose; style audit expects none")

    required_reference_fragments = (
        "The {{\\mbox{INTERSPEECH}}} 2016 Computational Paralinguistics",
        "Deception",
        "10.21437/Interspeech.2016-129",
        "2001--2005",
    )
    for fragment in required_reference_fragments:
        if fragment not in bib:
            fail(f"corrected INTERSPEECH 2016 reference is missing {fragment!r}")
    for obsolete in ("10.21437/Interspeech.2016-1299", "1201--1205"):
        if obsolete in bib:
            fail(f"obsolete INTERSPEECH 2016 metadata remains: {obsolete}")
    for fragment in (
        "delong1988correlated",
        "10.2307/2531595",
        "837--845",
    ):
        if fragment not in bib:
            fail(f"DeLong method reference is missing {fragment!r}")

    required_macros = {
        "FusionTestAUROC",
        "FusionAUCILow",
        "FusionAUCIHigh",
        "FusionAUPRCILow",
        "FusionAUPRCIHigh",
        "FusionTestAUPRC",
        "FusionTestN",
        "SpeechTestAUROC",
        "CandidateFeatureCount",
        "SelectedFeatureCount",
        "FusionMinusCoughAUROC",
        "FusionMinusCoughAUCILow",
        "FusionMinusCoughAUCIHigh",
        "FusionMinusCoughAUROCHolmPValue",
        "FusionMinusSpeechAUROC",
        "FusionMinusSpeechAUCILow",
        "FusionMinusSpeechAUCIHigh",
        "FusionMinusSpeechAUROCPValue",
        "FeatureFusionTestAUROC",
        "FeatureFusionTestAUPRC",
        "ScoreMinusFeatureAUROC",
        "ScoreMinusFeatureAUPRC",
        "MultiSeedAUROCMean",
        "MultiSeedAUROCSD",
        "UniformTestBrier",
        "UniformTestECE",
        "StackTestBrier",
        "StackTestECE",
    }
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", values))
    if required_macros - defined:
        fail(f"missing generated value macros: {sorted(required_macros - defined)}")
    for macro in required_macros:
        if f"\\{macro}" not in tex:
            fail(f"generated macro {macro} is not used in main.tex")

    prohibited = {
        r"\bstate[- ]of[- ]the[- ]art\b": "unsupported state-of-the-art language",
        r"\bSOTA\b": "unsupported SOTA language",
        r"metadata[- ]only": "journal-only metadata analysis",
        r"early[- ]to[- ]late": "journal-only temporal analysis",
        r"0\.543|0\.698|0\.964": "journal-only headline metric",
        r"0\.41\b": "known obsolete feature-stability value",
        r"paired uncertainty was unavailable": "obsolete paired-uncertainty claim",
        r"paired fusion-versus-speech interval was unavailable": "obsolete paired-uncertainty claim",
    }
    for pattern, description in prohibited.items():
        if re.search(pattern, tex, flags=re.IGNORECASE if "SOTA" not in pattern else 0):
            fail(f"{description} found in conference manuscript")

    if "\\documentclass[conference]{IEEEtran}" not in tex:
        fail("manuscript is not using IEEEtran conference mode")
    if "Participant-Disjoint Evaluation of Multisound Fusion" not in tex:
        fail("unexpected manuscript title")
    if "descriptive" not in tex or "does not establish" not in tex:
        fail("inferential boundary for the fusion comparison is missing")

    if not PDF.exists() or not LOG.exists():
        fail("compiled main.pdf/main.log not found; compile before auditing")
    pages = pdf_pages(PDF)
    if pages != 6:
        fail(f"paper has {pages} pages; the final submission must use the 6-page IATMSI limit")

    log = LOG.read_text(encoding="utf-8", errors="replace")
    fatal_patterns = [
        r"LaTeX Error",
        r"Undefined control sequence",
        r"Citation .+ undefined",
        r"There were undefined references",
        r"Overfull \\[hv]box",
    ]
    for pattern in fatal_patterns:
        match = re.search(pattern, log)
        if match:
            fail(f"LaTeX log contains: {match.group(0)}")

    extracted = subprocess.run(
        ["pdftotext", str(PDF), "-"],
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    ).stdout
    for text_fragment in (
        "0.895",
        "0.852",
        "0.933",
        "0.802",
        "0.909",
        "10,140",
        "800",
        "0.033",
        "0.012",
        "0.053",
        "0.004",
        "0.460",
        "0.878",
        "0.828",
        "0.017",
        "0.033",
        "0.8945",
        "0.0068",
        "0.8862",
        "0.9049",
    ):
        if text_fragment not in extracted:
            fail(f"compiled PDF is missing expected evidence value {text_fragment}")

    paired_path = ROOT / "tables" / "paired_fusion_branch_comparisons.csv"
    paired = paired_path.read_text(encoding="utf-8")
    paired_expectations = (
        "fusion_minus_cough,Cough,Cough+speech mean,auroc,314,102,212",
        "fusion_minus_speech,Speech,Cough+speech mean,auroc,314,102,212",
        "paired_delong",
        "paired_participant_bootstrap",
    )
    for fragment in paired_expectations:
        if fragment not in paired:
            fail(f"paired comparison evidence is missing {fragment!r}")

    with (ROOT / "tables" / "feature_level_fusion_comparator.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        feature_rows = list(csv.DictReader(handle))
    feature_test = [row for row in feature_rows if row["metric_split"] == "test"]
    if len(feature_test) != 1:
        fail("feature-level comparator does not contain one test row")
    if feature_test[0]["model_name"] != "xgboost_smote_f80":
        fail("feature-level comparator is not the validation-selected XGBoost row")
    if abs(float(feature_test[0]["auroc"]) - 0.878422) > 1e-6:
        fail("feature-level comparator AUROC has drifted")

    with (ROOT / "tables" / "multiseed_uniform_fusion.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        seed_rows = list(csv.DictReader(handle))
    observed_seeds = [int(float(row["random_state"])) for row in seed_rows]
    if observed_seeds != [42, 43, 44, 45, 46]:
        fail(f"unexpected multi-seed evidence: {observed_seeds}")
    if any(int(float(row["n_samples"])) != 314 for row in seed_rows):
        fail("multi-seed rows do not use the fixed 314-participant test cohort")

    with (ROOT / "tables" / "heldout_auroc_intervals.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        interval_rows = list(csv.DictReader(handle))
    expected_systems = {
        "Breathing",
        "Cough",
        "Speech",
        "Feature concat.",
        "Score fusion",
    }
    if {row["system"] for row in interval_rows} != expected_systems:
        fail("held-out AUROC interval systems do not match Figure 2")
    if len(interval_rows) != len(expected_systems):
        fail("held-out AUROC interval table contains duplicate systems")
    if any(int(float(row["n_bootstraps"])) != 1000 for row in interval_rows):
        fail("Figure 2 held-out AUROC intervals do not use 1,000 resamples")
    if any(
        float(row["ci_low"]) > float(row["point"])
        or float(row["point"]) > float(row["ci_high"])
        for row in interval_rows
    ):
        fail("Figure 2 held-out AUROC point lies outside its interval")

    malformed_reference_patterns = {
        "——": "repeated-author dash",
        "INTER-\nSPEECH": "broken INTERSPEECH name",
        "chal-\nlenge": "broken challenge title",
    }
    for fragment, description in malformed_reference_patterns.items():
        if fragment in extracted:
            fail(f"compiled bibliography contains {description}")

    figure_source = (ROOT / "scripts" / "build_assets.py").read_text(encoding="utf-8")
    for obsolete_label in ("Locked test", "Locked-test AUROC"):
        if obsolete_label in figure_source:
            fail(f"obsolete figure wording remains: {obsolete_label}")

    print(
        "AUDIT PASSED: "
        f"{pages} pages, {len(abstract_words)} abstract words, {len(cited)} cited sources, "
        "no blocked claims or LaTeX errors"
    )


if __name__ == "__main__":
    main()
