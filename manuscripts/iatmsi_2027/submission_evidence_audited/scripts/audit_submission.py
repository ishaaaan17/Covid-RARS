"""Fail-fast audit for the evidence-audited IATMSI submission."""

from __future__ import annotations

import re
import subprocess
import sys
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

    required_macros = {
        "FusionTestAUROC",
        "FusionAUCILow",
        "FusionAUCIHigh",
        "FusionTestAUPRC",
        "FusionTestN",
        "SpeechTestAUROC",
        "CandidateFeatureCount",
        "SelectedFeatureCount",
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
    }
    for pattern, description in prohibited.items():
        if re.search(pattern, tex, flags=re.IGNORECASE if "SOTA" not in pattern else 0):
            fail(f"{description} found in conference manuscript")

    if "\\documentclass[conference]{IEEEtran}" not in tex:
        fail("manuscript is not using IEEEtran conference mode")
    if "descriptive" not in tex or "does not establish" not in tex:
        fail("inferential boundary for the fusion comparison is missing")

    if not PDF.exists() or not LOG.exists():
        fail("compiled main.pdf/main.log not found; compile before auditing")
    pages = pdf_pages(PDF)
    if pages > 6:
        fail(f"paper has {pages} pages; IATMSI limit is 6")

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
    for text_fragment in ("0.895", "0.852", "0.933", "10,140", "800"):
        if text_fragment not in extracted:
            fail(f"compiled PDF is missing expected evidence value {text_fragment}")

    print(f"AUDIT PASSED: {pages} pages, {len(cited)} cited primary sources, no blocked claims or LaTeX errors")


if __name__ == "__main__":
    main()
