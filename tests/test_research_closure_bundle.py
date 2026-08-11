from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).parents[1] / "scripts" / "43_make_research_closure_bundle.py"
    spec = importlib.util.spec_from_file_location("research_closure_bundle_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_bundle_artifacts_include_core_docs_and_no_prediction_tables() -> None:
    module = _load_script_module()

    artifacts = module.default_bundle_artifacts()
    names = [str(path) for path in artifacts]

    assert "reports/final/COVID_RARS_PHASED_RESULTS_BRIEF_2026-06-15.md" in names
    assert "reports/final/COVID_RARS_PUBLICATION_RESULTS_REPORT.md" in names
    assert "reports/final/MANUSCRIPT_SUPPORT_ANALYSES.md" in names
    assert "reports/tables/publication_evidence_matrix.csv" in names
    assert "reports/tables/related_paper_comparison.csv" in names
    assert "reports/tables/manuscript_external_auprc_lift.csv" in names
    assert not any("predictions.csv" in name for name in names)


def test_bundle_cli_writes_zip_with_available_artifacts(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    project = tmp_path / "project"
    output = tmp_path / "bundle.zip"
    for rel_path in module.default_bundle_artifacts()[:5]:
        path = project / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"artifact: {rel_path}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "43_make_research_closure_bundle.py",
            "--project-root",
            str(project),
            "--output",
            str(output),
        ],
    )

    module.main()

    assert output.exists()
    with zipfile.ZipFile(output) as bundle:
        names = set(bundle.namelist())
    assert "MANIFEST.txt" in names
    assert "reports/final/COVID_RARS_PHASED_RESULTS_BRIEF_2026-06-15.md" in names
