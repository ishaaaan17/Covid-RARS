from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_script_module():
    script_path = Path(__file__).parents[1] / "scripts" / "36_make_related_paper_comparison.py"
    spec = importlib.util.spec_from_file_location("related_papers_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_related_papers_cli_writes_csv_and_markdown(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    csv_output = tmp_path / "related_paper_comparison.csv"
    markdown_output = tmp_path / "related_paper_comparison.md"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "36_make_related_paper_comparison.py",
            "--output",
            str(csv_output),
            "--markdown-output",
            str(markdown_output),
        ],
    )

    module.main()

    table = pd.read_csv(csv_output)
    assert len(table) >= 18
    assert "P1" in set(table["paper_id"])
    assert "Related-Paper Comparison" in markdown_output.read_text()
