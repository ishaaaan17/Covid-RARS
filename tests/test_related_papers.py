from __future__ import annotations

import pandas as pd


def test_related_paper_comparison_contains_original_doc_sources() -> None:
    from covid_rars.related_papers import build_related_paper_comparison

    table = build_related_paper_comparison()

    required = {
        "paper_id",
        "title",
        "source_year",
        "role",
        "datasets",
        "method",
        "reported_results",
        "main_limitation",
        "how_ours_compares",
        "source_doc",
    }
    assert required.issubset(table.columns)
    assert len(table) >= 18

    base = table[table["paper_id"].eq("P1")].iloc[0]
    assert "Drift-Adaptive" in base["title"]
    assert "69.1" in base["reported_results"]
    assert "cross-dataset" in base["how_ours_compares"]

    caution = table[table["paper_id"].eq("P5")].iloc[0]
    assert "Nature Machine Intelligence" in caution["source_year"]
    assert "confounding" in caution["main_limitation"].lower()
    assert "metadata confounding" in caution["how_ours_compares"].lower()


def test_related_paper_markdown_is_table_with_conservative_position() -> None:
    from covid_rars.related_papers import related_paper_comparison_to_markdown

    table = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "title": "Example paper",
                "source_year": "Journal, 2025",
                "role": "base",
                "datasets": "Coswara",
                "method": "CNN",
                "reported_results": "AUROC 0.70",
                "main_limitation": "Limited external validation",
                "how_ours_compares": "We add external validation",
                "source_doc": "source.md",
            }
        ]
    )

    markdown = related_paper_comparison_to_markdown(table)

    assert "# Related-Paper Comparison" in markdown
    assert "| paper_id | title | source_year |" in markdown
    assert "We add external validation" in markdown
    assert "conservative comparison" in markdown.lower()
