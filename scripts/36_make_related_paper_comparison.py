#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from covid_rars.related_papers import write_related_paper_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write source-backed related-paper comparison tables.")
    parser.add_argument("--output", type=Path, default=Path("reports/tables/related_paper_comparison.csv"))
    parser.add_argument("--markdown-output", type=Path, default=Path("reports/tables/related_paper_comparison.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = write_related_paper_comparison(args.output, args.markdown_output)
    print(f"Wrote related-paper comparison CSV: {args.output} ({len(table)} rows)")
    print(f"Wrote related-paper comparison markdown: {args.markdown_output}")


if __name__ == "__main__":
    main()
