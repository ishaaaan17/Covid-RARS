# JBHI Ground-Up Manuscript

This folder contains a new manuscript built from the repository's executable methods and completed evidence artifacts. Earlier COVID-RARS manuscripts are not source material for its prose, structure, figures, or tables.

## Scope

- Target: IEEE Journal of Biomedical and Health Informatics.
- Working title: *Multimodal COVID-19 Respiratory Sound Models: Temporal Stability and External Cough Transfer*.
- Study type: retrospective reliability and transportability evaluation of respiratory-audio screening models.
- Primary comparison: fixed cough modality and model family, with frozen Coswara source models scored on the COUGHVID external cohort.
- Supporting analyses: internal multimodal modeling, architecture checks, retrospective temporal stress testing, calibration, operating points, metadata associations, feature stability, support overlap, and complete-case incremental-value analysis.
- No new experiments are introduced for manuscript preparation.

## Rebuild

From this directory:

```powershell
& 'C:\Program Files\MATLAB\R2026a\bin\matlab.exe' -batch "addpath('figures'); build_figures"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

The MATLAB figure builder reads only repository result tables and writes vector PDF figures plus 300-DPI PNG inspection copies. It uses explicit panel positions rather than automatic subplot packing. The claim ledger in `evidence_map.csv` records the source and interpretation boundary for every central numerical statement.

The main paper uses four double-column figures: study design, controlled external transport, external endpoint reliability, and context/stability diagnostics. Result panels are separated by whitespace rather than decorative rules or enclosing boxes.

`PROTOCOL_AWARE_PAPER_COMPARISON.md` is a separate working note containing the numerical comparison, qualitative contribution matrix, claim boundaries, and concise reviewer defenses. It is not part of the manuscript or supplement.
