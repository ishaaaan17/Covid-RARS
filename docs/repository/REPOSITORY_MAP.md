# Covid-RARS Repository Map

This map explains the current root-level repository layout after the Covid-RARS cleanup.

## Top-Level Structure

| Path | Role |
|---|---|
| `src/covid_rars/` | Importable Python implementation |
| `scripts/` | Numbered command-line workflow scripts |
| `tests/` | Pytest suite for package modules and CLI behavior |
| `notebooks/` | Notebook workflow and review notebooks |
| `app/` | Local demonstration application |
| `configs/` | Versioned experiment and HST configuration |
| `docs/` | Current research briefing, repository notes, and operational guidance |
| `research_protocol/` | Versioned study protocols and methodological decisions |
| `data/` | Local datasets and generated experiment outputs; raw data follows source licenses |
| `reports/` | Selected evidence tables, figures, manifests, and final research notes |
| `results/frozen/` | Frozen publication, Coswara, CNN, and external-validation result folders |
| `results/representations/` | Frozen OpenSMILE/BEATs/PANNs representation result folders |
| `artifacts/bundles/` | Compressed zip/tar.gz bundles preserved as evidence packages |
| `manuscripts/` | Venue-specific manuscript drafts, generated PDFs, shared figures, and source artifacts |
| `archive/patches/` | Historical patch files retained for traceability |
| `archive/updates/` | Historical update notes |
| `archive/review_materials/` | Gemini/PDF review exports and duplicate root review snapshots |
| `archive/historical_project_docs/` | Superseded project notes and runbooks preserved without rewriting |
| `archive/historical_status_snapshots/` | Dated implementation and decision snapshots preserved for traceability |
| `docs/repository/` | Repository-level maps and hygiene documentation |

Generated files under `data/`, `reports/figures/`, `reports/tables/`, and
`reports/hst/` are ignored by default. Evidence files already selected for the
research record remain versioned. This prevents routine reruns from flooding
the working tree while preserving the results used by the manuscripts.

## Active Code Map

| Path | Role |
|---|---|
| `src/covid_rars/` | Canonical implementation package |
| `scripts/` | CLI entry points and one-shot experiment runners |
| `tests/` | Unit and integration tests |
| `notebooks/` | Notebook workflow and review notebooks |
| `configs/` | HST and experiment configuration files |
| `requirements*.txt` | Core, development, optional, HST, and GPU dependency sets |
| `pyproject.toml` | Package metadata and pytest configuration |

## Main Evidence Documents

| File | Use |
|---|---|
| `docs/research_briefing/COVID_RARS_E2E_PROJECT_BRIEF.md` | End-to-end explanation of the research pipeline |
| `docs/research_briefing/COVID_RARS_RESULTS_EVIDENCE.md` | Metrics ledger and safe interpretations |
| `docs/research_briefing/COVID_RARS_PLAIN_LANGUAGE_EXPLANATION_GUIDE.md` | Simple explanations for meeting and review questions |
| `docs/research_briefing/COVID_RARS_RESULTS_COMPARISON.md` | Paper/result comparison guidance |
| `references/verified_source_registry.md` | Source-backed guardrail for scope and claims |

## Main Code Families

| Area | Representative files |
|---|---|
| Data indexing and metadata | `datasets.py`, `data_index.py`, `metadata.py`, `labels.py`, `validation.py` |
| Preprocessing and quality | `audio_io.py`, `preprocess.py`, `quality.py`, `split.py` |
| Acoustic features | `features.py`, `strong_features.py`, `opensmile_features.py`, `representation_features.py`, `ssl_extractors.py` |
| Classical models and fusion | `models_ml.py`, `fusion.py`, `calibration.py`, `strong_baseline.py`, `compare_is10_final_validation.py` |
| Deep/representation models | `models_cnn.py`, `train_cnn.py`, `sota_ssl.py`, `spectrograms.py` |
| Reliability audits | `domain_shift_audit.py`, `metadata_confounding.py`, `temporal_holdout.py`, `temporal_month_causal.py`, `ipw_sensitivity.py`, `clinical_operating_points.py` |
| Reporting | `reporting.py`, `final_report.py`, `publication_evidence.py`, `manifest.py`, `related_papers.py` |

## Claim Discipline

Use this repository as a reliability and domain-shift artifact. The most defensible claim is that strong internal respiratory-audio performance does not imply deployment robustness without temporal validation, external transfer, calibration, metadata-confounding checks, and decision-oriented evaluation.

Do not describe the repository as a clinical diagnostic system.
