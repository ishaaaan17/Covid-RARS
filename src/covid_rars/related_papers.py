from __future__ import annotations

from pathlib import Path

import pandas as pd


RELATED_PAPER_COLUMNS = [
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
]


RELATED_PAPER_ROWS: list[dict[str, str]] = [
    {
        "paper_id": "P1",
        "title": "A Comprehensive Drift-Adaptive Framework for Sustaining Model Performance in COVID-19 Detection From Dynamic Cough Audio Data",
        "source_year": "JMIR, 2025",
        "role": "Main base paper",
        "datasets": "COVID-19 Sounds; Coswara",
        "method": "Cough audio, mel spectrograms, VGGish features, chronological splits, MMD drift detection, CUSUM alerts, UDA and active learning",
        "reported_results": "Development AUROC 69.1% on COVID-19 Sounds and 66.8% on Coswara; postdevelopment AUROC dropped to 60.7% and 59.7%; UDA improved balanced accuracy by roughly 10%-20%",
        "main_limitation": "Focuses on temporal drift; limited interpretation of causes, cross-dataset drift, and interdemographic variability; cough-only",
        "how_ours_compares": "We add Coswara-to-COUGHVID cross-dataset transfer, metadata confounding, operating points, and calibration-under-shift, but do not implement full UDA/active learning",
        "source_doc": "references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/source_plans/literature_matrix_v2_18_papers.md",
    },
    {
        "paper_id": "P2",
        "title": "Coswara: A respiratory sounds and symptoms dataset for remote screening of SARS-CoV-2 infection",
        "source_year": "Scientific Data, 2023",
        "role": "Primary dataset paper",
        "datasets": "Coswara: 2635 individuals, 23,700 recordings, about 65 hours, 9 sound categories",
        "method": "Crowdsourced cough, breath, vowel, and counting recordings plus symptoms and metadata",
        "reported_results": "Dataset paper; not a model leaderboard in our comparison table",
        "main_limitation": "Crowdsourced recordings and metadata-driven label associations require careful quality and confounding analysis",
        "how_ours_compares": "We use Coswara as the core source dataset with participant-level split, quality checks, metadata audit, and representation comparison",
        "source_doc": "references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/verified_source_registry.md",
    },
    {
        "paper_id": "P3",
        "title": "COVID-19 Sounds: A Large-Scale Audio Dataset for Digital Respiratory Screening",
        "source_year": "NeurIPS Datasets and Benchmarks, 2021",
        "role": "Large multimodal dataset reference",
        "datasets": "53,449 audio samples, 552+ hours, 36,116 participants, breathing/cough/voice",
        "method": "Large-scale respiratory audio dataset with participant metadata",
        "reported_results": "Dataset reference; raw access requires request/data agreement",
        "main_limitation": "Not instantly available for this implementation; cannot be required for reproducibility",
        "how_ours_compares": "We cite it for field context but rely on public Coswara plus COUGHVID for executable validation",
        "source_doc": "references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md",
    },
    {
        "paper_id": "P4",
        "title": "The COUGHVID crowdsourcing dataset",
        "source_year": "Scientific Data, 2021",
        "role": "External cough validation dataset",
        "datasets": "More than 25,000 crowdsourced cough recordings; more than 2800 expert-labeled recordings",
        "method": "Crowdsourced cough collection with metadata, cough detection, and quality tools",
        "reported_results": "Dataset paper; expert agreement was limited for some labels",
        "main_limitation": "Noisy labels and crowdsourced quality make it an external robustness target, not perfect ground truth",
        "how_ours_compares": "We use COUGHVID only as external validation/internal baseline context and report weak transfer honestly",
        "source_doc": "references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/verified_source_registry.md",
    },
    {
        "paper_id": "P5",
        "title": "Audio-based AI classifiers show no evidence of improved COVID-19 screening over simple symptoms checkers",
        "source_year": "Nature Machine Intelligence, 2024",
        "role": "Required confounding critique",
        "datasets": "UK COVID-19 Vocal Audio Dataset with PCR-referenced cough, exhalation, and speech",
        "method": "Audio classifiers compared against symptom/checker baselines under realistic evaluation",
        "reported_results": "Main conclusion: audio models may not outperform simple symptom checkers under realistic evaluation",
        "main_limitation": "Shows that apparent audio performance can reflect confounding signals, symptoms, and sampling differences",
        "how_ours_compares": "We add metadata confounding audits, symptom/demographic/protocol-only models, IPW-controlled audio evaluation, and conservative claims",
        "source_doc": "references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/source_plans/literature_matrix_v2_18_papers.md",
    },
    {
        "paper_id": "P6",
        "title": "Confidence-Calibrated Clinical Decision Support System for Reliable Respiratory Disease Screening",
        "source_year": "IEEE-BHI/OpenReview, 2024",
        "role": "Calibration base paper",
        "datasets": "Coswara; Cambridge COVID-19 Sounds",
        "method": "MFCC features, DNN, ensemble-based confidence calibration, LIME-style interpretability",
        "reported_results": "ENCL-DNN AUROC 0.834 on Coswara and 0.854 on Cambridge; ECE reduced by 50.0% on Coswara and 28.74% on Cambridge",
        "main_limitation": "Mostly cough-focused and not a full external shift/confounding pipeline",
        "how_ours_compares": "We include calibration metrics, Brier/NLL/ECE, calibration-under-shift, and avoid interpreting external probabilities as calibrated risks",
        "source_doc": "references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/verified_source_registry.md",
    },
    {
        "paper_id": "P7",
        "title": "COVID-19 Detection From Respiratory Sounds With Hierarchical Spectrogram Transformers",
        "source_year": "IEEE JBHI, 2023/2024",
        "role": "Advanced architecture reference",
        "datasets": "Crowdsourced respiratory sound datasets with cough and breathing sounds",
        "method": "Hierarchical Spectrogram Transformer with local-to-global attention",
        "reported_results": "Reports over 83% AUC for COVID-19 detection from respiratory sounds",
        "main_limitation": "Compute-heavy transformer architecture; not necessary for BTech reproducibility",
        "how_ours_compares": "We compare modern learned embeddings through BEATs/PANNs and keep the main contribution on reliability rather than architecture novelty",
        "source_doc": "references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/source_plans/literature_matrix_v2_18_papers.md",
    },
    {
        "paper_id": "P8",
        "title": "SympCoughNet: symptom assisted audio-based COVID-19 detection",
        "source_year": "Frontiers in Digital Health, 2025",
        "role": "Symptom-assisted recent paper",
        "datasets": "UK COVID-19 Vocal Audio Dataset; 72,999 participants and 25,766 PCR-positive cases reported in project notes",
        "method": "Log-mel spectrograms, CNN backbone, symptom-encoded channel attention, augmentation, VAD/noise preprocessing",
        "reported_results": "Accuracy 89.30%, AUROC 94.74%, PR 91.62%",
        "main_limitation": "Symptom-assisted performance can be dominated by symptom metadata rather than audio-specific signal",
        "how_ours_compares": "We use symptoms as confounding/audit metadata rather than central diagnostic input",
        "source_doc": "references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/source_plans/literature_matrix_v2_18_papers.md",
    },
    {
        "paper_id": "P9",
        "title": "Speech-based respiratory diagnostics: A study on COVID-19 detection with machine learning",
        "source_year": "PLOS ONE, 2025",
        "role": "Speech/vowel modality reference",
        "datasets": "Coswara vowel sounds /a/, /e/, and /o/",
        "method": "ITU-T P.56 normalization, OpenSMILE 1582-dimensional features, RF/SVM/Decision Tree/ANN, feature selection",
        "reported_results": "Random Forest with ANOVA-selected features; accuracy around 76.47% for vowel /a/ and 75.54% for /a/+/o/",
        "main_limitation": "Speech-only performance is moderate and task-specific",
        "how_ours_compares": "We prioritize cough plus representation robustness and keep speech-biased wav2vec2 as lower priority for the current cough transfer story",
        "source_doc": "references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/source_plans/literature_matrix_v2_18_papers.md",
    },
    {
        "paper_id": "P10",
        "title": "Robust COVID-19 detection from cough sounds using deep neural decision tree and forest: A comprehensive cross-datasets evaluation",
        "source_year": "Expert Systems with Applications, 2026",
        "role": "Latest cross-dataset robustness reference",
        "datasets": "Cambridge COVID-19 Sounds, Coswara, COUGHVID, Virufy, NoCoCoDa",
        "method": "Deep Neural Decision Tree/Forest, RFECV, Bayesian hyperparameter optimization, SMOTE, threshold moving",
        "reported_results": "AUC around 0.92 to 0.99 across individual settings; combined DNDF reports accuracy/AUC around 0.97",
        "main_limitation": "Cough-only and method-heavy; high reported metrics may depend on dataset construction and tuning",
        "how_ours_compares": "Our external transfer is far weaker but more conservative; we emphasize calibration/confounding and do not claim comparable SOTA accuracy",
        "source_doc": "references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/source_plans/literature_matrix_v2_18_papers.md",
    },
    {
        "paper_id": "P11",
        "title": "Audio-Based Screening of Respiratory Diseases",
        "source_year": "MDPI, 2026",
        "role": "Segmentation framework reference",
        "datasets": "Respiratory disease audio screening datasets",
        "method": "Event-focused respiratory audio preprocessing and segmentation pipeline",
        "reported_results": "Used for methodology guidance rather than direct COVID metric comparison in current docs",
        "main_limitation": "Broader respiratory screening, not directly reproduced here",
        "how_ours_compares": "We include quality filtering and note event segmentation as a future extension beyond current global feature extraction",
        "source_doc": "references/source_plans/literature_matrix_v2_18_papers.md",
    },
    {
        "paper_id": "P12",
        "title": "Sensors 2022 systematic review on audio-based COVID/respiratory screening",
        "source_year": "Sensors, 2022",
        "role": "Methodology breadth review",
        "datasets": "Review across COVID and respiratory audio studies",
        "method": "Systematic review of preprocessing, models, metrics, and study designs",
        "reported_results": "Review source; no single directly comparable metric",
        "main_limitation": "Review-level evidence cannot validate our model directly",
        "how_ours_compares": "We follow the review's multi-metric and preprocessing-documentation guidance",
        "source_doc": "references/source_plans/literature_matrix_v2_18_papers.md",
    },
    {
        "paper_id": "P13",
        "title": "COVID-19 detection in cough, breath and speech using deep transfer learning and bottleneck features",
        "source_year": "Computers in Biology and Medicine, 2022",
        "role": "Older multimodal baseline",
        "datasets": "Coswara, ComParE, Sarcos and related respiratory audio data",
        "method": "Deep transfer learning and bottleneck features across cough, breath, and speech",
        "reported_results": "AUC about 0.98 for cough, 0.94 for breath, and 0.92 for speech",
        "main_limitation": "Older high metrics require careful comparison because validation/control assumptions may differ",
        "how_ours_compares": "Our results are lower, but we add external COUGHVID transfer, confounding checks, calibration, and operating points",
        "source_doc": "references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/source_plans/literature_matrix_v2_18_papers.md",
    },
    {
        "paper_id": "P14",
        "title": "Pay Attention to the Speech",
        "source_year": "Alexandria Engineering Journal, 2022",
        "role": "Speech/fusion precedent",
        "datasets": "Coswara multimodal audio",
        "method": "Handcrafted audio features, class imbalance handling, attention/fusion over speech and respiratory sounds",
        "reported_results": "Used as speech/fusion precedent in project docs; exact metric extraction deferred to full paper table",
        "main_limitation": "Fusion can degrade if not calibrated and leakage-safe",
        "how_ours_compares": "We compare fusion against best single modality and add calibration/quality/confounding evidence",
        "source_doc": "references/source_plans/literature_matrix_v2_18_papers.md; references/verified_source_registry.md",
    },
    {
        "paper_id": "P15",
        "title": "Audio texture analysis of COVID-19 cough, breath, and speech sounds",
        "source_year": "Biomedical Signal Processing and Control / Frontiers, 2022",
        "role": "Interpretable feature-engineering baseline",
        "datasets": "Cambridge COVID-19 Sounds subset: 1141 cough, 392 breath, 893 speech samples",
        "method": "Audio texture and handcrafted spectrogram features",
        "reported_results": "5-class accuracy around 71.7% cough, 72.2% breath, and speech binary accuracy around 79.7%",
        "main_limitation": "Dataset-specific feature-engineering results; exact binary comparisons need full table extraction",
        "how_ours_compares": "We include handcrafted MFCC/OpenSMILE baselines and learned embeddings, then test external robustness",
        "source_doc": "references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/source_plans/literature_matrix_v2_18_papers.md",
    },
    {
        "paper_id": "P16",
        "title": "Audio feature ranking for sound-based COVID-19 patient detection",
        "source_year": "arXiv, 2021",
        "role": "Feature ranking reference",
        "datasets": "Cambridge and Coswara in project notes",
        "method": "Feature ranking and selection for classical sound-based COVID detection",
        "reported_results": "Supports feature-selection improvements; exact table metrics not captured in current docs",
        "main_limitation": "Feature-ranking evidence does not by itself solve external transfer or confounding",
        "how_ours_compares": "We use feature strategies and shift-based feature selection, then report weak external generalization honestly",
        "source_doc": "references/source_plans/literature_matrix_v2_18_papers.md; references/verified_source_registry.md",
    },
    {
        "paper_id": "P17",
        "title": "Investigating Feature Selection And Explainability For COVID-19 Diagnostics From Cough Sounds",
        "source_year": "Interspeech, 2021",
        "role": "Feature selection and explainability support",
        "datasets": "DiCOVA challenge context in project notes",
        "method": "High-dimensional acoustic features, log-mel CNN, probability-score fusion, explainability/feature selection",
        "reported_results": "Reported DiCOVA challenge improvements; exact metrics not captured in current docs",
        "main_limitation": "Explainability is inspection, not proof of causal COVID biomarkers",
        "how_ours_compares": "We use feature selection and evidence tables, but avoid causal biomarker claims",
        "source_doc": "references/source_plans/literature_matrix_v2_18_papers.md; references/verified_source_registry.md",
    },
    {
        "paper_id": "P18",
        "title": "COVID-19 cough classification using machine learning and global smartphone recordings",
        "source_year": "Computers in Biology and Medicine, 2021",
        "role": "Real-world recording variability reference",
        "datasets": "Coswara and Sarcos/South Africa smartphone cough recordings",
        "method": "Machine learning/deep residual architecture on global smartphone cough recordings",
        "reported_results": "Highest AUC around 0.98 from residual architecture in paper highlights",
        "main_limitation": "Real-world smartphone recordings introduce device/noise/domain variability",
        "how_ours_compares": "We directly show that cross-dataset robustness is difficult and keep device/noise/domain-shift limitations explicit",
        "source_doc": "references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/source_plans/literature_matrix_v2_18_papers.md",
    },
]


def build_related_paper_comparison() -> pd.DataFrame:
    return pd.DataFrame(RELATED_PAPER_ROWS, columns=RELATED_PAPER_COLUMNS)


def related_paper_comparison_to_markdown(table: pd.DataFrame) -> str:
    if table is None or table.empty:
        raise ValueError("Related-paper comparison requires at least one row")
    table = table[RELATED_PAPER_COLUMNS].copy()
    lines = [
        "# Related-Paper Comparison",
        "",
        "This is a conservative comparison against the exact source papers captured in the original project documents. It is not a claim that our metrics exceed prior work; it explains how our reliability checks differ from headline-accuracy studies.",
        "",
        "| " + " | ".join(RELATED_PAPER_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(RELATED_PAPER_COLUMNS)) + " |",
    ]
    for _, row in table.iterrows():
        values = []
        for col in RELATED_PAPER_COLUMNS:
            text = "" if pd.isna(row[col]) else str(row[col])
            values.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def write_related_paper_comparison(csv_output: Path, markdown_output: Path) -> pd.DataFrame:
    table = build_related_paper_comparison()
    csv_output = Path(csv_output)
    markdown_output = Path(markdown_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_output, index=False)
    markdown_output.write_text(related_paper_comparison_to_markdown(table), encoding="utf-8")
    return table
