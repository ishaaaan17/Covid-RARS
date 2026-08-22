# DNDT / DNDF Integrated Reliability Study Design

## Purpose

This study incorporates **Deep Neural Decision Trees (DNDT)** and **Deep Neural Decision Forests (DNDF)** into the COVID-RARS reliability evaluation ladder, following the methodology and code structure inspired by:

> **Rofiqul Islam, Nihad Karim Chowdhury, and Muhammad Ashad Kabir.**  
> *"Robust COVID-19 detection from cough sounds using deep neural decision tree and forest: A comprehensive cross-datasets evaluation."*  
> *Expert Systems with Applications*, Volume 310, 2026, Article 131235.  
> DOI: [10.1016/j.eswa.2026.131235](https://doi.org/10.1016/j.eswa.2026.131235)  
> GitHub: [Rofiquldk1/COVID-19-Detection-from-Cough-Sound](https://github.com/Rofiquldk1/COVID-19-Detection-from-Cough-Sound)

The goal is to test whether differentiable neural tree ensembles improve representation capacity and whether their reported cross-dataset resilience holds when evaluated under the strict, participant-level validation ladder of COVID-RARS.

---

## 1. DNDT and DNDF Model Architecture

### Deep Neural Decision Tree (DNDT)
- A soft, differentiable binary decision tree of depth $D$.
- Has $N_{\text{inner}} = 2^D - 1$ decision nodes and $N_{\text{leaves}} = 2^D$ leaf nodes.
- Each decision node $j$ computes a soft routing decision via linear projection with temperature scaling:
  $$d_j(x) = \sigma\left(\frac{w_j^T x + b_j}{\tau}\right) \in (0, 1)$$
- Leaf routing probabilities $\mu_l(x)$ are calculated as the differentiable product of path probabilities:
  $$\mu_l(x) = \prod_{j \in \text{path}(l)} (d_j(x))^{\mathbb{I}(M_{l,j}=+1)} \cdot (1 - d_j(x))^{\mathbb{I}(M_{l,j}=-1)}$$
- Each leaf maintains trainable class logits $\theta_l \in \mathbb{R}^C$, with class distribution $\pi_l = \text{softmax}(\theta_l)$.
- Output prediction:
  $$P(y = c \mid x) = \sum_{l=1}^{N_{\text{leaves}}} \mu_l(x) \cdot \pi_{l, c}$$

### Deep Neural Decision Forest (DNDF)
- An ensemble of $N_{\text{trees}}$ distinct `NeuralDecisionTree` models.
- Uses feature bagging (`used_features_rate` $\in (0, 1]$) to sample a random feature subspace for each tree.
- Forest output is the ensemble average:
  $$P_{\text{forest}}(y = c \mid x) = \frac{1}{N_{\text{trees}}} \sum_{m=1}^{N_{\text{trees}}} P_m(y = c \mid x)$$

---

## 2. Evaluation Hierarchy (Matching HST & COVID-RARS)

1. **Track A (Literature-Aligned Repeated Holdouts):**
   - 10 repeated stratified participant-disjoint holdouts (approx 70/10/20 train/val/test) using the standard benchmark seeds `[1, 2, 5, 12, 40, 52, 72, 2002, 4002, 6002]`.
   - Evaluated on individual modalities (**Cough**, **Breath**, **Speech**) and complete-case multimodal fusion (**Cough+Speech**, **Cough+Breath+Speech**).
2. **Track B (Matched-Cohort Split-Policy Contrast):**
   - Chronological early-to-late 60/20/20 holdout vs. Calendar-mixed date-balanced baseline.
3. **Track C (COUGHVID External Transfer):**
   - Model trained on Coswara cough evaluated directly on COUGHVID cough data without target domain tuning or threshold snooping.
4. **Reliability Audits:**
   - Platt probability calibration, ECE, Brier score, and NLL.
   - Fixed-sensitivity clinical operating points ($\ge 90\%$ sensitivity).
   - Decision Curve Analysis (DCA) computing net clinical benefit.
   - 1,000-replicate clustered bootstrap confidence intervals.

---

## 3. Implementation Codebase Mapping

| Component | File Path |
|---|---|
| DNDT / DNDF PyTorch Models | [`src/covid_rars/dndf_models.py`](file:///D:/Projects/Covid-RARS/src/covid_rars/dndf_models.py) |
| Training & Participant Aggregation | [`src/covid_rars/dndf_training.py`](file:///D:/Projects/Covid-RARS/src/covid_rars/dndf_training.py) |
| Protocol Execution (Tracks A, B, C) | [`src/covid_rars/dndf_protocols.py`](file:///D:/Projects/Covid-RARS/src/covid_rars/dndf_protocols.py) |
| Multimodal & Hybrid Fusion | [`src/covid_rars/dndf_fusion.py`](file:///D:/Projects/Covid-RARS/src/covid_rars/dndf_fusion.py) |
| Reliability, DCA & Calibration | [`src/covid_rars/dndf_reliability.py`](file:///D:/Projects/Covid-RARS/src/covid_rars/dndf_reliability.py) |
| Stages & Pipeline Controller | [`src/covid_rars/dndf_stages.py`](file:///D:/Projects/Covid-RARS/src/covid_rars/dndf_stages.py) |
| Comparative Reporting | [`src/covid_rars/dndf_reporting.py`](file:///D:/Projects/Covid-RARS/src/covid_rars/dndf_reporting.py) |
| CLI Workflow Script | [`scripts/79_run_dndf_reliability.py`](file:///D:/Projects/Covid-RARS/scripts/79_run_dndf_reliability.py) |
| Evidence Pack Script | [`scripts/80_make_dndf_evidence_pack.py`](file:///D:/Projects/Covid-RARS/scripts/80_make_dndf_evidence_pack.py) |
| Configuration | [`configs/dndf_reliability.json`](file:///D:/Projects/Covid-RARS/configs/dndf_reliability.json) |
| Google Colab Notebook | [`notebooks/11_DNDT_DNDF_RELIABILITY_E2E.ipynb`](file:///D:/Projects/Covid-RARS/notebooks/11_DNDT_DNDF_RELIABILITY_E2E.ipynb) |
| Unit & Integration Tests | [`tests/test_dndf_*.py`](file:///D:/Projects/Covid-RARS/tests/) |
