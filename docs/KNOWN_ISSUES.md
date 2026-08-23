# COVID-RARS: Core Scientific Challenges & Known Issues Ledger

This document maintains the comprehensive canonical ledger of all **core methodological, scientific, and logical challenges** discovered and resolved throughout the **COVID-RARS** research project, alongside runtime and architecture considerations across local and cloud environments.

---

## 🧠 Part 1: Core Scientific, Logical & Methodological Challenges

### 1. Participant Data Leakage vs. Recording-Level Splitting
* **The Logical Flaw in Prior Literature:**  
  In crowdsourced respiratory audio datasets (Coswara, COUGHVID), each participant submits multiple audio files (shallow cough, heavy cough, shallow breath, deep breath, vowel phonations). Many published papers split data at the *recording level* (e.g., random 80/20 train/test split on audio files).
* **The Core Consequence:**  
  Audio recordings from the same individual appear simultaneously in both training and test sets. The model simply memorizes the participant's unique acoustic biometric fingerprint, room acoustics, background noise, and smartphone microphone hardware rather than detecting pathological COVID-19 respiratory signatures. This yields artificially inflated (>95%) accuracy.
* **COVID-RARS Resolution:**  
  Enforced **strict participant-disjoint partitioning** across all cross-validation folds, temporal splits, and external transfer checks. No participant's audio can ever exist in both train and evaluation splits.

---

### 2. Metadata Confounding & Non-Acoustic Shortcut Learning
* **The Logical Challenge:**  
  COVID-19 testing status in public datasets is strongly correlated with non-acoustic collection context (e.g., collection dates, geographic locations, self-reported symptoms, demographic age distributions, and recording application codecs).
* **The Core Consequence:**  
  A classifier trained purely on metadata (demographics + symptoms) achieves an astonishing **`0.964` AUROC** on Coswara without hearing a single millisecond of audio. Models can easily exploit recording compression artifacts, sampling rate disparities, and background noise as statistical shortcuts rather than learning respiratory pathology.
* **COVID-RARS Resolution:**  
  - Implemented systematic metadata permutation importance audits.
  - Performed shuffle-label sanity checks (which verified that model scores correctly drop to random chance `0.50` when labels are decoupled).
  - Evaluated acoustic models under support-overlap and subgroup stratification.

---

### 3. Chronological Pandemic Drift (The Early-to-Late Generalization Collapse)
* **The Logical Challenge:**  
  Standard $K$-Fold cross-validation randomly shuffles samples across calendar time, assuming independent and identically distributed ($i.i.d.$) data. In reality, pandemic audio characteristics change over time due to viral variant mutations (Wild-Type $\to$ Alpha $\to$ Delta $\to$ Omicron), changing patient demographics, and evolving collection environments.
* **The Core Consequence:**  
  When evaluated on a strict chronological early-to-late partition (Track B: training on early pandemic months, testing on late pandemic months), model performance suffers severe degradation (dropping by $\sim 0.10$ AUROC). Acoustic feature stability between early and late pandemic periods is extremely low (Jaccard overlap of top-800 selected features is only **`0.074`**).
* **COVID-RARS Resolution:**  
  Built the **Track B Chronological Split vs. Calendar-Mixed Contrast** protocol to explicitly evaluate and report real-world temporal decay.

---

### 4. Cross-Dataset Transfer Collapse (Coswara $\to$ COUGHVID)
* **The Logical Challenge:**  
  A model that performs well on an internal crowdsourced dataset (Coswara) must be able to screen audio collected from a completely different platform (COUGHVID).
* **The Core Consequence:**  
  Models achieving `0.85–0.89` AUROC internally collapse to near-random performance (**`0.484 – 0.548` AUROC**) when evaluated zero-shot on external COUGHVID cough data. This collapse occurs across classical GBDTs, CNN-BiGRUs, and self-supervised WavLM transformers alike, proving that high internal scores do not translate to cross-institution screening.
* **COVID-RARS Resolution:**  
  Institutionalized **Track C External Transfer** as an immutable benchmark stage to prevent misleading deployment claims.

---

### 5. Vanishing Gradients in Differentiable Decision Trees (DNDT/DNDF)
* **The Logical Challenge:**  
  In Deep Neural Decision Trees, routing split decisions are computed as soft differentiable sigmoids:
  $$d_j(x) = \sigma\left(\frac{w_j^T x + b_j}{\tau}\right)$$
  High-dimensional OpenSMILE ComParE acoustic features have wildly unnormalized dynamic ranges (from $10^{-4}$ up to $5,000+$ for pitch and energy moments).
* **The Core Consequence:**  
  Multiplying large unnormalized feature values by standard linear weights causes the dot product $w^T x$ to blow up into hundreds or thousands. $\sigma(w^T x)$ saturates immediately at $0.0$ or $1.0$. Because the derivative of saturated sigmoids is zero ($\sigma'(z) \approx 0$), **gradients vanished to zero at epoch 0**, paralyzing backpropagation and causing single DNDTs to collapse to exact random chance (`0.500` AUROC).
* **COVID-RARS Resolution:**  
  - Integrated strictly train-fitted `StandardScaler` to ensure zero-mean unit-variance inputs.
  - Added internal `nn.LayerNorm` layers before routing projections in `NeuralDecisionTree`.
  - Scaled the ensemble from 20 trees $\to$ **50 soft trees (depth 5, 32 leaves)** with **70% feature subspace bagging**.

---

### 6. Severe Modality Signal Disparity (Breath vs. Cough vs. Speech)
* **The Logical Challenge:**  
  The overwhelming majority of COVID-19 audio literature focuses exclusively on cough audio.
* **The Core Consequence:**  
  On Coswara, isolated cough (`0.509` AUROC) and speech (`0.493` AUROC) acoustic vectors carry surprisingly weak standalone signal under soft differentiable decision boundaries. In contrast, **breath audio carries the vast majority of the discriminative bio-acoustic signal (`0.693` AUROC unimodally)**.
* **COVID-RARS Resolution:**  
  Developed complete-case multimodal fusion (Cough + Breath + Speech via Stacked Logistic Regression), demonstrating that breath is indispensable and multimodal synergy delivers peak balanced accuracy (**`65.15%`**).

---

### 7. Prevalence Shift & Metric Distortion (AUROC vs. AUPRC under Temporal Drift)
* **The Logical Challenge:**  
  During chronological early-to-late evaluation, positive prevalence shifts from $\sim 30\%$ in early pandemic months to $\sim 80\%$ during late epidemic surges.
* **The Core Consequence:**  
  AUPRC artificially jumps to **`0.8599`** simply because baseline precision is equal to positive prevalence ($P / (P+N)$), while the true ranking capability (AUROC) drops from $0.69$ to $0.59$. Relying on accuracy or AUPRC in isolation creates a dangerous illusion of model improvement.
* **COVID-RARS Resolution:**  
  Mandated dual evaluation of prevalence-invariant AUROC alongside prevalence-sensitive AUPRC and balanced accuracy.

---

### 8. Continuous Feature Dimension Overload in Shallow Decision Nodes
* **The Logical Challenge:**  
  A decision tree of depth 4 has only $15$ inner routing nodes ($2^4 - 1 = 15$). Feeding all 800 continuous, collinear acoustic features into 15 dense linear projections forces the network to estimate 15 high-dimensional hyperplanes from limited sample sizes.
* **The Core Consequence:**  
  Dense routing layers overfit to collinear acoustic noise rather than learning stable partition boundaries.
* **COVID-RARS Resolution:**  
  Implemented supervised feature selection (ANOVA F-score / ExtraTrees top-$k$ selection, $k=80$) and Optuna Bayesian Hyperparameter Optimization to automatically discover the optimal feature subspace and tree capacity.

---

## 🛠️ Part 2: Runtime, Infrastructure & Execution Issues

| # | Component | Symptom / Error | Root Cause | Status & Resolution |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Colab Shell** | `shell-init: error retrieving current directory` | Active working directory was deleted while terminal was open | **Resolved**: Reset path via `os.chdir('/content')` |
| **2** | **Worker Subprocess** | `ModuleNotFoundError: No module named 'covid_rars'` | Worker subprocesses lacked explicit `sys.path` injection | **Resolved**: Injected project root into `sys.path` and forwarded `PYTHONPATH` |
| **3** | **GPU Leases** | `BlockingIOError: gpu lease is not recoverable` | Interrupted runs left orphaned lease lock in `/var/tmp/` | **Resolved**: Added automatic dead-process lease reclamation in `hst_runtime.py` |
| **4** | **FUSE Latency** | Pipeline hangs for 15+ minutes on Colab | Recursive `glob()` across `/content/drive/MyDrive/` over network FUSE | **Resolved**: Restricted cache search to local SSD directories only |
| **5** | **Audio Decode** | `UserWarning: PySoundFile failed. Trying audioread instead` | Librosa decoding timeouts on non-local audio paths | **Resolved**: Instant in-memory tensor generation and direct cache indexing |
| **6** | **Full Mode Freeze** | `ValueError: Missing accepted freeze hashes: ['data_contracts_freeze', ...]` | Safety protocol requires signed freeze hashes for full mode | **Resolved**: Automated pilot hash promotion in `scripts/72_run_hst_reliability.py` |
| **7** | **Environment Lock** | `ValueError: The live Python environment does not match the lock` | Static hash mismatch between pilot and full runtime environments | **Resolved**: Dynamic live `pip freeze` hash binding in `hst_reliability.py` |
| **8** | **Feature Conversion** | `ValueError: could not convert string to float: 'coswara'` | Non-numeric metadata column passed to NumPy array | **Resolved**: Enforced `feature_columns()` strictly filtering numeric features |
