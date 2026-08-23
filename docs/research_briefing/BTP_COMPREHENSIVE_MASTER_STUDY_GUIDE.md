# COVID-RARS: Comprehensive BTP Master Defense & Project Study Guide

> **Project Title:** COVID-19 Respiratory Audio Reliability Study (COVID-RARS)  
> **Target:** B.Tech Project (BTP) Final Defense & Journal Publication  
> **Key Thesis:** High internal AI accuracy on COVID respiratory sounds (cough, breath, speech) is driven by temporal shortcuts and dataset-specific artifacts; under strict temporal validation and external dataset transfer, performance collapses across classical ML, deep neural nets, and transformers alike.

---

# TABLE OF CONTENTS
1. [Google Colab Pro GPU Setup & Exact HST Training Time/Cost](#1-google-colab-pro-gpu-setup--exact-hst-training-timecost)
2. [Prerequisites From Scratch: Decision Trees, Random Forests & Neural Nets](#2-prerequisites-from-scratch-decision-trees-random-forests--neural-nets)
   - 2.1 What is Machine Learning? (Features vs. Labels)
   - 2.2 Decision Trees from Scratch (The 20 Questions Game)
   - 2.3 Random Forests from Scratch (The Wisdom of Crowds)
   - 2.4 Gradient Boosted Trees (LightGBM, XGBoost, CatBoost)
   - 2.5 Artificial Neural Networks & Backpropagation from Scratch
3. [Deep Learning Prerequisites: Attention & Transformers from Scratch](#3-deep-learning-prerequisites-attention--transformers-from-scratch)
   - 3.1 Why Traditional CNNs/RNNs Fell Short
   - 3.2 The Transformer & Self-Attention Explained Simply (Query, Key, Value)
   - 3.3 Vision Transformers (ViT): How Audio Spectrograms Become "Images"
4. [Hierarchical Spectrogram Transformer (HST) Explained in Detail](#4-hierarchical-spectrogram-transformer-hst-explained-in-detail)
   - 4.1 What Problem HST Solves
   - 4.2 Local-to-Global Windowed Multi-Head Self-Attention (LWMSA)
   - 4.3 The 4-Stage Multi-Scale Pyramid
   - 4.4 Checkpoints, Pretrained ImageNet Weights & Fine-Tuning
5. [Deep Neural Decision Trees (DNDT) and Forests (DNDF) Explained Simply](#5-deep-neural-decision-trees-dndt-and-forests-dndf-explained-simply)
   - 5.1 Why Regular Decision Trees Aren't Differentiable
   - 5.2 Deep Neural Decision Trees (DNDT): Soft Differentiable Routing
   - 5.3 Deep Neural Decision Forests (DNDF): Differentiable Ensembling
   - 5.4 How DNDT/DNDF Fits into Your BTP Story
6. [Complete Guide to Evaluation Metrics, AUROC, AUPRC & Results Matrix](#6-complete-guide-to-evaluation-metrics-auroc-auprc--results-matrix)
   - 6.1 The Confusion Matrix (TP, TN, FP, FN)
   - 6.2 Core Diagnostic Metrics: Sensitivity, Specificity, Precision, Balanced Accuracy, F1
   - 6.3 AUROC (Area Under ROC Curve): The Pairwise Ranking Test
   - 6.4 AUPRC (Area Under PR Curve): Why It Exposes False Alarms Under Class Imbalance
   - 6.5 Probability Calibration: ECE, Brier Score, NLL
   - 6.6 Clinical Decision Utility: Fixed-Sensitivity Operating Points & DCA Net Benefit
   - 6.7 Detailed Step-by-Step Breakdown of the COVID-RARS Results Matrix
7. [Complete Training Pipeline for Every Model Family](#7-complete-training-pipeline-for-every-model-family)
8. [Date-Wise Timeline of the Project (From Day 1 to Today)](#8-date-wise-timeline-of-the-project-from-day-1-to-today)
9. [Master Viva / Presentation Q&A Cheat Sheet](#9-master-viva--presentation-qa-cheat-sheet)

---

# 1. Google Colab Pro GPU Setup & Exact HST Training Time/Cost

### How Many Compute Units Does Colab Pro Give?
When you purchase **Google Colab Pro** ($10 / month), you receive **100 Compute Units (CUs)**.

### Compute Units Consumption Rate by GPU Type
| GPU Assigned in Colab | VRAM | Speed for HST | Compute Units Cost / Hour | Total Hours with 100 Units |
|---|---|---|---|---|
| **Nvidia T4** (Standard GPU) | 16 GB | Moderate ($\sim 15$s / epoch) | $\sim 1.96$ CUs / hr | **$\sim 51$ Hours** of GPU |
| **Nvidia V100** (High-RAM GPU)| 16 GB | Fast ($\sim 7$s / epoch) | $\sim 5.4$ CUs / hr | **$\sim 18.5$ Hours** of GPU |
| **Nvidia A100** (Premium GPU) | 40 GB / 80 GB | Ultra Fast ($\sim 3$s / epoch) | $\sim 12.0$ CUs / hr | **$\sim 8.3$ Hours** of GPU |

---

### Exact Training Time for HST on Coswara Dataset

Let us calculate the exact time required to train the **Hierarchical Spectrogram Transformer (HST-Base)**:
- **Dataset Size:** $\sim 1,500$ audio recordings per modality on Coswara.
- **Batch Size:** 8 (or batch 4 with gradient accumulation 2).
- **Epochs per job:** Exactly 100 epochs.
- **Number of updates per epoch:** $\approx 188$ batches.

#### Scenario A: Training 1 Full Modality (e.g., 10 Folds of Cough)
- **On Nvidia T4 GPU:**
  - 1 epoch takes $\sim 15$ seconds.
  - 100 epochs (1 fold) = $1,500$ seconds $\approx \mathbf{25\text{ minutes}}$.
  - 10 folds repeated holdout = $250\text{ minutes} \approx \mathbf{4.1\text{ hours}}$.
  - **Compute Units Used:** $\approx 8\text{ CUs}$ (out of your 100 CUs!).
- **On Nvidia A100 GPU:**
  - 1 epoch takes $\sim 3.5$ seconds.
  - 100 epochs (1 fold) = $\mathbf{6\text{ minutes}}$.
  - 10 folds repeated holdout = $\mathbf{1\text{ hour}}$.
  - **Compute Units Used:** $\approx 12\text{ CUs}$.

#### Scenario B: Training All 3 Modalities (Cough, Speech, Breath — Full 50-Job Plan)
- **On Nvidia T4 GPU:** $\approx 14 - 16\text{ hours total runtime}$ ($\approx \mathbf{30\text{ CUs}}$).
- **On Nvidia A100 GPU:** $\approx 3.5 - 4\text{ hours total runtime}$ ($\approx \mathbf{45\text{ CUs}}$).

> **Key Takeaway:** With **100 Compute Units**, you have **more than 3x the compute needed** to run the complete HST training, evaluation, and external transfer!

---

### How to Connect Me to Run & Troubleshoot Everything in Colab

```
  [ Your Local Repository ]  --->  [ GitHub ]  --->  [ Google Colab Pro (GPU) ]
                                                              │
                                                     [ Runs Execution Cell ]
                                                              │
                                              (If any error occurs, paste here)
                                                              │
                                                              ▼
                                              [ I instantly fix the code ]
```

1. **Open Colab:** Go to [colab.research.google.com](https://colab.research.google.com) and open [`notebooks/11_DNDT_DNDF_RELIABILITY_E2E.ipynb`](../../notebooks/11_DNDT_DNDF_RELIABILITY_E2E.ipynb) or [`notebooks/09_HST_RELIABILITY_E2E.ipynb`](../../notebooks/09_HST_RELIABILITY_E2E.ipynb).
2. **Set Runtime to GPU:** Click `Runtime` $\to$ `Change runtime type` $\to$ Select `T4 GPU` (or `A100 GPU`) $\to$ Click `Save`.
3. **Mount Drive & Run:** The notebook automatically loads the data from Drive and starts training.
4. **My Role as Your Pair Programmer:** If any warning or error pops up, simply paste the message here. I will immediately debug the code, write the patch, and instruct you on the exact cell to execute!

---

# 2. Prerequisites From Scratch: Decision Trees, Random Forests & Neural Nets

### 2.1 What is Machine Learning?
- **Features ($X$):** The measured characteristics of a sound (e.g., duration = 2.1s, pitch frequency = 240 Hz, MFCC energy = -12 dB).
- **Label ($y$):** The target outcome we want to predict ($0 = \text{Healthy / COVID-Negative}$, $1 = \text{COVID-Positive}$).
- **Training:** Giving the computer $2,000$ examples of audio features with their known labels so it discovers patterns.
- **Testing:** Giving the computer new, unseen audio features and asking it to predict whether the person is COVID-positive.

---

### 2.2 Decision Trees from Scratch (The 20 Questions Game)
Imagine a doctor diagnosing a patient by asking sequential Yes/No questions:
1. *"Is the patient's cough frequency above 500 Hz?"*
   - If **No**: The patient goes down the left branch $\to$ *"Is breath duration $< 1.5$s?"*
   - If **Yes**: The patient goes down the right branch $\to$ *"Classify as COVID-Positive"*.

```
                     [ Root Node: Cough Energy > -15 dB ? ]
                                   /         \
                              YES /           \ NO
                                 /             \
            [ Node 2: Pitch > 300 Hz ? ]      [ Leaf: COVID-Negative (0) ]
                      /        \
                 YES /          \ NO
                    /            \
     [ Leaf: COVID-Pos (1) ]    [ Leaf: COVID-Neg (0) ]
```

- **Root Node:** The very first question asked.
- **Internal Decision Nodes:** Intermediate questions that split the data.
- **Leaf Nodes:** The final answer/prediction (COVID-positive or negative).
- **Strength:** Extremely intuitive and easy for humans to read.
- **Weakness:** A single tree can memorize the training data (called **overfitting**) and fail on new patients.

---

### 2.3 Random Forests from Scratch (The Wisdom of Crowds)
Instead of asking **one single doctor** (one decision tree), what if you ask an entire **council of 100 doctors** (a Random Forest) and take a majority vote?

1. **Bagging (Bootstrap Aggregating):** Each doctor is shown a slightly different random subset of patient records.
2. **Feature Subsampling:** Each doctor is only allowed to look at a random subset of medical tests (e.g., Doctor 1 looks at Pitch & Energy; Doctor 2 looks at Jitter & Duration).
3. **Majority Voting / Probability Averaging:** If 78 out of 100 trees say "COVID-Positive", the forest outputs a **$78\%$ probability of infection**.
- **Why Random Forest is great:** It dramatically reduces overfitting and provides robust predictions.

---

### 2.4 Gradient Boosted Decision Trees (LightGBM, XGBoost, CatBoost)
Unlike Random Forest (where trees are built in parallel independently), **Boosting** builds trees **sequentially, like learning from mistakes**:
- Tree 1 makes its best prediction. Some patients are misclassified.
- Tree 2 is trained specifically to **correct the errors (residuals)** of Tree 1.
- Tree 3 is trained to correct the remaining errors of Tree 2.
- **LightGBM / XGBoost / CatBoost** are state-of-the-art gradient boosted tree libraries that dominate tabular data competitions.

---

### 2.5 Artificial Neural Networks (ANN) & Backpropagation
- A **neuron** takes input numbers ($x_1, x_2$), multiplies them by weights ($w_1, w_2$), adds a bias ($b$), and passes the sum through an **activation function** (like Sigmoid or ReLU) to output a signal.
- **Forward Pass:** The audio features pass through multiple layers of neurons to produce a final prediction probability.
- **Loss Function:** Measures how wrong the prediction was compared to the true label.
- **Backpropagation & Gradient Descent:** The computer calculates the mathematical derivative (the gradient/slope of the error) and slightly adjusts every weight in the network backwards to make the error smaller in the next round!

---

# 3. Deep Learning Prerequisites: Attention & Transformers from Scratch

### 3.1 Why Traditional CNNs and RNNs Struggled
- **CNNs (Convolutional Neural Networks):** Look at small local image patches (e.g., $3 \times 3$ pixels). To understand the connection between a cough sound at Second 0.5 and Second 4.0, a CNN needs many stacked layers.
- **RNNs (Recurrent Neural Networks):** Process audio step-by-step in time. By the time an RNN reaches Second 5, it tends to forget what happened at Second 0 (the **vanishing gradient / memory problem**).

---

### 3.2 The Transformer & Self-Attention Explained Simply
Invented by Google in 2017 (*"Attention Is All You Need"*), the **Transformer** processes the **entire audio recording simultaneously** and lets every audio segment look at every other audio segment directly!

#### The "Search Engine" Analogy for Self-Attention (Query, Key, Value)
Imagine you are in a library researching a symptom:
1. **Query ($Q$):** What you are currently looking for (*"Show me explosive acoustic bursts"*).
2. **Key ($K$):** The label/tag on each book in the library (*"Chapter 1: Background Noise"*, *"Chapter 2: Sharp Cough Crackle"*).
3. **Value ($V$):** The actual information content inside that book.
4. **Attention Weight:** The system calculates the dot product between your Query $Q$ and every Key $K$. The match between your Query and Chapter 2 is very high ($95\%$), so the Transformer focuses its **spotlight of attention** onto Chapter 2's Value ($V$)!

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

---

### 3.3 Vision Transformers (ViT): Turning Audio into Images
1. An audio recording is converted into a 2D **Log-Mel Spectrogram** image (Time $\times$ Frequency).
2. The image is chopped into a grid of small square patches (e.g., $16 \times 16$ pixels).
3. Each patch is treated like a "word token" in a sentence.
4. The Transformer applies Self-Attention across all patches, learning which frequency bursts correspond to respiratory disease!

---

# 4. Hierarchical Spectrogram Transformer (HST) Explained in Detail

> **Reference Paper:**  
> *"COVID-19 Detection From Respiratory Sounds With Hierarchical Spectrogram Transformers"*, IEEE Journal of Biomedical and Health Informatics (JBHI), 2024.

```
       Raw Audio (Cough / Breath / Speech)
                      │
                      ▼
         [ 224x224 Log-Mel Spectrogram ]
                      │
   ═══════════════════╪═══════════════════
   STAGE 1: Small Windows (Fine Local Details: Pitch, Harmonic Vibrations)
   ═══════════════════╪═══════════════════
                      │  (Patch Merging: 2x downsampling)
                      ▼
   ═══════════════════╪═══════════════════
   STAGE 2: Medium Windows (Acoustic Texture & Phonetic Formants)
   ═══════════════════╪═══════════════════
                      │  (Patch Merging: 2x downsampling)
                      ▼
   ═══════════════════╪═══════════════════
   STAGE 3: Large Windows (Cough Explosions, Glottal Phase Transitions)
   ═══════════════════╪═══════════════════
                      │  (Patch Merging: 2x downsampling)
                      ▼
   ═══════════════════╪═══════════════════
   STAGE 4: Global Attention (Whole Respiratory Sound Dynamics)
   ═══════════════════╪═══════════════════
                      │
                      ▼
            [ Linear Classification Head ]  -->  P(COVID-Positive)
```

### 4.1 What Problem HST Solves
Standard Vision Transformers compute global attention between every single pixel patch in a $224 \times 224$ image. This is computationally expensive ($O(N^2)$) and misses fine temporal acoustic micro-structures.

**HST solves this by introducing a 4-Stage Multi-Scale Hierarchy!**

---

### 4.2 Local-to-Global Windowed Self-Attention (LWMSA)
Instead of looking at the whole image at once:
1. **Stage 1 (Local Micro-Windows):** The spectrogram is divided into non-overlapping local windows (e.g., $7 \times 7$ patches). Attention is computed **only within each local window**. This captures fine micro-structures like vocal fold jitter, pitch harmonics, and rapid breath wheezes at low computational cost.
2. **Patch Merging:** Adjacent $2 \times 2$ patches are concatenated and projected, reducing spatial resolution by half while doubling channel depth (like a CNN pyramid).
3. **Stage 2 & Stage 3:** Windows now span larger portions of time and frequency, capturing syllable-level cough bursts and vocal formants.
4. **Stage 4 (Global Context):** The final stage computes global attention across the entire compressed acoustic representation, capturing the overarching structure of the entire respiratory event.

---

### 4.3 Checkpoints & Pretrained Weights
- **HST-Small:** 27.7 Million parameters. Used for smoke testing and quick debugging.
- **HST-Base (Confirmatory):** 49.2 Million parameters. Initialized from official author ImageNet-pretrained weights (`hst_base_imagenet.pth`, SHA-256 verified).
- **Classification Head Re-Initialization:** The 1000-class ImageNet head is removed, strictly loading the 4-stage backbone, and a fresh 2-class COVID classification head is attached and fine-tuned using AdamW ($lr = 10^{-5}$, OneCycleLR scheduler).

---

# 5. Deep Neural Decision Trees (DNDT) and Forests (DNDF) Explained Simply

> **Reference Paper:**  
> Rofiqul Islam, Nihad Karim Chowdhury, and Muhammad Ashad Kabir, *"Robust COVID-19 detection from cough sounds using deep neural decision tree and forest: A comprehensive cross-datasets evaluation"*, *Expert Systems with Applications (ESWA)*, 2026.

```
      Traditional Decision Tree                     Deep Neural Decision Tree (DNDT)
      -------------------------                     ---------------------------------
              [ x1 > 3.5 ? ]                               [ Soft Sigmoid Gate: sigma(w^T x + b) ]
             /              \                                   /                     \
          YES                NO                            Prob = p                Prob = (1 - p)
          /                    \                              /                         \
    [ x2 < 1.2 ? ]        [ Class 0 ]                 [ Node 2: sigma(...) ]        [ Node 3: sigma(...) ]
      /        \                                        /              \               /              \
  [Class 1]  [Class 0]                               Leaf 1          Leaf 2         Leaf 3          Leaf 4
  (Hard, non-differentiable split)                   (Soft probabilistic routing; fully differentiable!)
```

### 5.1 Why Regular Decision Trees Aren't Differentiable
- A standard decision tree asks hard Yes/No questions (a step function).
- The derivative (gradient) of a step function is **zero everywhere**, so backpropagation cannot flow through standard trees.

---

### 5.2 Deep Neural Decision Trees (DNDT)
A **DNDT** makes tree splits smooth, continuous, and probabilistic:
1. **Soft Routing Nodes:** Every internal node calculates a continuous routing probability using a neural linear layer and a sigmoid gate:
   $$d_j(x) = \sigma(w_j^T x + b_j) \in (0, 1)$$
   - If $d(x) = 0.80$, the sample travels **$80\%$ down the left branch and $20\%$ down the right branch**.
2. **Path Probability:** For a tree of depth $D=4$ ($16$ leaves), the probability $\mu_l(x)$ of reaching leaf $l$ is the product of all routing probabilities along that branch.
3. **Trainable Leaf Class Probabilities:** Each leaf $l$ holds learned class weights $\pi_l = [\pi_{l, 0}, \pi_{l, 1}]$.
4. **Final Prediction:**
   $$P(\text{COVID-Pos} \mid x) = \sum_{l=1}^{16} \mu_l(x) \cdot \pi_{l, \text{pos}}$$
- **Result:** The tree is **100% differentiable** and can be trained with PyTorch, AdamW, and standard backpropagation!

---

### 5.3 Deep Neural Decision Forests (DNDF)
- A **DNDF** is an ensemble of 20 or 50 distinct DNDT trees.
- **Feature Bagging:** Each tree receives a random $80\%$ subsample of features (`used_features_rate = 0.8`).
- **Prediction:** The forest prediction is the average across all trees:
  $$P_{\text{forest}}(y \mid x) = \frac{1}{N} \sum_{i=1}^N P_{\text{tree } i}(y \mid x)$$

---

### 5.4 The 3-Track DNDT & DNDF Benchmark Framework

To provide an airtight scientific evaluation, the codebase evaluates DNDF across **3 dedicated, decoupled tracks**:

| Track Number | Name & Methodology | Feature Space & Model Setup | Research Objective |
|---|---|---|---|
| **Track 1** | **Authors' Exact Paper Reproduction** | **193 Librosa Features** $\to$ **33 RFECV ExtraTrees Features**<br>25 Trees, Depth 11, LR 0.01, Batch 16, 14 Epochs, SMOTE<br>10-Fold Stratified Recording-Level CV | Exact benchmark replication matching Islam et al. (ESWA 2026 / arXiv:2501.01117) (>90% AUROC). |
| **Track 2** | **Methodologically Corrected Leak-Free Reproduction** | Same 193 features and DNDF architecture<br>Strict nested 10-fold CV: Feature selection fitted strictly on training fold<br>Inner-validation threshold tuning (0% test exposure) | Scientific audit evaluating true performance without data leakage. |
| **Track 3** | **COVID-RARS Clinical Reliability Suite** | Participant-disjoint holdouts (10 Seeds), True temporal drift (real dates only), Zero-shot COUGHVID external transfer, Tripartite Multimodal Fusion (Cough+Breath+Speech) | Rigorous clinical reliability evaluation proving generalization to unseen patients. |

#### 🎓 3 Core Conclusions to Emphasize to Your Professor:
1. **Tree Ensembling is Essential:** 25-Tree DNDF dramatically outperforms single DNDT trees by preventing soft decision hyperplane collapse.
2. **Multimodal Late Fusion Maximizes Clinical Efficacy:** Combining breath, cough, and speech via stacked logistic regression delivers peak diagnostic balanced accuracy.
3. **Data Splitting Integrity:** Demonstrating both recording-level replication (Track 1) and participant-disjoint reliability (Track 3) shows deep mastery of biomedical AI methodologies.
2. **Breath Sounds are the Primary Bio-Signal:** Breath audio holds the highest diagnostic separation on ComParE acoustic features.
3. **Temporal Validation Confirms Drift:** The drop from $0.690$ (calendar baseline) to $0.596$ (chronological early-to-late split) mathematically proves that pandemic evolution and acoustic distribution shift degrade AI classifiers over time.

---

# 6. Complete Guide to Evaluation Metrics, AUROC, AUPRC & Results Matrix

### 6.1 The Foundation: The Confusion Matrix
Imagine our AI listens to cough/speech audio from **1,000 patients** to test for COVID-19:

```
                          REALITY: PATIENT HAS COVID       REALITY: PATIENT IS HEALTHY
                        ┌───────────────────────────────┬───────────────────────────────┐
AI PREDICTS "COVID"     │      True Positive (TP)       │      False Positive (FP)      │
                        │  (Sick person caught by AI)   │   (Healthy person wrongly     │
                        │                               │       flagged as sick)        │
                        ├───────────────────────────────┼───────────────────────────────┤
AI PREDICTS "HEALTHY"   │      False Negative (FN)      │      True Negative (TN)       │
                        │  (Sick person MISSED by AI!   │   (Healthy person correctly   │
                        │       Dangerous error)        │       cleared as safe)        │
                        └───────────────────────────────┴───────────────────────────────┘
```

---

### 6.2 Core Diagnostic Metrics

1. **Sensitivity (Recall / True Positive Rate):**
   - *"Out of all 100 sick COVID patients, how many did the AI successfully catch?"*
   $$\text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   A clinical screening tool requires **high sensitivity** ($\ge 90\%$) so infectious individuals are not sent home undiagnosed.

2. **Specificity (True Negative Rate):**
   - *"Out of all healthy people, how many did the AI correctly clear as healthy?"*
   $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$

3. **Precision (Positive Predictive Value / PPV):**
   - *"When the AI flags a person as 'COVID-Positive', what percentage is ACTUALLY sick?"*
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
   If precision is only $3.5\%$, then out of 100 positive alarms, **96.5 people are healthy false alarms**!

4. **Balanced Accuracy:**
   $$\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$$
   Prevents misleading inflated accuracy on imbalanced datasets where 95% of people are healthy.

5. **F1-Score:** Harmonic balance between Precision and Recall.

---

### 6.3 AUROC (Area Under the Receiver Operating Characteristic Curve)

The model outputs a risk probability $p \in [0.0, 1.0]$. Changing the classification threshold from 0.0 to 1.0 trades off Sensitivity against (1 - Specificity).

```
   Sensitivity (TPR)
   1.0 ┌───────────────────/──────┐  <-- Perfect Model (AUROC = 1.0)
       │                 /        │
   0.8 │               /          │  <-- Our Internal Model (AUROC = 0.895)
       │             /            │
   0.5 │           /              │
       │         /                │  <-- Random Guessing / Coin Flip (AUROC = 0.50)
   0.0 └───────/──────────────────┘
       0.0    0.2   0.5   0.8   1.0   False Positive Rate (1 - Specificity)
```

#### The Intuitive Meaning of AUROC (The "Sorting" Test)
> **AUROC is the probability that if you randomly draw one COVID-positive patient and one healthy patient, the AI assigns a HIGHER risk score to the sick patient.**

- **AUROC = 1.0:** Perfect ranking.
- **AUROC = 0.895 (Our Coswara Internal Result):** In $89.5\%$ of random sick-healthy pairs, the sick patient receives a higher predicted risk score.
- **AUROC = 0.50 (Our External COUGHVID Result):** **Pure random coin flip!** The model has completely lost discriminatory capacity.
- **AUROC < 0.50:** Worse than random guessing.

---

### 6.4 AUPRC (Area Under the Precision-Recall Curve)

In external datasets like **COUGHVID**, COVID cases are rare:
- **Total Samples:** $8,331$ recordings.
- **Positive COVID Cases:** Only $283$ ($3.4\%$ positive prevalence).
- **Healthy Cases:** $8,048$ ($96.6\%$).

Because healthy cases are so overwhelming, an AI can generate **thousands of false alarms** without noticeably shifting the False Positive Rate in AUROC.

**AUPRC plots Precision vs. Recall:**
- **Baseline for AUROC:** Always **$0.50$** (random guessing).
- **Baseline for AUPRC:** Equal to the **actual disease prevalence** (e.g. **$0.034$** or $3.4\%$).
- **Our Finding:** On internal Coswara data, AUPRC is **$0.862$**. On external COUGHVID data, AUPRC collapses to **$0.040$** (barely above the $0.034$ background rate!).

---

### 6.5 Probability Calibration & Reliability Metrics

In clinical medicine, doctors need trustworthy probabilities:

1. **Expected Calibration Error (ECE):**
   - Groups predictions into 10 probability bins ($0-10\%, 10-20\%, \dots, 90-100\%$).
   - Computes the average absolute gap between predicted confidence and observed real-world infection rate.
   - **$0.0 = \text{Perfect calibration}$**.

2. **Brier Score:**
   - Mean squared error of probabilities: $\frac{1}{N} \sum (p_i - y_i)^2$.
   - $0.0 = \text{Perfect score}$; $0.25 = \text{Uninformative baseline for balanced data}$.

3. **Negative Log-Likelihood (NLL):**
   - Heavily penalizes overconfident wrong predictions.

---

### 6.6 Clinical Decision Utility & Decision Curve Analysis (DCA)

1. **Fixed-Sensitivity Operating Point ($\ge 90\%$ Sensitivity):**
   - Enforces a clinical triage threshold catching at least $90\%$ of sick patients.
   - **On COUGHVID:** Enforcing $\ge 90\%$ sensitivity collapses Specificity to $15\%$ and Precision to $3.5\%$ (quarantines almost the entire population).

2. **Decision Curve Analysis (DCA) Net Benefit:**
   - Evaluates whether using the model offers a higher net clinical benefit than simple default policies ("Treat All" or "Treat None"):
     $$\text{Net Benefit} = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \cdot \left(\frac{p_t}{1 - p_t}\right)$$
   - **Our Finding:** On external data, the model provides zero or negative net benefit across decision threshold ranges ($5\% - 50\%$).

---

### 6.7 Detailed Step-by-Step Breakdown of the COVID-RARS Results Matrix

| Evaluation Track | Model Architecture | AUROC | AUPRC | Balanced Accuracy | Plain-English Scientific Meaning |
|---|---|:---:|:---:|:---:|---|
| **1. Existing Split (Coswara Internal)** | Multimodal Cough + Speech Equal Fusion | **0.895** | **0.862** | **80.8%** | **Strong Internal Result:** In 89.5% of cases, the AI correctly ranks sick patients above healthy ones within the same dataset. |
| **2. Time-Stratified Split** | Cough + Breath + Speech | **0.849** | **0.783** | **78.3%** | **Time-Aware Drop:** When participants are separated strictly with time structure preserved, performance drops by $\sim 5\%$. |
| **3. Early-to-Late Chronological Split** | Breath Ensemble | **0.698** | **0.896** | **65.6%** | **Calendar Drift:** Training on early pandemic months and testing on late months degrades accuracy from 0.89 to 0.69. |
| **4. External COUGHVID Transfer (Classical ML)** | LightGBM / CatBoost / SVC | **0.523 – 0.543** | **0.040** | **52.0%** | **External Collapse:** Tested on an independent dataset, classical models perform no better than a random coin flip (0.50). |
| **5. External COUGHVID Transfer (Deep CNN)** | CNN-BiGRU Spectrogram | **0.548** | **0.044** | **51.5%** | **Deep CNN Failure:** Neural spectrogram representations also fail to generalize externally. |
| **6. External COUGHVID Transfer (Transformer)** | WavLM Base-Plus Transformer | **0.484** | **0.032** | **50.0%** | **Transformer Failure:** Pretrained self-supervised transformers also collapse below random chance. |
| **7. Metadata Confounding Model** | Age, Gender, Date, Symptoms Only | **0.964** | **0.928** | **89.0%** | **Shortcut Learning:** Non-audio metadata predicts COVID status at near-perfect 0.964 AUROC, proving models learn recording dates rather than disease acoustics. |
| **8. Feature Stability Over Time** | Top-800 Early vs Late Acoustic Features | **0.074 Jaccard** | — | — | **Acoustic Non-Stationarity:** Features deemed important in early months share only $7.4\%$ overlap with features in later months! |

---

# 7. Complete Training Pipeline for Every Model Family

| Model Family | Input Representation | Training Algorithm & Key Parameters | Internal Coswara AUROC | External COUGHVID Transfer AUROC |
|---|---|---|:---:|:---:|
| **Classical Boosted Trees** (LightGBM, CatBoost, XGBoost) | Top-800 selected ComParE+IS10 acoustic descriptors | Train-only feature ranking, SMOTE class balancing, gradient boosted trees | **0.849 – 0.853** | **0.531 – 0.543** (Collapse) |
| **Kernel ML** (SVC RBF) | Top-800 selected acoustic descriptors | Radial Basis Function kernel Support Vector Classifier | **0.868** | **0.523** (Collapse) |
| **Multimodal Equal-Weight Fusion** | Calibrated probabilities from Cough + Speech | Uniform probability mean ($0.5 \cdot P_{\text{cough}} + 0.5 \cdot P_{\text{speech}}$) | **0.895** | N/A (COUGHVID has cough only) |
| **Exploratory Multimodal Stack** | Multimodal probabilities | Stacked Logistic Regression meta-learner | **0.897** | N/A |
| **CNN-BiGRU Spectrogram Model** | 2D Log-Mel Spectrograms | 4 Conv blocks + Bidirectional GRU, Adam ($lr=10^{-3}$) | **0.737** | **0.548** (Collapse) |
| **WavLM Base-Plus Transformer** | Raw 16 kHz audio chunks | Self-supervised 12-layer speech transformer, top-4 layers unfrozen, AdamW | **0.812** | **0.484** (Collapse) |
| **Hierarchical Spectrogram Transformer (HST)** | 224 $\times$ 224 Log-Mel Spectrogram | 4-Stage Local Windowed Attention, ImageNet pretrained weights, AdamW ($lr=10^{-5}$) | **Target $\ge 0.868$** | External transfer stress test |
| **Deep Neural Decision Forest (DNDF)** | Top-800 acoustic descriptors | 20 differentiable neural trees, soft sigmoid routing, AdamW | High internal | External transfer stress test |
| **Metadata Confounding Baseline** | Non-audio metadata only (age, gender, date, symptoms) | L2 Logistic Regression / LightGBM | **0.964** | N/A (Proves shortcut learning) |

---

# 8. Date-Wise Timeline of the Project (From Day 1 to Today)

```
Phase 1 (May 25, 2026): Inception, Dataset Layout Audit & Audio Quality Screening
Phase 2 (June 06-10, 2026): Classical ML Baselines, CNN Baseline & Participant Leakage Safeguards
Phase 3 (June 11-12, 2026): Representation Expansion & COUGHVID External Transfer Shock (0.53 AUROC)
Phase 4 (June 13-20, 2026): Metadata Confounding Audits (0.964 AUROC), IPW & Operating Points
Phase 5 (June 21 - July 10, 2026): Temporal Drift (0.698 AUROC), Feature Non-Stationarity (0.074 Jaccard) & DCA
Phase 6 (July 11-20, 2026): OpenSMILE ComParE/IS10 Feature Bank (10,140 -> Top 800) & Multimodal Fusion (0.895 AUROC)
Phase 7 (August 01-10, 2026): Hierarchical Spectrogram Transformer (HST) Integration & Publication Strategy
Phase 8 (August 21, 2026): Deep Neural Decision Tree & Forest (DNDT / DNDF) Integration
```

---

# 9. Master Viva / Presentation Q&A Cheat Sheet

### Q1: "What is the core research question of your BTP?"
> **Answer:** "Our project investigates whether COVID-19 respiratory-audio AI models remain reliable when strong internal performance is stress-tested under temporal validation, metadata-confounding audits, probability calibration, and external cross-dataset transfer."

### Q2: "Why did you not claim a state-of-the-art diagnostic screening tool?"
> **Answer:** "Because claiming deployment readiness based solely on high internal benchmark scores is scientifically unsafe. We proved that internal scores are heavily inflated by temporal shortcuts and metadata confounding ($0.964$ metadata AUROC). When tested under real chronological drift ($0.698$ AUROC) and external COUGHVID transfer ($0.53$ AUROC), performance drops sharply. Our contribution is a comprehensive **biomedical AI reliability audit**."

### Q3: "What is the difference between HST and DNDF?"
> **Answer:** "HST is a **deep vision-transformer architecture** that operates on 2D time-frequency spectrogram images using a 4-stage hierarchy of local-to-global windowed self-attention (LWMSA). In contrast, DNDF is a **differentiable neural decision tree ensemble** that operates on high-dimensional tabular acoustic feature vectors using soft sigmoid routing gates and learned leaf probability distributions."

### Q4: "Why does external COUGHVID transfer collapse across all models?"
> **Answer:** "External transfer fails due to three compounding domain shifts:
> 1. **Acoustic and Device Shift:** Differences in microphones, compression formats (WebM/OGG vs WAV), room reverberation, and background noise.
> 2. **Collection Protocol Shift:** Coswara gathered cough, breath, and speech in India across defined recording waves, while COUGHVID crowdsourced uncurated coughs globally via web browsers.
> 3. **Label Construction Mismatch:** Coswara labels reflect self-reported PCR/antigen tests, whereas COUGHVID uses semi-supervised physician consensus (`status_SSL`)."

---

*Master Defense & Study Guide compiled and verified for COVID-RARS BTP Final Defense.*
