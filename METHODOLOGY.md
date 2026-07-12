# 📐 Detailed Methodology

> Full technical documentation for the research pipeline.  
> For a high-level overview, see [`README.md`](./README.md).

---

## Table of Contents

- [1. Data Source & Collection](#1-data-source--collection)
- [2. Variable Engineering](#2-variable-engineering)
- [3. Text Preprocessing (NLP)](#3-text-preprocessing-nlp)
- [4. Feature Engineering](#4-feature-engineering)
- [5. Feature Selection](#5-feature-selection)
- [6. Machine Learning Models](#6-machine-learning-models)
- [7. Evaluation Metrics](#7-evaluation-metrics)
- [8. Statistical Testing Framework](#8-statistical-testing-framework)

---

## 1. Data Source & Collection

### 1.1 Platform Selection

**Amazon US** was selected as the sole data source based on:
- **39.7% market share** of US online retail (Digital Commerce 360, 2025)
- **28% of consumer electronics** sold through the platform
- Strict community guidelines ensuring review authenticity

### 1.2 Brand Selection

Three mainstream GPU brands covering ~59% global AIB market share:

| Brand | Market Share | Reviews Collected |
|---|---|---|
| GIGABYTE | 26% | 1,198 (40.08%) |
| ASUS | 19% | 1,000 (33.46%) |
| MSI | 14% | 791 (26.46%) |

### 1.3 Product Scope

- **NVIDIA RTX 40 Series** only (Oct 2022 – Apr 2025)
- RTX 30 series excluded to avoid crypto-mining bias
- RTX 50 series excluded due to insufficient review volume
- NVIDIA chosen over AMD due to **92% AIB market share** (Jon Peddie Research, 2025)

### 1.4 Label Definition

To address class imbalance, reviews were binarized:
- **BAD (Negative):** 1-3 stars → Positive class for detection
- **GOOD (Positive):** 4-5 stars → Negative class
- **Final ratio:** ~4.76:1 (GOOD:BAD)

---

## 2. Variable Engineering

### 2.1 Numerical Metadata Features (6)

| Feature | Mean | Std Dev | Description |
|---|---|---|---|
| Helpful | 2.35 | 4.93 | Community validation metric |
| Tokens | 59.22 | 77.97 | Review length (word count) |
| Images | 0.32 | 0.81 | Visual evidence provided |
| Videos | 0.03 | 0.17 | Video evidence provided |
| Avg_Sentence_Length | 17.41 | 15.81 | Linguistic complexity proxy |
| Volume | 2,125.37 | 1,007.53 | GPU physical volume (L×W×H) |

### 2.2 Categorical Metadata Features (3 → 16 via One-Hot Encoding)

- **Brand:** MSI, ASUS, GIGABYTE (3 columns)
- **Chipset:** 4060, 4060 Ti, 4070, 4070 SUPER, 4070 Ti, 4070 Ti SUPER, 4080, 4080 SUPER, 4090 (9 columns)
- **VRAM:** 8GB, 12GB, 16GB, 24GB (4 columns)

**Total non-text features:** 1 target + 6 numerical + 16 one-hot = 23 variables

### 2.3 How These Features Enter the Models (Hybrid Feature Strategy)

**All 52 experimental configurations use hybrid feature sets:** the text features (TF-IDF-selected terms or Word2Vec embeddings) are concatenated with the 22 metadata features above (6 numerical + 16 one-hot) to form the final training matrix. The dimension labels used throughout this repository (258–1,292 for TF-IDF; 100–300 for Word2Vec) refer to the **text-feature portion only**.

This is confirmed by the feature-importance analysis in the thesis (Fig. 6.2.1): metadata features such as `Tokens` (review length) and `chipset.4070_Ti` rank among the top-15 predictors of the best mRMR models.

**Leakage note.** Product attributes (Brand, Chipset, VRAM, Volume) are fixed properties known at prediction time and pose no leakage risk. `Helpful` votes, however, accumulate *after* a review is posted — a freshly scraped review starts at 0 — so in a real-time deployment this feature would be systematically lower than in the training data (see the deployment caveat in [`PAPER.md`](./PAPER.md), §7).

---

## 3. Text Preprocessing (NLP)

### Pipeline (R packages: quanteda, stopwords, textstem)

```
Raw Review Text
    │
    ▼
[1] Text Standardization & Cleaning
    → Lowercase, remove URLs/HTML/emojis/numbers/punctuation
    │
    ▼
[2] Tokenization
    → Split into individual word tokens
    │
    ▼
[3] Stop Word Removal
    → Snowball + SMART dictionaries
    → Custom domain terms: "card", "rtx", "gpu", "nvidia"
    │
    ▼
[4] Lemmatization
    → "ordered" → "order", "arrived" → "arrive"
    │
    ▼
[5] Domain-Specific Lexical Correction
    → "fp" → "fps", "temp" → "temperature", "trace" → "tracing"
    → Remove single-character tokens
    │
    ▼
[6] Empty Document Removal
    → 24 reviews removed (became empty after preprocessing)
    │
    ▼
Clean Token Sequences (2,989 reviews)
```

### Corpus Statistics
- **Total tokens:** 63,309
- **Vocabulary size:** 6,532 unique terms

---

## 4. Feature Engineering

### Approach 1: TF-IDF

**Term Frequency (TF):**

$$tf_{t,d} = \frac{n_{t,d}}{\sum_k n_{k,d}}$$

**Inverse Document Frequency (IDF):**

$$idf_{t,d} = \log\left(\frac{|D|}{1 + |\{t : t \in d\}|}\right)$$

**TF-IDF Weight:**

$$tfidf_{t,d} = tf_{t,d} \times idf_{t,d}$$

- Base vocabulary: **2,748 unique terms** (after removing terms with frequency < 2)
- Strict **fit-transform** protocol to prevent data leakage
- IDF weights computed on training set only, then applied to both sets

### Approach 2: Word2Vec (Skip-gram)

**Objective Function:**

$$P(w_{t-n}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+n} | w_t)$$

Skip-gram was chosen over CBOW for its superior handling of **rare/domain-specific terms** (e.g., "coil whine", "raytracing").

**Training Configuration:**

| Parameter | Value | Rationale |
|---|---|---|
| Architecture | Skip-gram | Better for rare words |
| Dimensions | 100 / 200 / 300 | Multi-scale comparison |
| Window size | 7 | Capture technical context |
| Min count | 2 | Filter noise terms |
| Negative sampling | 7 | Training efficiency |
| Subsampling | 0.001 | Downsample high-freq words |
| Iterations | 45 | Convergence assurance |
| Learning rate | 0.005 | Stable convergence |

**Document Vector:** Mean Word Vector aggregation per review.

---

## 5. Feature Selection

Applied to TF-IDF features only (Word2Vec dimensions are inherently low-dimensional).

### 5.1 Mutual Information (MI)

$$I(X;Y) = \sum_{x \in S_x} \sum_{y \in S_y} p(x,y) \log\left(\frac{p(x,y)}{p(x)p(y)}\right)$$

- Measures how much knowing whether a word appears reduces uncertainty about the sentiment label
- Higher MI → more discriminative feature

### 5.2 Minimum Redundancy Maximum Relevance (mRMR)

**Max-Relevance:**

$$D(S,y) = \frac{1}{|S|} \sum_{x_i \in S} I(x_i; y)$$

**Min-Redundancy:**

$$R(S) = \frac{1}{|S|^2} \sum_{x_i, x_j \in S} I(x_i; x_j)$$

- Balances feature relevance with inter-feature redundancy
- Produces more diverse, efficient feature subsets than MI alone

### Dimension Configurations

Features selected at **10%, 20%, 30%, 40%, 50%** of 2,748 total TF-IDF features:

| Percentage | Dimensions |
|---|---|
| 10% | 258 |
| 20% | 517 |
| 30% | 776 |
| 40% | 1,034 |
| 50% | 1,292 |

---

## 6. Machine Learning Models

### 6.1 SVM Linear

$$f(x_i) = w_i^T x_i + b_i$$

Key hyperparameter: **C** (cost) — controls misclassification tolerance.

### 6.2 SVM RBF

$$K(x_i, x) = \exp(-\gamma \|x_i - x\|^2)$$

Key hyperparameters: **C** (cost) + **γ** (kernel width).

### 6.3 Random Forest (Breiman, 2001)

- Bootstrap sampling + random feature subsets
- Classification via **majority voting**
- Key params: `ntree`, `mtry`, `min_n`

### 6.4 XGBoost (Chen & Guestrin, 2016)

$$L(\phi) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

- Sequential gradient boosting with L1/L2 regularization
- Key params: `trees`, `depth`, `learn_rate`, `mtry`, `sample_size`, `reg_alpha`, `reg_lambda`

### Hyperparameter Optimization

**Method:** Random Search (30 random configurations per model)  
**Primary metric:** Recall (for maximizing negative review detection)  
**Secondary metric:** F1 Score (tiebreaker)

| Model | Hyperparameter | Search Range |
|---|---|---|
| SVM Linear | cost (C) | 10⁻³ ~ 10² |
| SVM RBF | cost (C) | 10⁻³ ~ 10² |
| | gamma | 10⁻³ ~ 10¹ |
| Random Forest | mtry | 0.5–2.5× √features |
| | trees | 200–1000 |
| | min_n | 1–21 |
| XGBoost | mtry | 0.5–0.9 × total features |
| | trees | 100–1000 |
| | depth | 3–8 |
| | learn_rate | 0.01–0.3 |
| | sample_size | 0.5–1.0 |
| | min_n | 1–10 |
| | loss_reduction | 0.0–0.2 |
| | reg_alpha | 0.0–1.0 |
| | reg_lambda | 0.0–1.0 |

### Cross-Validation Strategy

- **10-fold stratified cross-validation**
- **30 repetitions** → 300 evaluation rounds per model
- Stratification preserves BAD:GOOD ratio in each fold

### Computing Environment (Thesis Table 5.4)

| Component | Specification |
|---|---|
| Machine | Mac Mini M4 Pro (2024) |
| CPU | 12-core (8 performance + 4 efficiency cores) |
| RAM | 48 GB |
| GPU | 16-core |
| OS | macOS Sequoia 15.6 |
| Language | R 4.5.0 |
| Key packages | quanteda · recipes 1.3.1 · tidymodels 1.3.0 |

All training-time figures reported in [`RESULTS.md`](./RESULTS.md) were measured in this environment. The full `sessionInfo()` output will accompany the analysis scripts in `src/`.

---

## 7. Evaluation Metrics

### Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP (True Positive) | FN (False Negative) |
| **Actual Negative** | FP (False Positive) | TN (True Negative) |

### Metrics

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1 Score} = \frac{2 \times Precision \times Recall}{Precision + Recall}$$

**Recall is the primary metric** — in this business context, missing a critical negative review (FN) is more costly than flagging a false alarm (FP).

---

## 8. Statistical Testing Framework

### Why Non-Parametric Tests?

- Performance distributions may not be normally distributed
- Paired repeated measures design (same data splits across all models)
- Recommended by Demšar (2006) for multi-classifier comparison

### Testing Protocol

```
Step 1: Friedman Test (Omnibus)
    → Tests if ANY significant difference exists among 52 models
    → H₀: All models perform equally
    │
    ▼ (if p < 0.05)
Step 2: Select Benchmark Model
    → Model with highest median Recall/F1 across 300 CV rounds
    │
    ▼
Step 3: Dunn's Post-hoc Test
    → Pairwise comparison: benchmark vs. all others
    → Holm p-value correction for multiple comparisons
    │
    ▼
Step 4: Define Top-Performing Group
    → All models where adjusted p ≥ 0.05 (not significantly different from benchmark)
```

### Results Summary

**Recall Metric:**
- Friedman χ²(51) = 11,827, **p < 0.001** ✓
- Top group: **12 models** (dominated by Word2Vec configurations)

**F1 Score Metric:**
- Friedman χ²(51) = 9,811.3, **p < 0.001** ✓
- Top group: **8 models** (exclusively Word2Vec-based, except mRMR 258d + SVM RBF)

---

## References

- Mikolov, T. et al. (2013). Efficient estimation of word representations in vector space.
- Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
- Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
- Peng, H. et al. (2005). Feature selection based on MI criteria of mRMR.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets.
- Salton, G. & McGill, M. J. (1983). Introduction to modern information retrieval.

See the [full reference list](./洪凱迪_碩士論文_0820.docx) in the original thesis.
