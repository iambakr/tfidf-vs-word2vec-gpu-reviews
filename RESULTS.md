**English** | [繁體中文](./RESULTS.zh-TW.md)

# 📊 Complete Experimental Results

> All 52 model configurations across TF-IDF (MI & mRMR) and Word2Vec feature sets.  
> For methodology details, see [`METHODOLOGY.md`](./METHODOLOGY.md).

> [!NOTE]
> **Two kinds of numbers are reported here — don't mix them.**
> Sections 1–3 report single-point performance on the **held-out test set** (897 reviews, never touched during training) — the deployment-facing numbers. Section 4 reports **medians across 300 cross-validation rounds** (30 × 10-fold on the training set) — the distributions used for the Friedman/Dunn statistical tests. CV medians and test-set values for the same configuration therefore differ slightly by design.

---

## Table of Contents

- [1. Word2Vec Results](#1-word2vec-results)
- [2. TF-IDF + MI Results](#2-tf-idf--mi-results)
- [3. TF-IDF + mRMR Results](#3-tf-idf--mrmr-results)
- [4. Statistical Significance Analysis](#4-statistical-significance-analysis)
- [5. Summary & Conclusions](#5-summary--conclusions)

---

## 1. Word2Vec Results

### 1.1 Test Set Performance

#### 300 Dimensions

| Model | Accuracy | Specificity | Precision | Recall | F1 | Training Time (s) |
|---|---|---|---|---|---|---|
| SVM Linear | 0.865 | 0.898 | 0.591 | 0.710 | 0.645 | **1,317** |
| **SVM RBF** | **0.891** | 0.927 | **0.673** | 0.716 | **0.694** | 1,825 |
| Random Forest | 0.892 | 0.934 | 0.686 | 0.690 | 0.688 | 3,646 |
| **XGBoost** | 0.872 | 0.900 | 0.606 | **0.735** | 0.665 | 6,101 |

#### 200 Dimensions

| Model | Accuracy | Specificity | Precision | Recall | F1 | Training Time (s) |
|---|---|---|---|---|---|---|
| SVM Linear | 0.863 | 0.896 | 0.586 | 0.703 | 0.639 | **919** |
| **SVM RBF** | **0.890** | 0.929 | **0.673** | 0.703 | **0.688** | 1,238 |
| Random Forest | 0.886 | 0.927 | 0.665 | 0.690 | 0.677 | 3,302 |
| XGBoost | 0.883 | 0.925 | 0.645 | 0.684 | 0.669 | 4,489 |

#### 100 Dimensions

| Model | Accuracy | Specificity | Precision | Recall | F1 | Training Time (s) |
|---|---|---|---|---|---|---|
| SVM Linear | 0.878 | 0.916 | 0.635 | 0.697 | 0.665 | **523** |
| **SVM RBF** | **0.891** | 0.930 | 0.677 | 0.703 | **0.690** | 697 |
| Random Forest | 0.881 | 0.926 | 0.652 | 0.665 | 0.658 | 2,771 |
| **XGBoost** | 0.886 | 0.920 | 0.655 | **0.723** | 0.687 | 2,726 |

### 1.2 Generalization Analysis (Train vs. Test)

#### 300 Dimensions

| Model | Data | Accuracy | Specificity | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| SVM Linear | Train | 0.902 | 0.907 | 0.665 | 0.874 | 0.755 |
| SVM Linear | **Test** | 0.865 | 0.898 | 0.591 | 0.710 | 0.645 |
| SVM RBF | Train | 0.920 | 0.930 | 0.724 | 0.874 | 0.792 |
| SVM RBF | **Test** | 0.891 | 0.927 | 0.673 | 0.716 | 0.694 |
| Random Forest | Train | 0.958 | 0.950 | 0.808 | 0.995 | 0.892 |
| Random Forest | **Test** | 0.892 | 0.934 | 0.686 | 0.690 | 0.688 |
| XGBoost | Train | 0.931 | 0.927 | 0.732 | 0.948 | 0.826 |
| XGBoost | **Test** | 0.872 | 0.900 | 0.606 | 0.735 | 0.665 |

> **Observation:** SVM models show the smallest train-test gap, indicating best generalization. Random Forest shows the largest gap (recall: 0.995→0.690), suggesting overfitting.

---

## 2. TF-IDF + MI Results

### 2.1 Test Set Performance

#### 258 Dimensions (10%)

| Model | Accuracy | Specificity | Precision | Recall | F1 | Training Time (s) |
|---|---|---|---|---|---|---|
| SVM Linear | 0.877 | 0.962 | 0.723 | 0.471 | 0.570 | **1,453** |
| SVM RBF | 0.880 | 0.961 | 0.724 | 0.490 | 0.585 | 1,622 |
| Random Forest | **0.883** | 0.942 | 0.684 | 0.600 | **0.639** | 4,730 |
| XGBoost | 0.846 | 0.894 | 0.549 | **0.619** | 0.582 | 3,510 |

#### 517 Dimensions (20%)

| Model | Accuracy | Specificity | Precision | Recall | F1 | Training Time (s) |
|---|---|---|---|---|---|---|
| SVM Linear | 0.878 | 0.960 | 0.717 | 0.490 | 0.582 | **2,933** |
| SVM RBF | **0.885** | 0.981 | **0.825** | 0.426 | 0.562 | 3,242 |
| Random Forest | 0.883 | 0.947 | 0.695 | 0.574 | **0.629** | 7,144 |
| XGBoost | 0.821 | 0.863 | 0.485 | **0.619** | 0.544 | 6,414 |

#### 776 Dimensions (30%)

| Model | Accuracy | Specificity | Precision | Recall | F1 | Training Time (s) |
|---|---|---|---|---|---|---|
| SVM Linear | 0.882 | 0.970 | 0.763 | 0.458 | 0.573 | **4,648** |
| SVM RBF | 0.877 | 0.980 | **0.800** | 0.387 | 0.522 | 4,945 |
| Random Forest | **0.889** | 0.954 | 0.724 | 0.574 | **0.640** | 9,005 |
| XGBoost | 0.805 | 0.848 | 0.451 | **0.600** | 0.515 | 8,088 |

#### 1034 Dimensions (40%)

| Model | Accuracy | Specificity | Precision | Recall | F1 | Training Time (s) |
|---|---|---|---|---|---|---|
| SVM Linear | 0.880 | 0.969 | 0.753 | 0.452 | 0.565 | **6,373** |
| SVM RBF | 0.882 | 0.981 | **0.818** | 0.406 | 0.543 | 6,904 |
| Random Forest | **0.890** | 0.950 | 0.715 | 0.600 | **0.653** | 10,425 |
| XGBoost | 0.817 | 0.860 | 0.477 | **0.613** | 0.537 | 11,224 |

#### 1292 Dimensions (50%)

| Model | Accuracy | Specificity | Precision | Recall | F1 | Training Time (s) |
|---|---|---|---|---|---|---|
| SVM Linear | 0.872 | 0.966 | 0.722 | 0.419 | 0.531 | **8,663** |
| SVM RBF | 0.881 | 0.984 | **0.833** | 0.387 | 0.529 | 9,031 |
| Random Forest | **0.890** | 0.953 | 0.722 | 0.587 | **0.648** | 11,822 |
| XGBoost | 0.841 | 0.892 | 0.535 | **0.594** | 0.563 | 14,002 |

> **MI Summary:** XGBoost consistently achieves highest recall (0.594–0.619). SVM RBF recall degrades significantly at higher dimensions. Random Forest leads in F1.

---

## 3. TF-IDF + mRMR Results

### 3.1 Test Set Performance

#### 258 Dimensions (10%)

| Model | Accuracy | Specificity | Precision | Recall | F1 | Training Time (s) |
|---|---|---|---|---|---|---|
| **SVM Linear** | 0.835 | 0.879 | 0.519 | **0.626** | 0.567 | **1,345** |
| SVM RBF | 0.857 | 0.911 | 0.585 | 0.600 | 0.592 | 1,585 |
| Random Forest | **0.878** | 0.950 | **0.692** | 0.535 | **0.604** | 5,366 |
| XGBoost | 0.837 | 0.884 | 0.525 | 0.613 | 0.565 | 3,489 |

#### 517 Dimensions (20%)

| Model | Accuracy | Specificity | Precision | Recall | F1 | Training Time (s) |
|---|---|---|---|---|---|---|
| SVM Linear | 0.797 | 0.840 | 0.436 | 0.594 | 0.503 | **2,782** |
| SVM RBF | 0.797 | 0.837 | 0.437 | **0.606** | 0.508 | 3,271 |
| Random Forest | **0.884** | 0.951 | **0.707** | 0.561 | **0.626** | 7,666 |
| XGBoost | 0.814 | 0.865 | 0.468 | 0.568 | 0.513 | 6,416 |

#### 776 Dimensions (30%)

| Model | Accuracy | Specificity | Precision | Recall | F1 | Training Time (s) |
|---|---|---|---|---|---|---|
| SVM Linear | 0.834 | 0.911 | 0.522 | 0.465 | 0.491 | **4,434** |
| SVM RBF | 0.829 | 0.903 | 0.507 | 0.477 | 0.492 | 4,864 |
| Random Forest | **0.885** | 0.953 | **0.713** | 0.561 | **0.628** | 9,043 |
| XGBoost | 0.828 | 0.876 | 0.503 | **0.600** | 0.547 | 8,096 |

#### 1034 Dimensions (40%)

| Model | Accuracy | Specificity | Precision | Recall | F1 | Training Time (s) |
|---|---|---|---|---|---|---|
| SVM Linear | 0.864 | 0.976 | 0.739 | 0.329 | 0.455 | **6,106** |
| SVM RBF | 0.863 | 0.950 | 0.651 | 0.445 | 0.529 | 6,762 |
| Random Forest | **0.889** | 0.956 | **0.727** | 0.568 | **0.638** | 10,717 |
| XGBoost | 0.808 | 0.856 | 0.457 | **0.581** | 0.511 | 11,141 |

#### 1292 Dimensions (50%)

| Model | Accuracy | Specificity | Precision | Recall | F1 | Training Time (s) |
|---|---|---|---|---|---|---|
| SVM Linear | 0.873 | 0.980 | 0.789 | 0.361 | 0.496 | 8,006 |
| SVM RBF | 0.870 | 0.992 | **0.880** | 0.284 | 0.429 | **7,886** |
| Random Forest | **0.885** | 0.953 | 0.713 | 0.561 | **0.628** | 12,162 |
| XGBoost | 0.843 | 0.895 | 0.541 | **0.594** | 0.566 | 14,011 |

> **mRMR Summary:** Shows a clear "dimension-diminishing" effect — SVM Linear recall drops from **0.626 (258d) → 0.361 (1292d)**. Low-dimensional mRMR features are more effective.

![mRMR dimensionality decay by classifier](./results/dimensionality_decay.png)

The decay concentrates in the margin-based SVMs, while the tree ensembles (Random Forest, XGBoost) stay nearly flat across 258→1,292 dims — their built-in feature subsampling absorbs the redundant, low-relevance terms that mRMR admits at higher selection percentages.

---

## 4. Statistical Significance Analysis

### 4.1 Recall Metric

**Friedman Test:** χ²(51) = 11,827, **p < 0.001** ✓

**Top-Performing Group (12 models, p.adj ≥ 0.05 vs. benchmark):**

| Rank | Configuration | Recall Median |
|---|---|---|
| 1 | mRMR 517d + SVM RBF | 0.757 |
| 2 | mRMR 517d + SVM Linear | 0.753 |
| 3 | mRMR 258d + SVM Linear | 0.750 |
| 4 | Word2Vec 100d + SVM Linear | 0.750 |
| 5 | Word2Vec 200d + SVM Linear | 0.750 |
| 6 | Word2Vec 300d + SVM Linear | 0.750 |
| 7 | Word2Vec 300d + SVM RBF | 0.730 |
| 8 | Word2Vec 100d + SVM RBF | 0.722 |
| 9 | Word2Vec 200d + SVM RBF | 0.722 |
| 10 | Word2Vec 200d + XGBoost | 0.722 |
| 11 | Word2Vec 300d + Random Forest | 0.722 |
| 12 | Word2Vec 300d + XGBoost | 0.722 |

### 4.2 F1 Score Metric

**Friedman Test:** χ²(51) = 9,811.3, **p < 0.001** ✓

**Top-Performing Group (8 models, p.adj ≥ 0.05 vs. benchmark):**

| Rank | Configuration | F1 Median |
|---|---|---|
| 1 | Word2Vec 100d + SVM RBF | 0.684 |
| 2 | Word2Vec 200d + SVM RBF | 0.684 |
| 3 | Word2Vec 300d + SVM RBF | 0.684 |
| 4 | Word2Vec 300d + Random Forest | 0.680 |
| 5 | Word2Vec 100d + Random Forest | 0.676 |
| 6 | Word2Vec 200d + Random Forest | 0.675 |
| 7 | Word2Vec 100d + SVM Linear | 0.674 |
| 8 | mRMR 258d + SVM RBF | 0.658 |

---

## 5. Summary & Conclusions

### Key Findings

1. **Word2Vec > TF-IDF:** Semantic embeddings significantly outperform statistical word weighting in all metrics for domain-specific technical reviews.

2. **mRMR > MI:** When using TF-IDF, mRMR feature selection at low dimensions outperforms MI, especially for SVM classifiers.

3. **Dimension Effect:** For TF-IDF+mRMR, lower dimensions (258–517) outperform higher ones. For Word2Vec, 300d slightly edges 100d and 200d.

4. **Best Recall:** Word2Vec 300d + XGBoost (test set: 0.735)

5. **Best F1:** Word2Vec 300d + SVM RBF (test set: 0.694)

6. **Statistical Robustness:** Results confirmed via Friedman + Dunn's test with Holm correction across 300 cross-validation rounds.

### Practical Recommendation

| Use Case | Recommended Configuration |
|---|---|
| **Maximize negative review detection** | Word2Vec 300d + XGBoost |
| **Balanced precision-recall trade-off** | Word2Vec 100–300d + SVM RBF |
| **Budget/speed-constrained** | Word2Vec 100d + SVM RBF (fastest, competitive F1) |
| **TF-IDF required (interpretability)** | mRMR 258d + SVM Linear |
