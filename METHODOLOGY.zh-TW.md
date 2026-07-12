[English](./METHODOLOGY.md) | **繁體中文**

# 📐 詳細方法論

> 本文件為研究流程的完整技術文件。  
> 高層次概覽請參閱 [`README.zh-TW.md`](./README.zh-TW.md)。

---

## 目錄

- [1. 資料來源與蒐集](#1-資料來源與蒐集)
- [2. 變數工程](#2-變數工程)
- [3. 文本預處理（NLP）](#3-文本預處理nlp)
- [4. 特徵工程](#4-特徵工程)
- [5. 特徵選取](#5-特徵選取)
- [6. 機器學習模型](#6-機器學習模型)
- [7. 評估指標](#7-評估指標)
- [8. 統計檢定框架](#8-統計檢定框架)

---

## 1. 資料來源與蒐集

### 1.1 平台選擇

選擇 **美國 Amazon** 作為唯一資料來源，理由如下：
- 佔美國線上零售 **39.7% 市佔率**（Digital Commerce 360, 2025）
- **28% 的消費性電子產品**透過該平台銷售
- 嚴格的社群守則，確保評論真實性

### 1.2 品牌選擇

三個主流 GPU 品牌，合計約佔全球 AIB 市佔率 59%：

| 品牌 | 市佔率 | 蒐集評論數 |
|---|---|---|
| GIGABYTE | 26% | 1,198（40.08%） |
| ASUS | 19% | 1,000（33.46%） |
| MSI | 14% | 791（26.46%） |

### 1.3 產品範圍

- 僅限 **NVIDIA RTX 40 系列**（2022 年 10 月 – 2025 年 4 月）
- 排除 RTX 30 系列，以避免加密貨幣挖礦造成的偏差
- 排除 RTX 50 系列，因評論數量不足
- 選擇 NVIDIA 而非 AMD，係因其 **92% 的 AIB 市佔率**（Jon Peddie Research, 2025）

### 1.4 標籤定義

為處理類別不平衡問題，將評論二元化：
- **BAD（負面評價）：** 1–3 星 → 偵測任務中的陽性類別（Positive class）
- **GOOD（正面評價）：** 4–5 星 → 陰性類別（Negative class）
- **最終比例：** 約 4.76:1（GOOD:BAD）

---

## 2. 變數工程

### 2.1 數值型 Metadata 特徵（6 個）

| 特徵 | 平均數 | 標準差 | 說明 |
|---|---|---|---|
| Helpful | 2.35 | 4.93 | 社群認可程度指標 |
| Tokens | 59.22 | 77.97 | 評論長度（詞數） |
| Images | 0.32 | 0.81 | 提供的圖片佐證數 |
| Videos | 0.03 | 0.17 | 提供的影片佐證數 |
| Avg_Sentence_Length | 17.41 | 15.81 | 語言複雜度的代理指標 |
| Volume | 2,125.37 | 1,007.53 | GPU 實體體積（長×寬×高） |

### 2.2 類別型 Metadata 特徵（3 個 → 經 One-Hot 編碼展開為 16 欄）

- **Brand（品牌）：** MSI、ASUS、GIGABYTE（3 欄）
- **Chipset（晶片組）：** 4060、4060 Ti、4070、4070 SUPER、4070 Ti、4070 Ti SUPER、4080、4080 SUPER、4090（9 欄）
- **VRAM（顯示記憶體）：** 8GB、12GB、16GB、24GB（4 欄）

**非文本特徵總計：** 1 個目標變數 + 6 個數值型 + 16 個 one-hot = 23 個變數

### 2.3 這些特徵如何進入模型（混合特徵策略）

**全部 52 組實驗組態皆使用混合特徵集：** 文本特徵（TF-IDF 選出的詞彙特徵或 Word2Vec 嵌入向量）會與上述 22 個 metadata 特徵（6 個數值型 + 16 個 one-hot）串接，構成最終的訓練矩陣。本 repository 通篇使用的維度標示（TF-IDF 的 258–1,292；Word2Vec 的 100–300）**僅指文本特徵的部分**。

此點可由論文中的特徵重要性分析（圖 6.2.1）證實：`Tokens`（評論長度）與 `chipset.4070_Ti` 等 metadata 特徵，名列最佳 mRMR 模型前 15 大預測因子之中。

**資料洩漏（data leakage）說明。** 產品屬性（Brand、Chipset、VRAM、Volume）是預測當下即已知的固定屬性，不構成洩漏風險。然而 `Helpful` 票數是在評論發佈*之後*才逐漸累積的——剛爬取下來的新評論起始值為 0——因此在即時部署情境中，此特徵的數值會系統性地低於訓練資料（參見 [`PAPER.zh-TW.md`](./PAPER.zh-TW.md) §7 的部署注意事項）。

---

## 3. 文本預處理（NLP）

### 處理流程（R 套件：quanteda、stopwords、textstem）

```
原始評論文本
    │
    ▼
[1] 文本標準化與清理
    → 轉為小寫，移除 URL／HTML 標籤／emoji／數字／標點符號
    │
    ▼
[2] 斷詞（Tokenization）
    → 切分為獨立的詞彙單元（token）
    │
    ▼
[3] 停用詞移除
    → Snowball + SMART 停用詞辭典
    → 自訂領域詞彙："card"、"rtx"、"gpu"、"nvidia"
    │
    ▼
[4] 詞形還原（Lemmatization）
    → "ordered" → "order"、"arrived" → "arrive"
    │
    ▼
[5] 領域特定詞彙校正
    → "fp" → "fps"、"temp" → "temperature"、"trace" → "tracing"
    → 移除單一字元的 token
    │
    ▼
[6] 空文件移除
    → 移除 24 篇評論（預處理後內容變為空白）
    │
    ▼
乾淨的 token 序列（2,989 篇評論）
```

### 語料庫統計
- **總 token 數：** 63,309
- **詞彙量：** 6,532 個獨特詞彙

---

## 4. 特徵工程

### 方法一：TF-IDF

**詞頻（Term Frequency, TF）：**

$$tf_{t,d} = \frac{n_{t,d}}{\sum_k n_{k,d}}$$

**逆文件頻率（Inverse Document Frequency, IDF）：**

$$idf_{t,d} = \log\left(\frac{|D|}{1 + |\{t : t \in d\}|}\right)$$

**TF-IDF 權重：**

$$tfidf_{t,d} = tf_{t,d} \times idf_{t,d}$$

- 基礎詞彙庫：**2,748 個獨特詞彙**（移除詞頻 < 2 的詞彙後）
- 採用嚴格的 **fit-transform** 流程，以防止資料洩漏（data leakage）
- IDF 權重僅以訓練集計算，再同時套用至訓練集與測試集

### 方法二：Word2Vec（Skip-gram）

**目標函數：**

$$P(w_{t-n}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+n} | w_t)$$

本研究選擇 Skip-gram 而非 CBOW，因其對**罕見詞彙／領域專門術語**（如 "coil whine"、"raytracing"）有較佳的處理能力。

**訓練參數設定：**

| 參數 | 數值 | 理由 |
|---|---|---|
| 架構 | Skip-gram | 對罕見詞彙表現較佳 |
| 維度 | 100 / 200 / 300 | 多尺度比較 |
| Window size | 7 | 捕捉技術性上下文 |
| Min count | 2 | 過濾雜訊詞彙 |
| Negative sampling | 7 | 提升訓練效率 |
| Subsampling | 0.001 | 高頻詞下採樣 |
| 迭代次數 | 45 | 確保收斂 |
| Learning rate | 0.005 | 穩定收斂 |

**文件向量（Document Vector）：** 以平均詞向量法（Mean Word Vector）聚合每篇評論的詞向量。

---

## 5. 特徵選取

僅套用於 TF-IDF 特徵（Word2Vec 的維度本身即為低維度）。

### 5.1 互資訊（Mutual Information, MI）

$$I(X;Y) = \sum_{x \in S_x} \sum_{y \in S_y} p(x,y) \log\left(\frac{p(x,y)}{p(x)p(y)}\right)$$

- 衡量「得知某詞彙是否出現」能降低多少對情感標籤的不確定性
- MI 越高 → 特徵的鑑別力越強

### 5.2 最小冗餘最大相關性（Minimal Redundancy Maximal Relevance, mRMR）

**最大相關（Max-Relevance）：**

$$D(S,y) = \frac{1}{|S|} \sum_{x_i \in S} I(x_i; y)$$

**最小冗餘（Min-Redundancy）：**

$$R(S) = \frac{1}{|S|^2} \sum_{x_i, x_j \in S} I(x_i; x_j)$$

- 在特徵與標籤的相關性、以及特徵彼此間的冗餘度之間取得平衡
- 相較於僅使用 MI，可產生更多元、更有效率的特徵子集

### 維度配置

以 2,748 個 TF-IDF 特徵總數的 **10%、20%、30%、40%、50%** 進行特徵選取：

| 百分比 | 維度 |
|---|---|
| 10% | 258 |
| 20% | 517 |
| 30% | 776 |
| 40% | 1,034 |
| 50% | 1,292 |

---

## 6. 機器學習模型

### 6.1 SVM Linear

$$f(x_i) = w_i^T x_i + b_i$$

關鍵超參數：**C**（cost）——控制對誤分類的容忍度。

### 6.2 SVM RBF

$$K(x_i, x) = \exp(-\gamma \|x_i - x\|^2)$$

關鍵超參數：**C**（cost）+ **γ**（kernel 寬度）。

### 6.3 Random Forest（Breiman, 2001）

- Bootstrap 抽樣 + 隨機特徵子集
- 以**多數決投票（majority voting）**進行分類
- 關鍵參數：`ntree`、`mtry`、`min_n`

### 6.4 XGBoost（Chen & Guestrin, 2016）

$$L(\phi) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

- 具 L1/L2 正規化的循序梯度提升（gradient boosting）
- 關鍵參數：`trees`、`depth`、`learn_rate`、`mtry`、`sample_size`、`reg_alpha`、`reg_lambda`

### 超參數優化

**方法：** 隨機搜尋（Random Search）——每個模型隨機抽樣 30 組參數組合  
**主要指標：** 召回率（Recall）（以最大化負面評論偵測能力）  
**次要指標：** F1 分數（F1 Score）（表現並列時的決勝指標）

| 模型 | 超參數 | 搜尋範圍 |
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

### 交叉驗證策略

- **10 摺分層交叉驗證（10-fold stratified cross-validation）**
- **重複 30 次** → 每個模型共 300 輪評估
- 「分層」機制確保每一摺中 BAD:GOOD 的比例與整體一致

### 實驗環境（論文表 5.4）

| 元件 | 規格 |
|---|---|
| 機器 | Mac Mini M4 Pro (2024) |
| CPU | 12-core (8 performance + 4 efficiency cores) |
| RAM | 48 GB |
| GPU | 16-core |
| 作業系統 | macOS Sequoia 15.6 |
| 語言 | R 4.5.0 |
| 主要套件 | quanteda · recipes 1.3.1 · tidymodels 1.3.0 |

[`RESULTS.zh-TW.md`](./RESULTS.zh-TW.md) 中回報的所有訓練時間數據，均在此環境下量測。完整的 `sessionInfo()` 輸出將隨 `src/` 中的分析腳本一併提供。

---

## 7. 評估指標

### 混淆矩陣（Confusion Matrix）

|  | 預測為陽性 | 預測為陰性 |
|---|---|---|
| **實際為陽性** | TP（真陽性，True Positive） | FN（偽陰性，False Negative） |
| **實際為陰性** | FP（偽陽性，False Positive） | TN（真陰性，True Negative） |

### 指標

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1 Score} = \frac{2 \times Precision \times Recall}{Precision + Recall}$$

上列依序為準確率（Accuracy）、精確率（Precision）、召回率（Recall）與 F1 分數（F1 Score）。**召回率為主要指標**——在本研究的商業情境中，漏掉一則關鍵負面評論（FN）的代價，遠高於誤報一次假警報（FP）。

---

## 8. 統計檢定框架

### 為何採用無母數（Non-Parametric）檢定？

- 效能分布未必符合常態分布
- 成對重複量測設計（所有模型使用完全相同的資料切分）
- Demšar（2006）建議用於多分類器比較

### 檢定流程

```
步驟 1：Friedman 檢定（全局檢定，omnibus test）
    → 檢驗 52 組模型之間是否存在任何顯著差異
    → H₀：所有模型效能相同
    │
    ▼（若 p < 0.05）
步驟 2：選定基準模型（benchmark model）
    → 300 輪交叉驗證中召回率／F1 中位數最高的模型
    │
    ▼
步驟 3：Dunn's 事後檢定（post-hoc test）
    → 成對比較：基準模型 vs. 其餘所有模型
    → 以 Holm 法對多重比較進行 p 值校正
    │
    ▼
步驟 4：界定效能最佳模型群組
    → 所有校正後 p ≥ 0.05（與基準模型無顯著差異）的模型
```

### 結果摘要

**召回率指標：**
- Friedman χ²(51) = 11,827，**p < 0.001** ✓
- 效能最佳群組：**12 組模型**（以 Word2Vec 組態為主）

**F1 分數指標：**
- Friedman χ²(51) = 9,811.3，**p < 0.001** ✓
- 效能最佳群組：**8 組模型**（除 mRMR 258 維 + SVM RBF 外，全部基於 Word2Vec）

---

## 參考文獻

- Mikolov, T. et al. (2013). Efficient estimation of word representations in vector space.
- Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
- Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
- Peng, H. et al. (2005). Feature selection based on MI criteria of mRMR.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets.
- Salton, G. & McGill, M. J. (1983). Introduction to modern information retrieval.

完整參考文獻列表請參閱[原始論文](./洪凱迪_碩士論文_0820.docx)。
