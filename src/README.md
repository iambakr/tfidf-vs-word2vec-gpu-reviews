# Analysis Code (R)

The full analysis was conducted in **R**. Scripts in this directory cover the thesis pipeline:

1. **Preprocessing** — cleaning, tokenization, stop-word removal, lemmatization, domain
   lexical correction (`quanteda`, `stopwords`, `textstem`)
2. **Feature engineering** — TF-IDF with MI / mRMR feature selection; Word2Vec Skip-gram
   (window 7, negative 7, 45 epochs; 100/200/300 dims) with mean document vectors
3. **Modeling** — SVM (Linear / RBF), Random Forest, XGBoost via `tidymodels` / `caret`,
   with randomized hyperparameter search and undersampling
4. **Evaluation** — recall-first metrics, 30 × 10-fold CV, Friedman + Dunn's tests
   (`PMCMRplus`)

See [`METHODOLOGY.md`](../METHODOLOGY.md) for full technical details and
[`README.md`](../README.md) for the package list.
