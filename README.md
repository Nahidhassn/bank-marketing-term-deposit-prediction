# Bank Marketing — Term Deposit Prediction 🏦

A machine learning project that predicts whether a bank customer will subscribe to a **term deposit** following a direct marketing campaign, using a **Decision Tree Classifier**, with a full pipeline covering exploratory data analysis, hyperparameter tuning, threshold analysis, and feature importance (both impurity-based and permutation-based).

---

## 🎯 Problem Statement

Direct marketing campaigns (phone calls) are costly and time-consuming. This project builds a classification model to predict which customers are likely to **subscribe to a term deposit**, so the bank can prioritize outreach toward the most promising leads rather than contacting customers at random.

**Dataset:** [Bank Marketing Data Set](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset) — direct marketing campaigns (phone calls) of a Portuguese banking institution.

---

## 🔍 Project Workflow

| Stage | What was done |
|---|---|
| **Data Cleaning** | Checked for missing values and duplicates; identified "unknown" categories in key fields (education, job, contact, poutcome) |
| **EDA** | Analyzed deposit subscription rate by job, education, previous campaign outcome, housing/personal loans, contact type, balance, age, and number of campaign contacts |
| **Preprocessing** | `OneHotEncoder` for categorical features via `ColumnTransformer` (no scaling needed — Decision Trees are scale-invariant) |
| **Modeling** | Decision Tree Classifier — baseline → `GridSearchCV`-tuned (criterion, max depth, min samples split/leaf) |
| **Evaluation** | 5-fold `StratifiedKFold` cross-validation + held-out test set: Accuracy, Precision, Recall, F1, ROC-AUC |
| **Threshold Tuning** | Swept decision thresholds (0.30–0.70) to find the F1-optimal cutoff instead of the default 0.5 |
| **Interpretability** | Impurity-based feature importance (aggregated back to original features) **and** permutation importance for a model-agnostic cross-check |
| **Deployment-ready** | Final tuned model saved with `joblib`, wrapped in a reusable `predict_customer()` function |

---

## 📊 Key Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline Decision Tree | `0.6314` | `0.6179` | `0.5822` | `0.5995` | `0.6290` |
| Tuned Decision Tree | `0.6744` | `0.6715` | `0.6125` | `0.6406` | `0.7160` |
| Tuned Tree (Threshold = 0.30) | `0.6234` | `0.5791` | `0.7505` | `0.6538` | — |

**Best hyperparameters found (GridSearchCV):** `criterion='entropy', max_depth=None, min_samples_leaf=10, min_samples_split=2`

**Interpretation:** Tuning improved every metric over the baseline, most notably ROC-AUC (0.63 → 0.72). Lowering the decision threshold trades precision for a large recall gain (58% → 75%) — meaning the model catches significantly more potential subscribers, at the cost of contacting more customers who won't ultimately subscribe. For an outbound calling campaign, this trade-off is often worthwhile: the cost of an extra call is generally much lower than the cost of missing a real subscriber.

---

## 🗂️ Project Structure

```
bank-marketing-term-deposit-prediction/
│
├── data/
│   └── bank.csv
│
├── notebooks/
│   └── bank_marketing_analysis.ipynb
│
├── models/
│   └── bank_marketing_decision_tree.pkl
│
├── visuals/
│   ├── deposit_rate_by_job.png
│   ├── deposit_rate_by_poutcome.png
│   ├── balance_vs_deposit.png
│   ├── campaign_contacts_vs_deposit.png
│   ├── feature_importance_impurity.png
│   ├── permutation_importance.png
│   └── ... (other charts)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Nahidhassn/bank-marketing-term-deposit-prediction.git
cd bank-marketing-term-deposit-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook notebooks/bank_marketing_analysis.ipynb
```

---

## 🧠 Using the Trained Model

```python
import joblib

artifact = joblib.load("models/bank_marketing_decision_tree.pkl")
model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

probability = model.predict_proba(new_customer_df[features])[:, 1]
prediction = "yes" if probability >= threshold else "no"
```

---

## 🤔 Why Decision Tree?

A Decision Tree was chosen for its **interpretability** — it directly shows which customer attributes (job, previous campaign outcome, contact type, etc.) drive the prediction, in a format that's easy to explain to non-technical marketing stakeholders. It also handles a mix of categorical and numerical features naturally and requires no feature scaling. Both impurity-based and permutation-based feature importance are included to cross-check which features the model actually relies on, since impurity-based importance alone can be biased toward high-cardinality categorical features.

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy
- **Visualization** — matplotlib, seaborn
- **Machine Learning** — scikit-learn (Decision Tree, Pipelines, ColumnTransformer, GridSearchCV, permutation_importance)
- **Model Persistence** — joblib

---

## 📌 Future Improvements

- Compare against ensemble models (Random Forest, Gradient Boosting/XGBoost) which typically outperform a single Decision Tree
- Address class imbalance explicitly with `class_weight="balanced"` or resampling techniques
- Deploy as a simple scoring API to rank leads before each new campaign

---

## 👤 Author

**Nahid Hasan**
[Email](mailto:nahidhassan.ctc@gmail.com)

*This project was built as part of my data science portfolio to demonstrate end-to-end ML workflow: from raw data to a tuned, evaluated, and deployable classification model.*
