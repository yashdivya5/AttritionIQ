# 🧠 AttritionIQ — Employee Attrition Predictor

> Predict whether an employee is likely to leave — with AI-powered explanations using SHAP.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yashdivya-attritioniq.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.42.1-green)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red)

---

## 📌 Overview

**AttritionIQ** is a machine learning web app that predicts employee attrition risk using the IBM HR Analytics dataset. It not only predicts whether an employee is likely to quit, but also **explains why** using SHAP (SHapley Additive exPlanations) — making it interpretable for HR decision-making.

---

## 🚀 Live Demo

👉 [attritioniq-yashdivya5.streamlit.app](https://yashdivya-attritioniq.streamlit.app/)

---

## ✨ Features

- **Single Prediction** — Enter employee details via an interactive form and get instant attrition risk with probability score
- **Batch Prediction** — Upload a CSV of employees and get predictions for all of them at once, with a downloadable results file
- **SHAP Explanations** — Waterfall plot showing which features drove each individual prediction
- **Interpretable ML** — Every prediction comes with a reason, not just a number

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | XGBoost (v1.7.6) |
| Explainability | SHAP (v0.42.1) |
| Frontend | Streamlit |
| Data Processing | Pandas, Scikit-learn |
| Dataset | IBM HR Analytics (Kaggle) |
| Deployment | Streamlit Cloud |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 81.3% |
| F1 Score (Attrition class) | 0.476 |
| ROC-AUC | 0.802 |
| Recall (Attrition class) | 53% |

> Note: Class imbalance handled using `scale_pos_weight` and a tuned prediction threshold of 0.35 to maximize recall on the minority (attrition) class.

---

## 📁 Project Structure

```
AttritionIQ/
├── streamlit_app.py       # Main Streamlit app
├── src/
│   └── preprocess.py      # Data loading & preprocessing
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── model/                 # Auto-generated at runtime
│   ├── xgb_model.pkl
│   ├── shap_summary.png
│   └── shap_waterfall.png
├── requirements.txt
└── README.md
```

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/yashdivya5/AttritionIQ.git
cd AttritionIQ

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

---

## 🔍 How It Works

1. **Preprocessing** — Categorical encoding, dropping irrelevant columns, train/test split with stratification
2. **Training** — XGBoost with class imbalance handling (`scale_pos_weight = 5.19`)
3. **Threshold Tuning** — Default 0.5 threshold lowered to 0.35 to improve recall on attrition cases
4. **Explainability** — SHAP TreeExplainer generates feature-level attributions for every prediction

---

## 📈 Key Insights from SHAP

Top features driving attrition predictions:
- **OverTime** — strongest predictor of attrition
- **MonthlyIncome** — lower income → higher risk
- **YearsAtCompany** — early tenure employees leave more
- **JobSatisfaction** — lower satisfaction = higher attrition
- **StockOptionLevel** — no stock options = higher risk

---

## 👤 Author

**Yash** — Final year engineering student at NIE Mysore  
[GitHub](https://github.com/yashdivya5) • [LinkedIn](https://www.linkedin.com/in/yash-divya-ab3aa6206/)
