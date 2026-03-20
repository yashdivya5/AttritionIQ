import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from src.preprocess import load_and_preprocess

# ─── Train model if not exists ───
MODEL_PATH = "model/xgb_model.pkl"
DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

os.makedirs("model", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    from sklearn.metrics import f1_score
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(DATA_PATH)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg/pos, random_state=42, eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
else:
    model = joblib.load(MODEL_PATH)

_, X_test, _, _, feature_names = load_and_preprocess(DATA_PATH)

# ─── UI ───
st.set_page_config(page_title="AttritionIQ", page_icon="🧠", layout="wide")
st.title("🧠 AttritionIQ — Employee Attrition Predictor")
st.markdown("Predict whether an employee is likely to leave, with AI-powered explanations.")

tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction"])

# ─── TAB 1 ───
with tab1:
    st.subheader("Enter Employee Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 60, 30)
        monthly_income = st.slider("Monthly Income", 1000, 20000, 5000)
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        years_at_company = st.slider("Years at Company", 0, 40, 5)

    with col2:
        overtime = st.selectbox("OverTime", ["Yes", "No"])
        work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
        environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
        distance_from_home = st.slider("Distance From Home", 1, 30, 5)

    with col3:
        num_companies_worked = st.slider("Num Companies Worked", 0, 9, 2)
        years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 2)
        stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])

    if st.button("Predict"):
        sample = pd.DataFrame(columns=feature_names)
        sample.loc[0] = 0
        sample["Age"] = age
        sample["MonthlyIncome"] = monthly_income
        sample["JobSatisfaction"] = job_satisfaction
        sample["YearsAtCompany"] = years_at_company
        sample["OverTime"] = 1 if overtime == "Yes" else 0
        sample["WorkLifeBalance"] = work_life_balance
        sample["EnvironmentSatisfaction"] = environment_satisfaction
        sample["DistanceFromHome"] = distance_from_home
        sample["NumCompaniesWorked"] = num_companies_worked
        sample["YearsSinceLastPromotion"] = years_since_last_promotion
        sample["StockOptionLevel"] = stock_option_level
        sample["JobLevel"] = job_level

        prob = model.predict_proba(sample)[0][1]
        pred = int(prob >= 0.35)

        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ High Attrition Risk — {prob*100:.1f}% probability")
        else:
            st.success(f"✅ Low Attrition Risk — {prob*100:.1f}% probability")

        st.subheader("Why this prediction?")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        fig, ax = plt.subplots()
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=sample.iloc[0],
                feature_names=feature_names
            ), show=False
        )
        st.pyplot(fig)

# ─── TAB 2 ───
with tab2:
    st.subheader("Upload CSV for Batch Prediction")
    uploaded = st.file_uploader("Upload employee CSV", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        df = df_raw.copy()

        drop_cols = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours", "Attrition"]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        le = LabelEncoder()
        for col in df.select_dtypes(include="object").columns:
            df[col] = le.fit_transform(df[col])

        df = df.reindex(columns=feature_names, fill_value=0)

        probs = model.predict_proba(df)[:, 1]
        df_raw["Attrition_Probability_%"] = (probs * 100).round(1)
        df_raw["Risk"] = ["High 🔴" if p >= 35 else "Low 🟢" for p in df_raw["Attrition_Probability_%"]]

        st.dataframe(df_raw)
        st.download_button("Download Results", df_raw.to_csv(index=False), "attrition_results.csv")