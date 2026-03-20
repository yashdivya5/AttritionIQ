import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from preprocess import load_and_preprocess

X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(
    "../data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# Class imbalance ratio
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
ratio = neg / pos
print(f"Class ratio (scale_pos_weight): {ratio:.2f}")

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Use 0.35 threshold instead of default 0.5
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.35).astype(int)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "../model/xgb_model.pkl")
print("\nModel saved!")