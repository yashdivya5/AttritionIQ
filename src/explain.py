import joblib
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess

try:
    import shap
except Exception as exc:
    raise SystemExit(
        "Failed to import SHAP dependencies. This commonly happens when NumPy/Pandas "
        "installations are broken or mismatched.\n"
        "Fix with one of:\n"
        "  - conda install -c conda-forge numpy pandas shap\n"
        "  - pip install --force-reinstall numpy pandas shap\n"
        f"Original error: {exc}"
    ) from exc

# Load model and data
model = joblib.load("../model/xgb_model.pkl")
X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(
    "../data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot — global feature importance
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("../model/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Summary plot saved!")

# Waterfall plot — explain first test prediction
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=feature_names
    ),
    show=False
)
plt.tight_layout()
plt.savefig("../model/shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("Waterfall plot saved!")
