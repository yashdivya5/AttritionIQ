try:
    import pandas as pd
except Exception as exc:
    raise SystemExit(
        "Failed to import Pandas. This commonly happens when NumPy/Pandas "
        "installations are broken or mismatched.\n"
        "Fix with one of:\n"
        "  - conda install -c conda-forge numpy pandas\n"
        "  - pip install --force-reinstall numpy pandas\n"
        f"Original error: {exc}"
    ) from exc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    # Drop useless columns
    df.drop(columns=["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"], inplace=True)

    # Encode target
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # Encode categorical columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Split features and target
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, X.columns.tolist()
