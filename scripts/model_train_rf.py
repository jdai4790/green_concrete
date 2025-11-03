import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
DATA_PATH = r"C:\Users\Jeffrey Dai\OneDrive - The University of Sydney (Students)\CIVL4022 Thesis\green_concrete\data\dataset.xlsx"
SHEET = "Sheet1"
TARGET = "Cylinder compressive strength (MPa)"
OTHER_OUTPUTS = ['Splitting tensile strength (MPa)', 'Flexural strength (MPa)', 'Embodied CO₂ (kg CO₂e/m³)', 'Cost ($/kg)']

def main():
    # --- Load dataset ---
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in [TARGET] + OTHER_OUTPUTS]

    X = df[feature_cols]
    y = df[TARGET]

    # --- Split data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Preprocessing ---
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), feature_cols),
        ],
        remainder="drop",
    )

    # --- Model definition ---
    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("prep", preprocess),
        ("model", model)
    ])

    # --- Fit model ---
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    # --- Evaluate performance ---
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"Features used: {len(feature_cols)}")
    print(f"MAE (MPa): {mae:.3f}")
    print(f"RMSE (MPa): {rmse:.3f}")
    print(f"R^2: {r2:.3f}")

    # ---------- SHAP ANALYSIS ----------
    print("\nGenerating SHAP feature importance...")

    trained_model = pipe.named_steps["model"]
    X_imputed = pipe.named_steps["prep"].transform(X)

    explainer = shap.TreeExplainer(trained_model)
    shap_values = explainer.shap_values(X_imputed)

    # --- SHAP Summary Plot ---
    plt.figure()
    shap.summary_plot(shap_values, pd.DataFrame(X_imputed, columns=feature_cols), show=False)
    plt.title("SHAP Feature Importance – Random Forest (Compressive Strength)")
    plt.tight_layout()
    plt.savefig("shap_summary_random_forest.png", dpi=300)
    plt.close()

    # --- SHAP Bar Plot ---
    plt.figure()
    shap.summary_plot(shap_values, pd.DataFrame(X_imputed, columns=feature_cols), plot_type="bar", show=False)
    plt.title("Mean |SHAP| Feature Importance – Random Forest")
    plt.tight_layout()
    plt.savefig("shap_bar_random_forest.png", dpi=300)
    plt.close()

    print("SHAP plots saved as:")
    print(" - shap_summary_random_forest.png")
    print(" - shap_bar_random_forest.png")

if __name__ == "__main__":
    main()
