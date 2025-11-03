"""
train_and_predict_linear_multi.py  (with output graphs + RMSE)

Adds scatter plots of Predicted vs Actual for each target.
Saved to ./outputs/linear_plot_<target>.png
"""

import sys, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error  # <-- added mean_squared_error

# ---------- config ----------
DATA_DIR    = Path("./data")
MODELS_DIR  = Path("./models")
OUTPUTS_DIR = Path("./outputs")

DEFAULT_EXCEL = DATA_DIR / "dataset.xlsx"
DEFAULT_SHEET = "Sheet1"

TARGETS_CANON = [
    "Cylinder compressive strength (MPa)",
    "Splitting tensile strength (MPa)",
    "Flexural strength (MPa)",
    "Elastic modulus (GPa)",
]
# ----------------------------


def ensure_dirs():
    """Ensure data, models, and outputs folders exist."""
    for d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def slugify(name: str) -> str:
    """Convert string to safe filename."""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def find_present_targets(columns):
    """Identify which known targets are present in dataset."""
    cols_lower = {c.lower(): c for c in columns}
    present = []
    for t in TARGETS_CANON:
        if t.lower() in cols_lower:
            present.append(cols_lower[t.lower()])
    return present


def train_one(df, target, feature_cols):
    """Train and evaluate one Linear Regression model."""
    X = df[feature_cols]
    y = df[target]
    
    # --- Split data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing: impute missing values + scale numeric features
    preprocess = ColumnTransformer(
        [("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), feature_cols)],
        remainder="drop"
    )

    # Linear regression model
    model = LinearRegression()

    # Combine preprocessing and model into a pipeline
    pipe = Pipeline([
        ("prep", preprocess),
        ("model", model)
    ])

    # Train the model
    pipe.fit(X_train, y_train)

    # Evaluate performance
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Save model and features
    features_path = MODELS_DIR / f"linear_{slugify(target)}_features.txt"
    with features_path.open("w", encoding="utf-8") as f:
        for c in feature_cols:
            f.write(c + "\n")

    model_path = MODELS_DIR / f"linear_{slugify(target)}_model.pkl"
    joblib.dump(pipe, model_path, compress=3)

    # Return all useful data for graphing
    return pipe, mae, rmse, r2, model_path, features_path, X_test, y_test, preds


def plot_results(y_test, preds, target):
    """Create and save a Predicted vs Actual scatter plot."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.7, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label="Ideal fit (y = x)")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Predicted vs Actual — {target}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    out_path = OUTPUTS_DIR / f"linear_plot_{slugify(target)}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot to {out_path}")


def main():
    ensure_dirs()

    # CLI args (optional)
    if len(sys.argv) >= 2:
        excel_path = Path(sys.argv[1])
    else:
        excel_path = DEFAULT_EXCEL

    sheet_name = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_SHEET

    if not excel_path.exists():
        raise FileNotFoundError(
            f"Could not find Excel file: {excel_path}\n"
            f"Tip: Put your file at {DEFAULT_EXCEL} or pass a path as an argument."
        )

    # Read dataset
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Identify present target columns
    present_targets = find_present_targets(df.columns)
    if not present_targets:
        raise RuntimeError(
            "No known target columns found. Expected one or more of:\n  - "
            + "\n  - ".join(TARGETS_CANON)
        )

    # Select numeric features (exclude target columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in present_targets]
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found after excluding targets.")

    metrics = []
    preds_df = df.copy()

    # Train one model per target
    for tgt in present_targets:
        pipe, mae, rmse, r2, model_path, feats_path, X_test, y_test, preds = train_one(df, tgt, feature_cols)
        metrics.append({
            "Target": tgt,
            "MAE": mae,
            "RMSE": rmse,  # <-- added
            "R2": r2,
            "Model File": model_path.name
        })

        # Predict for full dataset
        preds_full = pipe.predict(df[feature_cols])
        preds_df[f"Predicted {tgt}"] = preds_full

        # Plot actual vs predicted
        plot_results(y_test, preds, tgt)

    # Save predictions and metrics
    out_excel = OUTPUTS_DIR / "predictions_linear_multi.xlsx"
    preds_df.to_excel(out_excel, index=False)
    print(f"Wrote predictions to {out_excel}")

    out_metrics = OUTPUTS_DIR / "metrics_linear_multi.txt"
    with out_metrics.open("w", encoding="utf-8") as f:
        for m in metrics:
            f.write(f"{m['Target']}: MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}, R2={m['R2']:.3f}, Model={m['Model File']}\n")
    print(f"Wrote metrics to {out_metrics}")

    # Print summary to console
    print("\n=== Summary ===")
    for m in metrics:
        print(f"- {m['Target']}: MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}, R2={m['R2']:.3f}")


if __name__ == "__main__":
    main()
