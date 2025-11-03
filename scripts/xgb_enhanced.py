"""
train_and_predict_xgb_multi.py  
(Final working version without early stopping, compatible with older XGBoost)

- KFold CV improves R² stability
- Compatible with xgboost < 1.6
- No early stopping to avoid TypeError
- Predictions + metrics saved
"""

import sys, re
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt

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
    "Embodied CO₂ (kg CO₂e/m³)",
    "Cost ($/kg)",
    "Drying shrinkage (με) 28d",
    "Creep (με) 28d",
]
# ---------------------------------------------------------------------

def ensure_dirs():
    for d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

def slugify(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')

def find_present_targets(columns):
    cols_lower = {c.lower(): c for c in columns}
    present = []
    for t in TARGETS_CANON:
        if t.lower() in cols_lower:
            present.append(cols_lower[t.lower()])
    return present

def small_data_note(target: str, n: int) -> str:
    t = target.lower()
    note_parts = []
    if ("shrinkage" in t) or ("creep" in t):
        note_parts.append("limited domain data")
    if n < 20:
        note_parts.append(f"very small sample (n={n})")
    elif n < 50:
        note_parts.append(f"small sample (n={n})")
    return "; ".join(note_parts)

EXCLUDE_PATTERNS = [r"embodied\s*co", r"\bcost\b", r"shrinkage", r"\bcreep\b"]
def is_excluded(col: str) -> bool:
    cl = col.lower()
    return any(re.search(pat, cl, flags=re.IGNORECASE) for pat in EXCLUDE_PATTERNS)

def train_one(df, target, feature_cols, n_splits=5):
    """Train one XGB model with KFold CV (no early stopping)."""
    mask = df[target].notna()
    X = df.loc[mask, feature_cols]
    y = df.loc[mask, target]
    n = len(y)

    features_path = MODELS_DIR / "xgb_features.txt"
    with features_path.open("w", encoding="utf-8") as f:
        for c in feature_cols:
            f.write(c + "\n")

    note = small_data_note(target, n)
    if n < 2:
        print(f"[WARN] Skipping '{target}': not enough samples (n={n}). {note or ''}".strip())
        return None, None, None, None, features_path, n, note or "insufficient samples"

    preprocess = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), feature_cols)],
        remainder="drop"
    )

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.1,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )

    kf = KFold(n_splits=min(n_splits, n), shuffle=True, random_state=42)
    fold_mae, fold_rmse, fold_r2 = [], [], []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_proc = preprocess.fit_transform(X_train)
        X_val_proc = preprocess.transform(X_val)

        # 👇 No early stopping here
        model.fit(X_train_proc, y_train)

        preds = model.predict(X_val_proc)
        fold_mae.append(mean_absolute_error(y_val, preds))
        fold_rmse.append(np.sqrt(mean_squared_error(y_val, preds)))
        fold_r2.append(r2_score(y_val, preds))

    mae = np.mean(fold_mae)
    rmse = np.mean(fold_rmse)
    r2 = np.mean(fold_r2)

    # Retrain on full data
    X_proc = preprocess.fit_transform(X)
    model.fit(X_proc, y)

    booster = model.get_booster()
    model_json_path = MODELS_DIR / f"xgb_{slugify(target)}_model.json"
    booster.save_model(str(model_json_path))

    # --- Optional SHAP analysis ---
    # try:
    #     X_imputed = preprocess.transform(X)
    #     explainer = shap.Explainer(model, X_imputed, feature_names=feature_cols)
    #     shap_values = explainer(X_imputed)
    #
    #     plt.figure(figsize=(10, 6))
    #     shap.summary_plot(shap_values, X_imputed, feature_names=feature_cols, show=False)
    #     plt.title(f"SHAP Summary – {target}")
    #     plt.tight_layout()
    #     out_path = OUTPUTS_DIR / f"shap_{slugify(target)}.png"
    #     plt.savefig(out_path, dpi=300)
    #     plt.close()
    #
    #     plt.figure(figsize=(8, 5))
    #     shap.summary_plot(shap_values, X_imputed, feature_names=feature_cols,
    #                       plot_type="bar", show=False)
    #     plt.title(f"SHAP Feature Importance – {target}")
    #     plt.tight_layout()
    #     out_path_bar = OUTPUTS_DIR / f"shap_{slugify(target)}_bar.png"
    #     plt.savefig(out_path_bar, dpi=300)
    #     plt.close()
    # except Exception as e:
    #     print(f"[WARN] SHAP failed for {target}: {e}")
    # --- end SHAP analysis ---

    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X, y)

    return pipe, mae, rmse, r2, model_json_path, features_path, n, note

def main():
    ensure_dirs()

    excel_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_EXCEL
    sheet_name = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_SHEET

    if not excel_path.exists():
        raise FileNotFoundError(f"Could not find Excel file: {excel_path}")

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    # === Outlier Removal (IQR method) ===
    def remove_outliers_iqr_all(df, columns, factor=1.5):
        df_clean = df.copy()
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        return df_clean

    numeric_cols_for_outliers = df.select_dtypes(include=[np.number]).columns.tolist()
    rows_before = len(df)
    df = remove_outliers_iqr_all(df, numeric_cols_for_outliers, factor=1.5)
    rows_after = len(df)
    print(f"Removed {rows_before - rows_after} outliers ({rows_after} rows remain)")
    # === End Outlier Removal ===

    present_targets = find_present_targets(df.columns)
    print(f"Detected targets: {present_targets}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if (c not in present_targets) and (not is_excluded(c))]
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found after excluding targets.")

    metrics = []
    preds_df = df.copy()

    for tgt in present_targets:
        pipe, mae, rmse, r2, model_json_path, feats_path, n, note = train_one(df, tgt, feature_cols)
        if pipe is None:
            metrics.append({"Target": tgt, "MAE": None, "RMSE": None, "R2": None, "Samples": n,
                            "Model JSON": "N/A", "Note": note})
            continue

        metrics.append({"Target": tgt, "MAE": mae, "RMSE": rmse, "R2": r2, "Samples": n,
                        "Model JSON": model_json_path.name, "Note": note})
        preds = pipe.predict(df[feature_cols])
        preds_df[f"Predicted {tgt}"] = preds

    out_excel = OUTPUTS_DIR / "predictions_multi.xlsx"
    preds_df.to_excel(out_excel, index=False)
    print(f"Wrote predictions to {out_excel}")

    out_metrics = OUTPUTS_DIR / "metrics_multi.txt"
    with out_metrics.open("w", encoding="utf-8") as f:
        for m in metrics:
            mae_str = "N/A" if m["MAE"] is None else f"{m['MAE']:.3f}"
            rmse_str = "N/A" if m["RMSE"] is None else f"{m['RMSE']:.3f}"
            r2_str  = "N/A" if m["R2"]  is None else f"{m['R2']:.3f}"
            note    = f" | {m['Note']}" if m.get("Note") else ""
            f.write(
                f"{m['Target']}: MAE={mae_str}, RMSE={rmse_str}, R2={r2_str}, Samples={m.get('Samples','?')}, "
                f"Model={m['Model JSON']}{note}\n"
            )

    print(f"Wrote metrics to {out_metrics}")

    print("\n=== Summary ===")
    for m in metrics:
        mae_str = "N/A" if m["MAE"] is None else f"{m['MAE']:.3f}"
        rmse_str = "N/A" if m["RMSE"] is None else f"{m['RMSE']:.3f}"
        r2_str  = "N/A" if m["R2"]  is None else f"{m['R2']:.3f}"
        note    = f" | {m['Note']}" if m.get("Note") else ""
        print(f"- {m['Target']}: MAE={mae_str}, RMSE={rmse_str}, R2={r2_str}{note}")

if __name__ == "__main__":
    main()
