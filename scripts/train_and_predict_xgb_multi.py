"""
train_and_predict_xgb_multi.py
(Final version with SHAP explainability + CSV export + interaction plots + durability exclusion)

Generates SHAP analyses for each target to show which constituents
most influence each output (e.g., strength, carbon, cost, etc.).
Durability outputs (chloride permeability, shrinkage, creep, abrasion)
are automatically excluded from SHAP/interaction plots.

Plots:
    ./outputs/shap_<target>.png
    ./outputs/shap_bar_<target>.png
    ./outputs/shap_interactions_<target>.png

CSV:
    ./outputs/shap_top_features.csv
"""

import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
})

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

# Exclude sustainability and durability outputs from features
EXCLUDE_PATTERNS = [
    r"embodied\s*co",
    r"\bcost\b",
    r"shrinkage",
    r"\bcreep\b",
    r"chloride",
    r"abrasion",
]
def is_excluded(col: str) -> bool:
    cl = col.lower()
    return any(re.search(pat, cl, flags=re.IGNORECASE) for pat in EXCLUDE_PATTERNS)

# ---------- SHAP analysis ----------
def shap_analysis(pipe, X, target, feature_cols, shap_top_rows):
    """Generate SHAP summary + bar + interaction plots and store top features."""
    try:
        print(f"Generating SHAP analysis for {target}...")

        # Filter again just in case
        filtered_features = [f for f in feature_cols if not is_excluded(f)]
        X_imputed = pipe.named_steps["prep"].transform(X[filtered_features])
        X_imputed = np.asarray(X_imputed, dtype=float)
        model = pipe.named_steps["model"]

        explainer = shap.Explainer(model, X_imputed, feature_names=filtered_features)
        shap_values = explainer(X_imputed)

        # --- Beeswarm plot ---
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_imputed, feature_names=filtered_features, show=False, plot_size=None)
        fig.suptitle(f"SHAP Summary – {target}", fontsize=13, y=0.98)
        fig.subplots_adjust(top=0.92, bottom=0.1, left=0.35, right=0.95)
        fig.savefig(OUTPUTS_DIR / f"shap_{slugify(target)}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # --- Bar plot ---
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_imputed, feature_names=filtered_features,
            plot_type="bar", show=False, plot_size=None
        )
        fig.suptitle(f"Mean SHAP Importance – {target}", fontsize=13, y=0.98)
        fig.subplots_adjust(top=0.92, bottom=0.1, left=0.35, right=0.95)
        fig.savefig(OUTPUTS_DIR / f"shap_bar_{slugify(target)}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # --- Save top 10 mean(|SHAP|) features ---
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:10]
        for i in top_idx:
            shap_top_rows.append({
                "Target": target,
                "Feature": filtered_features[i],
                "Mean(|SHAP|)": mean_abs[i]
            })

        # --- Interaction plots (top 5 features) ---
        top_features = [filtered_features[i] for i in top_idx[:5]]
        n_feats = len(top_features)
        n_cols = 3
        n_rows = int(np.ceil(n_feats / n_cols))

        plt.close('all')
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

        for i, f in enumerate(top_features):
            color_by = top_features[(i + 1) % n_feats]
            shap.dependence_plot(
                f, shap_values.values, X_imputed, feature_names=filtered_features,
                interaction_index=color_by, show=False, ax=axes[i], alpha=0.8, dot_size=6
            )
            axes[i].set_title(f"{f} vs. {color_by}")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"Feature Interaction Plots – {target}", fontsize=14, y=0.92)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(OUTPUTS_DIR / f"shap_interactions_{slugify(target)}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved SHAP plots (summary, bar, and interaction) for {target}")

    except Exception as e:
        print(f"[WARN] SHAP failed for {target}: {e}")

# ---------- Model training ----------
def train_one(df, target, feature_cols):
    mask = df[target].notna()
    X = df.loc[mask, feature_cols]
    y = df.loc[mask, target]
    n = len(y)

    note = small_data_note(target, n)
    if n < 2:
        print(f"[WARN] Skipping '{target}': not enough samples (n={n}). {note or ''}".strip())
        return None, None, None, None, n, note or "insufficient samples"

    test_size = 0.2 if n > 10 else 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    preprocess = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), feature_cols)],
        remainder="drop"
    )

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )

    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2  = r2_score(y_test, preds)

    return pipe, mae, rmse, r2, n, note

# ---------- main ----------
def main():
    ensure_dirs()

    excel_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_EXCEL
    sheet_name = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_SHEET

    if not excel_path.exists():
        raise FileNotFoundError(f"Could not find Excel file: {excel_path}")

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    present_targets = find_present_targets(df.columns)
    print(f"Detected targets: {present_targets}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if (c not in present_targets) and (not is_excluded(c))]
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found after excluding targets.")

    metrics = []
    shap_top_rows = []
    preds_df = df.copy()

    for tgt in present_targets:
        pipe, mae, rmse, r2, n, note = train_one(df, tgt, feature_cols)
        if pipe is None:
            metrics.append({"Target": tgt, "MAE": None, "RMSE": None, "R2": None, "Samples": n, "Note": note})
            continue

        # ✅ Save trained XGBoost model (.json) - check if this works (riki)
        model = pipe.named_steps["model"]
        model_path = MODELS_DIR / f"xgb_{slugify(tgt)}_model.json"
        model.save_model(model_path)
        print(f"✅ Saved XGBoost model for {tgt} → {model_path}")

        metrics.append({"Target": tgt, "MAE": mae, "RMSE": rmse, "R2": r2, "Samples": n, "Note": note})
        preds_df[f"Predicted {tgt}"] = pipe.predict(df[feature_cols])

        # shap_analysis(pipe, df[feature_cols], tgt, feature_cols, shap_top_rows)

    # ---------- Outputs ----------
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
            f.write(f"{m['Target']}: MAE={mae_str}, RMSE={rmse_str}, R2={r2_str}{note}\n")
    print(f"Wrote metrics to {out_metrics}")

    # ---------- Save SHAP top features ----------
    if shap_top_rows:
        shap_df = pd.DataFrame(shap_top_rows)
        shap_df.to_csv(OUTPUTS_DIR / "shap_top_features.csv", index=False)
        print(f"Saved top SHAP features to {OUTPUTS_DIR / 'shap_top_features.csv'}")

    # ---------- Summary ----------
    print("\n=== Summary ===")
    for m in metrics:
        mae_str = "N/A" if m["MAE"] is None else f"{m['MAE']:.3f}"
        rmse_str = "N/A" if m["RMSE"] is None else f"{m["RMSE"]:.3f}"
        r2_str  = "N/A" if m["R2"]  is None else f"{m["R2"]:.3f}"
        note    = f" | {m['Note']}" if m.get("Note") else ""
        print(f"- {m['Target']}: MAE={mae_str}, RMSE={rmse_str}, R2={r2_str}{note}")

if __name__ == "__main__":
    main()
