import os
print(">>> RUNNING FILE:", os.path.abspath(__file__))

# app.py — Streamlit Concrete Mix Optimizer with NSGA-II
import sys
from pathlib import Path
from xgboost import XGBRegressor
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# -------------------------------------------------------
# Setup paths
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXCEL = BASE_DIR / "data" / "dataset.xlsx"

SCRIPTS_DIR = BASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from nsga2_optimize import run_nsga2  # ✅ optimisation function

# -------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------
st.set_page_config(page_title="Green Concrete Optimizer", layout="wide")
st.title("🌿 Green Concrete Mixture Optimizer")

# -------------------------------------------------------
# Load XGBoost models
# -------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    MODELS_DIR = BASE_DIR / "models"
    model_files = {
        "strength": "xgb_cylinder_compressive_strength_mpa_model.json",
        "co2": "xgb_embodied_co_kg_co_e_m_model.json",
        "cost": "xgb_cost_kg_model.json",
    }
    models = {}
    for key, fname in model_files.items():
        path = MODELS_DIR / fname
        model = XGBRegressor()
        model.load_model(path)
        models[key] = model
    return models

models = load_models()
st.success("✅ XGBoost models loaded successfully!")

# -------------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------------
st.sidebar.header("🔧 User Inputs")

strength_target = st.sidebar.number_input(
    "Target Compressive Strength (MPa)", 20.0, 100.0, 45.0, 1.0
)
strength_tolerance = st.sidebar.number_input(
    "Tolerance (± MPa)", 0.5, 10.0, 3.0, 0.5
)

st.sidebar.markdown("---")
st.sidebar.subheader("Material Ranges (kg/m³)")

# -------------------------------------------------------
# Slider helper (simple numeric range)
# -------------------------------------------------------
def slider_range(label, lo, hi, default_lo, default_hi, step=1.0):
    return st.sidebar.slider(
        label,
        min_value=float(lo),
        max_value=float(hi),
        value=(float(default_lo), float(default_hi)),
        step=float(step),
        key=f"{label}_range"
    )

# -------------------------------------------------------
# Material range inputs
# -------------------------------------------------------
user_bounds = {
    "cement": slider_range("Cement", 100.0, 500.0, 200.0, 350.0),
    "water": slider_range("Water", 100.0, 250.0, 140.0, 180.0),
    "fine_agg": slider_range("Fine Aggregate", 700.0, 1000.0, 800.0, 950.0),
    "coarse_agg": slider_range("Coarse Aggregate", 800.0, 1200.0, 900.0, 1100.0),
    "FA": slider_range("Fly Ash (FA)", 0.0, 200.0, 0.0, 150.0),
    "SF": slider_range("Silica Fume (SF)", 0.0, 60.0, 0.0, 30.0),
    "GGBFS": slider_range("GGBFS", 0.0, 250.0, 0.0, 200.0),
    "SP": slider_range("Superplasticizer (SP)", 0.0, 2.0, 0.5, 1.5, step=0.05),  # ✅ added SP
}

print(">>> Bounds keys:", list(user_bounds.keys()))

# -------------------------------------------------------
# Run Optimisation
# -------------------------------------------------------
if st.sidebar.button("🚀 Run Optimisation"):
    with st.spinner("Running NSGA-II optimisation... Please wait ⏳"):
        constraints = {
            "strength_target": strength_target,
            "strength_tolerance": strength_tolerance
        }

        # ✅ Fallback if SP missing (safety net)
        if "SP" not in user_bounds:
            user_bounds["SP"] = (0.8, 0.8)

        # ✅ Run optimisation
        results = run_nsga2(models, constraints, user_bounds, pop_size=100, n_gen=60)

        print(">>> Type of results:", type(results))
        if isinstance(results, pd.DataFrame):
            print(">>> Results shape:", results.shape)
            print(">>> Results columns:", list(results.columns))
            print(">>> First few rows:\n", results.head())
        else:
            print(">>> Results preview:", results)

    st.success("✅ Optimisation complete!")

    if results is None or len(results) == 0:
        st.warning("⚠️ No feasible mixtures found.")
        st.stop()

    # ---------------------------------------------------
    # Normalise column naming
    # ---------------------------------------------------
    results.columns = [c.strip().lower() for c in results.columns]
    co2_col = next((c for c in results.columns if "co2" in c), None)
    cost_col = next((c for c in results.columns if "cost" in c), None)
    strength_col = next((c for c in results.columns if "strength" in c), None)

    # ---------------------------------------------------
    # 🥇 Top Optimised Mixture (Lowest CO₂)
    # ---------------------------------------------------
    st.markdown("### 🥇 Top Optimised Mixture — Lowest Embodied CO₂")
    top_mix = results.loc[results[co2_col].idxmin()].to_dict()

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Strength (MPa)", f"{top_mix[strength_col]:.2f}")
    col2.metric("Embodied CO₂ (kg/m³)", f"{top_mix[co2_col]:.2f}")
    col3.metric("Cost ($/kg)", f"{top_mix[cost_col]:.2f}")

    st.markdown("#### Mix Composition (kg/m³)")
    mix_keys = ["cement", "water", "fine_agg", "coarse_agg", "fa", "sf", "ggbfs", "sp"]
    mix_data = {k: top_mix[k] for k in mix_keys if k in top_mix}
    mix_df = pd.DataFrame(list(mix_data.items()), columns=["Constituent", "Amount (kg/m³)"])
    st.dataframe(mix_df, use_container_width=True)

    # ---------------------------------------------------
    # 🌍 Pareto Front Visualisation
    # ---------------------------------------------------
    st.subheader("🌍 Pareto Front — CO₂ vs Cost")
    fig = px.scatter(
        results,
        x=co2_col,
        y=cost_col,
        color=strength_col,
        color_continuous_scale="Viridis",
        hover_data=["cement","water","fa","sf","ggbfs","fine_agg","coarse_agg","sp"],
        title="Trade-off between Embodied CO₂ and Cost",
        labels={
            co2_col: "Embodied CO₂ (kg/m³)",
            cost_col: "Cost ($/kg)",
            strength_col: "Strength (MPa)",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # 📊 Full Table + Download
    # ---------------------------------------------------
    st.subheader("📊 Full Optimised Mix Results")
    st.dataframe(results, use_container_width=True, height=500)

    csv = results.to_csv(index=False)
    st.download_button("⬇️ Download Results as CSV", csv, "optimised_mixtures.csv", "text/csv")

else:
    st.info("👈 Set your target strength and ranges, then click **Run Optimisation**.")

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.caption("Developed by Jeffrey Dai & Riki Endo — Sustainable Concrete Mix Design using AI & NSGA-II")
