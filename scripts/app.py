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

from nsga2_optimize import run_nsga2  # optimisation function

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
# Slider helper
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
# Material Sliders
# -------------------------------------------------------
user_bounds = {
    "cement": slider_range("Cement", 0.0, 500.0, 200.0, 350.0),
    "water": slider_range("Water", 0.0, 250.0, 140.0, 180.0),
    "fine_agg": slider_range("Fine Aggregate", 0.0, 1000.0, 800.0, 950.0),
    "coarse_agg": slider_range("Coarse Aggregate", 0.0, 1200.0, 900.0, 1100.0),
    "FA": slider_range("Fly Ash (FA)", 0.0, 200.0, 0.0, 150.0),
    "SF": slider_range("Silica Fume (SF)", 0.0, 60.0, 0.0, 30.0),
    "GGBFS": slider_range("GGBFS", 0.0, 300.0, 0.0, 200.0),
    "SP": slider_range("Superplasticizer (SP)", 0.0, 10.0, 0.5, 1.5, step=0.05),
}

# -------------------------------------------------------
# ⚠️ FEASIBILITY CHECK + WARNINGS
# -------------------------------------------------------
cement_lo, cement_hi = user_bounds["cement"]
water_lo, water_hi = user_bounds["water"]
FA_lo, FA_hi = user_bounds["FA"]
SF_lo, SF_hi = user_bounds["SF"]
GGBFS_lo, GGBFS_hi = user_bounds["GGBFS"]
fine_lo, fine_hi = user_bounds["fine_agg"]
coarse_lo, coarse_hi = user_bounds["coarse_agg"]

# compute mid-point estimates
cement = (cement_lo + cement_hi) / 2
water = (water_lo + water_hi) / 2
FA = (FA_lo + FA_hi) / 2
SF = (SF_lo + SF_hi) / 2
GGBFS = (GGBFS_lo + GGBFS_hi) / 2
fine_agg = (fine_lo + fine_hi) / 2
coarse_agg = (coarse_lo + coarse_hi) / 2

binder = cement + FA + GGBFS + SF
w_b = water / binder if binder > 0 else 999
SCM_pct = (FA + GGBFS + SF) / binder if binder > 0 else 999
agg_total = fine_agg + coarse_agg

warnings_list = []

# ---- W/B warnings ----
if w_b < 0.35:
    warnings_list.append("Water–binder ratio too LOW (<0.35). Mix may be unworkable without high SP.")
elif w_b > 0.60:
    warnings_list.append("Water–binder ratio too HIGH (>0.60). Mix likely too weak for structural concrete.")

# ---- Binder window ----
if binder < 280:
    warnings_list.append(f"Binder too LOW ({binder:.0f} kg/m³). May not reach target strength.")
elif binder > 480:
    warnings_list.append(f"Binder too HIGH ({binder:.0f} kg/m³). Excess cost, CO₂ and shrinkage risk.")

# ---- SCM caps ----
if FA / binder > 0.40:
    warnings_list.append("Fly Ash exceeds 40% of binder.")
if GGBFS / binder > 0.70:
    warnings_list.append("GGBFS exceeds 70% of binder.")
if SF / binder > 0.10:
    warnings_list.append("Silica Fume exceeds 10% of binder.")
if SCM_pct > 0.50:
    warnings_list.append("Total SCM exceeds 50% binder — may impair early strength.")

# ---- Aggregate total ----
if agg_total < 1700 or agg_total > 2000:
    warnings_list.append("Total aggregates should be 1700–2000 kg/m³ for normal-density concrete.")

# ---- Strength too high vs cement bounds ----
if strength_target > 70 and cement_hi < 350:
    warnings_list.append("Target strength >70 MPa may not be achievable with current cement upper bound.")

# -------------------------------------------------------
# 🚦 TRAFFIC LIGHT INDICATOR
# -------------------------------------------------------
if len(warnings_list) == 0:
    colour = "🟢"
    status_msg = "All constraints satisfied — optimisation likely to succeed."
elif any("too HIGH" in w or "exceeds" in w for w in warnings_list):
    colour = "🔴"
    status_msg = "Some inputs violate physical constraints — optimisation may return no solutions."
else:
    colour = "🟡"
    status_msg = "Some values are borderline — proceed with caution."

st.markdown(f"""
### {colour} Feasibility Status  
**{status_msg}**

""")

if warnings_list:
    st.markdown("#### ⚠️ Warnings:")
    for w in warnings_list:
        st.warning(w)

# -------------------------------------------------------
# Run Optimisation
# -------------------------------------------------------
if st.sidebar.button("🚀 Run Optimisation"):
    with st.spinner("Running NSGA-II optimisation... Please wait ⏳"):
        constraints = {
            "strength_target": strength_target,
            "strength_tolerance": strength_tolerance
        }

        if "SP" not in user_bounds:
            user_bounds["SP"] = (0.8, 0.8)

        raw_results = run_nsga2(models, constraints, user_bounds, pop_size=100, n_gen=60)

        # ---- UNPACK RETURN ----
        if isinstance(raw_results, tuple):
            # assume the first element is the dataframe
            results = raw_results[0]  
        else:
            results = raw_results


    st.success("✅ Optimisation complete!")

    if results is None or len(results) == 0:
        st.error("❌ No feasible mixtures found for the given constraints.")
        st.stop()

    # ---------------------------------------------------
    # Normalisation of column names
    # ---------------------------------------------------
    results.columns = [c.strip().lower() for c in results.columns]
    co2_col = next((c for c in results.columns if "co2" in c), None)
    cost_col = next((c for c in results.columns if "cost" in c), None)
    strength_col = next((c for c in results.columns if "strength" in c), None)

    # ---------------------------------------------------
    # 🥇 Top Mix
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
    st.dataframe(pd.DataFrame(list(mix_data.items()), columns=["Constituent", "Amount (kg/m³)"]))

    # ---------------------------------------------------
    # pareto scatter
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
        labels={co2_col: "Embodied CO₂ (kg/m³)", cost_col: "Cost ($/kg)", strength_col: "Strength (MPa)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # full table + download
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
