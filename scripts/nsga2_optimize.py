"""
nsga2_optimize.py
-----------------
Multi-objective optimisation using NSGA-II for green concrete mix design.
Integrates pre-trained XGBoost models for strength, CO₂, and cost prediction.

Objectives:
    1. Minimise embodied CO₂ (kg CO₂e/m³)
    2. Minimise cost ($/kg)
Constraint:
    |predicted_strength - target_strength| ≤ tolerance

Inputs from Streamlit UI:
    - models: dict of preloaded XGB models
    - user_constraints: { "strength_target": float, "strength_tolerance": float }
    - user_bounds: dict of min/max tuples for each constituent
    - pop_size, n_gen: NSGA-II parameters
"""

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem


# --------------------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------------------
def run_nsga2(models, user_constraints, user_bounds, pop_size=100, n_gen=60):
    """
    Runs NSGA-II multi-objective optimisation using pre-trained ML models.

    Parameters
    ----------
    models : dict
        Dictionary of pre-trained XGBoost models (keys: 'strength', 'co2', 'cost')
    user_constraints : dict
        Contains "strength_target" (float) and "strength_tolerance" (float)
    user_bounds : dict
        Variable bounds, e.g. {"cement": (200,350), "water": (140,180), ...}
    pop_size : int
        Population size for NSGA-II (default=100)
    n_gen : int
        Number of generations for NSGA-II (default=60)

    Returns
    -------
    results : pandas.DataFrame
        Pareto-optimal mix designs and predicted outputs.
    """

    # -----------------------------
    # Prepare variable definitions
    # -----------------------------
    var_names = list(user_bounds.keys())
    xl = np.array([v[0] for v in user_bounds.values()])
    xu = np.array([v[1] for v in user_bounds.values()])

    strength_target = user_constraints["strength_target"]
    tolerance = user_constraints["strength_tolerance"]

    # Explicit feature layout used during model training
    feature_cols = [
        "cement", "water", "fine_agg", "coarse_agg",
        "FA", "SF", "GGBFS", "SP", "binder",
        "w/b", "SCM_pct", "density"
    ]

    # -----------------------------
    # Define the optimisation problem
    # -----------------------------
    class ConcreteProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(var_names), n_obj=2, n_constr=1, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            df = pd.DataFrame(X, columns=var_names)

            # --- Derived features (must match training set) ---
            df["binder"] = df["cement"] + df["FA"] + df["GGBFS"] + df["SF"]
            df["w/b"] = df["water"] / df["cement"]
            df["SCM_pct"] = 100 * (df["FA"] + df["GGBFS"] + df["SF"]) / df["binder"]
            df["SP"] = 0.8
            df["density"] = 2400

            df = df[feature_cols]

            # --- Predictions ---
            strength = models["strength"].predict(df)
            co2 = models["co2"].predict(df)
            cost = models["cost"].predict(df)

            # --- Objectives and constraint ---
            out["F"] = np.column_stack([co2, cost])  # minimise both
            out["G"] = np.abs(strength - strength_target) - tolerance  # ≤ 0

    # -----------------------------
    # Run NSGA-II algorithm
    # -----------------------------
    algorithm = NSGA2(pop_size=pop_size)
    res = minimize(
        ConcreteProblem(),
        algorithm,
        ("n_gen", n_gen),
        verbose=True
    )

    # -----------------------------
    # Collect results
    # -----------------------------
    pareto_df = pd.DataFrame(res.X, columns=var_names)

    # Derived features for output display
    pareto_df["binder"] = pareto_df["cement"] + pareto_df["FA"] + pareto_df["GGBFS"] + pareto_df["SF"]
    pareto_df["w/b"] = pareto_df["water"] / pareto_df["cement"]
    pareto_df["SCM_pct"] = 100 * (pareto_df["FA"] + pareto_df["GGBFS"] + pareto_df["SF"]) / pareto_df["binder"]
    pareto_df["SP"] = 0.8
    pareto_df["density"] = 2400

    pareto_inputs = pareto_df[feature_cols].copy()

    # Debug check (useful if something still mismatches)
    print("Columns used for model prediction:", pareto_inputs.columns.tolist())
    print("Shape going into model:", pareto_inputs.shape)

    # --- Ensure all numeric columns are float ---
    pareto_inputs = pd.DataFrame(pareto_inputs)

    # Replace any non-numeric types with numeric, coercing strings/objects to floats
    pareto_inputs = pareto_inputs.apply(pd.to_numeric, errors='coerce')

    # Replace NaN and inf values (can appear when a range is 0–0)
    pareto_inputs = pareto_inputs.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Cast everything to float32 for XGBoost safety
    pareto_inputs = pareto_inputs.astype(np.float32)


    # --- Predictions ---
    pareto_df["pred_strength_MPa"] = models["strength"].predict(pareto_inputs)
    pareto_df["pred_co2_kg_per_m3"] = models["co2"].predict(pareto_inputs)
    pareto_df["pred_cost_per_kg"] = models["cost"].predict(pareto_inputs)

    # Sort by CO₂ ascending for readability
    pareto_df = pareto_df.sort_values(by="pred_co2_kg_per_m3", ascending=True).reset_index(drop=True)
    return pareto_df


# --------------------------------------------------------------
# OPTIONAL: Standalone run for testing
# --------------------------------------------------------------
if __name__ == "__main__":
    from xgboost import XGBRegressor
    from pathlib import Path

    MODELS_DIR = Path("./models")

    # --- Load JSON XGBoost models ---
    models = {}
    model_files = {
        "strength": "xgb_cylinder_compressive_strength_mpa_model.json",
        "co2": "xgb_embodied_co_kg_co_e_m_model.json",
        "cost": "xgb_cost_kg_model.json",
    }

    for key, filename in model_files.items():
        model_path = MODELS_DIR / filename
        print(f"Loading {model_path.name} ...")
        model = XGBRegressor()
        model.load_model(model_path)
        models[key] = model

    print("✅ All models loaded successfully!\n")

    # --- User input examples ---
    user_bounds = {
        "cement": (200, 350),
        "water": (140, 180),
        "fine_agg": (800, 950),
        "coarse_agg": (900, 1100),
        "FA": (0, 150),
        "SF": (0, 30),
        "GGBFS": (0, 200),
    }

    user_constraints = {"strength_target": 45, "strength_tolerance": 3}

    # --- Run optimisation ---
    results = run_nsga2(models, user_constraints, user_bounds)
    print("\n✅ Optimisation complete. Preview of results:")
    print(results.head())
