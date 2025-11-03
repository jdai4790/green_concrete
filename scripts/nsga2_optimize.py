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
"""

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pathlib import Path
from xgboost import XGBRegressor


# --------------------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------------------
def run_nsga2(models, user_constraints, user_bounds, pop_size=100, n_gen=60):
    """
    Runs NSGA-II multi-objective optimisation using pre-trained ML models.
    """

    # -----------------------------
    # Prepare variable definitions
    # -----------------------------
    var_names = list(user_bounds.keys())
    xl = np.array([v[0] for v in user_bounds.values()])
    xu = np.array([v[1] for v in user_bounds.values()])

    strength_target = user_constraints["strength_target"]
    tolerance = user_constraints["strength_tolerance"]

    # -----------------------------
    # Load and normalise feature names
    # -----------------------------
    MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
    FEATURE_FILE = MODELS_DIR / "xgb_features.txt"

    if FEATURE_FILE.exists():
        with open(FEATURE_FILE, "r") as f:
            raw_features = [line.strip() for line in f if line.strip()]

        # Map Excel-style headers to variable names used in optimisation
        rename_map = {
            "Cement(kg/m3)": "cement",
            "Water(kg/m3)": "water",
            "Coarse aggregate(kg/m3)": "coarse_agg",
            "Fine aggregate(kg/m3)": "fine_agg",
            "FA (kg/m3)": "FA",
            "SF (kg/m3)": "SF",
            "GGBFS (kg/m3)": "GGBFS",
            "SP (kg/m3)": "SP",
        }

        # Keep only features relevant to optimisation
        feature_names = [rename_map.get(f) for f in raw_features if f in rename_map]
        print(f"✅ Using {len(feature_names)} normalised feature names:", feature_names)
    else:
        raise FileNotFoundError(f"Missing feature file: {FEATURE_FILE}")

    # -----------------------------
    # Define the optimisation problem
    # -----------------------------
    class ConcreteProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(var_names), n_obj=2, n_constr=1, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            df = pd.DataFrame(X, columns=var_names)

            # Filter to match training features
            df_input = df[feature_names]

            # --- Predictions ---
            strength = models["strength"].predict(df_input)
            co2 = models["co2"].predict(df_input)
            cost = models["cost"].predict(df_input)

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
    pareto_inputs = pareto_df[feature_names].copy()
    pareto_inputs = pareto_inputs.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)

    pareto_df["pred_strength_MPa"] = models["strength"].predict(pareto_inputs)
    pareto_df["pred_co2_kg_per_m3"] = models["co2"].predict(pareto_inputs)
    pareto_df["pred_cost_per_kg"] = models["cost"].predict(pareto_inputs)

    pareto_df = pareto_df.sort_values(by="pred_co2_kg_per_m3", ascending=True).reset_index(drop=True)
    return pareto_df


# --------------------------------------------------------------
# OPTIONAL: Standalone run for testing
# --------------------------------------------------------------
if __name__ == "__main__":
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

    for name, model in models.items():
        try:
            booster = model.get_booster()
            if booster is not None:
                print(f"{name} expects {len(booster.feature_names)} features:")
                print(booster.feature_names)
            else:
                print(f"{name} has no booster feature names.")
        except Exception as e:
            print(f"{name} feature name check failed: {e}")

    # --- User input examples ---
    user_bounds = {
        "cement": (200, 350),
        "water": (140, 180),
        "fine_agg": (800, 950),
        "coarse_agg": (900, 1100),
        "FA": (0, 150),
        "SF": (0, 30),
        "GGBFS": (0, 200),
        "SP": (0.6, 1.0),
    }

    user_constraints = {"strength_target": 45, "strength_tolerance": 3}

    # --- Run optimisation ---
    results = run_nsga2(models, user_constraints, user_bounds)
    print("\n✅ Optimisation complete. Preview of results:")
    print(results.head())
