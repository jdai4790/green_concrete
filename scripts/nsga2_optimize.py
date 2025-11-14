"""
nsga2_optimize.py
-----------------
Multi-objective optimisation using NSGA-II for green concrete mix design.
Integrates pre-trained XGBoost models for strength, CO₂, and cost prediction.

Objectives:
    1. Minimise embodied CO₂ (kg CO₂e/m³)
    2. Minimise cost ($/kg)
Constraints:
    1. |predicted_strength - target_strength| ≤ tolerance
    2. 0.35 ≤ w/b ≤ 0.60 (structural concrete feasibility)
    3. 280 ≤ binder ≤ 480
    4. SCM caps: FA ≤ 40%, GGBFS ≤ 70%, SF ≤ 10%, Total SCM ≤ 50%
    5. Total aggregates 1700–2000 kg/m³
"""

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import json
from pathlib import Path
import matplotlib.pyplot as plt


# --------------------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------------------
def run_nsga2(models, user_constraints, user_bounds, pop_size=500, n_gen=200):

    # --- Save configuration for reproducibility ---
    config = {
        "population_size": pop_size,
        "generations": n_gen,
        "crossover_probability": 0.9,
        "mutation_probability": 0.14,
        "selection_operator": "tournament",
        "crossover_operator": "simulated binary crossover (SBX, eta=15)",
        "mutation_operator": "polynomial mutation (PM, eta=20)",
        "user_constraints": user_constraints,
        "variable_bounds": user_bounds,
        "w/b_constraints": {"min": 0.35, "max": 0.60},
        "binder_window": {"min": 280, "max": 480},
        "SCM_caps": {"FA": 0.40, "GGBFS": 0.70, "SF": 0.10, "total": 0.50},
        "aggregate_total": {"min": 1700, "max": 2000}
    }

    MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
    CONFIG_PATH = MODELS_DIR / "nsga2_config.json"

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

    # -----------------------------
    # Prepare variable definitions
    # -----------------------------
    var_names = list(user_bounds.keys())
    xl = np.array([v[0] for v in user_bounds.values()])
    xu = np.array([v[1] for v in user_bounds.values()])

    strength_target = user_constraints["strength_target"]
    tolerance = user_constraints["strength_tolerance"]

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
            super().__init__(n_var=len(var_names), n_obj=2, n_constr=11, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):

            df = pd.DataFrame(X, columns=var_names)

            # Derived features
            df["binder"] = df["cement"] + df["FA"] + df["GGBFS"] + df["SF"]
            df["w/b"] = df["water"] / df["binder"]
            df["SCM_pct"] = (df["FA"] + df["GGBFS"] + df["SF"]) / df["binder"]
            # df["SP"] = 0.8
            df["density"] = 2400

            # SCM fractions
            FA_pct = df["FA"] / df["binder"]
            GGBFS_pct = df["GGBFS"] / df["binder"]
            SF_pct = df["SF"] / df["binder"]

            df_model = df[feature_cols]

            # Predictions
            strength = models["strength"].predict(df_model)
            co2 = models["co2"].predict(df_model)
            cost = models["cost"].predict(df_model)

            # ----------------------
            # HARD CONSTRAINTS
            # ----------------------
            g_strength = np.abs(strength - strength_target) - tolerance
            g_wb_min = 0.35 - df["w/b"]
            g_wb_max = df["w/b"] - 0.60
            g_binder_min = 280 - df["binder"]
            g_binder_max = df["binder"] - 480
            g_fa = FA_pct - 0.40
            g_ggb = GGBFS_pct - 0.70
            g_sf = SF_pct - 0.10
            g_scmT = df["SCM_pct"] - 0.50

            agg_total = df["fine_agg"] + df["coarse_agg"]
            g_agg_min = 1700 - agg_total
            g_agg_max = agg_total - 2000

            out["G"] = np.column_stack([
                g_strength,
                g_wb_min, g_wb_max,
                g_binder_min, g_binder_max,
                g_fa, g_ggb, g_sf, g_scmT,
                g_agg_min, g_agg_max
            ])

            out["F"] = np.column_stack([co2, cost])

    # -----------------------------
    # Run NSGA-II
    # -----------------------------
    algorithm = NSGA2(
        pop_size=pop_size,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=1/len(user_bounds), eta=20),
        eliminate_duplicates=True
    )

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

    pareto_df["binder"] = pareto_df["cement"] + pareto_df["FA"] + pareto_df["GGBFS"] + pareto_df["SF"]
    pareto_df["w/b"] = pareto_df["water"] / pareto_df["binder"]
    pareto_df["SCM_pct"] = (pareto_df["FA"] + pareto_df["GGBFS"] + pareto_df["SF"]) / pareto_df["binder"]
    # pareto_df["SP"] = 0.8
    pareto_df["density"] = 2400
    pareto_df["agg_total"] = pareto_df["fine_agg"] + pareto_df["coarse_agg"]

    pareto_inputs = pareto_df[feature_cols].copy()
    pareto_inputs = pareto_inputs.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pareto_inputs = pareto_inputs.astype(np.float32)

    pareto_df["pred_strength_MPa"] = models["strength"].predict(pareto_inputs)
    pareto_df["pred_co2_kg_per_m3"] = models["co2"].predict(pareto_inputs)
    pareto_df["pred_cost_per_kg"] = models["cost"].predict(pareto_inputs)

    # --------------------------------------------------------------
    # NEW SECTION: AUTOMATIC CONSTRAINT SUMMARY TABLE
    # --------------------------------------------------------------
    bounds = {
        "w/b": (0.35, 0.60),
        "binder": (280, 480),
        "FA_over_b": (0, 0.40),
        "GGBFS_over_b": (0, 0.70),
        "SF_over_b": (0, 0.10),
        "SCM_total_over_b": (0, 0.50),
        "agg_total": (1700, 2000),
    }

    pareto_df["FA_over_b"] = pareto_df["FA"] / pareto_df["binder"]
    pareto_df["GGBFS_over_b"] = pareto_df["GGBFS"] / pareto_df["binder"]
    pareto_df["SF_over_b"] = pareto_df["SF"] / pareto_df["binder"]
    pareto_df["SCM_total_over_b"] = (pareto_df["FA"] + pareto_df["GGBFS"] + pareto_df["SF"]) / pareto_df["binder"]

    rows = []
    for var, (low, high) in bounds.items():
        rows.append({
            "Variable": var,
            "Theoretical Min": low,
            "Theoretical Max": high,
            "Observed Min": round(pareto_df[var].min(), 4),
            "Observed Max": round(pareto_df[var].max(), 4),
        })

    constraints_summary = pd.DataFrame(rows)

    summary_path = MODELS_DIR / "nsga2_constraint_summary.csv"
    constraints_summary.to_csv(summary_path, index=False)


    # --------------------------------------------------------------
    # NEW BLOCK: TRAINING vs OPTIMISED PARETO FIGURE
    # --------------------------------------------------------------
    try:
        DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset.xlsx"
        OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
        OUTPUT_DIR.mkdir(exist_ok=True)

        df_train = pd.read_excel(DATA_PATH)

        STRENGTH_COL = "Cylinder compressive strength (MPa)"
        CO2_COL = "Embodied CO₂ (kg CO₂e/m³)"
        COST_COL = "Cost ($/kg)"

        TARGET = user_constraints["strength_target"]
        TOL = user_constraints["strength_tolerance"]

        mask = df_train[STRENGTH_COL].between(TARGET - TOL, TARGET + TOL)
        df_train_band = df_train[mask]

        plt.figure(figsize=(10, 7))

        plt.scatter(
            df_train_band[CO2_COL],
            df_train_band[COST_COL],
            s=25, alpha=0.4, color="grey",
            label=f"Training mixes {TARGET} ± {TOL} MPa"
        )

        plt.scatter(
            pareto_df["pred_co2_kg_per_m3"],
            pareto_df["pred_cost_per_kg"],
            s=40, alpha=0.9, color="red",
            label="NSGA-II Pareto front"
        )

        plt.xlabel("Embodied CO₂ (kg CO₂e/m³)", fontsize=12)
        plt.ylabel("Cost ($/m³)", fontsize=12)
        plt.title(
            f"Optimised Pareto Front vs Training Data ({TARGET} ± {TOL} MPa)",
            fontsize=14
        )

        plt.legend()
        plt.grid(alpha=0.3)

        FIG_PATH = OUTPUT_DIR / f"pareto_vs_training_{TARGET}mpa.png"
        plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
        plt.close()

        print(f">>> Saved comparison figure to {FIG_PATH}")

    except Exception as e:
        print(">>> WARNING: Failed to generate comparison figure:", e)


    # --------------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------------
    return (
        pareto_df.sort_values(by="pred_co2_kg_per_m3").reset_index(drop=True),
        constraints_summary
    )



# --------------------------------------------------------------
# OPTIONAL: Standalone run for testing
# --------------------------------------------------------------
if __name__ == "__main__":
    from xgboost import XGBRegressor

    MODELS_DIR = Path("./models")

    models = {}
    model_files = {
        "strength": "xgb_cylinder_compressive_strength_mpa_model.json",
        "co2": "xgb_embodied_co_kg_co_e_m_model.json",
        "cost": "xgb_cost_kg_model.json",
    }

    for key, filename in model_files.items():
        model = XGBRegressor()
        model.load_model(MODELS_DIR / filename)
        models[key] = model

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

    results, summary = run_nsga2(models, user_constraints, user_bounds)
    print(results.head())
    print(summary)
