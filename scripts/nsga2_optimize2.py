"""
nsga2_optimize.py
-----------------
Multi-objective optimisation using NSGA-II for green concrete mix design.
Integrates pre-trained XGBoost models for strength, CO₂, and cost prediction.

Objectives:
    1. Minimise embodied CO₂ (kg CO₂e/m³)
    2. Minimise cost ($/kg)
    3. Maximise compressive strength (MPa)
"""

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
import json
from pathlib import Path


# --------------------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------------------
def run_nsga2(models, user_bounds, pop_size=500, n_gen=200):

    # --- Save configuration for reproducibility ---
    config = {
        "population_size": pop_size,
        "generations": n_gen,
        "crossover_probability": 0.9,
        "mutation_probability": 0.14,
        "selection_operator": "tournament",
        "crossover_operator": "simulated binary crossover (SBX, eta=15)",
        "mutation_operator": "polynomial mutation (PM, eta=20)",
        "variable_bounds": user_bounds,
        "constraints_removed": True
    }

    MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
    CONFIG_PATH = MODELS_DIR / "nsga2_config.json"
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

    # -----------------------------
    # Variable definitions
    # -----------------------------
    var_names = list(user_bounds.keys())
    xl = np.array([v[0] for v in user_bounds.values()])
    xu = np.array([v[1] for v in user_bounds.values()])

    # Explicit feature layout (must match model training)
    feature_cols = [
        "cement", "water", "fine_agg", "coarse_agg",
        "FA", "SF", "GGBFS", "SP", "binder", "w/b", "SCM_pct", "density"
    ]

    # -----------------------------
    # Define optimisation problem
    # -----------------------------
    class ConcreteProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(var_names), n_obj=3, n_constr=0, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            df = pd.DataFrame(X, columns=var_names)

            # Derived features (must match training data)
            df["binder"] = df["cement"] + df["FA"] + df["GGBFS"] + df["SF"]
            df["w/b"] = df["water"] / df["cement"]
            df["SCM_pct"] = 100 * (df["FA"] + df["GGBFS"] + df["SF"]) / df["binder"]
            df["SP"] = 0.8
            df["density"] = 2400
            df = df[feature_cols]

            # Model predictions
            strength = models["strength"].predict(df)
            co2 = models["co2"].predict(df)
            cost = models["cost"].predict(df)

            # Objective vector: minimise CO₂ and cost, maximise strength
            out["F"] = np.column_stack([co2, cost, -strength])  # negative → maximise

    # -----------------------------
    # Run NSGA-II algorithm
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
        verbose=True,
        save_history=True
    )

    # --------------------------------------------------------------
    # Convergence Tracking (Hypervolume)
    # --------------------------------------------------------------
    OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # compute hypervolume progression
    ref_point = np.array([500, 250, 0])  # slightly worse than expected maxima
    hv = HV(ref_point=ref_point)
    hv_values = [hv.do(e.pop.get("F")) for e in res.history]
    gens = np.arange(len(hv_values))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(gens, hv_values, "-o", markersize=3, linewidth=1.5)
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume Indicator")
    plt.title("NSGA-II Convergence Profile (Hypervolume vs. Generation)")
    plt.grid(True)
    plt.tight_layout()
    hv_path = OUTPUTS_DIR / "nsga2_convergence.png"
    plt.savefig(hv_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Convergence plot saved to: {hv_path}")

    # --------------------------------------------------------------
    # Collect Pareto Front Results
    # --------------------------------------------------------------
    pareto_df = pd.DataFrame(res.X, columns=var_names)
    pareto_df["binder"] = pareto_df["cement"] + pareto_df["FA"] + pareto_df["GGBFS"] + pareto_df["SF"]
    pareto_df["w/b"] = pareto_df["water"] / pareto_df["cement"]
    pareto_df["SCM_pct"] = 100 * (pareto_df["FA"] + pareto_df["GGBFS"] + pareto_df["SF"]) / pareto_df["binder"]
    pareto_df["SP"] = 0.8
    pareto_df["density"] = 2400

    pareto_inputs = pareto_df[feature_cols].astype(np.float32)
    pareto_df["pred_strength_MPa"] = models["strength"].predict(pareto_inputs)
    pareto_df["pred_co2_kg_per_m3"] = models["co2"].predict(pareto_inputs)
    pareto_df["pred_cost_per_kg"] = models["cost"].predict(pareto_inputs)

    pareto_df = pareto_df.sort_values(by="pred_co2_kg_per_m3").reset_index(drop=True)

    # # --------------------------------------------------------------
    # # 3D PARETO FRONT PLOT (and save)
    # # --------------------------------------------------------------
    # from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # fig = plt.figure(figsize=(9, 7))
    # ax = fig.add_subplot(111, projection="3d")

    # x = pareto_df["pred_co2_kg_per_m3"]
    # y = pareto_df["pred_cost_per_kg"]
    # z = pareto_df["pred_strength_MPa"]

    # norm_strength = (z - z.min()) / (z.max() - z.min())
    # sc = ax.scatter(x, y, z, c=norm_strength, cmap="jet", s=30, alpha=0.9, edgecolors="none")

    # ax.set_xlabel("Embodied CO₂ emission (kg e-CO₂/m³)", labelpad=10)
    # ax.set_ylabel("Price ($/kg)", labelpad=10)
    # ax.set_zlabel("28-day compressive strength (MPa)", labelpad=10)

    # cbar = plt.colorbar(sc, pad=0.15)
    # cbar.set_label("Normalized Strength (0–1)", rotation=270, labelpad=15)
    # ax.view_init(elev=25, azim=135)
    # plt.tight_layout()

    # save_path = OUTPUTS_DIR / "pareto_front_3D.png"
    # plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # print(f"✅ Pareto front plot saved to: {save_path}")
    # plt.show()

    return pareto_df


# --------------------------------------------------------------
# OPTIONAL: Standalone test run
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
        model_path = MODELS_DIR / filename
        print(f"Loading {model_path.name} ...")
        model = XGBRegressor()
        model.load_model(model_path)
        models[key] = model

    print("✅ All models loaded successfully!\n")

    user_bounds = {
        "cement": (200, 350),
        "water": (140, 180),
        "fine_agg": (800, 950),
        "coarse_agg": (900, 1100),
        "FA": (0, 150),
        "SF": (0, 30),
        "GGBFS": (0, 200),
    }

    results = run_nsga2(models, user_bounds, pop_size=500, n_gen=200)
    print("\n✅ Optimisation complete. Preview of results:")
    print(results.head())
