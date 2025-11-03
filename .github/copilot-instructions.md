# Copilot Instructions for green_concrete

## Project Overview
This repository focuses on machine learning modeling for predicting properties of green concrete. The main workflow involves training, evaluating, and saving regression models (XGBoost, Random Forest) using tabular data from Excel files. Outputs include model artifacts and metrics for multiple concrete properties.

## Key Directories & Files
- `data/`: Input datasets (main file: `dataset.xlsx`)
- `models/`: Saved model files (`xgb_*.json`), features list (`xgb_features.txt`)
- `outputs/`: Model predictions and metrics (`predictions_multi.xlsx`, `metrics_multi.txt`)
- `scripts/`: Main scripts for training and evaluation:
  - `train_and_predict_xgb_multi.py`: Multi-target XGBoost training, feature selection, metrics, and output generation
  - `model_train_rf.py`: Random Forest training for compressive strength
  - `linear_regression.py`: (empty, placeholder)

## Workflow Patterns
- **Default Data Path**: Most scripts expect `data/dataset.xlsx` with a sheet named `Sheet1`.
- **Script Usage**:
  - Run scripts directly with Python (e.g. `python scripts/train_and_predict_xgb_multi.py`)
  - Optionally specify input file and sheet: `python scripts/train_and_predict_xgb_multi.py "data/dataset.xlsx" "Sheet1"`
- **Directory Creation**: Scripts auto-create `data/`, `models/`, and `outputs/` if missing.
- **Targets**: Canonical output columns:
  - Cylinder compressive strength (MPa)
  - Splitting tensile strength (MPa)
  - Flexural strength (MPa)
  - Elastic modulus (GPa)
- **Feature Selection**: Numeric columns except target(s) and other outputs are used as features.

## Conventions & Patterns
- **Model Saving**: XGBoost models saved as JSON in `models/`; features list in `xgb_features.txt`.
- **Metrics**: MAE and R² printed and saved to `metrics_multi.txt`.
- **Naming**: Use slugified target names for model files (e.g. `xgb_cylinder_compressive_strength_mpa_model.json`).
- **Imputation**: Median imputation for missing values.
- **Random State**: Fixed at 42 for reproducibility.

## Integration Points
- **External Libraries**: `pandas`, `numpy`, `scikit-learn`, `xgboost`.
- **No explicit test or build system**: Scripts are run directly; outputs are written to disk.

## Example: Adding a New Model Script
- Follow the structure in `train_and_predict_xgb_multi.py` or `model_train_rf.py`.
- Use canonical target names and save models/metrics in the appropriate folders.
- Ensure new scripts create required directories if missing.

---

**If any conventions or workflows are unclear, please provide feedback so this guide can be improved.**
