## Neural activity decoding

Predict trial types (GO/NOGO × lick/reward) from calcium activity traces recorded across cortical areas, layers, and cell types.

### Repository layout
- `data/raw/train.csv`, `data/raw/test.csv` — tabular time-series features (≈30 time bins per population) with `session_id`, `trial_number`, and `TRIAL_TYPE` (train only).
- `data/raw/sample_submission.csv` — Kaggle-style template; generated submissions live in `data/` or notebook folder.
- `notebooks/mini_project_ml_leo_ammar.ipynb` — main classical ML pipeline (imputation, per-session z-score, ANOVA feature selection, logistic regression/forest/XGB sweeps, LOSO evaluation, plotting).

### Environment
Python 3.10+. Core deps: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `catboost`, `torch`, `jupyterlab`, `ipykernel`.

Quick start (pip, from repo root):
```bash
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Conda (uses `environment.yml`):
```bash
conda env create -f environment.yml
conda activate biological-data-science-ii
python -m ipykernel install --user --name biological-data-science-ii
```

### Data preprocessing
- Drop constant columns.
- Imputation: tested zero/mean, KNN (k=1/2/5/10), iterative; zero/mean best for linear baseline, KNN used in XGB variants.
- Normalization: **per-session z-score** (fit on train fold only) prevents leakage; global z-score inflates CV.
- Feature selection: `SelectKBest` ANOVA, best around top-400 features for linear/XGB.
- Optional features: windowed AUC aggregates; baseline subtraction for visualization.

### Models & best settings
- **Logistic Regression (OvR)**: per-session z-score → ANOVA top-400 → zero/mean imputer; L1 with C≈0.03 (flat 0.02–0.05), L2 around C≈7e-4. Evaluated with LOSO; submission example `data/submission_logreg_linear_pipeline.csv`.
- **XGBoost variants**: z-score per session, KNN imputer (k=1 or 2), top-400 features; e.g., 500 trees depth 7 lr 0.10, or depth 9 lr 0.03, saved as `submission__xgb_...csv`.


### How to reproduce
- **Linear/XGB**: run `notebooks/mini_project_ml_leo_ammar.ipynb` cells for preprocessing, feature selection, and model training. 


