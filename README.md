# Early Detection of the Risk of Diabetes Using Calibrations on Explainable Artificial Intelligence (XAI) with Philippine Data

This repository contains the full machine learning pipeline for early diabetes risk detection using the 2018–2021 Philippine Expanded National Nutrition Survey (ENNS) dataset.

## Project Overview

This study develops, evaluates, and explains machine learning models for diabetes risk prediction using Philippine population data. The pipeline covers:

1. Multi-dataset preprocessing with selective feature engineering
2. Hyperparameter-tuned training of 10 ML/DL models
3. Explainability via SHAP and LIME (XAI)
4. Probability calibration (Platt Scaling, Isotonic Regression, Temperature Scaling)

**Best model by F2-score**: kNN (Tuned) — F2=0.827, Recall=87.2%  
**Best model by discrimination**: Stacking Ensemble — ROC-AUC=0.857, AUPRC=0.870

---

## Directory Structure

```
├── DATASETS/
│   └── 2018-2021/
│       └── MAIN DATASET/
│           ├── data-set_anthrop.csv        # Anthropometric measurements
│           ├── data-set_biochemical.csv    # Biochemical/lipid panel
│           ├── data-set_clinical.csv       # Clinical (BP, hemoglobin, etc.)
│           ├── data-set_dietary_indiv.csv  # Dietary intake (excluded by default)
│           └── data-set_socio.csv          # Socioeconomic (age, sex)
│
├── MAIN_PREPROCESS_OUTPUT/                 # Preprocessed arrays, scalers, feature list
├── MAIN_TRAINED_MODELS/                    # Saved model files (.joblib / .zip)
├── MAIN_MODEL_RESULTS/                     # Metrics CSV, confusion matrices
├── MAIN_XAI/
│   ├── SHAP/                               # SHAP plots and global_importance.csv
│   └── LIME/                               # LIME local explanation plots
├── MAIN_CALIBRATION/                       # Calibration plots and comparison CSV
│
├── preprocessing_pipeline_v7.py            # Core preprocessing utilities
├── main_preprocessing.py                   # Pipeline entry point (preprocessing)
├── main_train.py                           # Pipeline entry point (training)
├── main_xai.py                             # Pipeline entry point (XAI)
└── main_calibration.py                     # Pipeline entry point (calibration)
```

---

## Requirements

```
pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
xgboost lightgbm catboost pytorch-tabnet torch optuna shap lime joblib
```

Install all at once:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost lightgbm catboost pytorch-tabnet optuna shap lime joblib
```

---

## Pipeline Execution

Run scripts in this order:

### Step 1 — Preprocessing

```bash
python main_preprocessing.py
```

**What it does:**
1. Loads and merges anthropometric, biochemical, and clinical datasets
2. Integrates socioeconomic dataset (adds `age` and `sex`)
3. Recodes special survey missing codes to `NaN`
4. Creates binary target: `diabetes = 1` if FBS ≥ 100 mg/dL
5. Applies **selective feature engineering**:
   - `bmi` from weight ÷ (height/100)² — drops weight and height
   - `tri_hdl_ratio`, `ldl_hdl_ratio`, `bmi_tri_interact` — kept alongside raw lipid features
6. KNN imputation (k optimized) for continuous; mode imputation for categorical
7. IQR-based outlier capping on continuous features
8. StandardScaler fitted on training data only
9. Explicit drop of `chol` before correlation filter (retains `ldl`)
10. Correlation-based feature dropping (threshold = 0.80)
11. SMOTE applied to full dataset before 70/15/15 re-split

**Outputs** → `MAIN_PREPROCESS_OUTPUT/`:
- `X_train.npy`, `X_val.npy`, `X_test.npy`, `y_train.npy`, `y_val.npy`, `y_test.npy`
- `feature_list.csv` — 22 final features
- `preprocessing_info.pkl` — scaler, imputer, feature metadata
- `EDA/` — demographic and descriptive plots

### Step 2 — Training

```bash
python main_train.py
```

Trains 10 models with RandomizedSearchCV (traditional ML) and Optuna BO (TabNet):

| # | Model | Optimization |
|---|-------|-------------|
| 1 | BO-TabNet | Optuna Bayesian Optimization (30 trials) |
| 2 | Naive Bayes | — |
| 3 | kNN | RandomizedSearchCV (F2, 5-fold) |
| 4 | AdaBoost | RandomizedSearchCV |
| 5 | XGBoost | RandomizedSearchCV (30 iter) |
| 6 | Random Forest | RandomizedSearchCV (30 iter) |
| 7 | CatBoost | RandomizedSearchCV (30 iter) |
| 8 | LightGBM | RandomizedSearchCV (30 iter) |
| 9 | Stacking Ensemble | RF + XGB + LightGBM → Logistic Regression meta |
| 10 | Voting Ensemble | RF + XGB + LightGBM (soft vote) |

**Config flags** (in `main_train.py → Config`):

| Flag | Default | Description |
|------|---------|-------------|
| `skip_tabnet` | `False` | Skip TabNet, load prior metrics from CSV |
| `tabnet_only` | `False` | Train only TabNet, load other model metrics from CSV |

**Outputs** → `MAIN_MODEL_RESULTS/` and `MAIN_TRAINED_MODELS/`

### Step 3 — Explainability (XAI)

```bash
python main_xai.py
```

- **SHAP**: TreeExplainer for tree models, KernelExplainer for BO-TabNet
  - Global bar, beeswarm, dependency, waterfall plots per model
  - `global_importance.csv` — mean SHAP across all models
- **LIME**: Local explanations for BO-TabNet and Stacking Ensemble

**Outputs** → `MAIN_XAI/SHAP/` and `MAIN_XAI/LIME/`

### Step 4 — Calibration

```bash
python main_calibration.py
```

Applies three calibration methods per model:
- **Sigmoid** (Platt Scaling)
- **Isotonic Regression**
- **Temperature Scaling**

Evaluates using Brier score and log-loss. Saves calibrated models and comparison CSV.

**Outputs** → `MAIN_CALIBRATION/`

---

## Final Results

### Model Performance (Test Set)

| Model | F2 | Recall | Precision | ROC-AUC | AUPRC | Accuracy |
|-------|----|--------|-----------|---------|-------|----------|
| **kNN (Tuned)** | **0.827** | **87.2%** | 68.7% | 0.817 | 0.759 | 73.7% |
| AdaBoost (Tuned) | 0.796 | 86.6% | 60.3% | 0.728 | 0.684 | 64.8% |
| Random Forest (Tuned) | 0.783 | 79.7% | 73.3% | 0.839 | 0.838 | 75.4% |
| **Stacking Ensemble** | 0.768 | 76.8% | **77.1%** | **0.857** | **0.870** | **77.0%** |
| Voting Ensemble | 0.768 | 77.8% | 73.1% | 0.831 | 0.837 | 74.6% |
| BO-TabNet | 0.749 | 77.4% | 66.6% | 0.757 | 0.727 | 69.3% |
| LightGBM (Tuned) | 0.748 | 75.1% | 73.4% | 0.828 | 0.843 | 74.0% |
| CatBoost (Tuned) | 0.747 | 74.4% | 75.7% | 0.841 | 0.858 | 75.3% |
| XGBoost (Tuned) | 0.745 | 75.4% | 71.1% | 0.809 | 0.814 | 72.4% |
| Naive Bayes | 0.742 | 77.6% | 63.3% | 0.722 | 0.692 | 66.3% |

### Calibration Results (Best Method per Model)

| Model | ROC-AUC | Best Log-Loss | Method | Brier Score |
|-------|---------|---------------|--------|-------------|
| **Stacking** | **0.857** | **0.470** | Isotonic | **0.154** |
| CatBoost | 0.841 | 0.482 | Isotonic | 0.162 |
| Random Forest | 0.839 | 0.498 | Sigmoid | 0.164 |
| LightGBM | 0.828 | 0.501 | Isotonic | 0.169 |
| Voting | 0.831 | 0.504 | Isotonic | 0.168 |
| XGBoost | 0.809 | 0.529 | Isotonic | 0.179 |
| BO-TabNet | 0.757 | 0.585 | Sigmoid | 0.200 |
| AdaBoost | 0.728 | 0.607 | Isotonic | 0.209 |
| Naive Bayes | 0.722 | 0.616 | Sigmoid | 0.213 |

### Top SHAP Features (Mean Across Models)

| Rank | Feature | Mean SHAP | Clinical Meaning |
|------|---------|-----------|-----------------|
| 1 | `age` | 0.330 | Age — strongest metabolic risk factor |
| 2 | `waist` | 0.214 | Waist circumference — central obesity |
| 3 | `Ave_SBP` | 0.160 | Systolic BP — hypertension indicator |
| 4 | `pa_met` | 0.087 | Physical activity — metabolic equivalent |
| 5 | `tri` | 0.078 | Triglycerides — lipid metabolism |
| 6 | `drnk_30d_num` | 0.067 | Alcohol frequency |
| 7 | `hemoglobin` | 0.055 | Hematologic marker |
| 8 | `Ave_DBP` | 0.040 | Diastolic BP |
| 9 | `sex` | 0.034 | Biological sex |
| 10 | `ldl_hdl_ratio` | 0.021 | Engineered lipid ratio |

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| SMOTE timing | Before re-split (on full data) | Avoids leakage while ensuring balanced classes in all splits |
| Feature engineering | Selective (ldl_hdl_ratio retained) | bmi, tri_hdl_ratio, bmi_tri_interact removed by correlation filter (r > 0.80) |
| chol vs ldl | chol dropped pre-correlation | Retains the more specific `ldl` marker |
| Population filter | None — all age groups included | Broader coverage aligns with national screening scope |
| Primary metric | F2-score (β=2) | Recall weighted 2× — missing a diabetic case is costlier than a false positive |
| Recommended deployment model | Stacking Ensemble + Isotonic calibration | Highest discrimination (AUC=0.857) with reliable probability estimates |

---

## Reproducibility

- Fixed random seed: `42` across all scripts
- Train / Val / Test split: **70 / 15 / 15**
- All preprocessing fitted on training data only (no leakage)
- Hyperparameters logged in `MAIN_MODEL_RESULTS/model_results.csv`

---

## Citation

```
Early Detection of the Risk of Diabetes Using Calibrations on Explainable
Artificial Intelligence (XAI) with Philippine Data
2018–2021 ENNS Dataset — FNRI, Department of Science and Technology
```
