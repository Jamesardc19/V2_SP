# Project Context

Last updated: 2026-11-03

## 1. Overview
- **Title:** Early Detection of the Risk of Diabetes Using Calibrated Explanations and Explainable Artificial Intelligence (XAI) with Philippine Data
- **Author:** James Angelo R. Dela Cruz; **Adviser:** Geoffrey A. Solano
- **Affiliation:** Department of Physical Sciences and Mathematics, UP Manila (CMSC 199, BS Computer Science, June 2026)
- **Repository:** https://github.com/Jamesardc19/V2_SP (public, branch `main`)
- **Data:** 2018-2021 Philippine Expanded National Nutrition Survey (ENNS), DOST-FNRI
- **Target:** `diabetes = 1` if FBS >= 100 mg/dL (pre-diabetes and diabetes combined)
- **Primary metric:** F2 (beta = 2); **seed:** 42

## 2. Final pipeline (V2 Hybrid)
| Step | Script | Output folder |
|---|---|---|
| 1 Preprocessing | `main_preprocessing.py`, `preprocessing_pipeline_v7.py` | `MAIN_PREPROCESS_OUTPUT/` |
| 1b Extra EDA | `main_eda_extra.py` | `MAIN_PREPROCESS_OUTPUT/EDA/eda2_*` |
| 2 Training (10 models) | `main_train.py` | `MAIN_TRAINED_MODELS/`, `MAIN_MODEL_RESULTS/` |
| 3 SHAP + LIME | `main_xai.py` | `MAIN_XAI/` |
| 4 Calibration (Platt, isotonic, temperature) | `main_calibration.py` | `MAIN_CALIBRATION/`, `MAIN_TRAINED_MODELS_CALIB/` |
| 5 Calibrated Explanations | `main_ce.py` + `ce_utils/factual/cce/global/mondrian.py` | `MAIN_CE_OUTPUT/` |

Preprocessing: merge anthropometric, biochemical, clinical and socio (age, sex) data; dietary excluded; drop rows with missing FBS; selective feature engineering (only `ldl_hdl_ratio` survives the correlation filter); KNN/mode imputation; IQR capping; StandardScaler; correlation filter at |r| > 0.80; SMOTE (strategy 1.0, k = 5); stratified 70/15/15 split. Final feature set: 22 features.

Models: BO-TabNet, Naive Bayes, kNN, AdaBoost, XGBoost, Random Forest, CatBoost, LightGBM, Stacking, Voting.

## 3. Key results (V2, see `THESIS_RESULTS_SUMMARY.md`)
- Best F2: kNN (tuned), F2 = 0.827, recall 87.2%
- Best discrimination: Stacking, ROC-AUC 0.857, AUPRC 0.870
- Stacking isotonic log-loss 0.470
- CE run on 9 models: factual, counterfactual (CCE and Ensured), global uncertainty, Mondrian sex-stratified
- Earlier Rivera-framework comparisons (V7 vs V7-Rivera, 2013 vs 2018-2021) are in `PIPELINE_COMPARISON.md`

## 4. Decisions
- F2 as primary metric: false negatives are costlier than false positives in screening.
- Dietary data excluded (noise, out of scope). `age` and `sex` added from the socio dataset (improved every model).
- Hybrid analysis: SHAP for global ranking, CE for per-patient uncertainty and counterfactuals, Mondrian CE for sex equity.
- Trained models are NOT pushed to GitHub (about 4.2 GB each folder); `ARCHIVE/` (about 33 GB) is local only.
- `Thesis Manuscript/Reference Papers/` is untracked (copyright); files remain local.
- Raw ENNS datasets (`DATASETS/`) and all outputs are pushed to the public repo for thesis documentation (user decision, 2026-11-03). Cite DOST-FNRI as source and confirm FNRI terms of use.
- LaTeX build artifacts are git-ignored.
- `V2_Pipeline.ipynb` documents the V2 pipeline as-is (including its known issues).

## 5. Completed
- Full V2 pipeline run and outputs saved
- `main_eda_extra.py` and eda2 outputs
- CE outputs (`MAIN_CE_OUTPUT/`)
- `V2_Pipeline.ipynb` (218 cells, intact; confirmed not corrupted)
- Thesis proposal chapters 1-4 in `Thesis Manuscript/Proposal Template/` (modified, tracked), Chapter 4 DrawIO diagrams
- `.gitignore` updated; commit staged locally (not pushed at time of writing)

## 6. Known issues
1. **SMOTE leakage:** V2 applies SMOTE to the full dataset, then splits 70/15/15, so val/test contain synthetic instances. Reported V2 metrics are optimistic. `THESIS_RESULTS_SUMMARY.md` describes the method as "no-leakage", which contradicts this and should be corrected.
2. **Calibrator ranking on test data:** `main_calibration.py` selects calibrators on the test set.
3. **Duplicate dataset files:** `DATASETS/2018-2021/MAIN DATASET/` duplicates the per-component folders (about 230 MB extra).
4. **IDE renderer error:** after moving from Windsurf to Devin Desktop, markdown cells in notebooks fail with "Error loading renderer vscode.markdown-it-renderer". The notebook file is fine. Tried or suggested: reload webviews/window, full restart, clear cache folders in `%APPDATA%`, unblock `index.js`, reinstall.
5. Chat history from Windsurf was lost in the migration; this file replaces it.
6. Older helper scripts open in the editor (`model_training_revised.py`, `create_feature_mapping.py`, `apply_calibration_fixed.py`, `revised_preprocessing.py`, `improved_preprocessing.py`); they live under `ARCHIVE/` (git-ignored) and are not part of V2.

## 7. Upcoming tasks
- Re-run the pipeline with SMOTE applied to the training split only (reference: `ARCHIVE/notebooks/SMOTE_Corrected_Pipeline.ipynb`) and compare against V2 metrics.
- Update the `V2_Pipeline.ipynb` header with the V2 vs. corrected comparison table once the corrected results exist.
- Fix the "no-leakage" wording in `THESIS_RESULTS_SUMMARY.md`.
- Rank calibrators on validation data instead of test data.
- Upload trained models to an external host (Google Drive, Hugging Face or a GitHub Release) and link them in `README.md`.
- Finish thesis chapters 5 onward using the final results.
- Push the staged commit when ready.
