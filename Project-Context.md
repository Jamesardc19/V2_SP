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

## 8. Planned pipeline changes (agreed to implement later, not yet run)
Run in one pass so the artifacts are regenerated only once. V2 outputs are overwritten, so back up `MAIN_TRAINED_MODELS/`, `MAIN_TRAINED_MODELS_CALIB/` and the output folders to `ARCHIVE/` first (the models are not in git).
1. Drop `drnk_30d_num` (22 to 21 predictors). Reason: 50.6% blank, plus the undocumented `999999` code in 26% of rows (about 23% real values). A test showed no metric change when it is removed.
2. Recode ENNS skip logic instead of mode imputation:
   - `alcohol` = 0 implies `con_alcohol` = 0 and `drnk_30days` = 0. `con_alcohol` = 0 implies `drnk_30days` = 0.
   - `ever_smk` blank for current smokers gets its own code or is derived. Remaining blanks get an explicit "missing" category.
3. SMOTE implementation stays as in V2 (applied to the full dataset before the 70/15/15 split). This is a documented limitation, not changed.
4. `main_calibration.py`: add `knn` to `models_to_calibrate`, add ECE to `compute_metrics`, rank calibrators on validation data, not test.
5. Fix Chapter 4 table row for `pa_met` (binary 0/1, categorical), update every "22 predictors" mention (Chapter 4, README, results summary, notebook), and refresh `THESIS_RESULTS_SUMMARY.md` with the new results.
6. Open decisions for the web app: decision threshold (0.5 now, re-tune on validation after the rerun), LIME/CE background data (do not bundle record-level ENNS data in a public deployment), SHAP for Stacking.
7. Data use: the ENNS data was formally requested and approved for academic research, so using it (including in the web app) is acceptable. Keep acknowledging DOST-FNRI as the source, do not attempt re-identification, and keep record-level data on the server, not in the browser. `raw_merged.csv` and `preprocessed_*.csv` remain tracked in the repo.
8. Retain the anthropometric and lipid measurements: `weight`, `height`, `bmi`, `waist`, `hip`, and `chol` stay as features (a person can supply them, and `chol` is clinically distinct from `ldl`). A test showed retaining `chol`, `hip` and `bmi` does not hurt performance (changes of about +0.001 to +0.002 ROC-AUC). The correlation filter (|r| > 0.80) therefore needs a protected-feature list. Still drop the derived `tri_hdl_ratio`, `bmi_tri_interact` and the duplicate `alcohol_status`. Proposed feature count: 26. Document the exemption in Chapter 4.
9. Calibration and CE outputs must be fully regenerated after items 1 to 8 (features and models change). Gaps to close in that pass: kNN not calibrated, no ECE anywhere, Stacking and Voting lack `factual_feature_weights.csv`, Stacking and Voting counterfactual outputs cover only 12 of 24 narratives, several Ensured CCE plots are missing, Mondrian groups are labeled `group_a`/`group_b` instead of Male/Female.

## 10. Implementation status of section 8 (2026-11-18)
Implemented in the scripts, re-run pending (the user runs it in the terminal). The V2 outputs were copied to `ARCHIVE/V2_final_backup/` first.
- `main_preprocessing.py`: all changes above except SMOTE, which is unchanged (V2 behavior). A test run on the real data gave 26 features.
- `main_calibration.py`: kNN, ECE, calibrated F2, validation-CV method selection.
- `ce_utils.py` / `main_ce.py` (sex index from feature list), `ce_mondrian.py` (Male/Female labels), `ce_cce.py` / `ce_factual.py` (clear stale files).
- `main_train.py`: `tabnet_only` set to False (it was True and would have trained only BO-TabNet).
- Correction to item 9: the earlier "gaps" in Stacking/Voting CE outputs were not real. They came from stale files left by older runs and a CSV that only the old combined script wrote.
- Run order: `main_preprocessing.py`, `main_eda_extra.py` (optional), `main_train.py`, `main_xai.py`, `main_calibration.py`, then `ce_factual.py`, `ce_cce.py`, `ce_global.py`, `ce_mondrian.py`.
- Still to do after the run: update Chapter 4 (26 predictors, protected-feature correlation filter, `pa_met` as categorical), the README, `THESIS_RESULTS_SUMMARY.md` and `V2_Pipeline.ipynb`.

## 11. Re-run completed (2026-11-19)
- Preprocessing, training, calibration and CE re-run. 26 predictors, SMOTE unchanged, 10 models calibrated. New metrics are in `MAIN_MODEL_RESULTS/model_results.csv` and `HANDOFF_DIASCOPE_AI.md`.
- Chapter 4 updated (26 predictors, skip-logic recoding, protected correlation filter, calibration selection). The notebook and README are not updated yet by request.
- Stale: `MAIN_XAI/` (SHAP and LIME still from V2, `main_xai.py` not producing new files) and `MAIN_CE_OUTPUT/global/pre_ce_calibration_quality.csv` (written only by the old `main_ce.py`).
- Temperature-scaled calibrators are pickled under `__main__`, so the app must define `TemperatureScaledModel` (and `TabNetSklearnWrapper`) before loading them.

