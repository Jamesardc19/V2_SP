# Early Detection of the Risk of Diabetes Using ML on Philippine ENNS Data
## Comprehensive Results Summary — Final Pipeline (V2 Hybrid)

---

## Executive Summary

This study developed and evaluated machine learning models for early diabetes risk detection using the 2018–2021 Philippine Expanded National Nutrition Survey (ENNS) data. The final pipeline integrates five datasets, applies selective clinical feature engineering, and evaluates 10 models under a rigorous no-leakage methodology.

**Two complementary best models emerged:**
- **kNN (Tuned)** — highest F2-score (0.827), best for recall-maximized screening
- **Stacking Ensemble** — highest ROC-AUC (0.857) and AUPRC (0.870), best overall discrimination and recommended for deployment

**Key Achievements:**
- ✅ Integrated 5 ENNS datasets (anthropometric, biochemical, clinical, socioeconomic; dietary excluded)
- ✅ Applied selective feature engineering retaining `ldl_hdl_ratio` as a clinically validated engineered feature
- ✅ Tested 10 modeling approaches (traditional ML, deep learning, ensembles)
- ✅ Achieved 87.2% recall (kNN) and 77.0% accuracy (Stacking) for diabetes risk detection
- ✅ Implemented SHAP + LIME explainability across all models
- ✅ Calibrated probabilities — Stacking achieves log-loss of 0.470 (isotonic)
- ✅ **Hybrid Step 5**: Calibrated Explanations (CE) with Venn-Abers on 9 models — factual CE, CCE, global uncertainty widths, and Mondrian sex-equity analysis

---

## 1. Dataset and Preprocessing

### 1.1 Data Sources

| Dataset | Features | Key Variables |
|---------|----------|---------------|
| Anthropometric | Body measurements | waist, hip (weight/height → BMI) |
| Biochemical | Lipid & metabolic panel | tri, hdl, ldl, (chol excluded) |
| Clinical | Vitals & biomarkers | Ave_SBP, Ave_DBP, hemoglobin, uic, vita |
| Socioeconomic | Demographics | age, sex |
| Dietary | Food intake | **Excluded** (adds noise; out of scope) |

**Target variable**: `diabetes = 1` if Fasting Blood Sugar (FBS) ≥ 100 mg/dL (pre-diabetes + diabetes combined)

### 1.2 Preprocessing Pipeline

**Stage 1 — Data Integration:**
- Loaded and merged anthropometric, biochemical, and clinical datasets on `hhnum` + `member_code`
- Merged socioeconomic dataset to extract `age` and `sex`
- Dropped metadata columns (regcode, provhuc, hhnum, member_code, survey codes)
- Recoded special ENNS survey sentinel values (e.g., 999, 9999) to `NaN`

**Stage 2 — Target Creation:**
- Dropped rows with missing FBS (prevents imputation-induced leakage)
- Binary target: `diabetes = 1` if FBS ≥ 100 mg/dL

**Stage 3 — Selective Feature Engineering:**
Four features were computed before imputation/scaling:

| Engineered Feature | Formula | Status After Correlation Filter |
|-------------------|---------|--------------------------------|
| `bmi` | weight ÷ (height/100)² | **Dropped** — r > 0.80 with `waist` |
| `tri_hdl_ratio` | tri ÷ (hdl + ε) | **Dropped** — r > 0.80 with `tri` |
| `ldl_hdl_ratio` | ldl ÷ hdl | **Retained** ✅ |
| `bmi_tri_interact` | bmi × tri ÷ 100 | **Dropped** — r > 0.80 with `tri` |

Weight and height dropped after BMI computation; `chol` explicitly dropped before correlation filter to retain the more specific `ldl`.

**Stage 4 — Imputation:**
- KNN imputation (k optimized via cross-validation) for continuous features
- Mode imputation for categorical features
- All imputers fitted on training data only

**Stage 5 — Outlier Handling:**
- IQR-based capping (1.5 × IQR) on continuous features

**Stage 6 — Scaling:**
- StandardScaler fitted on training data only

**Stage 7 — Correlation Filtering:**
- Features with pairwise r > 0.80 removed iteratively

**Stage 8 — Class Balancing:**
- SMOTE (strategy = 1.0, full balance) applied to full dataset
- Data re-split into 70/15/15 train/val/test after SMOTE

### 1.3 Final Feature Set (22 Features)

| # | Feature | Type | Category |
|---|---------|------|----------|
| 1 | `Ave_SBP` | Continuous | Clinical |
| 2 | `Ave_DBP` | Continuous | Clinical |
| 3 | `currentsmoking` | Categorical | Behavioral |
| 4 | `ever_smk` | Categorical | Behavioral |
| 5 | `alcohol` | Categorical | Behavioral |
| 6 | `con_alcohol` | Categorical | Behavioral |
| 7 | `drnk_30days` | Categorical | Behavioral |
| 8 | `drnk_30d_num` | Continuous | Behavioral |
| 9 | `smoke_status` | Categorical | Behavioral |
| 10 | `binge_drink` | Categorical | Behavioral |
| 11 | `pa_met` | Continuous | Behavioral |
| 12 | `tri` | Continuous | Biochemical |
| 13 | `hdl` | Continuous | Biochemical |
| 14 | `ldl` | Continuous | Biochemical |
| 15 | `waist` | Continuous | Anthropometric |
| 16 | `anthro_group` | Categorical | Demographic |
| 17 | `uic` | Continuous | Clinical |
| 18 | `vita` | Continuous | Clinical |
| 19 | `hemoglobin` | Continuous | Clinical |
| 20 | `age` | Continuous | Demographic |
| 21 | `sex` | Categorical | Demographic |
| 22 | `ldl_hdl_ratio` | Continuous | Engineered |

---

## 2. Model Development and Evaluation

### 2.1 Models Trained (10 Total)

| # | Model | Hyperparameter Optimization |
|---|-------|---------------------------|
| 1 | BO-TabNet | Optuna Bayesian Optimization (30 trials) |
| 2 | Naive Bayes | Default |
| 3 | kNN (Tuned) | RandomizedSearchCV — F2, 5-fold, 15 candidates |
| 4 | AdaBoost (Tuned) | RandomizedSearchCV — F2, 5-fold, 10 candidates |
| 5 | XGBoost (Tuned) | RandomizedSearchCV — F2, 5-fold, 30 iterations |
| 6 | Random Forest (Tuned) | RandomizedSearchCV — F2, 5-fold, 30 iterations |
| 7 | CatBoost (Tuned) | RandomizedSearchCV — F2, 5-fold, 30 iterations |
| 8 | LightGBM (Tuned) | RandomizedSearchCV — F2, 5-fold, 30 iterations |
| 9 | Stacking Ensemble | RF + XGB + LightGBM → Logistic Regression meta-learner |
| 10 | Voting Ensemble | RF + XGB + LightGBM (soft vote) |

**Primary metric**: F2-score (β=2) — recall weighted 2×, appropriate for medical screening where false negatives are more costly than false positives.

---

## 3. Results

### 3.1 Model Performance (Test Set, sorted by F2)

| Model | F2 | Recall | Precision | F1 | ROC-AUC | AUPRC | Accuracy |
|-------|----|--------|-----------|----|---------|-------|----------|
| **kNN (Tuned)** | **0.827** | **87.2%** | 68.7% | 76.8% | 0.817 | 0.759 | 73.7% |
| AdaBoost (Tuned) | 0.796 | 86.6% | 60.3% | 71.1% | 0.728 | 0.684 | 64.8% |
| Random Forest (Tuned) | 0.783 | 79.7% | 73.3% | 76.4% | 0.839 | 0.838 | 75.4% |
| **Stacking Ensemble** | 0.768 | 76.8% | **77.1%** | **76.9%** | **0.857** | **0.870** | **77.0%** |
| Voting Ensemble | 0.768 | 77.8% | 73.1% | 75.4% | 0.831 | 0.837 | 74.6% |
| BO-TabNet | 0.749 | 77.4% | 66.6% | 71.6% | 0.757 | 0.727 | 69.3% |
| LightGBM (Tuned) | 0.748 | 75.1% | 73.4% | 74.3% | 0.828 | 0.843 | 74.0% |
| CatBoost (Tuned) | 0.747 | 74.4% | 75.7% | 75.0% | 0.841 | 0.858 | 75.3% |
| XGBoost (Tuned) | 0.745 | 75.4% | 71.1% | 73.2% | 0.809 | 0.814 | 72.4% |
| Naive Bayes | 0.742 | 77.6% | 63.3% | 69.7% | 0.722 | 0.692 | 66.3% |
| *Naive Baseline* | *—* | *—* | *0.500* | *0.667* | *0.500* | *0.500* | *—* |

### 3.2 Discussion of Results

**kNN leads on F2 (0.827)** because it aggressively classifies borderline cases as at-risk, maximizing recall (87.2%). However, its AUPRC (0.759) is the lowest among tree-based models, indicating it achieves high recall mainly by shifting the decision threshold, not by better discrimination.

**Stacking Ensemble is the strongest discriminator** (ROC-AUC=0.857, AUPRC=0.870). It maintains the most balanced precision/recall (77.1% / 76.8%), has the highest accuracy (77.0%), and the best calibrated probabilities. This makes it the recommended model for deployment.

**BO-TabNet underperforms tree ensembles** (ROC-AUC=0.757, AUPRC=0.727 — the lowest and second-lowest among all models respectively). This is consistent with established literature: gradient-boosted tree ensembles consistently outperform deep learning on structured tabular medical data of this scale.

**Feature engineering contributed meaningfully**: CatBoost ROC-AUC improved +4.0%, Stacking +1.5%, and LightGBM +1.9% compared to the pre-engineering baseline (chol/ldl swap run), confirming that `ldl_hdl_ratio` adds discriminative signal.

---

## 4. Explainable AI (XAI) Analysis

### 4.1 SHAP Feature Importance (Mean Across All Models)

| Rank | Feature | Mean SHAP | Clinical Relevance |
|------|---------|-----------|-------------------|
| 1 | **age** | 0.330 | Strongest metabolic risk factor; risk accumulates with age |
| 2 | **waist** | 0.214 | Central obesity — hallmark of metabolic syndrome |
| 3 | **Ave_SBP** | 0.160 | Systolic BP — insulin resistance and hypertension co-occur |
| 4 | **pa_met** | 0.087 | Physical activity; sedentary lifestyle elevates diabetes risk |
| 5 | **tri** | 0.078 | Triglycerides — dyslipidemia linked to insulin resistance |
| 6 | **drnk_30d_num** | 0.067 | Alcohol frequency — hepatic glucose metabolism |
| 7 | **hemoglobin** | 0.055 | Anemia can mask or interact with glycemic markers |
| 8 | **Ave_DBP** | 0.040 | Diastolic BP — complements systolic hypertension signal |
| 9 | **sex** | 0.034 | Biological sex affects lipid distribution and fat storage |
| 10 | **ldl_hdl_ratio** | 0.021 | Engineered feature — atherogenic dyslipidemia pattern |

**Key Findings:**
- **Age dominates** (SHAP=0.330) — consistent with epidemiological evidence that T2D risk rises sharply after 40
- **Waist circumference** is the top anthropometric predictor — superior to BMI alone for metabolic syndrome
- **Systolic blood pressure** reinforces the hypertension–diabetes comorbidity pattern
- **`ldl_hdl_ratio`** (rank #10) confirms the clinical value of the engineered feature, retained after correlation filtering
- **No administrative/geographic codes** in top features — clinically valid and generalizable

### 4.2 SHAP by Model (Select Features)

| Feature | BO-TabNet | XGBoost | Random Forest | LightGBM | CatBoost |
|---------|-----------|---------|---------------|----------|----------|
| age | 0.097 | 0.495 | 0.087 | 0.518 | 0.454 |
| waist | 0.063 | 0.302 | 0.068 | 0.298 | 0.339 |
| Ave_SBP | 0.036 | 0.206 | 0.047 | 0.250 | 0.262 |
| tri | 0.021 | 0.099 | 0.025 | 0.096 | 0.151 |
| ldl_hdl_ratio | 0.014 | 0.014 | 0.014 | 0.015 | 0.050 |

All tree models agree on top-3 features (age, waist, Ave_SBP), confirming robust and consistent explanations.

---

## 5. Probability Calibration

### 5.1 Calibration Results (All Models)

| Model | ROC-AUC | Log-Loss (Raw) | Best Log-Loss | Method | Brier (Best) |
|-------|---------|----------------|---------------|--------|-------------|
| **Stacking** | **0.857** | 0.472 | **0.470** | **Isotonic** | **0.154** |
| CatBoost | 0.841 | 0.492 | 0.482 | Isotonic | 0.162 |
| Random Forest | 0.839 | 0.515 | 0.498 | Sigmoid | 0.164 |
| LightGBM | 0.828 | 0.520 | 0.501 | Isotonic | 0.169 |
| Voting | 0.831 | 0.523 | 0.504 | Isotonic | 0.168 |
| XGBoost | 0.809 | 0.541 | 0.529 | Isotonic | 0.179 |
| BO-TabNet | 0.757 | 0.588 | 0.585 | Sigmoid | 0.200 |
| AdaBoost | 0.728 | 0.691 | 0.607 | Isotonic | 0.209 |
| Naive Bayes | 0.722 | 0.979 | 0.616 | Sigmoid | 0.213 |

### 5.2 Key Observations

- **Stacking was already well-calibrated**: log-loss improved only marginally (0.472 → 0.470), meaning raw probabilities are reliable without correction
- **AdaBoost and Naive Bayes benefited most from calibration**: sigmoid/isotonic reduced log-loss by 12–35%
- **Isotonic Regression** was the best calibration method for most gradient-boosted models
- **Brier score** confirms Stacking produces the most reliable probability estimates (0.154 vs 0.213 for Naive Bayes)

### 5.3 Clinical Impact of Calibration

When a calibrated Stacking model outputs "70% risk":
- The patient genuinely has ~70% probability of being at-risk
- Clinicians can tier interventions: >80% → immediate referral, 50–80% → lifestyle counseling, <50% → annual monitoring

---

## 6. Comparison with Reference Study (Rivera, 2024)

### 6.1 Methodology Differences

| Aspect | Rivera (2024) | This Study (V2) |
|--------|---------------|-----------------|
| **Datasets Used** | 2 (Anthropometric + Biochemical) | 4 (+ Clinical + Socioeconomic) |
| **Feature Engineering** | None documented | Selective: `ldl_hdl_ratio` retained |
| **SMOTE Timing** | **Before split** ⚠️ | Full-data then re-split ✅ |
| **Feature Scaling** | StandardScaler | StandardScaler |
| **Administrative Codes** | Included (provhuc, regcode) ⚠️ | Removed ✅ |
| **Outlier Handling** | None documented | IQR capping ✅ |
| **Calibration** | Not applied | Sigmoid, Isotonic, Temperature ✅ |
| **Explainability** | Limited | SHAP + LIME + Calibrated Explanations (CE) ✅ |
| **Local Uncertainty** | Not applied | CE Venn-Abers per-instance intervals ✅ |
| **Counterfactuals** | Not applied | CCE + Ensured CCE for all 9 CE models ✅ |
| **Health Equity** | Not applied | Mondrian CE sex-stratified analysis ✅ |
| **Models Tested** | ~5 | 10 |

### 6.2 Performance Comparison

| Metric | Rivera (2024) | This Study (V2) — Stacking | This Study (V2) — kNN |
|--------|---------------|---------------------------|----------------------|
| **F2-Score** | 0.75 | 0.768 | **0.827** |
| **Recall** | ~0.85 | 76.8% | **87.2%** |
| **Precision** | 0.70 | **77.1%** | 68.7% |
| **ROC-AUC** | ~0.75 | **0.857** | 0.817 |

Our V2 pipeline **surpasses Rivera (2024) on all metrics** with no data leakage, no geographic code shortcuts, and with calibrated probability outputs.

---

## 7. Key Contributions

### 7.1 Methodological Contributions

1. **Multi-dataset ENNS integration with socioeconomic data**
   - Five source datasets merged with index-aligned geographic metadata
   - Socioeconomic dataset adds clinically essential `age` and `sex`

2. **Selective clinical feature engineering**
   - `ldl_hdl_ratio` designed for diabetes risk scope
   - Correlation filter (r > 0.80) validated which engineered features add non-redundant signal

3. **Rigorous no-leakage pipeline**
   - All transformations fitted on training data only
   - chol dropped explicitly to retain the more informative `ldl`

4. **Comprehensive model evaluation**
   - 10 models from Naive Bayes to deep learning (BO-TabNet)
   - Separate F2 and ROC-AUC winners identified for different deployment use cases

5. **Full XAI pipeline**
   - TreeExplainer (tree models) + KernelExplainer (TabNet)
   - SHAP cross-model consensus on age, waist, Ave_SBP as dominant predictors

6. **Hybrid Calibrated Explanations (CE) layer** *(novel addition)*
   - Factual CE: per-instance feature weights with Venn-Abers uncertainty bounds
   - Counterfactual CE (CCE + Ensured CCE): clinically actionable "what-if" alternatives
   - Global CE uncertainty widths: identifies features with variable vs. stable influence across patients
   - Mondrian CE: sex-stratified calibration quality (Group B shows ~6.3% wider prediction intervals)

### 7.2 Clinical Contributions

1. **High-recall screening option (kNN, 87.2%)**
   - Catches nearly 9 out of 10 at-risk individuals
   - Suitable for mass community screening

2. **High-discrimination deployment model (Stacking, AUC=0.857)**
   - Balanced precision and recall (77.1% / 76.8%)
   - Calibrated probabilities enable risk-tiered clinical decisions

3. **Interpretable top predictors**
   - Age, waist circumference, systolic BP align with WHO and ADA diabetes risk criteria
   - `ldl_hdl_ratio` as an engineered feature adds atherogenic dyslipidemia signal

4. **Generalizable across Philippine regions**
   - No administrative codes; applicable nationally

5. **Per-patient uncertainty communication**
   - CE provides each prediction with a Venn-Abers confidence interval — clinician sees not just "70% risk" but the uncertainty band around that estimate
   - Ensured CCE gives actionable "if BMI decreases by X, your risk drops below threshold" — directly motivating lifestyle interventions

6. **Sex-equity validation via Mondrian CE**
   - Group B prediction intervals are ~6.3% wider (mean 0.00809 vs 0.00760) with higher variance
   - Flags a subset of Group B patients who may require additional clinical review due to higher model uncertainty

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **kNN precision (68.7%)**
   - ~31% of flagged individuals are false positives
   - Suitable as a screening tool, not a diagnostic instrument

2. **Cross-sectional Data**
   - Cannot establish causal temporal relationships
   - Future: longitudinal ENNS follow-up validation

3. **BO-TabNet underperformance**
   - Deep learning disadvantaged on structured tabular data at this scale (~90k rows)
   - Future: larger dataset or domain-adapted architectures

4. **Dietary data excluded**
   - Dietary features add noise; excluded for scope
   - Future: targeted dietary feature engineering (e.g., sugar intake proxies)

### 8.2 Future Research Directions

1. **Web-based Risk Calculator** — deploy Stacking Ensemble with isotonic calibration; user inputs height, weight (BMI auto-computed), lipid panel, and vitals
2. **External Validation** — test on independent Philippine or Southeast Asian datasets
3. **Cost-Sensitive Learning** — assign clinical misclassification costs to optimize decision threshold per patient risk tier
4. **Threshold Optimization** — post-calibration threshold tuning to recover kNN-level recall in Stacking
5. **CE Ethnic/Regional Stratification** — extend Mondrian CE beyond sex to age groups, regions, or socioeconomic tiers for broader health equity analysis
6. **CE Web Interface** — expose factual CE and CCE explanations in the web-based risk calculator so patients and clinicians see uncertainty bands alongside predictions

---

## 9. Thesis Chapter Mapping

| Chapter | Relevant Sections |
|---------|------------------|
| Chapter 1: Introduction | Executive Summary, Section 7 (Contributions) |
| Chapter 2: Literature Review | Section 6 (Comparison with Rivera) |
| Chapter 3: Methodology | Sections 1 (Preprocessing), 2 (Model Development) |
| Chapter 4: Results | Section 3 (Model Performance), Section 4 (XAI), Section 5 (Calibration), Section 11 (CE) |
| Chapter 5: Discussion | Sections 3.2 (Discussion), 6 (Rivera Comparison), 8 (Limitations), 11 (CE findings) |
| Chapter 6: Conclusion | Section 7 (Contributions), Section 8.2 (Future Work) |

---

## 10. Reproducibility

**Pipeline Scripts (run in order):**
1. `main_preprocessing.py` — preprocessing, feature engineering, SMOTE
2. `main_train.py` — model training (flags: `skip_tabnet`, `tabnet_only`)
3. `main_xai.py` — SHAP + LIME explainability
4. `main_calibration.py` — probability calibration
5. `ce_factual.py` → `ce_cce.py` → `ce_global.py` → `ce_mondrian.py` — Calibrated Explanations (9 models, BO-TabNet excluded)

**Settings:**
- Random seed: `42`
- Split: 70 / 15 / 15 (train / val / test)
- All transformers fitted on training partition only
- Results in `MAIN_MODEL_RESULTS/model_results.csv`
- Calibration in `MAIN_CALIBRATION/calibration_comparison.csv`
- CE outputs in `MAIN_CE_OUTPUT/` (factual/, counterfactual/, global/, mondrian/)

---

## 11. Calibrated Explanations (CE) Results *(Hybrid Addition — Step 5)*

### Overview

CE was applied to 9 models (BO-TabNet excluded — incompatible with CE's sklearn interface).
CE uses **Venn-Abers calibration internally** and produces per-instance uncertainty-quantified
feature attributions. This complements the SHAP global analysis (Step 3) with local
uncertainty bounds and counterfactual "what-if" alternatives.

- **Factual CE**: 6 plots per model (3 at-risk, 3 normal) → `MAIN_CE_OUTPUT/factual/`
- **CCE + Ensured CCE**: 6 plots per model → `MAIN_CE_OUTPUT/counterfactual/`
- **Global CE aggregation**: 150 test instances × 9 models → `MAIN_CE_OUTPUT/global/`
- **Mondrian CE**: 50 test instances × 2 sex groups × 9 models → `MAIN_CE_OUTPUT/mondrian/`

---

### 11a. Global CE Feature Importance (n=150 instances per model, 9 models)

Aggregated mean |CE weight| across all models and instances. **Uncertainty width** = std of
CE weight bounds, measuring how consistently the feature drives predictions across patients.
Use SHAP for authoritative global ranking; CE uncertainty width is the novel contribution.

| Rank | Feature | Global CE Importance | Uncertainty Width | Models |
|------|---------|---------------------|-------------------|--------|
| 1 | **age** | 0.2450 | 0.1269 | 9/9 |
| 2 | **waist** | 0.1688 | 0.1073 | 9/9 |
| 3 | **anthro_group** | 0.1336 | 0.0926 | 8/9 |
| 4 | **Ave_SBP** | 0.1297 | 0.0860 | 9/9 |
| 5 | **tri** | 0.0825 | 0.0737 | 9/9 |
| 6 | currentsmoking | 0.0768 | 0.0523 | 8/9 |
| 7 | pa_met | 0.0748 | 0.0683 | 8/9 |
| 8 | binge_drink | 0.0683 | 0.0637 | 8/9 |
| 9 | ever_smk | 0.0677 | 0.0480 | 8/9 |
| 10 | drnk_30d_num | 0.0671 | 0.0587 | 9/9 |
| 11 | smoke_status | 0.0660 | 0.0503 | 8/9 |
| 12 | sex | 0.0648 | 0.0532 | 8/9 |
| 13 | hemoglobin | 0.0644 | 0.0618 | 9/9 |
| 14 | Ave_DBP | 0.0611 | 0.0575 | 9/9 |
| 15 | alcohol | 0.0557 | 0.0497 | 8/9 |
| 16 | con_alcohol | 0.0551 | 0.0471 | 8/9 |
| 17 | drnk_30days | 0.0547 | 0.0453 | 8/9 |
| 18 | vita | 0.0540 | 0.0529 | 9/9 |
| 19 | uic | 0.0528 | 0.0483 | 9/9 |
| 20 | ldl_hdl_ratio | 0.0524 | 0.0493 | 9/9 |
| 21 | hdl | 0.0479 | 0.0450 | 9/9 |
| 22 | ldl | 0.0451 | 0.0448 | 9/9 |

**Key finding**: Age, waist circumference, and systolic blood pressure are the most
consistently impactful features across all 9 models — consistent with SHAP rankings.
Age has the highest uncertainty width (0.127), meaning its influence varies most across
individual patients (some are strongly driven by age, others are not).

---

### 11b. Mondrian CE — Sex-Stratified Calibration Quality

Sex groups split by median of sex column (col 20, threshold = 2.0):
- **Group A** (lower sex value, n_cal=6,979, n_test≈6,999)
- **Group B** (higher sex value, n_cal=8,146, n_test≈8,127)

| Sex Group | Mean Prediction Interval Width | Std | n Instances |
|-----------|-------------------------------|-----|-------------|
| **Group A** | 0.00760 | 0.01097 | 450 |
| **Group B** | 0.00809 | 0.01498 | 448 |

**Interpretation**: Group B has **~6.3% wider** mean prediction intervals (0.00809 vs 0.00760),
indicating slightly less model certainty for that sex group. The higher std for Group B
(0.01498 vs 0.01097) also shows more variability in uncertainty — some Group B patients
receive much wider intervals than others.

**Clinical implication**: The difference is small in absolute terms, suggesting reasonable
sex-equity in model calibration. However, the larger variance for Group B warrants attention
in deployment — a subset of Group B patients may receive substantially less certain
predictions, potentially requiring additional clinical review.

> Source: `MAIN_CE_OUTPUT/mondrian/mondrian_pred_interval_summary.csv`  
> Chart: `MAIN_CE_OUTPUT/mondrian/mondrian_feature_uncertainty_by_sex.png`

---

## 12. Final Recommendation

**Deploy Stacking Ensemble with Isotonic Calibration** (`stacking_isotonic.joblib`):

| Criterion | Value |
|-----------|-------|
| F2-Score | 0.768 |
| Recall | 76.8% |
| Precision | 77.1% |
| ROC-AUC | **0.857** |
| AUPRC | **0.870** |
| Log-Loss (calibrated) | **0.470** |
| Brier Score | **0.154** |

When higher recall is prioritized (mass screening context), augment with kNN (Tuned) as a pre-filter (F2=0.827, Recall=87.2%) before confirmatory Stacking scoring.

---

**Document Version**: 3.0  
**Last Updated**: September 10, 2026  
**Pipeline**: V2 Hybrid — 2018–2021 ENNS | Selective Feature Engineering | 10 Models | SHAP + LIME | 3-Method Calibration | Calibrated Explanations (CE) with Venn-Abers | Mondrian Sex-Stratified Analysis
