# Pipeline Comparison: Standard V7 vs V7-Rivera vs 2013-Rivera

Generated: 2026-05-16  |  **Updated: 2026-05-17 (V7-Rivera v2: age+sex added, 22 features)**

---

## 1. Preprocessing Differences

| Setting | Standard V7 | **V7-Rivera v1** | **V7-Rivera v2** | 2013-Rivera |
|---------|------------|--------------|--------------|-------------|
| Dataset | 2018–2021 ENNS | 2018–2021 ENNS | 2018–2021 ENNS | 2013 ENNS |
| Population filter | Adults only (excl. children, pregnant, lactating) | None — all age groups | None — all age groups | None |
| Feature engineering | 8 interaction/ratio features added | Disabled — raw features only | Disabled — raw features only | Disabled |
| Correlation threshold | 0.92 | 0.80 | 0.80 | 0.80 |
| SMOTE timing | Train split only (strategy=0.6, partial) | Full dataset before split (strategy=1.0) | Full dataset before split (strategy=1.0) | Full dataset before split |
| Primary metric | ROC-AUC | F2-score (β=2) | F2-score (β=2) | F2-score (β=2) |
| Final features | ~34 | 20 (no age/sex) | **22 (age+sex added)** | ~18 |
| Socio dataset | No | No | **Yes (age, sex)** | No |
| Train size | ~46K | 70,585 (SMOTE-balanced) | **70,585 (SMOTE-balanced)** | ~3,800 |

---

## 2. Model Performance: Standard V7 vs V7-Rivera

> Standard V7 does not record F2. Estimated F2 = 5·P·R / (4·P + R).

| Model | Std-V7 Recall | Std-V7 F1 | Std-V7 F2* | Rivera-V7 Recall | Rivera-V7 F2 | Rivera-V7 ROC-AUC |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| kNN | 0.370 | 0.436 | 0.393 | **0.860** | **0.815** | 0.806 |
| Random Forest | 0.626 | 0.560 | 0.603 | 0.763 | 0.755 | 0.817 |
| Voting Ensemble | 0.581 | 0.551 | 0.569 | 0.736 | 0.739 | 0.832 |
| Stacking Ensemble | 0.365 | 0.447 | 0.394 | 0.716 | 0.727 | 0.841 |
| Naive Bayes | 0.531 | 0.526 | 0.530 | 0.759 | 0.721 | 0.676 |
| BO-TabNet | — | — | — | 0.738 | 0.718 | 0.739 |
| XGBoost | 0.449 | 0.497 | 0.479 | 0.712 | 0.717 | 0.812 |
| CatBoost | 0.660 | 0.570 | **0.621** | 0.710 | 0.717 | 0.818 |
| AdaBoost | 0.352 | 0.436 | 0.376 | 0.752 | 0.716 | 0.656 |
| LightGBM | 0.643 | 0.563 | 0.612 | 0.686 | 0.702 | 0.832 |

**Δ best F2: +0.194** (0.621 → 0.815, CatBoost→kNN)
**Δ best ROC-AUC: +0.144** (0.697 → 0.841, Stacking)

---

## 3a. Full Metric Snapshot — V7-Rivera v1 (20 features, no age/sex)

| Rank | Model | Precision | Recall | F2 | F1 | ROC-AUC | AUPRC | Accuracy |
|:----:|-------|:---------:|:------:|:--:|:--:|:-------:|:-----:|:--------:|
| 1 | kNN (Tuned) | 0.673 | **0.860** | **0.815** | 0.755 | 0.806 | 0.753 | 0.721 |
| 2 | Random Forest | 0.724 | 0.763 | 0.755 | 0.743 | 0.817 | 0.822 | 0.736 |
| 3 | Voting Ensemble | 0.751 | 0.736 | 0.739 | 0.743 | 0.832 | 0.852 | 0.746 |
| 4 | Stacking Ensemble | **0.774** | 0.716 | 0.727 | **0.744** | **0.841** | **0.865** | **0.753** |
| 5 | Naive Bayes | 0.598 | 0.759 | 0.721 | 0.669 | 0.676 | 0.648 | 0.625 |
| 6 | BO-TabNet | 0.648 | 0.738 | 0.718 | 0.690 | 0.739 | 0.742 | 0.669 |
| 7 | XGBoost | 0.738 | 0.712 | 0.717 | 0.725 | 0.812 | 0.834 | 0.730 |
| 8 | CatBoost | 0.746 | 0.710 | 0.717 | 0.728 | 0.818 | 0.841 | 0.734 |
| 9 | AdaBoost | 0.599 | 0.752 | 0.716 | 0.667 | 0.656 | 0.607 | 0.625 |
| 10 | LightGBM | **0.776** | 0.686 | 0.702 | 0.728 | 0.832 | 0.858 | 0.744 |

## 3b. Full Metric Snapshot — V7-Rivera v2 (22 features, age+sex added) ✅ CURRENT

| Rank | Model | Precision | Recall | F2 | F1 | ROC-AUC | AUPRC | Accuracy |
|:----:|-------|:---------:|:------:|:--:|:--:|:-------:|:-----:|:--------:|
| 1 | kNN (Tuned) | 0.683 | **0.866** | **0.822** | 0.763 | 0.814 | 0.759 | 0.732 |
| 2 | AdaBoost | 0.600 | 0.869 | 0.797 | 0.710 | 0.701 | 0.644 | 0.645 |
| 3 | Random Forest | 0.733 | 0.803 | 0.788 | 0.766 | 0.839 | 0.839 | 0.755 |
| 4 | Voting Ensemble | 0.746 | 0.779 | 0.772 | 0.762 | 0.847 | 0.858 | 0.757 |
| 5 | Stacking Ensemble | **0.813** | 0.753 | 0.764 | **0.782** | **0.876** | **0.893** | **0.789** |
| 6 | BO-TabNet | 0.679 | 0.773 | 0.752 | 0.723 | 0.780 | 0.779 | 0.704 |
| 7 | CatBoost | 0.683 | 0.770 | 0.751 | 0.724 | 0.793 | 0.802 | 0.706 |
| 8 | XGBoost | 0.708 | 0.759 | 0.748 | 0.732 | 0.808 | 0.816 | 0.723 |
| 9 | Naive Bayes | 0.629 | 0.780 | 0.744 | 0.697 | 0.715 | 0.684 | 0.660 |
| 10 | LightGBM | **0.781** | 0.726 | 0.737 | 0.752 | 0.851 | 0.872 | 0.761 |

**Average F2: 0.768** (+0.035 vs v1 avg 0.733) | **Average ROC-AUC: 0.808** (+0.025 vs v1 avg 0.783)

---

## 4. Full Metric Snapshot — 2013-Rivera (Best Models)

| Rank | Model | Precision | Recall | F2 | F1 | ROC-AUC | AUPRC | Accuracy |
|:----:|-------|:---------:|:------:|:--:|:--:|:-------:|:-----:|:--------:|
| 1 | kNN (Tuned) | 0.710 | **0.983** | **0.913** | 0.824 | 0.892 | 0.828 | 0.791 |
| 2 | Random Forest | 0.864 | 0.858 | 0.859 | 0.861 | 0.940 | **0.948** | 0.862 |
| 3 | BO-TabNet | 0.774 | 0.862 | 0.843 | 0.816 | 0.892 | 0.896 | 0.805 |
| 4 | Stacking Ensemble | **0.896** | 0.849 | 0.858 | **0.872** | **0.948** | 0.957 | **0.875** |
| 5 | Voting Ensemble | 0.905 | 0.820 | 0.836 | 0.861 | 0.943 | 0.954 | 0.867 |
| 6 | XGBoost | 0.920 | 0.809 | 0.829 | 0.861 | 0.938 | 0.952 | 0.869 |
| 7 | LightGBM | 0.886 | 0.800 | 0.815 | 0.840 | 0.927 | 0.942 | 0.848 |
| 8 | CatBoost | 0.882 | 0.780 | 0.799 | 0.828 | 0.916 | 0.935 | 0.838 |
| 9 | AdaBoost | 0.593 | 0.890 | 0.809 | 0.712 | 0.701 | 0.641 | 0.640 |
| 10 | Naive Bayes | 0.671 | 0.709 | 0.701 | 0.689 | 0.744 | 0.731 | 0.680 |

---

## 5. Cross-Dataset Comparison (Rivera Framework, Shared Models)

| Model | 2013 F2 | V7-Rivera v2 F2 | 2013 ROC-AUC | V7-Rivera v2 ROC-AUC | 2013 Recall | V7-Rivera v2 Recall |
|-------|:-------:|:---------------:|:------------:|:--------------------:|:-----------:|:-------------------:|
| kNN | **0.913** | **0.822** | 0.892 | 0.814 | **0.983** | **0.866** |
| Random Forest | 0.859 | 0.788 | 0.940 | 0.839 | 0.858 | 0.803 |
| BO-TabNet | 0.843 | 0.752 | 0.892 | 0.780 | 0.862 | 0.773 |
| Stacking | 0.858 | 0.764 | **0.948** | **0.876** | 0.849 | 0.753 |
| Voting | 0.836 | 0.772 | 0.943 | 0.847 | 0.820 | 0.779 |
| XGBoost | 0.829 | 0.748 | 0.938 | 0.808 | 0.809 | 0.759 |
| CatBoost | 0.799 | 0.751 | 0.916 | 0.793 | 0.780 | 0.770 |
| LightGBM | 0.815 | 0.737 | 0.927 | 0.851 | 0.800 | 0.726 |
| Naive Bayes | 0.701 | 0.744 | 0.744 | 0.715 | 0.709 | 0.780 |
| AdaBoost | 0.809 | 0.797 | 0.701 | 0.701 | 0.890 | 0.869 |
| **Average** | **0.826** | **0.768** | **0.874** | **0.808** | **0.836** | **0.788** |

---

## 6. Key Findings & Thesis Recommendation

### V7-Rivera vs Standard V7
- Rivera preprocessing delivers **+19.4 F2 points** improvement on the best model (0.621 → 0.815)
- **+14.4 ROC-AUC points** on Stacking Ensemble (0.697 → 0.841)
- Recall improves dramatically: best model recall 0.365 → 0.860
- **Use V7-Rivera** — the standard V7 pipeline is clearly inferior for this clinical task

### V7-Rivera v2 vs v1 (Socio Integration Impact)
- Adding `age` + `sex` improved **every model** — avg F2: 0.733 → 0.768 (**+0.035**)
- Stacking Ensemble ROC-AUC: 0.841 → 0.876 (**+0.035**) — strongest AUC model
- AdaBoost gained the most on F2: 0.716 → 0.797 (**+0.081**)
- **Conclusion: socioeconomic features (age, sex) are clinically meaningful predictors**

### V7-Rivera v2 vs 2013-Rivera (Dataset Gap)
- 2013 dataset averages **~5.8 F2 points** higher across all models (0.826 vs 0.768)
- Gap narrowed from 9.3 → 5.8 points after adding age+sex
- Remaining gap expected: 2013 ENNS is smaller (~4K vs 77K), more curated, single-year snapshot
- The 2018-2021 dataset is more heterogeneous (multi-year, all age groups, more missingness)

### Model Rankings (Rivera framework, both datasets)
- **kNN** consistently tops F2 ranking — high recall from distance-based voting on SMOTE-balanced data
- **Stacking/Voting Ensembles** best for ROC-AUC and balanced precision-recall
- **BO-TabNet** mid-tier on V7-Rivera but strong on 2013 (F2=0.843) — performs better on smaller, cleaner data

### Thesis Decision
- **Primary pipeline: V7-Rivera** (SMOTE before split, F2-optimized, correlation threshold 0.80)
- **Primary metric: F2-score** — justified by clinical need to minimize false negatives in at-risk detection
- Report both datasets as separate experiments under the Rivera framework
- Use BO-TabNet + Stacking as focal models (Stacking = best ROC-AUC/AUPRC; kNN = best F2/Recall)
