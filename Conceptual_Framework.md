# Conceptual Framework: Diabetes Risk Prediction System Using Machine Learning

## Overview
This conceptual framework illustrates the systematic approach to developing a diabetes risk prediction system using multiple machine learning models with Bayesian-optimized TabNet as the primary model.

---

## Framework Components

### **INPUT LAYER**
```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW DATASETS                              │
├─────────────────┬─────────────────┬─────────────┬───────────────┤
│  Anthropometric │   Biochemical   │   Clinical  │    Dietary    │
│     Dataset     │     Dataset     │   Dataset   │    Dataset    │
│   (444,400 x 18)│  (161,643 x 13) │(351,701 x 29)│(234,293 x 75)│
└─────────────────┴─────────────────┴─────────────┴───────────────┘
```

**Data Sources:**
- Anthropometric: Weight, height, waist, hip, BMI, ethnicity
- Biochemical: Vitamin A, hemoglobin, urinary iodine concentration
- Clinical: Blood pressure, FBS, cholesterol, triglycerides, HDL, LDL
- Dietary: Food group intake, nutrient consumption

---

### **PREPROCESSING LAYER**

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPROCESSING                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Dataset Merging                                             │
│     └─ Composite Key: [enns_year, regcode, provhuc,            │
│                         hhnum, member_code]                      │
│     └─ Left Join on Clinical Dataset (Base)                    │
│     └─ Output: Merged Dataset (351,701 x 105)                  │
│                                                                  │
│  2. Target Variable Creation                                    │
│     └─ diabetes_risk = 1 if FBS ≥ 100 mg/dL, else 0           │
│     └─ Remove missing FBS values                               │
│     └─ Final Dataset: 76,929 samples                           │
│     └─ Class Distribution: 65.5% Non-diabetic, 34.5% Diabetic  │
│                                                                  │
│  3. Train/Validation/Test Split                                 │
│     └─ Stratified Split (70/15/15)                             │
│     └─ Train: 53,849 | Val: 11,540 | Test: 11,540             │
│                                                                  │
│  4. Feature Processing Pipeline                                 │
│     ┌─────────────────────────────────────────────────┐        │
│     │ Numeric Features (96 features)                  │        │
│     │  ├─ Simple Imputation (Mean)                    │        │
│     │  ├─ Variance Threshold (threshold=0.01)         │        │
│     │  ├─ Correlation Dropper (threshold=0.90)        │        │
│     │  └─ MinMaxScaler (range: 0-1)                   │        │
│     └─────────────────────────────────────────────────┘        │
│     ┌─────────────────────────────────────────────────┐        │
│     │ Categorical Features (8 features)               │        │
│     │  ├─ Mode Imputation                             │        │
│     │  └─ One-Hot Encoding                            │        │
│     └─────────────────────────────────────────────────┘        │
│                                                                  │
│  5. Class Balancing (Training Set Only)                        │
│     └─ SMOTE (k_neighbors=5)                                   │
│     └─ Balanced Training Set: 70,584 samples                   │
│                                                                  │
│  6. Output: Preprocessed Features (1,557 features)             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### **FEATURE SELECTION LAYER**

```
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE SELECTION                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Combined Feature Selection Method                              │
│                                                                  │
│  Step 1: Statistical Filtering                                  │
│  ┌────────────────────────────────────────────┐                │
│  │ ANOVA F-test (SelectKBest)                 │                │
│  │ └─ Select top 80 features (2x target)      │                │
│  │ └─ Based on statistical significance       │                │
│  └────────────────────────────────────────────┘                │
│                    ↓                                             │
│  Step 2: Model-Based Selection                                  │
│  ┌────────────────────────────────────────────┐                │
│  │ Random Forest Feature Importance            │                │
│  │ └─ Select top 40 features from 80          │                │
│  │ └─ Based on feature importance scores      │                │
│  └────────────────────────────────────────────┘                │
│                    ↓                                             │
│  Output: 40 Most Important Features                            │
│  └─ 97% dimensionality reduction (1,557 → 40)                  │
│  └─ Reduced noise and improved model performance               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### **MODEL TRAINING LAYER**

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Baseline Models (6 models)                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. Naive Bayes (GaussianNB)                              │  │
│  │ 2. Logistic Regression (max_iter=1000)                   │  │
│  │ 3. Support Vector Machine (SVM with RBF kernel)          │  │
│  │ 4. Random Forest (default parameters)                    │  │
│  │ 5. k-Nearest Neighbors (k=5)                             │  │
│  │ 6. XGBoost (default parameters)                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Advanced Model                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Bayesian-Optimized TabNet (BO-TabNet)                    │  │
│  │                                                           │  │
│  │ Hyperparameter Optimization:                             │  │
│  │  ├─ Bayesian Optimization (25 iterations)                │  │
│  │  ├─ Optimized Parameters:                                │  │
│  │  │   • n_d, n_a: [8, 64]                                 │  │
│  │  │   • n_steps: [3, 10]                                  │  │
│  │  │   • gamma: [1.0, 2.0]                                 │  │
│  │  │   • lambda_sparse: [1e-6, 1e-3]                       │  │
│  │  │   • learning_rate: [1e-3, 1e-1]                       │  │
│  │  └─ Optimization Metric: AUPRC                           │  │
│  │                                                           │  │
│  │ Training Configuration:                                   │  │
│  │  ├─ Max Epochs: 100                                      │  │
│  │  ├─ Early Stopping: Patience=10                          │  │
│  │  ├─ Batch Size: 1024                                     │  │
│  │  ├─ Virtual Batch Size: 128                              │  │
│  │  ├─ Optimizer: Adam                                      │  │
│  │  └─ LR Scheduler: StepLR (step=10, gamma=0.5)           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### **MODEL EVALUATION LAYER**

```
┌─────────────────────────────────────────────────────────────────┐
│                   MODEL EVALUATION                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Performance Metrics                                            │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ • Accuracy                                             │    │
│  │ • Precision (focus on minimizing false positives)     │    │
│  │ • Recall (focus on identifying all diabetic cases)    │    │
│  │ • F1-Score (harmonic mean of precision & recall)      │    │
│  │ • F2-Score (weighted towards recall)                  │    │
│  │ • AUPRC (Area Under Precision-Recall Curve)           │    │
│  │ • ROC-AUC (Area Under ROC Curve)                      │    │
│  │ • Brier Score (calibration quality)                   │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Visualization Outputs                                          │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ • Confusion Matrices (per model)                       │    │
│  │ • ROC Curves (per model)                               │    │
│  │ • Precision-Recall Curves (per model)                  │    │
│  │ • Feature Importance Plots                             │    │
│  │ • Calibration Curves                                   │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### **MODEL CALIBRATION LAYER**

```
┌─────────────────────────────────────────────────────────────────┐
│                   MODEL CALIBRATION                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Calibration Methods (Applied to All Models)                    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ 1. Platt Scaling (Logistic Calibration)               │    │
│  │    └─ Fits logistic regression on model outputs       │    │
│  │    └─ Best for: SVM, Naive Bayes                      │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ 2. Isotonic Regression (Non-parametric)               │    │
│  │    └─ Fits monotonic function to outputs              │    │
│  │    └─ Best for: Tree-based models                     │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Purpose:                                                       │
│  └─ Improve probability estimates reliability                  │
│  └─ Ensure predicted probabilities match true frequencies      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### **MODEL INTERPRETABILITY LAYER**

```
┌─────────────────────────────────────────────────────────────────┐
│                 MODEL INTERPRETABILITY                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LIME (Local Interpretable Model-agnostic Explanations)        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ • Generates explanations for individual predictions    │    │
│  │ • Shows feature contributions to each prediction       │    │
│  │ • Provides local model interpretability                │    │
│  │ • Helps understand model decision-making process       │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Outputs:                                                       │
│  └─ Feature importance plots for sample predictions            │
│  └─ Positive/negative contribution visualization               │
│  └─ Per-model explanation reports                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### **OUTPUT LAYER**

```
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUTS                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Trained Models                                              │
│     └─ 7 saved models (.joblib format)                         │
│     └─ Ready for deployment                                    │
│                                                                  │
│  2. Model Performance Metrics                                   │
│     └─ Comparative analysis across all models                  │
│     └─ Best model identification                               │
│     └─ Performance reports (CSV format)                        │
│                                                                  │
│  3. Calibrated Models                                           │
│     └─ Improved probability estimates                          │
│     └─ 14 calibrated models (2 methods × 7 models)            │
│                                                                  │
│  4. Visualization Artifacts                                     │
│     └─ Confusion matrices                                      │
│     └─ ROC curves                                              │
│     └─ PR curves                                               │
│     └─ Calibration curves                                      │
│     └─ Feature importance plots                                │
│                                                                  │
│  5. LIME Explanations                                           │
│     └─ Individual prediction explanations                      │
│     └─ Feature contribution visualizations                     │
│                                                                  │
│  6. Preprocessed Data                                           │
│     └─ Feature-selected dataset (40 features)                  │
│     └─ Feature mapping documentation                           │
│     └─ Preprocessing pipeline (.joblib)                        │
│                                                                  │
│  7. Diabetes Risk Predictions                                   │
│     └─ Binary classification (Diabetic/Non-diabetic)           │
│     └─ Probability scores (0-1 range)                          │
│     └─ Confidence intervals                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
┌─────────────┐
│   INPUT     │
│  4 Datasets │
└──────┬──────┘
       │
       ↓
┌─────────────────────┐
│   PREPROCESSING     │
│  • Merge            │
│  • Clean            │
│  • Transform        │
│  • Split            │
│  • Balance (SMOTE)  │
└──────┬──────────────┘
       │
       ↓
┌─────────────────────┐
│ FEATURE SELECTION   │
│  • Statistical      │
│  • Model-based      │
│  • 1557 → 40        │
└──────┬──────────────┘
       │
       ├──────────────────────────────────┐
       │                                  │
       ↓                                  ↓
┌─────────────────┐            ┌──────────────────┐
│ BASELINE MODELS │            │   BO-TABNET      │
│  • Naive Bayes  │            │ • Bayesian Opt   │
│  • Log Reg      │            │ • Hyperparameter │
│  • SVM          │            │   Tuning         │
│  • Random Forest│            └────────┬─────────┘
│  • k-NN         │                     │
│  • XGBoost      │                     │
└────────┬────────┘                     │
         │                              │
         └──────────┬───────────────────┘
                    │
                    ↓
         ┌──────────────────┐
         │   EVALUATION     │
         │  • Metrics       │
         │  • Visualization │
         └────────┬─────────┘
                  │
                  ↓
         ┌──────────────────┐
         │   CALIBRATION    │
         │  • Platt Scaling │
         │  • Isotonic Reg  │
         └────────┬─────────┘
                  │
                  ↓
         ┌──────────────────┐
         │ INTERPRETABILITY │
         │  • LIME          │
         │  • Explanations  │
         └────────┬─────────┘
                  │
                  ↓
         ┌──────────────────┐
         │     OUTPUT       │
         │  • Models        │
         │  • Metrics       │
         │  • Predictions   │
         └──────────────────┘
```

---

## Key Innovations

1. **Composite Key Merging**: Robust dataset integration using multi-column keys
2. **Early Data Splitting**: Prevents data leakage by splitting before preprocessing
3. **Combined Feature Selection**: Two-stage approach for optimal feature subset
4. **Bayesian Optimization**: Automated hyperparameter tuning for TabNet
5. **Dual Calibration**: Both Platt Scaling and Isotonic Regression for reliability
6. **Model Interpretability**: LIME explanations for clinical decision support

---

## Expected Outcomes

1. **High-Performance Models**: Accurate diabetes risk prediction (target: >70% accuracy)
2. **Calibrated Probabilities**: Reliable risk scores for clinical use
3. **Interpretable Predictions**: Explainable AI for healthcare professionals
4. **Reduced Feature Space**: Efficient model with 40 key features
5. **Comparative Analysis**: Identification of best-performing model
6. **Deployment-Ready System**: Saved models and preprocessing pipeline

---

## Technical Stack

- **Language**: Python 3.12
- **ML Framework**: scikit-learn, XGBoost, PyTorch TabNet
- **Optimization**: Bayesian Optimization (bayes_opt)
- **Visualization**: Matplotlib, Seaborn
- **Interpretability**: LIME
- **Data Processing**: Pandas, NumPy
- **Class Balancing**: imbalanced-learn (SMOTE)

---

*Figure: Conceptual Framework for Diabetes Risk Prediction System*
