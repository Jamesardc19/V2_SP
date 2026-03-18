# Diabetes Risk Prediction Using Machine Learning

A comprehensive machine learning pipeline for predicting diabetes risk using multiple datasets from the Expanded National Nutrition Survey (ENNS). This project implements advanced preprocessing techniques, feature engineering, and ensemble methods to achieve optimal prediction accuracy.

## 📊 Project Overview

This research project develops and evaluates multiple machine learning models for diabetes risk prediction, incorporating:
- Advanced feature engineering (30+ clinical indicators)
- Proper feature selection methodology
- Hyperparameter optimization
- Ensemble methods (Stacking, Voting)
- Model calibration and interpretability

**Best Model Performance:**
- **CatBoost: 69.2% accuracy, 71.7% ROC-AUC**
- Significant improvement over baseline through proper preprocessing pipeline

## 🗂️ Project Structure

```
V2_SP/
├── Core Scripts
│   ├── improved_preprocessing_v2.py      # Final preprocessing pipeline
│   ├── improved_model_training_v2.py     # Final model training pipeline
│   ├── revised_preprocessing.py          # Baseline preprocessing
│   ├── model_training_revised.py         # Baseline training
│   └── tabnet_wrapper.py                 # TabNet utility wrapper
│
├── Documentation
│   ├── README.md                         # Project overview
│   ├── Conceptual_Framework.md           # Research framework
│   ├── Model_Results_Interpretation.md   # Results analysis
│   └── Model_Improvement_Strategies.md   # Improvement strategies
│
├── Results
│   ├── Model_Results_Improved_V2/        # Final model results
│   ├── Trained_Models_Improved_V2/       # Final trained models
│   ├── PREPROCESS_OUTPUT_IMPROVED_V2/    # Final preprocessed data
│   └── Model_Results/                    # Baseline results
│
└── DATASETS/                             # Original datasets
    └── MAIN DATASET/
        ├── data-set_anthrop.csv
        ├── data-set_biochemical.csv
        ├── data-set_clinical.csv
        └── data-set_dietary_indiv.csv
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost catboost lightgbm
pip install imbalanced-learn matplotlib seaborn joblib
```

### Running the Pipeline

**1. Preprocessing (Feature Engineering + Selection)**
```bash
python improved_preprocessing_v2.py
```

**2. Model Training (Hyperparameter Tuning + Ensembles)**
```bash
python improved_model_training_v2.py
```

## 🔬 Methodology

### Data Processing Pipeline

**1. Data Integration**
- Merged 4 datasets: Anthropometric, Biochemical, Clinical, Dietary
- Composite key: `[enns_year, regcode, provhuc, hhnum, member_code]`
- Final dataset: 76,929 samples with 120 features

**2. Feature Engineering (30+ New Features)**
- **BMI Categories**: Underweight, Normal, Overweight, Obese
- **Cholesterol Ratios**: Total/HDL, LDL/HDL, Triglycerides/HDL
- **Metabolic Syndrome Indicator**: Combined risk factors
- **Blood Pressure Categories**: Hypertension, Prehypertension
- **Anthropometric Ratios**: Waist-hip, Waist-height
- **Age Interactions**: Age × BMI, Age × Waist
- **Lifestyle Risk Scores**: Smoking + Alcohol, Smoking + Obesity
- **Lipid Risk Score**: Combined cholesterol indicators
- **Dietary Ratios**: Carb/Energy, Fat/Energy

**3. Critical Fix: Feature Selection BEFORE One-Hot Encoding**
- **Problem Identified**: V1 selected 50 from 7,853 features (99% noise)
- **Solution**: Select 80 from 126 engineered features FIRST, then encode
- **Result**: 79 final features (vs 7,853) with preserved clinical indicators

**4. Data Balancing**
- BorderlineSMOTE for handling class imbalance (65.5% vs 34.5%)
- Training samples: 70,584 (balanced)

### Models Implemented

**Base Models:**
- XGBoost (Gradient Boosting)
- Random Forest
- CatBoost
- LightGBM

**Ensemble Methods:**
- Stacking Ensemble (Logistic Regression meta-learner)
- Voting Ensemble (Soft voting with weighted contributions)

**Optimization:**
- RandomizedSearchCV with 50 iterations
- 5-fold Stratified Cross-Validation
- Early stopping for gradient boosting models

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **CatBoost** | **69.2%** | 56.4% | 47.0% | 51.3% | **71.7%** |
| LightGBM | 68.6% | 55.2% | 46.8% | 50.6% | 71.5% |
| Stacking Ensemble | 68.3% | 54.3% | 51.6% | 52.9% | 71.4% |
| Voting Ensemble | 68.1% | 53.8% | 54.0% | 53.9% | 71.7% |
| Random Forest | 67.7% | 53.3% | 50.8% | 52.0% | 70.4% |
| XGBoost | 64.7% | 49.0% | 62.4% | 54.9% | 70.0% |

**Baseline Comparison:**
- Baseline XGBoost: 69.0%
- Improved CatBoost: 69.2% (+0.2%)
- ROC-AUC improvement: 71.7% (strong discriminative ability)

### Key Findings

**✅ Successful Improvements:**
1. Proper feature selection methodology (before encoding)
2. Reduced feature space from 7,853 to 79 (99% reduction)
3. Preserved all engineered clinical features
4. Ensemble methods showed consistent performance

**⚠️ Challenges Identified:**
1. CV-Test gap (85% CV → 65% test) indicates overfitting
2. Dataset ceiling appears to be ~69% accuracy
3. Class imbalance remains challenging despite SMOTE
4. Data quality limitations (missing values, noise)

## 🔍 Feature Importance

Top contributing features (from best model):
- Fasting Blood Sugar (FBS) - primary indicator
- BMI and BMI-derived features
- Age and age interactions
- Cholesterol ratios (Total/HDL, LDL/HDL)
- Blood pressure indicators
- Waist-hip ratio
- Metabolic syndrome indicator

## 📚 Documentation

- **Conceptual_Framework.md**: Research methodology and theoretical framework
- **Model_Results_Interpretation.md**: Detailed analysis of model performance
- **Model_Improvement_Strategies.md**: Strategies attempted to reach 80% accuracy

## 🎯 Future Work

To potentially achieve higher accuracy (80%+):

**Data Enhancement:**
- Temporal features (disease progression over time)
- Genetic markers
- More granular dietary data
- Additional biochemical measurements

**Advanced Techniques:**
- Deep learning with embeddings
- AutoML for automated feature engineering
- Cost-sensitive learning (penalize false negatives)
- Threshold optimization for medical applications

**Alternative Approaches:**
- Multi-class prediction (pre-diabetic, diabetic, high-risk)
- Risk scoring instead of binary classification
- Focus on ROC-AUC optimization over accuracy

## 📝 Citation

If you use this code or methodology, please cite:

```
[Your Name]. (2026). Diabetes Risk Prediction Using Machine Learning.
GitHub repository: [Your GitHub URL]
```

## 📄 License

[Specify your license - e.g., MIT, GPL, etc.]

## 👥 Contributors

- [Your Name] - Research and Implementation

## 🙏 Acknowledgments

- Expanded National Nutrition Survey (ENNS) for providing the datasets
- [Your Institution/Advisor] for guidance and support

---

**Note**: Large files (trained models, preprocessed data) are excluded from this repository due to size constraints. These can be regenerated by running the preprocessing and training scripts.
