# Strategies to Improve Model Performance to 80% Accuracy

## Current Status
- **Current Best Accuracy**: 69% (XGBoost)
- **Target Accuracy**: 80%
- **Gap to Close**: +11 percentage points

---

## Is 80% Achievable? YES!

Based on published diabetes prediction studies, 80-85% accuracy is achievable with:
1. Enhanced feature engineering
2. Advanced hyperparameter tuning
3. Ensemble methods
4. Better handling of class imbalance
5. Additional data sources or features

---

## Strategy 1: Advanced Feature Engineering (Expected Impact: +3-5%)

### Current Limitation
You're using raw features after basic preprocessing. Many important patterns might be hidden.

### Improvements to Implement

#### 1.1 Create Domain-Specific Features

**Clinical Risk Indicators:**
```python
# BMI categories (WHO standards)
df['bmi_category'] = pd.cut(df['bmi'], 
                            bins=[0, 18.5, 25, 30, 100],
                            labels=['underweight', 'normal', 'overweight', 'obese'])

# Metabolic syndrome indicators
df['metabolic_syndrome'] = (
    (df['waist'] > 102) &  # Central obesity (male threshold)
    (df['tri'] >= 150) &    # High triglycerides
    (df['hdl'] < 40) &      # Low HDL
    (df['Ave_SBP'] >= 130)  # High blood pressure
).astype(int)

# Cholesterol ratios (better predictors than individual values)
df['total_hdl_ratio'] = df['chol'] / df['hdl']
df['ldl_hdl_ratio'] = df['ldl'] / df['hdl']
df['tri_hdl_ratio'] = df['tri'] / df['hdl']

# Blood pressure categories
df['hypertension'] = ((df['Ave_SBP'] >= 140) | (df['Ave_DBP'] >= 90)).astype(int)
df['prehypertension'] = (
    ((df['Ave_SBP'] >= 120) & (df['Ave_SBP'] < 140)) |
    ((df['Ave_DBP'] >= 80) & (df['Ave_DBP'] < 90))
).astype(int)
```

#### 1.2 Interaction Features

**Feature Interactions:**
```python
# Age-related risk factors
df['age_bmi_interaction'] = df['age'] * df['bmi']
df['age_waist_interaction'] = df['age'] * df['waist']

# Lifestyle combinations
df['smoking_alcohol'] = df['currentsmoking'] * df['alcohol']
df['smoking_bmi'] = df['currentsmoking'] * df['bmi']

# Anthropometric ratios
df['waist_hip_ratio'] = df['waist'] / df['hip']
df['waist_height_ratio'] = df['waist'] / df['height']
```

#### 1.3 Polynomial Features (for key predictors)

```python
from sklearn.preprocessing import PolynomialFeatures

# Apply to most important features only (to avoid explosion)
key_features = ['bmi', 'age', 'chol', 'tri', 'hdl', 'ldl', 'Ave_SBP']
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
poly_features = poly.fit_transform(df[key_features])
```

#### 1.4 Binning Continuous Variables

```python
# Age groups (risk increases with age)
df['age_group'] = pd.cut(df['age'], 
                         bins=[0, 30, 40, 50, 60, 100],
                         labels=['<30', '30-40', '40-50', '50-60', '60+'])

# FBS risk categories (even for training features, not target)
df['glucose_category'] = pd.cut(df['some_glucose_marker'], 
                                bins=[0, 100, 126, 200],
                                labels=['normal', 'prediabetic', 'diabetic'])
```

**Expected Impact**: +3-5% accuracy

---

## Strategy 2: Hyperparameter Optimization (Expected Impact: +2-4%)

### Current Limitation
Only TabNet has Bayesian Optimization. Other models use default parameters.

### Improvements to Implement

#### 2.1 Grid Search / Random Search for All Models

**XGBoost Hyperparameter Tuning:**
```python
from sklearn.model_selection import RandomizedSearchCV

xgb_params = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 500],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [0.5, 1, 1.5, 2]
}

xgb_random = RandomizedSearchCV(
    XGBClassifier(random_state=42),
    param_distributions=xgb_params,
    n_iter=100,  # Try 100 combinations
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)
xgb_random.fit(X_train, y_train)
best_xgb = xgb_random.best_estimator_
```

**Random Forest Tuning:**
```python
rf_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=rf_params,
    n_iter=100,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
```

**Expected Impact**: +2-4% accuracy

---

## Strategy 3: Advanced Ensemble Methods (Expected Impact: +3-6%)

### Current Limitation
Models are trained independently. No ensemble combining their strengths.

### Improvements to Implement

#### 3.1 Stacking Ensemble

**Concept**: Use predictions from multiple models as features for a meta-model.

```python
from sklearn.ensemble import StackingClassifier

# Base models (Level 0)
base_models = [
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=20)),
    ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.05)),
    ('lr', LogisticRegression(max_iter=1000, C=0.1)),
    ('svm', SVC(probability=True, C=1.0, kernel='rbf'))
]

# Meta-model (Level 1)
meta_model = LogisticRegression()

# Stacking ensemble
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

stacking_clf.fit(X_train, y_train)
```

#### 3.2 Voting Ensemble

```python
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('rf', best_rf_model),
        ('xgb', best_xgb_model),
        ('lr', best_lr_model)
    ],
    voting='soft',  # Use probability predictions
    weights=[2, 3, 1]  # Give more weight to better models
)
```

#### 3.3 Blending

```python
# Train models on 80% of training data
# Use remaining 20% to train meta-model
from sklearn.model_selection import train_test_split

X_train_base, X_train_meta, y_train_base, y_train_meta = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Train base models
base_predictions = []
for name, model in base_models:
    model.fit(X_train_base, y_train_base)
    preds = model.predict_proba(X_train_meta)[:, 1]
    base_predictions.append(preds)

# Stack predictions as features
X_meta = np.column_stack(base_predictions)

# Train meta-model
meta_model = LogisticRegression()
meta_model.fit(X_meta, y_train_meta)
```

**Expected Impact**: +3-6% accuracy

---

## Strategy 4: Better Class Imbalance Handling (Expected Impact: +2-3%)

### Current Limitation
Using SMOTE with default parameters. May not be optimal.

### Improvements to Implement

#### 4.1 Advanced Sampling Techniques

**ADASYN (Adaptive Synthetic Sampling):**
```python
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
```

**SMOTE + Tomek Links (Hybrid):**
```python
from imblearn.combine import SMOTETomek

smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
```

**Borderline-SMOTE:**
```python
from imblearn.over_sampling import BorderlineSMOTE

borderline_smote = BorderlineSMOTE(random_state=42, kind='borderline-1')
X_resampled, y_resampled = borderline_smote.fit_resample(X_train, y_train)
```

#### 4.2 Class Weights

```python
# For XGBoost
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
xgb_model = XGBClassifier(scale_pos_weight=scale_pos_weight)

# For Random Forest
rf_model = RandomForestClassifier(class_weight='balanced')

# For Logistic Regression
lr_model = LogisticRegression(class_weight='balanced')
```

#### 4.3 Threshold Optimization

```python
from sklearn.metrics import f1_score

# Find optimal threshold instead of default 0.5
thresholds = np.arange(0.3, 0.7, 0.05)
best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_val, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Optimal threshold: {best_threshold}")
```

**Expected Impact**: +2-3% accuracy

---

## Strategy 5: Feature Selection Refinement (Expected Impact: +1-3%)

### Current Limitation
Using combined method with fixed 40 features. May not be optimal number.

### Improvements to Implement

#### 5.1 Recursive Feature Elimination with Cross-Validation

```python
from sklearn.feature_selection import RFECV

rfecv = RFECV(
    estimator=RandomForestClassifier(n_estimators=100),
    step=1,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
rfecv.fit(X_train, y_train)

optimal_features = rfecv.n_features_
print(f"Optimal number of features: {optimal_features}")
X_train_selected = rfecv.transform(X_train)
```

#### 5.2 Mutual Information

```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_scores = pd.Series(mi_scores, index=feature_names)
top_features = mi_scores.nlargest(50).index.tolist()
```

#### 5.3 Permutation Importance

```python
from sklearn.inspection import permutation_importance

# Train a model first
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Calculate permutation importance
perm_importance = permutation_importance(
    rf, X_val, y_val, n_repeats=10, random_state=42
)

# Select features with importance > threshold
important_features = [
    feature_names[i] for i in range(len(feature_names))
    if perm_importance.importances_mean[i] > 0.01
]
```

**Expected Impact**: +1-3% accuracy

---

## Strategy 6: Data Augmentation & External Features (Expected Impact: +2-5%)

### Current Limitation
Limited to existing dataset features.

### Improvements to Implement

#### 6.1 Add Derived Temporal Features

If your data has temporal information:
```python
# Time-based features
df['year'] = df['enns_year']
df['years_since_baseline'] = df['enns_year'] - df['enns_year'].min()

# Cohort effects
df['cohort'] = pd.cut(df['enns_year'], bins=[2012, 2015, 2018, 2020])
```

#### 6.2 Geographic/Regional Features

```python
# Regional diabetes prevalence (if available)
regional_stats = df.groupby('provhuc')['diabetes_risk'].mean()
df['regional_diabetes_rate'] = df['provhuc'].map(regional_stats)

# Urban vs rural (if derivable from provhuc)
urban_provinces = [1, 2, 3, 13]  # Example
df['is_urban'] = df['provhuc'].isin(urban_provinces).astype(int)
```

#### 6.3 Family History Proxy

If you have household data:
```python
# Family diabetes history (from same household)
household_diabetes = df.groupby('hhnum')['diabetes_risk'].transform('sum')
df['family_diabetes_count'] = household_diabetes - df['diabetes_risk']
df['has_diabetic_family'] = (df['family_diabetes_count'] > 0).astype(int)
```

**Expected Impact**: +2-5% accuracy

---

## Strategy 7: Advanced Models & Deep Learning (Expected Impact: +2-4%)

### Current Limitation
TabNet is your only deep learning model.

### Improvements to Implement

#### 7.1 CatBoost (Often outperforms XGBoost)

```python
from catboost import CatBoostClassifier

catboost_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    bootstrap_type='Bernoulli',
    subsample=0.8,
    random_seed=42,
    verbose=False
)

catboost_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50
)
```

#### 7.2 LightGBM

```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=8,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)
```

#### 7.3 Neural Network with Embeddings

```python
import tensorflow as tf
from tensorflow import keras

# For categorical features
def create_nn_model(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_dim=input_dim),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    return model

nn_model = create_nn_model(X_train.shape[1])
nn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=256,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
)
```

**Expected Impact**: +2-4% accuracy

---

## Strategy 8: Cross-Validation & Robust Evaluation (Expected Impact: +1-2%)

### Current Limitation
Single train/val/test split may not be representative.

### Improvements to Implement

#### 8.1 Stratified K-Fold Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
for train_idx, val_idx in skf.split(X_train, y_train):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    model.fit(X_fold_train, y_fold_train)
    score = model.score(X_fold_val, y_fold_val)
    cv_scores.append(score)

print(f"CV Mean Accuracy: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
```

#### 8.2 Nested Cross-Validation (for hyperparameter tuning)

```python
from sklearn.model_selection import cross_val_score

# Outer CV for model evaluation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV for hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Nested CV
nested_scores = cross_val_score(
    RandomizedSearchCV(model, param_distributions, cv=inner_cv),
    X_train, y_train,
    cv=outer_cv,
    scoring='accuracy'
)
```

**Expected Impact**: +1-2% accuracy (more robust estimates)

---

## Strategy 9: Data Quality Improvements (Expected Impact: +2-4%)

### Current Limitation
Using mean imputation for missing values. May introduce bias.

### Improvements to Implement

#### 9.1 Advanced Imputation

**Iterative Imputer (MICE):**
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

iterative_imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10),
    max_iter=10,
    random_state=42
)
X_imputed = iterative_imputer.fit_transform(X_train)
```

**KNN Imputer (but optimized):**
```python
from sklearn.impute import KNNImputer

knn_imputer = KNNImputer(
    n_neighbors=5,
    weights='distance',  # Weight by distance
    metric='nan_euclidean'
)
X_imputed = knn_imputer.fit_transform(X_train)
```

#### 9.2 Outlier Detection & Treatment

```python
from sklearn.ensemble import IsolationForest

# Detect outliers
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso_forest.fit_predict(X_train)

# Option 1: Remove outliers
X_train_clean = X_train[outlier_labels == 1]
y_train_clean = y_train[outlier_labels == 1]

# Option 2: Cap outliers
from scipy import stats
z_scores = np.abs(stats.zscore(X_train))
X_train_capped = X_train.copy()
X_train_capped[z_scores > 3] = np.nan  # Then impute
```

#### 9.3 Feature Scaling Optimization

```python
from sklearn.preprocessing import RobustScaler, PowerTransformer

# RobustScaler (less sensitive to outliers)
robust_scaler = RobustScaler()
X_scaled = robust_scaler.fit_transform(X_train)

# PowerTransformer (make features more Gaussian)
power_transformer = PowerTransformer(method='yeo-johnson')
X_transformed = power_transformer.fit_transform(X_train)
```

**Expected Impact**: +2-4% accuracy

---

## Strategy 10: Cost-Sensitive Learning (Expected Impact: +1-2%)

### Current Limitation
Treating all errors equally. Missing diabetic patients is more costly.

### Improvements to Implement

#### 10.1 Custom Loss Functions

```python
# For XGBoost - penalize false negatives more
def custom_objective(y_true, y_pred):
    grad = y_pred - y_true
    hess = np.ones_like(y_pred)
    
    # Increase gradient for false negatives
    grad[y_true == 1] *= 2  # Double penalty for missing diabetic patients
    
    return grad, hess

xgb_model = XGBClassifier(objective=custom_objective)
```

#### 10.2 Focal Loss (for imbalanced data)

```python
import tensorflow as tf

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
        
        loss = weight * cross_entropy
        return tf.reduce_mean(loss)
    
    return focal_loss_fixed

model.compile(loss=focal_loss(gamma=2., alpha=0.75), ...)
```

**Expected Impact**: +1-2% accuracy (especially for minority class)

---

## Recommended Implementation Plan

### Phase 1: Quick Wins (1-2 weeks) - Expected: +5-7%

1. **Feature Engineering** (Strategy 1)
   - Add clinical risk indicators
   - Create interaction features
   - Implement cholesterol ratios

2. **Hyperparameter Tuning** (Strategy 2)
   - RandomizedSearchCV for XGBoost
   - RandomizedSearchCV for Random Forest

3. **Better Imputation** (Strategy 9.1)
   - Replace SimpleImputer with IterativeImputer

**Expected Accuracy After Phase 1**: 74-76%

### Phase 2: Advanced Techniques (2-3 weeks) - Expected: +3-5%

4. **Ensemble Methods** (Strategy 3)
   - Implement stacking ensemble
   - Try voting classifier

5. **Advanced Models** (Strategy 7)
   - Add CatBoost
   - Add LightGBM

6. **Class Imbalance** (Strategy 4)
   - Try ADASYN or BorderlineSMOTE
   - Optimize classification threshold

**Expected Accuracy After Phase 2**: 77-81%

### Phase 3: Fine-Tuning (1-2 weeks) - Expected: +1-2%

7. **Feature Selection Refinement** (Strategy 5)
   - RFECV to find optimal feature count
   - Permutation importance

8. **Cross-Validation** (Strategy 8)
   - Implement nested CV for robust estimates

9. **Data Quality** (Strategy 9.2-9.3)
   - Outlier detection and treatment
   - Advanced scaling methods

**Expected Accuracy After Phase 3**: 78-83%

---

## Priority Ranking (by Impact/Effort Ratio)

### High Priority (Do First)
1. ✅ **Feature Engineering** - High impact, moderate effort
2. ✅ **Hyperparameter Tuning** - High impact, low effort (automated)
3. ✅ **Ensemble Methods** - High impact, moderate effort
4. ✅ **Advanced Models (CatBoost/LightGBM)** - High impact, low effort

### Medium Priority (Do Second)
5. ⚡ **Better Imputation** - Medium impact, low effort
6. ⚡ **Class Imbalance Handling** - Medium impact, low effort
7. ⚡ **Threshold Optimization** - Medium impact, very low effort

### Lower Priority (Do If Needed)
8. 📊 **Deep Learning (Neural Networks)** - Medium impact, high effort
9. 📊 **External Features** - Variable impact, high effort
10. 📊 **Cost-Sensitive Learning** - Low impact, moderate effort

---

## Code Template for Quick Implementation

Here's a ready-to-use template combining the top strategies:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import BorderlineSMOTE

# 1. FEATURE ENGINEERING
def engineer_features(df):
    """Add clinical and interaction features"""
    # BMI categories
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100],
                                labels=[0, 1, 2, 3])
    
    # Cholesterol ratios
    df['total_hdl_ratio'] = df['chol'] / (df['hdl'] + 1e-6)
    df['ldl_hdl_ratio'] = df['ldl'] / (df['hdl'] + 1e-6)
    
    # Metabolic syndrome
    df['metabolic_syndrome'] = (
        (df['waist'] > 102) & (df['tri'] >= 150) & 
        (df['hdl'] < 40) & (df['Ave_SBP'] >= 130)
    ).astype(int)
    
    # Interactions
    df['age_bmi'] = df['age'] * df['bmi']
    df['waist_hip_ratio'] = df['waist'] / (df['hip'] + 1e-6)
    
    # Blood pressure categories
    df['hypertension'] = ((df['Ave_SBP'] >= 140) | (df['Ave_DBP'] >= 90)).astype(int)
    
    return df

# 2. ADVANCED IMPUTATION
def advanced_imputation(X_train, X_val, X_test):
    """Use iterative imputer instead of simple mean"""
    imputer = IterativeImputer(max_iter=10, random_state=42)
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_val_imputed, X_test_imputed

# 3. BETTER SAMPLING
def advanced_sampling(X_train, y_train):
    """Use BorderlineSMOTE instead of regular SMOTE"""
    smote = BorderlineSMOTE(random_state=42, kind='borderline-1')
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

# 4. HYPERPARAMETER TUNING
def tune_xgboost(X_train, y_train):
    """Tune XGBoost with RandomizedSearch"""
    params = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200, 300, 500],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [1, 2, 3]
    }
    
    xgb = XGBClassifier(random_state=42)
    search = RandomizedSearchCV(
        xgb, params, n_iter=50, cv=5, 
        scoring='roc_auc', n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

# 5. STACKING ENSEMBLE
def create_stacking_ensemble(X_train, y_train):
    """Create stacking ensemble with best models"""
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=42)),
        ('cat', CatBoostClassifier(iterations=300, learning_rate=0.05, random_state=42, verbose=False))
    ]
    
    meta_model = LogisticRegression(max_iter=1000)
    
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    
    stacking.fit(X_train, y_train)
    return stacking

# MAIN PIPELINE
def improved_pipeline(X_train, y_train, X_val, y_val, X_test, y_test):
    """Complete improved pipeline"""
    
    # 1. Feature engineering
    print("Engineering features...")
    X_train_eng = engineer_features(X_train.copy())
    X_val_eng = engineer_features(X_val.copy())
    X_test_eng = engineer_features(X_test.copy())
    
    # 2. Advanced imputation
    print("Applying advanced imputation...")
    X_train_imp, X_val_imp, X_test_imp = advanced_imputation(
        X_train_eng, X_val_eng, X_test_eng
    )
    
    # 3. Better sampling
    print("Applying BorderlineSMOTE...")
    X_train_balanced, y_train_balanced = advanced_sampling(X_train_imp, y_train)
    
    # 4. Hyperparameter tuning
    print("Tuning XGBoost...")
    best_xgb = tune_xgboost(X_train_balanced, y_train_balanced)
    
    # 5. Stacking ensemble
    print("Creating stacking ensemble...")
    stacking_model = create_stacking_ensemble(X_train_balanced, y_train_balanced)
    
    # Evaluate
    print("\nEvaluating models...")
    xgb_score = best_xgb.score(X_test_imp, y_test)
    stacking_score = stacking_model.score(X_test_imp, y_test)
    
    print(f"Tuned XGBoost Accuracy: {xgb_score:.3f}")
    print(f"Stacking Ensemble Accuracy: {stacking_score:.3f}")
    
    return stacking_model, best_xgb

# Run the improved pipeline
final_model, xgb_model = improved_pipeline(X_train, y_train, X_val, y_val, X_test, y_test)
```

---

## Expected Timeline & Results

### Week 1-2: Feature Engineering + Hyperparameter Tuning
- **Expected Accuracy**: 74-76%
- **Effort**: Moderate
- **Confidence**: High

### Week 3-4: Ensemble Methods + Advanced Models
- **Expected Accuracy**: 77-79%
- **Effort**: Moderate
- **Confidence**: High

### Week 5-6: Fine-tuning + Optimization
- **Expected Accuracy**: 79-82%
- **Effort**: Low-Moderate
- **Confidence**: Medium-High

---

## Important Considerations

### 1. Diminishing Returns
- First 5% improvement is easier than next 5%
- 80-85% is realistic, >85% may require external data

### 2. Overfitting Risk
- More complex models → higher overfitting risk
- Always validate on held-out test set
- Use cross-validation extensively

### 3. Computational Cost
- Hyperparameter tuning is time-consuming
- Ensemble methods require more memory
- Consider using cloud computing if needed

### 4. Interpretability Trade-off
- Stacking ensembles are less interpretable
- May need to balance accuracy vs. explainability
- Keep simpler models for clinical explanation

---

## Conclusion

**YES, 80% accuracy is achievable!**

**Recommended Path:**
1. Start with feature engineering (+3-5%)
2. Add hyperparameter tuning (+2-4%)
3. Implement stacking ensemble (+3-6%)
4. Fine-tune with advanced sampling (+2-3%)

**Total Expected Improvement**: +10-18 percentage points
**Target Accuracy**: 79-87% (from current 69%)

**Most Likely Outcome**: 78-82% accuracy with Phase 1 + Phase 2 implementation.

Focus on the high-priority strategies first - they give you the best return on effort invested!

---

*Document Created: March 7, 2026*
*Model Improvement Strategies for Diabetes Risk Prediction*
