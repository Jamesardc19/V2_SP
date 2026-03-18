"""
Improved Preprocessing Pipeline V2 - PROPER FIX
Key changes:
1. Feature selection BEFORE one-hot encoding (not after)
2. Select from 126 engineered features, not 7853 encoded features
3. This prevents feature explosion and preserves important features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import BorderlineSMOTE
import joblib


@dataclass
class Config:
    """Configuration for improved preprocessing V2"""
    dataset_dir: Path = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\DATASETS\MAIN DATASET")
    output_dir: Path = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\PREPROCESS_OUTPUT_IMPROVED_V2")
    
    merge_keys = ['enns_year', 'regcode', 'provhuc', 'hhnum', 'member_code']
    
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    
    smote_k_neighbors = 5
    random_state = 42
    
    # Feature selection parameters (BEFORE one-hot encoding)
    n_features_to_select = 80  # Select 80 from 126 engineered features


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add clinical and interaction features"""
    
    def __init__(self):
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Add engineered features"""
        df = X.copy()
        
        # 1. BMI-related features
        if 'bmi' in df.columns:
            df['bmi_underweight'] = (df['bmi'] < 18.5).fillna(0).astype(int)
            df['bmi_normal'] = ((df['bmi'] >= 18.5) & (df['bmi'] < 25)).fillna(0).astype(int)
            df['bmi_overweight'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).fillna(0).astype(int)
            df['bmi_obese'] = (df['bmi'] >= 30).fillna(0).astype(int)
            df['bmi_squared'] = df['bmi'] ** 2
        
        # 2. Cholesterol ratios
        if all(col in df.columns for col in ['chol', 'hdl']):
            df['total_hdl_ratio'] = df['chol'] / (df['hdl'] + 1e-6)
        
        if all(col in df.columns for col in ['ldl', 'hdl']):
            df['ldl_hdl_ratio'] = df['ldl'] / (df['hdl'] + 1e-6)
        
        if all(col in df.columns for col in ['tri', 'hdl']):
            df['tri_hdl_ratio'] = df['tri'] / (df['hdl'] + 1e-6)
        
        # 3. Metabolic syndrome indicator
        if all(col in df.columns for col in ['waist', 'tri', 'hdl', 'Ave_SBP']):
            df['metabolic_syndrome'] = (
                (df['waist'] > 102) &
                (df['tri'] >= 150) &
                (df['hdl'] < 40) &
                (df['Ave_SBP'] >= 130)
            ).fillna(0).astype(int)
        
        # 4. Blood pressure categories
        if all(col in df.columns for col in ['Ave_SBP', 'Ave_DBP']):
            df['hypertension'] = ((df['Ave_SBP'] >= 140) | (df['Ave_DBP'] >= 90)).fillna(0).astype(int)
            df['prehypertension'] = (
                ((df['Ave_SBP'] >= 120) & (df['Ave_SBP'] < 140)) |
                ((df['Ave_DBP'] >= 80) & (df['Ave_DBP'] < 90))
            ).fillna(0).astype(int)
            df['pulse_pressure'] = df['Ave_SBP'] - df['Ave_DBP']
        
        # 5. Anthropometric ratios
        if all(col in df.columns for col in ['waist', 'hip']):
            df['waist_hip_ratio'] = df['waist'] / (df['hip'] + 1e-6)
        
        if all(col in df.columns for col in ['waist', 'height']):
            df['waist_height_ratio'] = df['waist'] / (df['height'] + 1e-6)
        
        # 6. Age-related interactions
        if 'age' in df.columns:
            if 'bmi' in df.columns:
                df['age_bmi_interaction'] = df['age'] * df['bmi']
            
            if 'waist' in df.columns:
                df['age_waist_interaction'] = df['age'] * df['waist']
            
            df['age_group_young'] = (df['age'] < 40).fillna(0).astype(int)
            df['age_group_middle'] = ((df['age'] >= 40) & (df['age'] < 60)).fillna(0).astype(int)
            df['age_group_senior'] = (df['age'] >= 60).fillna(0).astype(int)
        
        # 7. Lifestyle risk combinations
        if all(col in df.columns for col in ['currentsmoking', 'alcohol']):
            df['smoking_alcohol_risk'] = (df['currentsmoking'] * df['alcohol']).fillna(0).astype(int)
        
        if all(col in df.columns for col in ['currentsmoking', 'bmi']):
            df['smoking_obesity_risk'] = (df['currentsmoking'] * (df['bmi'] >= 30).astype(int)).fillna(0).astype(int)
        
        # 8. Lipid profile risk score
        if all(col in df.columns for col in ['chol', 'tri', 'hdl', 'ldl']):
            df['lipid_risk_score'] = (
                (df['chol'] >= 200).astype(int) +
                (df['tri'] >= 150).astype(int) +
                (df['hdl'] < 40).astype(int) +
                (df['ldl'] >= 130).astype(int)
            )
        
        # 9. Dietary risk indicators
        if 'Total_Carb' in df.columns and 'Total_Ener' in df.columns:
            df['carb_energy_ratio'] = df['Total_Carb'] / (df['Total_Ener'] + 1e-6)
        
        if 'Total_Fat' in df.columns and 'Total_Ener' in df.columns:
            df['fat_energy_ratio'] = df['Total_Fat'] / (df['Total_Ener'] + 1e-6)
        
        self.feature_names_ = df.columns.tolist()
        return df


def load_datasets(dataset_dir):
    """Load all four datasets"""
    print("Loading datasets...")
    
    anthrop = pd.read_csv(dataset_dir / "data-set_anthrop.csv")
    biochem = pd.read_csv(dataset_dir / "data-set_biochemical.csv")
    clinical = pd.read_csv(dataset_dir / "data-set_clinical.csv")
    dietary = pd.read_csv(dataset_dir / "data-set_dietary_indiv.csv")
    
    print(f"  Anthropometric: {anthrop.shape}")
    print(f"  Biochemical: {biochem.shape}")
    print(f"  Clinical: {clinical.shape}")
    print(f"  Dietary: {dietary.shape}")
    
    return anthrop, biochem, clinical, dietary


def merge_datasets(anthrop, biochem, clinical, dietary, merge_keys):
    """Merge datasets using composite key"""
    print("\nMerging datasets...")
    
    merged = clinical.copy()
    merged = merged.merge(anthrop, on=merge_keys, how='left', suffixes=('', '_anthrop'))
    print(f"  After anthrop merge: {merged.shape}")
    
    merged = merged.merge(biochem, on=merge_keys, how='left', suffixes=('', '_biochem'))
    print(f"  After biochem merge: {merged.shape}")
    
    merged = merged.merge(dietary, on=merge_keys, how='left', suffixes=('', '_dietary'))
    print(f"  After dietary merge: {merged.shape}")
    
    return merged


def create_target_variable(df):
    """Create diabetes_risk target from FBS"""
    print("\nCreating target variable...")
    
    if 'fbs' not in df.columns:
        raise ValueError("FBS column not found in dataset")
    
    df = df.dropna(subset=['fbs'])
    print(f"  After removing missing FBS: {df.shape}")
    
    df['diabetes_risk'] = (df['fbs'] >= 100).astype(int)
    
    class_dist = df['diabetes_risk'].value_counts(normalize=True)
    print(f"  Class distribution:")
    print(f"    Non-diabetic (0): {class_dist[0]:.2%}")
    print(f"    Diabetic (1): {class_dist[1]:.2%}")
    
    return df


def split_data(df, merge_keys, train_ratio, val_ratio, test_ratio, random_state):
    """Split data into train/val/test sets"""
    print("\nSplitting data...")
    
    X = df.drop(columns=['diabetes_risk', 'fbs'])
    y = df['diabetes_risk']
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y
    )
    
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def select_features_before_encoding(X_train, y_train, X_val, X_test, n_features, random_state):
    """
    CRITICAL FIX: Select features BEFORE one-hot encoding
    This prevents selecting from 7853 noisy features
    """
    print(f"\n🔧 CRITICAL FIX: Feature selection BEFORE one-hot encoding")
    print(f"  Selecting top {n_features} from {X_train.shape[1]} engineered features...")
    
    # Separate numeric and categorical features
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    
    # Impute numeric features first (needed for feature selection)
    imputer = SimpleImputer(strategy='mean')
    X_train_num_imputed_array = imputer.fit_transform(X_train[numeric_features])
    
    # Get feature names after imputation (some may be dropped if all NaN)
    imputed_feature_names = [numeric_features[i] for i in range(len(numeric_features)) 
                             if i < X_train_num_imputed_array.shape[1]]
    
    X_train_num_imputed = pd.DataFrame(
        X_train_num_imputed_array,
        columns=imputed_feature_names,
        index=X_train.index
    )
    
    # Update numeric_features to only include successfully imputed features
    numeric_features = imputed_feature_names
    
    # Combined feature selection: Statistical + Model-based
    print("  Using combined feature selection (ANOVA F-test + Random Forest)...")
    
    # Step 1: Statistical selection (ANOVA F-test)
    k_statistical = min(int(n_features * 1.5), len(numeric_features))
    selector_statistical = SelectKBest(f_classif, k=k_statistical)
    selector_statistical.fit(X_train_num_imputed, y_train)
    
    # Get top features from statistical test
    statistical_scores = pd.Series(
        selector_statistical.scores_,
        index=numeric_features
    ).sort_values(ascending=False)
    
    top_statistical_features = statistical_scores.head(k_statistical).index.tolist()
    
    # Step 2: Model-based selection (Random Forest)
    print("  Training Random Forest for feature importance...")
    rf_selector = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1
    )
    rf_selector.fit(X_train_num_imputed[top_statistical_features], y_train)
    
    # Get feature importances
    feature_importance = pd.Series(
        rf_selector.feature_importances_,
        index=top_statistical_features
    ).sort_values(ascending=False)
    
    # Select top n_features
    selected_numeric_features = feature_importance.head(n_features).index.tolist()
    
    # For categorical features, only keep low-cardinality ones (to prevent explosion)
    selected_categorical = []
    for cat_col in categorical_features:
        n_unique = X_train[cat_col].nunique()
        if n_unique <= 20:  # Only keep if <= 20 unique values
            selected_categorical.append(cat_col)
            print(f"    Keeping categorical '{cat_col}' ({n_unique} unique values)")
        else:
            print(f"    Dropping categorical '{cat_col}' ({n_unique} unique values - too high cardinality)")
    
    selected_features = selected_numeric_features + selected_categorical
    
    print(f"  ✅ Selected {len(selected_numeric_features)} numeric + {len(selected_categorical)} categorical = {len(selected_features)} total features")
    
    # Filter datasets
    X_train_selected = X_train[selected_features].copy()
    X_val_selected = X_val[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()
    
    # Save feature selection info
    feature_info = {
        'selected_features': selected_features,
        'selected_numeric': selected_numeric_features,
        'selected_categorical': categorical_features,
        'feature_importance': feature_importance.to_dict(),
        'statistical_scores': statistical_scores.to_dict()
    }
    
    return X_train_selected, X_val_selected, X_test_selected, feature_info


def build_preprocessing_pipeline(X):
    """Build preprocessing pipeline (after feature selection)"""
    print("\nBuilding preprocessing pipeline...")
    
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    
    # Numeric pipeline
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler(feature_range=(0, 1)))
    ])
    
    # Categorical pipeline
    categorical_pipe = Pipeline(steps=[
        ("mode_impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor


def main():
    """Main preprocessing pipeline V2"""
    CFG = Config()
    
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("IMPROVED PREPROCESSING V2 - PROPER FIX")
    print("="*80)
    print("Key improvements:")
    print("  1. Feature selection BEFORE one-hot encoding")
    print("  2. Select from 126 engineered features (not 7853)")
    print("  3. Combined statistical + model-based selection")
    print("="*80)
    
    # Load and merge datasets
    anthrop, biochem, clinical, dietary = load_datasets(CFG.dataset_dir)
    merged = merge_datasets(anthrop, biochem, clinical, dietary, CFG.merge_keys)
    merged = create_target_variable(merged)
    
    # Apply feature engineering
    print("\nApplying feature engineering...")
    feature_engineer = FeatureEngineer()
    
    id_cols = CFG.merge_keys.copy()
    X_with_ids = merged.drop(columns=['diabetes_risk', 'fbs'])
    X_features = X_with_ids.drop(columns=id_cols, errors='ignore')
    
    X_engineered = feature_engineer.fit_transform(X_features)
    
    print(f"  Features after engineering: {X_engineered.shape[1]}")
    print(f"  Added {X_engineered.shape[1] - X_features.shape[1]} new features")
    
    # Add back ID columns and target
    for col in id_cols:
        if col in X_with_ids.columns:
            X_engineered[col] = X_with_ids[col].values
    
    df_engineered = X_engineered.copy()
    df_engineered['diabetes_risk'] = merged['diabetes_risk'].values
    df_engineered['fbs'] = merged['fbs'].values
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df_engineered, CFG.merge_keys, CFG.train_ratio, CFG.val_ratio, CFG.test_ratio, CFG.random_state
    )
    
    # Remove ID columns
    X_train = X_train.drop(columns=id_cols, errors='ignore')
    X_val = X_val.drop(columns=id_cols, errors='ignore')
    X_test = X_test.drop(columns=id_cols, errors='ignore')
    
    # Convert categorical to string
    cat_cols = X_train.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)
        X_test[col] = X_test[col].astype(str)
    
    # CRITICAL: Feature selection BEFORE one-hot encoding
    X_train_selected, X_val_selected, X_test_selected, feature_info = select_features_before_encoding(
        X_train, y_train, X_val, X_test, CFG.n_features_to_select, CFG.random_state
    )
    
    # NOW apply preprocessing (including one-hot encoding)
    preprocessor = build_preprocessing_pipeline(X_train_selected)
    
    print("\nFitting preprocessing pipeline...")
    preprocessor.fit(X_train_selected, y_train)
    
    print("Transforming data...")
    X_train_proc = preprocessor.transform(X_train_selected)
    X_val_proc = preprocessor.transform(X_val_selected)
    X_test_proc = preprocessor.transform(X_test_selected)
    
    print(f"  ✅ Processed features: {X_train_proc.shape[1]} (much better than 7853!)")
    
    # Apply SMOTE
    print("\nApplying BorderlineSMOTE...")
    smote = BorderlineSMOTE(k_neighbors=CFG.smote_k_neighbors, random_state=CFG.random_state, kind='borderline-1')
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_proc, y_train)
    
    print(f"  Training samples after SMOTE: {X_train_balanced.shape[0]}")
    print(f"  Class distribution: {np.bincount(y_train_balanced)}")
    
    # Save processed data
    print("\nSaving processed data...")
    np.save(CFG.output_dir / "X_train.npy", X_train_balanced)
    np.save(CFG.output_dir / "y_train.npy", y_train_balanced)
    np.save(CFG.output_dir / "X_val.npy", X_val_proc)
    np.save(CFG.output_dir / "y_val.npy", y_val.to_numpy())
    np.save(CFG.output_dir / "X_test.npy", X_test_proc)
    np.save(CFG.output_dir / "y_test.npy", y_test.to_numpy())
    
    # Save preprocessor and feature info
    joblib.dump(preprocessor, CFG.output_dir / "preprocessor.joblib")
    joblib.dump(feature_engineer, CFG.output_dir / "feature_engineer.joblib")
    joblib.dump(feature_info, CFG.output_dir / "feature_selection_info.joblib")
    
    # Save metadata
    meta = {
        "original_feature_count": X_features.shape[1],
        "engineered_feature_count": X_engineered.shape[1] - len(id_cols),
        "selected_before_encoding": len(feature_info['selected_features']),
        "processed_feature_count": X_train_proc.shape[1],
        "train_samples": X_train_balanced.shape[0],
        "val_samples": X_val_proc.shape[0],
        "test_samples": X_test_proc.shape[0],
        "improvement": "Feature selection BEFORE one-hot encoding"
    }
    joblib.dump(meta, CFG.output_dir / "meta.joblib")
    
    print("\n" + "="*80)
    print("✅ IMPROVED PREPROCESSING V2 COMPLETE!")
    print("="*80)
    print(f"Files saved to: {CFG.output_dir}")
    print(f"\nFeature count comparison:")
    print(f"  Original features: {meta['original_feature_count']}")
    print(f"  After engineering: {meta['engineered_feature_count']}")
    print(f"  Selected (before encoding): {meta['selected_before_encoding']}")
    print(f"  After preprocessing: {meta['processed_feature_count']}")
    print(f"\n🎯 Key improvement: {meta['processed_feature_count']} features instead of 7853!")
    print("="*80)


if __name__ == "__main__":
    main()
