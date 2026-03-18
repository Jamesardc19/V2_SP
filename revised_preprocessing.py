import os
import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import joblib


# =========================
# Config
# =========================
@dataclass
class Config:
    dataset_dir: str = r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\DATASETS\MAIN DATASET"
    output_dir: str = r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\PREPROCESS_OUTPUT"
    random_state: int = 42

    # Split ratios: 70/15/15 (train/val/test)
    test_size: float = 0.15
    val_size_from_train: float = 0.17647058823529413  # 0.15 / 0.85

    # Correlation threshold for dropping numeric features
    corr_threshold: float = 0.90

    # SMOTE settings
    smote_k_neighbors: int = 5


CFG = Config()


# =========================
# Custom transformer to drop highly correlated numeric features
# Fit only on TRAIN to avoid leakage
# =========================
class CorrelationDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.to_drop_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # Expect X as a pandas DataFrame of numeric features
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = X.columns.tolist()
        corr = X.corr().abs()

        # Upper triangle to avoid duplicates
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = [col for col in upper.columns if any(upper[col] > self.threshold)]
        self.to_drop_ = to_drop
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            # If numpy, rebuild DF with original column names when possible
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X.drop(columns=self.to_drop_, errors="ignore")


# =========================
# Load datasets
# =========================
def load_datasets(dataset_dir: str):
    anthrop_file = os.path.join(dataset_dir, "data-set_anthrop.csv")
    biochemical_file = os.path.join(dataset_dir, "data-set_biochemical.csv")
    clinical_file = os.path.join(dataset_dir, "data-set_clinical.csv")
    dietary_file = os.path.join(dataset_dir, "data-set_dietary_indiv.csv")

    # Add low_memory=False to handle mixed data types warning
    anthrop_df = pd.read_csv(anthrop_file, low_memory=False)
    biochemical_df = pd.read_csv(biochemical_file, low_memory=False)
    clinical_df = pd.read_csv(clinical_file, low_memory=False)
    dietary_df = pd.read_csv(dietary_file, low_memory=False)

    print("Loaded:")
    print("  anthrop:", anthrop_df.shape)
    print("  biochemical:", biochemical_df.shape)
    print("  clinical:", clinical_df.shape)
    print("  dietary:", dietary_df.shape)

    return anthrop_df, biochemical_df, clinical_df, dietary_df


# =========================
# Merge datasets (FIX: clinical as base, left joins)
# =========================
def merge_datasets(anthrop_df, biochemical_df, clinical_df, dietary_df):
    # Use recommended composite key for merging
    # This is the safest because it pins the person down by: year, geography, household, member
    common_cols = ['enns_year', 'regcode', 'provhuc', 'hhnum', 'member_code']
    
    # Verify all datasets have the required columns
    for df, name in [(anthrop_df, "anthrop"), (biochemical_df, "biochemical"), 
                     (clinical_df, "clinical"), (dietary_df, "dietary")]:
        missing_cols = [col for col in common_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: {name} dataset is missing columns: {missing_cols}")
    
    # Use clinical as the base to avoid row explosion & preserve target alignment
    merged = clinical_df.copy()
    
    # Perform merges with the composite key
    merged = merged.merge(anthrop_df, on=common_cols, how='left', suffixes=('', '_anthrop'))
    merged = merged.merge(biochemical_df, on=common_cols, how='left', suffixes=('', '_biochem'))
    merged = merged.merge(dietary_df, on=common_cols, how='left', suffixes=('', '_dietary'))
    
    # Drop duplicated suffix columns if they were created
    duplicate_cols = [c for c in merged.columns if c.endswith(('_anthrop', '_biochem', '_dietary'))]
    merged = merged.drop(columns=duplicate_cols, errors='ignore')
    
    # Check for potential duplicates in the merge keys
    key_counts = clinical_df.groupby(common_cols).size().reset_index(name='count')
    duplicates = key_counts[key_counts['count'] > 1]
    if len(duplicates) > 0:
        print(f"Warning: Found {len(duplicates)} duplicate composite keys in clinical dataset")
    
    print("Merged shape:", merged.shape)
    return merged


# =========================
# Target transform + remove missing target
# =========================
def add_target_and_filter(df: pd.DataFrame):
    if "fbs" not in df.columns:
        raise ValueError("Column 'fbs' not found. Ensure clinical dataset contains 'fbs' and merge kept it.")

    # Drop rows with missing target (same as paper logic)
    before = len(df)
    df = df.dropna(subset=["fbs"]).copy()
    print(f"Dropped {before - len(df)} rows with missing fbs")

    # Binary target
    df["diabetes_risk"] = (df["fbs"] >= 100).astype(int)

    print("Class counts:\n", df["diabetes_risk"].value_counts())
    return df


# =========================
# Build preprocessing pipeline
# - Numeric: SimpleImputer -> CorrDrop -> MinMaxScaler
# - Categorical: mode impute -> OneHotEncode
# =========================
def build_preprocess_pipeline(df: pd.DataFrame, corr_threshold=0.90):
    # Define feature columns (exclude target + raw fbs)
    y_col = "diabetes_risk"
    drop_cols = ["fbs", y_col]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    
    # Early feature selection - remove low variance features
    from sklearn.feature_selection import VarianceThreshold
    
    # Identify numeric/categorical
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Apply variance threshold to numeric features only (optional)
    if len(numeric_features) > 20:  # Only apply if we have many numeric features
        print("Applying variance threshold for early feature selection...")
        selector = VarianceThreshold(threshold=0.01)  # Features with variance < 0.01 will be removed
        X_numeric = X[numeric_features]
        selector.fit(X_numeric)
        # Get selected feature names
        selected_numeric = [numeric_features[i] for i, selected in enumerate(selector.get_support()) if selected]
        print(f"Variance threshold reduced numeric features from {len(numeric_features)} to {len(selected_numeric)}")
        numeric_features = selected_numeric

    # Numeric pipeline:
    # 1) Simple imputation (mean) - much faster than KNN
    # 2) Drop highly correlated features (fit only on train)
    # 3) Scale (MinMaxScaler to avoid negative values)
    numeric_pipe = Pipeline(steps=[
        ("simple_impute", SimpleImputer(strategy="mean")),  # Replace KNN with SimpleImputer for speed
        ("to_df", FunctionTransformerToDataFrame(numeric_features)),
        ("corr_drop", CorrelationDropper(threshold=corr_threshold)),
        ("scaler", MinMaxScaler(feature_range=(0, 1))),  # Use MinMaxScaler to keep values positive
    ])

    # Categorical pipeline:
    # 1) mode impute
    # 2) one hot encode (handle unknown categories)
    categorical_pipe = Pipeline(steps=[
        ("impute_mode", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor, X.columns.tolist()


class FunctionTransformerToDataFrame(BaseEstimator, TransformerMixin):
    """
    Helper transformer: converts numpy array back to DataFrame with stable column names,
    needed for correlation dropper.
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Handle dimension mismatch by using only the available columns
        # This can happen if KNNImputer or other transformers change the number of columns
        if isinstance(X, np.ndarray) and X.shape[1] != len(self.columns):
            print(f"Warning: Column count mismatch. Expected {len(self.columns)}, got {X.shape[1]}")
            # Use only the first X.shape[1] columns or extend with dummy names if needed
            if X.shape[1] < len(self.columns):
                cols_to_use = self.columns[:X.shape[1]]
            else:
                cols_to_use = self.columns + [f"extra_{i}" for i in range(X.shape[1] - len(self.columns))]
            return pd.DataFrame(X, columns=cols_to_use)
        return pd.DataFrame(X, columns=self.columns)


# =========================
# Split data (70/15/15) BEFORE any fitting (FIX leakage)
# =========================
def split_data(df: pd.DataFrame, random_state=42):
    y = df["diabetes_risk"].astype(int)
    X = df.drop(columns=["diabetes_risk"], errors="ignore")

    # First split off test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=CFG.test_size, stratify=y, random_state=random_state
    )

    # Split trainval into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=CFG.val_size_from_train,  # makes val = 15% overall
        stratify=y_trainval,
        random_state=random_state
    )

    print("Split shapes:")
    print("  Train:", X_train.shape, y_train.shape)
    print("  Val:", X_val.shape, y_val.shape)
    print("  Test:", X_test.shape, y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test


# =========================
# Main
# =========================
def main():
    output_dir = Path(CFG.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    anthrop_df, biochemical_df, clinical_df, dietary_df = load_datasets(CFG.dataset_dir)

    # 2) Merge (fixed)
    merged_df = merge_datasets(anthrop_df, biochemical_df, clinical_df, dietary_df)

    # 3) Target + drop missing target
    merged_df = add_target_and_filter(merged_df)

    # 4) Split first (fix leakage)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(merged_df, CFG.random_state)

    # IMPORTANT: ensure fbs is not used as a feature (paper uses it only for label)
    # drop it from features
    for frame in [X_train, X_val, X_test]:
        if "fbs" in frame.columns:
            frame.drop(columns=["fbs"], inplace=True)

    # Build preprocess pipeline based on TRAIN columns
    # Recombine train to infer types
    train_df_for_types = X_train.copy()
    train_df_for_types["diabetes_risk"] = y_train.values

    # Print column info for debugging
    print(f"Number of columns in X_train: {X_train.shape[1]}")
    print(f"Column types: {X_train.dtypes.value_counts()}")

    # Save original column names for later reference
    original_feature_names = X_train.columns.tolist()

    preprocessor, raw_feature_list = build_preprocess_pipeline(train_df_for_types, CFG.corr_threshold)

    # 6) Full training pipeline with SMOTE applied AFTER preprocessing (TRAIN ONLY)
    pipeline = ImbPipeline(steps=[
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=CFG.random_state, k_neighbors=CFG.smote_k_neighbors)),
        # model goes later (TabNet / baseline models) — keep preprocessing reusable
    ])

    # Fit preprocessing + SMOTE on training set only
    X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)

    # Transform val/test (NO SMOTE there)
    X_val_proc = pipeline.named_steps["prep"].transform(X_val)
    X_test_proc = pipeline.named_steps["prep"].transform(X_test)

    # Save outputs
    np.save(output_dir / "X_train.npy", X_train_res)
    np.save(output_dir / "y_train.npy", y_train_res.to_numpy())
    np.save(output_dir / "X_val.npy", X_val_proc)
    np.save(output_dir / "y_val.npy", y_val.to_numpy())
    np.save(output_dir / "X_test.npy", X_test_proc)
    np.save(output_dir / "y_test.npy", y_test.to_numpy())

    # Save feature names for later reference
    try:
        feature_names = pipeline.named_steps["prep"].get_feature_names_out()
        joblib.dump(feature_names, output_dir / "feature_names.joblib")
        print(f"Saved {len(feature_names)} feature names")
    except Exception as e:
        print(f"Warning: Could not save feature names: {e}")

    # Create a CSV file like processed_dataset.csv without SMOTE augmentation
    try:
        # Process the original data without SMOTE for CSV export
        # This ensures we have a 1:1 mapping with the original data
        X_train_for_csv = pipeline.named_steps["prep"].transform(X_train)
        
        # Create a new DataFrame for the CSV
        csv_df = pd.DataFrame()
        
        # Try to get transformed feature names
        try:
            # Get feature names from the preprocessor
            transformed_feature_names = pipeline.named_steps["prep"].get_feature_names_out()
            
            # Map numeric features directly
            numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
            numeric_cols_kept = [col for col in numeric_features if col not in pipeline.named_steps["prep"].transformers_[0][1].named_steps["corr_drop"].to_drop_]
            
            # Create a mapping between original feature names and transformed feature names
            # For numeric features that weren't dropped
            feature_mapping = {}
            
            # Track which transformed features have been mapped
            mapped_indices = set()
            
            # First map numeric features (they maintain a 1:1 relationship)
            for i, name in enumerate(transformed_feature_names):
                if i < len(numeric_cols_kept):
                    feature_mapping[name] = numeric_cols_kept[i]
                    mapped_indices.add(i)
            
            # For remaining features (one-hot encoded), use the format "category_value"
            categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
            for i, name in enumerate(transformed_feature_names):
                if i not in mapped_indices:
                    # This is likely a one-hot encoded feature
                    # Try to extract the original categorical feature name
                    for cat_feature in categorical_features:
                        if cat_feature in name:
                            feature_mapping[name] = name  # Keep the full name which includes the category value
                            break
                    else:
                        # If we can't determine the original name, keep the transformed name
                        feature_mapping[name] = name
            
            # Use the mapped names for the DataFrame columns
            mapped_feature_names = [feature_mapping.get(name, name) for name in transformed_feature_names]
            
        except Exception as e:
            print(f"Warning: Could not map feature names: {e}")
            # Fall back to original feature names where possible, or use generic names
            if X_train_for_csv.shape[1] == len(original_feature_names):
                mapped_feature_names = original_feature_names
            else:
                mapped_feature_names = [f"feature_{i}" for i in range(X_train_for_csv.shape[1])]
        
        # Combine processed data (without SMOTE augmentation)
        X_all_no_smote = np.vstack([X_train_for_csv, X_val_proc, X_test_proc])
        y_all_no_smote = np.concatenate([y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy()])
        
        # Create DataFrame with processed features
        processed_df = pd.DataFrame(X_all_no_smote, columns=mapped_feature_names)
        
        # Add original identifier columns
        # Recombine the original data frames in the same order
        original_df = pd.concat([X_train, X_val, X_test], axis=0)
        id_columns = ['regcode', 'provhuc', 'enns_year', 'hhnum', 'member_code']
        
        # Add ID columns first
        for col in id_columns:
            if col in original_df.columns:
                csv_df[col] = original_df[col].reset_index(drop=True)
        
        # Add processed features
        for col in processed_df.columns:
            csv_df[col] = processed_df[col].reset_index(drop=True)
        
        # Add target variable
        csv_df['diabetes_risk'] = y_all_no_smote
        
        # Add original fbs column if available
        if 'fbs' in original_df.columns:
            csv_df['fbs'] = original_df['fbs'].reset_index(drop=True)
        
        # Save as CSV
        csv_path = output_dir / "processed_dataset_revised.csv"
        csv_df.to_csv(csv_path, index=False)
        print(f"Saved CSV file to {csv_path}")
        
        # Also save a feature mapping file for reference
        try:
            feature_map_df = pd.DataFrame({
                'transformed_name': list(transformed_feature_names),
                'original_name': [feature_mapping.get(name, 'unknown') for name in transformed_feature_names]
            })
            feature_map_df.to_csv(output_dir / "feature_mapping.csv", index=False)
            print(f"Saved feature mapping to {output_dir / 'feature_mapping.csv'}")
        except Exception as e:
            print(f"Warning: Could not save feature mapping: {e}")
            
    except Exception as e:
        print(f"Warning: Could not save CSV file: {e}")

    # Save the preprocessor (for model training + web app inference consistency)
    joblib.dump(pipeline.named_steps["prep"], output_dir / "preprocessor.joblib")

    # Save metadata
    meta = {
        "raw_feature_list": raw_feature_list,
        "original_feature_names": original_feature_names,
        "corr_threshold": CFG.corr_threshold,
        "smote_k_neighbors": CFG.smote_k_neighbors,
        "random_state": CFG.random_state,
        "imputation_method": "mean",  # Document the imputation method used
        "variance_threshold": 0.01,   # Document the variance threshold used for feature selection
    }
    joblib.dump(meta, output_dir / "meta.joblib")

    # Print summary of outputs
    print("\nFiles saved:")
    print(f"  - NumPy arrays for model training: {output_dir}/X_*.npy, y_*.npy")
    print(f"  - Preprocessor: {output_dir}/preprocessor.joblib")
    print(f"  - Metadata: {output_dir}/meta.joblib")
    print(f"  - CSV dataset: {output_dir}/processed_dataset_revised.csv")
    print("\nThe CSV file uses MinMaxScaler to ensure all values are positive.")

    print("\n✅ Preprocessing finished.")
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()