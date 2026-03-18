"""
Script to create a proper feature mapping from the preprocessed data
This will help identify what each feature_# represents
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import the custom transformers from revised_preprocessing
sys.path.insert(0, r'd:\School Files\CMSC\CMSC 199\SP\V2_SP')
from revised_preprocessing import FunctionTransformerToDataFrame, CorrelationDropper

# Load the preprocessor and metadata
preprocess_dir = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\PREPROCESS_OUTPUT")
preprocessor = joblib.load(preprocess_dir / "preprocessor.joblib")
meta = joblib.load(preprocess_dir / "meta.joblib")

print("Original features before preprocessing:", len(meta['original_feature_names']))
print("\nOriginal feature names:")
for i, name in enumerate(meta['original_feature_names'][:30]):
    print(f"  {i}: {name}")

# Get the feature names after transformation
try:
    transformed_names = preprocessor.get_feature_names_out()
    print(f"\nTransformed features after preprocessing: {len(transformed_names)}")
    
    # Create a detailed mapping
    feature_mapping = []
    
    # Get numeric and categorical transformers
    numeric_transformer = preprocessor.transformers_[0][1]
    categorical_transformer = preprocessor.transformers_[1][1]
    
    numeric_features = preprocessor.transformers_[0][2]
    categorical_features = preprocessor.transformers_[1][2]
    
    print(f"\nNumeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Get features that were dropped by correlation dropper
    corr_dropper = numeric_transformer.named_steps['corr_drop']
    dropped_features = corr_dropper.to_drop_
    print(f"\nFeatures dropped by correlation threshold: {len(dropped_features)}")
    print("Dropped features:", dropped_features[:10] if len(dropped_features) > 10 else dropped_features)
    
    # Numeric features that were kept
    numeric_kept = [f for f in numeric_features if f not in dropped_features]
    print(f"\nNumeric features kept: {len(numeric_kept)}")
    
    # Build the mapping
    feature_idx = 0
    
    # Add numeric features
    for orig_name in numeric_kept:
        feature_mapping.append({
            'feature_index': feature_idx,
            'transformed_name': f'feature_{feature_idx}',
            'original_name': orig_name,
            'feature_type': 'numeric',
            'transformation': 'mean_impute -> corr_drop -> minmax_scale',
            'source_dataset': get_source_dataset(orig_name)
        })
        feature_idx += 1
    
    # Add categorical features (one-hot encoded)
    if len(categorical_features) > 0:
        # Get one-hot encoder
        onehot = categorical_transformer.named_steps['onehot']
        cat_feature_names = onehot.get_feature_names_out(categorical_features)
        
        for cat_name in cat_feature_names:
            # Extract original feature name and category value
            # Format is usually: feature_value
            parts = cat_name.split('_', 1)
            if len(parts) == 2:
                orig_feature = parts[0]
                category_value = parts[1]
            else:
                orig_feature = cat_name
                category_value = 'unknown'
            
            feature_mapping.append({
                'feature_index': feature_idx,
                'transformed_name': f'feature_{feature_idx}',
                'original_name': cat_name,
                'feature_type': 'categorical_onehot',
                'transformation': 'mode_impute -> onehot_encode',
                'source_dataset': get_source_dataset(orig_feature),
                'category_value': category_value
            })
            feature_idx += 1
    
    # Create DataFrame
    mapping_df = pd.DataFrame(feature_mapping)
    
    # Save to CSV
    mapping_df.to_csv(preprocess_dir / "detailed_feature_mapping.csv", index=False)
    print(f"\n✅ Saved detailed feature mapping to {preprocess_dir / 'detailed_feature_mapping.csv'}")
    print(f"Total features mapped: {len(mapping_df)}")
    
    # Print summary by source dataset
    print("\n📊 Feature Distribution by Source Dataset:")
    print(mapping_df.groupby('source_dataset').size())
    
    # Print first 20 mappings
    print("\n📋 First 20 Feature Mappings:")
    print(mapping_df[['feature_index', 'transformed_name', 'original_name', 'source_dataset']].head(20).to_string(index=False))
    
except Exception as e:
    print(f"Error creating feature mapping: {e}")
    import traceback
    traceback.print_exc()


def get_source_dataset(feature_name):
    """Determine which dataset a feature comes from based on its name"""
    # Anthropometric features
    anthrop_features = ['weight', 'height', 'waist', 'hip', 'ethnicity', 'anthro_group', 
                       'mos_lactation', 'mos_preg', 'bmi', 'whr']
    
    # Biochemical features
    biochem_features = ['uic', 'vita', 'hemoglobin', 'fwgti_natl_var', 'fwgti_prov',
                       'fwgti_natl2_var', 'fwgti_prov2', 'rep_natl', 'rep_prov']
    
    # Clinical features
    clinical_features = ['Ave_SBP', 'Ave_DBP', 'currentsmoking', 'ever_smk', 'alcohol',
                        'con_alcohol', 'drnk_30days', 'drnk_30d_num', 'chol', 'tri',
                        'hdl', 'ldl', 'ms_psucode']
    
    # Dietary features (usually start with fg, epwt_fg, or Total_)
    if any(feature_name.lower().startswith(prefix) for prefix in ['fg', 'epwt_fg', 'total_']):
        return 'dietary'
    
    # Check other datasets
    feature_lower = feature_name.lower()
    if any(f in feature_lower for f in anthrop_features):
        return 'anthropometric'
    elif any(f in feature_lower for f in biochem_features):
        return 'biochemical'
    elif any(f in feature_lower for f in clinical_features):
        return 'clinical'
    else:
        # Try to infer from common patterns
        if 'food' in feature_lower or 'ener' in feature_lower or 'prot' in feature_lower:
            return 'dietary'
        return 'unknown'


if __name__ == "__main__":
    pass
