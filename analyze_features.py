"""
Analyze and create feature mapping for the preprocessed dataset
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Load metadata
preprocess_dir = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\PREPROCESS_OUTPUT")
meta = joblib.load(preprocess_dir / "meta.joblib")

# Load the CSV to see what we actually have
csv_path = preprocess_dir / "processed_dataset_revised.csv"
df = pd.read_csv(csv_path, nrows=5)

print("="*80)
print("FEATURE ANALYSIS")
print("="*80)

# Get column names from CSV
csv_columns = df.columns.tolist()
print(f"\nTotal columns in CSV: {len(csv_columns)}")

# Separate ID columns, feature columns, and target
id_columns = ['regcode', 'provhuc', 'enns_year', 'hhnum', 'member_code']
target_columns = ['diabetes_risk', 'fbs']

feature_columns = [col for col in csv_columns if col not in id_columns + target_columns]
print(f"Feature columns: {len(feature_columns)}")
print(f"ID columns: {id_columns}")
print(f"Target columns: {[col for col in target_columns if col in csv_columns]}")

# Get original feature names
original_features = meta['original_feature_names']
print(f"\nOriginal features before preprocessing: {len(original_features)}")

# Categorize original features by source dataset
def get_source_dataset(feature_name):
    """Determine which dataset a feature comes from"""
    feature_lower = feature_name.lower()
    
    # Anthropometric features
    anthrop_keywords = ['weight', 'height', 'waist', 'hip', 'ethnicity', 'anthro', 
                        'lactation', 'preg', 'bmi', 'whr']
    
    # Biochemical features  
    biochem_keywords = ['uic', 'vita', 'hemoglobin', 'fwgti', 'rep_']
    
    # Clinical features
    clinical_keywords = ['sbp', 'dbp', 'smoking', 'smk', 'alcohol', 'drnk', 'chol', 
                        'tri', 'hdl', 'ldl', 'psucode', 'pa_met', 'binge']
    
    # Dietary features
    dietary_keywords = ['fg', 'epwt', 'total_', 'food', 'ener', 'prot', 'carb', 
                       'fat', 'fiber', 'calc', 'iron', 'zinc', 'vit']
    
    # Check each category
    if any(kw in feature_lower for kw in dietary_keywords):
        return 'Dietary'
    elif any(kw in feature_lower for kw in anthrop_keywords):
        return 'Anthropometric'
    elif any(kw in feature_lower for kw in biochem_keywords):
        return 'Biochemical'
    elif any(kw in feature_lower for kw in clinical_keywords):
        return 'Clinical'
    else:
        return 'Unknown'

# Analyze original features
feature_sources = {}
for feat in original_features:
    if feat not in id_columns + ['fbs', 'diabetes_risk']:
        source = get_source_dataset(feat)
        if source not in feature_sources:
            feature_sources[source] = []
        feature_sources[source].append(feat)

print("\n" + "="*80)
print("ORIGINAL FEATURES BY SOURCE DATASET")
print("="*80)
for source, features in sorted(feature_sources.items()):
    print(f"\n{source} ({len(features)} features):")
    for i, feat in enumerate(features[:10], 1):
        print(f"  {i}. {feat}")
    if len(features) > 10:
        print(f"  ... and {len(features) - 10} more")

# Create a comprehensive feature documentation
print("\n" + "="*80)
print("CREATING FEATURE DOCUMENTATION")
print("="*80)

# Create mapping DataFrame
mapping_data = []

# Add ID columns
for i, col in enumerate(id_columns):
    if col in csv_columns:
        mapping_data.append({
            'csv_column_name': col,
            'original_feature_name': col,
            'feature_type': 'identifier',
            'source_dataset': 'All datasets',
            'description': f'Composite key component: {col}',
            'preprocessing': 'None (preserved as-is)'
        })

# For feature_# columns, we need to infer what they represent
# Since we don't have the exact mapping, we'll document what we know
print(f"\nProcessing {len(feature_columns)} transformed features...")

# The preprocessing pipeline does:
# 1. Numeric features: mean imputation -> correlation drop -> MinMaxScaler
# 2. Categorical features: mode imputation -> one-hot encoding

# We know from the output that we have 1557 features after preprocessing
# This is much more than the 104 original features, indicating one-hot encoding expanded categorical features

# Create a note about the transformation
for i, col in enumerate(feature_columns):
    mapping_data.append({
        'csv_column_name': col,
        'original_feature_name': 'See detailed_analysis.txt',
        'feature_type': 'transformed',
        'source_dataset': 'Multiple',
        'description': 'Preprocessed feature (numeric scaled or categorical one-hot encoded)',
        'preprocessing': 'Imputation -> Correlation drop (numeric) or One-hot encoding (categorical) -> Scaling'
    })

# Add target columns
for col in target_columns:
    if col in csv_columns:
        desc = 'Binary diabetes risk (1 if FBS >= 100, else 0)' if col == 'diabetes_risk' else 'Fasting Blood Sugar (mg/dL)'
        mapping_data.append({
            'csv_column_name': col,
            'original_feature_name': col,
            'feature_type': 'target' if col == 'diabetes_risk' else 'continuous_target',
            'source_dataset': 'Clinical',
            'description': desc,
            'preprocessing': 'None' if col == 'fbs' else 'Derived from FBS'
        })

# Create DataFrame
mapping_df = pd.DataFrame(mapping_data)

# Save to CSV
output_path = preprocess_dir / "feature_documentation.csv"
mapping_df.to_csv(output_path, index=False)
print(f"\n✅ Saved feature documentation to: {output_path}")

# Create a detailed text file with original features
with open(preprocess_dir / "detailed_analysis.txt", 'w') as f:
    f.write("="*80 + "\n")
    f.write("DETAILED FEATURE ANALYSIS\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Total columns in CSV: {len(csv_columns)}\n")
    f.write(f"  - ID columns: {len(id_columns)}\n")
    f.write(f"  - Feature columns: {len(feature_columns)}\n")
    f.write(f"  - Target columns: {len([c for c in target_columns if c in csv_columns])}\n\n")
    
    f.write(f"Original features before preprocessing: {len(original_features)}\n")
    f.write(f"Features after preprocessing: {len(feature_columns)}\n")
    f.write(f"Expansion ratio: {len(feature_columns) / max(1, len([f for f in original_features if f not in id_columns])):.1f}x\n\n")
    
    f.write("="*80 + "\n")
    f.write("ORIGINAL FEATURES BY SOURCE DATASET\n")
    f.write("="*80 + "\n\n")
    
    for source, features in sorted(feature_sources.items()):
        f.write(f"\n{source} Dataset ({len(features)} features):\n")
        f.write("-" * 40 + "\n")
        for feat in features:
            f.write(f"  • {feat}\n")
    
    f.write("\n\n" + "="*80 + "\n")
    f.write("PREPROCESSING PIPELINE\n")
    f.write("="*80 + "\n\n")
    
    f.write("Numeric Features:\n")
    f.write("  1. Simple Imputation (mean strategy)\n")
    f.write("  2. Variance Threshold (removes features with variance < 0.01)\n")
    f.write("  3. Correlation Dropper (removes features with correlation > 0.90)\n")
    f.write("  4. MinMaxScaler (scales to range 0-1)\n\n")
    
    f.write("Categorical Features:\n")
    f.write("  1. Mode Imputation\n")
    f.write("  2. One-Hot Encoding (creates binary columns for each category)\n\n")
    
    f.write("Note: The large increase in features (104 -> 1557) is primarily due to\n")
    f.write("one-hot encoding of categorical variables, which creates a separate\n")
    f.write("binary column for each unique category value.\n\n")
    
    f.write("="*80 + "\n")
    f.write("RECOMMENDATION\n")
    f.write("="*80 + "\n\n")
    f.write("The feature_# naming is a limitation of the current preprocessing pipeline.\n")
    f.write("For your thesis, you should:\n\n")
    f.write("1. Use the feature selection step (which reduces to 40 features) to make\n")
    f.write("   the model more interpretable\n\n")
    f.write("2. After feature selection, you can examine which of the 40 selected features\n")
    f.write("   are most important and manually map them back to original features\n\n")
    f.write("3. Focus your analysis on the selected features rather than all 1557\n\n")
    f.write("4. The model training script will save feature importance scores that can\n")
    f.write("   help identify which features matter most for diabetes prediction\n")

print(f"✅ Saved detailed analysis to: {preprocess_dir / 'detailed_analysis.txt'}")

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n📊 Dataset Statistics:")
print(f"  • Original features: {len(original_features)}")
print(f"  • After preprocessing: {len(feature_columns)}")
print(f"  • Expansion: {len(feature_columns) / max(1, len([f for f in original_features if f not in id_columns])):.1f}x (due to one-hot encoding)")

print(f"\n📁 Files Created:")
print(f"  • {output_path}")
print(f"  • {preprocess_dir / 'detailed_analysis.txt'}")

print(f"\n💡 Key Insight:")
print(f"  The generic feature_# names are due to the preprocessing pipeline.")
print(f"  However, the feature selection step (in model training) will reduce")
print(f"  this to 40 most important features, making the model much more interpretable.")
print(f"  Focus your analysis on those 40 selected features for your thesis.")
