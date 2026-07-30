"""
Preprocessing Pipeline V7 - Clean Rebuild

Fixes from V6:
1. Clinical as LEFT base in merge (all records have FBS)
2. ALL enns_year_* merge artifacts dropped
3. Explicit categorical list -> mode imputation (not KNN)
4. Special missing codes -> NaN (ever_smk=99, con_alcohol/drnk_30days=9)
5. Population filter via anthro_group (children 1-3, pregnant=5, lactating=6)
6. Minimal feature engineering: bmi, waist_hip_ratio, chol_hdl_ratio, ldl_hdl_ratio ONLY
7. RAW unscaled CSV exported for EDA before any scaling
8. Single StandardScaler on continuous features only
9. Correlation drop computed on train set only
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')


class Config:
    dataset_dir = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\DATASETS\2018-2021\MAIN DATASET")
    output_dir  = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\PREPROCESS_OUTPUT_V7")
    merge_keys  = ['hhnum', 'member_code']

    fbs_threshold     = 100  # mg/dL; FBS >= 100 = At-Risk (pre-diabetes + diabetes)
    fbs_normal_max    = 100  # used for display only

    # anthro_group: 1-3=children, 5=pregnant, 6=lactating -> exclude
    exclude_anthro_groups = [1, 2, 3, 5, 6]

    use_dietary = False  # dietary features add noise; clinical+biochem+anthrop only

    test_size    = 0.15
    val_size     = 0.15
    random_state = 42

    resampling_method         = 'none'  # 'smotetomek' | 'smote' | 'undersample' | 'none'
    smote_sampling_strategy   = 0.6
    smote_k_neighbors         = 5
    undersample_strategy      = 0.7

    optimize_knn      = True
    knn_neighbors     = 5
    knn_candidates    = [3, 5, 7, 9, 11]
    masking_fraction  = 0.10

    handle_outliers  = True
    iqr_multiplier   = 1.5

    correlation_threshold = 0.92

    # Integer-coded categoricals -> mode imputation, NO StandardScaler
    CATEGORICAL_FEATURES = [
        'currentsmoking', 'ever_smk', 'alcohol', 'con_alcohol',
        'drnk_30days', 'smoke_status', 'alcohol_status', 'binge_drink', 'pa_met',
    ]

    # Special "not applicable" codes to recode as NaN
    SPECIAL_MISSING = {
        'ever_smk':   [99],
        'con_alcohol': [9],
        'drnk_30days': [9],
    }


# =============================================================================
# STEP 1 – LOAD & MERGE
# =============================================================================

def load_and_merge(config):
    print("\n" + "="*70)
    print("LOADING AND MERGING DATASETS")
    print("="*70)

    clinical = pd.read_csv(config.dataset_dir / 'data-set_clinical.csv',       low_memory=False)
    anthrop  = pd.read_csv(config.dataset_dir / 'data-set_anthrop.csv',        low_memory=False)
    biochem  = pd.read_csv(config.dataset_dir / 'data-set_biochemical.csv',    low_memory=False)
    dietary  = pd.read_csv(config.dataset_dir / 'data-set_dietary_indiv.csv',  low_memory=False)

    print(f"  Clinical : {clinical.shape}")
    print(f"  Anthrop  : {anthrop.shape}")
    print(f"  Biochem  : {biochem.shape}")
    print(f"  Dietary  : {dietary.shape}")

    # Clinical is the LEFT base -> every row has FBS
    df = clinical.merge(anthrop,  on=config.merge_keys, how='left', suffixes=('', '_anthrop'))
    df = df.merge(biochem,        on=config.merge_keys, how='left', suffixes=('', '_biochem'))
    if config.use_dietary:
        df = df.merge(dietary,    on=config.merge_keys, how='left', suffixes=('', '_dietary'))

    print(f"\n  Merged shape: {df.shape}")
    return df


# =============================================================================
# STEP 2 – DROP METADATA COLUMNS
# =============================================================================

def drop_metadata_columns(df):
    print("\n" + "="*70)
    print("DROPPING METADATA COLUMNS")
    print("="*70)

    # Patterns that identify non-feature columns (checked as prefix or exact)
    metadata_prefixes = [
        'regcode', 'provhuc', 'enns_year',
        'fwgti', 'rep_natl', 'rep_prov', 'ms_psucode',
        'hhnum', 'member_code',
        'ethnicity',
        'Total_Food',   # redundant sum of food groups (Total_Food and Total_Food_epwt)
        'epwt_fg',      # person-weighted food group amounts (redundant with fg* raw counts)
    ]
    # anthro_group / mos_* kept until AFTER population filtering

    to_drop = []
    for col in df.columns:
        for pat in metadata_prefixes:
            if col == pat or col.startswith(pat + '_') or col.startswith(pat):
                to_drop.append(col)
                break

    df = df.drop(columns=to_drop, errors='ignore')
    print(f"  Dropped {len(to_drop)} metadata columns -> {df.shape[1]} remaining")
    return df


# =============================================================================
# STEP 3 – RECODE SPECIAL MISSING VALUES
# =============================================================================

def recode_special_missing(df, config):
    print("\n" + "="*70)
    print("RECODING SPECIAL MISSING CODES -> NaN")
    print("="*70)
    for col, bad_vals in config.SPECIAL_MISSING.items():
        if col in df.columns:
            n = df[col].isin(bad_vals).sum()
            df[col] = df[col].replace(bad_vals, np.nan)
            print(f"  {col}: {n:,} values recoded")
    return df


# =============================================================================
# STEP 4 – POPULATION FILTERING
# =============================================================================

def filter_population(df, config):
    print("\n" + "="*70)
    print("POPULATION FILTERING")
    print("="*70)
    n0 = len(df)

    if 'anthro_group' in df.columns:
        mask = df['anthro_group'].isin(config.exclude_anthro_groups)
        df = df[~mask]
        print(f"  Removed {mask.sum():,} rows (children/pregnant/lactating via anthro_group)")

    # Drop exclusion-related columns after filtering
    df = df.drop(columns=['anthro_group', 'mos_lactation', 'mos_preg'], errors='ignore')

    print(f"  Rows: {n0:,} -> {len(df):,} ({100*len(df)/n0:.1f}% retained)")
    return df


# =============================================================================
# STEP 5 – CREATE TARGET VARIABLE
# =============================================================================

def create_target(df, config):
    print("\n" + "="*70)
    print("CREATING TARGET VARIABLE")
    print("="*70)

    n0 = len(df)
    df = df.dropna(subset=['fbs'])
    print(f"  Dropped {n0 - len(df):,} rows with missing FBS")

    df['diabetes'] = (df['fbs'] >= config.fbs_threshold).astype(int)
    df = df.drop(columns=['fbs'])

    n_normal  = (df['diabetes'] == 0).sum()
    n_atrisk  = (df['diabetes'] == 1).sum()
    print(f"  Class 0 (FBS < {config.fbs_threshold}): {n_normal:,}  ({100*n_normal/len(df):.1f}%)")
    print(f"  Class 1 (FBS >= {config.fbs_threshold}): {n_atrisk:,} ({100*n_atrisk/len(df):.1f}%)")
    print(f"  Imbalance ratio: {n_normal/n_atrisk:.2f}:1")
    return df


# =============================================================================
# STEP 6 – FEATURE ENGINEERING (minimal)
# =============================================================================

def engineer_features(df):
    print("\n" + "="*70)
    print("FEATURE ENGINEERING (minimal)")
    print("="*70)

    created = []

    if 'weight' in df.columns and 'height' in df.columns:
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        created.append('bmi')
        # Drop raw components — bmi is the clinically interpretable composite
        df = df.drop(columns=['weight', 'height'], errors='ignore')

    if 'waist' in df.columns and 'hip' in df.columns:
        df['waist_hip_ratio'] = df['waist'] / df['hip']
        created.append('waist_hip_ratio')
        # Drop raw components — waist_hip_ratio is the risk-relevant metric
        df = df.drop(columns=['waist', 'hip'], errors='ignore')

    if 'chol' in df.columns and 'hdl' in df.columns:
        df['chol_hdl_ratio'] = df['chol'] / df['hdl']
        created.append('chol_hdl_ratio')
        df['non_hdl_chol'] = df['chol'] - df['hdl']   # non-HDL cholesterol
        created.append('non_hdl_chol')

    if 'ldl' in df.columns and 'hdl' in df.columns:
        df['ldl_hdl_ratio'] = df['ldl'] / df['hdl']
        created.append('ldl_hdl_ratio')

    if 'tri' in df.columns and 'hdl' in df.columns:
        eps = 1e-6
        df['tri_hdl_ratio'] = df['tri'] / (df['hdl'] + eps)   # insulin-resistance proxy
        df['atherogenic_index'] = np.log((df['tri'] + eps) / (df['hdl'] + eps))
        created.extend(['tri_hdl_ratio', 'atherogenic_index'])

    if 'Ave_SBP' in df.columns and 'Ave_DBP' in df.columns:
        df['pulse_pressure'] = df['Ave_SBP'] - df['Ave_DBP']   # arterial stiffness marker
        df['mean_arterial_pressure'] = df['Ave_DBP'] + df['pulse_pressure'] / 3
        created.extend(['pulse_pressure', 'mean_arterial_pressure'])

    if 'bmi' in df.columns and 'Ave_SBP' in df.columns:
        df['bmi_sbp_interact'] = df['bmi'] * df['Ave_SBP'] / 100   # obesity + hypertension
        created.append('bmi_sbp_interact')

    if 'bmi' in df.columns and 'tri' in df.columns:
        df['bmi_tri_interact'] = df['bmi'] * df['tri'] / 100   # obesity + triglyceride risk
        created.append('bmi_tri_interact')

    print(f"  Created: {created}")
    print(f"  (Removed vs V6: 6 BMI flags, whr_high_risk, metabolic_risk_score, metabolic_syndrome_risk)")
    return df


# =============================================================================
# STEP 7 – EXPORT RAW UNSCALED CSV FOR EDA
# =============================================================================

def export_raw_csv(df, config):
    print("\n" + "="*70)
    print("EXPORTING RAW UNSCALED MERGED CSV")
    print("="*70)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / 'raw_merged.csv'
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    print(f"  Shape: {df.shape} (NOT scaled, for EDA use)")
    return path


# =============================================================================
# STEP 7b – EDA VISUALIZATIONS (on raw data)
# =============================================================================

def generate_eda_visualizations(df, config):
    print("\n" + "="*70)
    print("GENERATING EDA VISUALIZATIONS")
    print("="*70)
    eda_dir = config.output_dir / 'EDA'
    eda_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns if c != 'diabetes']
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    df['diabetes'].value_counts().sort_index().plot(kind='bar', ax=ax,
        color=['steelblue', 'tomato'], edgecolor='black')
    ax.set_xticklabels(['Normal (0)', 'At-Risk (1)'], rotation=0)
    ax.set_title('Target Class Distribution')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(eda_dir / 'class_distribution.png', dpi=100)
    plt.close()

    # 2. Correlation heatmap (continuous features only, top 30 by variance)
    num_df = df[num_cols + ['diabetes']].copy()
    if len(num_cols) > 30:
        variances = num_df[num_cols].var().nlargest(30).index.tolist()
        num_df = num_df[variances + ['diabetes']]
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr, cmap='coolwarm', center=0, linewidths=0.3,
                xticklabels=True, yticklabels=True, ax=ax, annot=False)
    ax.set_title('Correlation Heatmap (top 30 continuous features)')
    plt.tight_layout()
    plt.savefig(eda_dir / 'correlation_heatmap.png', dpi=100)
    plt.close()

    # 3. Feature-target correlation bar chart
    target_corr = num_df.corr()['diabetes'].drop('diabetes').abs().sort_values(ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(10, 7))
    target_corr.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Feature-Target Correlation (top 25, absolute Pearson)')
    ax.set_xlabel('|Correlation|')
    plt.tight_layout()
    plt.savefig(eda_dir / 'feature_target_correlation.png', dpi=100)
    plt.close()

    # 4. Boxplots for top clinical features
    key_features = [f for f in ['Ave_SBP', 'Ave_DBP', 'bmi', 'waist_hip_ratio',
                                 'chol', 'tri', 'hdl', 'ldl', 'hemoglobin']
                    if f in df.columns]
    if key_features:
        n_cols = 3
        n_rows = (len(key_features) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten()
        for i, feat in enumerate(key_features):
            df.boxplot(column=feat, by='diabetes', ax=axes[i])
            axes[i].set_title(feat)
            axes[i].set_xlabel('Class (0=Normal, 1=At-Risk)')
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle('Feature Distributions by Class')
        plt.tight_layout()
        plt.savefig(eda_dir / 'key_feature_boxplots.png', dpi=100)
        plt.close()

    print(f"  Saved EDA plots to: {eda_dir}")


# =============================================================================
# STEP 8 – TRAIN / VAL / TEST SPLIT
# =============================================================================

def split_data(df, config):
    print("\n" + "="*70)
    print("TRAIN / VAL / TEST SPLIT")
    print("="*70)

    y = df['diabetes']
    X = df.drop(columns=['diabetes'])

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.test_size,
        random_state=config.random_state, stratify=y)

    val_ratio = config.val_size / (1 - config.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio,
        random_state=config.random_state, stratify=y_temp)

    print(f"  Train : {X_train.shape}  | Class 1: {y_train.mean()*100:.1f}%")
    print(f"  Val   : {X_val.shape}   | Class 1: {y_val.mean()*100:.1f}%")
    print(f"  Test  : {X_test.shape}  | Class 1: {y_test.mean()*100:.1f}%")
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# STEP 9 – IDENTIFY FEATURE TYPES
# =============================================================================

def identify_feature_types(X_train, config):
    all_cols = list(X_train.columns)
    cat_cols  = [c for c in config.CATEGORICAL_FEATURES if c in all_cols]
    cont_cols = [c for c in all_cols if c not in cat_cols]
    print(f"\n  Categorical features : {len(cat_cols)}")
    print(f"  Continuous features  : {len(cont_cols)}")
    return cat_cols, cont_cols


# =============================================================================
# STEP 10 – KNN NEIGHBOR OPTIMIZATION
# =============================================================================

def optimize_knn_neighbors(X_train, cont_cols, config):
    print("\n" + "="*70)
    print("KNN IMPUTATION OPTIMIZATION")
    print("="*70)

    if not config.optimize_knn:
        print(f"  Using default k={config.knn_neighbors}")
        return config.knn_neighbors

    X_num = X_train[cont_cols].copy()
    missingness = X_num.isnull().mean()
    low_miss = missingness[missingness < 0.3].index.tolist()

    if len(low_miss) < 5:
        print(f"  Too few low-missingness features; using default k={config.knn_neighbors}")
        return config.knn_neighbors

    X_sub = X_num[low_miss].dropna()
    if len(X_sub) > 5000:
        X_sub = X_sub.sample(n=5000, random_state=config.random_state)

    X_orig   = X_sub.copy()
    X_masked = X_sub.copy()
    rng = np.random.RandomState(config.random_state)
    for col in X_masked.columns:
        idx = rng.choice(len(X_masked), size=int(len(X_masked) * 0.1), replace=False)
        X_masked.iloc[idx, X_masked.columns.get_loc(col)] = np.nan

    best_k, best_rmse = config.knn_neighbors, np.inf
    for k in config.knn_candidates:
        imp = KNNImputer(n_neighbors=k)
        X_imp = imp.fit_transform(X_masked)
        rmse = np.sqrt(mean_squared_error(X_orig.values.flatten(), X_imp.flatten()))
        print(f"    k={k}: RMSE={rmse:.4f}")
        if rmse < best_rmse:
            best_rmse, best_k = rmse, k

    print(f"  Optimal k={best_k} (RMSE={best_rmse:.4f})")
    return best_k


# =============================================================================
# STEP 11 – IMPUTATION
# =============================================================================

def impute_missing(X_train, X_val, X_test, cat_cols, cont_cols, optimal_k, config):
    print("\n" + "="*70)
    print("MISSING VALUE IMPUTATION")
    print("="*70)

    # --- Categorical: mode imputation (fit on train) ---
    cat_modes = {}
    for col in cat_cols:
        if col in X_train.columns:
            mode_val = X_train[col].mode()
            cat_modes[col] = mode_val[0] if len(mode_val) > 0 else 0
            X_train[col] = X_train[col].fillna(cat_modes[col])
            X_val[col]   = X_val[col].fillna(cat_modes[col])
            X_test[col]  = X_test[col].fillna(cat_modes[col])

    # --- Continuous: KNN imputation (fit on train) ---
    valid_cont = [c for c in cont_cols if c in X_train.columns and not X_train[c].isna().all()]
    imputer = KNNImputer(n_neighbors=optimal_k)
    X_train_cont = imputer.fit_transform(X_train[valid_cont])
    X_val_cont   = imputer.transform(X_val[valid_cont])
    X_test_cont  = imputer.transform(X_test[valid_cont])

    for i, col in enumerate(valid_cont):
        X_train[col] = X_train_cont[:, i]
        X_val[col]   = X_val_cont[:, i]
        X_test[col]  = X_test_cont[:, i]

    print(f"  Mode imputed  : {len(cat_cols)} categorical features")
    print(f"  KNN imputed   : {len(valid_cont)} continuous features (k={optimal_k})")
    return X_train, X_val, X_test, imputer, cat_modes


# =============================================================================
# STEP 12 – OUTLIER CAPPING (continuous only)
# =============================================================================

def handle_outliers(X_train, X_val, X_test, cont_cols, config):
    print("\n" + "="*70)
    print("OUTLIER CAPPING (IQR, continuous only)")
    print("="*70)

    if not config.handle_outliers:
        print("  Disabled")
        return X_train, X_val, X_test

    bounds = {}
    n_capped = 0
    valid_cont = [c for c in cont_cols if c in X_train.columns]
    for col in valid_cont:
        Q1, Q3 = X_train[col].quantile(0.25), X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - config.iqr_multiplier * IQR, Q3 + config.iqr_multiplier * IQR
        bounds[col] = (lo, hi)
        capped = int(((X_train[col] < lo) | (X_train[col] > hi)).sum())
        if capped > 0:
            n_capped += capped
            for ds in [X_train, X_val, X_test]:
                ds[col] = ds[col].clip(lower=lo, upper=hi)

    print(f"  Capped {n_capped:,} outlier values across continuous features")
    return X_train, X_val, X_test


# =============================================================================
# STEP 13 – STANDARD SCALING (continuous only)
# =============================================================================

def scale_features(X_train, X_val, X_test, cont_cols, config):
    print("\n" + "="*70)
    print("STANDARD SCALING (continuous features only)")
    print("="*70)

    valid_cont = [c for c in cont_cols if c in X_train.columns]
    scaler = StandardScaler()
    X_train[valid_cont] = scaler.fit_transform(X_train[valid_cont])
    X_val[valid_cont]   = scaler.transform(X_val[valid_cont])
    X_test[valid_cont]  = scaler.transform(X_test[valid_cont])

    print(f"  Scaled {len(valid_cont)} continuous features (fit on train only)")
    print(f"  Categorical features left as-is: {[c for c in X_train.columns if c in cont_cols[:5]][:3]}...")
    return X_train, X_val, X_test, scaler


# =============================================================================
# STEP 14 – CORRELATION-BASED FEATURE DROP
# =============================================================================

def drop_correlated_features(X_train, X_val, X_test, config):
    print("\n" + "="*70)
    print(f"DROPPING HIGHLY CORRELATED FEATURES (threshold={config.correlation_threshold})")
    print("="*70)

    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    to_drop = [col for col in upper.columns if (upper[col] > config.correlation_threshold).any()]

    X_train = X_train.drop(columns=to_drop, errors='ignore')
    X_val   = X_val.drop(columns=to_drop, errors='ignore')
    X_test  = X_test.drop(columns=to_drop, errors='ignore')

    print(f"  Dropped {len(to_drop)} features")
    if to_drop:
        print(f"  Examples: {to_drop[:10]}")
    print(f"  Remaining features: {X_train.shape[1]}")
    return X_train, X_val, X_test, to_drop


# =============================================================================
# STEP 15 – RESAMPLING (train only)
# =============================================================================

def resample_data(X_train, y_train, config):
    print("\n" + "="*70)
    print(f"RESAMPLING: {config.resampling_method.upper()}")
    print("="*70)

    feature_names = list(X_train.columns)
    X_np = X_train.values
    y_np = y_train.values

    if config.resampling_method == 'smotetomek':
        n1 = (y_np == 1).sum()
        smt = SMOTETomek(sampling_strategy=config.smote_sampling_strategy,
                         smote=SMOTE(k_neighbors=config.smote_k_neighbors,
                                     random_state=config.random_state),
                         random_state=config.random_state)
        X_np, y_np = smt.fit_resample(X_np, y_np)
        print(f"  SMOTETomek: {n1:,} -> {(y_np==1).sum():,} minority samples (boundary-cleaned)")

    elif config.resampling_method == 'smote':
        n0 = (y_np == 0).sum()
        n1 = (y_np == 1).sum()
        current_ratio = n1 / n0
        if current_ratio < config.smote_sampling_strategy:
            sm = SMOTE(sampling_strategy=config.smote_sampling_strategy,
                       k_neighbors=config.smote_k_neighbors,
                       random_state=config.random_state)
            X_np, y_np = sm.fit_resample(X_np, y_np)
            print(f"  SMOTE: {n1:,} -> {(y_np==1).sum():,} minority samples")
        else:
            print(f"  SMOTE skipped (ratio {current_ratio:.2f} >= target)")

    elif config.resampling_method == 'undersample':
        rus = RandomUnderSampler(sampling_strategy=config.undersample_strategy,
                                 random_state=config.random_state)
        X_np, y_np = rus.fit_resample(X_np, y_np)
        print(f"  Undersampled to: Class 0={( y_np==0).sum():,}, Class 1={(y_np==1).sum():,}")

    else:
        print("  No resampling applied")

    return X_np, y_np, feature_names


# =============================================================================
# STEP 16 – SAVE OUTPUTS
# =============================================================================

def save_outputs(X_train_np, y_train_np, X_val, y_val,
                 X_test, y_test, feature_names,
                 imputer, cat_modes, scaler, corr_dropped,
                 config):
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # NumPy arrays
    np.save(config.output_dir / 'X_train.npy', X_train_np)
    np.save(config.output_dir / 'y_train.npy', y_train_np)
    np.save(config.output_dir / 'X_val.npy',   X_val.values)
    np.save(config.output_dir / 'y_val.npy',   y_val.values)
    np.save(config.output_dir / 'X_test.npy',  X_test.values)
    np.save(config.output_dir / 'y_test.npy',  y_test.values)

    # Scaled CSVs (model-ready, scaled)
    pd.DataFrame(X_train_np, columns=feature_names).assign(diabetes=y_train_np).to_csv(
        config.output_dir / 'preprocessed_train.csv', index=False)
    X_val.assign(diabetes=y_val.values).to_csv(
        config.output_dir / 'preprocessed_val.csv',  index=False)
    X_test.assign(diabetes=y_test.values).to_csv(
        config.output_dir / 'preprocessed_test.csv', index=False)

    # Feature list
    feat_df = pd.DataFrame({'feature_name': feature_names,
                             'feature_index': range(len(feature_names))})
    feat_df.to_csv(config.output_dir / 'feature_list.csv', index=False)

    # Preprocessing artifacts
    info = {
        'feature_names':        feature_names,
        'n_features':           len(feature_names),
        'scaler':               scaler,
        'imputer':              imputer,
        'cat_modes':            cat_modes,
        'corr_dropped':         corr_dropped,
        'resampling_method':    config.resampling_method,
        'fbs_threshold':        config.fbs_threshold,
        'train_shape':          X_train_np.shape,
        'val_shape':            X_val.shape,
        'test_shape':           X_test.shape,
    }
    joblib.dump(info, config.output_dir / 'preprocessing_info.pkl')

    print(f"  Train  : {X_train_np.shape}")
    print(f"  Val    : {X_val.shape}")
    print(f"  Test   : {X_test.shape}")
    print(f"  Features saved: {len(feature_names)}")
    print(f"  Output dir: {config.output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = Config()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load & Merge
    df = load_and_merge(config)

    # Drop metadata columns
    df = drop_metadata_columns(df)

    # Recode special missing codes to NaN
    df = recode_special_missing(df, config)

    # Filter population (children, pregnant, lactating)
    df = filter_population(df, config)

    # Create binary target (fbs >= threshold)
    df = create_target(df, config)

    # Minimal feature engineering
    df = engineer_features(df)

    # Export raw unscaled CSV for EDA
    export_raw_csv(df, config)

    # EDA visualizations
    generate_eda_visualizations(df, config)

    # Train / Val / Test split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, config)

    # Identify categorical vs continuous features
    cat_cols, cont_cols = identify_feature_types(X_train, config)

    # Optimize KNN k
    optimal_k = optimize_knn_neighbors(X_train, cont_cols, config)

    # Impute missing values
    X_train, X_val, X_test, imputer, cat_modes = impute_missing(
        X_train, X_val, X_test, cat_cols, cont_cols, optimal_k, config)

    # Outlier capping (continuous only)
    X_train, X_val, X_test = handle_outliers(X_train, X_val, X_test, cont_cols, config)

    # StandardScaler (continuous only)
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test, cont_cols, config)

    # Drop highly correlated features (computed on train)
    X_train, X_val, X_test, corr_dropped = drop_correlated_features(X_train, X_val, X_test, config)

    # Resampling on train only
    X_train_np, y_train_np, feature_names = resample_data(X_train, y_train, config)

    # Save all outputs
    save_outputs(X_train_np, y_train_np, X_val, y_val,
                 X_test, y_test, feature_names,
                 imputer, cat_modes, scaler, corr_dropped, config)

    print("\n" + "="*70)
    print("PREPROCESSING V7 COMPLETE")
    print("="*70)
    print(f"  Final features : {len(feature_names)}")
    print(f"  Train samples  : {X_train_np.shape[0]:,} (after resampling)")
    print(f"  Output dir     : {config.output_dir}")


if __name__ == '__main__':
    main()
