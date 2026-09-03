"""
Main Preprocessing Pipeline (2018-2021 ENNS Dataset)

Key design decisions:
  1. No population filter  (all age groups included — children, pregnant, lactating)
  2. Selective feature engineering (bmi, tri_hdl_ratio, ldl_hdl_ratio, bmi_tri_interact)
  3. Correlation threshold = 0.80
  4. SMOTE applied to the FULL preprocessed dataset BEFORE re-splitting 70/15/15
  5. Socio dataset integrated: age and sex included as features
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Import all utility functions from the standard V7 pipeline
from preprocessing_pipeline_v7 import (
    load_and_merge,
    drop_metadata_columns,
    recode_special_missing,
    create_target,
    export_raw_csv,
    generate_eda_visualizations,
    split_data,
    identify_feature_types,
    optimize_knn_neighbors,
    impute_missing,
    handle_outliers,
    scale_features,
    drop_correlated_features,
    Config as ConfigV7,
)


# =============================================================================
# SELECTIVE FEATURE ENGINEERING  (diabetes risk scope)
# =============================================================================

def engineer_features_selective(df):
    """
    Compute only the engineered features clinically relevant to diabetes risk:
      - bmi            : weight / (height/100)²    drops weight & height
      - tri_hdl_ratio  : tri / (hdl + eps)          insulin resistance proxy
      - ldl_hdl_ratio  : ldl / hdl                  dyslipidaemia pattern
      - bmi_tri_interact: bmi * tri / 100           obesity × triglyceride risk

    tri, hdl, ldl are KEPT alongside their derived ratios.
    The correlation filter (0.80) will remove any that become redundant.
    """
    print("\n" + "="*70)
    print("SELECTIVE FEATURE ENGINEERING (diabetes risk scope)")
    print("="*70)
    created, dropped = [], []
    eps = 1e-6

    if 'weight' in df.columns and 'height' in df.columns:
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        df = df.drop(columns=['weight', 'height'])
        created.append('bmi')
        dropped.extend(['weight', 'height'])

    if 'tri' in df.columns and 'hdl' in df.columns:
        df['tri_hdl_ratio'] = df['tri'] / (df['hdl'] + eps)
        created.append('tri_hdl_ratio')

    if 'ldl' in df.columns and 'hdl' in df.columns:
        df['ldl_hdl_ratio'] = df['ldl'] / df['hdl']
        created.append('ldl_hdl_ratio')

    if 'bmi' in df.columns and 'tri' in df.columns:
        df['bmi_tri_interact'] = df['bmi'] * df['tri'] / 100
        created.append('bmi_tri_interact')

    print(f"  Created : {created}")
    print(f"  Dropped : {dropped}  (replaced by bmi)")
    print(f"  Retained: tri, hdl, ldl  (kept alongside derived ratios)")
    print(f"  Shape   : {df.shape}")
    return df


# =============================================================================
# CONFIG
# =============================================================================

class Config(ConfigV7):
    output_dir   = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\MAIN_PREPROCESS_OUTPUT")
    socio_path   = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\DATASETS\2018-2021\SOCIO DATASET\data-set_socio.csv")

    # All age groups included — no population filtering
    exclude_anthro_groups = []

    # Keep anthro_group as a feature (not dropped); add sex from socio dataset
    CATEGORICAL_FEATURES = ConfigV7.CATEGORICAL_FEATURES + ['anthro_group', 'sex']

    # No feature engineering (controlled via the main() flow)

    # Explicitly drop before correlation filter
    # chol (total cholesterol) dropped in favour of ldl (LDL — bad cholesterol);
    # ldl + hdl + tri give the full standard lipid panel
    drop_before_corr = ['chol']

    # More aggressive correlation pruning (align with 2013)
    correlation_threshold = 0.80

    # SMOTE settings (applied to full dataset before split)
    resampling_method       = 'smote'
    smote_sampling_strategy = 1.0   # full balance
    smote_k_neighbors       = 5


# =============================================================================
# SOCIO DATASET MERGE  (adds age, sex, regcode, provhuc)
# =============================================================================

REGION_MAP = {
    1: 'Region I', 2: 'Region II', 3: 'Region III',
    41: 'Region IV-A', 42: 'Region IV-B', 5: 'Region V',
    6: 'Region VI', 7: 'Region VII', 8: 'Region VIII',
    9: 'Region IX', 10: 'Region X', 11: 'Region XI',
    12: 'Region XII', 13: 'NCR', 14: 'CAR', 15: 'BARMM', 16: 'CARAGA',
}


def merge_socio(df, config):
    print("\n" + "="*70)
    print("MERGING SOCIO-ECONOMIC DATASET  (age, sex, region)")
    print("="*70)

    socio = pd.read_csv(config.socio_path, low_memory=False)
    print(f"  Socio raw shape : {socio.shape}")

    # Keep only merge keys + age/sex  (regcode/provhuc already exist in original data)
    keep = ['hhnum', 'member_code', 'age', 'sex']
    socio = socio[keep].copy()

    # Recode sex: 1=Male, 2=Female  (float so numpy handles it correctly)
    socio['sex'] = pd.to_numeric(socio['sex'], errors='coerce').astype(float)

    n0 = len(df)
    df = df.merge(socio, on=['hhnum', 'member_code'], how='left')
    match_rate = df['age'].notna().mean() * 100
    print(f"  Clinical rows   : {n0:,}")
    print(f"  After merge     : {df.shape}  (match rate: {match_rate:.1f}%)")
    print(f"  age  null : {df['age'].isna().sum():,}  "
          f"sex  null : {df['sex'].isna().sum():,}")
    return df


# =============================================================================
# DESCRIPTIVE STATISTICS  (EDA table + chart)
# =============================================================================

def generate_descriptive_stats(df, config):
    """Compute and save descriptive statistics for all features by diabetes class."""
    print("\n" + "="*70)
    print("GENERATING DESCRIPTIVE STATISTICS")
    print("="*70)
    eda_dir = config.output_dir / 'EDA'
    eda_dir.mkdir(parents=True, exist_ok=True)

    feat_cols = [c for c in df.columns if c not in ('diabetes', 'regcode', 'provhuc')]
    cat_feats  = [c for c in config.CATEGORICAL_FEATURES if c in feat_cols]
    cont_feats = [c for c in feat_cols if c not in cat_feats]

    rows = []
    for col in cont_feats + cat_feats:
        if col not in df.columns:
            continue
        is_cat = col in cat_feats
        for cls in [-1, 0, 1]:      # -1 = overall
            sub = df if cls == -1 else df[df['diabetes'] == cls]
            n   = sub[col].notna().sum()
            if is_cat:
                mode_val = sub[col].mode(dropna=True)
                rows.append({
                    'feature':   col,
                    'class':     'All' if cls == -1 else ('Normal' if cls == 0 else 'At-Risk'),
                    'type':      'categorical',
                    'n':         n,
                    'mean':      '',
                    'std':       '',
                    'median':    '',
                    'q1':        '',
                    'q3':        '',
                    'min':       sub[col].min(),
                    'max':       sub[col].max(),
                    'mode':      mode_val.iloc[0] if len(mode_val) else '',
                    'missing_pct': f"{df[col].isna().mean()*100:.1f}%" if cls == -1 else '',
                })
            else:
                s = sub[col].dropna()
                rows.append({
                    'feature':   col,
                    'class':     'All' if cls == -1 else ('Normal' if cls == 0 else 'At-Risk'),
                    'type':      'continuous',
                    'n':         n,
                    'mean':      round(s.mean(), 3)  if len(s) else '',
                    'std':       round(s.std(),  3)  if len(s) else '',
                    'median':    round(s.median(), 3) if len(s) else '',
                    'q1':        round(s.quantile(0.25), 3) if len(s) else '',
                    'q3':        round(s.quantile(0.75), 3) if len(s) else '',
                    'min':       round(s.min(), 3) if len(s) else '',
                    'max':       round(s.max(), 3) if len(s) else '',
                    'mode':      '',
                    'missing_pct': f"{df[col].isna().mean()*100:.1f}%" if cls == -1 else '',
                })

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(eda_dir / 'descriptive_statistics.csv', index=False)
    print(f"  Saved: descriptive_statistics.csv  ({len(cont_feats)} continuous, {len(cat_feats)} categorical)")

    # Summary chart: mean +/- 1 SD for top continuous features by class
    top_cont = cont_feats[:min(12, len(cont_feats))]
    sub_df   = stats_df[stats_df['feature'].isin(top_cont) & (stats_df['type'] == 'continuous')]
    fig, ax  = plt.subplots(figsize=(14, 6))
    x        = np.arange(len(top_cont))
    w        = 0.35
    for offset, cls_label, color in [(-w/2, 'Normal', '#2980b9'), (w/2, 'At-Risk', '#e74c3c')]:
        subset = sub_df[sub_df['class'] == cls_label].set_index('feature').reindex(top_cont)
        means  = pd.to_numeric(subset['mean'],  errors='coerce').values
        stds   = pd.to_numeric(subset['std'],   errors='coerce').values
        ax.bar(x + offset, means, w, yerr=stds, label=cls_label, color=color,
               alpha=0.8, capsize=4, error_kw={'linewidth': 1})
    ax.set_xticks(x)
    ax.set_xticklabels(top_cont, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Mean +/- 1 SD')
    ax.set_title('Feature Means by Diabetes Class (Normal vs At-Risk)\n2018-2021 ENNS',
                 fontweight='bold', fontsize=11)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(eda_dir / 'descriptive_stats_by_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: descriptive_stats_by_class.png")

    return stats_df


# =============================================================================
# DEMOGRAPHIC / REGIONAL ANALYSIS  (EDA charts)
# =============================================================================

def generate_demographic_analysis(df, config):
    """Charts: diabetes by region, by age group, by sex.  df must contain
    'diabetes', 'regcode', 'age', 'sex' columns before metadata drop."""
    print("\n" + "="*70)
    print("GENERATING DEMOGRAPHIC & REGIONAL ANALYSIS")
    print("="*70)
    eda_dir = config.output_dir / 'EDA'
    eda_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Diabetes prevalence by region ─────────────────────────────────────
    if 'regcode' in df.columns:
        df['region_label'] = df['regcode'].map(REGION_MAP).fillna('Unknown')
        reg = (df.groupby('region_label')['diabetes']
               .agg(['mean', 'sum', 'count'])
               .rename(columns={'mean': 'prevalence', 'sum': 'n_atrisk', 'count': 'n_total'})
               .sort_values('prevalence', ascending=True))
        reg['prevalence_pct'] = reg['prevalence'] * 100

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(reg.index, reg['prevalence_pct'],
                       color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(reg))),  # type: ignore
                       edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Diabetes At-Risk Prevalence (%)', fontsize=10)
        ax.set_title('Diabetes At-Risk Prevalence by Region\n2018-2021 ENNS',
                     fontsize=12, fontweight='bold')
        for bar, pct, n in zip(bars, reg['prevalence_pct'], reg['n_total']):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{pct:.1f}%  (n={n:,})', va='center', fontsize=8)
        ax.set_xlim(0, reg['prevalence_pct'].max() * 1.35)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(eda_dir / 'diabetes_by_region.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: diabetes_by_region.png")

        # Save regional summary CSV
        reg.to_csv(eda_dir / 'diabetes_by_region.csv')
        print(f"  Saved: diabetes_by_region.csv")

    # ── 2. Diabetes prevalence by age group (10-year bins) ───────────────────
    if 'age' in df.columns:
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 120],
            labels=['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'],
            right=False
        )
        age_g = (df.groupby('age_group', observed=True)['diabetes']
                 .agg(['mean', 'sum', 'count'])
                 .rename(columns={'mean': 'prevalence', 'sum': 'n_atrisk', 'count': 'n_total'}))
        age_g['prevalence_pct'] = age_g['prevalence'] * 100

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        x   = np.arange(len(age_g))
        ax1.bar(x, age_g['n_total'], color='#BDC3C7', alpha=0.7, label='Total (N)')
        ax1.bar(x, age_g['n_atrisk'], color='#e74c3c', alpha=0.8, label='At-Risk (N)')
        ax2.plot(x, age_g['prevalence_pct'], 'o-', color='#2c3e50',
                 linewidth=2.5, markersize=7, label='Prevalence %')
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax1.set_xticks(x)
        ax1.set_xticklabels(age_g.index, fontsize=9)
        ax1.set_xlabel('Age Group', fontsize=10)
        ax1.set_ylabel('Count', fontsize=10)
        ax2.set_ylabel('Prevalence (%)', fontsize=10)
        ax1.set_title('Diabetes At-Risk by Age Group\n2018-2021 ENNS',
                      fontsize=12, fontweight='bold')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(eda_dir / 'diabetes_by_age_group.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: diabetes_by_age_group.png")

        age_g.to_csv(eda_dir / 'diabetes_by_age_group.csv')
        print(f"  Saved: diabetes_by_age_group.csv")

    # ── 3. Diabetes prevalence by sex ─────────────────────────────────────────
    if 'sex' in df.columns:
        sex_map = {1: 'Male', 2: 'Female'}
        df['sex_label'] = df['sex'].map(sex_map).fillna('Unknown')
        sex_g = (df.groupby('sex_label')['diabetes']
                 .agg(['mean', 'sum', 'count'])
                 .rename(columns={'mean': 'prevalence', 'sum': 'n_atrisk', 'count': 'n_total'}))
        sex_g['prevalence_pct'] = sex_g['prevalence'] * 100

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        colors = ['#2980b9', '#e74c3c']

        axes[0].bar(sex_g.index, sex_g['prevalence_pct'], color=colors, alpha=0.85, edgecolor='white')
        axes[0].set_ylabel('Prevalence (%)', fontsize=10)
        axes[0].set_title('At-Risk Prevalence by Sex', fontsize=11, fontweight='bold')
        axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
        for i, (idx, row) in enumerate(sex_g.iterrows()):
            axes[0].text(i, row['prevalence_pct'] + 0.3, f"{row['prevalence_pct']:.1f}%",
                         ha='center', fontsize=10, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        axes[1].pie(sex_g['n_atrisk'], labels=sex_g.index, colors=colors,
                    autopct='%1.1f%%', startangle=90, pctdistance=0.8)
        axes[1].set_title('At-Risk Cases by Sex', fontsize=11, fontweight='bold')

        fig.suptitle('Diabetes At-Risk by Sex\n2018-2021 ENNS',
                     fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(eda_dir / 'diabetes_by_sex.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: diabetes_by_sex.png")

        sex_g.to_csv(eda_dir / 'diabetes_by_sex.csv')
        print(f"  Saved: diabetes_by_sex.csv")

    # ── 4. Age distribution histogram (all vs at-risk) ────────────────────────
    if 'age' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        df[df['diabetes'] == 0]['age'].dropna().plot(
            kind='hist', bins=40, ax=ax, alpha=0.6, color='#2980b9',
            label='Normal', density=True)
        df[df['diabetes'] == 1]['age'].dropna().plot(
            kind='hist', bins=40, ax=ax, alpha=0.6, color='#e74c3c',
            label='At-Risk', density=True)
        ax.set_xlabel('Age (years)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title('Age Distribution: Normal vs At-Risk\n2018-2021 ENNS',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(eda_dir / 'age_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: age_distribution.png")

    print(f"  All demographic analysis saved to: {eda_dir}")


# =============================================================================
# SAVE OUTPUTS  (numpy-array version — val/test are post-SMOTE numpy arrays)
# =============================================================================

def save_outputs(X_train_np, y_train_np, X_val_np, y_val_np,
                 X_test_np, y_test_np, feature_names,
                 imputer, cat_modes, scaler, corr_dropped, config):
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    np.save(config.output_dir / 'X_train.npy', X_train_np)
    np.save(config.output_dir / 'y_train.npy', y_train_np)
    np.save(config.output_dir / 'X_val.npy',   X_val_np)
    np.save(config.output_dir / 'y_val.npy',   y_val_np)
    np.save(config.output_dir / 'X_test.npy',  X_test_np)
    np.save(config.output_dir / 'y_test.npy',  y_test_np)

    pd.DataFrame(X_train_np, columns=feature_names).assign(
        diabetes=y_train_np).to_csv(
        config.output_dir / 'preprocessed_train.csv', index=False)
    pd.DataFrame(X_val_np, columns=feature_names).assign(
        diabetes=y_val_np).to_csv(
        config.output_dir / 'preprocessed_val.csv', index=False)
    pd.DataFrame(X_test_np, columns=feature_names).assign(
        diabetes=y_test_np).to_csv(
        config.output_dir / 'preprocessed_test.csv', index=False)

    pd.DataFrame({'feature_name':  feature_names,
                  'feature_index': range(len(feature_names))}).to_csv(
        config.output_dir / 'feature_list.csv', index=False)

    joblib.dump({
        'feature_names':     feature_names,
        'n_features':        len(feature_names),
        'scaler':            scaler,
        'imputer':           imputer,
        'cat_modes':         cat_modes,
        'corr_dropped':      corr_dropped,
        'resampling_method': config.resampling_method,
        'fbs_threshold':     config.fbs_threshold,
        'train_shape':       X_train_np.shape,
        'val_shape':         X_val_np.shape,
        'test_shape':        X_test_np.shape,
    }, config.output_dir / 'preprocessing_info.pkl')

    print(f"  Train  : {X_train_np.shape}  Class 1: {y_train_np.mean()*100:.1f}%")
    print(f"  Val    : {X_val_np.shape}  Class 1: {y_val_np.mean()*100:.1f}%")
    print(f"  Test   : {X_test_np.shape}  Class 1: {y_test_np.mean()*100:.1f}%")
    print(f"  Features ({len(feature_names)}): {feature_names[:8]}...")
    print(f"  Output dir: {config.output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import time
    start = time.time()

    config = Config()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("MAIN PREPROCESSING PIPELINE  (2018-2021 ENNS)")
    print("SMOTE before split | no population filter | selective feature engineering")
    print("Correlation threshold: 0.80  |  SMOTE strategy: 1.0 (full balance)")
    print("Socio dataset: age + sex included as features")
    print("="*70)

    # ── Steps 1-5 ────────────────────────────────────────────────────────────────────
    df = load_and_merge(config)

    # Save regcode/provhuc from original data BEFORE merge/drop — preserves index
    # for later alignment even after create_target drops FBS-missing rows
    geo_series = df[['regcode', 'provhuc']].copy()

    df = merge_socio(df, config)          # adds: age, sex
    df = drop_metadata_columns(df)        # drops regcode, provhuc, hhnum, member_code
                                           # age and sex are retained (not in metadata_prefixes)
    df = recode_special_missing(df, config)

    # filter_population: SKIPPED — all age groups included
    df = create_target(df, config)        # drops rows with NaN FBS — index shrinks

    # Drop admin columns near-entirely NaN outside their subgroup
    df = df.drop(columns=['mos_lactation', 'mos_preg'], errors='ignore')
    print("  Dropped admin columns: mos_lactation, mos_preg (near-all-NaN)")

    df = engineer_features_selective(df)    # bmi, tri_hdl_ratio, ldl_hdl_ratio, bmi_tri_interact

    # ── EDA: re-attach geo columns using index alignment (survives FBS row drops)
    df_eda = df.copy()
    df_eda['regcode'] = geo_series.loc[df.index, 'regcode'].values
    df_eda['provhuc'] = geo_series.loc[df.index, 'provhuc'].values

    generate_demographic_analysis(df_eda, config)
    generate_descriptive_stats(df, config)      # df has age+sex but no geo cols

    export_raw_csv(df, config)
    generate_eda_visualizations(df, config)

    # ── Steps 6-13: imputation, scaling, correlation drop ──────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, config)
    cat_cols, cont_cols = identify_feature_types(X_train, config)
    optimal_k = optimize_knn_neighbors(X_train, cont_cols, config)

    X_train, X_val, X_test, imputer, cat_modes = impute_missing(
        X_train, X_val, X_test, cat_cols, cont_cols, optimal_k, config)
    X_train, X_val, X_test = handle_outliers(
        X_train, X_val, X_test, cont_cols, config)
    X_train, X_val, X_test, scaler = scale_features(
        X_train, X_val, X_test, cont_cols, config)

    # Explicitly drop features before correlation filter (e.g. chol -> keep ldl instead)
    pre_drop = getattr(config, 'drop_before_corr', [])
    if pre_drop:
        X_train = X_train.drop(columns=pre_drop, errors='ignore')
        X_val   = X_val.drop(columns=pre_drop,   errors='ignore')
        X_test  = X_test.drop(columns=pre_drop,  errors='ignore')
        print(f"  [Config] Pre-dropped before corr filter: {pre_drop}")

    X_train, X_val, X_test, corr_dropped = drop_correlated_features(
        X_train, X_val, X_test, config)

    feature_names = list(X_train.columns)

    # ── Reconstitute full dataset, apply SMOTE, re-split 70/15/15 ──────────────
    X_all = np.vstack([X_train.values, X_val.values, X_test.values])
    y_all = np.concatenate([y_train.values, y_val.values, y_test.values])
    print(f"\n  Reconstituted full dataset: {X_all.shape}  "
          f"Class 0: {(y_all==0).sum():,}  Class 1: {(y_all==1).sum():,}  "
          f"({y_all.mean()*100:.1f}% at-risk)")

    # Ensure float64 (Int64 / object dtypes from nullable integers break np.isnan)
    X_all = X_all.astype(float)

    # Safety: fill residual NaN with column median (numpy — never drops columns)
    n_nan = int(np.isnan(X_all).sum())
    if n_nan > 0:
        print(f"  [Safety] {n_nan} residual NaN values — filling with column median")
        col_medians = np.nanmedian(X_all, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        nan_r, nan_c = np.where(np.isnan(X_all))
        X_all[nan_r, nan_c] = col_medians[nan_c]

    print("\n" + "="*70)
    print("RESAMPLING  (SMOTE — FULL DATASET BEFORE SPLIT)")
    print("="*70)
    sm = SMOTE(
        sampling_strategy=config.smote_sampling_strategy,
        k_neighbors=config.smote_k_neighbors,
        random_state=config.random_state)
    X_all_np, y_all_np = sm.fit_resample(X_all, y_all)
    print(f"  Before: {len(y_all):,}  (Class 1: {(y_all==1).sum():,})")
    print(f"  After : {len(y_all_np):,}  (Class 1: {(y_all_np==1).sum():,}  "
          f"| {y_all_np.mean()*100:.1f}%)")

    print("\n" + "="*70)
    print("RE-SPLITTING SMOTE-BALANCED DATASET  (70 / 15 / 15)")
    print("="*70)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_all_np, y_all_np, test_size=0.30,
        random_state=config.random_state, stratify=y_all_np)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50,
        random_state=config.random_state, stratify=y_tmp)
    print(f"  Train : {X_tr.shape}  Class 1: {y_tr.mean()*100:.1f}%")
    print(f"  Val   : {X_va.shape}  Class 1: {y_va.mean()*100:.1f}%")
    print(f"  Test  : {X_te.shape}  Class 1: {y_te.mean()*100:.1f}%")

    save_outputs(X_tr, y_tr, X_va, y_va, X_te, y_te, feature_names,
                 imputer, cat_modes, scaler, corr_dropped, config)

    elapsed = (time.time() - start) / 60
    print("\n" + "="*70)
    print("MAIN PREPROCESSING COMPLETE")
    print("="*70)
    print(f"  Final features : {len(feature_names)}")
    print(f"  Train samples  : {X_tr.shape[0]:,}  (post-SMOTE)")
    print(f"  Output dir     : {config.output_dir}")
    print(f"  [TIME] {elapsed:.1f} minutes")


if __name__ == '__main__':
    main()
