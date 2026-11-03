"""
Additional V2 EDA (read-only on V2 outputs).

Reads MAIN_PREPROCESS_OUTPUT/raw_merged.csv (raw, unscaled, pre-split) and writes NEW files
to MAIN_PREPROCESS_OUTPUT/EDA with the prefix 'eda2_'. Does not rerun preprocessing and does not
touch V2 arrays, models, or existing EDA files.

Added analyses:
  1. Missing-value and sentinel-value audit (table + chart)
  2. Density distributions of every continuous feature by class
  3. Boxplots of every continuous feature by class
  4. Annotated correlation matrix plus signed feature-target correlation

Usage:  python main_eda_extra.py
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\MAIN_PREPROCESS_OUTPUT")
EDA_DIR = BASE / 'EDA'
TARGET = 'diabetes'
CAT_COLS = ['currentsmoking', 'ever_smk', 'alcohol', 'con_alcohol', 'drnk_30days',
            'smoke_status', 'binge_drink', 'anthro_group', 'sex']
SENTINEL_MIN = 99999
C0, C1 = '#2980b9', '#e74c3c'


def load():
    df = pd.read_csv(BASE / 'raw_merged.csv', low_memory=False)
    for c in df.columns:
        if c != TARGET:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    cat = [c for c in CAT_COLS if c in df.columns]
    num = [c for c in df.columns if c not in cat + [TARGET]]
    return df, num, cat


def missing_and_sentinel_audit(df, num, cat):
    rows = []
    for c in num + cat:
        s = df[c]
        rows.append({
            'feature': c,
            'type': 'categorical' if c in cat else 'continuous',
            'missing_n': int(s.isna().sum()),
            'missing_pct': round(s.isna().mean() * 100, 2),
            'sentinel_like_n': int((s >= SENTINEL_MIN).sum()),
            'at_risk_pct_when_missing': round(df.loc[s.isna(), TARGET].mean() * 100, 2) if s.isna().any() else np.nan,
            'at_risk_pct_when_present': round(df.loc[s.notna(), TARGET].mean() * 100, 2),
        })
    tbl = pd.DataFrame(rows).sort_values('missing_pct', ascending=False)
    tbl.to_csv(EDA_DIR / 'eda2_missing_sentinel_audit.csv', index=False)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(tbl))))
    colors = [C1 if v > 40 else '#f39c12' if v > 10 else C0 for v in tbl['missing_pct']]
    ax.barh(tbl['feature'], tbl['missing_pct'], color=colors, alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel('Missing (%)')
    ax.set_title(f'Missing Values per Feature (n={len(df):,})\nred > 40%, orange > 10%',
                 fontweight='bold', fontsize=11)
    for y, v in enumerate(tbl['missing_pct']):
        ax.text(v + 0.5, y, f'{v:.1f}%', va='center', fontsize=7)
    ax.set_xlim(0, max(tbl['missing_pct'].max() * 1.15, 5))
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'eda2_missing_values.png', dpi=150, bbox_inches='tight')
    plt.close()

    flagged = tbl[tbl['sentinel_like_n'] > 0]
    if len(flagged):
        print('  Features containing sentinel-like values (>= %d):' % SENTINEL_MIN)
        print(flagged[['feature', 'sentinel_like_n']].to_string(index=False))
    print('  Saved: eda2_missing_sentinel_audit.csv, eda2_missing_values.png')


def _grid(n, n_cols=4):
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    return fig, np.atleast_1d(axes).flatten()


def _trim(s, lo=0.5, hi=99.5):
    """Clip display range to central percentiles so sentinels/extremes do not flatten the plot."""
    s = s.dropna()
    if s.empty:
        return s
    a, b = np.percentile(s, [lo, hi])
    return s[(s >= a) & (s <= b)]


def distributions_by_class(df, num):
    fig, axes = _grid(len(num))
    for ax, col in zip(axes, num):
        for cls, color, label in [(0, C0, 'Normal'), (1, C1, 'At-Risk')]:
            s = _trim(df.loc[df[TARGET] == cls, col])
            if s.nunique() > 1:
                ax.hist(s, bins=30, alpha=0.6, color=color, label=label, density=True)
        ax.set_title(col, fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in axes[len(num):]:
        ax.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C0, alpha=0.6), plt.Rectangle((0, 0), 1, 1, color=C1, alpha=0.6)]
    fig.legend(handles, ['Normal', 'At-Risk'], loc='lower right', fontsize=10, ncol=2)
    fig.suptitle('Continuous Feature Distributions by Class (0.5-99.5 percentile range)\n2018-2021 ENNS',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'eda2_feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: eda2_feature_distributions.png')


def boxplots_by_class(df, num):
    fig, axes = _grid(len(num))
    for ax, col in zip(axes, num):
        data = [df.loc[df[TARGET] == c, col].dropna() for c in (0, 1)]
        if any(len(d) == 0 for d in data):
            ax.set_visible(False)
            continue
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops=dict(color='black', linewidth=2))
        for box, color in zip(bp['boxes'], (C0, C1)):
            box.set_facecolor(color)
            box.set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Normal', 'At-Risk'], fontsize=7)
        ax.set_title(col, fontsize=8, fontweight='bold')
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(axis='y', alpha=0.3)
    for ax in axes[len(num):]:
        ax.set_visible(False)
    fig.suptitle('Boxplots by Class: Outlier Inspection (raw values)\n2018-2021 ENNS',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'eda2_boxplots_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: eda2_boxplots_all.png')


def correlation_analysis(df, num):
    clean = df[num + [TARGET]].copy()
    for c in num:
        clean.loc[clean[c] >= SENTINEL_MIN, c] = np.nan
    corr = clean.corr()
    corr.to_csv(EDA_DIR / 'eda2_correlation_matrix.csv')

    tcorr = corr[TARGET].drop(TARGET).dropna().sort_values(key=abs, ascending=False)
    tcorr.rename('pearson_r_with_target').to_csv(EDA_DIR / 'eda2_target_correlation.csv')

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1.6, 1]})
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.4,
                annot_kws={'size': 6}, ax=axes[0], cbar_kws={'shrink': 0.8})
    axes[0].set_title('Pairwise Pearson Correlation (pairwise-complete, sentinels excluded)',
                      fontweight='bold', fontsize=11)
    axes[0].tick_params(axis='x', rotation=60, labelsize=7)
    axes[0].tick_params(axis='y', rotation=0, labelsize=7)

    axes[1].barh(tcorr.index, tcorr.values, color=[C1 if v > 0 else C0 for v in tcorr.values], alpha=0.85)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_xlabel('Pearson r with target')
    axes[1].set_title('Feature Correlation with At-Risk Class\n(red = positive)', fontweight='bold', fontsize=11)
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'eda2_correlation_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    pairs = pairs[pairs.abs() > 0.8].rename('r').reset_index()
    pairs.columns = ['feature_a', 'feature_b', 'r']
    pairs.sort_values('r', key=abs, ascending=False).to_csv(EDA_DIR / 'eda2_high_correlation_pairs.csv', index=False)
    print('  Saved: eda2_correlation_analysis.png, eda2_correlation_matrix.csv, '
          'eda2_target_correlation.csv, eda2_high_correlation_pairs.csv')


def main():
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    df, num, cat = load()
    print(f'Loaded {df.shape} | at-risk prevalence {df[TARGET].mean() * 100:.1f}% | '
          f'{len(num)} continuous, {len(cat)} categorical')
    missing_and_sentinel_audit(df, num, cat)
    distributions_by_class(df, num)
    boxplots_by_class(df, num)
    correlation_analysis(df, num)
    print(f'\nDone. Outputs in {EDA_DIR}')


if __name__ == '__main__':
    main()
