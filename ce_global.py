"""
CE Step 5c — Global CE Feature Importance Aggregation

Extracts CE feature weights across 150 test instances per model, then
aggregates into a global importance ranking with uncertainty width.

  global_ce_importance  = mean |CE weight| across 150 instances × all models
  uncertainty_width     = how consistently each feature matters
                          (not available from SHAP — novel CE contribution)

CRASH-SAFE:
  Saves a per-model checkpoint CSV after each model finishes.
  If the script is interrupted, rerun it — completed models are skipped
  automatically (no recomputation). Only the remaining models are processed.

  Checkpoint files: MAIN_CE_OUTPUT/global/ckpt_<model_key>.csv
  Delete a checkpoint to force recomputation for that model.

Output : MAIN_CE_OUTPUT/global/
  - global_ce_importance.csv   (feature ranking + uncertainty width)
  - global_ce_importance.png   (bar chart with error bars)
  - global_ce_heatmap.png      (feature × model consistency heatmap)
  - global_ce_by_model.csv     (per-model detail)
  - ckpt_<model_key>.csv       (per-model checkpoints, safe to delete after run)

Runtime: ~30-90 min for all 10 models (150 instances each, hardware-dependent)

Compare global CE ranking with SHAP results in MAIN_XAI/SHAP/global_importance.csv
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from calibrated_explanations import CalibratedExplainer

from ce_utils import (
    Config, MODEL_REGISTRY,
    load_data, _load_single_model,
    _extract_weights_from_explanation,
    CE_AVAILABLE,
)


# =============================================================================
# PER-MODEL WEIGHT EXTRACTION  (crash-safe, with checkpoints)
# =============================================================================

def extract_weights_for_model(model_key, model_name, model,
                               X_cal, y_cal, X_test, y_test, features, config):
    """
    Extract CE feature weights for n_factual_global instances.
    Saves a checkpoint CSV immediately on completion.
    Returns the weights DataFrame (or None on total failure).
    """
    out_dir   = config.output_dir / 'global'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Global CE weights: {model_name}")
    print(f"{'='*70}")
    print(f"  Initializing CalibratedExplainer (n_cal={len(X_cal)})...")

    try:
        ce = CalibratedExplainer(
            model, X_cal, y_cal,
            feature_names=features,
            mode='classification',
        )
        print(f"  [OK] CalibratedExplainer ready")
    except Exception as e:
        print(f"  [ERROR] Init failed: {e}")
        return None

    rng     = np.random.default_rng(config.random_state)
    n       = min(config.n_factual_global, len(X_test))
    indices = rng.choice(len(X_test), n, replace=False)
    rows    = []
    t0      = time.time()

    print(f"  Extracting weights for {n} instances...")
    for i, idx in enumerate(indices):
        try:
            exp    = ce.explain_factual(X_test[[idx]])
            single = exp[0]
            w = _extract_weights_from_explanation(single, features)
            if w:
                w['instance_idx'] = int(idx)
                w['true_label']   = int(y_test[idx])
                rows.append(w)
        except Exception:
            pass

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / max(elapsed, 1e-9)
            eta     = (n - i - 1) / rate
            print(f"    {i+1}/{n}  |  {elapsed:.0f}s elapsed  |  ~{eta:.0f}s remaining")

    if not rows:
        print(f"  [warn] No weights collected for {model_name}")
        return None

    df        = pd.DataFrame(rows)
    ckpt_path = out_dir / f'ckpt_{model_key}.csv'
    df.to_csv(ckpt_path, index=False)
    elapsed   = time.time() - t0
    print(f"  [saved] checkpoint: {ckpt_path.name}  "
          f"({len(df)} instances, {elapsed:.0f}s)")
    return df


# =============================================================================
# AGGREGATION + CHARTS
# =============================================================================

def aggregate_global_ce(model_weights, features, config):
    """Aggregate per-model weight DataFrames → global CE importance."""
    print("\n" + "="*70)
    print("AGGREGATING GLOBAL CE IMPORTANCE")
    print("="*70)

    out_dir = config.output_dir / 'global'
    out_dir.mkdir(parents=True, exist_ok=True)

    registry_map   = {k: n for k, n, _ in MODEL_REGISTRY}
    per_model_rows = []

    for model_key, weights_df in model_weights.items():
        if weights_df is None:
            continue
        model_name = registry_map.get(model_key, model_key)
        for feat in features:
            if feat in weights_df.columns:
                vals = pd.to_numeric(weights_df[feat], errors='coerce').dropna()
                if len(vals) == 0:
                    continue
                per_model_rows.append({
                    'model':           model_name,
                    'feature':         feat,
                    'mean_abs_weight': float(vals.abs().mean()),
                    'std_weight':      float(vals.abs().std()),
                    'n_instances':     len(vals),
                })

    if not per_model_rows:
        print("  [warn] No weight data available for aggregation.")
        return pd.DataFrame()

    per_model_df = pd.DataFrame(per_model_rows)
    per_model_df.to_csv(out_dir / 'global_ce_by_model.csv', index=False)

    summary = (per_model_df.groupby('feature')
               .agg(
                   global_ce_importance=('mean_abs_weight', 'mean'),
                   uncertainty_width   =('std_weight',      'mean'),
                   n_models            =('model',           'count'),
               )
               .sort_values('global_ce_importance', ascending=False)
               .reset_index())
    summary.to_csv(out_dir / 'global_ce_importance.csv', index=False)

    print(f"\n  {'Rank':<5} {'Feature':<28} {'CE Importance':>15} {'Uncertainty':>13}")
    print("  " + "-"*63)
    for rank, (_, row) in enumerate(summary.head(15).iterrows(), 1):
        print(f"  {rank:<5} {row['feature']:<28} "
              f"{row['global_ce_importance']:>15.4f} "
              f"{row['uncertainty_width']:>13.4f}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    top = summary.head(15)
    fig, ax = plt.subplots(figsize=(11, 7))
    y_pos = np.arange(len(top))[::-1]
    ax.barh(y_pos, top['global_ce_importance'],
            xerr=top['uncertainty_width'],
            color='#2980b9', alpha=0.85, edgecolor='white',
            capsize=4, error_kw={'linewidth': 1.2, 'color': '#555'})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top['feature'], fontsize=9)
    ax.set_xlabel('Mean |CE Feature Weight|  '
                  f'({config.n_factual_global} instances × all models)', fontsize=10)
    ax.set_title(
        'Global CE Feature Importance (V2 Hybrid)\n'
        'Aggregated Calibrated Explanations — 2018-2021 ENNS\n'
        'Error bars = uncertainty width  |  '
        'Compare with SHAP global from main_xai.py',
        fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'global_ce_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: global_ce_importance.png")
    print(f"  Saved: global_ce_importance.csv")

    # ── Feature × model heatmap ───────────────────────────────────────────────
    try:
        pivot = per_model_df.pivot_table(
            index='feature', columns='model',
            values='mean_abs_weight', aggfunc='mean')
        pivot = pivot.loc[summary['feature'].head(15)]

        fig, ax = plt.subplots(figsize=(16, 7))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                    linewidths=0.5, ax=ax,
                    cbar_kws={'label': '|CE Weight|'})
        ax.set_title(
            'CE Feature Importance by Model — V2 Hybrid\n2018-2021 ENNS',
            fontsize=11, fontweight='bold')
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Feature', fontsize=10)
        plt.xticks(rotation=30, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / 'global_ce_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: global_ce_heatmap.png  (feature × model consistency)")
    except Exception as e:
        print(f"  [warn] Heatmap failed: {e}")

    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not CE_AVAILABLE:
        return

    config  = Config()
    out_dir = config.output_dir / 'global'
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, X_cal, y_cal, X_test, y_test, features = load_data(config)

    print(f"\n  Global CE: {config.n_factual_global} instances per model")
    print(f"  CRASH-SAFE: checkpoints in {out_dir}")
    print(f"  Rerun at any time — completed models are skipped automatically.\n")

    model_weights = {}
    for model_key, model_name, fmt in MODEL_REGISTRY:
        ckpt_path = out_dir / f'ckpt_{model_key}.csv'

        # Resume: skip model if checkpoint already exists
        if ckpt_path.exists():
            print(f"\n  [SKIP] {model_name} — checkpoint found, loading cached results")
            model_weights[model_key] = pd.read_csv(ckpt_path)
            continue

        model = _load_single_model(model_key, model_name, fmt, config)
        if model is None:
            continue
        try:
            df = extract_weights_for_model(
                model_key, model_name, model,
                X_cal, y_cal, X_test, y_test, features, config,
            )
            model_weights[model_key] = df
        finally:
            del model   # release RAM before loading next model

    aggregate_global_ce(model_weights, features, config)

    print("\n" + "="*70)
    print("[DONE] ce_global.py complete.")
    print(f"Outputs: {out_dir}")
    print("Note   : checkpoint CSVs (ckpt_*.csv) can be deleted after successful run.")
    print("Next   : python ce_mondrian.py")
    print("="*70)


if __name__ == '__main__':
    main()
