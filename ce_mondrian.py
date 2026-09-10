"""
CE Step 5d — Mondrian CE (Sex-Stratified Calibration Quality)

Fits separate Venn-Abers calibrators per sex group using the calibration set.
Compares CE prediction interval widths and feature uncertainty widths between
Group A (lower sex value) and Group B (higher sex value).

Answers: "Does the model have different calibration reliability for male vs
          female patients?" — wider interval = less certain for that group.

Reference: Löfström & Löfström (2024), Conditional Calibrated Explanations,
           xAI 2024 Conference.

Output : MAIN_CE_OUTPUT/mondrian/
  - mondrian_ce_sex.csv                    (raw per-instance records)
  - mondrian_pred_interval_summary.csv     (group-level prediction width stats)
  - mondrian_feature_uncertainty_by_sex.png (top-10 feature uncertainty bar chart)

Runtime: ~30-60 min for all 10 models (100 instances per group per model)
"""

import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calibrated_explanations import CalibratedExplainer

from ce_utils import (
    Config, MODEL_REGISTRY,
    load_data, _load_single_model,
    CE_AVAILABLE,
)


# =============================================================================
# PER-MODEL MONDRIAN CE
# =============================================================================

def run_mondrian_for_model(model_key, model_name, model,
                           X_cal, y_cal, X_test, y_test, features, config,
                           male_mask_cal, female_mask_cal,
                           male_mask_test, female_mask_test):
    """
    Fit sex-stratified Venn-Abers for one model.
    Returns a list of row dicts for the combined mondrian_ce_sex.csv.
    """
    print(f"  [{model_name}]")

    rng  = np.random.default_rng(config.random_state)
    rows = []

    group_specs = [
        ('group_a', male_mask_cal,   male_mask_test),
        ('group_b', female_mask_cal, female_mask_test),
    ]

    for group_label, mask_cal, mask_test in group_specs:
        # Create ONE explainer at a time — delete before creating the next
        try:
            ce_grp = CalibratedExplainer(
                model,
                X_cal[mask_cal], y_cal[mask_cal],
                feature_names=features, mode='classification',
            )
        except Exception as e:
            print(f"    [warn] CalibratedExplainer init failed ({group_label}): {e}")
            continue

        feat_names = list(getattr(ce_grp, 'feature_names', features))
        test_idx   = np.where(mask_test)[0]
        sample     = rng.choice(
            test_idx,
            min(config.mondrian_n_sample, len(test_idx)),
            replace=False,
        )

        for idx in sample:
            try:
                exp    = ce_grp.explain_factual(X_test[[idx]])
                single = exp[0]
                p_low, p_high = single.prediction_interval
                pred_width    = float(p_high) - float(p_low)

                rule_data = single.get_rules()
                feat_idxs = rule_data.get('feature',    [])
                wt_vals   = rule_data.get('weight',      [])
                wt_highs  = rule_data.get('weight_high', [])
                wt_lows   = rule_data.get('weight_low',  [])

                for fi, wt, wh, wl in zip(feat_idxs, wt_vals, wt_highs, wt_lows):
                    if isinstance(fi, (int, np.integer)) and 0 <= fi < len(feat_names):
                        rows.append({
                            'model':          model_name,
                            'sex_group':      group_label,
                            'instance_idx':   int(idx),
                            'true_label':     int(y_test[idx]),
                            'feature':        feat_names[fi],
                            'ce_weight':      float(wt),
                            'pred_width':     pred_width,
                            'feat_unc_width': float(wh) - float(wl),
                        })
                del exp, single
            except Exception:
                pass

        del ce_grp
        gc.collect()   # release before creating the next group's explainer

    n = sum(1 for r in rows if r['model'] == model_name)
    print(f"    Records collected: {n}")
    return rows


# =============================================================================
# SUMMARY + CHARTS
# =============================================================================

def save_mondrian_outputs(mondrian_df, config):
    mondrian_dir = config.output_dir / 'mondrian'

    # Prediction interval width summary by sex group
    pred_summary = (
        mondrian_df
        .drop_duplicates(['model', 'sex_group', 'instance_idx'])
        .groupby('sex_group')
        .agg(
            mean_pred_width=('pred_width', 'mean'),
            std_pred_width =('pred_width', 'std'),
            n              =('instance_idx', 'count'),
        )
        .reset_index()
    )
    pred_summary.to_csv(
        mondrian_dir / 'mondrian_pred_interval_summary.csv', index=False)

    print(f"\n  Prediction interval width by sex group (averaged across all models):")
    for _, r in pred_summary.iterrows():
        print(f"    {r['sex_group']:>10}: "
              f"mean={r['mean_pred_width']:.4f}  "
              f"+/-{r['std_pred_width']:.4f}  "
              f"(n={r['n']})")
    print("  Wider interval = model is less certain for that group")

    # Bar chart: feature-level uncertainty width by sex group (top 10 features)
    try:
        top_feats = (mondrian_df.groupby('feature')['feat_unc_width']
                     .mean().nlargest(10).index.tolist())
        plot_df   = mondrian_df[mondrian_df['feature'].isin(top_feats)]
        pivot     = (plot_df
                     .groupby(['feature', 'sex_group'])['feat_unc_width']
                     .mean()
                     .unstack('sex_group')
                     .reindex(top_feats))

        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(pivot))
        w = 0.35
        ax.bar(x - w / 2, pivot.get('group_a', 0), w,
               label='Group A (lower sex value)',  color='#2980b9', alpha=0.85)
        ax.bar(x + w / 2, pivot.get('group_b', 0), w,
               label='Group B (higher sex value)', color='#e74c3c', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Mean CE Feature Uncertainty Width', fontsize=10)
        ax.set_title(
            'Mondrian CE — Feature-Level Uncertainty Width by Sex Group\n'
            '2018-2021 ENNS  '
            '(wider bar = model less confident about that feature for this group)',
            fontsize=10, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            mondrian_dir / 'mondrian_feature_uncertainty_by_sex.png',
            dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: mondrian_feature_uncertainty_by_sex.png")
    except Exception as e:
        print(f"  [warn] Chart failed: {e}")

    print(f"  Saved: mondrian_ce_sex.csv")
    print(f"  Saved: mondrian_pred_interval_summary.csv")


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not CE_AVAILABLE:
        return

    config = Config()
    _, _, X_cal, y_cal, X_test, y_test, features = load_data(config)

    mondrian_dir = config.output_dir / 'mondrian'
    mondrian_dir.mkdir(parents=True, exist_ok=True)

    sex_col   = config.sex_col_idx
    threshold = np.median(X_cal[:, sex_col])

    male_mask_cal    = X_cal[:, sex_col] <  threshold
    female_mask_cal  = X_cal[:, sex_col] >= threshold
    male_mask_test   = X_test[:, sex_col] <  threshold
    female_mask_test = X_test[:, sex_col] >= threshold

    print(f"\n  Sex threshold (median of col {sex_col}): {threshold}")
    print(f"  Cal  — Group A: {male_mask_cal.sum():>5},  "
          f"Group B: {female_mask_cal.sum()}")
    print(f"  Test — Group A: {male_mask_test.sum():>5},  "
          f"Group B: {female_mask_test.sum()}")

    if male_mask_cal.sum() < 10 or female_mask_cal.sum() < 10:
        print("  [warn] Insufficient calibration samples per sex group. Exiting.")
        return

    print("\n" + "="*70)
    print("MONDRIAN CE  (Sex-Stratified Calibration — Health Equity Lens)")
    print(f"  {config.mondrian_n_sample} test instances per group per model")
    print("="*70)

    all_rows = []
    for model_key, model_name, fmt in MODEL_REGISTRY:
        model = _load_single_model(model_key, model_name, fmt, config)
        if model is None:
            continue
        try:
            rows = run_mondrian_for_model(
                model_key, model_name, model,
                X_cal, y_cal, X_test, y_test, features, config,
                male_mask_cal, female_mask_cal,
                male_mask_test, female_mask_test,
            )
            all_rows.extend(rows)
        finally:
            del model   # release RAM before loading next model

    if not all_rows:
        print("  [warn] No Mondrian data collected.")
        return

    mondrian_df = pd.DataFrame(all_rows)
    mondrian_df.to_csv(mondrian_dir / 'mondrian_ce_sex.csv', index=False)

    save_mondrian_outputs(mondrian_df, config)

    print("\n" + "="*70)
    print("[DONE] ce_mondrian.py complete.")
    print(f"Outputs: {mondrian_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
