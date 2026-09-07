"""
Main Calibrated Explanations Pipeline  (V2 Hybrid — 2018-2021 ENNS)

Step 5 in the V2 hybrid pipeline — adds CE alongside existing SHAP + calibration:

  Step 3: main_xai.py         — SHAP global feature importance (retained)
  Step 4: main_calibration.py — Platt / Isotonic / Temperature calibration (retained)
  Step 5: main_ce.py          — THIS SCRIPT — CE local uncertainty + CCE + Mondrian

Hybrid rationale:
  SHAP (Step 3)  → global feature ranking  (exact Shapley decomposition)
  CE   (Step 5)  → local uncertainty intervals per patient (Venn-Abers)
                   + counterfactual "what-if" with reliability filter (Ensured CCE)
                   + sex-stratified calibration quality (Mondrian CE)

For each trained model this script produces:
  1. CP       — Calibrated Predictions with uncertainty interval (Venn-Abers)
  2. CE       — Calibrated Explanations: feature weights with confidence intervals (factual)
  3. Narrative — Human-readable text explanation per patient (to_narrative)
  4. CCE      — Counterfactual Calibrated Explanations: "what-if" with uncertainty
  5. Ensured  — Filtered CCE keeping only reliable alternatives
                (Lofstrom et al., arXiv:2410.05479)
  6. Global CE — mean |weight| + uncertainty width per feature across all models
  7. Mondrian CE — sex-stratified Venn-Abers: uncertainty comparison by sex group

Data split mapping:
  X_train / y_train  → used for model fitting  (in main_train.py)
  X_val   / y_val    → used as CALIBRATION SET for CalibratedExplainer (Venn-Abers)
  X_test  / y_test   → final unseen evaluation set

NOTE: Load uncalibrated models from MAIN_TRAINED_MODELS/ — CE handles calibration
      internally via Venn-Abers. Do NOT use models from MAIN_TRAINED_MODELS_CALIB/.
"""

import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import time
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message='.*deep-copy.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*CalibratedExplainer.*', category=UserWarning)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score, average_precision_score
)

try:
    from calibrated_explanations import CalibratedExplainer
    CE_AVAILABLE = True
except ImportError:
    CE_AVAILABLE = False
    print("[ERROR] calibrated-explanations not installed.")
    print("        Run: pip install calibrated-explanations")

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False


# =============================================================================
# CONFIG
# =============================================================================

class Config:
    data_dir   = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\MAIN_PREPROCESS_OUTPUT")
    models_dir = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\MAIN_TRAINED_MODELS")
    output_dir = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\MAIN_CE_OUTPUT")

    # Number of test instances to explain
    n_factual_plots      = 6    # per model: 3 at-risk + 3 normal individual plots
    n_factual_global     = 500  # for global CE aggregation (None = all test instances)
    n_counterfactual     = 6    # CCE plots: 3 at-risk + 3 normal

    # Mondrian (sex-stratified) calibration
    sex_col_idx       = 20   # feature index for 'sex' (see feature_list.csv)
    mondrian_n_sample = 100  # test instances per sex group

    random_state = 42


# Models to load — botabnet uses .zip (TabNet format), all others use .joblib
MODEL_REGISTRY = [
    ('naive_bayes',   'Naive Bayes',           'joblib'),
    ('knn',           'kNN (Tuned)',            'joblib'),
    ('adaboost',      'AdaBoost (Tuned)',        'joblib'),
    ('xgboost',       'XGBoost (Tuned)',         'joblib'),
    ('random_forest', 'Random Forest (Tuned)',   'joblib'),
    ('catboost',      'CatBoost (Tuned)',         'joblib'),
    ('lightgbm',      'LightGBM (Tuned)',         'joblib'),
    ('stacking',      'Stacking Ensemble',        'joblib'),
    ('voting',        'Voting Ensemble',          'joblib'),
    ('botabnet',      'BO-TabNet',                'zip'),
]


# =============================================================================
# LOAD DATA & MODELS
# =============================================================================

def load_data(config):
    print("\n" + "="*80)
    print("LOADING DATA  (V2 Hybrid — 2018-2021 ENNS)")
    print("="*80)
    X_train = np.load(config.data_dir / 'X_train.npy')
    y_train = np.load(config.data_dir / 'y_train.npy')
    X_cal   = np.load(config.data_dir / 'X_val.npy')   # val = calibration set for CE
    y_cal   = np.load(config.data_dir / 'y_val.npy')
    X_test  = np.load(config.data_dir / 'X_test.npy')
    y_test  = np.load(config.data_dir / 'y_test.npy')
    info    = joblib.load(config.data_dir / 'preprocessing_info.pkl')

    features = info.get('feature_names', [f'f{i}' for i in range(X_train.shape[1])])
    print(f"  Train : {X_train.shape}")
    print(f"  Cal   : {X_cal.shape}   (calibration set for Venn-Abers)")
    print(f"  Test  : {X_test.shape}")
    print(f"  Features ({len(features)}): {features[:6]}...")
    return X_train, y_train, X_cal, y_cal, X_test, y_test, features


def _load_single_model(model_key, model_name, fmt, config):
    """Load a single model by key. Handles joblib and TabNet zip formats."""
    if fmt == 'zip':
        path = config.models_dir / f'{model_key}.zip'
        if not path.exists():
            print(f"  [SKIP] {model_name} — {path.name} not found")
            return None
        if not TABNET_AVAILABLE:
            print(f"  [SKIP] {model_name} — pytorch_tabnet not installed")
            return None
        try:
            model = TabNetClassifier()
            model.load_model(str(path))
            print(f"  [OK]   {model_name}  (TabNet zip)")
            return model
        except Exception as e:
            print(f"  [FAIL] {model_name}: {e}")
            return None
    else:
        path = config.models_dir / f'{model_key}.joblib'
        if not path.exists():
            print(f"  [SKIP] {model_name} — {path.name} not found")
            return None
        try:
            model = joblib.load(path)
            print(f"  [OK]   {model_name}")
            return model
        except Exception as e:
            print(f"  [FAIL] {model_name}: {e}")
            return None


def load_models(config):
    print("\n" + "="*80)
    print("LOADING TRAINED MODELS  (uncalibrated — CE handles calibration via Venn-Abers)")
    print("="*80)
    loaded = {}
    for model_key, model_name, fmt in MODEL_REGISTRY:
        model = _load_single_model(model_key, model_name, fmt, config)
        if model is not None:
            loaded[model_key] = model
    return loaded


# =============================================================================
# CALIBRATION QUALITY EVALUATION  (raw model probabilities on test set)
# =============================================================================

def evaluate_calibration_quality(models, X_test, y_test, config):
    """
    Evaluate raw model probability quality BEFORE CE wrapping.
    Reports ROC-AUC, Brier score, and log-loss.
    Complements V2's main_calibration.py results (Platt/Isotonic/Temperature).
    CalibratedExplainer will apply Venn-Abers internally.
    """
    print("\n" + "="*80)
    print("CALIBRATION QUALITY  (raw model probabilities — pre-CE Venn-Abers)")
    print("Compare with main_calibration.py results for Platt/Isotonic/Temperature")
    print("="*80)

    rows = []
    for model_key, model_name, _ in MODEL_REGISTRY:
        if model_key not in models:
            continue
        model = models[model_key]
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            rows.append({
                'model':       model_name,
                'roc_auc':     roc_auc_score(y_test, y_proba),
                'brier_score': brier_score_loss(y_test, y_proba),
                'log_loss':    log_loss(y_test, y_proba),
                'auprc':       average_precision_score(y_test, y_proba),
            })
        except Exception as e:
            print(f"  [warn] {model_name}: {e}")

    if not rows:
        return pd.DataFrame()

    cal_df = pd.DataFrame(rows).sort_values('brier_score')
    print(f"\n  {'Model':<28} {'ROC-AUC':>8} {'Brier':>8} {'Log-Loss':>10} {'AUPRC':>8}")
    print("  " + "-"*66)
    for _, r in cal_df.iterrows():
        print(f"  {r['model']:<28} {r['roc_auc']:>8.4f} {r['brier_score']:>8.4f} "
              f"{r['log_loss']:>10.4f} {r['auprc']:>8.4f}")

    out = config.output_dir / 'global'
    out.mkdir(parents=True, exist_ok=True)
    cal_df.to_csv(out / 'pre_ce_calibration_quality.csv', index=False)
    print(f"\n  Saved: pre_ce_calibration_quality.csv")
    return cal_df


# =============================================================================
# PLOT + NARRATIVE HELPERS
# =============================================================================

def _save_narrative(single_exp, plot_path):
    """
    Save to_narrative() text output alongside the CE plot.
    Uses 'advanced' expertise level — includes calibrated prob, interval, and
    signed feature weights with uncertainty envelopes (as shown in CE README).
    """
    try:
        narrative = single_exp.to_narrative(
            output_format='text', expertise_level='advanced')
        narrative_path = str(plot_path).replace('.png', '_narrative.txt')
        with open(narrative_path, 'w', encoding='utf-8') as f:
            f.write(narrative)
        return True
    except Exception:
        return False


def _save_plot(single_exp, plot_path):
    """
    Work around the library's hardcoded 'plots/' prefix on the filename arg.

    Strategy:
      1. Change CWD to the target directory.
      2. Call plot(filename=basename) — library saves to plots/<basename>
         relative to CWD, i.e. <target_dir>/plots/<basename>.
      3. Move the file to the final path and restore CWD.

    This avoids the WinError 123 from prepending 'plots/' to an absolute
    Windows path, and avoids the blank-figure problem from plt.gcf() returning
    a new empty canvas after the library closes its own figure internally.
    """
    plot_path    = Path(plot_path)
    target_dir   = plot_path.parent
    basename     = plot_path.name
    plots_sub    = target_dir / 'plots'
    saved_by_lib = plots_sub / basename

    plots_sub.mkdir(parents=True, exist_ok=True)
    original_cwd = os.getcwd()
    try:
        os.chdir(target_dir)
        single_exp.plot(show=False, filename=basename)
        if saved_by_lib.exists():
            shutil.move(str(saved_by_lib), str(plot_path))
            return True
        return False
    except Exception as e:
        print(f"    [plot error] {e}")
        return False
    finally:
        os.chdir(original_cwd)
        try:
            if plots_sub.exists() and not any(plots_sub.iterdir()):
                plots_sub.rmdir()
        except Exception:
            pass


# =============================================================================
# EXTRACT FEATURE WEIGHTS  (v1.0.0 API: get_rules() is the reliable source)
# =============================================================================

def _extract_weights_from_explanation(factual_exp, features):
    """
    Extract feature weight point estimates from a FactualExplanation object.
    Uses get_rules() which is stable in calibrated-explanations v1.0.0.

    Returns: dict {feature_name: weight_value} or None if extraction fails.
    """
    weights = {}

    # Method 1: get_rules() — stable in v1.0.0
    try:
        rules = factual_exp.get_rules()
        feat_indices = rules.get('feature', [])
        weight_vals  = rules.get('weight', [])
        explainer    = factual_exp.get_explainer()
        feat_names   = list(getattr(explainer, 'feature_names', features))
        for fi, w in zip(feat_indices, weight_vals):
            if isinstance(fi, (int, np.integer)) and 0 <= fi < len(feat_names):
                fname = feat_names[fi]
                if fname not in weights:
                    weights[fname] = float(w)
        if weights:
            return weights
    except Exception:
        pass

    # Method 2: feature_weights attribute (fallback)
    try:
        fw = factual_exp.feature_weights
        if isinstance(fw, dict):
            arr = None
            for key in fw:
                candidate = np.asarray(fw[key]).flatten()
                if len(candidate) == len(features):
                    arr = candidate
                    break
            if arr is not None:
                for i, feat in enumerate(features):
                    weights[feat] = float(arr[i])
                return weights
    except Exception:
        pass

    return None


# =============================================================================
# RUN CE FOR ONE MODEL
# =============================================================================

def run_ce_model(model_key, model_name, model, X_cal, y_cal,
                 X_test, y_test, features, config):
    """
    Run Calibrated Explanations for a single model.
    Returns: DataFrame of feature weights across factual instances (for global CE).
    """
    print(f"\n{'='*80}")
    print(f"  CE: {model_name}")
    print(f"{'='*80}")

    factual_dir = config.output_dir / 'factual' / model_key
    ccf_dir     = config.output_dir / 'counterfactual' / model_key
    factual_dir.mkdir(parents=True, exist_ok=True)
    ccf_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.random_state)

    # ── Initialize CalibratedExplainer ───────────────────────────────────────
    print(f"  Initializing CalibratedExplainer (Venn-Abers, cal_n={len(X_cal)})...")
    try:
        ce = CalibratedExplainer(
            model,
            X_cal,
            y_cal,
            feature_names=features,
            mode='classification',
        )
        print(f"  [OK] CalibratedExplainer initialized")
    except Exception as e:
        print(f"  [ERROR] CalibratedExplainer init failed: {e}")
        return None

    weights_collection = []

    # ── 1. FACTUAL CE — individual plots ─────────────────────────────────────
    at_risk_idx = np.where(y_test == 1)[0]
    normal_idx  = np.where(y_test == 0)[0]

    n_each = config.n_factual_plots // 2
    plot_idx = np.concatenate([
        rng.choice(at_risk_idx, min(n_each, len(at_risk_idx)), replace=False),
        rng.choice(normal_idx,  min(n_each, len(normal_idx)),  replace=False),
    ])
    plot_labels = (
        ['at_risk'] * min(n_each, len(at_risk_idx)) +
        ['normal']  * min(n_each, len(normal_idx))
    )

    print(f"  Generating factual CE plots ({len(plot_idx)} instances)...")
    for idx, label in zip(plot_idx, plot_labels):
        instance = X_test[[idx]]
        try:
            exp    = ce.explain_factual(instance)
            single = exp[0]

            try:
                p_low, p_high = single.prediction_interval
            except Exception:
                p_low = p_high = None

            plot_path = str(factual_dir / f'ce_factual_{label}_{idx}.png')
            _save_plot(single, plot_path)
            _save_narrative(single, plot_path)

            w = _extract_weights_from_explanation(single, features)
            if w:
                w['instance_idx'] = idx
                w['true_label']   = int(y_test[idx])
                w['label_group']  = label
                if p_low is not None:
                    w['pred_low']  = float(p_low)
                    w['pred_high'] = float(p_high)
                weights_collection.append(w)

        except Exception as e:
            print(f"    [warn] instance {idx} factual CE failed: {e}")

    print(f"  Factual CE plots saved → {factual_dir}")

    # ── 2. FACTUAL CE — global aggregation sample ─────────────────────────────
    n_global = config.n_factual_global
    if n_global is None:
        global_idx = np.arange(len(X_test))
    else:
        n_global = min(n_global, len(X_test))
        global_idx = rng.choice(len(X_test), n_global, replace=False)

    print(f"  Extracting feature weights for global CE ({len(global_idx)} instances)...")
    success_count = 0
    plot_idx_set = set(plot_idx.tolist())
    for idx in global_idx:
        if idx in plot_idx_set:
            continue
        try:
            exp    = ce.explain_factual(X_test[[idx]])
            single = exp[0]
            w = _extract_weights_from_explanation(single, features)
            if w:
                w['instance_idx'] = idx
                w['true_label']   = int(y_test[idx])
                w['label_group']  = 'at_risk' if y_test[idx] == 1 else 'normal'
                weights_collection.append(w)
                success_count += 1
        except Exception:
            pass

    print(f"  Global CE weights extracted: {success_count + len(plot_idx)} instances")

    # ── 3. COUNTERFACTUAL CCE ─────────────────────────────────────────────────
    n_each_ccf = config.n_counterfactual // 2
    ccf_at  = rng.choice(at_risk_idx, min(n_each_ccf, len(at_risk_idx)), replace=False)
    ccf_nrm = rng.choice(normal_idx,  min(n_each_ccf, len(normal_idx)),  replace=False)
    ccf_pairs = (list(zip(ccf_at,  ['at_risk'] * len(ccf_at))) +
                 list(zip(ccf_nrm, ['normal']  * len(ccf_nrm))))

    print(f"  Generating counterfactual CCE ({len(ccf_pairs)} instances)...")
    for idx, label in ccf_pairs:
        instance = X_test[[idx]]
        try:
            counter = ce.explore_alternatives(instance)
            single  = counter[0]

            plot_path = str(ccf_dir / f'cce_{label}_{idx}.png')
            _save_plot(single, plot_path)
            _save_narrative(single, plot_path)

            try:
                ensured      = single.ensured_explanations()
                ensured_path = str(ccf_dir / f'cce_ensured_{label}_{idx}.png')
                _save_plot(ensured, ensured_path)
                _save_narrative(ensured, ensured_path)
            except Exception as ee:
                print(f"    [warn] Ensured CCE {idx}: {ee}")

        except Exception as e:
            print(f"    [warn] CCE instance {idx} failed: {e}")

    print(f"  CCE plots saved → {ccf_dir}")

    if not weights_collection:
        print(f"  [warn] No feature weights collected for {model_name}")
        return None

    weights_df = pd.DataFrame(weights_collection)
    weights_df.to_csv(factual_dir / 'factual_feature_weights.csv', index=False)
    print(f"  Saved: factual_feature_weights.csv ({len(weights_df)} instances)")
    return weights_df


# =============================================================================
# MONDRIAN CE  (Sex-Stratified Calibration — Health Equity Lens)
# =============================================================================

def run_mondrian_ce(models, X_cal, y_cal, X_test, y_test, features, config):
    """
    Mondrian (Conditional) Calibrated Explanations — stratified by sex.

    Fits separate Venn-Abers calibrators for male and female subgroups using
    the calibration set (X_val / y_val).  Compares CE uncertainty interval
    widths between groups — answering:
      'Does the model have different calibration quality for male vs. female patients?'

    Wider uncertainty for a group = model is less reliable for that group.
    Reference: Lofstrom & Lofstrom (2024), Conditional Calibrated Explanations, xAI 2024.
    """
    print("\n" + "="*80)
    print("MONDRIAN CE  (Sex-Stratified Calibration — Health Equity Lens)")
    print("Comparing CE uncertainty intervals: Group A vs. Group B (by sex column)")
    print("Reference: Lofstrom & Lofstrom (2024), Conditional Calibrated Explanations")
    print("="*80)

    mondrian_dir = config.output_dir / 'mondrian'
    mondrian_dir.mkdir(parents=True, exist_ok=True)

    sex_col   = config.sex_col_idx
    threshold = np.median(X_cal[:, sex_col])

    male_mask_cal    = X_cal[:, sex_col] <  threshold
    female_mask_cal  = X_cal[:, sex_col] >= threshold
    male_mask_test   = X_test[:, sex_col] <  threshold
    female_mask_test = X_test[:, sex_col] >= threshold

    print(f"  Sex threshold (median): {threshold}")
    print(f"  Cal  — Group A (<threshold): {male_mask_cal.sum()},  "
          f"Group B (>=threshold): {female_mask_cal.sum()}")
    print(f"  Test — Group A: {male_mask_test.sum()},  Group B: {female_mask_test.sum()}")

    if male_mask_cal.sum() < 10 or female_mask_cal.sum() < 10:
        print("  [warn] Insufficient calibration samples per sex group. Skipping Mondrian CE.")
        return

    rng      = np.random.default_rng(config.random_state)
    n_sample = config.mondrian_n_sample
    rows     = []

    for model_key, model_name, _ in MODEL_REGISTRY:
        if model_key not in models:
            continue
        model = models[model_key]
        print(f"\n  [{model_name}]")

        try:
            ce_grp_a = CalibratedExplainer(
                model, X_cal[male_mask_cal], y_cal[male_mask_cal],
                feature_names=features, mode='classification')
            ce_grp_b = CalibratedExplainer(
                model, X_cal[female_mask_cal], y_cal[female_mask_cal],
                feature_names=features, mode='classification')
        except Exception as e:
            print(f"    [warn] init failed: {e}")
            continue

        for group_label, ce_grp, mask_test in [
            ('group_a', ce_grp_a, male_mask_test),
            ('group_b', ce_grp_b, female_mask_test),
        ]:
            test_idx = np.where(mask_test)[0]
            sample   = rng.choice(test_idx, min(n_sample, len(test_idx)), replace=False)

            for idx in sample:
                try:
                    exp    = ce_grp.explain_factual(X_test[[idx]])
                    single = exp[0]
                    p_low, p_high = single.prediction_interval
                    pred_width = float(p_high) - float(p_low)

                    rules      = single.get_rules()
                    feat_idxs  = rules.get('feature', [])
                    wt_vals    = rules.get('weight', [])
                    wt_highs   = rules.get('weight_high', [])
                    wt_lows    = rules.get('weight_low', [])
                    feat_names = list(getattr(ce_grp, 'feature_names', features))

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
                except Exception:
                    pass

        n_model = sum(1 for r in rows if r['model'] == model_name)
        print(f"    Records collected: {n_model}")

    if not rows:
        print("  [warn] No Mondrian data collected.")
        return

    mondrian_df = pd.DataFrame(rows)
    mondrian_df.to_csv(mondrian_dir / 'mondrian_ce_sex.csv', index=False)

    # ── Prediction interval width comparison ─────────────────────────────────
    pred_summary = (mondrian_df.drop_duplicates(['model', 'sex_group', 'instance_idx'])
                    .groupby('sex_group')
                    .agg(mean_pred_width=('pred_width', 'mean'),
                         std_pred_width =('pred_width', 'std'),
                         n              =('instance_idx', 'count'))
                    .reset_index())
    pred_summary.to_csv(mondrian_dir / 'mondrian_pred_interval_summary.csv', index=False)
    print(f"\n  Prediction interval width by sex group (averaged across all models):")
    for _, r in pred_summary.iterrows():
        print(f"    {r['sex_group']:>10}: mean={r['mean_pred_width']:.4f}  "
              f"+/-{r['std_pred_width']:.4f}  (n={r['n']})")
    print("  Wider interval = model is less certain for that group")

    # ── Plot: feature CE uncertainty width by sex group ───────────────────────
    try:
        top_feats = (mondrian_df.groupby('feature')['feat_unc_width']
                     .mean().nlargest(10).index.tolist())
        plot_df   = mondrian_df[mondrian_df['feature'].isin(top_feats)]
        pivot     = (plot_df.groupby(['feature', 'sex_group'])['feat_unc_width']
                     .mean().unstack('sex_group').reindex(top_feats))

        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(pivot))
        w = 0.35
        ax.bar(x - w/2, pivot.get('group_a', 0), w,
               label='Group A (lower sex value)', color='#2980b9', alpha=0.85)
        ax.bar(x + w/2, pivot.get('group_b', 0), w,
               label='Group B (higher sex value)', color='#e74c3c', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Mean CE Feature Uncertainty Width', fontsize=10)
        ax.set_title(
            'Mondrian CE — Feature-Level Uncertainty Width by Sex Group\n'
            '2018-2021 ENNS  (wider bar = model less confident about feature for this group)',
            fontsize=10, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(mondrian_dir / 'mondrian_feature_uncertainty_by_sex.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: mondrian_feature_uncertainty_by_sex.png")
    except Exception as e:
        print(f"  [warn] Mondrian plot failed: {e}")

    print(f"  Saved: mondrian_ce_sex.csv")
    print(f"  Mondrian CE complete -> {mondrian_dir}")


# =============================================================================
# GLOBAL CE AGGREGATION
# =============================================================================

def compute_global_ce(model_weights, features, config):
    """
    Aggregate feature weights across all models and test instances.

    Produces:
      - mean |weight|       -> global CE importance (compare with SHAP global from main_xai.py)
      - uncertainty_width   -> how consistently each feature matters across patients
                               (not available from SHAP — novel contribution of CE)

    Note: For authoritative global feature ranking, use SHAP results from main_xai.py.
    This CE aggregation adds the uncertainty dimension that SHAP cannot provide.
    """
    print("\n" + "="*80)
    print("GLOBAL CE AGGREGATION")
    print("mean |feature weight| + uncertainty width across all models and instances")
    print("Compare global ranking with SHAP results from main_xai.py")
    print("="*80)

    global_dir = config.output_dir / 'global'
    global_dir.mkdir(parents=True, exist_ok=True)

    registry_map = {k: n for k, n, _ in MODEL_REGISTRY}
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
                    'mean_abs_weight': vals.abs().mean(),
                    'std_weight':      vals.abs().std(),
                    'n_instances':     len(vals),
                })

    if not per_model_rows:
        print("  [warn] No weight data available for global CE aggregation.")
        return pd.DataFrame()

    per_model_df = pd.DataFrame(per_model_rows)
    per_model_df.to_csv(global_dir / 'global_ce_by_model.csv', index=False)

    summary = (per_model_df.groupby('feature')
               .agg(
                   global_ce_importance=('mean_abs_weight', 'mean'),
                   uncertainty_width=('std_weight', 'mean'),
                   n_models=('model', 'count'),
               )
               .sort_values('global_ce_importance', ascending=False)
               .reset_index())
    summary.to_csv(global_dir / 'global_ce_importance.csv', index=False)

    print(f"\n  {'Rank':<5} {'Feature':<28} {'Global CE Importance':>22} {'Uncertainty Width':>20}")
    print("  " + "-"*77)
    for rank, (_, row) in enumerate(summary.head(15).iterrows(), 1):
        print(f"  {rank:<5} {row['feature']:<28} {row['global_ce_importance']:>22.4f} "
              f"{row['uncertainty_width']:>20.4f}")

    # ── Plot 1: Global CE importance bar chart ────────────────────────────────
    top = summary.head(15)
    fig, ax = plt.subplots(figsize=(11, 7))
    y_pos = np.arange(len(top))[::-1]
    ax.barh(y_pos, top['global_ce_importance'],
            xerr=top['uncertainty_width'],
            color='#2980b9', alpha=0.85, edgecolor='white',
            capsize=4, error_kw={'linewidth': 1.2, 'color': '#555'})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top['feature'], fontsize=9)
    ax.set_xlabel('Mean |CE Feature Weight|  (aggregated across all models)', fontsize=10)
    ax.set_title(
        'Global CE Feature Importance (V2 Hybrid)\n'
        'Aggregated Calibrated Explanations — 2018-2021 ENNS\n'
        'Error bars = uncertainty width  |  Compare with SHAP global from main_xai.py',
        fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(global_dir / 'global_ce_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: global_ce_importance.png")
    print(f"  Saved: global_ce_importance.csv")

    # ── Plot 2: Consistency heatmap (feature x model) ─────────────────────────
    try:
        pivot = per_model_df.pivot_table(
            index='feature', columns='model', values='mean_abs_weight', aggfunc='mean')
        pivot = pivot.loc[summary['feature'].head(15)]

        fig, ax = plt.subplots(figsize=(16, 7))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                    linewidths=0.5, ax=ax, cbar_kws={'label': '|CE Weight|'})
        ax.set_title('CE Feature Importance by Model — V2 Hybrid\n2018-2021 ENNS',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Feature', fontsize=10)
        plt.xticks(rotation=30, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(global_dir / 'global_ce_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: global_ce_heatmap.png  (feature x model consistency)")
    except Exception as e:
        print(f"  [warn] Heatmap failed: {e}")

    return summary


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_ce_summary(global_summary, cal_quality_df, config):
    print("\n" + "="*80)
    print("CE PIPELINE SUMMARY  (V2 Hybrid)")
    print("="*80)

    print("\n  [1] Pre-CE Venn-Abers Calibration Quality:")
    if not cal_quality_df.empty:
        best_brier = cal_quality_df.loc[cal_quality_df['brier_score'].idxmin()]
        best_auc   = cal_quality_df.loc[cal_quality_df['roc_auc'].idxmax()]
        print(f"      Best Brier : {best_brier['model']}  ({best_brier['brier_score']:.4f})")
        print(f"      Best AUC   : {best_auc['model']}  ({best_auc['roc_auc']:.4f})")
        print(f"      (Compare with Platt/Isotonic from main_calibration.py)")

    print("\n  [2] Top 5 Global CE Features (compare with SHAP from main_xai.py):")
    if not global_summary.empty:
        for rank, (_, row) in enumerate(global_summary.head(5).iterrows(), 1):
            print(f"      {rank}. {row['feature']}  "
                  f"(CE importance: {row['global_ce_importance']:.4f}, "
                  f"uncertainty: +/-{row['uncertainty_width']:.4f})")

    print(f"\n  [3] Output directory: {config.output_dir}")
    print(f"\n  Outputs:")
    print(f"    factual/<model>/        -- CE plots + _narrative.txt per patient")
    print(f"    counterfactual/<model>/ -- CCE plots + Ensured CCE + narratives")
    print(f"    global/                 -- global CE importance + heatmap")
    print(f"    mondrian/               -- sex-stratified uncertainty comparison")
    print(f"\n  Hybrid pipeline complete:")
    print(f"    Step 3 MAIN_XAI/SHAP/   -- SHAP global importance (authoritative ranking)")
    print(f"    Step 4 MAIN_CALIBRATION/ -- Platt/Isotonic/Temperature calibration")
    print(f"    Step 5 MAIN_CE_OUTPUT/  -- CE local uncertainty + CCE + Mondrian")
    print("="*80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not CE_AVAILABLE:
        print("\n[ABORT] calibrated-explanations package not found.")
        print("  Install: pip install calibrated-explanations")
        return

    config = Config()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("MAIN CALIBRATED EXPLANATIONS PIPELINE  (V2 Hybrid — 2018-2021 ENNS)")
    print("Step 5: CE local uncertainty + CCE + Mondrian")
    print("        alongside SHAP (Step 3) + Platt/Isotonic (Step 4)")
    print("="*80)

    start_time = time.time()

    X_train, y_train, X_cal, y_cal, X_test, y_test, features = load_data(config)
    models = load_models(config)

    if not models:
        print("\n[ABORT] No models found. Run main_train.py first.")
        return

    # ── Pre-CE calibration quality ──────────────────────────────────────────
    cal_quality_df = evaluate_calibration_quality(models, X_test, y_test, config)

    # ── Run CE per model ────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("RUNNING CALIBRATED EXPLANATIONS  (CE + CCE per model)")
    print("="*80)

    registry_keys = {k for k, _, _ in MODEL_REGISTRY}
    model_weights  = {}
    for model_key, model_name, _ in MODEL_REGISTRY:
        if model_key not in models:
            print(f"\n  [SKIP] {model_name} — not loaded")
            continue
        try:
            weights_df = run_ce_model(
                model_key, model_name, models[model_key],
                X_cal, y_cal, X_test, y_test, features, config)
            model_weights[model_key] = weights_df
        except Exception as e:
            print(f"\n  [ERROR] {model_name} CE failed: {e}")
            model_weights[model_key] = None

    # ── Global CE aggregation ───────────────────────────────────────────────
    global_summary = compute_global_ce(model_weights, features, config)

    # ── Mondrian CE (sex-stratified) ────────────────────────────────────────
    run_mondrian_ce(models, X_cal, y_cal, X_test, y_test, features, config)

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = (time.time() - start_time) / 60
    print_ce_summary(global_summary, cal_quality_df, config)

    print(f"\n[TIME] Total CE pipeline time: {elapsed:.1f} minutes")
    print(f"[+] CE pipeline complete!")
    print("="*80)


if __name__ == '__main__':
    main()
