"""
Main XAI (Explainable AI) Pipeline (2018-2021 ENNS)

BO-TabNet (primary model):
  - SHAP KernelExplainer: global bar, beeswarm, waterfall (local)
  - LIME: local explanations for 1 at-risk + 1 normal sample

Tree models (XGBoost, RF, LightGBM, CatBoost):
  - SHAP TreeExplainer: global bar, beeswarm, dependency plots (top-5), waterfall
  - Combined importance comparison chart

LIME — Stacking Ensemble (secondary)
"""

import numpy as np
import pandas as pd
import joblib
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import shap
from lime import lime_tabular
from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIG
# =============================================================================

class Config:
    data_dir   = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\MAIN_PREPROCESS_OUTPUT")
    models_dir = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\MAIN_TRAINED_MODELS")
    output_dir = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\MAIN_XAI")

    # Tree models — use TreeExplainer (fast, exact)
    tree_models = ['xgboost', 'random_forest', 'lightgbm', 'catboost']

    shap_background_size = 100    # KernelExplainer background samples
    shap_kernel_samples  = 200    # test samples for KernelExplainer
    shap_nsamples        = 150    # Monte-Carlo samples per KernelExplainer call
    lime_num_features    = 15
    lime_num_samples     = 500

    # LIME targets: tree models to explain locally (stacking skipped — too large)
    lime_tree_models     = ['xgboost', 'lightgbm', 'catboost']
    skip_lime_stacking   = True   # set False to run LIME on 883 MB stacking model

    random_state = 42


# =============================================================================
# LOAD
# =============================================================================

def load_all(config):
    print("\n" + "="*70)
    print("LOADING DATA AND MODELS")
    print("="*70)

    X_train  = np.load(config.data_dir / 'X_train.npy')
    y_train  = np.load(config.data_dir / 'y_train.npy')
    X_test   = np.load(config.data_dir / 'X_test.npy')
    y_test   = np.load(config.data_dir / 'y_test.npy')
    features = pd.read_csv(config.data_dir / 'feature_list.csv')['feature_name'].tolist()

    print(f"  Train : {X_train.shape}  | Class 1: {y_train.mean()*100:.1f}%")
    print(f"  Test  : {X_test.shape}   | Class 1: {y_test.mean()*100:.1f}%")
    print(f"  Features ({len(features)}): {features}")

    # Load BO-TabNet from zip
    tab_path = config.models_dir / 'botabnet.zip'
    if tab_path.exists():
        tab_model = TabNetClassifier()
        tab_model.load_model(str(tab_path))
        print(f"  Loaded: botabnet  (from {tab_path.name})")
    else:
        tab_model = None
        print(f"  MISSING: botabnet.zip  (skipping TabNet XAI)")

    # Load tree models
    tree_models = {}
    for name in config.tree_models:
        path = config.models_dir / f'{name}.joblib'
        if path.exists():
            tree_models[name] = joblib.load(path)
            print(f"  Loaded: {name}")
        else:
            print(f"  MISSING: {name}.joblib  (skipping)")

    return X_train, y_train, X_test, y_test, features, tab_model, tree_models


# =============================================================================
# HELPERS
# =============================================================================

def pick_samples(X_test, y_test, n=1, random_state=42):
    rng = np.random.default_rng(random_state)
    at_risk_idx = np.where(y_test == 1)[0]
    normal_idx  = np.where(y_test == 0)[0]
    return (rng.choice(at_risk_idx, size=n, replace=False),
            rng.choice(normal_idx,  size=n, replace=False))


# =============================================================================
# SHAP — BO-TABNET (KernelExplainer)
# =============================================================================

def run_shap_botabnet(tab_model, X_train, X_test, y_test, features, config):
    print("\n" + "="*70)
    print("SHAP — BO-TABNET  (KernelExplainer)")
    print("="*70)
    out = config.output_dir / 'SHAP' / 'botabnet'
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.random_state)

    bg_idx     = rng.choice(len(X_train), size=config.shap_background_size, replace=False)
    background = X_train[bg_idx].astype(np.float32)

    te_idx   = rng.choice(len(X_test), size=min(config.shap_kernel_samples, len(X_test)),
                           replace=False)
    X_sample = X_test[te_idx].astype(np.float32)
    y_sample = y_test[te_idx]

    def predict_fn(X):
        return tab_model.predict_proba(X.astype(np.float32))

    print(f"  Building KernelExplainer (background={config.shap_background_size} samples)...")
    explainer = shap.KernelExplainer(predict_fn, background)

    print(f"  Computing SHAP values for {len(X_sample)} test samples "
          f"(nsamples={config.shap_nsamples})  — this may take a few minutes...")
    shap_values_all = explainer.shap_values(X_sample, nsamples=config.shap_nsamples,
                                             l1_reg='num_features(10)')

    sv = shap_values_all[1] if isinstance(shap_values_all, list) else shap_values_all
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    base_val = explainer.expected_value
    base_val = base_val[1] if isinstance(base_val, (list, np.ndarray)) else float(base_val)

    # ── 1. Global bar (mean |SHAP|) ───────────────────────────────────────
    mean_abs = np.abs(sv).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]
    top_n    = min(20, len(features))

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(range(top_n), mean_abs[order[:top_n]][::-1],
                   color='#8e44ad', alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([features[i] for i in order[:top_n]][::-1], fontsize=9)
    ax.set_xlabel('Mean |SHAP value|', fontsize=10)
    ax.set_title('BO-TabNet — Global Feature Importance (SHAP)\n2018-2021 ENNS',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar in bars:
        ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.4f}', va='center', fontsize=7)
    plt.tight_layout()
    plt.savefig(out / 'shap_bar_global.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: shap_bar_global.png")

    # ── 2. Beeswarm ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, feature_names=features,
                      show=False, max_display=20, plot_size=None)
    plt.title('BO-TabNet — SHAP Beeswarm (At-Risk class)\n2018-2021 ENNS',
              fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(out / 'shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: shap_beeswarm.png")

    # ── 3. Waterfall — 1 at-risk + 1 normal ──────────────────────────────
    rng2       = np.random.default_rng(config.random_state + 1)
    risk_local = np.where(y_sample == 1)[0]
    norm_local = np.where(y_sample == 0)[0]
    for label, local_pool in [('at_risk', risk_local), ('normal', norm_local)]:
        s_i = int(rng2.choice(local_pool))
        exp_obj = shap.Explanation(
            values=sv[s_i],
            base_values=base_val,
            data=X_sample[s_i],
            feature_names=features
        )
        fig, ax = plt.subplots(figsize=(9, 7))
        shap.waterfall_plot(exp_obj, max_display=15, show=False)
        prob     = tab_model.predict_proba(X_sample[s_i:s_i+1])[0, 1]
        true_lbl = 'At-Risk' if y_sample[s_i] == 1 else 'Normal'
        plt.title(
            f'BO-TabNet — SHAP Waterfall ({label.replace("_", " ").title()})\n'
            f'True: {true_lbl}  |  P(At-Risk)={prob:.3f}',
            fontsize=10, fontweight='bold', pad=10)
        plt.tight_layout()
        plt.savefig(out / f'shap_waterfall_{label}.png', dpi=150, bbox_inches='tight')
        plt.close()
    print(f"  Saved: waterfall plots (at-risk + normal)")

    # Save importance CSV
    pd.DataFrame({'feature': features, 'mean_abs_shap': mean_abs}) \
      .sort_values('mean_abs_shap', ascending=False) \
      .to_csv(out / 'botabnet_shap_importance.csv', index=False)
    print(f"  Saved: botabnet_shap_importance.csv")
    print(f"\n  Top-10 features (BO-TabNet SHAP):")
    for i in order[:10]:
        print(f"    {features[i]:<30s}  {mean_abs[i]:.5f}")

    return mean_abs, order


# =============================================================================
# SHAP — TREE EXPLAINER
# =============================================================================

def run_shap_tree(name, model, X_train, X_test, y_test, features, config):
    print(f"\n  [{name}] TreeExplainer SHAP")
    out = config.output_dir / 'SHAP' / name
    out.mkdir(parents=True, exist_ok=True)

    rng      = np.random.default_rng(config.random_state)
    idx      = rng.choice(len(X_test), size=min(1000, len(X_test)), replace=False)
    X_sample = X_test[idx]
    y_sample = y_test[idx]

    is_xgb = 'xgboost' in type(model).__module__.lower()
    if is_xgb:
        import xgboost as xgb
        booster  = model.get_booster()
        dmat     = xgb.DMatrix(X_sample)
        contribs = booster.predict(dmat, pred_contribs=True)
        sv           = contribs[:, :-1]
        base_contrib = float(contribs[0, -1])
        explainer    = None
    else:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        ev = explainer.expected_value
        base_contrib = ev[1] if isinstance(ev, (list, np.ndarray)) else float(ev)

    mean_abs = np.abs(sv).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]
    top_n    = min(20, len(features))

    # Global bar
    try:
        fig, ax = plt.subplots(figsize=(9, 7))
        bars = ax.barh(range(top_n), mean_abs[order[:top_n]][::-1],
                       color='#2980b9', alpha=0.85)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([features[i] for i in order[:top_n]][::-1], fontsize=9)
        ax.set_xlabel('Mean |SHAP value|', fontsize=10)
        ax.set_title(f'{name.replace("_"," ").title()} — Global Feature Importance (SHAP)',
                     fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for bar in bars:
            ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                    f'{bar.get_width():.4f}', va='center', fontsize=7)
        plt.tight_layout()
        plt.savefig(out / 'shap_bar_global.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: shap_bar_global.png")
    except Exception as e:
        print(f"    [warn] bar plot failed: {e}")
        plt.close('all')

    # Beeswarm
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_sample, feature_names=features,
                          show=False, max_display=20)
        plt.title(f'{name.replace("_"," ").title()} — SHAP Beeswarm (At-Risk class)',
                  fontsize=11, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(out / 'shap_beeswarm.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: shap_beeswarm.png")
    except Exception as e:
        print(f"    [warn] beeswarm failed: {e}")
        plt.close('all')

    # Dependency plots — top 5  (manual scatter; shap.dependence_plot removed in 0.41+)
    try:
        for feat_i in order[:5]:
            feat_name  = features[feat_i]
            feat_vals  = X_sample[:, feat_i]
            shap_col   = sv[:, feat_i]
            # colour by top interacting feature (highest |correlation| with shap_col)
            corr = np.array([abs(np.corrcoef(X_sample[:, j], shap_col)[0, 1])
                             if j != feat_i else -1
                             for j in range(X_sample.shape[1])])
            interact_i = int(np.argmax(corr))
            interact_vals = X_sample[:, interact_i]

            fig, ax = plt.subplots(figsize=(7, 5))
            sc = ax.scatter(feat_vals, shap_col, c=interact_vals,
                            cmap='coolwarm', alpha=0.4, s=10)
            plt.colorbar(sc, ax=ax, label=features[interact_i])
            ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
            ax.set_xlabel(feat_name, fontsize=10)
            ax.set_ylabel('SHAP value', fontsize=10)
            ax.set_title(
                f'{name.replace("_"," ").title()} — SHAP Dependency: {feat_name}',
                fontsize=10, fontweight='bold')
            plt.tight_layout()
            safe = feat_name.replace('/', '_').replace(' ', '_')
            plt.savefig(out / f'shap_dep_{safe}.png', dpi=150, bbox_inches='tight')
            plt.close()
        print(f"    Saved: dependency plots (top-5 features)")
    except Exception as e:
        print(f"    [warn] dependency plots failed: {e}")
        plt.close('all')

    # Waterfall — 1 at-risk + 1 normal
    try:
        risk_idx, norm_idx = pick_samples(X_test, y_test, n=1,
                                          random_state=config.random_state)
        for label, idx_arr in [('at_risk', risk_idx), ('normal', norm_idx)]:
            i = int(idx_arr[0])
            if is_xgb:
                import xgboost as xgb
                dmat_i    = xgb.DMatrix(X_test[i:i+1])
                contrib_i = model.get_booster().predict(dmat_i, pred_contribs=True)
                sv_vals   = contrib_i[0, :-1]
                base_v    = float(contrib_i[0, -1])
            else:
                single_sv = explainer.shap_values(X_test[i:i+1])
                if isinstance(single_sv, list):
                    sv_vals = single_sv[1][0]
                elif single_sv.ndim == 3:
                    sv_vals = single_sv[0, :, 1]
                elif single_sv.ndim == 2 and single_sv.shape[1] == 2:
                    sv_vals = single_sv[:, 1]
                else:
                    sv_vals = single_sv[0]
                base_v = base_contrib

            exp_obj = shap.Explanation(values=sv_vals, base_values=base_v,
                                       data=X_test[i], feature_names=features)
            fig, ax = plt.subplots(figsize=(9, 7))
            shap.waterfall_plot(exp_obj, max_display=15, show=False)
            plt.title(
                f'{name.replace("_"," ").title()} — Waterfall ({label.replace("_"," ").title()})',
                fontsize=10, fontweight='bold', pad=10)
            plt.tight_layout()
            plt.savefig(out / f'shap_waterfall_{label}.png', dpi=150, bbox_inches='tight')
            plt.close()
        print(f"    Saved: waterfall plots (at-risk + normal)")
    except Exception as e:
        print(f"    [warn] waterfall plots failed: {e}")
        plt.close('all')

    return mean_abs, order


# =============================================================================
# SHAP — COMBINED IMPORTANCE (all models)
# =============================================================================

def plot_combined_importance(importance_dict, features, config):
    df     = pd.DataFrame(importance_dict, index=features)
    top_n  = 15
    top_feats = df.mean(axis=1).sort_values(ascending=False).index[:top_n].tolist()
    df_top = df.loc[top_feats]

    colors = ['#8e44ad', '#e74c3c', '#2980b9', '#27ae60', '#f39c12']
    fig, ax = plt.subplots(figsize=(13, 7))
    x     = np.arange(top_n)
    width = 0.8 / len(df_top.columns)

    for k, (col, color) in enumerate(zip(df_top.columns, colors)):
        ax.bar(x + k * width, df_top[col].values,
               width, label=col.replace('_', ' ').title(), color=color, alpha=0.8)

    ax.set_xticks(x + width * (len(df_top.columns) - 1) / 2)
    ax.set_xticklabels(top_feats, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Mean |SHAP value|', fontsize=10)
    ax.set_title('SHAP Feature Importance — All Models, Top-15\n2018-2021 ENNS',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.output_dir / 'SHAP' / 'combined_feature_importance.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: combined_feature_importance.png")


# =============================================================================
# LIME
# =============================================================================

def run_lime(name, predict_proba_fn, X_train, X_test, y_test, features, config):
    print(f"\n  [{name}] LIME explanations")
    out = config.output_dir / 'LIME' / name
    out.mkdir(parents=True, exist_ok=True)

    explainer = lime_tabular.LimeTabularExplainer(
        training_data         = X_train,
        feature_names         = features,
        class_names           = ['Normal', 'At-Risk'],
        mode                  = 'classification',
        discretize_continuous = True,
        random_state          = config.random_state
    )

    risk_idx, norm_idx = pick_samples(X_test, y_test, n=1,
                                      random_state=config.random_state)

    for label, idx_arr in [('at_risk', risk_idx), ('normal', norm_idx)]:
        i   = int(idx_arr[0])
        exp = explainer.explain_instance(
            X_test[i],
            predict_proba_fn,
            num_features = config.lime_num_features,
            num_samples  = config.lime_num_samples,
            labels       = (1,)
        )

        contrib     = exp.as_list(label=1)
        feat_labels = [c[0] for c in contrib]
        feat_values = [c[1] for c in contrib]
        bar_colors  = ['#e74c3c' if v > 0 else '#2980b9' for v in feat_values]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(feat_labels)), feat_values[::-1],
                color=bar_colors[::-1], alpha=0.85)
        ax.set_yticks(range(len(feat_labels)))
        ax.set_yticklabels(feat_labels[::-1], fontsize=9)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('LIME Contribution  (positive = towards At-Risk)', fontsize=10)

        prob     = predict_proba_fn(X_test[i:i+1])[0, 1]
        true_lbl = 'At-Risk' if y_test[i] == 1 else 'Normal'
        ax.set_title(
            f'{name.replace("_"," ").title()} — LIME Local Explanation\n'
            f'Sample: {label.replace("_"," ").title()}  |  '
            f'True: {true_lbl}  |  P(At-Risk)={prob:.3f}',
            fontsize=10, fontweight='bold'
        )
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / f'lime_{label}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: lime_{label}.png  [P(at-risk)={prob:.3f}]")


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = Config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / 'SHAP').mkdir(parents=True, exist_ok=True)
    (config.output_dir / 'LIME').mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("MAIN XAI PIPELINE  (2018-2021 ENNS)  (SHAP + LIME)")
    print("="*70)

    start = time.time()

    X_train, y_train, X_test, y_test, features, tab_model, tree_models = load_all(config)

    importance_dict = {}

    # ── BO-TABNET SHAP (KernelExplainer) ────────────────────────────────
    if tab_model is not None:
        tab_imp, _ = run_shap_botabnet(
            tab_model, X_train, X_test, y_test, features, config)
        importance_dict['BO-TabNet'] = tab_imp
    else:
        print("\n[skip] BO-TabNet not loaded — skipping TabNet SHAP")

    # ── TREE MODEL SHAP (TreeExplainer) ──────────────────────────────────
    print("\n" + "="*70)
    print("SHAP — TREE MODELS")
    print("="*70)

    for name in config.tree_models:
        if name not in tree_models:
            print(f"  [skip] {name}: not loaded")
            continue
        mean_abs, _ = run_shap_tree(
            name, tree_models[name], X_train, X_test, y_test, features, config)
        importance_dict[name] = mean_abs

    # Combined importance chart
    if len(importance_dict) > 1:
        plot_combined_importance(importance_dict, features, config)

    # Global importance CSV
    imp_df = pd.DataFrame(importance_dict, index=features)
    imp_df['mean_across_models'] = imp_df.mean(axis=1)
    imp_df = imp_df.sort_values('mean_across_models', ascending=False)
    imp_df.to_csv(config.output_dir / 'SHAP' / 'global_importance.csv')
    print(f"\n  Saved: global_importance.csv")
    print(f"\n  Top-10 features (mean |SHAP| across all models):")
    print(imp_df['mean_across_models'].head(10).round(5).to_string())

    # ── LIME — BO-TABNET ──────────────────────────────────────────────────
    print("\n" + "="*70)
    print("LIME — BO-TABNET  (primary model)")
    print("="*70)

    if tab_model is not None:
        def tab_predict_proba(X):
            return tab_model.predict_proba(X.astype(np.float32))
        try:
            run_lime('botabnet', tab_predict_proba, X_train, X_test, y_test, features, config)
        except Exception as e:
            print(f"  [warn] LIME botabnet failed: {e}")

    # ── LIME — TREE MODELS ────────────────────────────────────────────────
    print("\n" + "="*70)
    print("LIME — TREE MODELS")
    print("="*70)

    for lime_name in config.lime_tree_models:
        if lime_name not in tree_models:
            print(f"  [skip] {lime_name}: not loaded")
            continue
        print(f"\n  [{lime_name}] LIME")
        try:
            run_lime(lime_name, tree_models[lime_name].predict_proba,
                     X_train, X_test, y_test, features, config)
        except Exception as e:
            print(f"  [warn] LIME {lime_name} failed: {e}")

    # ── LIME — STACKING ENSEMBLE (optional — large model) ─────────────────
    stacking_path = config.models_dir / 'stacking.joblib'
    if not config.skip_lime_stacking and stacking_path.exists():
        stacking = joblib.load(stacking_path)
        print("\n" + "="*70)
        print("LIME — STACKING ENSEMBLE  (best ROC-AUC model)")
        print("="*70)
        try:
            run_lime('stacking', stacking.predict_proba, X_train, X_test, y_test, features, config)
        except Exception as e:
            print(f"  [warn] LIME stacking failed: {e}")
    elif config.skip_lime_stacking:
        print("\n  [skip] LIME stacking — skipped (skip_lime_stacking=True; set False to enable)")

    elapsed = (time.time() - start) / 60
    print(f"\n[TIME] {elapsed:.1f} minutes")
    print(f"[+] Done!  Outputs -> {config.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
