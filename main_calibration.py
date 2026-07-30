"""
Main Model Calibration Pipeline (2018-2021 ENNS)
Applies Platt scaling (sigmoid), isotonic regression, and temperature scaling.
Evaluates reliability, Brier score, log-loss, and AUC before/after calibration.
Handles BO-TabNet via a float32-safe sklearn wrapper.
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

from scipy.optimize import minimize_scalar
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    brier_score_loss, roc_auc_score, log_loss,
    average_precision_score, f1_score, fbeta_score,
    precision_score, recall_score
)
from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIG
# =============================================================================

class Config:
    data_dir         = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\MAIN_PREPROCESS_OUTPUT")
    models_dir       = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\MAIN_TRAINED_MODELS")
    output_dir       = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\MAIN_CALIBRATION")
    calib_models_dir = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\MAIN_TRAINED_MODELS_CALIB")

    models_to_calibrate = [
        'botabnet',
        'xgboost', 'random_forest', 'lightgbm', 'catboost',
        'adaboost', 'naive_bayes', 'stacking', 'voting',
    ]

    n_bins       = 10
    random_state = 42


# =============================================================================
# BO-TABNET SKLEARN WRAPPER
# =============================================================================

class TabNetSklearnWrapper:
    """Makes TabNetClassifier compatible with CalibratedClassifierCV (cv='prefit')."""
    _estimator_type = "classifier"

    def __init__(self, tab_model):
        self.tab_model = tab_model
        self.classes_  = np.array([0, 1])

    def predict_proba(self, X):
        return self.tab_model.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X):
        return self.tab_model.predict(np.asarray(X, dtype=np.float32))

    def fit(self, X, y):
        return self


# =============================================================================
# LOAD
# =============================================================================

def load_data(config):
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    X_train  = np.load(config.data_dir / 'X_train.npy')
    y_train  = np.load(config.data_dir / 'y_train.npy')
    X_val    = np.load(config.data_dir / 'X_val.npy')
    y_val    = np.load(config.data_dir / 'y_val.npy')
    X_test   = np.load(config.data_dir / 'X_test.npy')
    y_test   = np.load(config.data_dir / 'y_test.npy')
    features = pd.read_csv(config.data_dir / 'feature_list.csv')['feature_name'].tolist()
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"  Features: {len(features)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, features


def load_models(config):
    print("\n" + "="*70)
    print("LOADING TRAINED MODELS")
    print("="*70)
    models = {}
    for name in config.models_to_calibrate:
        if name == 'botabnet':
            path = config.models_dir / 'botabnet.zip'
            if path.exists():
                raw = TabNetClassifier()
                raw.load_model(str(path))
                models['botabnet'] = TabNetSklearnWrapper(raw)
                print(f"  Loaded: botabnet  (TabNetSklearnWrapper)")
            else:
                print(f"  MISSING: botabnet.zip  (skipping)")
        else:
            path = config.models_dir / f'{name}.joblib'
            if path.exists():
                models[name] = joblib.load(path)
                print(f"  Loaded: {name}")
            else:
                print(f"  MISSING: {name}.joblib  (skipping)")
    return models


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(name, model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred  = model.predict(X)
    return {
        'model':     name,
        'roc_auc':   roc_auc_score(y, proba),
        'auprc':     average_precision_score(y, proba),
        'brier':     brier_score_loss(y, proba),
        'log_loss':  log_loss(y, proba),
        'f2':        fbeta_score(y, pred, beta=2, zero_division=0),
        'f1':        f1_score(y, pred, zero_division=0),
        'precision': precision_score(y, pred, zero_division=0),
        'recall':    recall_score(y, pred, zero_division=0),
    }


# =============================================================================
# TEMPERATURE SCALING
# =============================================================================

class TemperatureScaledModel:
    """P_calibrated = sigmoid(logit(P_raw) / T).  Minimises NLL on val set."""
    def __init__(self, base_model, temperature):
        self.base_model  = base_model
        self.temperature = float(temperature)

    def predict_proba(self, X):
        p     = np.clip(self.base_model.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
        logit = np.log(p / (1 - p))
        p_cal = 1.0 / (1.0 + np.exp(-logit / self.temperature))
        return np.column_stack([1 - p_cal, p_cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def find_temperature(model, X_val, y_val):
    p     = np.clip(model.predict_proba(X_val)[:, 1], 1e-7, 1 - 1e-7)
    logit = np.log(p / (1 - p))

    def nll(T):
        p_cal = np.clip(1.0 / (1.0 + np.exp(-logit / T)), 1e-7, 1 - 1e-7)
        return -np.mean(y_val * np.log(p_cal) + (1 - y_val) * np.log(1 - p_cal))

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method='bounded')
    return res.x, res.fun


# =============================================================================
# CALIBRATION
# =============================================================================

def calibrate_model(model, X_val, y_val, method='sigmoid'):
    calib = CalibratedClassifierCV(estimator=model, cv='prefit', method=method)
    calib.fit(X_val, y_val)
    return calib


# =============================================================================
# PLOTS
# =============================================================================

def plot_reliability_diagram(ax, y_true, proba, label, n_bins=10, color='steelblue'):
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy='uniform')
    ax.plot(mean_pred, frac_pos, 's-', color=color, label=label,
            linewidth=1.5, markersize=5)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6, label='Perfect')
    ax.set_xlabel('Mean predicted probability', fontsize=9)
    ax.set_ylabel('Fraction of positives', fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_all_reliability(models_raw, models_sig, models_iso, X_test, y_test, config):
    names = list(models_raw.keys())
    n     = len(names)
    fig, axes = plt.subplots(n, 3, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle('Reliability Diagrams — Pre vs Post Calibration\n2018-2021 ENNS',
                 fontsize=13, fontweight='bold', y=1.01)

    for ax_col, title in zip(axes[0], ['Uncalibrated', 'Platt (Sigmoid)', 'Isotonic']):
        ax_col.set_title(title, fontsize=11, fontweight='bold')

    colors = ['#e74c3c', '#2980b9', '#27ae60']
    for i, name in enumerate(names):
        for j, (mdict, color, lbl) in enumerate([
            (models_raw, colors[0], 'Uncalibrated'),
            (models_sig, colors[1], 'Platt'),
            (models_iso, colors[2], 'Isotonic'),
        ]):
            if name in mdict:
                proba = mdict[name].predict_proba(X_test)[:, 1]
                plot_reliability_diagram(axes[i, j], y_test, proba,
                                         label=f'{name} ({lbl})',
                                         n_bins=config.n_bins, color=color)
                axes[i, j].set_title(f'{name.replace("_"," ").title()} — {lbl}', fontsize=9)

    plt.tight_layout()
    plt.savefig(config.output_dir / 'reliability_diagrams_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: reliability_diagrams_all.png")


def plot_logloss_comparison(df_before, df_sig, df_iso, df_temp, baseline_ll, config):
    models = df_before['model'].tolist()
    x, w   = np.arange(len(models)), 0.18
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 1.5*w, df_before['log_loss'], w, label='Uncalibrated',        color='#e74c3c', alpha=0.85)
    ax.bar(x - 0.5*w, df_sig['log_loss'],    w, label='Platt (Sigmoid)',     color='#2980b9', alpha=0.85)
    ax.bar(x + 0.5*w, df_iso['log_loss'],    w, label='Isotonic',            color='#27ae60', alpha=0.85)
    ax.bar(x + 1.5*w, df_temp['log_loss'],   w, label='Temperature Scaling', color='#8e44ad', alpha=0.85)
    ax.axhline(baseline_ll, color='black', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'Prior baseline ({baseline_ll:.3f})')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=9)
    ax.set_ylabel('Log-Loss (lower = better)', fontsize=10)
    ax.set_title('Log-Loss: All Calibration Methods — 2018-2021 ENNS',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.output_dir / 'logloss_all_methods.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: logloss_all_methods.png")


def plot_brier_comparison(df_before, df_sig, df_iso, config):
    models = df_before['model'].tolist()
    x, w   = np.arange(len(models)), 0.25
    fig, ax = plt.subplots(figsize=(13, 5))
    b1 = ax.bar(x - w, df_before['brier'], w, label='Uncalibrated',        color='#e74c3c', alpha=0.85)
    b2 = ax.bar(x,     df_sig['brier'],    w, label='Platt (Sigmoid)',     color='#2980b9', alpha=0.85)
    b3 = ax.bar(x + w, df_iso['brier'],    w, label='Isotonic Regression', color='#27ae60', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=9)
    ax.set_ylabel('Brier Score (lower = better)', fontsize=10)
    ax.set_title('Brier Score: Calibration Comparison — 2018-2021 ENNS',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig(config.output_dir / 'brier_score_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: brier_score_comparison.png")


def plot_auc_comparison(df_before, df_sig, df_iso, config):
    models = df_before['model'].tolist()
    x, w   = np.arange(len(models)), 0.25
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w, df_before['roc_auc'], w, label='Uncalibrated',        color='#e74c3c', alpha=0.85)
    ax.bar(x,     df_sig['roc_auc'],    w, label='Platt (Sigmoid)',     color='#2980b9', alpha=0.85)
    ax.bar(x + w, df_iso['roc_auc'],    w, label='Isotonic Regression', color='#27ae60', alpha=0.85)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Random (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=9)
    ax.set_ylabel('ROC-AUC', fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.set_title('ROC-AUC: Calibration Comparison — 2018-2021 ENNS',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.output_dir / 'roc_auc_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: roc_auc_comparison.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = Config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.calib_models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("MAIN MODEL CALIBRATION  (2018-2021 ENNS)")
    print("Methods: Platt (Sigmoid) | Isotonic | Temperature Scaling")
    print("="*70)

    start = time.time()

    X_train, y_train, X_val, y_val, X_test, y_test, features = load_data(config)
    models_raw = load_models(config)

    rows_before, rows_sig, rows_iso, rows_temp = [], [], [], []
    models_sig, models_iso, models_temp = {}, {}, {}

    print("\n" + "="*70)
    print("CALIBRATING MODELS")
    print("="*70)

    for name, model in models_raw.items():
        print(f"\n  [{name}]")

        m_before = compute_metrics(name, model, X_test, y_test)
        rows_before.append(m_before)
        print(f"    Before   -> AUC={m_before['roc_auc']:.4f}  "
              f"Brier={m_before['brier']:.4f}  "
              f"LogLoss={m_before['log_loss']:.4f}  "
              f"F2={m_before['f2']:.4f}")

        m_sig = calibrate_model(model, X_val, y_val, method='sigmoid')
        models_sig[name] = m_sig
        r_sig = compute_metrics(f'{name}_sigmoid', m_sig, X_test, y_test)
        rows_sig.append(r_sig)
        print(f"    Sigmoid  -> AUC={r_sig['roc_auc']:.4f}  "
              f"Brier={r_sig['brier']:.4f}  "
              f"LogLoss={r_sig['log_loss']:.4f}  "
              f"[Brier d: {r_sig['brier']-m_before['brier']:+.4f}]")

        m_iso = calibrate_model(model, X_val, y_val, method='isotonic')
        models_iso[name] = m_iso
        r_iso = compute_metrics(f'{name}_isotonic', m_iso, X_test, y_test)
        rows_iso.append(r_iso)
        print(f"    Isotonic -> AUC={r_iso['roc_auc']:.4f}  "
              f"Brier={r_iso['brier']:.4f}  "
              f"LogLoss={r_iso['log_loss']:.4f}  "
              f"[Brier d: {r_iso['brier']-m_before['brier']:+.4f}]")

        T_opt, val_nll = find_temperature(model, X_val, y_val)
        m_temp = TemperatureScaledModel(model, T_opt)
        models_temp[name] = m_temp
        r_temp = compute_metrics(f'{name}_temp', m_temp, X_test, y_test)
        rows_temp.append(r_temp)
        print(f"    TempScal -> AUC={r_temp['roc_auc']:.4f}  "
              f"Brier={r_temp['brier']:.4f}  "
              f"LogLoss={r_temp['log_loss']:.4f}  "
              f"T*={T_opt:.4f}  "
              f"[LL d: {r_temp['log_loss']-m_before['log_loss']:+.4f}]")

        lls = {'sigmoid': r_sig['log_loss'],
               'isotonic': r_iso['log_loss'],
               'temperature': r_temp['log_loss']}
        best_method = min(lls, key=lls.get)
        best_calib  = {'sigmoid': m_sig, 'isotonic': m_iso, 'temperature': m_temp}[best_method]
        save_path   = config.calib_models_dir / f'{name}_calib.joblib'
        joblib.dump(best_calib, save_path)
        print(f"    Saved best ({best_method}) -> {save_path.name}")

    df_before  = pd.DataFrame(rows_before)
    df_sig     = pd.DataFrame(rows_sig)
    df_iso     = pd.DataFrame(rows_iso)
    df_temp    = pd.DataFrame(rows_temp)

    df_sig_al  = df_sig.copy();  df_sig_al['model']  = df_before['model'].values
    df_iso_al  = df_iso.copy();  df_iso_al['model']  = df_before['model'].values
    df_temp_al = df_temp.copy(); df_temp_al['model'] = df_before['model'].values

    baseline_ll = log_loss(y_test, np.full(len(y_test), y_test.mean()))
    print(f"\n  Prior baseline log-loss: {baseline_ll:.4f}")

    print("\n" + "="*70)
    print("CALIBRATION RESULTS SUMMARY  (TEST SET)")
    print("="*70)
    compare = pd.DataFrame({
        'model':            df_before['model'],
        'auc_raw':          df_before['roc_auc'],
        'auc_sigmoid':      df_sig['roc_auc'].values,
        'auc_isotonic':     df_iso['roc_auc'].values,
        'auc_temp':         df_temp['roc_auc'].values,
        'brier_raw':        df_before['brier'],
        'brier_sigmoid':    df_sig['brier'].values,
        'brier_isotonic':   df_iso['brier'].values,
        'brier_temp':       df_temp['brier'].values,
        'logloss_raw':      df_before['log_loss'],
        'logloss_sigmoid':  df_sig['log_loss'].values,
        'logloss_isotonic': df_iso['log_loss'].values,
        'logloss_temp':     df_temp['log_loss'].values,
        'f2_raw':           df_before['f2'],
    })
    print(compare.to_string(index=False))
    compare.to_csv(config.output_dir / 'calibration_comparison.csv', index=False)
    print(f"\n  Saved: calibration_comparison.csv")

    print("\n" + "="*70)
    print(f"BEST LOG-LOSS PER MODEL  (prior baseline = {baseline_ll:.4f})")
    print("="*70)
    for i, name in enumerate(df_before['model']):
        lls = {
            'raw':         df_before['log_loss'].iloc[i],
            'sigmoid':     df_sig['log_loss'].iloc[i],
            'isotonic':    df_iso['log_loss'].iloc[i],
            'temperature': df_temp['log_loss'].iloc[i],
        }
        best        = min(lls, key=lls.get)
        improvement = (baseline_ll - lls[best]) / baseline_ll * 100
        print(f"  {name:22s}  best={lls[best]:.4f} ({best:<12s})  "
              f"vs baseline: {improvement:.1f}% improvement")

    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    plot_all_reliability(models_raw, models_sig, models_iso, X_test, y_test, config)
    plot_brier_comparison(df_before, df_sig_al, df_iso_al, config)
    plot_auc_comparison(df_before, df_sig_al, df_iso_al, config)
    plot_logloss_comparison(df_before, df_sig_al, df_iso_al, df_temp_al, baseline_ll, config)

    elapsed = (time.time() - start) / 60
    print(f"\n[TIME] {elapsed:.1f} minutes")
    print(f"[+] Done!  Outputs         -> {config.output_dir}")
    print(f"[+] Calibrated models      -> {config.calib_models_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
