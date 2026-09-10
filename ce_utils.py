"""
Shared utilities for the V2 CE pipeline scripts.

Imported by: ce_factual.py, ce_cce.py, ce_global.py, ce_mondrian.py

Contains: Config, MODEL_REGISTRY, data/model loaders, plot/narrative helpers,
          and feature-weight extractor.
"""

import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

    n_factual_plots   = 6    # per model: 3 at-risk + 3 normal (ce_factual.py)
    n_factual_global  = 150  # instances per model for global aggregation (ce_global.py)
    n_counterfactual  = 6    # per model: 3 at-risk + 3 normal (ce_cce.py)

    sex_col_idx       = 20   # feature index for 'sex' (see feature_list.csv)
    mondrian_n_sample = 50   # test instances per sex group (ce_mondrian.py)

    random_state = 42


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODEL_REGISTRY = [
    ('naive_bayes',   'Naive Bayes',           'joblib'),
    ('knn',           'kNN (Tuned)',            'joblib'),
    ('adaboost',      'AdaBoost (Tuned)',       'joblib'),
    ('xgboost',       'XGBoost (Tuned)',        'joblib'),
    ('random_forest', 'Random Forest (Tuned)',  'joblib'),
    ('catboost',      'CatBoost (Tuned)',        'joblib'),
    ('lightgbm',      'LightGBM (Tuned)',        'joblib'),
    ('stacking',      'Stacking Ensemble',       'joblib'),
    ('voting',        'Voting Ensemble',         'joblib'),
    # BO-TabNet excluded from CE: TabNet's load_model() does not restore sklearn's
    # fitted state in a way that CalibratedExplainer accepts. BO-TabNet performance
    # is evaluated via main_train.py and main_xai.py (Steps 2-3).
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(config):
    print("\n" + "="*70)
    print("LOADING DATA  (V2 Hybrid — 2018-2021 ENNS)")
    print("="*70)
    X_train = np.load(config.data_dir / 'X_train.npy')
    y_train = np.load(config.data_dir / 'y_train.npy')
    X_cal   = np.load(config.data_dir / 'X_val.npy')
    y_cal   = np.load(config.data_dir / 'y_val.npy')
    X_test  = np.load(config.data_dir / 'X_test.npy')
    y_test  = np.load(config.data_dir / 'y_test.npy')
    info    = joblib.load(config.data_dir / 'preprocessing_info.pkl')

    features = info.get('feature_names', [f'f{i}' for i in range(X_train.shape[1])])
    print(f"  Train : {X_train.shape}")
    print(f"  Cal   : {X_cal.shape}   (Venn-Abers calibration set)")
    print(f"  Test  : {X_test.shape}")
    print(f"  Features ({len(features)}): {list(features[:5])}...")
    return X_train, y_train, X_cal, y_cal, X_test, y_test, features


# =============================================================================
# MODEL LOADING  (one at a time — caller should del model after use)
# =============================================================================

def _load_single_model(model_key, model_name, fmt, config):
    """Load one model by key. Handles .joblib and TabNet .zip formats."""
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


# =============================================================================
# PLOT + NARRATIVE HELPERS
# =============================================================================

def _save_narrative(single_exp, plot_path):
    """Save to_narrative() text alongside the CE plot (.png → _narrative.txt)."""
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
    Save a CE explanation plot, working around the library's hardcoded
    'plots/' prefix on the filename argument.

    Strategy: chdir to target dir → call plot(filename=basename) →
    library writes to plots/<basename> relative to CWD → move to final path.
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
# FEATURE WEIGHT EXTRACTOR
# =============================================================================

def _extract_weights_from_explanation(factual_exp, features):
    """
    Extract feature weight point estimates from a FactualExplanation object.
    Uses get_rules() (stable in CE v1.0.0) with feature_weights as fallback.
    Returns dict {feature_name: weight_value} or None.
    """
    weights = {}

    # Method 1: get_rules() — stable in v1.0.0
    try:
        rules      = factual_exp.get_rules()
        feat_idxs  = rules.get('feature', [])
        weight_vals = rules.get('weight', [])
        explainer  = factual_exp.get_explainer()
        feat_names = list(getattr(explainer, 'feature_names', features))
        for fi, w in zip(feat_idxs, weight_vals):
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
