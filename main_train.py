"""
Main Model Training Pipeline (2018-2021 ENNS)

  - Data from MAIN_PREPROCESS_OUTPUT  (SMOTE-before-split, corr=0.80)
  - Primary metric: F2-score (β=2, recall-weighted)
  - BO-TabNet (Bayesian-Optimized via Optuna) as primary model
  - All standard ML models + Stacking + Voting ensembles
  - Outputs to MAIN_MODEL_RESULTS / MAIN_TRAINED_MODELS
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    fbeta_score, make_scorer
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier,
    StackingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# CONFIG
# =============================================================================

class Config:
    data_dir   = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\MAIN_PREPROCESS_OUTPUT")
    output_dir = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\MAIN_MODEL_RESULTS")
    models_dir = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\MAIN_TRAINED_MODELS")

    random_state  = 42
    n_jobs        = -1
    cv_folds      = 5
    n_iter        = 30   # RandomizedSearch + Optuna BO trials
    skip_tabnet   = False  # Set True to skip TabNet (loads prior metrics from CSV instead)
    tabnet_only   = True   # Set True to train ONLY TabNet (loads prior ML metrics from CSV)

    # Primary metric: F2 (recall-weighted, beta=2)
    f2_scorer           = make_scorer(fbeta_score, beta=2, zero_division=0)
    optimization_scorer = f2_scorer


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data(config):
    print("\n" + "="*80)
    print("LOADING PREPROCESSED DATA  (2018-2021 ENNS)")
    print("="*80)
    X_train = np.load(config.data_dir / 'X_train.npy')
    y_train = np.load(config.data_dir / 'y_train.npy')
    X_val   = np.load(config.data_dir / 'X_val.npy')
    y_val   = np.load(config.data_dir / 'y_val.npy')
    X_test  = np.load(config.data_dir / 'X_test.npy')
    y_test  = np.load(config.data_dir / 'y_test.npy')
    info    = joblib.load(config.data_dir / 'preprocessing_info.pkl')

    n0, n1 = (y_train == 0).sum(), (y_train == 1).sum()
    class_weights = {0: len(y_train) / (2 * n0), 1: len(y_train) / (2 * n1)}
    info['class_weights'] = class_weights

    print(f"  Train : {X_train.shape}  | Class 1: {y_train.mean()*100:.1f}%")
    print(f"  Val   : {X_val.shape}  | Class 1: {y_val.mean()*100:.1f}%")
    print(f"  Test  : {X_test.shape}  | Class 1: {y_test.mean()*100:.1f}%")
    features = info.get('feature_names', [f'f{i}' for i in range(X_train.shape[1])])
    print(f"  Features ({len(features)}): {features[:6]}...")
    return X_train, y_train, X_val, y_val, X_test, y_test, info


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def evaluate_model(model, X, y, model_name):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
    roc_auc = roc_auc_score(y, y_proba[:, 1]) if y_proba is not None else 0.0
    auprc   = average_precision_score(y, y_proba[:, 1]) if y_proba is not None else 0.0
    p_pc = precision_score(y, y_pred, average=None, zero_division=0)
    r_pc = recall_score(y, y_pred, average=None, zero_division=0)
    f_pc = f1_score(y, y_pred, average=None, zero_division=0)
    return {
        'model':             model_name,
        'accuracy':          accuracy_score(y, y_pred),
        'precision':         precision_score(y, y_pred, zero_division=0),
        'recall':            recall_score(y, y_pred, zero_division=0),
        'f1':                f1_score(y, y_pred, zero_division=0),
        'f2':                fbeta_score(y, y_pred, beta=2, zero_division=0),
        'roc_auc':           roc_auc,
        'auprc':             auprc,
        'precision_normal':  p_pc[0], 'recall_normal':  r_pc[0], 'f1_normal':  f_pc[0],
        'precision_at_risk': p_pc[1], 'recall_at_risk': r_pc[1], 'f1_at_risk': f_pc[1],
    }, y_pred, y_proba


def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'At-Risk'],
                yticklabels=['Normal', 'At-Risk'])
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name.replace(" ","_")}_confusion_matrix.png', dpi=150)
    plt.close()


def print_metrics(m):
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall   : {m['recall']:.4f}")
    print(f"  F1-Score : {m['f1']:.4f}")
    print(f"  F2-Score : {m['f2']:.4f}  (primary metric)")
    print(f"  ROC-AUC  : {m['roc_auc']:.4f}")
    print(f"  AUPRC    : {m['auprc']:.4f}")
    print(f"\n  Per-class metrics:")
    print(f"    Normal  - P: {m['precision_normal']:.4f}, R: {m['recall_normal']:.4f}, "
          f"F1: {m['f1_normal']:.4f}")
    print(f"    At-Risk - P: {m['precision_at_risk']:.4f}, R: {m['recall_at_risk']:.4f}, "
          f"F1: {m['f1_at_risk']:.4f}")


# =============================================================================
# [1/11] BO-TABNET  (Bayesian-Optimized via Optuna)
# =============================================================================

def train_botabnet(X_train, y_train, X_val, y_val, X_test, y_test,
                   feature_names, config):
    print("\n" + "="*80)
    print("[1/11] BO-TABNET  (Bayesian-Optimized via Optuna)")
    print("="*80)

    Xtr = X_train.astype(np.float32); ytr = y_train.astype(int)
    Xv  = X_val.astype(np.float32);   yv  = y_val.astype(int)
    Xte = X_test.astype(np.float32);  yte = y_test.astype(int)

    # Load existing model if already trained (skip costly Optuna re-run)
    saved_zip = config.models_dir / 'botabnet.zip'
    if saved_zip.exists():
        print(f"  [Skip] Found saved BO-TabNet — loading from {saved_zip}")
        model = TabNetClassifier()
        model.load_model(str(saved_zip))
        print("\n[*] Evaluating BO-TabNet on test set...")
        m, y_pred, _ = evaluate_model(model, Xte, yte, 'BO-TabNet')
        print_metrics(m)
        plot_confusion_matrix(yte, y_pred, 'BO-TabNet', config.output_dir)
        return model, m

    n_trials = config.n_iter
    print(f"  Bayesian Optimisation: {n_trials} trials  |  metric: F2 on val set")

    def objective(trial):
        n_d      = trial.suggest_int('n_d', 8, 64, step=4)
        n_steps  = trial.suggest_int('n_steps', 3, 10)
        gamma    = trial.suggest_float('gamma', 1.0, 2.0)
        lam      = trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True)
        lr       = trial.suggest_float('lr', 1e-4, 5e-2, log=True)
        bs       = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])
        momentum = trial.suggest_float('momentum', 0.01, 0.4)
        clf = TabNetClassifier(
            n_d=n_d, n_a=n_d, n_steps=n_steps, gamma=gamma,
            lambda_sparse=lam, momentum=momentum,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={'lr': lr},
            verbose=0, seed=config.random_state)
        clf.fit(Xtr, ytr, eval_set=[(Xv, yv)], eval_metric=['logloss'],
                max_epochs=100, patience=15,
                batch_size=bs, virtual_batch_size=max(16, bs // 4))
        return fbeta_score(yv, clf.predict(Xv), beta=2, zero_division=0)

    study = optuna.create_study(
        direction='maximize', sampler=TPESampler(seed=config.random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)

    bp = study.best_params
    print(f"\n  Best params (val F2={study.best_value:.4f}):")
    for k, v in bp.items():
        print(f"    {k}: {v}")

    print("\n  Training final BO-TabNet with best hyperparameters...")
    model = TabNetClassifier(
        n_d=bp['n_d'], n_a=bp['n_d'], n_steps=bp['n_steps'],
        gamma=bp['gamma'], lambda_sparse=bp['lambda_sparse'],
        momentum=bp['momentum'],
        optimizer_fn=torch.optim.Adam,
        optimizer_params={'lr': bp['lr']},
        verbose=5, seed=config.random_state)
    model.fit(Xtr, ytr, eval_set=[(Xv, yv)], eval_metric=['logloss'],
              max_epochs=200, patience=20,
              batch_size=bp['batch_size'],
              virtual_batch_size=max(16, bp['batch_size'] // 4))

    print("\n[*] Evaluating BO-TabNet on test set...")
    m, y_pred, _ = evaluate_model(model, Xte, yte, 'BO-TabNet')
    print_metrics(m)
    plot_confusion_matrix(yte, y_pred, 'BO-TabNet', config.output_dir)

    fi = model.feature_importances_
    n  = min(len(feature_names), len(fi))
    pd.DataFrame({'feature': feature_names[:n], 'importance': fi[:n]}) \
      .sort_values('importance', ascending=False) \
      .to_csv(config.output_dir / 'botabnet_feature_importance.csv', index=False)
    print("  Feature importance saved.")
    return model, m


# =============================================================================
# TRADITIONAL ML MODELS
# =============================================================================

def train_naive_bayes(X_train, y_train, X_test, y_test, config):
    print("\n" + "="*80); print("[2/11] TRAINING NAIVE BAYES"); print("="*80)
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("\n[*] Evaluating Naive Bayes on test set...")
    m, y_pred, _ = evaluate_model(model, X_test, y_test, "Naive Bayes")
    print_metrics(m); plot_confusion_matrix(y_test, y_pred, "Naive Bayes", config.output_dir)
    return model, m


def train_knn(X_train, y_train, X_test, y_test, config):
    print("\n" + "="*80); print("[3/11] TUNING K-NEAREST NEIGHBORS"); print("="*80)
    search = RandomizedSearchCV(
        KNeighborsClassifier(),
        {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'],
         'metric': ['euclidean', 'manhattan']},
        n_iter=15, cv=config.cv_folds, scoring=config.optimization_scorer,
        random_state=config.random_state, n_jobs=config.n_jobs, verbose=1)
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}  | CV F2: {search.best_score_:.4f}")
    model = search.best_estimator_
    print("\n[*] Evaluating kNN on test set...")
    m, y_pred, _ = evaluate_model(model, X_test, y_test, "kNN (Tuned)")
    print_metrics(m); plot_confusion_matrix(y_test, y_pred, "kNN (Tuned)", config.output_dir)
    return model, m


def train_adaboost(X_train, y_train, X_test, y_test, config):
    print("\n" + "="*80); print("[4/11] TUNING ADABOOST"); print("="*80)
    search = RandomizedSearchCV(
        AdaBoostClassifier(algorithm='SAMME', random_state=config.random_state),
        {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5, 1.0]},
        n_iter=10, cv=config.cv_folds, scoring=config.optimization_scorer,
        random_state=config.random_state, n_jobs=config.n_jobs, verbose=1)
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}  | CV F2: {search.best_score_:.4f}")
    model = search.best_estimator_
    print("\n[*] Evaluating AdaBoost on test set...")
    m, y_pred, _ = evaluate_model(model, X_test, y_test, "AdaBoost (Tuned)")
    print_metrics(m); plot_confusion_matrix(y_test, y_pred, "AdaBoost", config.output_dir)
    return model, m


def train_xgboost(X_train, y_train, X_test, y_test, class_weights, config):
    print("\n" + "="*80); print("[5/11] TUNING XGBOOST"); print("="*80)
    scale_pos = class_weights[1] / class_weights[0]
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'scale_pos_weight': [scale_pos],
    }
    search = RandomizedSearchCV(
        XGBClassifier(eval_metric='logloss',
                      random_state=config.random_state, n_jobs=config.n_jobs),
        param_dist, n_iter=config.n_iter, cv=config.cv_folds,
        scoring=config.optimization_scorer,
        random_state=config.random_state, n_jobs=config.n_jobs, verbose=1)
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}  | CV F2: {search.best_score_:.4f}")
    model = search.best_estimator_
    print("\n[*] Evaluating XGBoost on test set...")
    m, y_pred, _ = evaluate_model(model, X_test, y_test, "XGBoost (Tuned)")
    print_metrics(m); plot_confusion_matrix(y_test, y_pred, "XGBoost", config.output_dir)
    return model, m


def train_random_forest(X_train, y_train, X_test, y_test, config):
    print("\n" + "="*80); print("[6/11] TUNING RANDOM FOREST"); print("="*80)
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample'],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=config.random_state, n_jobs=config.n_jobs),
        param_dist, n_iter=config.n_iter, cv=config.cv_folds,
        scoring=config.optimization_scorer,
        random_state=config.random_state, n_jobs=config.n_jobs, verbose=1)
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}  | CV F2: {search.best_score_:.4f}")
    model = search.best_estimator_
    print("\n[*] Evaluating Random Forest on test set...")
    m, y_pred, _ = evaluate_model(model, X_test, y_test, "Random Forest (Tuned)")
    print_metrics(m); plot_confusion_matrix(y_test, y_pred, "Random Forest", config.output_dir)
    return model, m


def train_catboost(X_train, y_train, X_test, y_test, class_weights, config):
    print("\n" + "="*80); print("[7/11] TUNING CATBOOST"); print("="*80)
    cw = [class_weights[0], class_weights[1]]
    param_dist = {
        'iterations':    [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'depth':         [4, 6, 8, 10],
        'l2_leaf_reg':   [1, 3, 5, 7, 9],
        'bagging_temperature': [0.0, 0.5, 1.0],
    }
    search = RandomizedSearchCV(
        CatBoostClassifier(
            auto_class_weights='Balanced', random_state=config.random_state,
            verbose=0, eval_metric='F:beta=2'),
        param_dist, n_iter=config.n_iter, cv=config.cv_folds,
        scoring=config.optimization_scorer,
        random_state=config.random_state, n_jobs=config.n_jobs, verbose=1)
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}  | CV F2: {search.best_score_:.4f}")
    model = search.best_estimator_
    print("\n[*] Evaluating CatBoost on test set...")
    m, y_pred, _ = evaluate_model(model, X_test, y_test, "CatBoost (Tuned)")
    print_metrics(m); plot_confusion_matrix(y_test, y_pred, "CatBoost", config.output_dir)
    return model, m


def train_lightgbm(X_train, y_train, X_test, y_test, class_weights, config):
    print("\n" + "="*80); print("[8/11] TUNING LIGHTGBM"); print("="*80)
    param_dist = {
        'n_estimators':  [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves':    [15, 31, 63, 127],
        'max_depth':     [-1, 5, 10, 15],
        'min_child_samples': [10, 20, 30, 50],
        'subsample':     [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    }
    search = RandomizedSearchCV(
        LGBMClassifier(
            class_weight='balanced', random_state=config.random_state,
            n_jobs=config.n_jobs, verbose=-1),
        param_dist, n_iter=config.n_iter, cv=config.cv_folds,
        scoring=config.optimization_scorer,
        random_state=config.random_state, n_jobs=1, verbose=1)
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}  | CV F2: {search.best_score_:.4f}")
    model = search.best_estimator_
    print("\n[*] Evaluating LightGBM on test set...")
    m, y_pred, _ = evaluate_model(model, X_test, y_test, "LightGBM (Tuned)")
    print_metrics(m); plot_confusion_matrix(y_test, y_pred, "LightGBM", config.output_dir)
    return model, m


def train_stacking(rf, xgb, lgbm, X_train, y_train, X_test, y_test, config):
    print("\n" + "="*80); print("[9/11] CREATING STACKING ENSEMBLE"); print("="*80)
    estimators = [('rf', rf), ('xgb', xgb), ('lgbm', lgbm)]
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=config.random_state),
        cv=config.cv_folds, n_jobs=config.n_jobs, passthrough=False)
    print("  Fitting stacking ensemble (RF + XGB + LightGBM)...")
    model.fit(X_train, y_train)
    print("\n[*] Evaluating Stacking on test set...")
    m, y_pred, _ = evaluate_model(model, X_test, y_test, "Stacking Ensemble")
    print_metrics(m); plot_confusion_matrix(y_test, y_pred, "Stacking", config.output_dir)
    return model, m


def train_voting(rf, xgb, lgbm, X_train, y_train, X_test, y_test, config):
    print("\n" + "="*80); print("[10/11] CREATING VOTING ENSEMBLE"); print("="*80)
    model = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)],
        voting='soft', n_jobs=config.n_jobs)
    print("  Fitting voting ensemble (RF + XGB + LightGBM)...")
    model.fit(X_train, y_train)
    print("\n[*] Evaluating Voting Ensemble on test set...")
    m, y_pred, _ = evaluate_model(model, X_test, y_test, "Voting Ensemble")
    print_metrics(m); plot_confusion_matrix(y_test, y_pred, "Voting", config.output_dir)
    return model, m


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = Config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("MAIN MODEL TRAINING  (2018-2021 ENNS)")
    print("Primary metric: F2-score (beta=2)  |  BO-TabNet + 9 traditional models")
    print("="*80)

    start_time = time.time()

    X_train, y_train, X_val, y_val, X_test, y_test, info = load_data(config)
    class_weights = info['class_weights']
    feature_names = info.get('feature_names', [])

    all_metrics, all_models = [], {}

    # ── MODE: tabnet_only — train only BO-TabNet, load prior ML metrics ────────
    if config.tabnet_only:
        print("\n" + "="*80)
        print("MODE: TABNET-ONLY  (traditional ML loaded from prior CSV)")
        print("="*80)
        # Load prior traditional ML metrics
        prior_csv = config.output_dir / 'model_results.csv'
        if prior_csv.exists():
            prior = pd.read_csv(prior_csv)
            ml_rows = prior[~prior['model'].str.contains('TabNet', case=False)]
            for _, row in ml_rows.iterrows():
                all_metrics.append(row.to_dict())
            print(f"  Loaded {len(ml_rows)} prior traditional ML rows from CSV")
        else:
            print("  [warn] No prior results CSV — traditional ML rows will be missing")
        # Train BO-TabNet
        tab_model, tab_m = train_botabnet(
            X_train, y_train, X_val, y_val, X_test, y_test, feature_names, config)
        all_metrics.append(tab_m)
        tab_path = str(config.models_dir / 'botabnet')
        tab_model.save_model(tab_path)
        print(f"  BO-TabNet saved: {tab_path}.zip")

    else:
        # ── 1. BO-TabNet (normal mode) ────────────────────────────────────────
        if config.skip_tabnet:
            print("\n" + "="*80)
            print("[1/11] BO-TABNET  (skipped — loading prior metrics from CSV)")
            print("="*80)
            prior_csv = config.output_dir / 'model_results.csv'
            if prior_csv.exists():
                prior = pd.read_csv(prior_csv)
                tab_rows = prior[prior['model'].str.contains('TabNet', case=False)]
                if not tab_rows.empty:
                    tab_m = tab_rows.iloc[0].to_dict()
                    all_metrics.append(tab_m)
                    print(f"  Loaded prior TabNet metrics (F2={tab_m.get('f2', 'N/A'):.4f})")
                else:
                    print("  No prior TabNet row found in CSV — skipping entirely.")
            else:
                print("  No prior results CSV found — skipping entirely.")
        else:
            tab_model, tab_m = train_botabnet(
                X_train, y_train, X_val, y_val, X_test, y_test, feature_names, config)
            all_metrics.append(tab_m)
            tab_path = str(config.models_dir / 'botabnet')
            tab_model.save_model(tab_path)
            print(f"  BO-TabNet saved: {tab_path}.zip")

        # ── 2–10. Traditional ML models ───────────────────────────────────────
        nb_model,  nb_m  = train_naive_bayes(X_train, y_train, X_test, y_test, config)
        knn_model, knn_m = train_knn(X_train, y_train, X_test, y_test, config)
        ada_model, ada_m = train_adaboost(X_train, y_train, X_test, y_test, config)
        xgb_model, xgb_m = train_xgboost(X_train, y_train, X_test, y_test, class_weights, config)
        rf_model,  rf_m  = train_random_forest(X_train, y_train, X_test, y_test, config)
        cat_model, cat_m = train_catboost(X_train, y_train, X_test, y_test, class_weights, config)
        lgbm_model, lgbm_m = train_lightgbm(X_train, y_train, X_test, y_test, class_weights, config)
        stack_model, stack_m = train_stacking(
            rf_model, xgb_model, lgbm_model, X_train, y_train, X_test, y_test, config)
        vote_model, vote_m = train_voting(
            rf_model, xgb_model, lgbm_model, X_train, y_train, X_test, y_test, config)

        for m in [nb_m, knn_m, ada_m, xgb_m, rf_m, cat_m, lgbm_m, stack_m, vote_m]:
            all_metrics.append(m)
        for name, model in [('naive_bayes', nb_model), ('knn', knn_model),
                            ('adaboost', ada_model), ('xgboost', xgb_model),
                            ('random_forest', rf_model), ('catboost', cat_model),
                            ('lightgbm', lgbm_model), ('stacking', stack_model),
                            ('voting', vote_model)]:
            joblib.dump(model, config.models_dir / f'{name}.joblib')
            all_models[name] = model

    # Results summary — sorted by F2 (primary metric)
    results_df = pd.DataFrame(all_metrics).sort_values('f2', ascending=False)

    pos_rate = y_test.mean()
    print("\n" + "="*80)
    print("RESULTS SUMMARY  (sorted by F2-score)")
    print("="*80)
    cols = ['model', 'precision', 'recall', 'f2', 'auprc', 'f1', 'roc_auc', 'accuracy']
    print(results_df[cols].to_string(index=False))
    print(f"\n  --- Naive baseline: P={pos_rate:.3f}  F1={2*pos_rate/(1+pos_rate):.3f}"
          f"  AUC=0.500 ---")

    best = results_df.iloc[0]
    print("\n" + "="*80)
    print("BEST MODEL (BY F2-SCORE)")
    print("="*80)
    print(f"  Model     : {best['model']}")
    print(f"  F2-Score  : {best['f2']:.4f}  (primary metric — recall-weighted)")
    print(f"  AUPRC     : {best['auprc']:.4f}  (baseline={pos_rate:.3f})")
    print(f"  ROC-AUC   : {best['roc_auc']:.4f}")
    print(f"  F1-Score  : {best['f1']:.4f}")
    print(f"  Precision : {best['precision']:.4f} ({best['precision']*100:.1f}%)")
    print(f"  Recall    : {best['recall']:.4f} ({best['recall']*100:.1f}%)")
    print(f"  Accuracy  : {best['accuracy']:.4f} ({best['accuracy']*100:.1f}%)")

    elapsed = (time.time() - start_time) / 60
    print(f"\n[TIME] Total training time: {elapsed:.1f} minutes")

    results_df.to_csv(config.output_dir / 'model_results.csv', index=False)
    print(f"\n[+] Training complete!")
    print(f"   Results: {config.output_dir / 'model_results.csv'}")
    print(f"   Models : {config.models_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
