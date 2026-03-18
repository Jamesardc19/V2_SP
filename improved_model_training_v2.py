"""
Improved Model Training V2 - Works with properly preprocessed data
Uses data from improved_preprocessing_v2.py (feature selection before encoding)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb


class Config:
    """Configuration for improved model training V2"""
    preprocess_dir = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\PREPROCESS_OUTPUT_IMPROVED_V2")
    results_dir = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\Model_Results_Improved_V2")
    models_dir = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\Trained_Models_Improved_V2")
    
    random_state = 42
    n_iter_search = 50
    cv_folds = 5


CFG = Config()
CFG.results_dir.mkdir(parents=True, exist_ok=True)
CFG.models_dir.mkdir(parents=True, exist_ok=True)


def load_preprocessed_data():
    """Load preprocessed data from V2 preprocessing"""
    print("Loading preprocessed data...")
    
    X_train = np.load(CFG.preprocess_dir / "X_train.npy")
    y_train = np.load(CFG.preprocess_dir / "y_train.npy")
    X_val = np.load(CFG.preprocess_dir / "X_val.npy")
    y_val = np.load(CFG.preprocess_dir / "y_val.npy")
    X_test = np.load(CFG.preprocess_dir / "X_test.npy")
    y_test = np.load(CFG.preprocess_dir / "y_test.npy")
    
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Load metadata
    meta = joblib.load(CFG.preprocess_dir / "meta.joblib")
    print(f"\n  Preprocessing info:")
    print(f"    Selected features (before encoding): {meta['selected_before_encoding']}")
    print(f"    Final features (after encoding): {meta['processed_feature_count']}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def tune_xgboost(X_train, y_train):
    """Hyperparameter tuning for XGBoost"""
    print("\n🔧 Tuning XGBoost hyperparameters...")
    
    param_dist = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200, 300, 500],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1, 1.5],
        'scale_pos_weight': [1, 2, 3]
    }
    
    xgb = XGBClassifier(random_state=CFG.random_state, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        xgb, param_distributions=param_dist,
        n_iter=CFG.n_iter_search,
        cv=CFG.cv_folds,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=CFG.random_state,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"  Best parameters: {random_search.best_params_}")
    print(f"  Best CV score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_


def tune_random_forest(X_train, y_train):
    """Hyperparameter tuning for Random Forest"""
    print("\n🔧 Tuning Random Forest hyperparameters...")
    
    param_dist = {
        'n_estimators': [200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.3],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    rf = RandomForestClassifier(random_state=CFG.random_state, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist,
        n_iter=CFG.n_iter_search,
        cv=CFG.cv_folds,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=CFG.random_state,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"  Best parameters: {random_search.best_params_}")
    print(f"  Best CV score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_


def train_catboost(X_train, y_train, X_val, y_val):
    """Train CatBoost model"""
    print("\n🚀 Training CatBoost...")
    
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        random_seed=CFG.random_state,
        verbose=False
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    
    return model


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM model"""
    print("\n🚀 Training LightGBM...")
    
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=CFG.random_state,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    return model


def create_stacking_ensemble(X_train, y_train, tuned_models):
    """Create stacking ensemble from tuned models"""
    print("\n🎯 Creating Stacking Ensemble...")
    
    base_estimators = [
        ('rf', tuned_models['Random Forest']),
        ('xgb', tuned_models['XGBoost']),
        ('cat', tuned_models['CatBoost']),
        ('lgb', tuned_models['LightGBM'])
    ]
    
    meta_model = LogisticRegression(max_iter=1000, random_state=CFG.random_state)
    
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    
    print("  Fitting stacking ensemble...")
    stacking.fit(X_train, y_train)
    
    return stacking


def create_voting_ensemble(tuned_models):
    """Create voting ensemble from tuned models"""
    print("\n🎯 Creating Voting Ensemble...")
    
    estimators = [
        ('rf', tuned_models['Random Forest']),
        ('xgb', tuned_models['XGBoost']),
        ('cat', tuned_models['CatBoost']),
        ('lgb', tuned_models['LightGBM'])
    ]
    
    voting = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=[1, 2, 2, 2],
        n_jobs=-1
    )
    
    return voting


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics"""
    print(f"\n📊 Evaluating {model_name}...")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0
    roc_auc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    
    results = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'F2-Score': f2,
        'ROC-AUC': roc_auc,
        'AUPRC': auprc
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    
    return results, y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(CFG.results_dir / f"{model_name.replace(' ', '_')}_confusion_matrix.png", dpi=100, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_test, y_prob, model_name):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(CFG.results_dir / f"{model_name.replace(' ', '_')}_roc_curve.png", dpi=100, bbox_inches='tight')
    plt.close()


def main():
    """Main training pipeline V2"""
    start_time = time.time()
    
    print("="*80)
    print("IMPROVED MODEL TRAINING V2 - WITH PROPER PREPROCESSING")
    print("="*80)
    print("Using data with feature selection BEFORE one-hot encoding")
    print("="*80)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed_data()
    
    tuned_models = {}
    all_results = []
    
    # 1. Tune XGBoost
    tuned_models['XGBoost'] = tune_xgboost(X_train, y_train)
    xgb_results, xgb_pred, xgb_prob = evaluate_model(tuned_models['XGBoost'], X_test, y_test, 'XGBoost (Tuned)')
    all_results.append(xgb_results)
    plot_confusion_matrix(y_test, xgb_pred, 'XGBoost_Tuned_V2')
    plot_roc_curve(y_test, xgb_prob, 'XGBoost_Tuned_V2')
    joblib.dump(tuned_models['XGBoost'], CFG.models_dir / "xgboost_tuned_v2.joblib")
    
    # 2. Tune Random Forest
    tuned_models['Random Forest'] = tune_random_forest(X_train, y_train)
    rf_results, rf_pred, rf_prob = evaluate_model(tuned_models['Random Forest'], X_test, y_test, 'Random Forest (Tuned)')
    all_results.append(rf_results)
    plot_confusion_matrix(y_test, rf_pred, 'RandomForest_Tuned_V2')
    plot_roc_curve(y_test, rf_prob, 'RandomForest_Tuned_V2')
    joblib.dump(tuned_models['Random Forest'], CFG.models_dir / "random_forest_tuned_v2.joblib")
    
    # 3. Train CatBoost
    tuned_models['CatBoost'] = train_catboost(X_train, y_train, X_val, y_val)
    cat_results, cat_pred, cat_prob = evaluate_model(tuned_models['CatBoost'], X_test, y_test, 'CatBoost')
    all_results.append(cat_results)
    plot_confusion_matrix(y_test, cat_pred, 'CatBoost_V2')
    plot_roc_curve(y_test, cat_prob, 'CatBoost_V2')
    joblib.dump(tuned_models['CatBoost'], CFG.models_dir / "catboost_v2.joblib")
    
    # 4. Train LightGBM
    tuned_models['LightGBM'] = train_lightgbm(X_train, y_train, X_val, y_val)
    lgb_results, lgb_pred, lgb_prob = evaluate_model(tuned_models['LightGBM'], X_test, y_test, 'LightGBM')
    all_results.append(lgb_results)
    plot_confusion_matrix(y_test, lgb_pred, 'LightGBM_V2')
    plot_roc_curve(y_test, lgb_prob, 'LightGBM_V2')
    joblib.dump(tuned_models['LightGBM'], CFG.models_dir / "lightgbm_v2.joblib")
    
    # 5. Create Stacking Ensemble
    stacking_model = create_stacking_ensemble(X_train, y_train, tuned_models)
    stack_results, stack_pred, stack_prob = evaluate_model(stacking_model, X_test, y_test, 'Stacking Ensemble')
    all_results.append(stack_results)
    plot_confusion_matrix(y_test, stack_pred, 'Stacking_Ensemble_V2')
    plot_roc_curve(y_test, stack_prob, 'Stacking_Ensemble_V2')
    joblib.dump(stacking_model, CFG.models_dir / "stacking_ensemble_v2.joblib")
    
    # 6. Create Voting Ensemble
    voting_model = create_voting_ensemble(tuned_models)
    print("\n  Fitting voting ensemble...")
    voting_model.fit(X_train, y_train)
    vote_results, vote_pred, vote_prob = evaluate_model(voting_model, X_test, y_test, 'Voting Ensemble')
    all_results.append(vote_results)
    plot_confusion_matrix(y_test, vote_pred, 'Voting_Ensemble_V2')
    plot_roc_curve(y_test, vote_prob, 'Voting_Ensemble_V2')
    joblib.dump(voting_model, CFG.models_dir / "voting_ensemble_v2.joblib")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    results_df.to_csv(CFG.results_dir / "improved_model_results_v2.csv", index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY - V2 (PROPER FIX)")
    print("="*80)
    print(results_df.to_string(index=False))
    
    best_model = results_df.iloc[0]
    print(f"\n🏆 Best Model: {best_model['Model']}")
    print(f"   Accuracy: {best_model['Accuracy']:.4f} ({best_model['Accuracy']*100:.1f}%)")
    print(f"   ROC-AUC: {best_model['ROC-AUC']:.4f}")
    
    # Compare with baseline
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINE")
    print("="*80)
    print("Baseline (original training):")
    print("  - XGBoost: 69.0% accuracy")
    print("  - Random Forest: 68.0% accuracy")
    print(f"\nImproved V2 (with proper fix):")
    print(f"  - {best_model['Model']}: {best_model['Accuracy']*100:.1f}% accuracy")
    improvement = (best_model['Accuracy'] - 0.69) * 100
    print(f"  - Improvement: {improvement:+.1f} percentage points")
    
    if best_model['Accuracy'] >= 0.80:
        print("\n🎉 SUCCESS! Achieved 80%+ accuracy target!")
    elif improvement >= 5:
        print("\n✅ GOOD! Significant improvement achieved!")
    elif improvement >= 2:
        print("\n👍 MODERATE improvement. Better than baseline.")
    else:
        print("\n⚠️  Limited improvement. May need additional strategies.")
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Total training time: {elapsed_time/60:.1f} minutes")
    
    print(f"\n✅ Improved model training V2 complete!")
    print(f"Results saved to: {CFG.results_dir}")
    print(f"Models saved to: {CFG.models_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
