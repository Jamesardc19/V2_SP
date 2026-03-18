import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve,
    brier_score_loss
)
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, RFE, f_classif, mutual_info_classif, chi2
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# Import custom modules
from tabnet_wrapper import TabNetClassifierWrapper
from apply_calibration_fixed import apply_calibration

# Configuration
class Config:
    # Paths
    preprocess_dir = r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\PREPROCESS_OUTPUT"
    models_dir = r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\Trained_Models"
    results_dir = r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\Model_Results"
    lime_dir = r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\LIME_Explanations"
    
    # Model parameters
    random_state = 42
    
    # Feature selection parameters
    feature_selection_method = 'combined'  # 'statistical', 'model_based', 'rfe', or 'combined'
    n_features_to_select = 40             # Number of features to select
    
    # Evaluation metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'f2', 'auprc', 'roc_auc', 'brier_score']
    
    # TabNet optimization parameters
    tabnet_pbounds = {
        'n_d': (8, 64),
        'n_a': (8, 64),
        'n_steps': (3, 10),
        'gamma': (1.0, 2.0),
        'lambda_sparse': (1e-6, 1e-3),
        'learning_rate': (1e-3, 1e-1)
    }
    tabnet_init_points = 5
    tabnet_n_iter = 25

# Create configuration instance
CFG = Config()

# Create output directories if they don't exist
for directory in [CFG.models_dir, CFG.results_dir, CFG.lime_dir]:
    Path(directory).mkdir(parents=True, exist_ok=True)

def load_preprocessed_data():
    """
    Load preprocessed data from the preprocessing output directory
    """
    preprocess_dir = Path(CFG.preprocess_dir)
    
    # Load numpy arrays
    X_train = np.load(preprocess_dir / "X_train.npy")
    y_train = np.load(preprocess_dir / "y_train.npy")
    X_val = np.load(preprocess_dir / "X_val.npy")
    y_val = np.load(preprocess_dir / "y_val.npy")
    X_test = np.load(preprocess_dir / "X_test.npy")
    y_test = np.load(preprocess_dir / "y_test.npy")
    
    # Load preprocessor and feature names if available
    try:
        preprocessor = joblib.load(preprocess_dir / "preprocessor.joblib")
        feature_names = joblib.load(preprocess_dir / "feature_names.joblib")
    except:
        print("Warning: Could not load preprocessor or feature names")
        preprocessor = None
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    
    print(f"Loaded preprocessed data:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, feature_names

def apply_feature_selection(X_train, y_train, X_val, X_test, feature_names, method='combined', n_features=None):
    """
    Apply feature selection to reduce dimensionality and noise in the dataset
    
    Parameters:
    -----------
    X_train, y_train : numpy arrays
        Training data and labels
    X_val, X_test : numpy arrays
        Validation and test data
    feature_names : list
        Names of features
    method : str
        Feature selection method to use: 'statistical', 'model_based', 'rfe', or 'combined'
    n_features : int or None
        Number of features to select. If None, will use 50% of features or a minimum of 20
    
    Returns:
    --------
    X_train_selected, X_val_selected, X_test_selected : numpy arrays
        Selected features for train, validation and test sets
    selected_feature_names : list
        Names of selected features
    selector : object
        The fitted feature selector
    """
    if n_features is None:
        # Default to 50% of features or minimum 20, whichever is larger
        n_features = max(20, X_train.shape[1] // 2)
    
    print(f"Applying {method} feature selection to reduce from {X_train.shape[1]} to {n_features} features")
    
    if method == 'statistical':
        # Use ANOVA F-test for feature selection
        selector = SelectKBest(f_classif, k=n_features)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        
    elif method == 'model_based':
        # Use Random Forest for feature importance-based selection
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=CFG.random_state)
        selector = SelectFromModel(rf, threshold=-np.inf, max_features=n_features)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        
    elif method == 'rfe':
        # Use Recursive Feature Elimination
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=CFG.random_state)
        selector = RFE(rf, n_features_to_select=n_features, step=0.1)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        
    elif method == 'combined':
        # Combine multiple methods for more robust selection
        # 1. First filter using statistical test
        stat_selector = SelectKBest(f_classif, k=min(n_features*2, X_train.shape[1]))
        X_train_stat = stat_selector.fit_transform(X_train, y_train)
        X_val_stat = stat_selector.transform(X_val)
        X_test_stat = stat_selector.transform(X_test)
        
        # Get indices of features selected by statistical test
        stat_indices = stat_selector.get_support(indices=True)
        
        # Map back to original feature indices
        stat_feature_names = [feature_names[i] for i in stat_indices]
        
        # 2. Then use model-based selection on the reduced feature set
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=CFG.random_state)
        model_selector = SelectFromModel(rf, threshold=-np.inf, max_features=n_features)
        X_train_selected = model_selector.fit_transform(X_train_stat, y_train)
        X_val_selected = model_selector.transform(X_val_stat)
        X_test_selected = model_selector.transform(X_test_stat)
        
        # Get indices of features selected by model-based selection
        model_indices = model_selector.get_support(indices=True)
        
        # Map back to original feature indices
        selected_indices = [stat_indices[i] for i in model_indices]
        
        # Create a combined selector for convenience
        class CombinedSelector:
            def __init__(self, stat_selector, model_selector, selected_indices):
                self.stat_selector = stat_selector
                self.model_selector = model_selector
                self.selected_indices = selected_indices
                
            def transform(self, X):
                X_stat = self.stat_selector.transform(X)
                return self.model_selector.transform(X_stat)
                
            def get_support(self, indices=False):
                if indices:
                    return self.selected_indices
                else:
                    support = np.zeros(len(feature_names), dtype=bool)
                    support[self.selected_indices] = True
                    return support
        
        selector = CombinedSelector(stat_selector, model_selector, selected_indices)
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    # Get names of selected features
    selected_feature_names = [feature_names[i] for i in selected_indices]
    
    print(f"Feature selection complete. Selected {len(selected_feature_names)} features.")
    
    # Plot feature importance if possible
    if method in ['model_based', 'rfe']:
        try:
            if method == 'model_based':
                importances = selector.estimator_.feature_importances_
            else:  # rfe
                importances = selector.estimator_.feature_importances_[selector.get_support()]
                
            # Plot feature importances
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(importances)), importances)
            plt.title('Feature Importances')
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(CFG.results_dir, f"{method}_feature_importances.png"))
            plt.close()
        except:
            print("Could not plot feature importances")
    
    return X_train_selected, X_val_selected, X_test_selected, selected_feature_names, selector

def initialize_models():
    """
    Initialize all models to be trained and evaluated
    """
    models = {
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=CFG.random_state),
        'SVM': SVC(probability=True, max_iter=10000, random_state=CFG.random_state),
        'Random Forest': RandomForestClassifier(random_state=CFG.random_state),
        'k-NN': KNeighborsClassifier(n_neighbors=5),
        'XGBoost': XGBClassifier(random_state=CFG.random_state)
    }
    return models

def calculate_f2_score(y_true, y_pred):
    """
    Calculate F2 score which emphasizes recall over precision
    """
    return fbeta_score(y_true, y_pred, beta=2)

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a model on test data and return metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # F2 score (emphasizes recall over precision)
    from sklearn.metrics import fbeta_score
    f2 = fbeta_score(y_test, y_pred, beta=2)
    
    # ROC AUC and PR AUC
    roc_auc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    
    # Brier score (calibration metric)
    brier = brier_score_loss(y_test, y_prob)
    
    # Store results
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'roc_auc': roc_auc,
        'auprc': auprc,
        'brier_score': brier
    }
    
    return results, y_pred, y_prob

def optimize_tabnet(X_train, y_train, X_val, y_val):
    """
    Optimize TabNet hyperparameters using Bayesian Optimization
    """
    # Convert to numpy arrays if they aren't already
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
    y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
    
    def tabnet_cv(n_d, n_a, n_steps, gamma, lambda_sparse, learning_rate):
        # Convert float parameters to int where needed
        n_d = int(n_d)
        n_a = int(n_a)
        n_steps = int(n_steps)
        
        # Initialize TabNet
        model = TabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": learning_rate},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,  # Use StepLR instead of ReduceLROnPlateau
            scheduler_params={"step_size": 10, "gamma": 0.5},  # Reduce LR by half every 10 epochs
            mask_type='entmax',
            verbose=0,
            seed=CFG.random_state
        )
        
        # Fit model
        model.fit(
            X_train=X_train_np, y_train=y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            max_epochs=100,
            patience=10,
            batch_size=1024,
            virtual_batch_size=128,
            eval_metric=['auc']
        )
        
        # Predict on validation set
        y_pred = model.predict_proba(X_val_np)[:, 1]
        
        # Return AUPRC as optimization metric
        return average_precision_score(y_val_np, y_pred)
    
    # Initialize Bayesian Optimization
    optimizer = BayesianOptimization(
        f=tabnet_cv,
        pbounds=CFG.tabnet_pbounds,
        random_state=CFG.random_state
    )
    
    # Optimize
    optimizer.maximize(
        init_points=CFG.tabnet_init_points,
        n_iter=CFG.tabnet_n_iter
    )
    
    # Get best parameters
    best_params = optimizer.max['params']
    best_params['n_d'] = int(best_params['n_d'])
    best_params['n_a'] = int(best_params['n_a'])
    best_params['n_steps'] = int(best_params['n_steps'])
    
    print("Best TabNet Parameters:", best_params)
    
    # Train final model with best parameters
    final_model = TabNetClassifier(
        n_d=best_params['n_d'],
        n_a=best_params['n_a'],
        n_steps=best_params['n_steps'],
        gamma=best_params['gamma'],
        lambda_sparse=best_params['lambda_sparse'],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=best_params['learning_rate']),
        mask_type='entmax',
        scheduler_fn=torch.optim.lr_scheduler.StepLR,  # Use StepLR instead of ReduceLROnPlateau
        scheduler_params={"step_size": 10, "gamma": 0.5},  # Reduce LR by half every 10 epochs
        verbose=1,
        seed=CFG.random_state
    )
    
    # Combine train and validation for final training
    X_train_val = np.vstack([X_train_np, X_val_np])
    y_train_val = np.concatenate([y_train_np, y_val_np])
    
    # Fit final model
    final_model.fit(
        X_train=X_train_val, y_train=y_train_val,
        eval_set=[(X_train_val, y_train_val)],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
        eval_metric=['auc']
    )
    
    return final_model, best_params

def evaluate_calibration(calibrated_models, X_test, y_test):
    """
    Evaluate and plot calibration curves for all models
    """
    if not calibrated_models:
        print("No calibrated models to evaluate.")
        return
    
    n_models = len(calibrated_models)
    n_rows = max(1, (n_models + 2) // 3)  # Calculate number of rows needed, at least 1
    n_cols = min(3, n_models)     # Maximum 3 columns
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    
    # Handle single subplot case
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(-1)
    else:
        axes = axes.flatten()
    
    for i, (model_name, models_dict) in enumerate(calibrated_models.items()):
        ax = axes[i]
        plt.sca(ax)
        
        # Original model
        y_prob = models_dict['original'].predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Original')
        
        # Platt scaling
        if 'platt' in models_dict:
            y_prob_platt = models_dict['platt'].predict_proba(X_test)[:, 1]
            prob_true_platt, prob_pred_platt = calibration_curve(y_test, y_prob_platt, n_bins=10)
            plt.plot(prob_pred_platt, prob_true_platt, marker='s', linewidth=1, label='Platt')
        
        # Isotonic regression
        if 'isotonic' in models_dict:
            y_prob_iso = models_dict['isotonic'].predict_proba(X_test)[:, 1]
            prob_true_iso, prob_pred_iso = calibration_curve(y_test, y_prob_iso, n_bins=10)
            plt.plot(prob_pred_iso, prob_true_iso, marker='^', linewidth=1, label='Isotonic')
        
        # Plot diagonal (perfect calibration)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        plt.title(f'Calibration Curve - {model_name}')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.legend(loc='best')
        plt.grid(True)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.results_dir, 'calibration_curves.png'), dpi=100, bbox_inches='tight')
    plt.close()

def generate_lime_explanations(models, X_train, X_test, y_test, feature_names):
    """
    Generate LIME explanations for each model
    """
    from lime.lime_tabular import LimeTabularExplainer
    
    # Create LIME explainer
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['No Diabetes Risk', 'Diabetes Risk'],
        mode='classification'
    )
    
    # Select a few test instances for explanation
    np.random.seed(CFG.random_state)
    test_indices = np.random.choice(len(X_test), size=5, replace=False)
    
    for model_name, model in models.items():
        print(f"Generating LIME explanations for {model_name}...")
        
        for idx in test_indices:
            try:
                # Generate explanation
                explanation = explainer.explain_instance(
                    X_test[idx], 
                    model.predict_proba,
                    num_features=10
                )
                
                # Save explanation as figure
                fig = plt.figure(figsize=(10, 6))
                explanation.as_pyplot_figure()
                plt.title(f"{model_name} - LIME Explanation for Instance {idx}")
                plt.tight_layout()
                plt.savefig(os.path.join(CFG.lime_dir, f"{model_name}_lime_instance_{idx}.png"))
                plt.close()
                
            except Exception as e:
                print(f"Error generating LIME explanation for {model_name}, instance {idx}: {e}")

def main():
    # Load preprocessed data
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, feature_names = load_preprocessed_data()
    
    # Apply feature selection to reduce dimensionality and noise
    print("\nApplying feature selection...")
    X_train_selected, X_val_selected, X_test_selected, selected_feature_names, selector = apply_feature_selection(
        X_train, y_train, X_val, X_test, feature_names, 
        method='combined',  # Use combined method for robust selection
        n_features=40       # Select top 40 features (adjust based on your dataset)
    )
    
    # Save selected feature names for reference
    feature_selection_info = {
        'selected_features': selected_feature_names,
        'original_feature_count': len(feature_names),
        'selected_feature_count': len(selected_feature_names),
        'selection_method': 'combined'
    }
    joblib.dump(feature_selection_info, os.path.join(CFG.results_dir, "feature_selection_info.joblib"))
    
    # Create a DataFrame with feature selection results for visualization
    feature_selection_df = pd.DataFrame({
        'feature': feature_names,
        'selected': [feature in selected_feature_names for feature in feature_names]
    })
    feature_selection_df.to_csv(os.path.join(CFG.results_dir, "feature_selection_results.csv"), index=False)
    
    print(f"\nReduced features from {len(feature_names)} to {len(selected_feature_names)}")
    print(f"Selected features saved to {os.path.join(CFG.results_dir, 'feature_selection_info.joblib')}")
    
    # Initialize models
    models = initialize_models()
    
    # Train and evaluate models
    results = []
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train_selected, y_train)
        
        # Evaluate model
        model_results, y_pred, y_prob = evaluate_model(model, X_test_selected, y_test, model_name)
        results.append(model_results)
        trained_models[model_name] = model
        
        # Save model
        joblib.dump(model, os.path.join(CFG.models_dir, f"{model_name.replace(' ', '_')}.joblib"))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(CFG.results_dir, f"{model_name.replace(' ', '_')}_confusion_matrix.png"))
        plt.close()
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {model_results["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(CFG.results_dir, f"{model_name.replace(' ', '_')}_roc_curve.png"))
        plt.close()
        
        # Plot PR curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUPRC = {model_results["auprc"]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(CFG.results_dir, f"{model_name.replace(' ', '_')}_pr_curve.png"))
        plt.close()
        
        # Print classification report
        print(f"\nClassification Report - {model_name}:")
        print(classification_report(y_test, y_pred))
    
    # Train and evaluate TabNet with Bayesian Optimization
    print("\nTraining BO-TabNet...")
    tabnet_model, best_params = optimize_tabnet(X_train_selected, y_train, X_val_selected, y_val)
    
    # Create a wrapper for TabNet to make it compatible with sklearn's calibration methods
    tabnet_wrapper = TabNetClassifierWrapper(tabnet_model)
    
    # Evaluate TabNet
    tabnet_results, tabnet_y_pred, tabnet_y_prob = evaluate_model(
        tabnet_wrapper, X_test_selected, y_test, "BO-TabNet"
    )
    results.append(tabnet_results)
    trained_models["BO-TabNet"] = tabnet_wrapper
    
    # Save TabNet model
    joblib.dump(tabnet_model, os.path.join(CFG.models_dir, "BO_TabNet.joblib"))
    
    # Plot TabNet confusion matrix, ROC, and PR curves
    # Confusion Matrix
    cm = confusion_matrix(y_test, tabnet_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix - BO-TabNet')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(CFG.results_dir, "BO_TabNet_confusion_matrix.png"))
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, tabnet_y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {tabnet_results["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - BO-TabNet')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CFG.results_dir, "BO_TabNet_roc_curve.png"))
    plt.close()
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, tabnet_y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AUPRC = {tabnet_results["auprc"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - BO-TabNet')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CFG.results_dir, "BO_TabNet_pr_curve.png"))
    plt.close()
    
    # Print TabNet classification report
    print("\nClassification Report - BO-TabNet:")
    print(classification_report(y_test, tabnet_y_pred))
    
    # Apply calibration to all models
    print("\nApplying calibration to all models...")
    calibrated_models = apply_calibration(trained_models, X_train_selected, y_train, X_test_selected, y_test, CFG.models_dir)
    
    # Evaluate calibration
    evaluate_calibration(calibrated_models, X_test_selected, y_test)
    
    # Generate LIME explanations
    generate_lime_explanations(trained_models, X_train_selected, X_test_selected, y_test, selected_feature_names)
    
    # Create results DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(CFG.results_dir, "model_results.csv"), index=False)
    
    # Print summary of results
    print("\nModel Performance Summary:")
    print(results_df[['model_name', 'accuracy', 'precision', 'recall', 'f1', 'f2', 'auprc', 'roc_auc']].to_string(index=False))
    
    print("\nTraining and evaluation complete!")
    print(f"Results saved to {CFG.results_dir}")
    print(f"Models saved to {CFG.models_dir}")
    print(f"LIME explanations saved to {CFG.lime_dir}")

if __name__ == "__main__":
    main()
