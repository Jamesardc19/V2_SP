import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

# For data preprocessing and model evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, precision_recall_curve,
    auc, roc_curve, roc_auc_score, average_precision_score,
    fbeta_score, brier_score_loss
)

# For calibration
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# For model interpretability
import lime
import lime.lime_tabular

# Import TabNet wrapper for calibration compatibility
from tabnet_wrapper import TabNetClassifierWrapper

# Set paths
models_dir = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\Trained_Models")
results_dir = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\Model_Results")
lime_dir = Path(r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\LIME_Explanations")
processed_dataset_path = r"d:\School Files\CMSC\CMSC 199\SP\V2_SP\processed_dataset.csv"

# Create directories if they don't exist
for directory in [models_dir, results_dir, lime_dir]:
    directory.mkdir(exist_ok=True)

def load_data():
    """Load the preprocessed dataset"""
    print("Loading preprocessed dataset...")
    df = pd.read_csv(processed_dataset_path)
    
    # Check for any string columns that might cause issues
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"Column {col} has object dtype and will be dropped")
            df = df.drop(columns=[col])
    
    # Separate features and target
    X = df.drop(['diabetes_risk', 'fbs'], axis=1)
    y = df['diabetes_risk']
    
    # Handle NaN values by imputing with mean
    nan_cols = X.columns[X.isna().any()].tolist()
    for col in nan_cols:
        X[col] = X[col].fillna(X[col].mean())
    
    # Drop any columns with all NaN values
    if X.isna().any().any():
        X = X.dropna(axis=1)
    
    # Get feature names for later use
    feature_names = X.columns.tolist()
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    print(f"Class distribution (%):\n{y.value_counts() / len(y) * 100}")
    
    return X, y, feature_names

def load_models():
    """Load all trained models"""
    print("\nLoading trained models...")
    
    models = {}
    model_files = list(models_dir.glob("*_model.pkl"))
    
    for model_file in model_files:
        model_name = model_file.stem.replace('_', ' ').title()
        if "Bo Tabnet" in model_name:
            model_name = "BO-TabNet"  # Fix naming
        
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
                models[model_name] = model
                print(f"Loaded {model_name} model")
        except Exception as e:
            print(f"Error loading {model_name} model: {str(e)}")
    
    return models

def generate_detailed_confusion_matrix(y_true, y_pred, model_name):
    """Generate and save a detailed confusion matrix visualization"""
    print(f"\nGenerating detailed confusion matrix for {model_name}...")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a more detailed visualization
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix as heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    
    # Add labels, title, and ticks
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # Add class labels
    plt.xticks([0.5, 1.5], ['Non-diabetic (0)', 'Diabetic (1)'])
    plt.yticks([0.5, 1.5], ['Non-diabetic (0)', 'Diabetic (1)'])
    
    # Add metrics in the figure
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision_val * sensitivity) / (precision_val + sensitivity) if (precision_val + sensitivity) > 0 else 0
    f2 = (1 + 2**2) * (precision_val * sensitivity) / ((2**2 * precision_val) + sensitivity) if (precision_val + sensitivity) > 0 else 0
    
    plt.figtext(0.15, 0.01, f'Accuracy: {accuracy:.4f}', fontsize=10)
    plt.figtext(0.35, 0.01, f'Sensitivity: {sensitivity:.4f}', fontsize=10)
    plt.figtext(0.55, 0.01, f'Specificity: {specificity:.4f}', fontsize=10)
    plt.figtext(0.75, 0.01, f'F2-score: {f2:.4f}', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    plt.savefig(results_dir / f"{model_name.replace(' ', '_').lower()}_detailed_confusion_matrix.png", dpi=300)
    plt.close()
    
    print(f"Detailed confusion matrix saved for {model_name}")
    
    return cm

def generate_roc_curve(y_true, y_prob, model_name):
    """Generate and save ROC curve for a specific model"""
    print(f"\nGenerating ROC curve for {model_name}...")
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # Plot random guess line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    
    # Add labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(results_dir / f"{model_name.replace(' ', '_').lower()}_roc_curve.png", dpi=300)
    plt.close()
    
    print(f"ROC curve saved for {model_name}")
    
    return roc_auc

def generate_precision_recall_curve(y_true, y_prob, model_name):
    """Generate and save Precision-Recall curve for a specific model"""
    print(f"\nGenerating Precision-Recall curve for {model_name}...")
    
    # Calculate Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall, precision)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot Precision-Recall curve
    plt.plot(recall, precision, color='green', lw=2, label=f'{model_name} (AUPRC = {auprc:.4f})')
    
    # Add labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc='best')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(results_dir / f"{model_name.replace(' ', '_').lower()}_precision_recall_curve.png", dpi=300)
    plt.close()
    
    print(f"Precision-Recall curve saved for {model_name}")
    
    return auprc

def generate_classification_report(y_true, y_pred, model_name):
    """Generate and save classification report for a specific model"""
    print(f"\nGenerating classification report for {model_name}...")
    
    # Calculate classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Convert to DataFrame for better visualization
    report_df = pd.DataFrame(report).transpose()
    
    # Add F2-score (which emphasizes recall)
    f2_score = fbeta_score(y_true, y_pred, beta=2)
    report_df.loc['weighted avg', 'f2-score'] = f2_score
    
    # Save the report to CSV
    report_df.to_csv(results_dir / f"{model_name.replace(' ', '_').lower()}_classification_report.csv")
    
    # Print the report
    print(f"\nClassification Report for {model_name}:")
    print(report_df)
    
    return report_df

def generate_lime_explanations(model, X_train, X_test, y_test, feature_names, model_name, num_samples=5):
    """Generate and save LIME explanations for a specific model"""
    print(f"\nGenerating LIME explanations for {model_name}...")
    
    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        class_names=['Non-diabetic', 'Diabetic'],
        mode='classification'
    )
    
    # Select random test instances
    np.random.seed(42)
    test_indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    # Handle TabNet differently
    if "TabNet" in model_name:
        # Create a wrapper function for TabNet prediction
        def tabnet_predict_fn(x):
            # Convert to numpy array if needed
            x_np = x if isinstance(x, np.ndarray) else np.array(x)
            return model.predict_proba(x_np)
        
        predict_fn = tabnet_predict_fn
    else:
        predict_fn = model.predict_proba
    
    # Generate explanations
    for i, idx in enumerate(test_indices):
        try:
            # Generate explanation
            exp = explainer.explain_instance(
                X_test.iloc[idx].values,
                predict_fn,
                num_features=10
            )
            
            # Create a figure with both the explanation and actual vs predicted values
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot LIME explanation
            exp.as_pyplot_figure(ax=ax1)
            ax1.set_title(f"LIME Explanation - {model_name}")
            
            # Plot actual vs predicted for this instance
            if "TabNet" in model_name:
                y_pred = model.predict(X_test.iloc[idx].values.reshape(1, -1))[0]
                y_prob = model.predict_proba(X_test.iloc[idx].values.reshape(1, -1))[0, 1]
            else:
                y_pred = model.predict(X_test.iloc[idx].values.reshape(1, -1))[0]
                y_prob = model.predict_proba(X_test.iloc[idx].values.reshape(1, -1))[0, 1]
            
            # Create a simple bar chart showing the prediction probability
            bars = ax2.bar(['Non-diabetic', 'Diabetic'], 
                          [1-y_prob, y_prob], 
                          color=['blue', 'red'])
            
            # Add a horizontal line for the decision threshold (0.5)
            ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.7)
            
            # Add text annotations
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height:.2f}', ha='center', va='bottom')
            
            # Add actual class marker
            actual_class = y_test.iloc[idx]
            ax2.scatter(actual_class, 0.02, marker='^', color='green', s=200, label='Actual Class')
            
            # Set labels and title
            ax2.set_ylim(0, 1.1)
            ax2.set_ylabel('Probability')
            ax2.set_title(f"Prediction for Instance {idx}\nActual: {'Diabetic' if actual_class == 1 else 'Non-diabetic'}, Predicted: {'Diabetic' if y_pred == 1 else 'Non-diabetic'}")
            ax2.legend()
            
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(lime_dir / f"{model_name.replace(' ', '_').lower()}_lime_explanation_{i+1}.png", dpi=300)
            plt.close(fig)
            
        except Exception as e:
            print(f"Error generating LIME explanation for {model_name}, instance {idx}: {str(e)}")
    
    print(f"LIME explanations saved for {model_name}")

def evaluate_calibration(model, X_test, y_test, model_name):
    """Evaluate and visualize calibration for a specific model"""
    print(f"\nEvaluating calibration for {model_name}...")
    
    # Handle TabNet differently
    if "TabNet" in model_name:
        # Create a wrapper function for TabNet prediction
        def tabnet_predict_fn(x):
            # Convert to numpy array if needed
            x_np = x if isinstance(x, np.ndarray) else np.array(x)
            return model.predict_proba(x_np)[:, 1]
        
        y_prob = tabnet_predict_fn(X_test.values)
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10)
    
    # Calculate Brier score
    brier_score = brier_score_loss(y_test, y_prob)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot calibration curve
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", color='red', label=f'{model_name} (Brier score = {brier_score:.4f})')
    
    # Plot perfectly calibrated line
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    # Add labels and title
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curve - {model_name}")
    plt.legend(loc="best")
    plt.grid(True)
    
    # Save the figure
    plt.savefig(results_dir / f"{model_name.replace(' ', '_').lower()}_calibration_curve.png", dpi=300)
    plt.close()
    
    print(f"Calibration curve saved for {model_name}")
    
    return brier_score

def main():
    """Main function to run all evaluations"""
    print("Starting enhanced model evaluation...")
    
    # Load data
    X, y, feature_names = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load models
    models = load_models()
    
    # Create a DataFrame to store all metrics
    metrics_df = pd.DataFrame(columns=[
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'F2-score', 
        'AUPRC', 'ROC AUC', 'Brier Score'
    ])
    
    # Evaluate each model
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}...")
        print(f"{'='*50}")
        
        # Handle TabNet differently
        if "TabNet" in model_name:
            # Convert to numpy arrays
            X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
            
            # Make predictions
            y_pred = model.predict(X_test_np)
            y_prob = model.predict_proba(X_test_np)[:, 1]
        else:
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        
        # Generate detailed confusion matrix
        cm = generate_detailed_confusion_matrix(y_test, y_pred, model_name)
        
        # Generate ROC curve
        roc_auc = generate_roc_curve(y_test, y_prob, model_name)
        
        # Generate Precision-Recall curve
        auprc = generate_precision_recall_curve(y_test, y_prob, model_name)
        
        # Generate classification report
        report_df = generate_classification_report(y_test, y_pred, model_name)
        
        # Generate LIME explanations
        generate_lime_explanations(model, X_train, X_test, y_test, feature_names, model_name)
        
        # Evaluate calibration
        brier_score = evaluate_calibration(model, X_test, y_test, model_name)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2)
        
        # Add metrics to DataFrame
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'Model': [model_name],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1-score': [f1],
            'F2-score': [f2],
            'AUPRC': [auprc],
            'ROC AUC': [roc_auc],
            'Brier Score': [brier_score]
        })], ignore_index=True)
    
    # Save metrics to CSV
    metrics_df.to_csv(results_dir / "all_models_metrics.csv", index=False)
    
    # Create a bar chart comparing all models
    plt.figure(figsize=(15, 10))
    
    # Set up the metrics to plot
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'F2-score', 'AUPRC', 'ROC AUC']
    
    # Create a grouped bar chart
    metrics_df_plot = metrics_df.set_index('Model')
    ax = metrics_df_plot[metrics_to_plot].plot(kind='bar', figsize=(15, 10))
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Performance Comparison of All Models')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(results_dir / "all_models_comparison.png", dpi=300)
    
    print("\nModel evaluation completed successfully!")
    print(f"All results saved to {results_dir}")

if __name__ == "__main__":
    main()
