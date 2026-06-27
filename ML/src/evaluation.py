# src/evaluation.py
"Day 2: Model Evaluation Utilities, Focus on precision-recall for imbalanced classification"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from typing import Dict, Any, Tuple
from pathlib import Path
from sklearn.metrics import make_scorer


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    "Compute comprehensive classification metrics. Focus on PR-AUC for imbalanced data."
    return {
        # Primary metrics for imbalanced data
        'pr_auc': average_precision_score(y_true, y_proba),
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        
        # Secondary metrics
        'roc_auc': roc_auc_score(y_true, y_proba),
        'accuracy': np.mean(y_true == y_pred),
        
        # Class distribution
        'support_positive': np.sum(y_true == 1),
        'support_negative': np.sum(y_true == 0),
        'imbalance_ratio': np.sum(y_true == 0) / np.sum(y_true == 1)
    }


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: Path = None
) -> plt.Figure:
    """
    Plot Precision-Recall curve.
    Better than ROC for imbalanced datasets.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
    ax.axhline(y=np.mean(y_true), color='r', linestyle='--', 
               label=f'Baseline (No Skill = {np.mean(y_true):.3f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve: {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: Path = None
) -> plt.Figure:
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (ROC-AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', label='Random (AUC = 0.500)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve: {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Path = None
) -> plt.Figure:

    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def compare_models_pr(
    models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Path = None
) -> plt.Figure:
    "Compare multiple models on same PR curve plot."
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for model_name, (y_true, y_proba) in models_data.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        ax.plot(recall, precision, linewidth=2, label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
    
    # Baseline
    y_true_sample = list(models_data.values())[0][0]
    ax.axhline(y=np.mean(y_true_sample), color='r', linestyle='--', 
               label=f'No Skill Baseline ({np.mean(y_true_sample):.3f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Model Comparison: Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"):
    print(f"Classification Report: {model_name}")
    print(classification_report(y_true, y_pred, target_names=['No Churn', 'Churn']))
    
    # Additional context
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Negatives:  {tn:5d} (Correctly predicted No Churn)")
    print(f"  False Positives: {fp:5d} (Incorrectly predicted Churn)")
    print(f"  False Negatives: {fn:5d} (Missed Churn cases)")
    print(f"  True Positives:  {tp:5d} (Correctly predicted Churn)")

def analyze_class_imbalance(y: np.ndarray) -> Dict[str, Any]:
    unique, counts = np.unique(y, return_counts=True)
    majority_count = counts[0]  # Assuming 0 is majority (No Churn)
    minority_count = counts[1]  # Assuming 1 is minority (Churn)
    
    imbalance_ratio = majority_count / minority_count
    minority_pct = (minority_count / len(y)) * 100
    
    analysis = {
        'total_samples': len(y),
        'majority_class': int(unique[0]),
        'majority_count': int(majority_count),
        'minority_class': int(unique[1]),
        'minority_count': int(minority_count),
        'imbalance_ratio': float(imbalance_ratio),
        'minority_percentage': float(minority_pct),
        'is_imbalanced': imbalance_ratio > 1.5,  # Rule of thumb
        'severity': 'severe' if imbalance_ratio > 3 else 'moderate' if imbalance_ratio > 1.5 else 'balanced'
    }
    
    return analysis


def cost_sensitive_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Custom scorer for GridSearchCV that minimizes business cost.
    Returns:
        Negative total cost (GridSearchCV maximizes, so negate for minimization)
    Note:
        Import calculate_business_cost locally to avoid circular imports
    """
    from utils import calculate_business_cost
    
    cost_breakdown = calculate_business_cost(y_true, y_pred)
    return -cost_breakdown['total_cost']  # Negative because GridSearchCV maximizes


def get_cost_scorer():
    return make_scorer(cost_sensitive_scorer)


# Day 4: Calibration Functions

def calibrate_model(model, X_calib, y_calib, method='sigmoid'):
    """
    Calibrate model probabilities using CalibratedClassifierCV with cv='prefit'.  
    CRITICAL: To avoid data leakage, you MUST:
    1. Split your training data into train_proper (75%) and calibration holdout (25%)
    2. Train the base model ONLY on train_proper
    3. Pass the fitted model and calibration holdout to this function
    """
    from sklearn.calibration import CalibratedClassifierCV
    
    # Use cv='prefit' to avoid refitting the base model
    # This ensures calibration only happens on the held-out calibration set
    calibrated_model = CalibratedClassifierCV(
        model, 
        method=method, 
        cv='prefit'  # CRITICAL: prevents data leakage
    )
    calibrated_model.fit(X_calib, y_calib)
    
    return calibrated_model


def compute_calibration_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Compute calibration quality metrics.
    Returns:
        Dictionary with Brier score and Expected Calibration Error (ECE)
    Notes:
        - Brier Score: Lower is better (0 = perfect, 1 = worst). Measures MSE of probabilities.
        - ECE: Lower is better (0 = perfect calibration). Measures gap between confidence and accuracy.
    """
    from sklearn.metrics import brier_score_loss
    
    # Brier score (lower is better)
    brier = brier_score_loss(y_true, y_proba)
    
    # Expected Calibration Error (ECE)
    ece = compute_ece(y_true, y_proba, n_bins=10)
    
    return {
        'brier_score': brier,
        'ece': ece
    }


def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    "ECE = sum(|accuracy - confidence| * bin_weight) across probability bins"
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find samples in this bin
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        
        if np.sum(in_bin) > 0:
            # Average confidence in bin
            bin_confidence = np.mean(y_proba[in_bin])
            
            # Accuracy in bin
            bin_accuracy = np.mean(y_true[in_bin])
            
            # Weighted contribution to ECE
            bin_weight = np.sum(in_bin) / len(y_true)
            ece += np.abs(bin_accuracy - bin_confidence) * bin_weight
    
    return ece


def plot_reliability_curve(
    y_true: np.ndarray,
    y_proba_uncalibrated: np.ndarray,
    y_proba_calibrated: np.ndarray,
    model_name: str = "Model",
    save_path: Path = None
) -> plt.Figure:
    """
    Plot reliability diagram (calibration curve) comparing before/after calibration.
    Note:
        Perfect calibration = diagonal line (predicted probability = actual frequency)
    """
    from sklearn.calibration import calibration_curve
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Reliability curves
    # Uncalibrated
    fraction_of_positives_uncal, mean_predicted_value_uncal = calibration_curve(
        y_true, y_proba_uncalibrated, n_bins=10
    )
    
    # Calibrated
    fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
        y_true, y_proba_calibrated, n_bins=10
    )
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax1.plot(mean_predicted_value_uncal, fraction_of_positives_uncal, 
             's-', label='Uncalibrated', linewidth=2, markersize=8)
    ax1.plot(mean_predicted_value_cal, fraction_of_positives_cal, 
             'o-', label='Calibrated', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title(f'Reliability Diagram: {model_name}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    
    # Plot 2: Probability histograms
    ax2.hist(y_proba_uncalibrated, bins=20, alpha=0.5, label='Uncalibrated', 
             color='orange', edgecolor='black')
    ax2.hist(y_proba_calibrated, bins=20, alpha=0.5, label='Calibrated', 
             color='blue', edgecolor='black')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Probability Distribution', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def compare_calibration(
    y_true: np.ndarray,
    y_proba_uncalibrated: np.ndarray,
    y_proba_calibrated: np.ndarray,
    model_name: str = "Model"
) -> pd.DataFrame:
    uncal_metrics = compute_calibration_metrics(y_true, y_proba_uncalibrated)
    cal_metrics = compute_calibration_metrics(y_true, y_proba_calibrated)
    
    comparison_df = pd.DataFrame({
        'Model': [f'{model_name} (Uncalibrated)', f'{model_name} (Calibrated)'],
        'Brier_Score': [uncal_metrics['brier_score'], cal_metrics['brier_score']],
        'ECE': [uncal_metrics['ece'], cal_metrics['ece']]
    })
    
    # Calculate improvement
    brier_improvement = ((uncal_metrics['brier_score'] - cal_metrics['brier_score']) 
                         / uncal_metrics['brier_score']) * 100
    ece_improvement = ((uncal_metrics['ece'] - cal_metrics['ece']) 
                       / uncal_metrics['ece']) * 100
    
    print(f"\nCalibration Improvement for {model_name}:")
    print(f"  Brier Score: {uncal_metrics['brier_score']:.4f} → {cal_metrics['brier_score']:.4f} "
          f"({brier_improvement:+.1f}%)")
    print(f"  ECE:         {uncal_metrics['ece']:.4f} → {cal_metrics['ece']:.4f} "
          f"({ece_improvement:+.1f}%)")
    
    return comparison_df

# Day 5: Fairness Analysis Functions

def slice_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    X: pd.DataFrame,
    slice_feature: str,
    bins: int = None,
    bin_labels: list = None
) -> pd.DataFrame:
    """
    Compute metrics across slices of a feature (e.g., tenure bins, Contract types).    
    Returns: DataFrame with metrics per slice
    """
    # Create slice groups
    if pd.api.types.is_numeric_dtype(X[slice_feature]) and bins is not None:
        # Bin numeric features
        slice_groups = pd.cut(X[slice_feature], bins=bins, labels=bin_labels)
        slice_name = f"{slice_feature}_binned"
    else:
        # Use categorical as-is
        slice_groups = X[slice_feature]
        slice_name = slice_feature
    
    results = []
    
    for group_value in slice_groups.unique():
        if pd.isna(group_value):
            continue
            
        mask = (slice_groups == group_value)
        
        if np.sum(mask) < 10:  # Skip groups with too few samples
            continue
        
        y_true_slice = y_true[mask]
        y_pred_slice = y_pred[mask]
        y_proba_slice = y_proba[mask]
        
        # Compute metrics for this slice
        try:
            metrics = compute_metrics(y_true_slice, y_pred_slice, y_proba_slice)
            
            results.append({
                'slice_feature': slice_name,
                'slice_value': str(group_value),
                'sample_count': int(np.sum(mask)),
                'churn_rate': float(np.mean(y_true_slice)),
                'pr_auc': metrics['pr_auc'],
                'f1_score': metrics['f1_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'roc_auc': metrics['roc_auc']
            })
        except:
            # Skip if metrics can't be computed (e.g., only one class)
            continue
    
    return pd.DataFrame(results)


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    slice_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Returns:
        Dictionary with fairness statistics including:
        - max_disparity: Maximum difference in any metric across slices
        - min_max_ratio: Ratio of worst to best performing slice
        - coefficient_of_variation: Std/mean of performance across slices
    """
    if len(slice_df) < 2:
        return {
            'max_pr_auc_disparity': 0.0,
            'max_recall_disparity': 0.0,
            'max_precision_disparity': 0.0,
            'pr_auc_cv': 0.0,
            'recall_cv': 0.0,
            'fair_slices': [],
            'concerning_slices': []
        }
    
    # Disparities (max - min)
    max_pr_auc_disparity = slice_df['pr_auc'].max() - slice_df['pr_auc'].min()
    max_recall_disparity = slice_df['recall'].max() - slice_df['recall'].min()
    max_precision_disparity = slice_df['precision'].max() - slice_df['precision'].min()
    
    # Coefficient of variation (std / mean)
    pr_auc_cv = slice_df['pr_auc'].std() / slice_df['pr_auc'].mean()
    recall_cv = slice_df['recall'].std() / slice_df['recall'].mean()
    
    # Identify concerning slices (below 80% of mean performance)
    mean_pr_auc = slice_df['pr_auc'].mean()
    threshold = 0.8 * mean_pr_auc
    
    fair_slices = slice_df[slice_df['pr_auc'] >= threshold]['slice_value'].tolist()
    concerning_slices = slice_df[slice_df['pr_auc'] < threshold][
        ['slice_value', 'pr_auc', 'sample_count']
    ].to_dict('records')
    
    return {
        'max_pr_auc_disparity': float(max_pr_auc_disparity),
        'max_recall_disparity': float(max_recall_disparity),
        'max_precision_disparity': float(max_precision_disparity),
        'pr_auc_cv': float(pr_auc_cv),
        'recall_cv': float(recall_cv),
        'fair_slices': fair_slices,
        'concerning_slices': concerning_slices
    }


def plot_slice_comparison(
    slice_df: pd.DataFrame,
    metric: str = 'pr_auc',
    save_path: Path = None
) -> plt.Figure:
    """
    Visualize metric performance across slices.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart of metric by slice
    slice_df_sorted = slice_df.sort_values(metric, ascending=False)
    
    ax1.barh(slice_df_sorted['slice_value'], slice_df_sorted[metric], 
             color='skyblue', edgecolor='navy')
    ax1.axvline(slice_df[metric].mean(), color='red', linestyle='--', 
                label=f'Mean: {slice_df[metric].mean():.3f}')
    ax1.set_xlabel(metric.upper().replace('_', '-'), fontsize=12)
    ax1.set_ylabel('Slice', fontsize=12)
    ax1.set_title(f'{metric.upper()} by Slice', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='x')
    
    # Sample size vs performance
    ax2.scatter(slice_df['sample_count'], slice_df[metric], 
                s=100, alpha=0.6, edgecolor='navy')
    
    for idx, row in slice_df.iterrows():
        ax2.annotate(row['slice_value'], 
                    (row['sample_count'], row[metric]),
                    fontsize=8, alpha=0.7)
    
    ax2.set_xlabel('Sample Count', fontsize=12)
    ax2.set_ylabel(metric.upper().replace('_', '-'), fontsize=12)
    ax2.set_title('Performance vs Sample Size', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def analyze_fairness_across_features(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    X: pd.DataFrame,
    features_to_analyze: list
) -> Dict[str, Any]:
    "Returns: Dictionary with slice dataframes and fairness metrics per feature"
    results = {}
    
    for feature_config in features_to_analyze:
        if len(feature_config) == 3:
            feature, bins, labels = feature_config
        else:
            feature, bins, labels = feature_config[0], None, None
        
        # Compute slice metrics
        slice_df = slice_metrics(y_true, y_pred, y_proba, X, feature, bins, labels)
        
        if len(slice_df) > 0:
            # Compute fairness metrics
            fairness_metrics = compute_fairness_metrics(y_true, y_pred, slice_df)
            
            results[feature] = {
                'slice_metrics': slice_df,
                'fairness_metrics': fairness_metrics
            }
    
    return results
