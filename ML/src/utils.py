# src/utils.py
"Utility functions: Simple business cost definitions for churn prediction. This module defines simple cost assumptions to frame the business problem."
import pandas as pd
import numpy as np

# Simple business cost assumptions (Day 1 requirement)
BUSINESS_COSTS = {
    'false_negative_cost': 100,  # Cost of missing a churner (lost revenue)
    'false_positive_cost': 10,   # Cost of wasted retention effort
}


def get_cost_ratio() -> float:
    return BUSINESS_COSTS['false_negative_cost'] / BUSINESS_COSTS['false_positive_cost']


def calculate_business_cost(y_true, y_pred) -> dict:
    from sklearn.metrics import confusion_matrix
    
    # Get confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate costs
    fn_cost = fn * BUSINESS_COSTS['false_negative_cost']
    fp_cost = fp * BUSINESS_COSTS['false_positive_cost']
    total_cost = fn_cost + fp_cost
    
    return {
        'false_negatives': int(fn),
        'false_positives': int(fp),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'fn_cost': fn_cost,
        'fp_cost': fp_cost,
        'total_cost': total_cost
    }


def print_business_context():
    print("BUSINESS COST ASSUMPTIONS")
    print(f"\nCost Definitions:")
    print(f"  False Negative (Missed Churn):  ${BUSINESS_COSTS['false_negative_cost']:>3d} per customer")
    print(f"  False Positive (Wasted Effort): ${BUSINESS_COSTS['false_positive_cost']:>3d} per customer")
    
    ratio = get_cost_ratio()
    print(f"\nCost Ratio: {ratio:.1f}:1")
    print(f"  → Missing a churner is {ratio:.0f}x more expensive than a false alarm")
    
    print(f"\nImplication:")
    print(f"  • This is an imbalanced cost problem (FN >> FP)")
    print(f"  • We should prioritize Recall over Precision")
    print(f"  • Standard accuracy metric is misleading")
    print(f"  • PR-AUC (Precision-Recall) is better than ROC-AUC for evaluation")


# Day 4: Threshold Optimization

def find_optimal_threshold(y_true, y_proba, thresholds=None):
    "Returns:  Dictionary with optimal threshold, cost, and threshold analysis"
    import numpy as np
    
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.01)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cost_breakdown = calculate_business_cost(y_true, y_pred)
        
        results.append({
            'threshold': threshold,
            'total_cost': cost_breakdown['total_cost'],
            'fn': cost_breakdown['false_negatives'],
            'fp': cost_breakdown['false_positives'],
            'tp': cost_breakdown['true_positives'],
            'tn': cost_breakdown['true_negatives'],
            'fn_cost': cost_breakdown['fn_cost'],
            'fp_cost': cost_breakdown['fp_cost']
        })
    
    # Find minimum cost threshold
    results_df = pd.DataFrame(results)
    optimal_idx = results_df['total_cost'].idxmin()
    optimal_result = results_df.iloc[optimal_idx]
    
    return {
        'optimal_threshold': optimal_result['threshold'],
        'optimal_cost': optimal_result['total_cost'],
        'optimal_fn': int(optimal_result['fn']),
        'optimal_fp': int(optimal_result['fp']),
        'threshold_analysis': results_df
    }


def compute_expected_utility(y_true, y_proba, threshold, cost_fn=100, cost_fp=10):
    """
    Compute expected utility (negative cost) for a given threshold.
    
    Expected Utility = -(cost_fn * FN + cost_fp * FP)

    Returns:
        Expected utility (negative cost, higher is better)
    """
    y_pred = (y_proba >= threshold).astype(int)
    cost_breakdown = calculate_business_cost(y_true, y_pred)
    
    # Utility is negative cost (we want to maximize utility = minimize cost)
    utility = -cost_breakdown['total_cost']
    
    return utility


def plot_cost_curve(y_true, y_proba, model_name="Model", save_path=None):
    """
    Shows:
    1. Total cost vs threshold
    2. FN and FP counts vs threshold
    3. Expected utility vs threshold
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Find optimal threshold
    optimal_result = find_optimal_threshold(y_true, y_proba)
    threshold_df = optimal_result['threshold_analysis']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Total Cost vs Threshold
    ax1 = axes[0, 0]
    ax1.plot(threshold_df['threshold'], threshold_df['total_cost'], 
             linewidth=2, color='red', label='Total Cost')
    ax1.axvline(optimal_result['optimal_threshold'], color='green', 
                linestyle='--', linewidth=2, label=f"Optimal Threshold = {optimal_result['optimal_threshold']:.3f}")
    ax1.axhline(optimal_result['optimal_cost'], color='green', 
                linestyle=':', alpha=0.5)
    ax1.set_xlabel('Classification Threshold', fontsize=12)
    ax1.set_ylabel('Total Business Cost ($)', fontsize=12)
    ax1.set_title(f'Cost vs Threshold: {model_name}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    
    # Plot 2: FN and FP Counts vs Threshold
    ax2 = axes[0, 1]
    ax2.plot(threshold_df['threshold'], threshold_df['fn'], 
             linewidth=2, color='orange', label=f'False Negatives (cost=${BUSINESS_COSTS["false_negative_cost"]})')
    ax2.plot(threshold_df['threshold'], threshold_df['fp'], 
             linewidth=2, color='blue', label=f'False Positives (cost=${BUSINESS_COSTS["false_positive_cost"]})')
    ax2.axvline(optimal_result['optimal_threshold'], color='green', 
                linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Classification Threshold', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Error Counts vs Threshold', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)
    
    # Plot 3: Cost Breakdown (FN vs FP costs)
    ax3 = axes[1, 0]
    ax3.plot(threshold_df['threshold'], threshold_df['fn_cost'], 
             linewidth=2, color='orange', label='FN Cost')
    ax3.plot(threshold_df['threshold'], threshold_df['fp_cost'], 
             linewidth=2, color='blue', label='FP Cost')
    ax3.axvline(optimal_result['optimal_threshold'], color='green', 
                linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Classification Threshold', fontsize=12)
    ax3.set_ylabel('Cost ($)', fontsize=12)
    ax3.set_title('Cost Breakdown: FN vs FP', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(alpha=0.3)
    
    # Plot 4: Expected Utility (negative cost)
    ax4 = axes[1, 1]
    utility = -threshold_df['total_cost']
    ax4.plot(threshold_df['threshold'], utility, 
             linewidth=2, color='purple', label='Expected Utility')
    ax4.axvline(optimal_result['optimal_threshold'], color='green', 
                linestyle='--', linewidth=2, label=f"Max Utility = ${-optimal_result['optimal_cost']:,.0f}")
    ax4.axhline(-optimal_result['optimal_cost'], color='green', 
                linestyle=':', alpha=0.5)
    ax4.set_xlabel('Classification Threshold', fontsize=12)
    ax4.set_ylabel('Expected Utility (Negative Cost)', fontsize=12)
    ax4.set_title('Expected Utility vs Threshold', fontsize=14, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, optimal_result


def print_threshold_summary(optimal_result, default_threshold=0.5):
    import numpy as np
    
    print(f"\nTHRESHOLD OPTIMIZATION SUMMARY")
    
    # Optimal threshold results
    print(f"\nOptimal Threshold: {optimal_result['optimal_threshold']:.3f}")
    print(f"  Total Cost: ${optimal_result['optimal_cost']:,.0f}")
    print(f"  False Negatives: {optimal_result['optimal_fn']}")
    print(f"  False Positives: {optimal_result['optimal_fp']}")
    
    # Find default threshold results for comparison
    threshold_df = optimal_result['threshold_analysis']
    default_idx = (np.abs(threshold_df['threshold'] - default_threshold)).idxmin()
    default_result = threshold_df.iloc[default_idx]
    
    print(f"\nDefault Threshold ({default_threshold}):")
    print(f"  Total Cost: ${default_result['total_cost']:,.0f}")
    print(f"  False Negatives: {int(default_result['fn'])}")
    print(f"  False Positives: {int(default_result['fp'])}")
    
    # Calculate improvement
    cost_savings = default_result['total_cost'] - optimal_result['optimal_cost']
    cost_improvement_pct = (cost_savings / default_result['total_cost']) * 100
    
    print(f"\nImprovement:")
    print(f"  Cost Savings: ${cost_savings:,.0f} ({cost_improvement_pct:+.1f}%)")
    print(f"  FN Change: {int(default_result['fn']) - optimal_result['optimal_fn']:+d}")
    print(f"  FP Change: {int(default_result['fp']) - optimal_result['optimal_fp']:+d}")
    
