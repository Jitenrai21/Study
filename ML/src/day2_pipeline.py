"""
Day 2: Baselines & Class Imbalance
Train baseline models, handle imbalance, evaluate with PR curves, log to MLflow
"""
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_and_validate_data, stratified_split
from modeling import get_model
from evaluation import (
    compute_metrics,
    plot_pr_curve,
    plot_confusion_matrix,
    compare_models_pr,
    print_classification_report,
    analyze_class_imbalance
)
from utils import calculate_business_cost

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', './mlruns')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'telco-churn-baselines')
RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}\n")

def train_and_evaluate_model(
    model_pipeline,
    model_name: str,
    X_train, y_train,
    X_test, y_test,
    log_mlflow: bool = True
):
    print(f"Training: {model_name}")
    
    if log_mlflow:
        mlflow.start_run(run_name=model_name)
    
    try:
        # Train model
        model_pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model_pipeline.predict(X_train)
        y_pred_test = model_pipeline.predict(X_test)
        
        # Probabilities (handle DummyClassifier that may not have predict_proba)
        try:
            y_proba_train = model_pipeline.predict_proba(X_train)[:, 1]
            y_proba_test = model_pipeline.predict_proba(X_test)[:, 1]
        except AttributeError:
            # DummyClassifier with strategy='most_frequent' doesn't have predict_proba
            y_proba_train = y_pred_train.astype(float)
            y_proba_test = y_pred_test.astype(float)
        
        # Compute metrics
        train_metrics = compute_metrics(y_train, y_pred_train, y_proba_train)
        test_metrics = compute_metrics(y_test, y_pred_test, y_proba_test)
        
        # Print results
        print(f"\nTraining Metrics:")
        print(f"  PR-AUC:    {train_metrics['pr_auc']:.4f}")
        print(f"  F1-Score:  {train_metrics['f1_score']:.4f}")
        print(f"  Precision: {train_metrics['precision']:.4f}")
        print(f"  Recall:    {train_metrics['recall']:.4f}")
        
        print(f"\nTest Metrics:")
        print(f"  PR-AUC:    {test_metrics['pr_auc']:.4f}")
        print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        
        # Classification report
        print_classification_report(y_test, y_pred_test, model_name)
       
        # Log to MLflow
        if log_mlflow:
            # Parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", RANDOM_STATE)
            
            # Metrics
            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)
            
            # Create plots directory
            plots_dir = Path('../deliverables/plots/day2')
            plots_dir.mkdir(exist_ok=True)
            
            # Plot and log PR curve
            pr_fig = plot_pr_curve(
                y_test, y_proba_test, model_name,
                save_path=plots_dir / f'{model_name}_pr_curve.png'
            )
            mlflow.log_artifact(str(plots_dir / f'{model_name}_pr_curve.png'))
            
            # Plot and log confusion matrix
            cm_fig = plot_confusion_matrix(
                y_test, y_pred_test, model_name,
                save_path=plots_dir / f'{model_name}_confusion_matrix.png'
            )
            mlflow.log_artifact(str(plots_dir / f'{model_name}_confusion_matrix.png'))
            
            # Log model
            mlflow.sklearn.log_model(model_pipeline, "model")
            
            print(f"✅ Logged to MLflow: {model_name}")
        
        return {
            'model_name': model_name,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'y_pred_test': y_pred_test,
            'y_proba_test': y_proba_test
        }
    
    finally:
        if log_mlflow:
            mlflow.end_run()


def main():
    print("DAY 2: BASELINES & CLASS IMBALANCE")
    
    # Setup MLflow
    setup_mlflow()
    
    # Load and split data
    data_path = Path(__file__).parent.parent / "data" / "telco-customer-churn-by-IBM.csv"
    df = load_and_validate_data(str(data_path))
    train_df, test_df = stratified_split(df, test_size=0.2, random_state=RANDOM_STATE)
    
    # Prepare data
    X_train, y_train = train_df.drop('Churn', axis=1), train_df['Churn']
    X_test, y_test = test_df.drop('Churn', axis=1), test_df['Churn']
    
    # Analyze class imbalance
    print("CLASS IMBALANCE ANALYSIS")
    train_imbalance = analyze_class_imbalance(y_train.values)
    test_imbalance = analyze_class_imbalance(y_test.values)
    
    print(f"\nTraining Set:")
    print(f"  Total samples: {train_imbalance['total_samples']}")
    print(f"  Majority (No Churn): {train_imbalance['majority_count']} ({100-train_imbalance['minority_percentage']:.1f}%)")
    print(f"  Minority (Churn):    {train_imbalance['minority_count']} ({train_imbalance['minority_percentage']:.1f}%)")
    print(f"  Imbalance ratio: {train_imbalance['imbalance_ratio']:.2f}:1")
    print(f"  Severity: {train_imbalance['severity'].upper()}")
    
    print(f"\nTest Set:")
    print(f"  Total samples: {test_imbalance['total_samples']}")
    print(f"  Imbalance ratio: {test_imbalance['imbalance_ratio']:.2f}:1")
    
    # Decision on imbalance handling
    print(f"\nImbalance Strategy Decision:")
    if train_imbalance['imbalance_ratio'] > 2:
        print(f"  Ratio {train_imbalance['imbalance_ratio']:.2f}:1 indicates MODERATE imbalance")
        print(f"  Testing: class_weight='balanced' AND SMOTE")
        print(f"  Justification: PR-AUC will guide final choice")
    else:
        print(f"  Ratio {train_imbalance['imbalance_ratio']:.2f}:1 is relatively balanced")
        print(f"  Testing: standard model and class_weight='balanced'")
    
    # Train models
    results = {}
    
    # 1. Majority Baseline (DummyClassifier - most_frequent)
    print("\nMODEL 1/4: Dummy Classifier (Most Frequent)")
    results['majority_baseline'] = train_and_evaluate_model(
        get_model('majority_baseline'),
        'majority_baseline',
        X_train, y_train, X_test, y_test
    )
    
    # 2. Stratified Baseline (DummyClassifier - stratified)
    print("\nMODEL 2/4: Dummy Classifier (Stratified)")
    results['stratified_baseline'] = train_and_evaluate_model(
        get_model('stratified_baseline'),
        'stratified_baseline',
        X_train, y_train, X_test, y_test
    )
    
    # 3. Logistic Regression with class_weight='balanced'
    print("\nMODEL 3/4: Logistic Regression (Balanced Weights)")
    results['logistic_balanced'] = train_and_evaluate_model(
        get_model('logistic_balanced'),
        'logistic_balanced',
        X_train, y_train, X_test, y_test
    )
    
    # 4. Logistic Regression with SMOTE
    print("\nMODEL 4/4: Logistic Regression (SMOTE)")
    results['logistic_smote'] = train_and_evaluate_model(
        get_model('logistic_smote'),
        'logistic_smote',
        X_train, y_train, X_test, y_test
    )
    
    # Compare models
    print("MODEL COMPARISON")
    
    comparison_df = pd.DataFrame({
        name: {
            'PR-AUC': data['test_metrics']['pr_auc'],
            'F1-Score': data['test_metrics']['f1_score'],
            'Precision': data['test_metrics']['precision'],
            'Recall': data['test_metrics']['recall'],
            'ROC-AUC': data['test_metrics']['roc_auc']
        }
        for name, data in results.items()
    }).T
    
    print("\n" + comparison_df.to_string())
    
    # Best model by PR-AUC
    best_model_prauc = comparison_df['PR-AUC'].idxmax()
    print(f"\nBest Model (by PR-AUC): {best_model_prauc}")
    print(f"   PR-AUC: {comparison_df.loc[best_model_prauc, 'PR-AUC']:.4f}")
    
    # Business cost comparison
    print("\nBUSINESS COST COMPARISON")
    print(f"\nCost Assumptions:")
    print(f"  False Negative (Missed Churn):  $100 per customer")
    print(f"  False Positive (Wasted Effort): $10 per customer")
    
    cost_comparison = []
    for name, data in results.items():
        cost_breakdown = calculate_business_cost(y_test, data['y_pred_test'])
        cost_comparison.append({
            'Model': name,
            'FN': cost_breakdown['false_negatives'],
            'FP': cost_breakdown['false_positives'],
            'FN_Cost': f"${cost_breakdown['fn_cost']:,}",
            'FP_Cost': f"${cost_breakdown['fp_cost']:,}",
            'Total_Cost': cost_breakdown['total_cost']
        })
    
    cost_df = pd.DataFrame(cost_comparison)
    cost_df = cost_df.sort_values('Total_Cost')
    
    print("\n" + cost_df.to_string(index=False))
    
    best_cost_model = cost_df.iloc[0]['Model']
    best_cost = cost_df.iloc[0]['Total_Cost']
    print(f"\nLowest Cost Model: {best_cost_model}")
    print(f"   Total Cost: ${best_cost:,}")
    
    # Create PR curve comparison plot
    plots_dir = Path('../deliverables/plots/day2')
    models_data = {
        name: (y_test.values, data['y_proba_test'])
        for name, data in results.items()
        if 'y_proba_test' in data
    }
    compare_fig = compare_models_pr(models_data, save_path=plots_dir / 'model_comparison.png')
    
    print(f"\nPR curve comparison saved: plots/model_comparison.png")
    
    # Final summary
    print("\nDAY 2 COMPLETE!")
    print(f"Models Trained: 4")
    print(f"Best PR-AUC: {comparison_df['PR-AUC'].max():.4f}")
    print(f"Artifacts saved: ./plots/")
    print(f"MLflow experiments: mlflow ui --port 5000")

if __name__ == "__main__":
    main()
