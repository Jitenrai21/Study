"""
Day 3: Model Suite & Nested CV
Compare models rigorously with nested cross-validation and cost-aware tuning
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
    get_cost_scorer,
    analyze_class_imbalance
)
from utils import calculate_business_cost, BUSINESS_COSTS

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV #StratifiedKFold (preserves imbalance), GridSearchCV (inner tuning)
from sklearn.metrics import make_scorer, average_precision_score, recall_score

# Load environment variables
load_dotenv()

# Configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', './mlruns')
MLFLOW_EXPERIMENT_NAME = 'telco-churn-nested-cv'
RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))
OUTER_CV_FOLDS = 5
INNER_CV_FOLDS = 3

# Hyperparameter grids optimized for FN reduction
PARAM_GRIDS = {
    'xgboost_balanced': {
        'classifier__max_depth': [3, 5, 7], # Tree depth (complexity)
        'classifier__learning_rate': [0.01, 0.1, 0.2], # Step size (slower = more careful learning)
        'classifier__scale_pos_weight': [2.77, 3.5, 5.0],  # Imbalance ratios
        'classifier__min_child_weight': [1, 3, 5],  # Higher = fewer FN
        'classifier__n_estimators': [100, 200], # Number of trees
        'classifier__subsample': [0.8, 1.0] # Row sampling (regularization)
    },
    'rf_balanced': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],  # Smaller = fewer FN
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    },
    'logistic_smote': {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__penalty': ['l2'],
        'smote__k_neighbors': [3, 5, 7]
    }
}


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}\n")


def nested_cv_with_tuning(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
):
    """
    Nested cross-validation with cost-aware hyperparameter tuning.
    
    Outer CV: 5-fold StratifiedKFold (unbiased performance estimate)
    Inner CV: 3-fold GridSearchCV with cost-sensitive scoring
    
    Returns:
        Dictionary with fold results, best model, and stability metrics
    """
    print(f"MODEL: {model_name.upper()}")
    
    # Get base model and param grid
    base_model = get_model(model_name)
    param_grid = PARAM_GRIDS.get(model_name, {})
    
    if not param_grid:
        print(f"No param grid defined for {model_name}, using default hyperparameters")
        # Train on full train set, evaluate on test set
        base_model.fit(X_train, y_train)
        y_pred = base_model.predict(X_test)
        y_proba = base_model.predict_proba(X_test)[:, 1]
        
        test_metrics = compute_metrics(y_test.values, y_pred, y_proba)
        cost_breakdown = calculate_business_cost(y_test.values, y_pred)
        
        return {
            'model_name': model_name,
            'best_params': {},
            'test_metrics': test_metrics,
            'cost_breakdown': cost_breakdown,
            'fold_costs': [cost_breakdown['total_cost']],
            'fold_fn_counts': [cost_breakdown['false_negatives']],
            'best_model': base_model
        }
    
    # Setup cross-validation
    outer_cv = StratifiedKFold(n_splits=OUTER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Scoring metrics
    cost_scorer = get_cost_scorer()
    scoring = {
        'cost': cost_scorer,
        'pr_auc': make_scorer(average_precision_score, needs_proba=True),
        'recall': make_scorer(recall_score)
    }
    
    # Track results across folds
    fold_results = {
        'costs': [],
        'fn_counts': [],
        'fp_counts': [],
        'pr_aucs': [],
        'recalls': [],
        'best_params': []
    }
    
    print(f"\nOuter CV: {OUTER_CV_FOLDS} folds | Inner CV: {INNER_CV_FOLDS} folds (GridSearchCV)")
    
    # Calculate total hyperparameter combinations
    if param_grid:
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"Hyperparameter space: {int(total_combinations)} combinations")
    else:
        print(f"Hyperparameter space: 0 combinations (using defaults)")
    
    print(f"Optimization metric: Business Cost (FN=$100, FP=$10)")
    
    # Nested CV loop
    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train), 1):
        print(f"\n--- Fold {fold_idx}/{OUTER_CV_FOLDS} ---")
        
        # Split data
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Inner CV: Hyperparameter tuning with GridSearchCV
        grid_search = GridSearchCV(
            estimator=get_model(model_name),
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            refit='cost',  # Choose best params by cost
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_fold_train, y_fold_train)
        
        # Best model from inner CV
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Evaluate on outer fold validation set
        y_pred = best_model.predict(X_fold_val)
        y_proba = best_model.predict_proba(X_fold_val)[:, 1]
        
        # Compute metrics
        metrics = compute_metrics(y_fold_val.values, y_pred, y_proba)
        cost_breakdown = calculate_business_cost(y_fold_val.values, y_pred)
        
        # Store results
        fold_results['costs'].append(cost_breakdown['total_cost'])
        fold_results['fn_counts'].append(cost_breakdown['false_negatives'])
        fold_results['fp_counts'].append(cost_breakdown['false_positives'])
        fold_results['pr_aucs'].append(metrics['pr_auc'])
        fold_results['recalls'].append(metrics['recall'])
        fold_results['best_params'].append(best_params)
        
        print(f"  Best params: {best_params}")
        print(f"  Cost: ${cost_breakdown['total_cost']:,} | FN: {cost_breakdown['false_negatives']} | FP: {cost_breakdown['false_positives']}")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f} | Recall: {metrics['recall']:.4f}")
    
    # Aggregate statistics
    print(f"\nNESTED CV RESULTS (Outer Fold Statistics)")
    print(f"Cost:    ${np.mean(fold_results['costs']):>7,.0f} ± ${np.std(fold_results['costs']):>6,.0f}")
    print(f"FN:      {np.mean(fold_results['fn_counts']):>7.1f} ± {np.std(fold_results['fn_counts']):>6.1f}")
    print(f"FP:      {np.mean(fold_results['fp_counts']):>7.1f} ± {np.std(fold_results['fp_counts']):>6.1f}")
    print(f"PR-AUC:  {np.mean(fold_results['pr_aucs']):>7.4f} ± {np.std(fold_results['pr_aucs']):>6.4f}")
    print(f"Recall:  {np.mean(fold_results['recalls']):>7.4f} ± {np.std(fold_results['recalls']):>6.4f}")
    
    # Train final model on full training set with best hyperparameters
    print(f"\nTraining final model on full training set...")
    
    # Use most common best params across folds
    from collections import Counter
    param_counts = Counter([str(p) for p in fold_results['best_params']])
    most_common_params_str = param_counts.most_common(1)[0][0]
    final_params = eval(most_common_params_str)
    
    print(f"  Using params: {final_params}")
    
    # Strip 'classifier__' and 'smote__' prefixes from GridSearchCV params
    # GridSearchCV uses pipeline step names as prefixes, but get_model() doesn't need them
    cleaned_params = {}
    for key, value in final_params.items():
        if key.startswith('classifier__'):
            cleaned_params[key.replace('classifier__', '')] = value
        elif key.startswith('smote__'):
            cleaned_params[key.replace('smote__', '')] = value
        else:
            cleaned_params[key] = value
    
    final_model = get_model(model_name, **cleaned_params)
    final_model.fit(X_train, y_train)
    
    # Evaluate on held-out test set
    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    
    test_metrics = compute_metrics(y_test.values, y_test_pred, y_test_proba)
    test_cost_breakdown = calculate_business_cost(y_test.values, y_test_pred)
    
    print(f"\nHELD-OUT TEST SET PERFORMANCE:")
    print(f"  Cost: ${test_cost_breakdown['total_cost']:,}")
    print(f"  FN: {test_cost_breakdown['false_negatives']} | FP: {test_cost_breakdown['false_positives']}")
    print(f"  PR-AUC: {test_metrics['pr_auc']:.4f} | Recall: {test_metrics['recall']:.4f}")
    
    return {
        'model_name': model_name,
        'best_params': final_params,
        'cv_stats': {
            'cost_mean': np.mean(fold_results['costs']),
            'cost_std': np.std(fold_results['costs']),
            'fn_mean': np.mean(fold_results['fn_counts']),
            'fn_std': np.std(fold_results['fn_counts']),
            'pr_auc_mean': np.mean(fold_results['pr_aucs']),
            'pr_auc_std': np.std(fold_results['pr_aucs'])
        },
        'test_metrics': test_metrics,
        'test_cost': test_cost_breakdown,
        'fold_results': fold_results,
        'best_model': final_model
    }


def main():
    print("DAY 3: MODEL SUITE & NESTED CV")
    
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
    print("\nCLASS IMBALANCE ANALYSIS")
    train_imbalance = analyze_class_imbalance(y_train.values)
    print(f"Training Set: {train_imbalance['imbalance_ratio']:.2f}:1 ({train_imbalance['severity'].upper()})")
    print(f"  Minority (Churn): {train_imbalance['minority_count']} ({train_imbalance['minority_percentage']:.1f}%)")
    
    # Business context
    print(f"\nBUSINESS COST CONTEXT")
    print(f"  False Negative (Missed Churn): ${BUSINESS_COSTS['false_negative_cost']}")
    print(f"  False Positive (Wasted Effort): ${BUSINESS_COSTS['false_positive_cost']}")
    print(f"  Cost Ratio: {BUSINESS_COSTS['false_negative_cost'] / BUSINESS_COSTS['false_positive_cost']:.0f}:1")
    
    # Day 2 Baseline (for comparison)
    print(f"\nDAY 2 BASELINE TO BEAT (logistic_smote):")
    print(f"  Cost: $10,510 | FN: 76 | FP: 291")
    
    # Models to evaluate
    models_to_test = ['xgboost_balanced', 'rf_balanced', 'logistic_smote']
    
    results = {}
    
    # Train and evaluate each model with nested CV
    for idx, model_name in enumerate(models_to_test, 1):
        result = nested_cv_with_tuning(model_name, X_train, y_train, X_test, y_test)
        results[model_name] = result
        
        # Create descriptive run name with hierarchy
        # Format: Day3_{index}_{model}_{cost_optimized}
        run_name = f"Day3_{idx:02d}_{model_name}_cost_optimized"
        
        # Log to MLflow with enhanced metadata
        with mlflow.start_run(run_name=run_name):
            # Add run tags for better organization
            mlflow.set_tags({
                'day': 'day3',
                'task': 'nested_cv_hyperparameter_tuning',
                'model_family': model_name.split('_')[0],  # xgboost, rf, logistic
                'imbalance_strategy': model_name.split('_')[-1] if '_' in model_name else 'none',
                'optimization_metric': 'business_cost',
                'pipeline_stage': 'production_candidate'
            })
            
            # Log hyperparameters with prefix for clarity
            for param_name, param_value in result['best_params'].items():
                mlflow.log_param(f"best_{param_name}", param_value)
            
            # Log CV configuration
            mlflow.log_params({
                'model_name': model_name,
                'outer_cv_folds': OUTER_CV_FOLDS,
                'inner_cv_folds': INNER_CV_FOLDS,
                'total_cv_iterations': OUTER_CV_FOLDS * INNER_CV_FOLDS,
                'random_state': RANDOM_STATE
            })
            
            # Log CV statistics with proper naming
            for stat_name, stat_value in result['cv_stats'].items():
                mlflow.log_metric(f"nested_cv_{stat_name}", stat_value)
            
            # Log test set performance metrics
            for metric_name, value in result['test_metrics'].items():
                mlflow.log_metric(f"holdout_test_{metric_name}", value)
            
            # Log business cost breakdown (primary optimization target)
            mlflow.log_metric("business_cost_total", result['test_cost']['total_cost'])
            mlflow.log_metric("business_cost_fn", result['test_cost']['fn_cost'])
            mlflow.log_metric("business_cost_fp", result['test_cost']['fp_cost'])
            mlflow.log_metric("confusion_fn_count", result['test_cost']['false_negatives'])
            mlflow.log_metric("confusion_fp_count", result['test_cost']['false_positives'])
            mlflow.log_metric("confusion_tn_count", result['test_cost']['true_negatives'])
            mlflow.log_metric("confusion_tp_count", result['test_cost']['true_positives'])
            
            # Log model
            mlflow.sklearn.log_model(result['best_model'], "model")
            
            print(f"✅ Logged to MLflow: {model_name}")
    
    # Final comparison
    print(f"\nFINAL MODEL COMPARISON (Held-Out Test Set)")
    
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'Test_Cost': result['test_cost']['total_cost'],
            'FN': result['test_cost']['false_negatives'],
            'FP': result['test_cost']['false_positives'],
            'PR-AUC': result['test_metrics']['pr_auc'],
            'Recall': result['test_metrics']['recall'],
            'CV_Cost_Mean': result['cv_stats']['cost_mean'],
            'CV_Cost_Std': result['cv_stats']['cost_std']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test_Cost')
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_cost = comparison_df.iloc[0]['Test_Cost']
    best_fn = comparison_df.iloc[0]['FN']
    
    print(f"\nWINNER: {best_model_name.upper()}")
    print(f"Test Set Cost: ${best_cost:,.0f}")
    print(f"False Negatives: {int(best_fn)}")
    
    # Compare to Day 2 baseline
    baseline_cost = 10510
    baseline_fn = 76
    
    if best_cost < baseline_cost:
        savings = baseline_cost - best_cost
        print(f"\nIMPROVEMENT over Day 2 baseline:")
        print(f"Cost savings: ${savings:,.0f} ({savings/baseline_cost*100:.1f}%)")
    else:
        print(f"\nDid not beat Day 2 baseline (${baseline_cost:,})")
    
    if best_fn < baseline_fn:
        fn_reduction = baseline_fn - best_fn
        print(f"   FN reduction: {fn_reduction} fewer missed churners ({fn_reduction/baseline_fn*100:.1f}%)")
    
    # Save best model in a separate run for easy retrieval
    print(f"\nSaving best model to MLflow...")
    best_result = results[best_model_name]
    
    with mlflow.start_run(run_name=f"Day3_BEST_{best_model_name}_winner"):
        mlflow.set_tags({
            'day': 'day3',
            'status': 'WINNER',
            'model_name': best_model_name,
            'optimization_metric': 'business_cost',
            'pipeline_stage': 'production_ready'
        })
        
        # Log best params
        for param_name, param_value in best_result['best_params'].items():
            mlflow.log_param(f"best_{param_name}", param_value)
        
        # Log final metrics
        mlflow.log_metric("business_cost_total", best_result['test_cost']['total_cost'])
        mlflow.log_metric("confusion_fn_count", best_result['test_cost']['false_negatives'])
        mlflow.log_metric("confusion_fp_count", best_result['test_cost']['false_positives'])
        mlflow.log_metric("holdout_test_pr_auc", best_result['test_metrics']['pr_auc'])
        mlflow.log_metric("holdout_test_recall", best_result['test_metrics']['recall'])
        
        # Save the best model
        mlflow.sklearn.log_model(
            best_result['best_model'], 
            "model",
            registered_model_name="telco_churn_best_model"
        )
        
        print(f"✅ Best model saved: {best_model_name}")
        print(f"   Model registered as: telco_churn_best_model")
    
    print(f"\nDAY 3 COMPLETE!")
    print(f"Models evaluated: {len(models_to_test)}")
    print(f"Best model: {best_model_name}")
    print(f"Nested CV: {OUTER_CV_FOLDS} outer folds × {INNER_CV_FOLDS} inner folds")
    print(f"MLflow UI: mlflow ui --port 5000")


if __name__ == "__main__":
    main()
