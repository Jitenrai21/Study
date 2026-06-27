"Day 2-3: Baseline Models, Imbalance Handling, and Advanced Models"
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from preprocessing import preprocessor
from typing import Literal


def get_majority_baseline(random_state: int = 42) -> Pipeline:
    "Returns: Pipeline with DummyClassifier (strategy='most_frequent')"
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DummyClassifier(strategy='most_frequent', random_state=random_state))
    ]) # Note: Establishes "what if we did nothing smart?"—expect ~0.73 accuracy but 0 recall for churn.

def get_stratified_baseline(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DummyClassifier(strategy='stratified', random_state=random_state))
    ]) # Tests if your model beats "random proportional(e.g., 27% chance of churn) guessing"—useful for AUC baselines.

def get_logistic_baseline(
    class_weight: Literal['balanced', None] = None, #penalizes FN more (churn is rare)
    max_iter: int = 1000,
    random_state: int = 42
) -> Pipeline:
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs'  # Good for small datasets (~5k train rows)
        ))
    ]) 

def get_logistic_smote(
    k_neighbors: int = 5,
    max_iter: int = 1000,
    random_state: int = 42,
    C: float = 1.0,
    penalty: str = 'l2',
    **kwargs
) -> ImbPipeline:
    """
    Logistic Regression with SMOTE oversampling.
    Uses imblearn Pipeline to apply SMOTE before model training.
    
    Args:
        k_neighbors: Number of neighbors for SMOTE
        max_iter: Maximum iterations for LogisticRegression
        random_state: Random state for reproducibility
        C: Regularization strength (inverse)
        penalty: Regularization penalty ('l2' or 'l1')
        **kwargs: Additional parameters for LogisticRegression
        
    Returns:
        imblearn Pipeline with SMOTE + LogisticRegression
        
    Note:
        SMOTE is applied AFTER preprocessing, BEFORE model training.
        This is leakage-free because SMOTE only sees training data.
    """
    return ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(
            k_neighbors=k_neighbors,
            random_state=random_state,
            sampling_strategy='auto'  # Balance minority to majority
        )),
        ('classifier', LogisticRegression(
            C=C,
            penalty=penalty,
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs',
            **kwargs
        ))
    ])


# Day 3: Advanced Models
def get_xgboost_baseline(
    random_state: int = 42,
    **kwargs
) -> Pipeline:
    "XGBoost baseline model without imbalance handling."
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            **kwargs
        ))
    ])


def get_xgboost_balanced(
    scale_pos_weight: float = 2.77,  # Default: imbalance ratio from Day 2
    random_state: int = 42,
    **kwargs
) -> Pipeline:
    "XGBoost with scale_pos_weight for imbalance handling."
    #Note: scale_pos_weight = sum(negative instances) / sum(positive instances) Day 2 found ratio of 2.77:1, so default is 2.77
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            **kwargs
        ))
    ])


def get_random_forest_baseline(
    random_state: int = 42,
    **kwargs
) -> Pipeline:
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            random_state=random_state,
            **kwargs
        ))
    ])


def get_random_forest_balanced(
    class_weight: Literal['balanced', 'balanced_subsample'] = 'balanced',
    random_state: int = 42,
    **kwargs
) -> Pipeline:
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            class_weight=class_weight,
            random_state=random_state,
            **kwargs
        ))
    ])

# Model registry for easy access
MODEL_REGISTRY = {
    # Day 2: Baselines
    'majority_baseline': get_majority_baseline,
    'stratified_baseline': get_stratified_baseline,
    'logistic_baseline': get_logistic_baseline,
    'logistic_balanced': lambda **kwargs: get_logistic_baseline(class_weight='balanced', **kwargs),
    'logistic_smote': get_logistic_smote,
    
    # Day 3: Advanced Models
    'xgboost_baseline': get_xgboost_baseline,
    'xgboost_balanced': get_xgboost_balanced,
    'rf_baseline': get_random_forest_baseline,
    'rf_balanced': get_random_forest_balanced,
}


def get_model(model_name: str, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name](**kwargs)
