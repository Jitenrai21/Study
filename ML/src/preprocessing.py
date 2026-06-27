# src/preprocessing.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# Feature definitions
NUMERICAL = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# NOTE: TotalCharges has perfect multicollinearity with tenure × MonthlyCharges
# dropping TotalCharges in future iterations for model interpretability
# For now, keeping all features for baseline model

# Numerical preprocessing sub-pipeline (impute THEN scale)
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),  # Leakage-free: 0 for new customers
    ('scaler', StandardScaler())
])

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, NUMERICAL),
        ('cat', OneHotEncoder(
            categories='auto', 
            drop='first',  # Avoid dummy variable trap
            handle_unknown='ignore'  # Production safety for unseen categories
        ), CATEGORICAL)
    ],
    remainder='drop'  # No other columns
)

# Full pipeline template (add model later)
base_pipeline = Pipeline([
    ('preprocessor', preprocessor)
])
