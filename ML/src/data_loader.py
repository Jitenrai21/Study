# src/data_loader.py
import pandas as pd
import hashlib
import pandera as pa
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


def compute_checksum(file_path: Path) -> str:
    "House Rules: Verify data integrity. → Computes MD5 checksum of the exact file bytes"
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# Pandera schema with corrected syntax
schema = pa.DataFrameSchema({
    "gender": pa.Column(pa.String, pa.Check.isin(["Female", "Male"])),
    "SeniorCitizen": pa.Column("int64", pa.Check.isin([0, 1])),  # Explicit int64
    "Partner": pa.Column(pa.String, pa.Check.isin(["Yes", "No"])),
    "Dependents": pa.Column(pa.String, pa.Check.isin(["Yes", "No"])),
    "tenure": pa.Column("int64", pa.Check.in_range(0, 72)),  # Explicit int64
    "PhoneService": pa.Column(pa.String, pa.Check.isin(["Yes", "No"])),
    "MultipleLines": pa.Column(pa.String, pa.Check.isin(["Yes", "No", "No phone service"])),
    "InternetService": pa.Column(pa.String, pa.Check.isin(["DSL", "Fiber optic", "No"])),
    "OnlineSecurity": pa.Column(pa.String, pa.Check.isin(["Yes", "No", "No internet service"])),
    "OnlineBackup": pa.Column(pa.String, pa.Check.isin(["Yes", "No", "No internet service"])),
    "DeviceProtection": pa.Column(pa.String, pa.Check.isin(["Yes", "No", "No internet service"])),
    "TechSupport": pa.Column(pa.String, pa.Check.isin(["Yes", "No", "No internet service"])),
    "StreamingTV": pa.Column(pa.String, pa.Check.isin(["Yes", "No", "No internet service"])),
    "StreamingMovies": pa.Column(pa.String, pa.Check.isin(["Yes", "No", "No internet service"])),
    "Contract": pa.Column(pa.String, pa.Check.isin(["Month-to-month", "One year", "Two year"])),
    "PaperlessBilling": pa.Column(pa.String, pa.Check.isin(["Yes", "No"])),
    "PaymentMethod": pa.Column(pa.String, pa.Check.isin([
        "Electronic check", "Mailed check", "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ])),
    "MonthlyCharges": pa.Column("float64", pa.Check.in_range(0, 120)),
    "TotalCharges": pa.Column("float64", pa.Check.in_range(0, 10000, include_min=True), nullable=True),  # Nulls handled by pipeline
    "Churn": pa.Column("int64", pa.Check.isin([0, 1]))
}, strict=True)  # Strict mode: fail on unexpected columns


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    "Load, clean, and validate telco customer churn data. Returns: Validated and cleaned DataFrame"
    # Convert to absolute path if needed
    path = Path(file_path)
    if not path.is_absolute():
        # Assume relative to project root
        path = Path(__file__).parent.parent / path
    
    df = pd.read_csv(path)
    
    # LOG CHECKSUM
    checksum = compute_checksum(path)
    print(f"Data checksum: {checksum}")
    
    # CLEANING (fixes your issues)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # Fix spaces
    
    # NOTE: TotalCharges nulls (11 rows) will be handled by pipeline imputer to prevent any potential leakage from pre-split imputation
    
    # Ensure integer columns are int64 (avoid int32/int64 Pandera mismatches)
    df['tenure'] = df['tenure'].astype('int64')
    df['SeniorCitizen'] = df['SeniorCitizen'].astype('int64')
    
    # TARGET ENCODING (safe to do before split)
    # Churn is the target (y), not a feature (X)
    # It's separated from features before pipeline processing
    # Deterministic mapping: "Yes" → 1, "No" → 0 (no statistics computed)
    df['Churn'] = (df['Churn'] == 'Yes').astype('int64')  # Explicit int64 for Pandera
    
    # Drop ID (leakage prevention)
    df = df.drop('customerID', axis=1)
    
    # Validate
    validated_df = schema(df)  # Pandera fails fast if drift
    
    print(f" Loaded: {len(df)} rows, Churn rate: {df['Churn'].mean():.1%}")
    return validated_df


def stratified_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    "Leakage-proof stratified train-test split. Returns: Tuple of (train_df, test_df)"
    train, test = train_test_split(
        df, test_size=test_size, 
        stratify=df['Churn'],  # Preserves class balance
        random_state=random_state
    )
    print(f"Train: {len(train)} ({train['Churn'].mean():.1%} churn)")
    print(f"Test:  {len(test)} ({test['Churn'].mean():.1%} churn)")
    return train, test
