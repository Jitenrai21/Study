# src/day1_pipeline.py
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_and_validate_data, stratified_split
from preprocessing import base_pipeline
from utils import print_business_context
from sklearn import set_config
set_config(display='diagram')


if __name__ == "__main__":
    print("DAY 1: LEAKAGE-PROOF PIPELINE SETUP")
    
    # Dynamic path resolution
    data_path = Path(__file__).parent.parent / "data" / "telco-customer-churn-by-IBM.csv"
    
    # Load & split
    df = load_and_validate_data(str(data_path))
    train, test = stratified_split(df)
    
    # Leakage-proof: fit only on train
    X_train, y_train = train.drop('Churn', axis=1), train['Churn']
    base_pipeline.fit(X_train, y_train)
    
    print("PIPELINE VALIDATION")
    print(f"Train features shape: {base_pipeline.transform(X_train).shape}")
    print(f"Feature names (first 10): {base_pipeline.named_steps['preprocessor'].get_feature_names_out()[:10]}")
    
    # Business Context (Day 1 requirement)
    print_business_context()
    
    print("✅ DAY 1 COMPLETE: FOUNDATION READY FOR MODELING")