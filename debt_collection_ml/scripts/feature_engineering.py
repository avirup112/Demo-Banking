#!/usr/bin/env python3
import sys
import os
import yaml
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.feature_engineering import AdvancedFeatureEngineer
from data.data_preprocessor import AdvancedDataPreprocessor

def main():
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    fe_params = params['feature_engineering']
    
    # Load processed data
    X_processed = np.load('data/processed/X_processed.npy')
    y_encoded = np.load('data/processed/y_encoded.npy')
    
    # Load preprocessor to get feature names
    preprocessor = AdvancedDataPreprocessor()
    preprocessor.load_preprocessor('models/artifacts/preprocessor.joblib')
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(X_processed, columns=preprocessor.feature_names)
    
    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineer(**fe_params)
    
    # Engineer features
    X_engineered = feature_engineer.fit_transform(feature_df, y_encoded)
    
    # Save engineered features
    np.save('data/processed/X_engineered.npy', X_engineered)
    
    # Save feature engineer
    feature_engineer.save_feature_engineer('models/artifacts/feature_engineer.joblib')
    
    print(f"Engineered features shape: {X_engineered.shape}")

if __name__ == "__main__":
    main()