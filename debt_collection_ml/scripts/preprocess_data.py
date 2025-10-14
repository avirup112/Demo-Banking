#!/usr/bin/env python3
import sys
import os
import yaml
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_preprocessor import AdvancedDataPreprocessor

def main():
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    preprocess_params = params['preprocess']
    
    # Load data
    df = pd.read_csv('data/raw/debt_collection_data.csv')
    
    # Initialize preprocessor
    preprocessor = AdvancedDataPreprocessor(**preprocess_params)
    
    # Preprocess data
    X_processed, y_encoded = preprocessor.fit_transform(df)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/X_processed.npy', X_processed)
    np.save('data/processed/y_encoded.npy', y_encoded)
    
    # Save preprocessor
    os.makedirs('models/artifacts', exist_ok=True)
    preprocessor.save_preprocessor('models/artifacts/preprocessor.joblib')
    
    print(f"Preprocessed data shape: {X_processed.shape}")

if __name__ == "__main__":
    main()