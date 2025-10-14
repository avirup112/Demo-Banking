#!/usr/bin/env python3
"""
Explainability Analysis for Debt Collection ML Models
"""

import sys
import os
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import joblib

warnings.filterwarnings('ignore')
sys.path.append('src')

from explainability.shap_explainer import DebtCollectionExplainer
from utils.dagshub_integration import DagsHubTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_data():
    """Load the best trained model and test data"""
    
    logger.info("Loading trained model and data...")
    
    # Load model
    model_files = list(Path("models/optimized").glob("*.joblib"))
    if not model_files:
        model_files = list(Path("models/trained").glob("*.joblib"))
    
    if not model_files:
        raise FileNotFoundError("No trained models found.")
    
    model_path = model_files[0]
    model = joblib.load(model_path)
    model_name = model_path.stem
    
    # Load test data
    try:
        X_test = np.load('data/processed/X_engineered.npy')
        y_test = np.load('data/processed/y_encoded.npy')
    except FileNotFoundError:
        X_test = np.load('data/processed/X_processed.npy')
        y_test = np.load('data/processed/y_encoded.npy')
    
    # Generate feature names
    feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
    
    return model, model_name, X_test, y_test, feature_names

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Explainability Analysis')
    parser.add_argument('--sample-size', type=int, default=100)
    parser.add_argument('--dagshub-owner', type=str)
    parser.add_argument('--dagshub-repo', type=str)
    
    args = parser.parse_args()
    
    print("🚀 Starting Explainability Analysis...")
    
    try:
        # Load model and data
        model, model_name, X_test, y_test, feature_names = load_model_and_data()
        
        # Determine model type
        model_type = 'tree' if 'forest' in model_name.lower() else 'linear'
        
        # Create explainer
        explainer = DebtCollectionExplainer(model, model_type)
        
        # Use subset for analysis
        sample_size = min(args.sample_size, len(X_test))
        indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test[indices]
        
        # Create explainer
        explainer.create_explainer(X_sample[:50], feature_names)
        
        # Generate report
        report_path, summary_path = explainer.generate_explanation_report(
            X_sample, report_name=f"explanations_{model_name}"
        )
        
        # Get global importance
        global_importance = explainer.global_feature_importance(X_sample)
        
        print(f"\n🎯 TOP 10 FEATURES:")
        for i, feature in enumerate(global_importance['top_features'][:10], 1):
            print(f"{i:2d}. {feature['feature']:<25} ({feature['importance']:.4f})")
        
        print(f"\n📁 Reports generated:")
        print(f"   • {report_path}")
        print(f"   • {summary_path}")
        
        print("✅ Explainability analysis completed!")
        
    except Exception as e:
  

if __name__ == "__main__":
    main()