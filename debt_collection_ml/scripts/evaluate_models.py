#!/usr/bin/env python3
import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.ml_model import DebtCollectionMLModel, ModelComparison

def main():
    """Evaluate trained models and generate metrics"""
    
    # Load test data
    X_engineered = np.load('data/processed/X_engineered.npy')
    y_encoded = np.load('data/processed/y_encoded.npy')
    
    # Split data (use same split as training)
    from sklearn.model_selection import train_test_split
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_engineered, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Load trained models
    models_dir = Path('models/trained')
    model_files = list(models_dir.glob('*.joblib'))
    
    if not model_files:
        print("No trained models found!")
        return
    
    # Initialize model comparison
    comparison = ModelComparison()
    
    # Load and evaluate each model
    for model_file in model_files:
        model_name = model_file.stem.replace('_model', '').replace('_', ' ').title()
        
        print(f"Evaluating {model_name}...")
        
        # Load model
        model = DebtCollectionMLModel()
        model.load_model(str(model_file))
        
        # Add to comparison
        comparison.add_model(model_name, model)
    
    # Compare models
    comparison_df = comparison.compare_models(X_test, y_test)
    
    # Save evaluation metrics
    os.makedirs('reports', exist_ok=True)
    
    # Save comparison results
    comparison_df.to_csv('reports/model_comparison.csv', index=False)
    
    # Save as JSON for DVC metrics
    comparison_dict = comparison_df.to_dict('records')
    with open('reports/model_comparison.json', 'w') as f:
        json.dump(comparison_dict, f, indent=2)
    
    # Generate evaluation metrics JSON
    evaluation_metrics = {}
    
    for model_name, model in comparison.models.items():
        if model.model is None:
            continue
            
        results = model.evaluate(X_test, y_test, model_name)
        
        evaluation_metrics[model_name] = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'roc_auc': float(results['roc_auc']),
            'business_metrics': {
                k: float(v) for k, v in results['business_metrics'].items()
            }
        }
    
    # Save evaluation metrics
    with open('reports/evaluation_metrics.json', 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    
    # Create evaluation plots directory
    plots_dir = Path('reports/evaluation_plots')
    plots_dir.mkdir(exist_ok=True)
    
    # Generate and save plots
    comparison.plot_comparison()
    
    print("Model evaluation completed!")
    print(f"Results saved to reports/")
    print(f"Best model: {comparison.get_best_model('Business F1')}")

if __name__ == "__main__":
    main()