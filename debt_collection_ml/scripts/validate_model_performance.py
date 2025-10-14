#!/usr/bin/env python3
"""
Model performance validation script for CI/CD
"""

import json
import sys
from pathlib import Path

def validate_performance():
    """Validate that model performance meets minimum thresholds"""
    
    metrics_file = Path('metrics/train_metrics.json')
    
    if not metrics_file.exists():
        print("❌ Metrics file not found!")
        return False
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Performance thresholds
    thresholds = {
        'accuracy': 0.60,      # Minimum 60% accuracy
        'f1_weighted': 0.55,   # Minimum 55% weighted F1
        'f1_macro': 0.45,      # Minimum 45% macro F1
        'roc_auc': 0.65        # Minimum 65% ROC-AUC
    }
    
    best_model_name = metrics['best_model']['name']
    best_model_metrics = None
    
    # Find best model metrics
    for model_name, model_metrics in metrics.items():
        if model_name == best_model_name:
            best_model_metrics = model_metrics
            break
    
    if not best_model_metrics:
        print(f"❌ Best model '{best_model_name}' metrics not found!")
        return False
    
    print(f"🔍 Validating performance for best model: {best_model_name}")
    print("="*50)
    
    all_passed = True
    
    for metric, threshold in thresholds.items():
        if metric in best_model_metrics:
            value = best_model_metrics[metric]
            passed = value >= threshold
            status = "✅ PASS" if passed else "❌ FAIL"
            
            print(f"{metric:15}: {value:.4f} (threshold: {threshold:.4f}) {status}")
            
            if not passed:
                all_passed = False
        else:
            print(f"{metric:15}: NOT FOUND ❌")
            all_passed = False
    
    print("="*50)
    
    if all_passed:
        print("🎉 All performance thresholds met!")
        return True
    else:
        print("❌ Performance validation failed!")
        print("💡 Consider:")
        print("  - Adjusting hyperparameters")
        print("  - Adding more training data")
        print("  - Improving feature engineering")
        return False

if __name__ == "__main__":
    success = validate_performance()
    sys.exit(0 if success else 1)