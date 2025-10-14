#!/usr/bin/env python3
"""
Model Training Script for DVC Pipeline
"""

import sys
import os
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.ml_model import DebtCollectionMLModel
from utils.dagshub_integration import DagsHubTracker

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def main():
    """Train models based on DVC parameters"""
    
    # Load parameters
    params = load_params()
    training_params = params['training']
    dagshub_params = params.get('dagshub', {})
    
    print("Starting model training...")
    
    # Create output directories
    Path('models/trained').mkdir(parents=True, exist_ok=True)
    Path('reports/plots').mkdir(parents=True, exist_ok=True)
    Path('reports').mkdir(parents=True, exist_ok=True)
    
    # Load engineered data
    X_engineered = np.load('data/processed/X_engineered.npy')
    y_encoded = np.load('data/processed/y_encoded.npy')
    
    print(f"Loaded engineered data: {X_engineered.shape}")
    
    # Train-test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_engineered, y_encoded, 
        test_size=training_params['test_size'] + training_params['validation_size'],
        random_state=training_params['random_state'],
        stratify=y_encoded
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=training_params['test_size'] / (training_params['test_size'] + training_params['validation_size']),
        random_state=training_params['random_state'],
        stratify=y_temp
    )
    
    print(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Initialize DagsHub tracker if configured
    dagshub_tracker = None
    if dagshub_params.get('repo_owner'):
        try:
            dagshub_tracker = DagsHubTracker(
                dagshub_params['repo_owner'], 
                dagshub_params['repo_name']
            )
            print(f"DagsHub tracking enabled: {dagshub_params['repo_owner']}/{dagshub_params['repo_name']}")
        except Exception as e:
            print(f"Failed to initialize DagsHub tracker: {e}")
    
    # Train models
    models = {}
    training_results = {}
    model_metrics = {}
    
    for model_type in training_params['model_types']:
        print(f"Training {model_type} model...")
        
        # Initialize model
        model = DebtCollectionMLModel(
            model_type=model_type,
            random_state=training_params['random_state'],
            dagshub_tracker=dagshub_tracker
        )
        
        # Train model
        training_result = model.train(
            X_train, y_train, X_val, y_val,
            optimize=training_params['optimize_hyperparameters'],
            n_trials=training_params['n_trials']
        )
        
        # Evaluate on test set
        test_results = model.evaluate(X_test, y_test, f"{model_type}_test")
        
        # Save model
        model_path = f"models/trained/{model_type}_model.joblib"
        model.save_model(model_path)
        
        # Store results
        models[model_type] = model
        training_results[model_type] = training_result
        model_metrics[model_type] = {
            'accuracy': test_results['accuracy'],
            'f1_score': test_results['f1_score'],
            'roc_auc': test_results['roc_auc'],
            'business_f1': test_results['business_metrics']['business_f1'],
            'recovery_precision': test_results['business_metrics']['recovery_precision'],
            'collection_recall': test_results['business_metrics']['collection_recall']
        }
        
        print(f"{model_type} training completed - F1: {test_results['f1_score']:.4f}")
    
    # Generate plots
    generate_plots(models, X_test, y_test, model_metrics)
    
    # Save metrics
    metrics = {
        'training_config': training_params,
        'model_metrics': model_metrics,
        'best_model': max(model_metrics.keys(), key=lambda k: model_metrics[k]['business_f1']),
        'data_splits': {
            'train_size': X_train.shape[0],
            'validation_size': X_val.shape[0],
            'test_size': X_test.shape[0]
        }
    }
    
    with open('reports/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model training completed successfully")
    print(f"Best model: {metrics['best_model']}")
    print(f"Best Business F1: {model_metrics[metrics['best_model']]['business_f1']:.4f}")

def generate_plots(models, X_test, y_test, model_metrics):
    """Generate visualization plots"""
    
    # Model comparison plot
    plt.figure(figsize=(12, 8))
    
    metrics_df = pd.DataFrame(model_metrics).T
    
    # Plot 1: Model comparison
    plt.subplot(2, 2, 1)
    metrics_df[['accuracy', 'f1_score', 'roc_auc']].plot(kind='bar', ax=plt.gca())
    plt.title('Technical Metrics Comparison')
    plt.ylabel('Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    
    # Plot 2: Business metrics
    plt.subplot(2, 2, 2)
    metrics_df[['business_f1', 'recovery_precision', 'collection_recall']].plot(kind='bar', ax=plt.gca())
    plt.title('Business Metrics Comparison')
    plt.ylabel('Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    
    # Plot 3: Feature importance (for tree-based models)
    plt.subplot(2, 2, 3)
    for model_name, model in models.items():
        if hasattr(model, 'feature_importance') and model.feature_importance is not None:
            top_features = model.feature_importance[:10]  # Top 10 features
            plt.barh(range(len(top_features)), top_features, alpha=0.7, label=model_name)
            break  # Just show one model's feature importance
    
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature Index')
    
    # Plot 4: Confusion matrix for best model
    plt.subplot(2, 2, 4)
    best_model_name = max(model_metrics.keys(), key=lambda k: model_metrics[k]['business_f1'])
    best_model = models[best_model_name]
    
    y_pred = best_model.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('reports/plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual confusion matrices
    for model_name, model in models.items():
        plt.figure(figsize=(8, 6))
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.savefig(f'reports/plots/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Feature importance plot
    plt.figure(figsize=(10, 8))
    for model_name, model in models.items():
        if hasattr(model, 'feature_importance') and model.feature_importance is not None:
            top_indices = np.argsort(model.feature_importance)[-20:]  # Top 20
            plt.barh(range(len(top_indices)), model.feature_importance[top_indices], alpha=0.7)
            plt.title(f'Top 20 Feature Importances - {model_name}')
            plt.xlabel('Importance')
            plt.ylabel('Feature Index')
            plt.savefig(f'reports/plots/feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            break  # Just create one feature importance plot

if __name__ == "__main__":
    main()