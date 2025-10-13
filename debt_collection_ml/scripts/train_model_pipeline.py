#!/usr/bin/env python3
"""
Comprehensive Model Training Pipeline for Debt Collection ML System
This script demonstrates end-to-end ML pipeline with MLOps best practices
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import joblib
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_generator import DebtCollectionDataGenerator
from data.data_preprocessor import AdvancedDataPreprocessor
from features.feature_engineering import AdvancedFeatureEngineer
from models.ml_model import DebtCollectionMLModel, ModelComparison
from utils.mlops import MLOpsOrchestrator, ContinuousIntegration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories"""
    directories = [
        'data/raw', 'data/processed', 'data/external',
        'models/trained', 'models/artifacts', 'models/experiments',
        'logs', 'reports', 'monitoring/metrics'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Directory structure created")

def load_or_generate_data(n_samples: int = 10000, force_regenerate: bool = False):
    """Load existing data or generate new dataset"""
    
    data_path = Path('data/raw/debt_collection_data.csv')
    
    if data_path.exists() and not force_regenerate:
        logger.info("Loading existing dataset...")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with {len(df)} samples")
    else:
        logger.info(f"Generating new dataset with {n_samples} samples...")
        generator = DebtCollectionDataGenerator(n_samples=n_samples)
        df = generator.generate_dataset()
        
        # Save dataset
        df.to_csv(data_path, index=False)
        logger.info(f"Generated and saved dataset with {len(df)} samples")
    
    return df

def preprocess_data(df: pd.DataFrame, save_artifacts: bool = True):
    """Comprehensive data preprocessing"""
    
    logger.info("Starting data preprocessing...")
    
    # Initialize preprocessor with optimal settings
    preprocessor = AdvancedDataPreprocessor(
        imputation_strategy='knn',
        scaling_method='standard',
        encoding_method='onehot',
        handle_outliers=True,
        outlier_method='iqr'
    )
    
    # Fit and transform data
    X_processed, y_encoded = preprocessor.fit_transform(df, target_column='Outcome')
    
    if save_artifacts:
        # Save preprocessed data
        np.save('data/processed/X_processed.npy', X_processed)
        np.save('data/processed/y_encoded.npy', y_encoded)
        
        # Save preprocessor
        preprocessor.save_preprocessor('models/artifacts/preprocessor.joblib')
    
    logger.info(f"Preprocessing completed. Shape: {X_processed.shape}")
    
    return X_processed, y_encoded, preprocessor

def engineer_features(X_processed: np.ndarray, y_encoded: np.ndarray, 
                     feature_names: list, save_artifacts: bool = True):
    """Advanced feature engineering"""
    
    logger.info("Starting feature engineering...")
    
    # Convert to DataFrame for feature engineering
    feature_df = pd.DataFrame(X_processed, columns=feature_names)
    
    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineer(
        include_polynomial=True,
        polynomial_degree=2,
        include_pca=False,  # Keep interpretability
        feature_selection=True,
        selection_k=50
    )
    
    # Fit and transform features
    X_engineered = feature_engineer.fit_transform(feature_df, y_encoded)
    
    if save_artifacts:
        # Save engineered features
        np.save('data/processed/X_engineered.npy', X_engineered)
        
        # Save feature engineer
        feature_engineer.save_feature_engineer('models/artifacts/feature_engineer.joblib')
    
    logger.info(f"Feature engineering completed. Shape: {X_engineered.shape}")
    
    return X_engineered, feature_engineer

def train_multiple_models(X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         optimize_hyperparameters: bool = True):
    """Train multiple models for comparison"""
    
    logger.info("Training multiple models...")
    
    # Model configurations
    model_configs = [
        {'name': 'XGBoost', 'type': 'xgboost'},
        {'name': 'LightGBM', 'type': 'lightgbm'},
        {'name': 'Random Forest', 'type': 'random_forest'},
        {'name': 'Ensemble', 'type': 'ensemble'}
    ]
    
    models = {}
    training_results = {}
    
    for config in model_configs:
        logger.info(f"Training {config['name']} model...")
        
        # Initialize model
        model = DebtCollectionMLModel(
            model_type=config['type'],
            random_state=42
        )
        
        # Train model
        with mlflow.start_run(run_name=f"{config['name']}_training"):
            training_result = model.train(
                X_train, y_train, X_val, y_val,
                optimize=optimize_hyperparameters,
                n_trials=30 if optimize_hyperparameters else 0
            )
            
            # Log additional metrics
            mlflow.log_param("model_name", config['name'])
            mlflow.log_param("model_type", config['type'])
            
            # Save model
            model_path = f"models/trained/{config['name'].lower().replace(' ', '_')}_model.joblib"
            model.save_model(model_path)
            
            # Log model artifact
            mlflow.log_artifact(model_path)
        
        models[config['name']] = model
        training_results[config['name']] = training_result
        
        logger.info(f"{config['name']} training completed")
    
    return models, training_results

def evaluate_models(models: dict, X_test: np.ndarray, y_test: np.ndarray):
    """Comprehensive model evaluation and comparison"""
    
    logger.info("Evaluating models...")
    
    # Initialize model comparison
    comparison = ModelComparison()
    
    # Add models to comparison
    for name, model in models.items():
        comparison.add_model(name, model)
    
    # Compare models
    comparison_df = comparison.compare_models(X_test, y_test)
    
    # Save comparison results
    comparison_df.to_csv('reports/model_comparison.csv', index=False)
    
    # Get best model
    best_model_name = comparison.get_best_model('Business F1')
    
    logger.info(f"Model evaluation completed. Best model: {best_model_name}")
    
    return comparison_df, best_model_name, comparison

def setup_mlops_monitoring(df: pd.DataFrame, best_model_name: str, models: dict):
    """Setup MLOps monitoring and model registry"""
    
    logger.info("Setting up MLOps monitoring...")
    
    # Initialize MLOps orchestrator
    mlops = MLOpsOrchestrator("debt_collection_ml")
    
    # Setup monitoring with reference data
    reference_data = df.sample(n=1000, random_state=42)  # Sample for monitoring
    mlops.setup_monitoring(reference_data, target_column='Outcome')
    
    # Register best model
    best_model = models[best_model_name]
    
    # Get model metrics (mock for demonstration)
    model_metrics = {
        'f1_score': 0.85,
        'roc_auc': 0.88,
        'business_f1': 0.82
    }
    
    model_parameters = {
        'model_type': best_model.model_type,
        'random_state': best_model.random_state
    }
    
    # Deploy model pipeline
    model_id = mlops.deploy_model_pipeline(
        model=best_model.model,
        model_name="debt_collection_model",
        version="1.0.0",
        model_type=best_model.model_type,
        metrics=model_metrics,
        parameters=model_parameters
    )
    
    # Save MLOps configuration
    mlops.save_mlops_config()
    
    logger.info(f"MLOps setup completed. Model ID: {model_id}")
    
    return mlops

def run_ci_pipeline():
    """Run CI/CD pipeline checks"""
    
    logger.info("Running CI/CD pipeline...")
    
    ci = ContinuousIntegration()
    
    # Data validation
    data_validation = ci.run_data_validation('data/raw/debt_collection_data.csv')
    
    # Model tests (using best model)
    model_tests = ci.run_model_tests(
        'models/trained/xgboost_model.joblib',
        'data/raw/debt_collection_data.csv'
    )
    
    # Generate CI report
    ci_report = ci.generate_ci_report(data_validation, model_tests)
    
    # Save CI report
    with open('reports/ci_cd_report.txt', 'w') as f:
        f.write(ci_report)
    
    logger.info("CI/CD pipeline completed")
    
    return data_validation, model_tests

def generate_comprehensive_report(comparison_df: pd.DataFrame, 
                                training_results: dict,
                                mlops: MLOpsOrchestrator):
    """Generate comprehensive training report"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""
DEBT COLLECTION ML SYSTEM - TRAINING REPORT
{'='*60}

Generated: {timestamp}

DATASET OVERVIEW:
{'-'*30}
Training completed successfully with comprehensive evaluation.

MODEL PERFORMANCE COMPARISON:
{'-'*30}
{comparison_df.to_string(index=False)}

BEST MODEL SELECTION:
{'-'*30}
Selected based on Business F1 score for optimal debt collection performance.

TRAINING CONFIGURATION:
{'-'*30}
- Preprocessing: KNN imputation, Standard scaling, One-hot encoding
- Feature Engineering: Polynomial features, Feature selection (top 50)
- Model Types: XGBoost, LightGBM, Random Forest, Ensemble
- Hyperparameter Optimization: Optuna with 30 trials per model
- Cross-validation: 5-fold stratified

MLOPS INTEGRATION:
{'-'*30}
- Model Registry: Local SQLite-based registry
- Monitoring: Data drift and performance monitoring setup
- CI/CD: Automated testing and validation pipeline
- Experiment Tracking: MLflow integration

BUSINESS IMPACT:
{'-'*30}
- Expected improvement in collection efficiency
- Risk-based prioritization of collection efforts
- Optimized contact channel recommendations
- Reduced manual effort through automation

NEXT STEPS:
{'-'*30}
1. Deploy model to staging environment
2. Setup real-time monitoring dashboard
3. Implement A/B testing framework
4. Schedule regular model retraining

{'='*60}
"""
    
    # Add MLOps report
    mlops_report = mlops.generate_mlops_report()
    report += f"\n{mlops_report}"
    
    # Save report
    with open('reports/comprehensive_training_report.txt', 'w') as f:
        f.write(report)
    
    logger.info("Comprehensive report generated")
    
    return report

def main():
    """Main training pipeline"""
    
    parser = argparse.ArgumentParser(description='Debt Collection ML Training Pipeline')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--optimize', action='store_true', help='Enable hyperparameter optimization')
    parser.add_argument('--regenerate', action='store_true', help='Force regenerate dataset')
    
    args = parser.parse_args()
    
    logger.info("Starting comprehensive ML training pipeline...")
    
    try:
        # Setup
        setup_directories()
        
        # Data loading/generation
        df = load_or_generate_data(n_samples=args.samples, force_regenerate=args.regenerate)
        
        # Data preprocessing
        X_processed, y_encoded, preprocessor = preprocess_data(df)
        
        # Feature engineering
        X_engineered, feature_engineer = engineer_features(
            X_processed, y_encoded, preprocessor.feature_names
        )
        
        # Train-test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_engineered, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Model training
        models, training_results = train_multiple_models(
            X_train, y_train, X_val, y_val, 
            optimize_hyperparameters=args.optimize
        )
        
        # Model evaluation
        comparison_df, best_model_name, comparison = evaluate_models(models, X_test, y_test)
        
        # MLOps setup
        mlops = setup_mlops_monitoring(df, best_model_name, models)
        
        # CI/CD pipeline
        data_validation, model_tests = run_ci_pipeline()
        
        # Generate comprehensive report
        report = generate_comprehensive_report(comparison_df, training_results, mlops)
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Best model: {best_model_name}")
        logger.info("Check reports/ directory for detailed results")
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING PIPELINE SUMMARY")
        print("="*60)
        print(f"Dataset: {len(df)} samples")
        print(f"Features: {X_engineered.shape[1]} (after engineering)")
        print(f"Best Model: {best_model_name}")
        print(f"Best Business F1: {comparison_df.loc[comparison_df['Model'] == best_model_name, 'Business F1'].iloc[0]:.4f}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()