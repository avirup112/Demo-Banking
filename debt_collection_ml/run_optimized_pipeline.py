#!/usr/bin/env python3
"""
Enhanced Debt Collection ML Pipeline with Optuna Optimization
Integrates hyperparameter optimization for maximum performance
"""

import sys
import os
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

# Import modules
from data.data_generator import DebtDataGenerator
from data.data_preprocessor import AdvancedDataPreprocessor
from features.feature_engineering import AdvancedFeatureEngineer
from optimization.optuna_optimizer import DebtCollectionOptimizer
from explainability.shap_explainer import DebtCollectionExplainer
from utils.dagshub_integration import DagsHubTracker

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/raw', 'data/processed', 'data/features',
        'models/trained', 'models/artifacts', 'models/optimized',
        'reports', 'optimization_results', 'experiments'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Directory structure created")

def generate_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic data"""
    
    logger.info(f"Generating {n_samples} samples...")
    
    generator = DebtDataGenerator()
    df = generator.generate_data(n_samples=n_samples)
    
    # Save raw data
    df.to_csv('data/raw/debt_collection_data.csv', index=False)
    
    logger.info(f"Generated dataset with {len(df)} samples")
    logger.info(f"Target distribution:\n{df['payment_status'].value_counts()}")
    
    return df

def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocess the data"""
    
    logger.info("Starting data preprocessing...")
    
    # Drop customer_id column if it exists
    if 'customer_id' in df.columns:
        df = df.drop('customer_id', axis=1)
    
    # Initialize preprocessor
    preprocessor = AdvancedDataPreprocessor(
        imputation_strategy='iterative',
        scaling_method='robust',
        encoding_method='target',
        handle_outliers=True
    )
    
    # Fit and transform
    X_processed, y_encoded = preprocessor.fit_transform(df, target_column='payment_status')
    
    # Ensure X_processed is numeric
    if hasattr(X_processed, 'dtype') and X_processed.dtype == 'object':
        # Convert to numeric, replacing non-numeric with NaN
        X_processed = pd.DataFrame(X_processed).apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    # Save processed data
    np.save('data/processed/X_processed.npy', X_processed)
    np.save('data/processed/y_encoded.npy', y_encoded)
    
    # Save preprocessor
    joblib.dump(preprocessor, 'models/artifacts/preprocessor.joblib')
    
    logger.info(f"Preprocessing completed. Shape: {X_processed.shape}")
    
    return X_processed, y_encoded, preprocessor

def engineer_features(X_processed: np.ndarray, y_encoded: np.ndarray) -> tuple:
    """Engineer advanced features with optimization focus"""
    
    logger.info("Starting optimized feature engineering...")
    
    # For optimization, we'll use a simpler approach
    # Add polynomial features manually
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_processed)
    
    logger.info(f"Created polynomial features. Shape: {X_poly.shape}")
    
    # Select best features
    selector = SelectKBest(score_func=f_classif, k=min(100, X_poly.shape[1]))
    X_engineered = selector.fit_transform(X_poly, y_encoded)
    
    # Save engineered data
    np.save('data/processed/X_engineered.npy', X_engineered)
    
    # Save feature engineering components
    joblib.dump({'poly': poly, 'selector': selector}, 'models/artifacts/feature_engineer.joblib')
    
    logger.info(f"Feature engineering completed. Shape: {X_engineered.shape}")
    
    return X_engineered, {'poly': poly, 'selector': selector}

def optimize_models(X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray,
                   n_trials: int = 50) -> dict:
    """Optimize models using Optuna"""
    
    logger.info("Starting hyperparameter optimization...")
    
    # Initialize optimizer
    optimizer = DebtCollectionOptimizer(
        study_name="debt_collection_optimization",
        n_trials=n_trials,
        cv_folds=5,
        random_state=42
    )
    
    # Set data
    optimizer.set_data(X_train, y_train, X_val, y_val)
    
    # Optimize all models
    models_to_optimize = ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression']
    optimization_results = optimizer.optimize_all_models(models_to_optimize)
    
    logger.info("Hyperparameter optimization completed")
    
    return optimization_results, optimizer

def train_optimized_models(X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          optimizer: DebtCollectionOptimizer) -> dict:
    """Train models with optimized parameters"""
    
    logger.info("Training optimized models...")
    
    results = {}
    models = {}
    
    for model_name in optimizer.best_params.keys():
        try:
            logger.info(f"Training optimized {model_name}...")
            
            # Create optimized model
            model = optimizer.create_optimized_model(model_name)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            # ROC AUC for multiclass
            if y_pred_proba is not None:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            else:
                roc_auc = 0.0
            
            # Store results
            results[f"{model_name}_optimized"] = {
                'model': model,
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'roc_auc': roc_auc,
                'best_params': optimizer.best_params[model_name],
                'optimization_score': optimizer.best_scores[model_name]
            }
            
            models[f"{model_name}_optimized"] = model
            
            # Save model
            model_path = f"models/optimized/{model_name}_optimized.joblib"
            joblib.dump(model, model_path)
            
            logger.info(f"{model_name}_optimized - Accuracy: {accuracy:.4f}, F1: {f1_weighted:.4f}, ROC-AUC: {roc_auc:.4f}")
            
            # Print classification report
            print(f"\n{'='*50}")
            print(f"{model_name}_optimized Classification Report:")
            print(f"{'='*50}")
            print(classification_report(y_test, y_pred, 
                                      target_names=['Not Paid', 'Paid', 'Partially Paid']))
            
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            results[f"{model_name}_optimized"] = {'error': str(e)}
    
    return results, models

def evaluate_and_compare(results: dict, optimization_results: dict) -> pd.DataFrame:
    """Evaluate and compare all models"""
    
    logger.info("Generating model comparison...")
    
    comparison_data = []
    
    for model_name, metrics in results.items():
        if 'error' not in metrics:
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'F1-Score': metrics['f1_weighted'],
                'F1-Macro': metrics['f1_macro'],
                'ROC-AUC': metrics['roc_auc'],
                'Optimization Score': metrics.get('optimization_score', 0),
                'Business F1': metrics['f1_weighted']  # Using weighted F1 as business metric
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Business F1', ascending=False)
    
    # Save comparison
    comparison_df.to_csv('reports/optimized_model_comparison.csv', index=False)
    
    return comparison_df

def generate_optimization_report(comparison_df: pd.DataFrame, 
                               optimization_results: dict,
                               df: pd.DataFrame) -> str:
    """Generate comprehensive optimization report"""
    
    best_model = comparison_df.iloc[0]
    
    report = f"""
DEBT COLLECTION ML OPTIMIZATION REPORT
=====================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW:
- Total samples: {len(df):,}
- Target distribution:
{df['payment_status'].value_counts().to_string()}

OPTIMIZATION SUMMARY:
- Models optimized: {len(optimization_results) - 1}  # Excluding 'best_overall'
- Best overall model: {optimization_results.get('best_overall', {}).get('model_name', 'Unknown')}

OPTIMIZED MODEL PERFORMANCE:
{'='*50}

"""
    
    for _, row in comparison_df.iterrows():
        report += f"""
{row['Model']}:
  - Accuracy: {row['Accuracy']:.4f}
  - F1-Score (Weighted): {row['F1-Score']:.4f}
  - F1-Score (Macro): {row['F1-Macro']:.4f}
  - Business F1: {row['Business F1']:.4f}
  - ROC-AUC: {row['ROC-AUC']:.4f}
  - Optimization Score: {row['Optimization Score']:.4f}
"""
    
    report += f"""

BEST MODEL: {best_model['Model']}
Best F1-Score: {best_model['Business F1']:.4f}

PERFORMANCE IMPROVEMENT:
- Target F1 Score: 0.65
- Achieved F1 Score: {best_model['Business F1']:.4f}
- Performance Status: {'TARGET EXCEEDED' if best_model['Business F1'] >= 0.65 else 'TARGET NOT MET'}

FILES GENERATED:
- data/raw/debt_collection_data.csv
- data/processed/X_processed.npy
- data/processed/X_engineered.npy
- data/processed/y_encoded.npy
- models/artifacts/preprocessor.joblib
- models/artifacts/feature_engineer.joblib
- models/optimized/*.joblib
- optimization_results/*
- reports/optimized_model_comparison.csv

NEXT STEPS:
1. Deploy the best model ({best_model['Model']})
2. Set up monitoring and drift detection
3. Implement A/B testing framework
4. Create web interface for predictions

=====================================================
"""
    
    return report

def run_explainability_analysis(models: dict, results: dict, 
                               X_test: np.ndarray, y_test: np.ndarray,
                               comparison_df: pd.DataFrame) -> dict:
    """Run SHAP explainability analysis on the best model"""
    
    try:
        logger.info("Starting SHAP explainability analysis...")
        
        # Get best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = models[best_model_name]
        
        # Determine model type
        model_type = 'tree' if any(tree_type in best_model_name.lower() 
                                  for tree_type in ['forest', 'tree', 'xgb', 'lgb']) else 'linear'
        
        # Create explainer
        explainer = DebtCollectionExplainer(best_model, model_type)
        
        # Use subset for analysis (SHAP can be slow)
        sample_size = min(100, len(X_test))
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test[sample_indices]
        
        # Generate feature names
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
        
        # Create explainer
        background_size = min(50, len(X_sample))
        explainer.create_explainer(X_sample[:background_size], feature_names)
        
        # Get global feature importance
        global_importance = explainer.global_feature_importance(X_sample)
        
        # Create summary plot
        summary_plot_path = explainer.create_summary_plot(X_sample[:50])
        
        # Generate explanation report
        customer_ids = [f"CUST_{i:06d}" for i in sample_indices]
        report_path, summary_path = explainer.generate_explanation_report(
            X_sample, customer_ids, f"explanations_{best_model_name}"
        )
        
        logger.info("SHAP explainability analysis completed")
        
        return {
            'best_model_name': best_model_name,
            'model_type': model_type,
            'samples_analyzed': sample_size,
            'features_analyzed': len(feature_names),
            'top_features': [f['feature'] for f in global_importance['top_features'][:10]],
            'top_feature_importance': global_importance['top_features'][0]['importance'],
            'summary_plot': summary_plot_path,
            'report_path': report_path,
            'summary_path': summary_path,
            'global_importance': global_importance
        }
        
    except Exception as e:
        logger.error(f"Explainability analysis failed: {e}")
        return None

def main():
    """Main pipeline execution"""
    
    parser = argparse.ArgumentParser(description='Optimized Debt Collection ML Pipeline')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials per model')
    parser.add_argument('--dagshub-owner', type=str, help='DagsHub repository owner')
    parser.add_argument('--dagshub-repo', type=str, help='DagsHub repository name')
    
    args = parser.parse_args()
    
    print("🚀 Starting Optimized Debt Collection ML Pipeline...")
    print("=" * 60)
    
    # Initialize DagsHub tracking if provided
    tracker = None
    if args.dagshub_owner and args.dagshub_repo:
        try:
            tracker = DagsHubTracker(args.dagshub_owner, args.dagshub_repo)
            with tracker.start_experiment("optimized_pipeline"):
                tracker.log_params({
                    'samples': args.samples,
                    'optimization_trials': args.trials,
                    'pipeline_version': 'optimized_v1.0',
                    'timestamp': datetime.now().isoformat()
                })
            print(f"📊 DagsHub tracking enabled: {args.dagshub_owner}/{args.dagshub_repo}")
        except Exception as e:
            logger.warning(f"DagsHub tracking failed: {e}")
    
    try:
        # Create directories
        create_directories()
        
        # Generate data
        df = generate_data(args.samples)
        
        # Preprocess data
        X_processed, y_encoded, preprocessor = preprocess_data(df)
        
        # Engineer features
        X_engineered, engineer = engineer_features(X_processed, y_encoded)
        
        # Split data for optimization
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Further split training data for validation
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Apply SMOTE for class balancing
        logger.info("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_opt, y_train_opt)
        
        logger.info(f"Balanced training set shape: {X_train_balanced.shape}")
        
        # Optimize models
        optimization_results, optimizer = optimize_models(
            X_train_balanced, y_train_balanced, 
            X_val_opt, y_val_opt,
            n_trials=args.trials
        )
        
        # Train optimized models
        results, models = train_optimized_models(
            X_train_balanced, y_train_balanced,
            X_test, y_test,
            optimizer
        )
        
        # Evaluate and compare
        comparison_df = evaluate_and_compare(results, optimization_results)
        
        # Generate report
        report = generate_optimization_report(comparison_df, optimization_results, df)
        
        # Save report
        with open('reports/optimization_report.txt', 'w') as f:
            f.write(report)
        
        # Create DVC metrics file
        best_model_metrics = results[comparison_df.iloc[0]['Model']]
        dvc_metrics = {
            'model_performance': {
                'accuracy': float(best_model_metrics['accuracy']),
                'f1_weighted': float(best_model_metrics['f1_weighted']),
                'f1_macro': float(best_model_metrics['f1_macro']),
                'roc_auc': float(best_model_metrics['roc_auc']),
                'optimization_score': float(best_model_metrics.get('optimization_score', 0))
            },
            'dataset_info': {
                'total_samples': len(df),
                'features_engineered': X_engineered.shape[1],
                'target_distribution': df['payment_status'].value_counts().to_dict()
            }
        }
        
        # Add explainability metrics if available
        if explainability_results:
            dvc_metrics['explainability'] = {
                'top_feature': explainability_results['top_features'][0],
                'top_feature_importance': float(explainability_results['top_feature_importance']),
                'features_analyzed': explainability_results['features_analyzed'],
                'samples_analyzed': explainability_results['samples_analyzed']
            }
        
        # Save DVC metrics
        import json
        with open('reports/pipeline_metrics.json', 'w') as f:
            json.dump(dvc_metrics, f, indent=2)
        
        # Print report
        print(report)
        
        # Run explainability analysis
        logger.info("Running SHAP explainability analysis...")
        explainability_results = run_explainability_analysis(
            models, results, X_test, y_test, comparison_df
        )
        
        # Log to DagsHub if available
        if tracker:
            best_model_name = comparison_df.iloc[0]['Model']
            best_metrics = results[best_model_name]
            
            with tracker.start_experiment("complete_pipeline_results"):
                # Log model performance metrics
                tracker.log_metrics({
                    'best_accuracy': best_metrics['accuracy'],
                    'best_f1_weighted': best_metrics['f1_weighted'],
                    'best_f1_macro': best_metrics['f1_macro'],
                    'best_roc_auc': best_metrics['roc_auc'],
                    'target_achieved': 1 if best_metrics['f1_weighted'] >= 0.65 else 0,
                    'optimization_score': best_metrics.get('optimization_score', 0)
                })
                
                # Log explainability metrics
                if explainability_results:
                    tracker.log_metrics({
                        'top_feature_importance': explainability_results['top_feature_importance'],
                        'features_analyzed': explainability_results['features_analyzed'],
                        'explainability_samples': explainability_results['samples_analyzed']
                    })
                    
                    # Log top 5 features as parameters
                    for i, feature in enumerate(explainability_results['top_features'][:5]):
                        tracker.log_params({f'top_feature_{i+1}': feature})
                
                # Log best model
                tracker.log_model(models[best_model_name], f"best_model_{best_model_name}")
                
                # Log artifacts
                tracker.log_artifacts("reports/optimization_report.txt", "reports")
                if explainability_results and explainability_results.get('summary_plot'):
                    tracker.log_artifacts(explainability_results['summary_plot'], "explainability")
        
        print("✅ Optimized pipeline completed successfully!")
        print(f"📊 Best model: {comparison_df.iloc[0]['Model']}")
        print(f"📈 Best F1 Score: {comparison_df.iloc[0]['Business F1']:.4f}")
        print("📁 Check reports/optimization_report.txt for detailed results")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")
        
        if tracker:
            with tracker.start_experiment("pipeline_error"):
                tracker.log_params({'error': str(e)})

if __name__ == "__main__":
    main()