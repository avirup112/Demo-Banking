#!/usr/bin/env python3
"""
Complete DVC + DagsHub + CI/CD Pipeline for Debt Collection ML System
This script integrates DVC, DagsHub, and CI/CD for MLOps
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import warnings
import json
import yaml
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

# DagsHub and DVC integration
try:
    import dagshub
    from utils.dagshub_integration import DagsHubTracker
    DAGSHUB_AVAILABLE = True
except ImportError:
    DAGSHUB_AVAILABLE = False
    print("DagsHub not available. Install with: pip install dagshub")

# Import modules
from data.data_generator import DebtCollectionDataGenerator
from data.data_preprocessor import AdvancedDataPreprocessor
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import mlflow
import mlflow.sklearn

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    import mlflow.xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    import mlflow.lightgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dvc_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DVCDagsHubPipeline:
    """Complete MLOps pipeline with DVC and DagsHub integration"""
    
    def __init__(self, dagshub_owner=None, dagshub_repo="debt-collection-ml"):
        self.dagshub_owner = dagshub_owner
        self.dagshub_repo = dagshub_repo
        self.dagshub_tracker = None
        
        # Initialize DagsHub if credentials provided
        if dagshub_owner and DAGSHUB_AVAILABLE:
            try:
                self.dagshub_tracker = DagsHubTracker(dagshub_owner, dagshub_repo)
                logger.info(f"DagsHub tracking enabled: {dagshub_owner}/{dagshub_repo}")
            except Exception as e:
                logger.warning(f"Failed to initialize DagsHub: {e}")
                self.dagshub_tracker = None
        
        # Setup MLflow
        if not self.dagshub_tracker:
            mlflow.set_experiment("debt_collection_ml_dvc")
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            'data/raw', 'data/processed', 'models/trained', 
            'models/artifacts', 'reports', 'logs', 'metrics'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Directory structure created")
    
    def load_params(self):
        """Load parameters from params.yaml"""
        try:
            with open('params.yaml', 'r') as f:
                params = yaml.safe_load(f)
            return params
        except FileNotFoundError:
            logger.warning("params.yaml not found, using default parameters")
            return {
                'generate_data': {'n_samples': 10000, 'random_state': 42},
                'preprocess': {'imputation_strategy': 'knn', 'scaling_method': 'standard'},
                'feature_engineering': {'selection_k': 100},
                'training': {'optimize_hyperparameters': True, 'cv_folds': 5}
            }
    
    def generate_data(self, params):
        """Generate synthetic dataset"""
        
        gen_params = params.get('generate_data', {})
        n_samples = gen_params.get('n_samples', 10000)
        random_state = gen_params.get('random_state', 42)
        
        logger.info(f"Generating {n_samples} samples...")
        
        generator = DebtCollectionDataGenerator(n_samples=n_samples, random_state=random_state)
        df = generator.generate_dataset()
        
        # Save raw data
        df.to_csv('data/raw/debt_collection_data.csv', index=False)
        
        # Log dataset info to DagsHub
        if self.dagshub_tracker:
            with self.dagshub_tracker.start_run("data_generation"):
                self.dagshub_tracker.log_params(gen_params)
                self.dagshub_tracker.log_dataset_info(df, "raw_data")
        
        logger.info(f"Generated dataset with {len(df)} samples")
        return df
    
    def preprocess_data(self, df, params):
        """Preprocess the data"""
        
        preprocess_params = params.get('preprocess', {})
        
        logger.info("Starting data preprocessing...")
        
        # Initialize preprocessor
        preprocessor = AdvancedDataPreprocessor(
            imputation_strategy=preprocess_params.get('imputation_strategy', 'knn'),
            scaling_method=preprocess_params.get('scaling_method', 'standard'),
            encoding_method=preprocess_params.get('encoding_method', 'onehot'),
            handle_outliers=preprocess_params.get('handle_outliers', True),
            outlier_method=preprocess_params.get('outlier_method', 'iqr')
        )
        
        # Fit and transform
        X_processed, y_encoded = preprocessor.fit_transform(df, target_column='Outcome')
        
        # Save processed data
        np.save('data/processed/X_processed.npy', X_processed)
        np.save('data/processed/y_encoded.npy', y_encoded)
        
        # Save preprocessor
        preprocessor.save_preprocessor('models/artifacts/preprocessor.joblib')
        
        # Log preprocessing info to DagsHub
        if self.dagshub_tracker:
            with self.dagshub_tracker.start_run("preprocessing"):
                self.dagshub_tracker.log_params(preprocess_params)
                self.dagshub_tracker.log_metrics({
                    'processed_features': X_processed.shape[1],
                    'processed_samples': X_processed.shape[0]
                })
        
        logger.info(f"Preprocessing completed. Shape: {X_processed.shape}")
        return X_processed, y_encoded, preprocessor
    
    def engineer_features(self, X_processed, y_encoded, preprocessor, params):
        """Engineer and select features"""
        
        fe_params = params.get('feature_engineering', {})
        selection_k = fe_params.get('selection_k', 100)
        
        logger.info("Starting feature engineering...")
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(X_processed, columns=preprocessor.feature_names)
        
        # Simple feature engineering (create essential features only)
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns[:10]
        
        if len(numeric_cols) >= 4:
            try:
                col1, col2, col3, col4 = numeric_cols[0], numeric_cols[1], numeric_cols[2], numeric_cols[3]
                
                # Essential financial ratios
                feature_df['Debt_Ratio'] = feature_df[col4] / (feature_df[col2] + 1)
                feature_df['Utilization'] = feature_df[col4] / (feature_df[col3] + 1)
                
                if len(numeric_cols) >= 5:
                    col5 = numeric_cols[4]
                    feature_df['High_Risk'] = (feature_df[col5] > 90).astype(int)
                
                feature_df['Young'] = (feature_df[col1] < 30).astype(int)
                feature_df['Senior'] = (feature_df[col1] > 55).astype(int)
                
                logger.info("Created essential engineered features")
            except Exception as e:
                logger.warning(f"Feature engineering failed: {e}")
        
        # Feature selection
        n_features = min(selection_k, feature_df.shape[1])
        logger.info(f"Selecting top {n_features} features from {feature_df.shape[1]}")
        
        selector = SelectKBest(f_classif, k=n_features)
        X_selected = selector.fit_transform(feature_df, y_encoded)
        
        # Save engineered features
        np.save('data/processed/X_engineered.npy', X_selected)
        joblib.dump(selector, 'models/artifacts/feature_selector.joblib')
        
        # Log feature engineering info to DagsHub
        if self.dagshub_tracker:
            with self.dagshub_tracker.start_run("feature_engineering"):
                self.dagshub_tracker.log_params(fe_params)
                self.dagshub_tracker.log_metrics({
                    'original_features': feature_df.shape[1],
                    'selected_features': X_selected.shape[1],
                    'feature_reduction_ratio': X_selected.shape[1] / feature_df.shape[1]
                })
        
        logger.info(f"Feature engineering completed. Shape: {X_selected.shape}")
        return X_selected
    
    def train_models(self, X_engineered, y_encoded, params):
        """Train multiple models with DagsHub tracking"""
        
        training_params = params.get('training', {})
        
        logger.info("Starting model training with DagsHub tracking...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Check class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"Original class distribution: {class_dist}")
        
        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        unique_bal, counts_bal = np.unique(y_train_balanced, return_counts=True)
        class_dist_bal = dict(zip(unique_bal, counts_bal))
        logger.info(f"Balanced class distribution: {class_dist_bal}")
        
        # Define optimized models
        models = {
            'RandomForest_Optimized': RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'ExtraTrees_Optimized': ExtraTreesClassifier(
                n_estimators=300,
                max_depth=25,
                min_samples_split=8,
                min_samples_leaf=3,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'LogisticRegression_Optimized': LogisticRegression(
                random_state=42,
                max_iter=3000,
                class_weight='balanced',
                C=0.1,
                solver='liblinear'
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost_Optimized'] = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['LightGBM_Optimized'] = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Start DagsHub run for this model
            run_context = self.dagshub_tracker.start_run(f"{name}_training") if self.dagshub_tracker else mlflow.start_run(run_name=f"{name}_training")
            
            with run_context:
                try:
                    # Log model parameters
                    model_params = model.get_params()
                    if self.dagshub_tracker:
                        self.dagshub_tracker.log_params(model_params)
                    else:
                        for key, value in model_params.items():
                            mlflow.log_param(key, value)
                    
                    # Train model
                    model.fit(X_train_balanced, y_train_balanced)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    f1_weighted = f1_score(y_test, y_pred, average='weighted')
                    f1_macro = f1_score(y_test, y_pred, average='macro')
                    
                    try:
                        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                    except:
                        roc_auc = 0.0
                    
                    # Calculate class-wise metrics
                    from sklearn.metrics import precision_recall_fscore_support
                    precision, recall, f1_scores, support = precision_recall_fscore_support(y_test, y_pred)
                    
                    # Store results
                    results[name] = {
                        'accuracy': accuracy,
                        'f1_weighted': f1_weighted,
                        'f1_macro': f1_macro,
                        'roc_auc': roc_auc,
                        'precision_per_class': precision.tolist(),
                        'recall_per_class': recall.tolist(),
                        'f1_per_class': f1_scores.tolist(),
                        'model': model
                    }
                    
                    # Log metrics to DagsHub
                    metrics = {
                        'accuracy': accuracy,
                        'f1_weighted': f1_weighted,
                        'f1_macro': f1_macro,
                        'roc_auc': roc_auc,
                        'precision_not_paid': precision[0],
                        'recall_not_paid': recall[0],
                        'f1_not_paid': f1_scores[0],
                        'precision_paid': precision[1],
                        'recall_paid': recall[1],
                        'f1_paid': f1_scores[1],
                        'precision_partially_paid': precision[2],
                        'recall_partially_paid': recall[2],
                        'f1_partially_paid': f1_scores[2]
                    }
                    
                    if self.dagshub_tracker:
                        self.dagshub_tracker.log_metrics(metrics)
                        # Log model
                        if 'XGBoost' in name:
                            self.dagshub_tracker.log_model(model, f"{name}_model", "xgboost")
                        elif 'LightGBM' in name:
                            self.dagshub_tracker.log_model(model, f"{name}_model", "lightgbm")
                        else:
                            self.dagshub_tracker.log_model(model, f"{name}_model", "sklearn")
                    else:
                        for key, value in metrics.items():
                            mlflow.log_metric(key, value)
                        
                        # Log model to MLflow
                        if 'XGBoost' in name:
                            mlflow.xgboost.log_model(model, f"{name}_model")
                        elif 'LightGBM' in name:
                            mlflow.lightgbm.log_model(model, f"{name}_model")
                        else:
                            mlflow.sklearn.log_model(model, f"{name}_model")
                    
                    # Save model locally
                    model_path = f'models/trained/{name.lower()}_model.joblib'
                    joblib.dump(model, model_path)
                    
                    logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1-Weighted: {f1_weighted:.4f}, F1-Macro: {f1_macro:.4f}")
                    
                    # Print classification report
                    print(f"\n{'='*50}")
                    print(f"{name} Classification Report:")
                    print('='*50)
                    print(classification_report(y_test, y_pred, target_names=['Not Paid', 'Paid', 'Partially Paid']))
                    
                except Exception as e:
                    logger.error(f"Failed to train {name}: {e}")
                    continue
        
        return results, X_test, y_test
    
    def save_metrics_for_dvc(self, results):
        """Save metrics in DVC format"""
        
        # Create metrics for DVC
        dvc_metrics = {}
        
        for model_name, metrics in results.items():
            dvc_metrics[model_name] = {
                'accuracy': float(metrics['accuracy']),
                'f1_weighted': float(metrics['f1_weighted']),
                'f1_macro': float(metrics['f1_macro']),
                'roc_auc': float(metrics['roc_auc'])
            }
        
        # Find best model
        best_model = max(results.keys(), key=lambda k: results[k]['f1_weighted'])
        dvc_metrics['best_model'] = {
            'name': best_model,
            'f1_weighted': float(results[best_model]['f1_weighted'])
        }
        
        # Save metrics for DVC
        with open('metrics/train_metrics.json', 'w') as f:
            json.dump(dvc_metrics, f, indent=2)
        
        # Save detailed results
        detailed_results = {}
        for model_name, metrics in results.items():
            detailed_results[model_name] = {
                'accuracy': float(metrics['accuracy']),
                'f1_weighted': float(metrics['f1_weighted']),
                'f1_macro': float(metrics['f1_macro']),
                'roc_auc': float(metrics['roc_auc']),
                'precision_per_class': metrics['precision_per_class'],
                'recall_per_class': metrics['recall_per_class'],
                'f1_per_class': metrics['f1_per_class']
            }
        
        with open('reports/detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        logger.info("Metrics saved for DVC tracking")
        
        return dvc_metrics
    
    def run_complete_pipeline(self):
        """Run the complete DVC + DagsHub pipeline"""
        
        print("🚀 Starting Complete DVC + DagsHub ML Pipeline...")
        print("="*60)
        
        try:
            # Step 1: Setup
            self.create_directories()
            
            # Step 2: Load parameters
            params = self.load_params()
            
            # Step 3: Generate data
            df = self.generate_data(params)
            
            # Step 4: Preprocess data
            X_processed, y_encoded, preprocessor = self.preprocess_data(df, params)
            
            # Step 5: Engineer features
            X_engineered = self.engineer_features(X_processed, y_encoded, preprocessor, params)
            
            # Step 6: Train models
            results, X_test, y_test = self.train_models(X_engineered, y_encoded, params)
            
            # Step 7: Save metrics for DVC
            dvc_metrics = self.save_metrics_for_dvc(results)
            
            # Step 8: Generate final report
            best_model = max(results.keys(), key=lambda k: results[k]['f1_weighted'])
            
            print(f"\n✅ Pipeline completed successfully!")
            print(f"📊 Best model: {best_model}")
            print(f"📈 Best F1-Weighted: {results[best_model]['f1_weighted']:.4f}")
            
            if self.dagshub_tracker:
                print(f"🔗 View experiments: https://dagshub.com/{self.dagshub_owner}/{self.dagshub_repo}.mlflow")
            
            print("📁 Check metrics/train_metrics.json for DVC metrics")
            print("📁 Check reports/detailed_results.json for detailed results")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            print(f"❌ Pipeline failed: {e}")
            return False

def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='DVC + DagsHub ML Pipeline')
    parser.add_argument('--dagshub-owner', type=str, help='DagsHub repository owner')
    parser.add_argument('--dagshub-repo', type=str, default='debt-collection-ml', help='DagsHub repository name')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DVCDagsHubPipeline(
        dagshub_owner=args.dagshub_owner,
        dagshub_repo=args.dagshub_repo
    )
    
    # Run pipeline
    success = pipeline.run_complete_pipeline()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)