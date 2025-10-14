#!/usr/bin/env python3
"""
Complete ML Pipeline for Debt Collection System
This script runs the entire pipeline from data generation to model training
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

# Import modules
from data.data_generator import DebtCollectionDataGenerator
from data.data_preprocessor import AdvancedDataPreprocessor
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import joblib

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleFeatureEngineer:
    """Simple feature engineering that works with preprocessed data"""
    
    def __init__(self):
        self.feature_names = []
        
    def fit_transform(self, X_df, y=None):
        """Create simple engineered features"""
        
        X_new = X_df.copy()
        
        # Find numeric columns (first few are usually the original numeric features)
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns[:10]  # Take first 10 numeric
        
        if len(numeric_cols) >= 4:
            # Assume order: Age, Income, Loan_Amount, Outstanding_Balance, Days_Past_Due, etc.
            try:
                # Simple ratios using positional indexing
                col1, col2, col3, col4 = numeric_cols[0], numeric_cols[1], numeric_cols[2], numeric_cols[3]
                
                # Financial ratios
                X_new['Debt_Ratio'] = X_df[col4] / (X_df[col2] + 1)  # Outstanding/Income
                X_new['Utilization'] = X_df[col4] / (X_df[col3] + 1)  # Outstanding/Loan
                
                # Risk indicators
                if len(numeric_cols) >= 5:
                    col5 = numeric_cols[4]  # Days_Past_Due
                    X_new['High_Risk'] = (X_df[col5] > 90).astype(int)
                
                # Age categories
                X_new['Young'] = (X_df[col1] < 30).astype(int)
                X_new['Senior'] = (X_df[col1] > 55).astype(int)
                
                # Income categories
                income_median = X_df[col2].median()
                X_new['High_Income'] = (X_df[col2] > income_median).astype(int)
                
                logger.info("Created engineered features successfully")
                
            except Exception as e:
                logger.warning(f"Feature engineering failed: {e}")
                # Continue with original features
        
        return X_new
    
    def transform(self, X_df):
        """Transform new data"""
        return self.fit_transform(X_df)
    
    def save_feature_engineer(self, filepath):
        """Save feature engineer"""
        joblib.dump(self, filepath)

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/raw', 'data/processed', 'models/trained', 
        'models/artifacts', 'reports', 'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Directory structure created")

def generate_data(n_samples=10000):
    """Generate synthetic dataset"""
    
    logger.info(f"Generating {n_samples} samples...")
    
    generator = DebtCollectionDataGenerator(n_samples=n_samples)
    df = generator.generate_dataset()
    
    # Save raw data
    df.to_csv('data/raw/debt_collection_data.csv', index=False)
    
    logger.info(f"Generated dataset with {len(df)} samples")
    logger.info(f"Outcome distribution:\n{df['Outcome'].value_counts()}")
    
    return df

def preprocess_data(df):
    """Preprocess the data"""
    
    logger.info("Starting data preprocessing...")
    
    # Initialize preprocessor
    preprocessor = AdvancedDataPreprocessor(
        imputation_strategy='knn',
        scaling_method='standard',
        encoding_method='onehot',
        handle_outliers=True,
        outlier_method='iqr'
    )
    
    # Fit and transform
    X_processed, y_encoded = preprocessor.fit_transform(df, target_column='Outcome')
    
    # Save processed data
    np.save('data/processed/X_processed.npy', X_processed)
    np.save('data/processed/y_encoded.npy', y_encoded)
    
    # Save preprocessor
    preprocessor.save_preprocessor('models/artifacts/preprocessor.joblib')
    
    logger.info(f"Preprocessing completed. Shape: {X_processed.shape}")
    
    return X_processed, y_encoded, preprocessor

def engineer_features(X_processed, y_encoded, preprocessor):
    """Engineer features from preprocessed data with feature selection"""
    
    logger.info("Starting optimized feature engineering...")
    
    # Convert to DataFrame using feature names from preprocessor
    feature_df = pd.DataFrame(X_processed, columns=preprocessor.feature_names)
    
    # Use simple feature engineer
    feature_engineer = SimpleFeatureEngineer()
    X_engineered_df = feature_engineer.fit_transform(feature_df, y_encoded)
    
    # Feature selection to reduce dimensionality
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Select top 100 features instead of all
    n_features = min(100, X_engineered_df.shape[1])
    logger.info(f"Selecting top {n_features} features from {X_engineered_df.shape[1]}")
    
    selector = SelectKBest(f_classif, k=n_features)
    X_selected = selector.fit_transform(X_engineered_df, y_encoded)
    
    # Convert back to numpy array
    X_engineered = X_selected
    
    # Save engineered features
    np.save('data/processed/X_engineered.npy', X_engineered)
    
    # Save feature engineer and selector
    feature_engineer.save_feature_engineer('models/artifacts/feature_engineer.joblib')
    joblib.dump(selector, 'models/artifacts/feature_selector.joblib')
    
    logger.info(f"Feature engineering completed. Shape: {X_engineered.shape}")
    
    return X_engineered, feature_engineer

def train_models(X_engineered, y_encoded):
    """Train multiple models with class imbalance handling"""
    
    logger.info("Starting model training with class balancing...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Check original class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))
    logger.info(f"Original class distribution: {class_dist}")
    print(f"Original class distribution: {class_dist}")
    print(f"Class percentages: {counts / len(y_train) * 100}")
    
    # Apply SMOTE for class balancing
    logger.info("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Check balanced class distribution
    unique_bal, counts_bal = np.unique(y_train_balanced, return_counts=True)
    class_dist_bal = dict(zip(unique_bal, counts_bal))
    logger.info(f"Balanced class distribution: {class_dist_bal}")
    print(f"Balanced class distribution: {class_dist_bal}")
    
    logger.info(f"Train shape: {X_train_balanced.shape}, Test shape: {X_test.shape}")
    
    # Define optimized models with better hyperparameters
    models = {}
    
    # Random Forest with optimized parameters
    models['RandomForest_Optimized'] = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Extra Trees with optimized parameters
    models['ExtraTrees_Optimized'] = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=8,
        min_samples_leaf=3,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Logistic Regression with optimized parameters
    models['LogisticRegression_Optimized'] = LogisticRegression(
        random_state=42,
        max_iter=3000,
        class_weight='balanced',
        C=0.1,
        solver='liblinear'
    )
    
    # Add XGBoost if available with optimized parameters
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
    
    # Add LightGBM if available with optimized parameters
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
        
        try:
            # Train model on balanced data
            model.fit(X_train_balanced, y_train_balanced)
            
            # Make predictions on original test set
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                roc_auc = 0.0
            
            # Calculate business-specific metrics
            business_f1 = f1_score(y_test, y_pred, average='weighted')  # Weighted F1 for business
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'f1_macro': f1_macro,
                'roc_auc': roc_auc,
                'business_f1': business_f1,
                'model': model
            }
            
            # Save model
            model_path = f'models/trained/{name.lower()}_model.joblib'
            joblib.dump(model, model_path)
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, F1-Macro: {f1_macro:.4f}, ROC-AUC: {roc_auc:.4f}")
            
            # Print detailed classification report
            print(f"\n{'='*50}")
            print(f"{name} Classification Report:")
            print('='*50)
            print(classification_report(y_test, y_pred, target_names=['Not Paid', 'Paid', 'Partially Paid']))
            
            # Print class-wise performance
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1_scores, support = precision_recall_fscore_support(y_test, y_pred)
            
            print(f"\nClass-wise Performance:")
            class_names = ['Not Paid', 'Paid', 'Partially Paid']
            for i, class_name in enumerate(class_names):
                print(f"{class_name:15} - Precision: {precision[i]:.3f}, Recall: {recall[i]:.3f}, F1: {f1_scores[i]:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")
            continue
    
    return results, X_test, y_test

def generate_report(results, df):
    """Generate comprehensive report"""
    
    logger.info("Generating final report...")
    
    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['f1_score'])
    
    # Build report string
    report_lines = [
        "DEBT COLLECTION ML PIPELINE REPORT",
        "=" * 50,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "DATASET OVERVIEW:",
        f"- Total samples: {len(df):,}",
        f"- Features after preprocessing: {np.load('data/processed/X_processed.npy').shape[1]}",
        f"- Features after engineering: {np.load('data/processed/X_engineered.npy').shape[1]}",
        "",
        "TARGET DISTRIBUTION:",
        df['Outcome'].value_counts().to_string(),
        "",
        "MODEL PERFORMANCE:",
        "-" * 30
    ]
    
    # Add model results
    for name, metrics in results.items():
        report_lines.extend([
            "",
            f"{name}:",
            f"  - Accuracy: {metrics['accuracy']:.4f}",
            f"  - F1-Score (Weighted): {metrics['f1_score']:.4f}",
            f"  - F1-Score (Macro): {metrics['f1_macro']:.4f}",
            f"  - Business F1: {metrics['business_f1']:.4f}",
            f"  - ROC-AUC: {metrics['roc_auc']:.4f}"
        ])
    
    # Add summary and next steps
    report_lines.extend([
        "",
        f"BEST MODEL: {best_model}",
        f"Best F1-Score: {results[best_model]['f1_score']:.4f}",
        "",
        "FILES GENERATED:",
        "- data/raw/debt_collection_data.csv",
        "- data/processed/X_processed.npy",
        "- data/processed/X_engineered.npy",
        "- data/processed/y_encoded.npy",
        "- models/artifacts/preprocessor.joblib",
        "- models/artifacts/feature_engineer.joblib",
        "- models/trained/*.joblib",
        "",
        "NEXT STEPS:",
        "1. Use the best model for predictions",
        "2. Deploy the model using the saved artifacts",
        "3. Monitor model performance over time",
        "4. Retrain with new data as needed",
        "",
        "=" * 50
    ])
    
    # Join all lines into final report
    report = "\n".join(report_lines)
    
    # Save report
    with open('reports/pipeline_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    
    return report

def main():
    """Main pipeline execution"""
    
    print("🚀 Starting Complete Debt Collection ML Pipeline...")
    print("="*60)
    
    try:
        # Step 1: Setup
        create_directories()
        
        # Step 2: Generate data
        df = generate_data(n_samples=10000)
        
        # Step 3: Preprocess data
        X_processed, y_encoded, preprocessor = preprocess_data(df)
        
        # Step 4: Engineer features
        X_engineered, feature_engineer = engineer_features(X_processed, y_encoded, preprocessor)
        
        # Step 5: Train models
        results, X_test, y_test = train_models(X_engineered, y_encoded)
        
        # Step 6: Generate report
        report = generate_report(results, df)
        
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 Best model: {max(results.keys(), key=lambda k: results[k]['f1_score'])}")
        print("📁 Check reports/pipeline_report.txt for detailed results")
        
        # Save results summary
        import json
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(df),
            'models_trained': len(results),
            'best_model': max(results.keys(), key=lambda k: results[k]['f1_score']),
            'best_f1_score': max(results[k]['f1_score'] for k in results.keys()),
            'best_business_f1': max(results[k]['business_f1'] for k in results.keys()),
            'model_results': {k: {
                'accuracy': float(v['accuracy']),
                'f1_score': float(v['f1_score']),
                'f1_macro': float(v['f1_macro']),
                'roc_auc': float(v['roc_auc']),
                'business_f1': float(v['business_f1'])
            } for k, v in results.items()}
        }
        
        with open('reports/results_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)