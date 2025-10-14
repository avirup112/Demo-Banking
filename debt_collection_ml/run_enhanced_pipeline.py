#!/usr/bin/env python3
"""
Enhanced Debt Collection ML Pipeline with SHAP Explainability
Includes DagsHub tracking, SHAP explanations, Streamlit dashboard, and DVC compatibility
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
import json
import subprocess
import threading

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

# Import modules
from data.data_generator import DebtDataGenerator
from data.data_preprocessor import AdvancedDataPreprocessor
from utils.dagshub_integration import DagsHubTracker
from explainability.shap_explainer import DebtCollectionExplainer

from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import shap

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories including DagsHub storage"""
    directories = [
        'data/raw', 'data/processed', 'data/features',
        'models/trained', 'models/artifacts', 'models/dagshub',
        'reports', 'explanations', 'experiments', 'experiments/dagshub',
        'metrics/dagshub', 'artifacts/dagshub'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Directory structure created with DagsHub storage")

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
    """Preprocess the data with enhanced error handling"""
    
    logger.info("Starting data preprocessing...")
    
    # Drop customer_id if it exists
    if 'customer_id' in df.columns:
        df = df.drop('customer_id', axis=1)
    
    # Initialize preprocessor
    preprocessor = AdvancedDataPreprocessor(
        imputation_strategy='simple',  # Use simple instead of iterative for stability
        scaling_method='robust',
        encoding_method='target',
        handle_outliers=True
    )
    
    # Fit and transform
    X_processed, y_encoded = preprocessor.fit_transform(df, target_column='payment_status')
    
    # Ensure X_processed is numeric
    if hasattr(X_processed, 'dtype') and X_processed.dtype == 'object':
        X_processed = pd.DataFrame(X_processed).apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    # Convert to float array
    X_processed = np.array(X_processed, dtype=float)
    
    # Save processed data
    np.save('data/processed/X_processed.npy', X_processed)
    np.save('data/processed/y_encoded.npy', y_encoded)
    
    # Save preprocessor
    joblib.dump(preprocessor, 'models/artifacts/preprocessor.joblib')
    
    logger.info(f"Preprocessing completed. Shape: {X_processed.shape}")
    
    return X_processed, y_encoded, preprocessor

def engineer_features(X_processed: np.ndarray, y_encoded: np.ndarray) -> tuple:
    """Engineer advanced features with proper data type handling"""
    
    logger.info("Starting advanced feature engineering...")
    
    # Ensure X_processed is numeric
    X_processed = np.array(X_processed, dtype=float)
    
    logger.info(f"Data converted to numeric. Shape: {X_processed.shape}")
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_processed)
    
    logger.info(f"Created polynomial features. Shape: {X_poly.shape}")
    
    # Select best features
    selector = SelectKBest(score_func=f_classif, k=min(50, X_poly.shape[1]))
    X_engineered = selector.fit_transform(X_poly, y_encoded)
    
    # Save engineered data
    np.save('data/processed/X_engineered.npy', X_engineered)
    
    # Save feature engineering components
    feature_engineer = {'poly': poly, 'selector': selector}
    joblib.dump(feature_engineer, 'models/artifacts/feature_engineer.joblib')
    
    logger.info(f"Feature engineering completed. Shape: {X_engineered.shape}")
    
    return X_engineered, feature_engineer

def train_models(X_engineered: np.ndarray, y_encoded: np.ndarray, dagshub_tracker=None):
    """Train multiple models with class imbalance handling and DagsHub storage"""
    
    logger.info("Starting model training with class balancing...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Apply SMOTE for class balancing
    logger.info("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    logger.info(f"Train shape: {X_train_balanced.shape}, Test shape: {X_test.shape}")
    
    # Define models
    models = {
        'RandomForest_Optimized': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced', random_state=42
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced', random_state=42
        ),
        'LogisticRegression_Balanced': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        )
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1
        )
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        try:
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train_balanced, y_train_balanced)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'f1_score': f1_weighted,
                'f1_macro': f1_macro,
                'roc_auc': roc_auc,
                'business_f1': f1_weighted
            }
            
            # Store trained model
            trained_models[name] = model
            
            # Save model locally
            joblib.dump(model, f'models/trained/{name}.joblib')
            
            # Save model to DagsHub storage
            if dagshub_tracker:
                dagshub_tracker.log_model_to_dagshub(model, name)
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1_weighted:.4f}, ROC-AUC: {roc_auc:.4f}")
            
            # Log to DagsHub
            if dagshub_tracker:
                with dagshub_tracker.start_experiment(f"model_{name}"):
                    dagshub_tracker.log_metrics({
                        f'{name}_accuracy': accuracy,
                        f'{name}_f1_weighted': f1_weighted,
                        f'{name}_f1_macro': f1_macro,
                        f'{name}_roc_auc': roc_auc
                    })
            
            # Print classification report
            print(f"\n{'='*50}")
            print(f"{name} Classification Report:")
            print(f"{'='*50}")
            print(classification_report(y_test, y_pred, target_names=['Not Paid', 'Paid', 'Partially Paid']))
            
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")
            continue
    
    return results, trained_models, X_test, y_test

def generate_shap_explanations(trained_models: dict, X_test: np.ndarray, 
                              y_test: np.ndarray, dagshub_tracker=None):
    """Generate SHAP explanations for the best model"""
    
    logger.info("Generating SHAP explanations...")
    
    try:
        # Get the best model (first available model)
        best_model_name = list(trained_models.keys())[0]
        best_model = trained_models[best_model_name]
        
        logger.info(f"Creating SHAP explanations for {best_model_name}")
        
        # Determine model type
        if 'forest' in best_model_name.lower() or 'tree' in best_model_name.lower():
            model_type = 'tree'
        elif 'xgb' in best_model_name.lower() or 'lgb' in best_model_name.lower():
            model_type = 'tree'
        else:
            model_type = 'linear'
        
        # Create explainer
        explainer = DebtCollectionExplainer(best_model, model_type)
        
        # Use subset for speed
        sample_size = min(100, len(X_test))
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test[sample_indices]
        
        # Generate feature names
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
        
        # Create explainer
        explainer.create_explainer(X_sample[:50], feature_names)
        
        # Get global feature importance
        global_importance = explainer.global_feature_importance(X_sample)
        
        # Create summary plot
        summary_plot_path = explainer.create_summary_plot(X_sample[:50])
        
        # Store SHAP artifacts to DagsHub
        if dagshub_tracker:
            dagshub_tracker.log_artifact_to_dagshub(summary_plot_path, "shap_summary_plot.png", "visualization")
        
        # Log to DagsHub if available
        if dagshub_tracker:
            with dagshub_tracker.start_experiment("shap_explanations"):
                # Log top features as metrics
                for i, feature in enumerate(global_importance['top_features'][:10]):
                    dagshub_tracker.log_metrics({
                        f'top_feature_{i+1}_importance': feature['importance']
                    })
                
                # Log model explainability metrics
                dagshub_tracker.log_metrics({
                    'shap_samples_analyzed': sample_size,
                    'shap_features_analyzed': len(feature_names),
                    'top_feature_importance': global_importance['top_features'][0]['importance']
                })
        
        logger.info("SHAP explanations generated successfully")
        
        return {
            'global_importance': global_importance,
            'summary_plot': summary_plot_path,
            'model_name': best_model_name,
            'samples_analyzed': sample_size
        }
        
    except Exception as e:
        logger.error(f"SHAP explanation generation failed: {e}")
        return None

def generate_comprehensive_report(results: dict, df: pd.DataFrame, 
                                shap_results: dict = None, dagshub_tracker=None):
    """Generate comprehensive report with SHAP insights and DagsHub storage"""
    
    logger.info("Generating comprehensive report...")
    
    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['f1_score'])
    
    # Create model comparison DataFrame
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'F1-Score': metrics['f1_score'],
            'F1-Macro': metrics['f1_macro'],
            'ROC-AUC': metrics['roc_auc'],
            'Business F1': metrics['business_f1']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Business F1', ascending=False)
    
    # Save comparison
    comparison_df.to_csv('reports/model_comparison.csv', index=False)
    
    # Build comprehensive report (without unicode characters)
    target_achieved = results[best_model]['f1_score'] >= 0.65
    target_status = "ACHIEVED" if target_achieved else "NOT MET"
    
    report_lines = [
        "ENHANCED DEBT COLLECTION ML PIPELINE REPORT",
        "=" * 55,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "DATASET OVERVIEW:",
        f"- Total samples: {len(df):,}",
        f"- Features after preprocessing: {np.load('data/processed/X_processed.npy').shape[1]}",
        f"- Features after engineering: {np.load('data/processed/X_engineered.npy').shape[1]}",
        "",
        "TARGET DISTRIBUTION:",
        df['payment_status'].value_counts().to_string(),
        "",
        "MODEL PERFORMANCE:",
        "-" * 30
    ]
    
    # Add model results
    for _, row in comparison_df.iterrows():
        report_lines.extend([
            "",
            f"{row['Model']}:",
            f"  - Accuracy: {row['Accuracy']:.4f}",
            f"  - F1-Score (Weighted): {row['F1-Score']:.4f}",
            f"  - F1-Score (Macro): {row['F1-Macro']:.4f}",
            f"  - Business F1: {row['Business F1']:.4f}",
            f"  - ROC-AUC: {row['ROC-AUC']:.4f}"
        ])
    
    # Add SHAP insights if available
    if shap_results:
        report_lines.extend([
            "",
            "EXPLAINABILITY INSIGHTS (SHAP):",
            "=" * 35,
            f"Model Analyzed: {shap_results['model_name']}",
            f"Samples Analyzed: {shap_results['samples_analyzed']}",
            "",
            "TOP 10 FEATURES DRIVING REPAYMENT DECISIONS:"
        ])
        
        for i, feature in enumerate(shap_results['global_importance']['top_features'][:10], 1):
            report_lines.append(f"{i:2d}. {feature['feature']:<25} (Importance: {feature['importance']:.4f})")
        
        report_lines.extend([
            "",
            f"SHAP Summary Plot: {shap_results['summary_plot']}"
        ])
    
    # Add summary and next steps
    report_lines.extend([
        "",
        f"BEST MODEL: {best_model}",
        f"Best F1-Score: {results[best_model]['f1_score']:.4f}",
        f"Performance Target (0.65): {target_status}",
        "",
        "DAGSHUB INTEGRATION:",
        "- Models stored in: models/dagshub/",
        "- Metrics stored in: metrics/dagshub/",
        "- Experiments tracked in: experiments/dagshub/",
        "- Artifacts stored in: artifacts/dagshub/",
        "",
        "FILES GENERATED:",
        "- data/raw/debt_collection_data.csv",
        "- data/processed/X_processed.npy",
        "- data/processed/X_engineered.npy", 
        "- data/processed/y_encoded.npy",
        "- models/artifacts/preprocessor.joblib",
        "- models/artifacts/feature_engineer.joblib",
        "- models/trained/*.joblib",
        "- models/dagshub/*.joblib (DagsHub storage)",
        "- reports/model_comparison.csv",
        "- explanations/shap_summary.png (if SHAP enabled)",
        "",
        "STREAMLIT DASHBOARD:",
        "- Launch with: streamlit run streamlit_dashboard.py",
        "- Access at: http://localhost:8501",
        "",
        "NEXT STEPS:",
        "1. Deploy the best model for production use",
        "2. Set up monitoring and drift detection",
        "3. Use SHAP insights for feature engineering",
        "4. Implement A/B testing framework",
        "5. Access interactive dashboard for predictions",
        "",
        "=" * 55
    ])
    
    # Join all lines into final report
    report = "\n".join(report_lines)
    
    # Save report with UTF-8 encoding
    with open('reports/enhanced_pipeline_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Store report to DagsHub
    if dagshub_tracker:
        dagshub_tracker.log_artifact_to_dagshub('reports/enhanced_pipeline_report.txt', 
                                               'enhanced_pipeline_report.txt', 'report')
    
    # Save summary metrics for DVC
    summary_metrics = {
        'best_model': best_model,
        'best_f1_score': results[best_model]['f1_score'],
        'best_accuracy': results[best_model]['accuracy'],
        'best_roc_auc': results[best_model]['roc_auc'],
        'target_achieved': bool(target_achieved),
        'timestamp': datetime.now().isoformat(),
        'dagshub_integration': True,
        'streamlit_available': True
    }
    
    if shap_results:
        summary_metrics.update({
            'shap_top_feature': str(shap_results['global_importance']['top_features'][0]['feature']),
            'shap_top_importance': str(shap_results['global_importance']['top_features'][0]['importance'])
        })
    
    # Save metrics for DVC with UTF-8 encoding
    with open('reports/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(summary_metrics, f, indent=2)
    
    # Store metrics to DagsHub
    if dagshub_tracker:
        dagshub_tracker.log_metrics_to_dagshub(summary_metrics)
    
    print(report)
    
    return report, summary_metrics

def launch_streamlit_dashboard():
    """Launch Streamlit dashboard in a separate thread"""
    
    def run_streamlit():
        try:
            logger.info("Starting Streamlit dashboard...")
            subprocess.run([
                "streamlit", "run", "streamlit_dashboard.py", 
                "--server.port", "8501",
                "--server.headless", "true"
            ])
        except Exception as e:
            logger.error(f"Failed to launch Streamlit: {e}")
    
    # Launch in separate thread
    dashboard_thread = threading.Thread(target=run_streamlit, daemon=True)
    dashboard_thread.start()
    
    return dashboard_thread

def main():
    """Main pipeline execution with enhanced features"""
    
    parser = argparse.ArgumentParser(description='Enhanced Debt Collection ML Pipeline')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--dagshub-owner', type=str, help='DagsHub repository owner')
    parser.add_argument('--dagshub-repo', type=str, help='DagsHub repository name')
    parser.add_argument('--enable-shap', action='store_true', help='Enable SHAP explanations')
    parser.add_argument('--launch-dashboard', action='store_true', help='Launch Streamlit dashboard')
    
    args = parser.parse_args()
    
    print("🚀 Starting Enhanced Debt Collection ML Pipeline...")
    print("=" * 65)
    
    # Initialize DagsHub tracking if provided
    dagshub_tracker = None
    if args.dagshub_owner and args.dagshub_repo:
        try:
            dagshub_tracker = DagsHubTracker(args.dagshub_owner, args.dagshub_repo)
            print(f"📊 DagsHub tracking enabled: {args.dagshub_owner}/{args.dagshub_repo}")
        except Exception as e:
            logger.warning(f"DagsHub tracking failed: {e}")
    
    try:
        # Start main experiment tracking
        if dagshub_tracker:
            with dagshub_tracker.start_experiment("enhanced_pipeline"):
                dagshub_tracker.log_params({
                    'samples': args.samples,
                    'pipeline_version': 'enhanced_v2.0',
                    'shap_enabled': args.enable_shap,
                    'streamlit_enabled': args.launch_dashboard,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Step 1: Setup
        create_directories()
        
        # Step 2: Generate data
        df = generate_data(n_samples=args.samples)
        
        # Step 3: Preprocess data
        X_processed, y_encoded, preprocessor = preprocess_data(df)
        
        # Step 4: Engineer features
        X_engineered, engineer = engineer_features(X_processed, y_encoded)
        
        # Step 5: Train models
        results, trained_models, X_test, y_test = train_models(X_engineered, y_encoded, dagshub_tracker)
        
        # Step 6: Generate SHAP explanations (if enabled)
        shap_results = None
        if args.enable_shap and trained_models:
            shap_results = generate_shap_explanations(trained_models, X_test, y_test, dagshub_tracker)
        
        # Step 7: Generate comprehensive report
        report, summary_metrics = generate_comprehensive_report(results, df, shap_results, dagshub_tracker)
        
        # Step 8: Launch Streamlit Dashboard (if enabled)
        dashboard_thread = None
        if args.launch_dashboard:
            try:
                dashboard_thread = launch_streamlit_dashboard()
                print("🌐 Streamlit dashboard launching at: http://localhost:8501")
                print("📊 Dashboard will be available in a few seconds...")
            except Exception as e:
                logger.warning(f"Failed to launch dashboard: {e}")
        
        # Log final metrics to DagsHub
        if dagshub_tracker:
            with dagshub_tracker.start_experiment("final_results"):
                dagshub_tracker.log_metrics(summary_metrics)
        
        print("✅ Enhanced pipeline completed successfully!")
        print(f"📊 Best model: {summary_metrics['best_model']}")
        print(f"📈 Best F1 Score: {summary_metrics['best_f1_score']:.4f}")
        print("📁 Check reports/ directory for detailed results")
        print("🔗 Check DagsHub directories for remote artifacts")
        
        if args.launch_dashboard:
            print("🌐 Streamlit dashboard available at: http://localhost:8501")
            print("   Press Ctrl+C to stop the pipeline (dashboard will continue running)")
            
            # Keep main thread alive if dashboard is running
            if dashboard_thread:
                try:
                    dashboard_thread.join()
                except KeyboardInterrupt:
                    print("\n👋 Pipeline stopped. Dashboard may still be running.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")
        
        if dagshub_tracker:
            with dagshub_tracker.start_experiment("pipeline_error"):
                dagshub_tracker.log_params({'error': str(e)})

if __name__ == "__main__":
    main()