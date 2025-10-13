import os
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import warnings
from pathlib import Path
import hashlib
import yaml

# MLOps and Monitoring
import mlflow
import mlflow.sklearn
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metrics import *

# Model versioning and registry
from sklearn.base import BaseEstimator
import sqlite3

warnings.filterwarnings('ignore')

class ModelRegistry:
    """Local model registry for version control and management"""
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database for metadata
        self.db_path = self.registry_path / "model_registry.db"
        self._init_database()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """Initialize model registry database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                model_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metrics TEXT,
                parameters TEXT,
                file_path TEXT,
                model_hash TEXT,
                status TEXT DEFAULT 'registered',
                description TEXT,
                tags TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS deployments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                environment TEXT,
                deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active',
                endpoint_url TEXT,
                FOREIGN KEY (model_id) REFERENCES models (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_model(self, model: BaseEstimator, name: str, version: str,
                      model_type: str, metrics: Dict[str, float],
                      parameters: Dict[str, Any], description: str = "",
                      tags: List[str] = None) -> str:
        """Register a new model version"""
        
        # Create model directory
        model_dir = self.registry_path / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)
        
        # Calculate model hash for integrity
        model_hash = self._calculate_model_hash(model_path)
        
        # Save metadata
        metadata = {
            'name': name,
            'version': version,
            'model_type': model_type,
            'metrics': metrics,
            'parameters': parameters,
            'description': description,
            'tags': tags or [],
            'created_at': datetime.now().isoformat(),
            'model_hash': model_hash
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Register in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO models (name, version, model_type, metrics, parameters, 
                              file_path, model_hash, description, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            name, version, model_type, json.dumps(metrics), json.dumps(parameters),
            str(model_path), model_hash, description, json.dumps(tags or [])
        ))
        
        model_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        self.logger.info(f"Model {name} v{version} registered with ID {model_id}")
        return str(model_id)
    
    def load_model(self, name: str, version: str = "latest") -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Load a model from registry"""
        
        if version == "latest":
            version = self.get_latest_version(name)
        
        model_path = self.registry_path / name / version / "model.joblib"
        metadata_path = self.registry_path / name / version / "metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model {name} v{version} not found")
        
        # Load model
        model = joblib.load(model_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.logger.info(f"Loaded model {name} v{version}")
        return model, metadata
    
    def get_latest_version(self, name: str) -> str:
        """Get latest version of a model"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT version FROM models 
            WHERE name = ? 
            ORDER BY created_at DESC 
            LIMIT 1
        ''', (name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]
        else:
            raise ValueError(f"No models found with name {name}")
    
    def list_models(self) -> pd.DataFrame:
        """List all registered models"""
        
        conn = sqlite3.connect(self.db_path)
        
        df = pd.read_sql_query('''
            SELECT name, version, model_type, created_at, status, description
            FROM models
            ORDER BY created_at DESC
        ''', conn)
        
        conn.close()
        return df
    
    def promote_model(self, name: str, version: str, environment: str) -> str:
        """Promote model to an environment"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get model ID
        cursor.execute('SELECT id FROM models WHERE name = ? AND version = ?', (name, version))
        result = cursor.fetchone()
        
        if not result:
            raise ValueError(f"Model {name} v{version} not found")
        
        model_id = result[0]
        
        # Deactivate previous deployments in this environment
        cursor.execute('''
            UPDATE deployments 
            SET status = 'inactive' 
            WHERE environment = ? AND status = 'active'
        ''', (environment,))
        
        # Create new deployment
        cursor.execute('''
            INSERT INTO deployments (model_id, environment, status)
            VALUES (?, ?, 'active')
        ''', (model_id, environment))
        
        deployment_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        self.logger.info(f"Model {name} v{version} promoted to {environment}")
        return str(deployment_id)
    
    def _calculate_model_hash(self, model_path: Path) -> str:
        """Calculate hash of model file for integrity checking"""
        
        hash_md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

class DataDriftMonitor:
    """Monitor data drift and model performance degradation"""
    
    def __init__(self, reference_data: pd.DataFrame, target_column: str = None):
        self.reference_data = reference_data
        self.target_column = target_column
        self.drift_reports = []
        
        # Setup column mapping for Evidently
        self.column_mapping = ColumnMapping()
        if target_column:
            self.column_mapping.target = target_column
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def detect_data_drift(self, current_data: pd.DataFrame, 
                         report_path: str = None) -> Dict[str, Any]:
        """Detect data drift between reference and current data"""
        
        # Create drift report
        data_drift_report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        
        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Save report if path provided
        if report_path:
            data_drift_report.save_html(report_path)
        
        # Extract key metrics
        drift_results = self._extract_drift_metrics(data_drift_report)
        
        # Store report
        self.drift_reports.append({
            'timestamp': datetime.now(),
            'report': data_drift_report,
            'results': drift_results
        })
        
        self.logger.info(f"Data drift analysis completed. Drift detected: {drift_results['drift_detected']}")
        
        return drift_results
    
    def detect_target_drift(self, current_data: pd.DataFrame,
                           report_path: str = None) -> Dict[str, Any]:
        """Detect target drift"""
        
        if self.target_column is None:
            raise ValueError("Target column not specified")
        
        target_drift_report = Report(metrics=[
            TargetDriftPreset(),
        ])
        
        target_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        if report_path:
            target_drift_report.save_html(report_path)
        
        # Extract target drift metrics
        target_drift_results = self._extract_target_drift_metrics(target_drift_report)
        
        self.logger.info(f"Target drift analysis completed. Drift detected: {target_drift_results['target_drift_detected']}")
        
        return target_drift_results
    
    def _extract_drift_metrics(self, report) -> Dict[str, Any]:
        """Extract key metrics from drift report"""
        
        # This is a simplified extraction - in practice, you'd parse the report JSON
        return {
            'drift_detected': True,  # Placeholder - would extract from actual report
            'drift_score': 0.15,     # Placeholder
            'drifted_features': [],  # Placeholder
            'timestamp': datetime.now().isoformat()
        }
    
    def _extract_target_drift_metrics(self, report) -> Dict[str, Any]:
        """Extract target drift metrics"""
        
        return {
            'target_drift_detected': False,  # Placeholder
            'target_drift_score': 0.05,     # Placeholder
            'timestamp': datetime.now().isoformat()
        }
    
    def get_drift_summary(self) -> pd.DataFrame:
        """Get summary of all drift reports"""
        
        if not self.drift_reports:
            return pd.DataFrame()
        
        summary_data = []
        for report_data in self.drift_reports:
            summary_data.append({
                'timestamp': report_data['timestamp'],
                'drift_detected': report_data['results']['drift_detected'],
                'drift_score': report_data['results']['drift_score'],
                'num_drifted_features': len(report_data['results']['drifted_features'])
            })
        
        return pd.DataFrame(summary_data)

class ModelPerformanceMonitor:
    """Monitor model performance over time"""
    
    def __init__(self, model_name: str, metrics_storage_path: str = "monitoring/metrics"):
        self.model_name = model_name
        self.metrics_storage_path = Path(metrics_storage_path)
        self.metrics_storage_path.mkdir(parents=True, exist_ok=True)
        
        self.performance_history = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def log_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_pred_proba: np.ndarray = None, 
                       additional_metrics: Dict[str, float] = None):
        """Log model performance metrics"""
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Calculate standard metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Add ROC-AUC if probabilities provided
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                metrics['roc_auc'] = 0.0
        
        # Add additional metrics
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Store metrics
        self.performance_history.append(metrics)
        
        # Save to file
        metrics_file = self.metrics_storage_path / f"{self.model_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
        
        self.logger.info(f"Performance logged for {self.model_name}: F1={metrics['f1_score']:.4f}")
    
    def detect_performance_degradation(self, threshold: float = 0.05) -> Dict[str, Any]:
        """Detect if model performance has degraded"""
        
        if len(self.performance_history) < 2:
            return {'degradation_detected': False, 'message': 'Insufficient data for comparison'}
        
        # Compare recent performance with baseline (first few measurements)
        baseline_metrics = self.performance_history[:3]  # First 3 measurements as baseline
        recent_metrics = self.performance_history[-3:]   # Last 3 measurements
        
        baseline_f1 = np.mean([m['f1_score'] for m in baseline_metrics])
        recent_f1 = np.mean([m['f1_score'] for m in recent_metrics])
        
        degradation = baseline_f1 - recent_f1
        degradation_detected = degradation > threshold
        
        result = {
            'degradation_detected': degradation_detected,
            'baseline_f1': baseline_f1,
            'recent_f1': recent_f1,
            'degradation_amount': degradation,
            'threshold': threshold,
            'recommendation': 'Consider retraining the model' if degradation_detected else 'Model performance is stable'
        }
        
        if degradation_detected:
            self.logger.warning(f"Performance degradation detected: {degradation:.4f} drop in F1 score")
        
        return result
    
    def get_performance_trends(self) -> pd.DataFrame:
        """Get performance trends over time"""
        
        if not self.performance_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.performance_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')

class MLOpsOrchestrator:
    """Orchestrate MLOps workflows"""
    
    def __init__(self, project_name: str = "debt_collection_ml"):
        self.project_name = project_name
        self.model_registry = ModelRegistry()
        self.drift_monitor = None
        self.performance_monitor = None
        
        # Setup MLflow
        mlflow.set_experiment(project_name)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_monitoring(self, reference_data: pd.DataFrame, target_column: str = None):
        """Setup monitoring components"""
        
        self.drift_monitor = DataDriftMonitor(reference_data, target_column)
        self.performance_monitor = ModelPerformanceMonitor(self.project_name)
        
        self.logger.info("Monitoring components initialized")
    
    def deploy_model_pipeline(self, model, model_name: str, version: str,
                             model_type: str, metrics: Dict[str, float],
                             parameters: Dict[str, Any]) -> str:
        """Deploy complete model pipeline"""
        
        # Register model
        model_id = self.model_registry.register_model(
            model=model,
            name=model_name,
            version=version,
            model_type=model_type,
            metrics=metrics,
            parameters=parameters,
            description=f"Debt collection model v{version}"
        )
        
        # Promote to staging
        deployment_id = self.model_registry.promote_model(model_name, version, "staging")
        
        self.logger.info(f"Model pipeline deployed: {model_name} v{version}")
        
        return model_id
    
    def run_monitoring_pipeline(self, current_data: pd.DataFrame,
                               y_true: np.ndarray = None, y_pred: np.ndarray = None,
                               y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """Run complete monitoring pipeline"""
        
        results = {}
        
        # Data drift monitoring
        if self.drift_monitor:
            drift_results = self.drift_monitor.detect_data_drift(current_data)
            results['data_drift'] = drift_results
            
            if self.drift_monitor.target_column and self.drift_monitor.target_column in current_data.columns:
                target_drift_results = self.drift_monitor.detect_target_drift(current_data)
                results['target_drift'] = target_drift_results
        
        # Performance monitoring
        if self.performance_monitor and y_true is not None and y_pred is not None:
            self.performance_monitor.log_performance(y_true, y_pred, y_pred_proba)
            
            degradation_results = self.performance_monitor.detect_performance_degradation()
            results['performance_degradation'] = degradation_results
        
        # Generate alerts
        alerts = self._generate_alerts(results)
        results['alerts'] = alerts
        
        return results
    
    def _generate_alerts(self, monitoring_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate alerts based on monitoring results"""
        
        alerts = []
        
        # Data drift alerts
        if 'data_drift' in monitoring_results:
            if monitoring_results['data_drift']['drift_detected']:
                alerts.append({
                    'type': 'data_drift',
                    'severity': 'warning',
                    'message': 'Data drift detected in input features',
                    'recommendation': 'Consider retraining the model with recent data'
                })
        
        # Target drift alerts
        if 'target_drift' in monitoring_results:
            if monitoring_results['target_drift']['target_drift_detected']:
                alerts.append({
                    'type': 'target_drift',
                    'severity': 'high',
                    'message': 'Target distribution has changed significantly',
                    'recommendation': 'Immediate model retraining recommended'
                })
        
        # Performance degradation alerts
        if 'performance_degradation' in monitoring_results:
            if monitoring_results['performance_degradation']['degradation_detected']:
                alerts.append({
                    'type': 'performance_degradation',
                    'severity': 'high',
                    'message': f"Model performance degraded by {monitoring_results['performance_degradation']['degradation_amount']:.4f}",
                    'recommendation': 'Schedule model retraining'
                })
        
        return alerts
    
    def generate_mlops_report(self) -> str:
        """Generate comprehensive MLOps report"""
        
        report = f"""
MLOps MONITORING REPORT - {self.project_name.upper()}
{'='*60}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL REGISTRY STATUS:
{'-'*30}
"""
        
        # Model registry info
        models_df = self.model_registry.list_models()
        if not models_df.empty:
            report += f"Total registered models: {len(models_df)}\n"
            report += f"Latest model: {models_df.iloc[0]['name']} v{models_df.iloc[0]['version']}\n"
        else:
            report += "No models registered\n"
        
        # Drift monitoring
        if self.drift_monitor:
            drift_summary = self.drift_monitor.get_drift_summary()
            if not drift_summary.empty:
                report += f"\nDATA DRIFT MONITORING:\n{'-'*30}\n"
                report += f"Total drift checks: {len(drift_summary)}\n"
                report += f"Drift detected: {drift_summary['drift_detected'].sum()} times\n"
        
        # Performance monitoring
        if self.performance_monitor:
            perf_trends = self.performance_monitor.get_performance_trends()
            if not perf_trends.empty:
                report += f"\nPERFORMANCE MONITORING:\n{'-'*30}\n"
                report += f"Performance measurements: {len(perf_trends)}\n"
                report += f"Latest F1 score: {perf_trends.iloc[-1]['f1_score']:.4f}\n"
                report += f"Average F1 score: {perf_trends['f1_score'].mean():.4f}\n"
        
        report += f"\n{'='*60}\n"
        
        return report
    
    def save_mlops_config(self, config_path: str = "config/mlops_config.yaml"):
        """Save MLOps configuration"""
        
        config = {
            'project_name': self.project_name,
            'model_registry': {
                'path': str(self.model_registry.registry_path),
                'database': str(self.model_registry.db_path)
            },
            'monitoring': {
                'drift_threshold': 0.1,
                'performance_threshold': 0.05,
                'alert_email': 'admin@company.com'
            },
            'deployment': {
                'staging_environment': 'staging',
                'production_environment': 'production',
                'auto_promote': False
            }
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.logger.info(f"MLOps configuration saved to {config_path}")

# CI/CD Pipeline Components
class ContinuousIntegration:
    """CI/CD pipeline for ML models"""
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def run_data_validation(self, data_path: str) -> Dict[str, Any]:
        """Run data validation tests"""
        
        try:
            df = pd.read_csv(data_path)
            
            validation_results = {
                'data_loaded': True,
                'row_count': len(df),
                'column_count': len(df.columns),
                'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'validation_passed': True
            }
            
            # Validation rules
            if validation_results['missing_data_percentage'] > 50:
                validation_results['validation_passed'] = False
                validation_results['error'] = "Too much missing data"
            
            if validation_results['duplicate_rows'] > len(df) * 0.1:
                validation_results['validation_passed'] = False
                validation_results['error'] = "Too many duplicate rows"
            
            self.logger.info(f"Data validation {'passed' if validation_results['validation_passed'] else 'failed'}")
            
            return validation_results
            
        except Exception as e:
            return {
                'data_loaded': False,
                'validation_passed': False,
                'error': str(e)
            }
    
    def run_model_tests(self, model_path: str, test_data_path: str) -> Dict[str, Any]:
        """Run model validation tests"""
        
        try:
            # Load model and test data
            model = joblib.load(model_path)
            test_data = pd.read_csv(test_data_path)
            
            # Basic model tests
            test_results = {
                'model_loaded': True,
                'can_predict': False,
                'prediction_shape_correct': False,
                'no_nan_predictions': False,
                'tests_passed': False
            }
            
            # Test prediction
            X_test = test_data.drop(['Outcome'], axis=1, errors='ignore')
            if 'Customer_ID' in X_test.columns:
                X_test = X_test.drop(['Customer_ID'], axis=1)
            
            predictions = model.predict(X_test[:10])  # Test with small sample
            test_results['can_predict'] = True
            
            # Check prediction shape
            if len(predictions) == 10:
                test_results['prediction_shape_correct'] = True
            
            # Check for NaN predictions
            if not np.isnan(predictions).any():
                test_results['no_nan_predictions'] = True
            
            # Overall test result
            test_results['tests_passed'] = all([
                test_results['can_predict'],
                test_results['prediction_shape_correct'],
                test_results['no_nan_predictions']
            ])
            
            self.logger.info(f"Model tests {'passed' if test_results['tests_passed'] else 'failed'}")
            
            return test_results
            
        except Exception as e:
            return {
                'model_loaded': False,
                'tests_passed': False,
                'error': str(e)
            }
    
    def generate_ci_report(self, data_validation: Dict, model_tests: Dict) -> str:
        """Generate CI/CD report"""
        
        report = f"""
CI/CD PIPELINE REPORT
{'='*40}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA VALIDATION:
{'-'*20}
Status: {'PASSED' if data_validation.get('validation_passed', False) else 'FAILED'}
Rows: {data_validation.get('row_count', 'N/A')}
Columns: {data_validation.get('column_count', 'N/A')}
Missing Data: {data_validation.get('missing_data_percentage', 0):.2f}%
Duplicates: {data_validation.get('duplicate_rows', 'N/A')}

MODEL TESTS:
{'-'*20}
Status: {'PASSED' if model_tests.get('tests_passed', False) else 'FAILED'}
Model Loaded: {'✓' if model_tests.get('model_loaded', False) else '✗'}
Can Predict: {'✓' if model_tests.get('can_predict', False) else '✗'}
Correct Shape: {'✓' if model_tests.get('prediction_shape_correct', False) else '✗'}
No NaN Predictions: {'✓' if model_tests.get('no_nan_predictions', False) else '✗'}

OVERALL STATUS: {'PASSED' if data_validation.get('validation_passed', False) and model_tests.get('tests_passed', False) else 'FAILED'}
{'='*40}
"""
        
        return report