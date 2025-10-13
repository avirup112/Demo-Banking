import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score,
                           precision_score, recall_score, log_loss)
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from typing import Dict, List, Any, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation with business metrics"""
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ['Not Paid', 'Partially Paid', 'Paid']
        self.evaluation_results = {}
        
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str = "Model") -> Dict[str, Any]:
        """Comprehensive model evaluation with business metrics"""
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ROC-AUC for multiclass
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            roc_auc = 0.0
        
        # Log loss
        try:
            logloss = log_loss(y_test, y_pred_proba)
        except:
            logloss = np.inf
        
        # Business metrics for debt collection
        business_metrics = self._calculate_business_metrics(y_test, y_pred, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, 
                                           target_names=self.class_names, 
                                           output_dict=True)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'log_loss': logloss,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'business_metrics': business_metrics,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        self.evaluation_results[model_name] = results
        return results
    
    def _calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate business-specific metrics for debt collection"""
        
        # Assuming class mapping: 0=Not Paid, 1=Partially Paid, 2=Paid
        
        # Recovery rate (percentage of cases predicted to pay that actually pay)
        predicted_to_pay = (y_pred >= 1)  # Partially paid or paid
        actually_paid = (y_true >= 1)
        
        if predicted_to_pay.sum() > 0:
            recovery_precision = (predicted_to_pay & actually_paid).sum() / predicted_to_pay.sum()
        else:
            recovery_precision = 0.0
        
        # Collection efficiency (percentage of actual payers identified)
        if actually_paid.sum() > 0:
            collection_recall = (predicted_to_pay & actually_paid).sum() / actually_paid.sum()
        else:
            collection_recall = 0.0
        
        # Priority accuracy (how well we identify high-priority cases)
        high_priority_actual = (y_true == 2)  # Fully paid cases
        high_priority_pred = (y_pred == 2)
        
        if high_priority_actual.sum() > 0:
            priority_accuracy = (high_priority_actual & high_priority_pred).sum() / high_priority_actual.sum()
        else:
            priority_accuracy = 0.0
        
        # Expected recovery value (using probabilities)
        # Assuming partial payment = 50% recovery, full payment = 100% recovery
        recovery_weights = np.array([0.0, 0.5, 1.0])  # Not paid, partial, full
        expected_recovery = np.sum(y_pred_proba * recovery_weights, axis=1).mean()
        
        # Risk assessment accuracy (identifying non-payers)
        non_payers_actual = (y_true == 0)
        non_payers_pred = (y_pred == 0)
        
        if non_payers_actual.sum() > 0:
            risk_accuracy = (non_payers_actual & non_payers_pred).sum() / non_payers_actual.sum()
        else:
            risk_accuracy = 0.0
        
        return {
            'recovery_precision': recovery_precision,
            'collection_recall': collection_recall,
            'priority_accuracy': priority_accuracy,
            'expected_recovery_rate': expected_recovery,
            'risk_assessment_accuracy': risk_accuracy,
            'business_f1': 2 * (recovery_precision * collection_recall) / (recovery_precision + collection_recall) if (recovery_precision + collection_recall) > 0 else 0.0
        }
    
    def plot_evaluation_results(self, model_name: str = None):
        """Create comprehensive evaluation plots"""
        
        if model_name and model_name in self.evaluation_results:
            results = self.evaluation_results[model_name]
        else:
            results = list(self.evaluation_results.values())[0]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confusion Matrix', 'ROC Curves', 'Precision-Recall', 'Business Metrics'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Confusion Matrix
        cm = results['confusion_matrix']
        fig.add_trace(
            go.Heatmap(z=cm, x=self.class_names, y=self.class_names,
                      colorscale='Blues', showscale=False),
            row=1, col=1
        )
        
        # Business Metrics Bar Chart
        business_metrics = results['business_metrics']
        fig.add_trace(
            go.Bar(x=list(business_metrics.keys()), 
                  y=list(business_metrics.values()),
                  name='Business Metrics'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title=f"Model Evaluation: {results['model_name']}")
        fig.show()
    
    def compare_models(self) -> pd.DataFrame:
        """Compare multiple models"""
        
        if len(self.evaluation_results) < 2:
            print("Need at least 2 models for comparison")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, results in self.evaluation_results.items():
            row = {
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc'],
                'Log Loss': results['log_loss'],
                'Business F1': results['business_metrics']['business_f1'],
                'Recovery Precision': results['business_metrics']['recovery_precision'],
                'Collection Recall': results['business_metrics']['collection_recall'],
                'Expected Recovery': results['business_metrics']['expected_recovery_rate']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.round(4)

class DebtCollectionMLModel:
    """Advanced ML Model for debt collection with MLOps integration"""
    
    def __init__(self, model_type: str = 'ensemble', random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.cv_scores = None
        self.evaluator = ModelEvaluator()
        
        # MLflow setup
        mlflow.set_experiment("debt_collection_ml")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_model_configs(self) -> Dict[str, Dict]:
        """Get model configurations for different algorithms"""
        
        configs = {
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=self.random_state, eval_metric='mlogloss'),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
        }
        
        return configs
    
    def create_ensemble_model(self, X_train: np.ndarray, y_train: np.ndarray) -> VotingClassifier:
        """Create ensemble model with multiple algorithms"""
        
        # Base models for ensemble
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=self.random_state, eval_metric='mlogloss'
        )
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=self.random_state, verbose=-1
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            random_state=self.random_state
        )
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('rf', rf_model)
            ],
            voting='soft'
        )
        
        return ensemble
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                                cv_folds: int = 5, n_trials: int = 100) -> Dict[str, Any]:
        """Advanced hyperparameter optimization using Optuna"""
        
        def objective(trial):
            if self.model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
                }
                model = xgb.XGBClassifier(random_state=self.random_state, eval_metric='mlogloss', **params)
                
            elif self.model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
                }
                model = lgb.LGBMClassifier(random_state=self.random_state, verbose=-1, **params)
                
            elif self.model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
                model = RandomForestClassifier(random_state=self.random_state, **params)
                
            else:
                # Default to XGBoost
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2)
                }
                model = xgb.XGBClassifier(random_state=self.random_state, eval_metric='mlogloss', **params)
            
            # Handle class imbalance
            sampling_strategy = trial.suggest_categorical('sampling_strategy', ['SMOTE', 'ADASYN', 'SMOTETomek'])
            
            if sampling_strategy == 'SMOTE':
                sampler = SMOTE(random_state=self.random_state)
            elif sampling_strategy == 'ADASYN':
                sampler = ADASYN(random_state=self.random_state)
            else:
                sampler = SMOTETomek(random_state=self.random_state)
            
            # Create pipeline
            pipeline = ImbPipeline([
                ('sampler', sampler),
                ('classifier', model)
            ])
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted')
            
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        
        self.logger.info(f"Best parameters: {self.best_params}")
        self.logger.info(f"Best score: {study.best_value:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              optimize: bool = True, n_trials: int = 50) -> Dict[str, Any]:
        """Train the model with MLflow tracking"""
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("optimize_hyperparameters", optimize)
            
            if optimize:
                self.logger.info("Starting hyperparameter optimization...")
                optimization_results = self.optimize_hyperparameters(X_train, y_train, n_trials=n_trials)
                
                # Log optimization results
                mlflow.log_param("optimization_trials", n_trials)
                mlflow.log_metric("best_cv_score", optimization_results['best_score'])
                
                # Extract best parameters
                best_params = {k: v for k, v in self.best_params.items() if k != 'sampling_strategy'}
                sampling_strategy = self.best_params.get('sampling_strategy', 'SMOTE')
            else:
                best_params = {}
                sampling_strategy = 'SMOTE'
            
            # Create final model
            if self.model_type == 'ensemble':
                model = self.create_ensemble_model(X_train, y_train)
                sampler = SMOTE(random_state=self.random_state)
            else:
                configs = self.get_model_configs()
                if self.model_type in configs:
                    model_class = configs[self.model_type]['model'].__class__
                    model = model_class(random_state=self.random_state, **best_params)
                else:
                    model = xgb.XGBClassifier(random_state=self.random_state, eval_metric='mlogloss', **best_params)
                
                # Setup sampler
                if sampling_strategy == 'ADASYN':
                    sampler = ADASYN(random_state=self.random_state)
                elif sampling_strategy == 'SMOTETomek':
                    sampler = SMOTETomek(random_state=self.random_state)
                else:
                    sampler = SMOTE(random_state=self.random_state)
            
            # Create final pipeline
            self.model = ImbPipeline([
                ('sampler', sampler),
                ('classifier', model)
            ])
            
            # Train model
            self.logger.info("Training final model...")
            self.model.fit(X_train, y_train)
            
            # Cross-validation scores
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='f1_weighted')
            self.cv_scores = cv_scores
            
            # Log cross-validation results
            mlflow.log_metric("cv_mean_f1", cv_scores.mean())
            mlflow.log_metric("cv_std_f1", cv_scores.std())
            
            # Validation evaluation if provided
            if X_val is not None and y_val is not None:
                val_results = self.evaluator.evaluate_model(self.model, X_val, y_val, "Validation")
                
                # Log validation metrics
                for metric, value in val_results['business_metrics'].items():
                    mlflow.log_metric(f"val_{metric}", value)
                
                mlflow.log_metric("val_accuracy", val_results['accuracy'])
                mlflow.log_metric("val_f1", val_results['f1_score'])
                mlflow.log_metric("val_roc_auc", val_results['roc_auc'])
            
            # Feature importance
            if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
                self.feature_importance = self.model.named_steps['classifier'].feature_importances_
            
            # Log model
            if self.model_type == 'xgboost':
                mlflow.xgboost.log_model(self.model.named_steps['classifier'], "model")
            elif self.model_type == 'lightgbm':
                mlflow.lightgbm.log_model(self.model.named_steps['classifier'], "model")
            else:
                mlflow.sklearn.log_model(self.model, "model")
            
            self.logger.info(f"Training completed. CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return {
                'cv_scores': cv_scores,
                'best_params': self.best_params,
                'model': self.model
            }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                model_name: str = "Test") -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        
        results = self.evaluator.evaluate_model(self.model, X_test, y_test, model_name)
        
        # Print results
        print(f"\n=== {model_name} Evaluation Results ===")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        print(f"ROC-AUC: {results['roc_auc']:.4f}")
        
        print(f"\n=== Business Metrics ===")
        for metric, value in results['business_metrics'].items():
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        
        return results
    
    def plot_feature_importance(self, feature_names: List[str], top_n: int = 20):
        """Plot feature importance"""
        
        if self.feature_importance is None:
            print("Feature importance not available for this model type")
            return
        
        # Get top features
        indices = np.argsort(self.feature_importance)[::-1][:top_n]
        
        fig = go.Figure(data=go.Bar(
            x=self.feature_importance[indices][::-1],
            y=[feature_names[i] for i in indices[::-1]],
            orientation='h'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importances',
            xaxis_title='Importance',
            yaxis_title='Features',
            height=600
        )
        
        fig.show()
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'cv_scores': self.cv_scores,
            'evaluator': self.evaluator
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.best_params = model_data['best_params']
        self.feature_importance = model_data['feature_importance']
        self.cv_scores = model_data['cv_scores']
        self.evaluator = model_data.get('evaluator', ModelEvaluator())
        
        self.logger.info(f"Model loaded from {filepath}")

class ModelComparison:
    """Compare multiple models for debt collection"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.comparison_df = None
        
    def add_model(self, name: str, model: DebtCollectionMLModel):
        """Add model to comparison"""
        self.models[name] = model
    
    def compare_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Compare all models"""
        
        comparison_data = []
        
        for name, model in self.models.items():
            if model.model is None:
                print(f"Model {name} not trained, skipping...")
                continue
            
            results = model.evaluate(X_test, y_test, name)
            self.results[name] = results
            
            row = {
                'Model': name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc'],
                'Business F1': results['business_metrics']['business_f1'],
                'Recovery Precision': results['business_metrics']['recovery_precision'],
                'Collection Recall': results['business_metrics']['collection_recall'],
                'Expected Recovery': results['business_metrics']['expected_recovery_rate']
            }
            comparison_data.append(row)
        
        self.comparison_df = pd.DataFrame(comparison_data)
        return self.comparison_df.round(4)
    
    def plot_comparison(self):
        """Plot model comparison"""
        
        if self.comparison_df is None:
            print("No comparison data available. Run compare_models first.")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Technical Metrics', 'Business Metrics', 'ROC-AUC Comparison', 'F1-Score Comparison')
        )
        
        models = self.comparison_df['Model']
        
        # Technical metrics
        fig.add_trace(
            go.Bar(x=models, y=self.comparison_df['Accuracy'], name='Accuracy'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=models, y=self.comparison_df['Precision'], name='Precision'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=models, y=self.comparison_df['Recall'], name='Recall'),
            row=1, col=1
        )
        
        # Business metrics
        fig.add_trace(
            go.Bar(x=models, y=self.comparison_df['Business F1'], name='Business F1'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=models, y=self.comparison_df['Recovery Precision'], name='Recovery Precision'),
            row=1, col=2
        )
        
        # ROC-AUC comparison
        fig.add_trace(
            go.Bar(x=models, y=self.comparison_df['ROC-AUC'], name='ROC-AUC'),
            row=2, col=1
        )
        
        # F1-Score comparison
        fig.add_trace(
            go.Bar(x=models, y=self.comparison_df['F1-Score'], name='F1-Score'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title="Model Comparison Dashboard")
        fig.show()
    
    def get_best_model(self, metric: str = 'Business F1') -> str:
        """Get best model based on specified metric"""
        
        if self.comparison_df is None:
            return None
        
        best_idx = self.comparison_df[metric].idxmax()
        best_model = self.comparison_df.loc[best_idx, 'Model']
        
        print(f"Best model based on {metric}: {best_model}")
        print(f"{metric} score: {self.comparison_df.loc[best_idx, metric]:.4f}")
        
        return best_model