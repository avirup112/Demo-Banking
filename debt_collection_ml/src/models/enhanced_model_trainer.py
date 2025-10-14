import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
import optuna
import joblib
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced models
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

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedModelTrainer:
    """Enhanced model trainer with Optuna hyperparameter optimization and ensemble methods"""
    
    def __init__(self, 
                 optimize_hyperparameters: bool = True,
                 n_trials: int = 100,
                 cv_folds: int = 5,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 use_time_series_cv: bool = False,
                 target_metric: str = 'f1_weighted',
                 class_balance_method: str = 'smote'):
        
        self.optimize_hyperparameters = optimize_hyperparameters
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.use_time_series_cv = use_time_series_cv
        self.target_metric = target_metric
        self.class_balance_method = class_balance_method
        
        # Model storage
        self.trained_models = {}
        self.best_models = {}
        self.model_results = {}
        self.ensemble_model = None
        
        # Optimization studies
        self.studies = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Set up Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def _get_class_balancer(self, method: str = 'smote'):
        """Get class balancing method"""
        
        balancers = {
            'smote': SMOTE(random_state=self.random_state, k_neighbors=3),
            'adasyn': ADASYN(random_state=self.random_state),
            'borderline': BorderlineSMOTE(random_state=self.random_state),
            'smote_tomek': SMOTETomek(random_state=self.random_state),
            'smote_enn': SMOTEENN(random_state=self.random_state)
        }
        
        return balancers.get(method, SMOTE(random_state=self.random_state))
    
    def _get_cv_splitter(self):
        """Get cross-validation splitter"""
        
        if self.use_time_series_cv:
            return TimeSeriesSplit(n_splits=self.cv_folds)
        else:
            from sklearn.model_selection import StratifiedKFold
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
    
    def _objective_random_forest(self, trial, X_train, y_train, cv_splitter):
        """Optuna objective function for Random Forest"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
            'class_weight': 'balanced',
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        
        # Cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, 
                               scoring='f1_weighted', n_jobs=-1)
        
        return scores.mean()
    
    def _objective_extra_trees(self, trial, X_train, y_train, cv_splitter):
        """Optuna objective function for Extra Trees"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 35),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
            'class_weight': 'balanced',
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        model = ExtraTreesClassifier(**params)
        
        scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, 
                               scoring='f1_weighted', n_jobs=-1)
        
        return scores.mean()
    
    def _objective_logistic_regression(self, trial, X_train, y_train, cv_splitter):
        """Optuna objective function for Logistic Regression"""
        
        params = {
            'C': trial.suggest_float('C', 0.001, 100, log=True),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'saga']),
            'max_iter': trial.suggest_int('max_iter', 1000, 5000),
            'class_weight': 'balanced',
            'random_state': self.random_state
        }
        
        # Handle solver-specific parameters
        if params['solver'] == 'liblinear':
            params['penalty'] = trial.suggest_categorical('penalty', ['l1', 'l2'])\n        elif params['solver'] == 'saga':\n            params['penalty'] = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])\n            if params['penalty'] == 'elasticnet':\n                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)\n        else:  # lbfgs\n            params['penalty'] = 'l2'\n        \n        model = LogisticRegression(**params)\n        \n        scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, \n                               scoring='f1_weighted', n_jobs=-1)\n        \n        return scores.mean()\n    \n    def _objective_xgboost(self, trial, X_train, y_train, cv_splitter):\n        \"\"\"Optuna objective function for XGBoost\"\"\"\n        \n        params = {\n            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),\n            'max_depth': trial.suggest_int('max_depth', 3, 15),\n            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),\n            'subsample': trial.suggest_float('subsample', 0.6, 1.0),\n            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),\n            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),\n            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),\n            'random_state': self.random_state,\n            'n_jobs': -1,\n            'eval_metric': 'mlogloss'\n        }\n        \n        model = xgb.XGBClassifier(**params)\n        \n        scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, \n                               scoring='f1_weighted', n_jobs=-1)\n        \n        return scores.mean()\n    \n    def _objective_lightgbm(self, trial, X_train, y_train, cv_splitter):\n        \"\"\"Optuna objective function for LightGBM\"\"\"\n        \n        params = {\n            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),\n            'max_depth': trial.suggest_int('max_depth', 3, 15),\n            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),\n            'subsample': trial.suggest_float('subsample', 0.6, 1.0),\n            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),\n            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),\n            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),\n            'class_weight': 'balanced',\n            'random_state': self.random_state,\n            'n_jobs': -1,\n            'verbose': -1\n        }\n        \n        model = lgb.LGBMClassifier(**params)\n        \n        scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, \n                               scoring='f1_weighted', n_jobs=-1)\n        \n        return scores.mean()\n    \n    def optimize_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:\n        \"\"\"Optimize hyperparameters for a specific model\"\"\"\n        \n        self.logger.info(f\"Optimizing {model_name} hyperparameters...\")\n        \n        cv_splitter = self._get_cv_splitter()\n        \n        # Create study\n        study = optuna.create_study(direction='maximize', \n                                  study_name=f'{model_name}_optimization')\n        \n        # Define objective function based on model type\n        if model_name == 'RandomForest':\n            objective = lambda trial: self._objective_random_forest(trial, X_train, y_train, cv_splitter)\n        elif model_name == 'ExtraTrees':\n            objective = lambda trial: self._objective_extra_trees(trial, X_train, y_train, cv_splitter)\n        elif model_name == 'LogisticRegression':\n            objective = lambda trial: self._objective_logistic_regression(trial, X_train, y_train, cv_splitter)\n        elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:\n            objective = lambda trial: self._objective_xgboost(trial, X_train, y_train, cv_splitter)\n        elif model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:\n            objective = lambda trial: self._objective_lightgbm(trial, X_train, y_train, cv_splitter)\n        else:\n            raise ValueError(f\"Unknown model: {model_name}\")\n        \n        # Optimize\n        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)\n        \n        self.studies[model_name] = study\n        \n        self.logger.info(f\"{model_name} optimization completed. Best score: {study.best_value:.4f}\")\n        \n        return study.best_params\n    \n    def train_optimized_model(self, model_name: str, best_params: Dict[str, Any], \n                            X_train: np.ndarray, y_train: np.ndarray) -> Any:\n        \"\"\"Train model with optimized hyperparameters\"\"\"\n        \n        if model_name == 'RandomForest':\n            model = RandomForestClassifier(**best_params)\n        elif model_name == 'ExtraTrees':\n            model = ExtraTreesClassifier(**best_params)\n        elif model_name == 'LogisticRegression':\n            model = LogisticRegression(**best_params)\n        elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:\n            model = xgb.XGBClassifier(**best_params)\n        elif model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:\n            model = lgb.LGBMClassifier(**best_params)\n        else:\n            raise ValueError(f\"Unknown model: {model_name}\")\n        \n        # Train model\n        model.fit(X_train, y_train)\n        \n        return model\n    \n    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, \n                      model_name: str) -> Dict[str, float]:\n        \"\"\"Evaluate model performance\"\"\"\n        \n        # Make predictions\n        y_pred = model.predict(X_test)\n        y_pred_proba = model.predict_proba(X_test)\n        \n        # Calculate metrics\n        accuracy = accuracy_score(y_test, y_pred)\n        f1_weighted = f1_score(y_test, y_pred, average='weighted')\n        f1_macro = f1_score(y_test, y_pred, average='macro')\n        f1_micro = f1_score(y_test, y_pred, average='micro')\n        \n        try:\n            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')\n        except:\n            roc_auc = 0.0\n        \n        # Calculate precision and recall\n        precision, recall, f1_scores, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')\n        \n        # Business-specific metrics\n        business_f1 = f1_weighted  # Use weighted F1 as business metric\n        \n        # Class-wise performance\n        class_f1_scores = f1_score(y_test, y_pred, average=None)\n        \n        results = {\n            'accuracy': float(accuracy),\n            'f1_weighted': float(f1_weighted),\n            'f1_macro': float(f1_macro),\n            'f1_micro': float(f1_micro),\n            'precision_weighted': float(precision),\n            'recall_weighted': float(recall),\n            'roc_auc': float(roc_auc),\n            'business_f1': float(business_f1),\n            'class_f1_scores': [float(score) for score in class_f1_scores]\n        }\n        \n        # Log detailed results\n        self.logger.info(f\"{model_name} Results:\")\n        self.logger.info(f\"  Accuracy: {accuracy:.4f}\")\n        self.logger.info(f\"  F1 (Weighted): {f1_weighted:.4f}\")\n        self.logger.info(f\"  F1 (Macro): {f1_macro:.4f}\")\n        self.logger.info(f\"  ROC-AUC: {roc_auc:.4f}\")\n        \n        return results\n    \n    def create_ensemble_model(self, models: Dict[str, Any]) -> VotingClassifier:\n        \"\"\"Create ensemble model from best performing models\"\"\"\n        \n        self.logger.info(\"Creating ensemble model...\")\n        \n        # Select top 3 models based on F1 score\n        model_scores = [(name, self.model_results[name]['f1_weighted']) \n                       for name in models.keys()]\n        model_scores.sort(key=lambda x: x[1], reverse=True)\n        \n        top_models = model_scores[:3]\n        \n        estimators = [(name, models[name]) for name, _ in top_models]\n        \n        ensemble = VotingClassifier(\n            estimators=estimators,\n            voting='soft',  # Use probability-based voting\n            n_jobs=-1\n        )\n        \n        self.logger.info(f\"Ensemble created with models: {[name for name, _ in top_models]}\")\n        \n        return ensemble\n    \n    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:\n        \"\"\"Train all models with optimization\"\"\"\n        \n        self.logger.info(\"Starting enhanced model training pipeline...\")\n        \n        # Split data\n        X_train, X_test, y_train, y_test = train_test_split(\n            X, y, test_size=self.test_size, random_state=self.random_state, \n            stratify=y\n        )\n        \n        # Check class distribution\n        unique, counts = np.unique(y_train, return_counts=True)\n        class_dist = dict(zip(unique, counts))\n        self.logger.info(f\"Original class distribution: {class_dist}\")\n        \n        # Apply class balancing\n        balancer = self._get_class_balancer(self.class_balance_method)\n        X_train_balanced, y_train_balanced = balancer.fit_resample(X_train, y_train)\n        \n        unique_bal, counts_bal = np.unique(y_train_balanced, return_counts=True)\n        class_dist_bal = dict(zip(unique_bal, counts_bal))\n        self.logger.info(f\"Balanced class distribution: {class_dist_bal}\")\n        \n        # Define models to train\n        models_to_train = ['RandomForest', 'ExtraTrees', 'LogisticRegression']\n        \n        if XGBOOST_AVAILABLE:\n            models_to_train.append('XGBoost')\n        \n        if LIGHTGBM_AVAILABLE:\n            models_to_train.append('LightGBM')\n        \n        trained_models = {}\n        \n        # Train each model\n        for model_name in models_to_train:\n            try:\n                self.logger.info(f\"Training {model_name}...\")\n                \n                if self.optimize_hyperparameters:\n                    # Optimize hyperparameters\n                    best_params = self.optimize_model(model_name, X_train_balanced, y_train_balanced)\n                    \n                    # Train with best parameters\n                    model = self.train_optimized_model(model_name, best_params, \n                                                     X_train_balanced, y_train_balanced)\n                else:\n                    # Use default parameters with some optimization\n                    if model_name == 'RandomForest':\n                        model = RandomForestClassifier(\n                            n_estimators=300, max_depth=20, min_samples_split=10,\n                            min_samples_leaf=4, class_weight='balanced',\n                            random_state=self.random_state, n_jobs=-1\n                        )\n                    elif model_name == 'ExtraTrees':\n                        model = ExtraTreesClassifier(\n                            n_estimators=300, max_depth=25, min_samples_split=8,\n                            min_samples_leaf=3, class_weight='balanced',\n                            random_state=self.random_state, n_jobs=-1\n                        )\n                    elif model_name == 'LogisticRegression':\n                        model = LogisticRegression(\n                            C=0.1, class_weight='balanced', max_iter=3000,\n                            random_state=self.random_state\n                        )\n                    elif model_name == 'XGBoost':\n                        model = xgb.XGBClassifier(\n                            n_estimators=300, max_depth=8, learning_rate=0.05,\n                            subsample=0.8, colsample_bytree=0.8,\n                            random_state=self.random_state, n_jobs=-1\n                        )\n                    elif model_name == 'LightGBM':\n                        model = lgb.LGBMClassifier(\n                            n_estimators=300, max_depth=10, learning_rate=0.05,\n                            subsample=0.8, colsample_bytree=0.8, class_weight='balanced',\n                            random_state=self.random_state, n_jobs=-1, verbose=-1\n                        )\n                    \n                    model.fit(X_train_balanced, y_train_balanced)\n                \n                # Evaluate model\n                results = self.evaluate_model(model, X_test, y_test, model_name)\n                \n                # Store results\n                trained_models[model_name] = model\n                self.model_results[model_name] = results\n                \n                # Save model\n                model_path = f'models/trained/{model_name.lower()}_optimized.joblib'\n                joblib.dump(model, model_path)\n                \n            except Exception as e:\n                self.logger.error(f\"Failed to train {model_name}: {e}\")\n                continue\n        \n        # Create ensemble model\n        if len(trained_models) >= 2:\n            try:\n                ensemble = self.create_ensemble_model(trained_models)\n                ensemble.fit(X_train_balanced, y_train_balanced)\n                \n                # Evaluate ensemble\n                ensemble_results = self.evaluate_model(ensemble, X_test, y_test, 'Ensemble')\n                \n                trained_models['Ensemble'] = ensemble\n                self.model_results['Ensemble'] = ensemble_results\n                \n                # Save ensemble\n                joblib.dump(ensemble, 'models/trained/ensemble_model.joblib')\n                \n            except Exception as e:\n                self.logger.error(f\"Failed to create ensemble: {e}\")\n        \n        self.trained_models = trained_models\n        \n        # Find best model\n        best_model_name = max(self.model_results.keys(), \n                            key=lambda k: self.model_results[k]['f1_weighted'])\n        \n        self.logger.info(f\"Training completed. Best model: {best_model_name}\")\n        self.logger.info(f\"Best F1 score: {self.model_results[best_model_name]['f1_weighted']:.4f}\")\n        \n        return {\n            'models': trained_models,\n            'results': self.model_results,\n            'best_model': best_model_name,\n            'X_test': X_test,\n            'y_test': y_test\n        }\n    \n    def get_optimization_summary(self) -> Dict[str, Any]:\n        \"\"\"Get summary of hyperparameter optimization\"\"\"\n        \n        summary = {}\n        \n        for model_name, study in self.studies.items():\n            summary[model_name] = {\n                'best_score': study.best_value,\n                'best_params': study.best_params,\n                'n_trials': len(study.trials),\n                'optimization_history': [trial.value for trial in study.trials if trial.value is not None]\n            }\n        \n        return summary\n    \n    def save_training_results(self, filepath: str):\n        \"\"\"Save complete training results\"\"\"\n        \n        results_data = {\n            'model_results': self.model_results,\n            'optimization_summary': self.get_optimization_summary(),\n            'training_config': {\n                'optimize_hyperparameters': self.optimize_hyperparameters,\n                'n_trials': self.n_trials,\n                'cv_folds': self.cv_folds,\n                'test_size': self.test_size,\n                'class_balance_method': self.class_balance_method,\n                'target_metric': self.target_metric\n            },\n            'timestamp': datetime.now().isoformat()\n        }\n        \n        joblib.dump(results_data, filepath)\n        self.logger.info(f\"Training results saved to {filepath}\")\n\nif __name__ == \"__main__\":\n    # Test the enhanced model trainer\n    import pandas as pd\n    \n    # Create sample data\n    np.random.seed(42)\n    X = np.random.randn(1000, 20)\n    y = np.random.randint(0, 3, 1000)\n    \n    # Test trainer\n    trainer = EnhancedModelTrainer(\n        optimize_hyperparameters=True,\n        n_trials=20,  # Reduced for testing\n        cv_folds=3,\n        class_balance_method='smote'\n    )\n    \n    results = trainer.train_models(X, y)\n    \n    print(f\"Trained models: {list(results['models'].keys())}\")\n    print(f\"Best model: {results['best_model']}\")\n    print(\"Enhanced model training test completed!\")