#!/usr/bin/env python3
"""
Advanced Hyperparameter Optimization with Optuna
Multi-objective optimization for debt collection ML models
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import joblib
import json
from datetime import datetime

# ML Libraries
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, make_scorer
import xgboost as xgb
import lightgbm as lgb

# Optuna integrations
from optuna.integration import XGBoostPruningCallback, LightGBMPruningCallback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebtCollectionOptimizer:
    """Advanced hyperparameter optimizer for debt collection models"""
    
    def __init__(self, 
                 study_name: str = "debt_collection_optimization",
                 storage: Optional[str] = None,
                 n_trials: int = 100,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize the optimizer
        
        Args:
            study_name: Name of the Optuna study
            storage: Database URL for study persistence
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.study_name = study_name
        self.storage = storage
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Initialize study
        self.study = None
        self.best_params = {}
        self.best_scores = {}
        
        # Data placeholders
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        
        # Results storage
        self.results_dir = Path("optimization_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Optimizer initialized: {study_name}")
    
    def set_data(self, X_train: np.ndarray, y_train: np.ndarray, 
                 X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Set training and validation data"""
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        logger.info(f"Data set - Train: {X_train.shape}, Val: {X_val.shape if X_val is not None else 'None'}")
    
    def create_study(self, direction: str = "maximize", 
                    sampler: Optional[optuna.samplers.BaseSampler] = None,
                    pruner: Optional[optuna.pruners.BasePruner] = None):
        """Create or load Optuna study"""
        
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
        
        if pruner is None:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )
        
        logger.info(f"Study created: {self.study_name}")
        return self.study
    
    def objective_random_forest(self, trial):
        """Objective function for Random Forest optimization"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
            'random_state': self.random_state
        }
        
        model = RandomForestClassifier(**params)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, self.X_train, self.y_train, 
                               cv=cv, scoring='f1_weighted', n_jobs=-1)
        
        return scores.mean()
    
    def objective_xgboost(self, trial):
        """Objective function for XGBoost optimization"""
        
        params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.2, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
            'random_state': self.random_state,
            'verbosity': 0
        }
        
        if params['booster'] == 'gbtree' or params['booster'] == 'dart':
            params['max_depth'] = trial.suggest_int('max_depth', 1, 9)
            params['eta'] = trial.suggest_float('eta', 1e-8, 1.0, log=True)
            params['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
            params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        
        if params['booster'] == 'dart':
            params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            params['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 1.0, log=True)
            params['one_drop'] = trial.suggest_categorical('one_drop', [0, 1])
        
        # Cross-validation with pruning
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(self.X_train, self.y_train)):
            X_fold_train, X_fold_val = self.X_train[train_idx], self.X_train[val_idx]
            y_fold_train, y_fold_val = self.y_train[train_idx], self.y_train[val_idx]
            
            dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
            dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dval, 'validation')],
                early_stopping_rounds=50,
                verbose_eval=False,
                callbacks=[XGBoostPruningCallback(trial, 'validation-mlogloss')]
            )
            
            preds = model.predict(dval)
            preds_class = np.argmax(preds, axis=1)
            score = f1_score(y_fold_val, preds_class, average='weighted')
            scores.append(score)
            
            # Report intermediate value for pruning
            trial.report(score, fold)
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return np.mean(scores)
    
    def objective_lightgbm(self, trial):
        """Objective function for LightGBM optimization"""
        
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(self.y_train)),
            'metric': 'multi_logloss',
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'random_state': self.random_state,
            'verbosity': -1
        }
        
        # Cross-validation with pruning
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(self.X_train, self.y_train)):
            X_fold_train, X_fold_val = self.X_train[train_idx], self.X_train[val_idx]
            y_fold_train, y_fold_val = self.y_train[train_idx], self.y_train[val_idx]
            
            train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
            val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(0),
                    LightGBMPruningCallback(trial, 'multi_logloss')
                ]
            )
            
            preds = model.predict(X_fold_val, num_iteration=model.best_iteration)
            preds_class = np.argmax(preds, axis=1)
            score = f1_score(y_fold_val, preds_class, average='weighted')
            scores.append(score)
            
            # Report intermediate value for pruning
            trial.report(score, fold)
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return np.mean(scores)
    
    def objective_logistic_regression(self, trial):
        """Objective function for Logistic Regression optimization"""
        
        params = {
            'C': trial.suggest_float('C', 1e-10, 1e10, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None]),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']),
            'max_iter': trial.suggest_int('max_iter', 100, 2000),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            'random_state': self.random_state
        }
        
        # Handle solver-penalty compatibility
        if params['penalty'] == 'elasticnet':
            if params['solver'] != 'saga':
                params['solver'] = 'saga'
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)
        elif params['penalty'] == 'l1':
            if params['solver'] not in ['liblinear', 'saga']:
                params['solver'] = 'liblinear'
        elif params['penalty'] is None:
            if params['solver'] not in ['lbfgs', 'newton-cg', 'sag', 'saga']:
                params['solver'] = 'lbfgs'
        
        model = LogisticRegression(**params)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, self.X_train, self.y_train, 
                               cv=cv, scoring='f1_weighted', n_jobs=-1)
        
        return scores.mean()
    
    def optimize_model(self, model_name: str, n_trials: Optional[int] = None) -> Dict[str, Any]:
        """Optimize a specific model"""
        
        if n_trials is None:
            n_trials = self.n_trials
        
        # Create study for this model
        study_name = f"{self.study_name}_{model_name}"
        self.create_study()
        
        # Select objective function
        objective_functions = {
            'random_forest': self.objective_random_forest,
            'xgboost': self.objective_xgboost,
            'lightgbm': self.objective_lightgbm,
            'logistic_regression': self.objective_logistic_regression
        }
        
        if model_name not in objective_functions:
            raise ValueError(f"Model {model_name} not supported. Available: {list(objective_functions.keys())}")
        
        objective = objective_functions[model_name]
        
        logger.info(f"Starting optimization for {model_name} with {n_trials} trials...")
        
        # Optimize
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Store results
        best_params = self.study.best_params
        best_score = self.study.best_value
        
        self.best_params[model_name] = best_params
        self.best_scores[model_name] = best_score
        
        logger.info(f"Optimization complete for {model_name}")
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")
        
        # Save results
        self.save_optimization_results(model_name)
        
        return {
            'model_name': model_name,
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(self.study.trials)
        }
    
    def optimize_all_models(self, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Optimize all supported models"""
        
        if models is None:
            models = ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression']
        
        results = {}
        
        for model_name in models:
            try:
                result = self.optimize_model(model_name)
                results[model_name] = result
            except Exception as e:
                logger.error(f"Optimization failed for {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        # Find best overall model
        best_model = max(results.keys(), 
                        key=lambda x: results[x].get('best_score', 0) if 'error' not in results[x] else 0)
        
        results['best_overall'] = {
            'model_name': best_model,
            'best_score': results[best_model].get('best_score', 0)
        }
        
        logger.info(f"Best overall model: {best_model} with score {results[best_model].get('best_score', 0):.4f}")
        
        return results
    
    def save_optimization_results(self, model_name: str):
        """Save optimization results to files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save study
        study_file = self.results_dir / f"{model_name}_study_{timestamp}.pkl"
        joblib.dump(self.study, study_file)
        
        # Save best parameters
        params_file = self.results_dir / f"{model_name}_best_params_{timestamp}.json"
        with open(params_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'best_params': self.best_params.get(model_name, {}),
                'best_score': self.best_scores.get(model_name, 0),
                'timestamp': timestamp,
                'n_trials': len(self.study.trials)
            }, f, indent=2)
        
        # Save trials dataframe
        trials_df = self.study.trials_dataframe()
        trials_file = self.results_dir / f"{model_name}_trials_{timestamp}.csv"
        trials_df.to_csv(trials_file, index=False)
        
        logger.info(f"Results saved for {model_name}")
    
    def get_optimization_summary(self) -> pd.DataFrame:
        """Get summary of all optimization results"""
        
        summary_data = []
        
        for model_name, params in self.best_params.items():
            summary_data.append({
                'model': model_name,
                'best_score': self.best_scores.get(model_name, 0),
                'n_params': len(params),
                'params': str(params)
            })
        
        return pd.DataFrame(summary_data).sort_values('best_score', ascending=False)
    
    def create_optimized_model(self, model_name: str, params: Optional[Dict[str, Any]] = None):
        """Create model with optimized parameters"""
        
        if params is None:
            params = self.best_params.get(model_name, {})
        
        if model_name == 'random_forest':
            return RandomForestClassifier(**params)
        elif model_name == 'xgboost':
            return xgb.XGBClassifier(**params)
        elif model_name == 'lightgbm':
            return lgb.LGBMClassifier(**params)
        elif model_name == 'logistic_regression':
            return LogisticRegression(**params)
        else:
            raise ValueError(f"Model {model_name} not supported")

def main():
    """Example usage of the optimizer"""
    
    # This would typically be called from your main pipeline
    logger.info("Optuna optimizer module loaded successfully")
    logger.info("Available models: random_forest, xgboost, lightgbm, logistic_regression")

if __name__ == "__main__":
    main()