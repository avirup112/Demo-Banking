#!/usr/bin/env python3
"""
Simple and Fast Hyperparameter Optimizer
Uses basic grid search with predefined parameter sets for speed
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import joblib
import json
from datetime import datetime
import time

# ML Libraries
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
import xgboost as xgb
import lightgbm as lgb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleOptimizer:
    """Simple and fast hyperparameter optimizer"""
    
    def __init__(self, cv_folds: int = 3, random_state: int = 42, n_jobs: int = -1):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.best_params = {}
        self.best_scores = {}
        
        # Data placeholders
        self.X_train = None
        self.y_train = None
        
        # Results storage
        self.results_dir = Path("optimization_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("Simple optimizer initialized")
    
    def set_data(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None):
        """Set training data"""
        self.X_train = X_train
        self.y_train = y_train
        logger.info(f"Data set - Train: {X_train.shape}")
    
    def get_simple_param_sets(self) -> Dict[str, List[Dict]]:
        """Get simple parameter sets for fast optimization"""
        
        param_sets = {
            'random_forest': [
                {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'random_state': self.random_state},
                {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 2, 'random_state': self.random_state},
                {'n_estimators': 150, 'max_depth': None, 'min_samples_split': 10, 'class_weight': 'balanced', 'random_state': self.random_state}
            ],
            
            'xgboost': [
                {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': self.random_state},
                {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05, 'random_state': self.random_state},
                {'n_estimators': 150, 'max_depth': 8, 'learning_rate': 0.2, 'subsample': 0.8, 'random_state': self.random_state}
            ],
            
            'lightgbm': [
                {'n_estimators': 100, 'num_leaves': 31, 'learning_rate': 0.1, 'random_state': self.random_state},
                {'n_estimators': 200, 'num_leaves': 50, 'learning_rate': 0.05, 'random_state': self.random_state},
                {'n_estimators': 150, 'num_leaves': 20, 'learning_rate': 0.2, 'random_state': self.random_state}
            ],
            
            'logistic_regression': [
                {'C': 1.0, 'random_state': self.random_state},
                {'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear', 'random_state': self.random_state},
                {'C': 10.0, 'class_weight': 'balanced', 'random_state': self.random_state}
            ]
        }
        
        return param_sets
    
    def optimize_model(self, model_name: str) -> Dict[str, Any]:
        """Optimize a specific model using simple parameter sets"""
        
        start_time = time.time()
        
        param_sets = self.get_simple_param_sets()
        
        if model_name not in param_sets:
            raise ValueError(f"Model {model_name} not supported")
        
        logger.info(f"Optimizing {model_name} with {len(param_sets[model_name])} parameter sets...")
        
        best_score = 0
        best_params = None
        all_results = []
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for i, params in enumerate(param_sets[model_name]):
            try:
                # Create model
                if model_name == 'random_forest':
                    model = RandomForestClassifier(n_jobs=self.n_jobs, **params)
                elif model_name == 'xgboost':
                    model = xgb.XGBClassifier(n_jobs=self.n_jobs, verbosity=0, **params)
                elif model_name == 'lightgbm':
                    model = lgb.LGBMClassifier(n_jobs=self.n_jobs, verbosity=-1, **params)
                elif model_name == 'logistic_regression':
                    model = LogisticRegression(n_jobs=self.n_jobs, **params)
                
                # Cross-validate
                scores = cross_val_score(model, self.X_train, self.y_train, 
                                       cv=cv, scoring='f1_weighted', n_jobs=self.n_jobs)
                
                mean_score = scores.mean()
                std_score = scores.std()
                
                all_results.append({
                    'params': params,
                    'mean_score': mean_score,
                    'std_score': std_score
                })
                
                logger.info(f"  Set {i+1}: F1={mean_score:.4f} (+/-{std_score:.4f})")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"  Set {i+1} failed: {e}")
                all_results.append({
                    'params': params,
                    'error': str(e)
                })
        
        optimization_time = time.time() - start_time
        
        # Store results
        self.best_params[model_name] = best_params
        self.best_scores[model_name] = best_score
        
        logger.info(f"Best {model_name}: F1={best_score:.4f} in {optimization_time:.2f}s")
        
        return {
            'model_name': model_name,
            'best_params': best_params,
            'best_score': best_score,
            'optimization_time': optimization_time,
            'all_results': all_results
        }
    
    def optimize_all_models(self, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Optimize all supported models"""
        
        if models is None:
            models = ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression']
        
        results = {}
        total_start_time = time.time()
        
        for model_name in models:
            try:
                result = self.optimize_model(model_name)
                results[model_name] = result
            except Exception as e:
                logger.error(f"Optimization failed for {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        total_time = time.time() - total_start_time
        
        # Find best overall model
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_model = max(valid_results.keys(), 
                           key=lambda x: valid_results[x].get('best_score', 0))
            
            results['summary'] = {
                'best_model': best_model,
                'best_score': valid_results[best_model].get('best_score', 0),
                'total_optimization_time': total_time,
                'models_optimized': len(valid_results)
            }
            
            logger.info(f"Best overall: {best_model} with F1={valid_results[best_model].get('best_score', 0):.4f}")
            logger.info(f"Total time: {total_time:.2f} seconds")
        
        return results
    
    def create_optimized_model(self, model_name: str, params: Optional[Dict[str, Any]] = None):
        """Create model with optimized parameters"""
        
        if params is None:
            params = self.best_params.get(model_name, {})
        
        if model_name == 'random_forest':
            return RandomForestClassifier(n_jobs=self.n_jobs, **params)
        elif model_name == 'xgboost':
            return xgb.XGBClassifier(n_jobs=self.n_jobs, verbosity=0, **params)
        elif model_name == 'lightgbm':
            return lgb.LGBMClassifier(n_jobs=self.n_jobs, verbosity=-1, **params)
        elif model_name == 'logistic_regression':
            return LogisticRegression(n_jobs=self.n_jobs, **params)
        else:
            raise ValueError(f"Model {model_name} not supported")

def main():
    logger.info("Simple optimizer module loaded successfully")

if __name__ == "__main__":
    main()