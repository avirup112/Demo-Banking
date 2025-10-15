#!/usr/bin/env python3
"""
Fast Grid Search Alternative to Optuna
Pre-defined parameter grids with good defaults for quick optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

class FastGridSearchOptimizer:
    """Fast hyperparameter optimization using grid/random search with good defaults"""
    
    def __init__(self, cv_folds: int = 3, random_state: int = 42, n_jobs: int = -1):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_params = {}
        self.best_scores = {}
        
        logger.info("Fast Grid Search optimizer initialized")
    
    def get_fast_param_grids(self) -> Dict[str, Dict]:
        """Get pre-defined parameter grids optimized for speed and performance"""
        
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': ['balanced']
            },
            
            'xgboost': {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            
            'lightgbm': {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'verbosity': [-1]
            },
            
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'class_weight': ['balanced'],
                'max_iter': [1000]
            }
        }
        
        return param_grids
    
    def optimize_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Any, Dict, float]:
        """Fast optimization using GridSearchCV"""
        
        logger.info(f"Fast optimization for {model_name}...")
        
        # Get base model and param grid
        base_model, param_grid = self._get_model_and_grid(model_name)
        
        if base_model is None:
            logger.error(f"Model {model_name} not supported")
            return None, {}, 0.0
        
        # Use RandomizedSearchCV for speed (samples subset of grid)
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=8,  # Only try 8 combinations - very fast
            cv=self.cv_folds,
            scoring='f1_weighted',
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=0
        )
        
        # Fit
        search.fit(X_train, y_train)
        
        # Store results
        self.best_params[model_name] = search.best_params_
        self.best_scores[model_name] = search.best_score_
        
        logger.info(f"{model_name} optimization complete - Best F1: {search.best_score_:.4f}")
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def _get_model_and_grid(self, model_name: str) -> Tuple[Any, Dict]:
        """Get base model and parameter grid"""
        
        param_grids = self.get_fast_param_grids()
        
        if model_name == 'random_forest':
            return RandomForestClassifier(random_state=self.random_state), param_grids['random_forest']
        
        elif model_name == 'xgboost':
            return xgb.XGBClassifier(random_state=self.random_state), param_grids['xgboost']
        
        elif model_name == 'lightgbm':
            return lgb.LGBMClassifier(random_state=self.random_state), param_grids['lightgbm']
        
        elif model_name == 'logistic_regression':
            return LogisticRegression(random_state=self.random_state), param_grids['logistic_regression']
        
        else:
            return None, {}
    
    def optimize_all_models(self, models: List[str], X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Optimize all specified models quickly"""
        
        results = {}
        
        for model_name in models:
            try:
                best_model, best_params, best_score = self.optimize_model(model_name, X_train, y_train)
                
                results[model_name] = {
                    'model': best_model,
                    'best_params': best_params,
                    'best_score': best_score
                }
                
            except Exception as e:
                logger.error(f"Optimization failed for {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results

def main():
    """Example usage"""
    logger.info("Fast Grid Search optimizer module loaded")

if __name__ == "__main__":
    main()