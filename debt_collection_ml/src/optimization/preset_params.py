#!/usr/bin/env python3
"""
Pre-tuned Parameter Sets for Instant Model Creation
Based on common best practices for debt collection ML
"""

import logging
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

class PresetParameterOptimizer:
    """Instant model creation with pre-tuned parameters - no optimization time"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        logger.info("Preset parameter optimizer initialized - instant model creation")
    
    def get_optimized_models(self) -> dict:
        """Get pre-tuned models with good parameters for debt collection"""
        
        models = {
            'RandomForest_Optimized': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'XGBoost_Optimized': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'LightGBM_Optimized': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1
            ),
            
            'LogisticRegression_Optimized': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        logger.info(f"Created {len(models)} pre-optimized models instantly")
        return models
    
    def get_model_info(self) -> dict:
        """Get information about the pre-tuned parameters"""
        
        return {
            'RandomForest_Optimized': {
                'description': 'Balanced RF with good depth and feature sampling',
                'best_for': 'General purpose, handles mixed data types well',
                'speed': 'Medium'
            },
            'XGBoost_Optimized': {
                'description': 'Gradient boosting with regularization',
                'best_for': 'High performance, handles missing values',
                'speed': 'Medium-Fast'
            },
            'LightGBM_Optimized': {
                'description': 'Fast gradient boosting with leaf-wise growth',
                'best_for': 'Large datasets, very fast training',
                'speed': 'Fast'
            },
            'LogisticRegression_Optimized': {
                'description': 'Linear model with L2 regularization',
                'best_for': 'Interpretable results, fast predictions',
                'speed': 'Very Fast'
            }
        }

def main():
    """Example usage"""
    logger.info("Preset parameter optimizer module loaded")

if __name__ == "__main__":
    main()