#!/usr/bin/env python3
"""
Advanced Model Performance Validation and Selection
Implements automated model comparison, performance thresholds, and confidence scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings

# ML Libraries
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, 
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPerformanceValidator:
    """Comprehensive model performance validation and comparison"""
    
    def __init__(self, 
                 performance_threshold: float = 0.65,
                 cv_folds: int = 5,
                 random_state: int = 42,
                 use_time_series_cv: bool = True):
        """
        Initialize model validator
        
        Args:
            performance_threshold: Minimum F1 score threshold for model acceptance
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            use_time_series_cv: Whether to use time series cross-validation
        """
        self.performance_threshold = performance_threshold
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.use_time_series_cv = use_time_series_cv
        
        # Results storage
        self.validation_results = {}
        self.model_rankings = {}
        self.best_model = None
        self.best_model_name = None
        
        # Results directory
        self.results_dir = Path("validation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Model validator initialized with threshold: {performance_threshold}")
    
    def validate_single_model(self, 
                             model: Any, 
                             model_name: str,
                             X_train: np.ndarray, 
                             y_train: np.ndarray,
                             X_test: np.ndarray, 
                             y_test: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive validation of a single model
        
        Args:
            model: Trained model to validate
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Validating model: {model_name}")
        
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'threshold_met': False,
            'validation_passed': False
        }
        
        try:
            # Cross-validation scores
            cv_results = self._cross_validation_analysis(model, X_train, y_train)
            results.update(cv_results)
            
            # Test set performance
            test_results = self._test_set_analysis(model, X_test, y_test)
            results.update(test_results)
            
            # Confidence analysis
            confidence_results = self._confidence_analysis(model, X_test, y_test)
            results.update(confidence_results)
            
            # Calibration analysis
            calibration_results = self._calibration_analysis(model, X_test, y_test)
            results.update(calibration_results)
            
            # Performance threshold check
            f1_score_test = results['test_metrics']['f1_weighted']
            results['threshold_met'] = f1_score_test >= self.performance_threshold
            results['validation_passed'] = results['threshold_met']
            
            # Overall assessment
            results['performance_grade'] = self._calculate_performance_grade(results)
            
            logger.info(f"{model_name} validation complete - F1: {f1_score_test:.4f}, Threshold met: {results['threshold_met']}")
            
        except Exception as e:
            logger.error(f"Validation failed for {model_name}: {e}")
            results['error'] = str(e)
            results['validation_passed'] = False
        
        return results
    
    def _cross_validation_analysis(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform cross-validation analysis"""
        
        # Choose CV strategy
        if self.use_time_series_cv:
            cv = TimeSeriesSplit(n_splits=self.cv_folds)
        else:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Calculate multiple metrics
        scoring_metrics = ['f1_weighted', 'f1_macro', 'accuracy', 'precision_weighted', 'recall_weighted']
        cv_results = {}
        
        for metric in scoring_metrics:
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
                cv_results[f'cv_{metric}'] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores.tolist()
                }
            except Exception as e:
                logger.warning(f"Failed to calculate {metric}: {e}")
                cv_results[f'cv_{metric}'] = {'error': str(e)}
        
        return {'cv_results': cv_results}
    
    def _test_set_analysis(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Perform test set analysis"""
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted')
        }
        
        # ROC-AUC if model supports predict_proba
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)
                test_metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, 
                                                       multi_class='ovr', average='weighted')
            except Exception as e:
                logger.warning(f"ROC-AUC calculation failed: {e}")
                test_metrics['roc_auc'] = None
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        
        return {
            'test_metrics': test_metrics,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }
    
    def _confidence_analysis(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction confidence"""
        
        confidence_results = {}
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate confidence scores (max probability for each prediction)
            confidence_scores = np.max(y_pred_proba, axis=1)
            
            # Confidence statistics
            confidence_results['confidence_stats'] = {
                'mean': float(confidence_scores.mean()),
                'std': float(confidence_scores.std()),
                'min': float(confidence_scores.min()),
                'max': float(confidence_scores.max()),
                'median': float(np.median(confidence_scores)),
                'q25': float(np.percentile(confidence_scores, 25)),
                'q75': float(np.percentile(confidence_scores, 75))
            }
            
            # Confidence distribution by correctness
            y_pred = model.predict(X_test)
            correct_predictions = (y_pred == y_test)
            
            confidence_results['confidence_by_correctness'] = {
                'correct_mean': float(confidence_scores[correct_predictions].mean()) if np.any(correct_predictions) else 0.0,
                'incorrect_mean': float(confidence_scores[~correct_predictions].mean()) if np.any(~correct_predictions) else 0.0,
                'separation_score': float(confidence_scores[correct_predictions].mean() - confidence_scores[~correct_predictions].mean()) if np.any(correct_predictions) and np.any(~correct_predictions) else 0.0
            }
            
            # High confidence predictions analysis
            high_confidence_threshold = 0.8
            high_confidence_mask = confidence_scores >= high_confidence_threshold
            
            if np.any(high_confidence_mask):
                high_conf_accuracy = accuracy_score(y_test[high_confidence_mask], y_pred[high_confidence_mask])
                confidence_results['high_confidence_analysis'] = {
                    'threshold': high_confidence_threshold,
                    'percentage': float(np.mean(high_confidence_mask) * 100),
                    'accuracy': float(high_conf_accuracy),
                    'count': int(np.sum(high_confidence_mask))
                }
            else:
                confidence_results['high_confidence_analysis'] = {
                    'threshold': high_confidence_threshold,
                    'percentage': 0.0,
                    'accuracy': 0.0,
                    'count': 0
                }
        
        else:
            confidence_results['confidence_stats'] = {'error': 'Model does not support predict_proba'}
        
        return confidence_results
    
    def _calibration_analysis(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Analyze model calibration"""
        
        calibration_results = {}
        
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)
                
                # For multi-class, analyze calibration for each class
                n_classes = y_pred_proba.shape[1]
                calibration_results['calibration_analysis'] = {}
                
                for class_idx in range(n_classes):
                    # Binary calibration for this class
                    y_binary = (y_test == class_idx).astype(int)
                    prob_pos = y_pred_proba[:, class_idx]
                    
                    if len(np.unique(y_binary)) > 1:  # Only if both classes present
                        fraction_of_positives, mean_predicted_value = calibration_curve(
                            y_binary, prob_pos, n_bins=10, strategy='uniform'
                        )
                        
                        # Calculate calibration error (Brier score)
                        brier_score = np.mean((prob_pos - y_binary) ** 2)
                        
                        calibration_results['calibration_analysis'][f'class_{class_idx}'] = {
                            'brier_score': float(brier_score),
                            'fraction_of_positives': fraction_of_positives.tolist(),
                            'mean_predicted_value': mean_predicted_value.tolist()
                        }
                
                # Overall calibration score (average Brier score)
                if calibration_results['calibration_analysis']:
                    avg_brier = np.mean([
                        result['brier_score'] 
                        for result in calibration_results['calibration_analysis'].values()
                    ])
                    calibration_results['overall_calibration_score'] = float(avg_brier)
                
            except Exception as e:
                logger.warning(f"Calibration analysis failed: {e}")
                calibration_results['calibration_analysis'] = {'error': str(e)}
        
        return calibration_results
    
    def _calculate_performance_grade(self, results: Dict[str, Any]) -> str:
        """Calculate overall performance grade"""
        
        try:
            f1_score = results['test_metrics']['f1_weighted']
            
            if f1_score >= 0.85:
                return 'A'
            elif f1_score >= 0.75:
                return 'B'
            elif f1_score >= self.performance_threshold:
                return 'C'
            elif f1_score >= 0.50:
                return 'D'
            else:
                return 'F'
        except:
            return 'Unknown'
    
    def compare_models(self, 
                      models: Dict[str, Any], 
                      X_train: np.ndarray, 
                      y_train: np.ndarray,
                      X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, Any]:
        """
        Compare multiple models and select the best one
        
        Args:
            models: Dictionary of model_name -> trained_model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing comparison results and best model selection
        """
        logger.info(f"Comparing {len(models)} models...")
        
        comparison_results = {
            'models_evaluated': len(models),
            'threshold': self.performance_threshold,
            'validation_results': {},
            'model_rankings': {},
            'best_model': None,
            'models_above_threshold': [],
            'summary_statistics': {}
        }
        
        # Validate each model
        for model_name, model in models.items():
            validation_result = self.validate_single_model(
                model, model_name, X_train, y_train, X_test, y_test
            )
            comparison_results['validation_results'][model_name] = validation_result
            
            # Track models above threshold
            if validation_result.get('threshold_met', False):
                comparison_results['models_above_threshold'].append(model_name)
        
        # Rank models
        comparison_results['model_rankings'] = self._rank_models(comparison_results['validation_results'])
        
        # Select best model
        best_model_info = self._select_best_model(comparison_results['validation_results'], models)
        comparison_results['best_model'] = best_model_info
        
        # Calculate summary statistics
        comparison_results['summary_statistics'] = self._calculate_summary_statistics(
            comparison_results['validation_results']
        )
        
        # Store results
        self.validation_results = comparison_results['validation_results']
        self.model_rankings = comparison_results['model_rankings']
        self.best_model = best_model_info['model'] if best_model_info else None
        self.best_model_name = best_model_info['name'] if best_model_info else None
        
        logger.info(f"Model comparison complete. Best model: {self.best_model_name}")
        logger.info(f"Models above threshold: {len(comparison_results['models_above_threshold'])}")
        
        return comparison_results
    
    def _rank_models(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Rank models based on multiple criteria"""
        
        ranking_data = []
        
        for model_name, results in validation_results.items():
            if 'error' not in results and 'test_metrics' in results:
                metrics = results['test_metrics']
                
                # Calculate composite score
                f1_score = metrics.get('f1_weighted', 0)
                accuracy = metrics.get('accuracy', 0)
                roc_auc = metrics.get('roc_auc', 0) or 0
                
                # Weighted composite score
                composite_score = (0.5 * f1_score + 0.3 * accuracy + 0.2 * roc_auc)
                
                ranking_data.append({
                    'model_name': model_name,
                    'f1_weighted': f1_score,
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'composite_score': composite_score,
                    'threshold_met': results.get('threshold_met', False),
                    'performance_grade': results.get('performance_grade', 'Unknown')
                })
        
        # Sort by composite score
        ranking_data.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Create rankings
        rankings = {
            'by_composite_score': ranking_data,
            'by_f1_score': sorted(ranking_data, key=lambda x: x['f1_weighted'], reverse=True),
            'by_accuracy': sorted(ranking_data, key=lambda x: x['accuracy'], reverse=True),
            'above_threshold': [item for item in ranking_data if item['threshold_met']]
        }
        
        return rankings
    
    def _select_best_model(self, validation_results: Dict[str, Any], models: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select the best model based on validation results"""
        
        best_model_info = None
        best_score = -1
        
        for model_name, results in validation_results.items():
            if 'error' not in results and 'test_metrics' in results:
                # Primary criterion: F1 score
                f1_score = results['test_metrics'].get('f1_weighted', 0)
                
                # Secondary criteria for tie-breaking
                accuracy = results['test_metrics'].get('accuracy', 0)
                roc_auc = results['test_metrics'].get('roc_auc', 0) or 0
                
                # Composite score for selection
                composite_score = f1_score + 0.1 * accuracy + 0.05 * roc_auc
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_model_info = {
                        'name': model_name,
                        'model': models.get(model_name),
                        'f1_score': f1_score,
                        'accuracy': accuracy,
                        'roc_auc': roc_auc,
                        'composite_score': composite_score,
                        'threshold_met': results.get('threshold_met', False),
                        'performance_grade': results.get('performance_grade', 'Unknown')
                    }
        
        return best_model_info
    
    def _calculate_summary_statistics(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics across all models"""
        
        f1_scores = []
        accuracies = []
        threshold_met_count = 0
        
        for results in validation_results.values():
            if 'error' not in results and 'test_metrics' in results:
                f1_scores.append(results['test_metrics'].get('f1_weighted', 0))
                accuracies.append(results['test_metrics'].get('accuracy', 0))
                if results.get('threshold_met', False):
                    threshold_met_count += 1
        
        summary = {
            'total_models': len(validation_results),
            'models_above_threshold': threshold_met_count,
            'threshold_success_rate': threshold_met_count / len(validation_results) if validation_results else 0
        }
        
        if f1_scores:
            summary['f1_statistics'] = {
                'mean': float(np.mean(f1_scores)),
                'std': float(np.std(f1_scores)),
                'min': float(np.min(f1_scores)),
                'max': float(np.max(f1_scores)),
                'median': float(np.median(f1_scores))
            }
        
        if accuracies:
            summary['accuracy_statistics'] = {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies)),
                'median': float(np.median(accuracies))
            }
        
        return summary
    
    def generate_confidence_scores(self, model: Any, X: np.ndarray) -> np.ndarray:
        """
        Generate confidence scores for predictions
        
        Args:
            model: Trained model
            X: Features for prediction
            
        Returns:
            Array of confidence scores
        """
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)
            # Return max probability as confidence score
            return np.max(y_pred_proba, axis=1)
        else:
            # For models without predict_proba, return uniform confidence
            logger.warning("Model does not support predict_proba, returning uniform confidence")
            return np.ones(len(X)) * 0.5
    
    def save_validation_results(self, results: Dict[str, Any], filename: str = None):
        """Save validation results to file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_validation_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Validation results saved to {filepath}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return float(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        elif hasattr(obj, '__dict__'):
            # Skip complex objects like models
            return str(type(obj).__name__)
        else:
            return obj
    
    def create_validation_report(self, results: Dict[str, Any]) -> str:
        """Create a comprehensive validation report"""
        
        report_lines = [
            "MODEL VALIDATION AND SELECTION REPORT",
            "=" * 50,
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Performance Threshold: {self.performance_threshold}",
            f"Cross-Validation Folds: {self.cv_folds}",
            f"Time Series CV: {self.use_time_series_cv}",
            "",
            "SUMMARY:",
            f"- Total Models Evaluated: {results.get('models_evaluated', 0)}",
            f"- Models Above Threshold: {len(results.get('models_above_threshold', []))}",
            f"- Success Rate: {results.get('summary_statistics', {}).get('threshold_success_rate', 0):.2%}",
            ""
        ]
        
        # Best model information
        best_model = results.get('best_model')
        if best_model:
            report_lines.extend([
                "BEST MODEL:",
                f"- Name: {best_model['name']}",
                f"- F1 Score: {best_model['f1_score']:.4f}",
                f"- Accuracy: {best_model['accuracy']:.4f}",
                f"- ROC-AUC: {best_model.get('roc_auc', 'N/A')}",
                f"- Performance Grade: {best_model['performance_grade']}",
                f"- Threshold Met: {'Yes' if best_model['threshold_met'] else 'No'}",
                ""
            ])
        
        # Model rankings
        rankings = results.get('model_rankings', {})
        if 'by_composite_score' in rankings:
            report_lines.extend([
                "MODEL RANKINGS (by Composite Score):",
                "-" * 40
            ])
            
            for i, model_info in enumerate(rankings['by_composite_score'][:10], 1):
                report_lines.append(
                    f"{i:2d}. {model_info['model_name']:<25} "
                    f"F1: {model_info['f1_weighted']:.4f} "
                    f"Acc: {model_info['accuracy']:.4f} "
                    f"Grade: {model_info['performance_grade']}"
                )
            
            report_lines.append("")
        
        # Summary statistics
        summary_stats = results.get('summary_statistics', {})
        if 'f1_statistics' in summary_stats:
            f1_stats = summary_stats['f1_statistics']
            report_lines.extend([
                "F1 SCORE STATISTICS:",
                f"- Mean: {f1_stats['mean']:.4f} ± {f1_stats['std']:.4f}",
                f"- Range: {f1_stats['min']:.4f} - {f1_stats['max']:.4f}",
                f"- Median: {f1_stats['median']:.4f}",
                ""
            ])
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
            "- " + ("Deploy best model to production" if best_model and best_model['threshold_met'] 
                   else "Continue model development - threshold not met"),
            "- Set up monitoring for model performance drift",
            "- Implement confidence-based prediction filtering",
            "- Consider ensemble methods for improved performance",
            "",
            "=" * 50
        ])
        
        return "\n".join(report_lines)

def main():
    """Example usage of model validator"""
    
    logger.info("Model performance validator module loaded successfully")
    logger.info("Features: automated comparison, threshold validation, confidence scoring")

if __name__ == "__main__":
    main()