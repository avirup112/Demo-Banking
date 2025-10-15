#!/usr/bin/env python3
"""
A/B Testing Framework for Model Comparison
Statistical testing and experiment management for ML models
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import uuid
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ABTestFramework:
    """Comprehensive A/B testing framework for ML models"""
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 min_sample_size: int = 100,
                 max_duration_days: int = 30):
        """
        Initialize A/B testing framework
        
        Args:
            confidence_level: Statistical confidence level (default: 0.95)
            min_sample_size: Minimum sample size per variant
            max_duration_days: Maximum test duration in days
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.min_sample_size = min_sample_size
        self.max_duration_days = max_duration_days
        
        # Test storage
        self.experiments_dir = Path("ab_experiments")
        self.experiments_dir.mkdir(exist_ok=True)
        
        # Active experiments
        self.active_experiments = {}
        
        logger.info(f"A/B testing framework initialized (confidence: {confidence_level})")
    
    def create_experiment(self, 
                         experiment_name: str,
                         model_a: Any,
                         model_b: Any,
                         model_a_name: str = "Control",
                         model_b_name: str = "Treatment",
                         traffic_split: float = 0.5,
                         primary_metric: str = "f1_score",
                         secondary_metrics: List[str] = None) -> str:
        """
        Create a new A/B test experiment
        
        Args:
            experiment_name: Name of the experiment
            model_a: Control model (baseline)
            model_b: Treatment model (new model)
            model_a_name: Name for control model
            model_b_name: Name for treatment model
            traffic_split: Fraction of traffic for treatment (0.0-1.0)
            primary_metric: Primary metric for comparison
            secondary_metrics: Additional metrics to track
            
        Returns:
            Experiment ID
        """
        
        experiment_id = str(uuid.uuid4())[:8]
        
        if secondary_metrics is None:
            secondary_metrics = ["accuracy", "precision", "recall"]
        
        experiment = {
            'id': experiment_id,
            'name': experiment_name,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'models': {
                'control': {
                    'name': model_a_name,
                    'model': model_a,
                    'traffic_allocation': 1 - traffic_split
                },
                'treatment': {
                    'name': model_b_name,
                    'model': model_b,
                    'traffic_allocation': traffic_split
                }
            },
            'metrics': {
                'primary': primary_metric,
                'secondary': secondary_metrics
            },
            'results': {
                'control': {'predictions': [], 'actuals': [], 'metrics': {}},
                'treatment': {'predictions': [], 'actuals': [], 'metrics': {}}
            },
            'statistics': {},
            'conclusion': None
        }
        
        self.active_experiments[experiment_id] = experiment
        self._save_experiment(experiment)
        
        logger.info(f"Created experiment '{experiment_name}' (ID: {experiment_id})")
        logger.info(f"Traffic split: {model_a_name} {(1-traffic_split)*100:.1f}% | {model_b_name} {traffic_split*100:.1f}%")
        
        return experiment_id
    
    def assign_variant(self, experiment_id: str, user_id: str = None) -> str:
        """
        Assign a user to a variant (control or treatment)
        
        Args:
            experiment_id: ID of the experiment
            user_id: User identifier (optional, uses random if None)
            
        Returns:
            Variant assignment ('control' or 'treatment')
        """
        
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        traffic_split = experiment['models']['treatment']['traffic_allocation']
        
        # Deterministic assignment based on user_id if provided
        if user_id:
            # Use hash of user_id for consistent assignment
            hash_value = hash(user_id) % 100
            variant = 'treatment' if hash_value < (traffic_split * 100) else 'control'
        else:
            # Random assignment
            variant = 'treatment' if np.random.random() < traffic_split else 'control'
        
        return variant
    
    def record_prediction(self, 
                         experiment_id: str,
                         variant: str,
                         prediction: Union[int, float, np.ndarray],
                         actual: Union[int, float, np.ndarray] = None,
                         user_id: str = None) -> None:
        """
        Record a prediction result for the experiment
        
        Args:
            experiment_id: ID of the experiment
            variant: Variant that made the prediction ('control' or 'treatment')
            prediction: Model prediction
            actual: Actual outcome (if available)
            user_id: User identifier (optional)
        """
        
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        
        # Store prediction
        experiment['results'][variant]['predictions'].append({
            'value': prediction,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id
        })
        
        # Store actual if provided
        if actual is not None:
            experiment['results'][variant]['actuals'].append({
                'value': actual,
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id
            })
        
        # Update experiment
        self.active_experiments[experiment_id] = experiment
        self._save_experiment(experiment)
    
    def get_predictions(self, experiment_id: str, variant: str) -> Tuple[List, List]:
        """Get predictions and actuals for a variant"""
        
        experiment = self.active_experiments[experiment_id]
        results = experiment['results'][variant]
        
        predictions = [p['value'] for p in results['predictions']]
        actuals = [a['value'] for a in results['actuals']]
        
        return predictions, actuals
    
    def calculate_metrics(self, experiment_id: str) -> Dict[str, Any]:
        """Calculate metrics for both variants"""
        
        experiment = self.active_experiments[experiment_id]
        metrics_config = experiment['metrics']
        
        results = {}
        
        for variant in ['control', 'treatment']:
            predictions, actuals = self.get_predictions(experiment_id, variant)
            
            if len(predictions) == 0 or len(actuals) == 0:
                results[variant] = {'error': 'No data available'}
                continue
            
            # Ensure equal length
            min_len = min(len(predictions), len(actuals))
            predictions = predictions[:min_len]
            actuals = actuals[:min_len]
            
            variant_metrics = {}
            
            try:
                # Calculate primary and secondary metrics
                all_metrics = [metrics_config['primary']] + metrics_config['secondary']
                
                for metric_name in all_metrics:
                    if metric_name == 'accuracy':
                        variant_metrics[metric_name] = accuracy_score(actuals, predictions)
                    elif metric_name == 'f1_score':
                        variant_metrics[metric_name] = f1_score(actuals, predictions, average='weighted')
                    elif metric_name == 'precision':
                        variant_metrics[metric_name] = precision_score(actuals, predictions, average='weighted')
                    elif metric_name == 'recall':
                        variant_metrics[metric_name] = recall_score(actuals, predictions, average='weighted')
                    elif metric_name == 'roc_auc':
                        try:
                            variant_metrics[metric_name] = roc_auc_score(actuals, predictions, multi_class='ovr')
                        except:
                            variant_metrics[metric_name] = 0.5  # Default for binary case
                
                variant_metrics['sample_size'] = len(predictions)
                
            except Exception as e:
                variant_metrics = {'error': str(e)}
            
            results[variant] = variant_metrics
        
        # Update experiment with calculated metrics
        experiment['results']['control']['metrics'] = results.get('control', {})
        experiment['results']['treatment']['metrics'] = results.get('treatment', {})
        
        return results
    
    def run_statistical_test(self, experiment_id: str) -> Dict[str, Any]:
        """
        Run statistical significance test
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Statistical test results
        """
        
        experiment = self.active_experiments[experiment_id]
        primary_metric = experiment['metrics']['primary']
        
        # Get metric values for both variants
        control_predictions, control_actuals = self.get_predictions(experiment_id, 'control')
        treatment_predictions, treatment_actuals = self.get_predictions(experiment_id, 'treatment')
        
        if len(control_actuals) < self.min_sample_size or len(treatment_actuals) < self.min_sample_size:
            return {
                'significant': False,
                'reason': f'Insufficient sample size (min: {self.min_sample_size})',
                'control_size': len(control_actuals),
                'treatment_size': len(treatment_actuals)
            }
        
        # Calculate metrics for comparison
        metrics = self.calculate_metrics(experiment_id)
        
        if 'error' in metrics['control'] or 'error' in metrics['treatment']:
            return {
                'significant': False,
                'reason': 'Error calculating metrics',
                'errors': {
                    'control': metrics['control'].get('error'),
                    'treatment': metrics['treatment'].get('error')
                }
            }
        
        control_metric = metrics['control'][primary_metric]
        treatment_metric = metrics['treatment'][primary_metric]
        
        # Perform t-test (assuming normal distribution of metric)
        # For more robust testing, could use bootstrap or other methods
        
        # Create metric arrays (simplified approach)
        control_values = [control_metric] * len(control_actuals)
        treatment_values = [treatment_metric] * len(treatment_actuals)
        
        try:
            t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(control_values) - 1) * np.var(control_values) + 
                                 (len(treatment_values) - 1) * np.var(treatment_values)) / 
                                (len(control_values) + len(treatment_values) - 2))
            
            cohens_d = (treatment_metric - control_metric) / pooled_std if pooled_std > 0 else 0
            
            # Determine significance
            is_significant = p_value < self.alpha
            
            # Calculate confidence interval for difference
            diff = treatment_metric - control_metric
            se_diff = pooled_std * np.sqrt(1/len(control_values) + 1/len(treatment_values))
            t_critical = stats.t.ppf(1 - self.alpha/2, len(control_values) + len(treatment_values) - 2)
            ci_lower = diff - t_critical * se_diff
            ci_upper = diff + t_critical * se_diff
            
            results = {
                'significant': is_significant,
                'p_value': p_value,
                'alpha': self.alpha,
                'confidence_level': self.confidence_level,
                'effect_size': cohens_d,
                'difference': diff,
                'confidence_interval': [ci_lower, ci_upper],
                'control_metric': control_metric,
                'treatment_metric': treatment_metric,
                'control_sample_size': len(control_actuals),
                'treatment_sample_size': len(treatment_actuals),
                'test_statistic': t_stat,
                'winner': 'treatment' if treatment_metric > control_metric and is_significant else 'control'
            }
            
        except Exception as e:
            results = {
                'significant': False,
                'reason': f'Statistical test failed: {str(e)}'
            }
        
        # Store results in experiment
        experiment['statistics'] = results
        self._save_experiment(experiment)
        
        return results
    
    def conclude_experiment(self, experiment_id: str, force: bool = False) -> Dict[str, Any]:
        """
        Conclude an experiment and determine the winner
        
        Args:
            experiment_id: ID of the experiment
            force: Force conclusion even if criteria not met
            
        Returns:
            Experiment conclusion
        """
        
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        
        # Run statistical test
        stats_results = self.run_statistical_test(experiment_id)
        
        # Check if experiment can be concluded
        can_conclude = (
            stats_results.get('significant', False) or
            force or
            self._is_experiment_expired(experiment)
        )
        
        if not can_conclude:
            return {
                'concluded': False,
                'reason': 'Experiment criteria not met for conclusion',
                'statistics': stats_results
            }
        
        # Determine conclusion
        conclusion = {
            'concluded': True,
            'concluded_at': datetime.now().isoformat(),
            'winner': stats_results.get('winner', 'inconclusive'),
            'significant': stats_results.get('significant', False),
            'statistics': stats_results,
            'recommendation': self._generate_recommendation(stats_results)
        }
        
        # Update experiment
        experiment['status'] = 'concluded'
        experiment['conclusion'] = conclusion
        self._save_experiment(experiment)
        
        logger.info(f"Experiment {experiment_id} concluded: Winner = {conclusion['winner']}")
        
        return conclusion
    
    def _is_experiment_expired(self, experiment: Dict[str, Any]) -> bool:
        """Check if experiment has exceeded maximum duration"""
        
        created_at = datetime.fromisoformat(experiment['created_at'])
        max_duration = timedelta(days=self.max_duration_days)
        
        return datetime.now() - created_at > max_duration
    
    def _generate_recommendation(self, stats_results: Dict[str, Any]) -> str:
        """Generate recommendation based on statistical results"""
        
        if not stats_results.get('significant', False):
            return "No significant difference detected. Consider running longer or with larger sample size."
        
        winner = stats_results.get('winner', 'inconclusive')
        effect_size = abs(stats_results.get('effect_size', 0))
        
        if winner == 'treatment':
            if effect_size > 0.8:
                return "Strong evidence for treatment model. Recommend full rollout."
            elif effect_size > 0.5:
                return "Moderate evidence for treatment model. Consider gradual rollout."
            else:
                return "Weak evidence for treatment model. Consider additional testing."
        elif winner == 'control':
            return "Control model performs better. Do not deploy treatment model."
        else:
            return "Inconclusive results. Consider extending test or redesigning experiment."
    
    def _save_experiment(self, experiment: Dict[str, Any]):
        """Save experiment to file"""
        
        # Create a serializable copy
        experiment_copy = experiment.copy()
        
        # Remove non-serializable model objects
        if 'models' in experiment_copy:
            for variant in experiment_copy['models']:
                if 'model' in experiment_copy['models'][variant]:
                    experiment_copy['models'][variant]['model'] = str(type(experiment_copy['models'][variant]['model']))
        
        experiment_file = self.experiments_dir / f"experiment_{experiment['id']}.json"
        with open(experiment_file, 'w') as f:
            json.dump(experiment_copy, f, indent=2, default=str)
    
    def get_experiment_summary(self, experiment_id: str) -> str:
        """Generate a summary report for an experiment"""
        
        experiment = self.active_experiments[experiment_id]
        metrics = self.calculate_metrics(experiment_id)
        
        lines = [
            f"A/B TEST EXPERIMENT SUMMARY",
            f"=" * 50,
            f"Experiment: {experiment['name']} (ID: {experiment_id})",
            f"Status: {experiment['status']}",
            f"Created: {experiment['created_at']}",
            f"Primary Metric: {experiment['metrics']['primary']}",
            "",
            f"RESULTS:",
            f"-" * 20
        ]
        
        for variant in ['control', 'treatment']:
            variant_name = experiment['models'][variant]['name']
            variant_metrics = metrics.get(variant, {})
            
            if 'error' in variant_metrics:
                lines.append(f"{variant_name}: Error - {variant_metrics['error']}")
            else:
                primary_metric = experiment['metrics']['primary']
                primary_value = variant_metrics.get(primary_metric, 'N/A')
                sample_size = variant_metrics.get('sample_size', 0)
                
                lines.append(f"{variant_name}: {primary_metric}={primary_value:.4f} (n={sample_size})")
        
        # Add statistical results if available
        if 'statistics' in experiment and experiment['statistics']:
            stats = experiment['statistics']
            lines.extend([
                "",
                f"STATISTICAL ANALYSIS:",
                f"-" * 30,
                f"Significant: {stats.get('significant', 'N/A')}",
                f"P-value: {stats.get('p_value', 'N/A')}",
                f"Effect Size: {stats.get('effect_size', 'N/A')}",
                f"Winner: {stats.get('winner', 'N/A')}"
            ])
        
        # Add conclusion if available
        if experiment.get('conclusion'):
            conclusion = experiment['conclusion']
            lines.extend([
                "",
                f"CONCLUSION:",
                f"-" * 20,
                f"Winner: {conclusion['winner']}",
                f"Recommendation: {conclusion['recommendation']}"
            ])
        
        return "\n".join(lines)

def main():
    """Example usage of A/B testing framework"""
    logger.info("A/B testing framework module loaded successfully")

if __name__ == "__main__":
    main()