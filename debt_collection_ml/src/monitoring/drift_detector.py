#!/usr/bin/env python3
"""
Data Drift Detection and Monitoring System
Uses statistical tests and distribution comparisons to detect drift
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings
from scipy import stats
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDriftDetector:
    """Comprehensive data drift detection system"""
    
    def __init__(self, reference_data: pd.DataFrame,feature_columns: List[str],target_column: str = None,drift_threshold: float = 0.05):
        """
        Initialize drift detector
        
        Args:
            reference_data: Reference dataset (training data)
            feature_columns: List of feature column names
            target_column: Target column name (optional)
            drift_threshold: P-value threshold for drift detection
        """
        self.reference_data = reference_data
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.drift_threshold = drift_threshold
        
        # Calculate reference statistics
        self.reference_stats = self._calculate_reference_stats()
        
        # Results storage
        self.results_dir = Path("monitoring_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Drift detector initialized with {len(feature_columns)} features")
    
    def _calculate_reference_stats(self) -> Dict[str, Any]:
        """Calculate reference statistics for drift detection"""
        
        reference_stats = {}
        
        for col in self.feature_columns:
            if col in self.reference_data.columns:
                data = self.reference_data[col].dropna()
                
                reference_stats[col] = {
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'median': data.median(),
                    'q25': data.quantile(0.25),
                    'q75': data.quantile(0.75),
                    'skewness': float(stats.skew(data)),
                    'kurtosis': float(stats.kurtosis(data)),
                    'distribution': data.values
                }
        
        # Target statistics if available
        if self.target_column and self.target_column in self.reference_data.columns:
            target_data = self.reference_data[self.target_column]
            reference_stats['target'] = {
                'distribution': target_data.value_counts().to_dict(),
                'proportions': target_data.value_counts(normalize=True).to_dict()
            }
        
        return reference_stats
    
    def detect_drift(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift in new data compared to reference
        
        Args:
            new_data: New dataset to check for drift
            
        Returns:
            Dictionary with drift detection results
        """
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'total_features': len(self.feature_columns),
            'features_with_drift': 0,
            'drift_detected': False,
            'feature_results': {},
            'overall_drift_score': 0.0,
            'alerts': []
        }
        
        drift_scores = []
        
        for col in self.feature_columns:
            if col not in new_data.columns:
                drift_results['alerts'].append(f"Missing feature: {col}")
                continue
            
            if col not in self.reference_stats:
                drift_results['alerts'].append(f"No reference stats for: {col}")
                continue
            
            # Perform drift tests
            feature_result = self._test_feature_drift(col, new_data[col])
            drift_results['feature_results'][col] = feature_result
            
            if feature_result['drift_detected']:
                drift_results['features_with_drift'] += 1
            
            drift_scores.append(feature_result['drift_score'])
        
        # Calculate overall drift
        if drift_scores:
            drift_results['overall_drift_score'] = np.mean(drift_scores)
            drift_results['drift_detected'] = drift_results['features_with_drift'] > 0
        
        # Check target drift if available
        if self.target_column and self.target_column in new_data.columns:
            target_result = self._test_target_drift(new_data[self.target_column])
            drift_results['target_drift'] = target_result
        
        # Generate alerts
        self._generate_alerts(drift_results)
        
        # Save results
        self._save_drift_results(drift_results)
        
        logger.info(f"Drift detection complete: {drift_results['features_with_drift']}/{drift_results['total_features']} features with drift")
        
        return drift_results
    
    def _test_feature_drift(self, feature_name: str, new_data: pd.Series) -> Dict[str, Any]:
        """Test for drift in a single feature"""
        
        reference_data = self.reference_stats[feature_name]['distribution']
        new_data_clean = new_data.dropna().values
        
        result = {
            'feature': feature_name,
            'drift_detected': False,
            'drift_score': 0.0,
            'tests': {},
            'statistics': {}
        }
        
        # Calculate new data statistics
        if len(new_data_clean) > 0:
            result['statistics'] = {
                'mean': np.mean(new_data_clean),
                'std': np.std(new_data_clean),
                'min': np.min(new_data_clean),
                'max': np.max(new_data_clean),
                'median': np.median(new_data_clean),
                'sample_size': len(new_data_clean)
            }
        
        # Kolmogorov-Smirnov test
        try:
            ks_stat, ks_pvalue = ks_2samp(reference_data, new_data_clean)
            result['tests']['ks_test'] = {
                'statistic': ks_stat,
                'p_value': ks_pvalue,
                'drift_detected': ks_pvalue < self.drift_threshold
            }
        except Exception as e:
            result['tests']['ks_test'] = {'error': str(e)}
        
        # Mann-Whitney U test (for distribution differences)
        try:
            mw_stat, mw_pvalue = stats.mannwhitneyu(reference_data, new_data_clean, alternative='two-sided')
            result['tests']['mannwhitney_test'] = {
                'statistic': mw_stat,
                'p_value': mw_pvalue,
                'drift_detected': mw_pvalue < self.drift_threshold
            }
        except Exception as e:
            result['tests']['mannwhitney_test'] = {'error': str(e)}
        
        # Population Stability Index (PSI)
        try:
            psi_score = self._calculate_psi(reference_data, new_data_clean)
            result['tests']['psi'] = {
                'score': psi_score,
                'drift_detected': psi_score > 0.2  # PSI > 0.2 indicates significant drift
            }
        except Exception as e:
            result['tests']['psi'] = {'error': str(e)}
        
        # Determine overall drift for this feature
        drift_indicators = []
        for test_name, test_result in result['tests'].items():
            if isinstance(test_result, dict) and 'drift_detected' in test_result:
                drift_indicators.append(test_result['drift_detected'])
        
        if drift_indicators:
            # Drift detected if majority of tests indicate drift
            result['drift_detected'] = sum(drift_indicators) >= len(drift_indicators) / 2
            result['drift_score'] = sum(drift_indicators) / len(drift_indicators)
        
        return result
    
    def _calculate_psi(self, reference: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        
        # Create bins based on reference data
        bin_edges = np.histogram_bin_edges(reference, bins=bins)
        
        # Calculate distributions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        new_counts, _ = np.histogram(new, bins=bin_edges)
        
        # Convert to proportions
        ref_props = ref_counts / len(reference)
        new_props = new_counts / len(new)
        
        # Avoid division by zero
        ref_props = np.where(ref_props == 0, 0.0001, ref_props)
        new_props = np.where(new_props == 0, 0.0001, new_props)
        
        # Calculate PSI
        psi = np.sum((new_props - ref_props) * np.log(new_props / ref_props))
        
        return psi
    
    def _test_target_drift(self, new_target: pd.Series) -> Dict[str, Any]:
        """Test for drift in target variable"""
        
        if 'target' not in self.reference_stats:
            return {'error': 'No reference target statistics available'}
        
        new_distribution = new_target.value_counts(normalize=True).to_dict()
        ref_distribution = self.reference_stats['target']['proportions']
        
        result = {
            'drift_detected': False,
            'new_distribution': new_distribution,
            'reference_distribution': ref_distribution,
            'chi2_test': {}
        }
        
        # Chi-square test for categorical target
        try:
            # Align categories
            all_categories = set(list(ref_distribution.keys()) + list(new_distribution.keys()))
            
            ref_counts = [ref_distribution.get(cat, 0) * len(self.reference_data) for cat in all_categories]
            new_counts = [new_distribution.get(cat, 0) * len(new_target) for cat in all_categories]
            
            chi2_stat, chi2_pvalue = stats.chisquare(new_counts, ref_counts)
            
            result['chi2_test'] = {
                'statistic': chi2_stat,
                'p_value': chi2_pvalue,
                'drift_detected': chi2_pvalue < self.drift_threshold
            }
            
            result['drift_detected'] = chi2_pvalue < self.drift_threshold
            
        except Exception as e:
            result['chi2_test'] = {'error': str(e)}
        
        return result
    
    def _generate_alerts(self, drift_results: Dict[str, Any]):
        """Generate alerts based on drift detection results"""
        
        if drift_results['drift_detected']:
            drift_results['alerts'].append(f"DRIFT ALERT: {drift_results['features_with_drift']} features showing drift")
        
        if drift_results['overall_drift_score'] > 0.5:
            drift_results['alerts'].append(f"HIGH DRIFT: Overall drift score {drift_results['overall_drift_score']:.3f}")
        
        # Feature-specific alerts
        for feature, result in drift_results['feature_results'].items():
            if result['drift_detected']:
                drift_results['alerts'].append(f"Feature '{feature}' showing significant drift")
    
    def _save_drift_results(self, drift_results: Dict[str, Any]):
        """Save drift detection results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.results_dir / f"drift_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(drift_results, f, indent=2, default=str)
        
        # Save summary
        summary = {
            'timestamp': drift_results['timestamp'],
            'drift_detected': drift_results['drift_detected'],
            'features_with_drift': drift_results['features_with_drift'],
            'total_features': drift_results['total_features'],
            'overall_drift_score': drift_results['overall_drift_score'],
            'alerts': drift_results['alerts']
        }
        
        summary_file = self.results_dir / f"drift_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Drift results saved to {results_file}")
    
    def create_drift_report(self, drift_results: Dict[str, Any]) -> str:
        """Create a comprehensive drift report"""
        
        report_lines = [
            "DATA DRIFT DETECTION REPORT",
            "=" * 50,
            f"Timestamp: {drift_results['timestamp']}",
            f"Total Features Analyzed: {drift_results['total_features']}",
            f"Features with Drift: {drift_results['features_with_drift']}",
            f"Overall Drift Score: {drift_results['overall_drift_score']:.3f}",
            f"Drift Detected: {'YES' if drift_results['drift_detected'] else 'NO'}",
            "",
            "FEATURE ANALYSIS:",
            "-" * 30
        ]
        
        for feature, result in drift_results['feature_results'].items():
            status = "DRIFT" if result['drift_detected'] else "OK"
            score = result['drift_score']
            report_lines.append(f"  {feature}: {status} (score: {score:.3f})")
        
        if drift_results.get('alerts'):
            report_lines.extend([
                "",
                "ALERTS:",
                "-" * 20
            ])
            for alert in drift_results['alerts']:
                report_lines.append(f"  ⚠️  {alert}")
        
        return "\n".join(report_lines)

def main():
    """Example usage of drift detector"""
    logger.info("Data drift detector module loaded successfully")

if __name__ == "__main__":
    main()