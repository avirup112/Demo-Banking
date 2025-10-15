#!/usr/bin/env python3
"""
SHAP-based Model Explainability for Debt Collection ML
Provides individual and global explanations for model predictions
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebtCollectionExplainer:
    """SHAP-based explainer for debt collection models"""
    
    def __init__(self, model, model_type: str = 'tree'):
        """
        Initialize the explainer
        
        Args:
            model: Trained ML model
            model_type: Type of model ('tree', 'linear', 'deep', 'kernel')
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.feature_names = None
        self.class_names = ['Not Paid', 'Paid', 'Partially Paid']
        
        # Results storage
        self.explanations_dir = Path("explanations")
        self.explanations_dir.mkdir(exist_ok=True)
        
        logger.info(f"Explainer initialized for {model_type} model")
    
    def create_explainer(self, X_background: np.ndarray, feature_names: List[str] = None):
        """Create SHAP explainer based on model type"""
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_background.shape[1])]
        
        if self.model_type == 'tree':
            # For tree-based models (RandomForest, XGBoost, LightGBM)
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Created TreeExplainer")
            
        elif self.model_type == 'linear':
            # For linear models (LogisticRegression)
            self.explainer = shap.LinearExplainer(self.model, X_background)
            logger.info("Created LinearExplainer")
            
        elif self.model_type == 'kernel':
            # For any model (slower but universal)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, X_background)
            logger.info("Created KernelExplainer")
            
        else:
            # Default to Kernel explainer
            self.explainer = shap.KernelExplainer(self.model.predict_proba, X_background)
            logger.info("Created default KernelExplainer")
    
    def explain_instance(self, X_instance: np.ndarray, 
                        customer_id: str = None) -> Dict[str, Any]:
        """Explain a single prediction"""
        
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_instance.reshape(1, -1))
        
        # Handle different output formats
        if isinstance(shap_values, list):
            # Multi-class case - shap_values is a list of arrays
            shap_values_dict = {
                self.class_names[i]: shap_values[i][0] 
                for i in range(len(shap_values))
            }
        else:
            # Binary case or single array
            if len(shap_values.shape) == 3:
                # Multi-class in single array format
                shap_values_dict = {
                    self.class_names[i]: shap_values[0, :, i] 
                    for i in range(shap_values.shape[2])
                }
            else:
                # Binary classification
                shap_values_dict = {'prediction': shap_values[0]}
        
        # Get prediction
        prediction_proba = self.model.predict_proba(X_instance.reshape(1, -1))[0]
        prediction_class = self.model.predict(X_instance.reshape(1, -1))[0]
        
        # Create explanation dictionary
        explanation = {
            'customer_id': customer_id or 'unknown',
            'prediction_class': self.class_names[prediction_class],
            'prediction_probabilities': {
                self.class_names[i]: float(prediction_proba[i]) 
                for i in range(len(prediction_proba))
            },
            'shap_values': shap_values_dict,
            'feature_values': {
                self.feature_names[i]: float(X_instance[i]) 
                for i in range(len(X_instance))
            }
        }
        
        # Get top contributing features
        if isinstance(shap_values, list):
            # Use the predicted class SHAP values
            main_shap_values = shap_values[prediction_class][0]
        else:
            main_shap_values = shap_values[0]
        
        # Get feature importance for this prediction
        feature_importance = list(zip(self.feature_names, main_shap_values, X_instance))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        explanation['top_features'] = [
            {
                'feature': feat,
                'shap_value': float(shap_val),
                'feature_value': float(feat_val),
                'impact': 'positive' if shap_val > 0 else 'negative'
            }
            for feat, shap_val, feat_val in feature_importance[:10]
        ]
        
        return explanation
    
    def explain_batch(self, X_batch: np.ndarray, 
                     customer_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Explain multiple predictions"""
        
        explanations = []
        customer_ids = customer_ids or [f"customer_{i}" for i in range(len(X_batch))]
        
        for i, X_instance in enumerate(X_batch):
            explanation = self.explain_instance(X_instance, customer_ids[i])
            explanations.append(explanation)
        
        return explanations
    
    def global_feature_importance(self, X_sample: np.ndarray, 
                                 max_display: int = 20) -> Dict[str, Any]:
        """Calculate global feature importance using SHAP"""
        
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        logger.info("Calculating global feature importance...")
        
        # Get SHAP values for sample
        shap_values = self.explainer.shap_values(X_sample)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            # Multi-class case - average across classes
            shap_values_combined = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            if len(shap_values.shape) == 3:
                # Multi-class in single array
                shap_values_combined = np.mean(np.abs(shap_values), axis=(0, 2))
            else:
                # Binary case
                shap_values_combined = np.abs(shap_values)
        
        # Calculate mean absolute SHAP values
        mean_shap_values = np.mean(shap_values_combined, axis=0)
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap_values
        }).sort_values('importance', ascending=False)
        
        # Create summary
        global_importance = {
            'top_features': importance_df.head(max_display).to_dict('records'),
            'total_features': len(self.feature_names),
            'importance_scores': importance_df['importance'].tolist(),
            'feature_names': self.feature_names
        }
        
        return global_importance
    
    def create_summary_plot(self, X_sample: np.ndarray, 
                           save_path: str = None) -> str:
        """Create SHAP summary plot"""
        
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_sample)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        if isinstance(shap_values, list):
            # Multi-class case - show for each class
            for i, class_name in enumerate(self.class_names):
                plt.subplot(1, len(self.class_names), i + 1)
                shap.summary_plot(
                    shap_values[i], 
                    X_sample, 
                    feature_names=self.feature_names,
                    show=False,
                    max_display=15
                )
                plt.title(f'SHAP Summary - {class_name}')
        else:
            shap.summary_plot(
                shap_values, 
                X_sample, 
                feature_names=self.feature_names,
                show=False,
                max_display=20
            )
            plt.title('SHAP Summary Plot')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.explanations_dir / "shap_summary_plot.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary plot saved to {save_path}")
        return str(save_path)
    
    def create_waterfall_plot(self, X_instance: np.ndarray, 
                             customer_id: str = None,
                             save_path: str = None) -> str:
        """Create SHAP waterfall plot for individual prediction"""
        
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_instance.reshape(1, -1))
        
        # Get prediction
        prediction_class = self.model.predict(X_instance.reshape(1, -1))[0]
        
        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        
        if isinstance(shap_values, list):
            # Use the predicted class
            shap_values_to_plot = shap_values[prediction_class][0]
        else:
            if len(shap_values.shape) == 3:
                shap_values_to_plot = shap_values[0, :, prediction_class]
            else:
                shap_values_to_plot = shap_values[0]
        
        # Create explanation object for waterfall plot
        explanation = shap.Explanation(
            values=shap_values_to_plot,
            base_values=self.explainer.expected_value[prediction_class] if isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value,
            data=X_instance,
            feature_names=self.feature_names
        )
        
        shap.waterfall_plot(explanation, show=False, max_display=15)
        
        customer_label = customer_id or "Individual"
        predicted_class = self.class_names[prediction_class]
        plt.title(f'SHAP Waterfall Plot - {customer_label}\nPredicted: {predicted_class}')
        
        # Save plot
        if save_path is None:
            save_path = self.explanations_dir / f"waterfall_{customer_id or 'individual'}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Waterfall plot saved to {save_path}")
        return str(save_path)
    
    def create_force_plot(self, X_instance: np.ndarray, 
                         customer_id: str = None,
                         save_path: str = None) -> str:
        """Create SHAP force plot for individual prediction"""
        
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_instance.reshape(1, -1))
        
        # Get prediction
        prediction_class = self.model.predict(X_instance.reshape(1, -1))[0]
        
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[prediction_class][0]
            expected_value = self.explainer.expected_value[prediction_class]
        else:
            if len(shap_values.shape) == 3:
                shap_values_to_plot = shap_values[0, :, prediction_class]
                expected_value = self.explainer.expected_value[prediction_class]
            else:
                shap_values_to_plot = shap_values[0]
                expected_value = self.explainer.expected_value
        
        # Create force plot
        force_plot = shap.force_plot(
            expected_value,
            shap_values_to_plot,
            X_instance,
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        # Save plot
        if save_path is None:
            save_path = self.explanations_dir / f"force_{customer_id or 'individual'}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Force plot saved to {save_path}")
        return str(save_path)
    
    def generate_explanation_report(self, X_sample: np.ndarray,
                                   customer_ids: List[str] = None,
                                   report_name: str = "explanation_report") -> str:
        """Generate comprehensive explanation report"""
        
        logger.info("Generating comprehensive explanation report...")
        
        # Get global importance
        global_importance = self.global_feature_importance(X_sample)
        
        # Get sample explanations
        sample_size = min(10, len(X_sample))
        sample_indices = np.random.choice(len(X_sample), sample_size, replace=False)
        X_sample_subset = X_sample[sample_indices]
        
        if customer_ids:
            customer_ids_subset = [customer_ids[i] for i in sample_indices]
        else:
            customer_ids_subset = [f"customer_{i}" for i in sample_indices]
        
        sample_explanations = self.explain_batch(X_sample_subset, customer_ids_subset)
        
        # Create visualizations
        summary_plot_path = self.create_summary_plot(X_sample_subset)
        
        # Create report
        report = {
            'report_metadata': {
                'generated_at': pd.Timestamp.now().isoformat(),
                'model_type': self.model_type,
                'sample_size': len(X_sample),
                'features_count': len(self.feature_names)
            },
            'global_feature_importance': global_importance,
            'sample_explanations': sample_explanations,
            'visualizations': {
                'summary_plot': summary_plot_path
            },
            'insights': self._generate_insights(global_importance, sample_explanations)
        }
        
        # Save report
        report_path = self.explanations_dir / f"{report_name}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Explanation report saved to {report_path}")
        
        # Generate human-readable summary
        summary_path = self._generate_human_readable_summary(report, report_name)
        
        return str(report_path), str(summary_path)
    
    def _generate_insights(self, global_importance: Dict, 
                          sample_explanations: List[Dict]) -> Dict[str, Any]:
        """Generate insights from explanations"""
        
        insights = {
            'top_global_features': [
                feat['feature'] for feat in global_importance['top_features'][:5]
            ],
            'feature_impact_patterns': {},
            'prediction_confidence_analysis': {}
        }
        
        # Analyze feature impact patterns
        for explanation in sample_explanations:
            predicted_class = explanation['prediction_class']
            if predicted_class not in insights['feature_impact_patterns']:
                insights['feature_impact_patterns'][predicted_class] = {
                    'common_positive_features': [],
                    'common_negative_features': []
                }
            
            # Get top positive and negative features
            positive_features = [
                f['feature'] for f in explanation['top_features'][:3] 
                if f['impact'] == 'positive'
            ]
            negative_features = [
                f['feature'] for f in explanation['top_features'][:3] 
                if f['impact'] == 'negative'
            ]
            
            insights['feature_impact_patterns'][predicted_class]['common_positive_features'].extend(positive_features)
            insights['feature_impact_patterns'][predicted_class]['common_negative_features'].extend(negative_features)
        
        # Analyze prediction confidence
        confidences = []
        for explanation in sample_explanations:
            max_prob = max(explanation['prediction_probabilities'].values())
            confidences.append(max_prob)
        
        insights['prediction_confidence_analysis'] = {
            'average_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'high_confidence_threshold': 0.8,
            'high_confidence_predictions': sum(1 for c in confidences if c > 0.8)
        }
        
        return insights
    
    def generate_shap_insights(self, X_sample: np.ndarray, feature_names: List[str] = None, 
                              max_samples: int = 100) -> Dict[str, Any]:
        """Generate comprehensive SHAP insights and recommendations for feature engineering"""
        
        if X_sample.shape[0] > max_samples:
            # Sample for performance
            indices = np.random.choice(X_sample.shape[0], max_samples, replace=False)
            X_sample = X_sample[indices]
        
        insights = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'sample_size': X_sample.shape[0],
            'feature_insights': {},
            'recommendations': [],
            'feature_engineering_suggestions': [],
            'model_behavior_analysis': {},
            'actionable_insights': []
        }
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_sample)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                # Use the first class for insights (or could aggregate)
                shap_matrix = shap_values[0]
            else:
                shap_matrix = shap_values
            
            # Feature importance analysis
            feature_importance = np.abs(shap_matrix).mean(axis=0)
            
            if feature_names is None:
                feature_names = self.feature_names or [f"feature_{i}" for i in range(len(feature_importance))]
            
            # Analyze each feature
            for i, (importance, feature_name) in enumerate(zip(feature_importance, feature_names)):
                feature_shap = shap_matrix[:, i]
                
                insights['feature_insights'][feature_name] = {
                    'importance': float(importance),
                    'mean_shap': float(np.mean(feature_shap)),
                    'std_shap': float(np.std(feature_shap)),
                    'positive_impact_ratio': float(np.mean(feature_shap > 0)),
                    'impact_consistency': float(1 - np.std(feature_shap) / (np.abs(np.mean(feature_shap)) + 1e-8)),
                    'percentile_95': float(np.percentile(feature_shap, 95)),
                    'percentile_5': float(np.percentile(feature_shap, 5)),
                    'outlier_impact': float(np.mean(np.abs(feature_shap) > 2 * np.std(feature_shap)))
                }
            
            # Model behavior analysis
            insights['model_behavior_analysis'] = self._analyze_model_behavior(shap_matrix, feature_names)
            
            # Generate recommendations
            insights['recommendations'] = self._generate_feature_recommendations(insights['feature_insights'])
            insights['feature_engineering_suggestions'] = self._generate_engineering_suggestions(insights['feature_insights'])
            insights['actionable_insights'] = self._generate_actionable_insights(insights['feature_insights'])
            
        except Exception as e:
            insights['error'] = str(e)
            logger.error(f"Failed to generate SHAP insights: {e}")
        
        return insights
    
    def _analyze_model_behavior(self, shap_matrix: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze overall model behavior patterns"""
        
        from scipy import stats
        
        behavior = {
            'prediction_stability': float(np.mean(np.std(shap_matrix, axis=0))),
            'feature_interactions': {},
            'decision_patterns': {},
            'risk_factors': []
        }
        
        # Analyze feature interactions (simplified correlation of SHAP values)
        if shap_matrix.shape[1] > 1:
            shap_corr = np.corrcoef(shap_matrix.T)
            
            # Find strong interactions
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    corr = shap_corr[i, j]
                    if abs(corr) > 0.3:  # Threshold for significant interaction
                        behavior['feature_interactions'][f"{feature_names[i]}_x_{feature_names[j]}"] = float(corr)
        
        # Identify decision patterns
        total_shap = np.sum(shap_matrix, axis=1)
        behavior['decision_patterns'] = {
            'high_confidence_predictions': float(np.mean(np.abs(total_shap) > np.std(total_shap))),
            'prediction_distribution': {
                'mean': float(np.mean(total_shap)),
                'std': float(np.std(total_shap)),
                'skewness': float(stats.skew(total_shap)) if len(total_shap) > 2 else 0.0
            }
        }
        
        # Identify risk factors (features with high negative impact)
        feature_impacts = np.mean(shap_matrix, axis=0)
        for i, (impact, name) in enumerate(zip(feature_impacts, feature_names)):
            if impact < -0.1:  # Significant negative impact
                behavior['risk_factors'].append({
                    'feature': name,
                    'average_impact': float(impact),
                    'frequency': float(np.mean(shap_matrix[:, i] < 0))
                })
        
        return behavior
    
    def _generate_feature_recommendations(self, feature_insights: Dict[str, Any]) -> List[str]:
        """Generate feature engineering recommendations"""
        
        recommendations = []
        
        # Sort features by importance
        sorted_features = sorted(feature_insights.items(), 
                               key=lambda x: x[1]['importance'], reverse=True)
        
        # High importance, low consistency features
        inconsistent_important = [
            name for name, data in sorted_features[:10] 
            if data['impact_consistency'] < 0.3
        ]
        if inconsistent_important:
            recommendations.append(f"Consider feature transformations for inconsistent high-impact features: {', '.join(inconsistent_important[:3])}")
        
        # Low importance features
        low_importance = [name for name, data in feature_insights.items() if data['importance'] < 0.01]
        if len(low_importance) > 5:
            recommendations.append(f"Consider removing {len(low_importance)} low-impact features to reduce model complexity")
        
        # Features with extreme outlier impact
        outlier_features = [
            name for name, data in feature_insights.items() 
            if data['outlier_impact'] > 0.1 and data['importance'] > 0.05
        ]
        if outlier_features:
            recommendations.append(f"Apply outlier handling to features with extreme impacts: {', '.join(outlier_features[:3])}")
        
        return recommendations
    
    def _generate_engineering_suggestions(self, feature_insights: Dict[str, Any]) -> List[str]:
        """Generate specific feature engineering suggestions"""
        
        suggestions = []
        
        # Features with high positive impact ratio
        positive_features = [
            name for name, data in feature_insights.items() 
            if data['positive_impact_ratio'] > 0.8 and data['importance'] > 0.05
        ]
        if positive_features:
            suggestions.append(f"Create interaction terms with consistently positive features: {', '.join(positive_features[:2])}")
        
        # Features with balanced impact (could benefit from binning)
        balanced_features = [
            name for name, data in feature_insights.items() 
            if 0.3 < data['positive_impact_ratio'] < 0.7 and data['importance'] > 0.05
        ]
        if balanced_features:
            suggestions.append(f"Consider binning or polynomial features for balanced-impact features: {', '.join(balanced_features[:2])}")
        
        # High variance features
        high_variance = [
            name for name, data in feature_insights.items() 
            if data['std_shap'] > 0.2 and data['importance'] > 0.05
        ]
        if high_variance:
            suggestions.append(f"Apply normalization or scaling to high-variance features: {', '.join(high_variance[:3])}")
        
        return suggestions
    
    def _generate_actionable_insights(self, feature_insights: Dict[str, Any]) -> List[str]:
        """Generate actionable business insights from SHAP analysis"""
        
        insights = []
        
        # Sort features by importance
        sorted_features = sorted(feature_insights.items(), 
                               key=lambda x: x[1]['importance'], reverse=True)
        
        # Top important features
        top_features = sorted_features[:3]
        insights.append(f"Focus on top 3 drivers: {', '.join([f[0] for f in top_features])}")
        
        # Features with consistent positive impact
        positive_features = [name for name, data in feature_insights.items() 
                           if data['positive_impact_ratio'] > 0.7 and data['importance'] > 0.05]
        if positive_features:
            insights.append(f"Consistently beneficial factors: {', '.join(positive_features[:3])}")
        
        # Features with high variability (potential for improvement)
        variable_features = [name for name, data in feature_insights.items() 
                           if data['impact_consistency'] < 0.5 and data['importance'] > 0.05]
        if variable_features:
            insights.append(f"Investigate inconsistent features: {', '.join(variable_features[:2])}")
        
        # Risk factors
        risk_features = [name for name, data in feature_insights.items() 
                        if data['mean_shap'] < -0.1 and data['importance'] > 0.05]
        if risk_features:
            insights.append(f"Monitor risk factors: {', '.join(risk_features[:2])}")
        
        # Feature engineering opportunities
        low_impact_features = [name for name, data in feature_insights.items() 
                             if data['importance'] < 0.01]
        if len(low_impact_features) > 5:
            insights.append(f"Consider removing {len(low_impact_features)} low-impact features for model simplification")
        
        return insights

    def _generate_human_readable_summary(self, report: Dict, 
                                       report_name: str) -> str:
        """Generate human-readable summary"""
        
        summary_text = f"""
DEBT COLLECTION MODEL EXPLAINABILITY REPORT
==========================================

Generated: {report['report_metadata']['generated_at']}
Model Type: {report['report_metadata']['model_type']}
Sample Size: {report['report_metadata']['sample_size']}
Features: {report['report_metadata']['features_count']}

TOP FEATURES DRIVING REPAYMENT DECISIONS:
========================================

"""
        
        # Add top features
        for i, feature in enumerate(report['global_feature_importance']['top_features'][:10], 1):
            summary_text += f"{i:2d}. {feature['feature']:<30} (Importance: {feature['importance']:.4f})\n"
        
        summary_text += f"""

PREDICTION INSIGHTS:
===================

Average Prediction Confidence: {report['insights']['prediction_confidence_analysis']['average_confidence']:.2%}
High Confidence Predictions: {report['insights']['prediction_confidence_analysis']['high_confidence_predictions']} / {len(report['sample_explanations'])}

FEATURE IMPACT PATTERNS:
=======================

"""
        
        # Add feature impact patterns
        for class_name, patterns in report['insights']['feature_impact_patterns'].items():
            summary_text += f"\n{class_name.upper()}:\n"
            summary_text += f"  Common Positive Drivers: {', '.join(set(patterns['common_positive_features'][:5]))}\n"
            summary_text += f"  Common Negative Drivers: {', '.join(set(patterns['common_negative_features'][:5]))}\n"
        
        summary_text += f"""

BUSINESS RECOMMENDATIONS:
========================

1. Focus collection efforts on customers with high 'Partially Paid' probability
2. Monitor top features: {', '.join(report['insights']['top_global_features'][:3])}
3. Use feature insights to personalize collection strategies
4. Review low-confidence predictions for manual assessment

FILES GENERATED:
===============
- Detailed Report: {report_name}.json
- Summary Plot: {report['visualizations']['summary_plot']}
- Individual explanations available for sample customers

"""
        
        # Save summary
        summary_path = self.explanations_dir / f"{report_name}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        
        return str(summary_path)

def main():
    """Example usage of the explainer"""
    logger.info("SHAP explainer module loaded successfully")
    logger.info("Use DebtCollectionExplainer to explain model predictions")

if __name__ == "__main__":
    main()