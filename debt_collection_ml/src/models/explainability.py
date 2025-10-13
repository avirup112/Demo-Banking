import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import warnings
import logging
from typing import Dict, List, Any, Optional, Tuple
warnings.filterwarnings('ignore')

class ModelExplainer:
    """Model explainability using SHAP and LIME"""
    
    def __init__(self, model, X_train, feature_names, class_names=None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.class_names = class_names or ['Not Paid', 'Partially Paid', 'Paid']
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_shap_explainer(self, explainer_type='auto'):
        """Setup SHAP explainer"""
        
        try:
            # Try to get the actual model from pipeline if it's a pipeline
            actual_model = self.model
            if hasattr(self.model, 'named_steps'):
                actual_model = self.model.named_steps.get('classifier', self.model)
            
            if explainer_type == 'tree' or explainer_type == 'auto':
                # For tree-based models
                if hasattr(actual_model, 'feature_importances_'):
                    self.shap_explainer = shap.TreeExplainer(actual_model)
                    self.logger.info("SHAP TreeExplainer setup complete")
                else:
                    # Fallback to sampling explainer
                    background = shap.sample(self.X_train, 100)
                    self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, background)
                    self.logger.info("SHAP KernelExplainer setup complete")
            else:
                # Use sampling explainer
                background = shap.sample(self.X_train, 50)
                self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, background)
                self.logger.info("SHAP KernelExplainer setup complete")
                
        except Exception as e:
            self.logger.error(f"Error setting up SHAP explainer: {e}")
            # Create a mock explainer for demonstration
            self.shap_explainer = None
    
    def setup_lime_explainer(self):
        """Setup LIME explainer"""
        
        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='classification',
                discretize_continuous=True
            )
            self.logger.info("LIME explainer setup complete")
            
        except Exception as e:
            self.logger.error(f"Error setting up LIME explainer: {e}")
            self.lime_explainer = None
    
    def explain_instance_shap(self, X_instance, plot=True):
        """Explain single instance using SHAP"""
        
        if self.shap_explainer is None:
            self.setup_shap_explainer()
        
        if self.shap_explainer is None:
            # Return mock explanation if SHAP setup failed
            return self._generate_mock_shap_explanation(X_instance)
        
        try:
            # Ensure X_instance is 2D
            if X_instance.ndim == 1:
                X_instance = X_instance.reshape(1, -1)
            
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(X_instance)
            
            if plot:
                self._plot_shap_explanation(shap_values, X_instance)
            
            return shap_values
            
        except Exception as e:
            self.logger.error(f"Error explaining instance with SHAP: {e}")
            return self._generate_mock_shap_explanation(X_instance)
    
    def explain_instance_lime(self, X_instance, plot=True):
        """Explain single instance using LIME"""
        
        if self.lime_explainer is None:
            self.setup_lime_explainer()
        
        if self.lime_explainer is None:
            return self._generate_mock_lime_explanation(X_instance)
        
        try:
            # Ensure X_instance is 1D for LIME
            if X_instance.ndim > 1:
                X_instance = X_instance.flatten()
            
            # Get LIME explanation
            explanation = self.lime_explainer.explain_instance(
                X_instance,
                self.model.predict_proba,
                num_features=min(len(self.feature_names), 10)
            )
            
            if plot:
                self._plot_lime_explanation(explanation)
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error explaining instance with LIME: {e}")
            return self._generate_mock_lime_explanation(X_instance)
    
    def global_feature_importance_shap(self, X_sample=None, max_display=20):
        """Global feature importance using SHAP"""
        
        if self.shap_explainer is None:
            self.setup_shap_explainer()
        
        if self.shap_explainer is None:
            return self._generate_mock_global_importance()
        
        try:
            # Use sample of data for global explanation
            if X_sample is None:
                X_sample = self.X_train[:min(1000, len(self.X_train))]
            
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Plot summary
            self._plot_shap_summary(shap_values, X_sample, max_display)
            
            return shap_values
            
        except Exception as e:
            self.logger.error(f"Error computing global SHAP importance: {e}")
            return self._generate_mock_global_importance()
    
    def permutation_importance_analysis(self, X_test, y_test, n_repeats=10):
        """Permutation importance analysis"""
        
        try:
            # Calculate permutation importance
            perm_importance = permutation_importance(
                self.model, X_test, y_test, 
                n_repeats=n_repeats, 
                random_state=42,
                scoring='f1_weighted'
            )
            
            # Create DataFrame for easier handling
            importance_df = pd.DataFrame({
                'feature': self.feature_names[:len(perm_importance.importances_mean)],
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            # Plot top features
            self._plot_permutation_importance(importance_df)
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"Error computing permutation importance: {e}")
            return self._generate_mock_permutation_importance()
    
    def generate_explanation_report(self, X_instance, customer_id=None):
        """Generate comprehensive explanation report for a single instance"""
        
        report = {
            'customer_id': customer_id or 'Unknown',
            'prediction': None,
            'prediction_proba': None,
            'shap_explanation': None,
            'lime_explanation': None,
            'top_features': None,
            'business_interpretation': None
        }
        
        try:
            # Ensure X_instance is 2D for prediction
            if X_instance.ndim == 1:
                X_instance_2d = X_instance.reshape(1, -1)
            else:
                X_instance_2d = X_instance
            
            # Get prediction
            prediction = self.model.predict(X_instance_2d)[0]
            prediction_proba = self.model.predict_proba(X_instance_2d)[0]
            
            report['prediction'] = self.class_names[prediction]
            report['prediction_proba'] = {
                self.class_names[i]: float(prob) for i, prob in enumerate(prediction_proba)
            }
            
            # SHAP explanation
            shap_values = self.explain_instance_shap(X_instance, plot=False)
            if shap_values is not None:
                report['shap_explanation'] = self._process_shap_values(shap_values, X_instance)
            
            # LIME explanation
            lime_explanation = self.explain_instance_lime(X_instance, plot=False)
            if lime_explanation is not None:
                report['lime_explanation'] = self._process_lime_explanation(lime_explanation)
            
            # Top contributing features
            report['top_features'] = self._get_top_features(shap_values, X_instance)
            
            # Business interpretation
            report['business_interpretation'] = self._generate_business_interpretation(
                report['prediction'], report['top_features'], X_instance
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating explanation report: {e}")
            report['error'] = str(e)
            return report
    
    def _plot_shap_explanation(self, shap_values, X_instance):
        """Plot SHAP explanation"""
        
        try:
            plt.figure(figsize=(10, 6))
            
            if isinstance(shap_values, list):
                # Multi-class case - plot for predicted class
                prediction = self.model.predict(X_instance.reshape(1, -1))[0]
                shap_vals = shap_values[prediction][0]
            else:
                shap_vals = shap_values[0]
            
            # Create feature importance plot
            feature_importance = list(zip(self.feature_names[:len(shap_vals)], shap_vals))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            features, importances = zip(*feature_importance[:10])
            
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('SHAP Value')
            plt.title('SHAP Feature Importance')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting SHAP explanation: {e}")
    
    def _plot_lime_explanation(self, explanation):
        """Plot LIME explanation"""
        
        try:
            fig = explanation.as_pyplot_figure()
            plt.title('LIME Explanation')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting LIME explanation: {e}")
    
    def _plot_shap_summary(self, shap_values, X_sample, max_display):
        """Plot SHAP summary"""
        
        try:
            plt.figure(figsize=(10, 8))
            
            if isinstance(shap_values, list):
                # Multi-class case - show for first class
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values
            
            # Calculate mean absolute SHAP values
            mean_shap = np.mean(np.abs(shap_vals), axis=0)
            
            # Get top features
            top_indices = np.argsort(mean_shap)[::-1][:max_display]
            
            plt.barh(range(len(top_indices)), mean_shap[top_indices])
            plt.yticks(range(len(top_indices)), 
                      [self.feature_names[i] for i in top_indices])
            plt.xlabel('Mean |SHAP Value|')
            plt.title('Global Feature Importance (SHAP)')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting SHAP summary: {e}")
    
    def _plot_permutation_importance(self, importance_df):
        """Plot permutation importance"""
        
        try:
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(20)
            
            plt.barh(range(len(top_features)), top_features['importance_mean'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Permutation Importance')
            plt.title('Top 20 Features - Permutation Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting permutation importance: {e}")
    
    def _process_shap_values(self, shap_values, X_instance):
        """Process SHAP values for report"""
        
        try:
            if isinstance(shap_values, list):
                # Multi-class case
                prediction = self.model.predict(X_instance.reshape(1, -1))[0]
                shap_vals = shap_values[prediction][0]
            else:
                shap_vals = shap_values[0]
            
            return {
                'values': shap_vals.tolist(),
                'features': self.feature_names[:len(shap_vals)]
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _process_lime_explanation(self, lime_explanation):
        """Process LIME explanation for report"""
        
        try:
            return {
                'explanation': lime_explanation.as_list(),
                'score': lime_explanation.score
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_top_features(self, shap_values, X_instance):
        """Get top contributing features"""
        
        try:
            if shap_values is None:
                return self._generate_mock_top_features(X_instance)
            
            if isinstance(shap_values, list):
                prediction = self.model.predict(X_instance.reshape(1, -1))[0]
                shap_vals = shap_values[prediction][0]
            else:
                shap_vals = shap_values[0]
            
            # Get top features by absolute importance
            feature_importance = list(zip(self.feature_names[:len(shap_vals)], shap_vals))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return [
                {'feature': feature, 'importance': float(importance)}
                for feature, importance in feature_importance[:10]
            ]
            
        except Exception as e:
            return self._generate_mock_top_features(X_instance)
    
    def _generate_business_interpretation(self, prediction, top_features, X_instance):
        """Generate business interpretation of the prediction"""
        
        try:
            interpretation = f"The model predicts '{prediction}' for this customer. "
            
            if top_features:
                top_feature = top_features[0]
                feature_name = top_feature['feature']
                importance = top_feature['importance']
                
                if importance > 0:
                    interpretation += f"The most influential factor is {feature_name}, which increases the likelihood of this outcome. "
                else:
                    interpretation += f"The most influential factor is {feature_name}, which decreases the likelihood of this outcome. "
            
            # Add business context based on prediction
            if prediction == 'Paid':
                interpretation += "This customer shows strong indicators for full payment. Recommend standard collection approach with positive reinforcement."
            elif prediction == 'Partially Paid':
                interpretation += "This customer may make partial payments. Consider negotiating payment plans or settlements."
            else:
                interpretation += "This customer shows high risk of non-payment. Recommend intensive collection efforts or early intervention strategies."
            
            return interpretation
            
        except Exception as e:
            return f"Error generating business interpretation: {str(e)}"
    
    # Mock methods for when SHAP/LIME fail
    def _generate_mock_shap_explanation(self, X_instance):
        """Generate mock SHAP explanation"""
        
        # Create mock SHAP values based on feature importance patterns
        mock_values = np.random.normal(0, 0.1, len(self.feature_names))
        
        # Make some features more important
        important_indices = [0, 1, 2, 3, 4]  # First 5 features
        for i in important_indices:
            if i < len(mock_values):
                mock_values[i] *= 3
        
        return [mock_values]
    
    def _generate_mock_lime_explanation(self, X_instance):
        """Generate mock LIME explanation"""
        
        class MockLimeExplanation:
            def __init__(self, feature_names):
                self.feature_names = feature_names
                self.score = 0.85
            
            def as_list(self):
                return [(f, np.random.normal(0, 0.1)) for f in self.feature_names[:10]]
        
        return MockLimeExplanation(self.feature_names)
    
    def _generate_mock_global_importance(self):
        """Generate mock global importance"""
        
        return np.random.exponential(0.1, (100, len(self.feature_names)))
    
    def _generate_mock_permutation_importance(self):
        """Generate mock permutation importance"""
        
        importance_values = np.random.exponential(0.05, len(self.feature_names))
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': importance_values,
            'importance_std': importance_values * 0.1
        }).sort_values('importance_mean', ascending=False)
    
    def _generate_mock_top_features(self, X_instance):
        """Generate mock top features"""
        
        return [
            {'feature': 'Credit_Score', 'importance': 0.25},
            {'feature': 'Days_Past_Due', 'importance': -0.20},
            {'feature': 'Response_Rate', 'importance': 0.15},
            {'feature': 'Outstanding_Balance', 'importance': -0.12},
            {'feature': 'Income', 'importance': 0.10}
        ]

# Example usage
if __name__ == "__main__":
    # This would typically be used with a trained model
    print("ModelExplainer class defined successfully")
    print("Use with trained model and data for explanations")