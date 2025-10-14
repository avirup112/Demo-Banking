import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
import joblib
from scipy import stats
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TimeSeriesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Extract time-series features from payment history data"""
    
    def __init__(self):
        self.feature_names = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Create time-series features from payment history patterns"""
        
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        
        # Assume first few columns contain payment-related time series data
        n_features = min(10, X_df.shape[1])
        
        try:
            # Payment velocity (trend analysis)
            for i in range(min(3, n_features)):
                col_data = X_df.iloc[:, i]
                
                # Calculate rolling statistics (simulated)
                X_df[f'payment_velocity_{i}'] = col_data.rolling(window=3, min_periods=1).mean()
                X_df[f'payment_volatility_{i}'] = col_data.rolling(window=3, min_periods=1).std().fillna(0)
                
                # Trend direction
                X_df[f'trend_positive_{i}'] = (col_data > col_data.shift(1)).astype(int)
                X_df[f'trend_negative_{i}'] = (col_data < col_data.shift(1)).astype(int)
            
            # Payment consistency score (coefficient of variation)
            if n_features >= 3:
                payment_cols = X_df.iloc[:, :3]
                X_df['payment_consistency'] = 1 / (payment_cols.std(axis=1) / (payment_cols.mean(axis=1) + 0.001) + 0.001)
                X_df['payment_range'] = payment_cols.max(axis=1) - payment_cols.min(axis=1)
                X_df['payment_stability'] = (payment_cols.std(axis=1) < payment_cols.mean(axis=1) * 0.2).astype(int)
            
            # Seasonal patterns (using modulo operations to simulate seasonality)
            if n_features >= 2:
                X_df['seasonal_pattern_1'] = np.sin(2 * np.pi * X_df.iloc[:, 0] / 12)  # Monthly pattern
                X_df['seasonal_pattern_2'] = np.cos(2 * np.pi * X_df.iloc[:, 1] / 4)   # Quarterly pattern
                
            # Payment frequency indicators
            if n_features >= 4:
                payment_data = X_df.iloc[:, :4]
                X_df['frequent_payer'] = (payment_data > 0).sum(axis=1) / 4
                X_df['irregular_payer'] = ((payment_data > 0).sum(axis=1) < 2).astype(int)
                
        except Exception as e:
            logger.warning(f"Time series feature engineering failed: {e}")
            # Add basic fallback features
            X_df['time_feature_1'] = X_df.iloc[:, 0] if X_df.shape[1] > 0 else 0
            X_df['time_feature_2'] = X_df.iloc[:, 1] if X_df.shape[1] > 1 else 0
        
        return X_df

class FinancialRatioEngineer(BaseEstimator, TransformerMixin):
    """Create financial domain-specific ratios and risk indicators"""
    
    def __init__(self):
        self.feature_names = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Create financial ratios and risk indicators"""
        
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        
        n_features = min(8, X_df.shape[1])
        
        try:
            # Assume columns represent: Age, Income, Debt_Amount, Outstanding_Balance, etc.
            if n_features >= 4:
                # Financial ratios
                X_df['debt_to_income_ratio'] = X_df.iloc[:, 2] / (X_df.iloc[:, 1] + 1)  # Debt/Income
                X_df['utilization_ratio'] = X_df.iloc[:, 3] / (X_df.iloc[:, 2] + 1)    # Outstanding/Debt
                X_df['payment_capacity'] = X_df.iloc[:, 1] / (X_df.iloc[:, 3] + 1)     # Income/Outstanding
                
                # Risk concentration index
                debt_features = X_df.iloc[:, 2:4]
                X_df['risk_concentration'] = debt_features.max(axis=1) / (debt_features.sum(axis=1) + 1)
                
            # Age-based risk categories
            if n_features >= 1:
                age_col = X_df.iloc[:, 0]
                X_df['young_borrower'] = (age_col < np.percentile(age_col, 25)).astype(int)
                X_df['senior_borrower'] = (age_col > np.percentile(age_col, 75)).astype(int)
                X_df['prime_age'] = ((age_col >= np.percentile(age_col, 25)) & 
                                   (age_col <= np.percentile(age_col, 75))).astype(int)
            
            # Income-based categories
            if n_features >= 2:
                income_col = X_df.iloc[:, 1]
                income_median = np.median(income_col)
                X_df['high_income'] = (income_col > income_median * 1.5).astype(int)
                X_df['low_income'] = (income_col < income_median * 0.5).astype(int)
                X_df['income_stability'] = (np.abs(income_col - income_median) < income_median * 0.2).astype(int)
            
            # Debt burden categories
            if n_features >= 3:
                debt_col = X_df.iloc[:, 2]
                debt_percentiles = np.percentile(debt_col, [25, 50, 75])
                X_df['low_debt'] = (debt_col <= debt_percentiles[0]).astype(int)
                X_df['medium_debt'] = ((debt_col > debt_percentiles[0]) & 
                                     (debt_col <= debt_percentiles[2])).astype(int)
                X_df['high_debt'] = (debt_col > debt_percentiles[2]).astype(int)
            
            # Risk scoring based on multiple factors
            if n_features >= 4:
                # Create composite risk score
                risk_factors = []
                
                # High debt-to-income ratio
                debt_income_ratio = X_df.iloc[:, 2] / (X_df.iloc[:, 1] + 1)
                risk_factors.append((debt_income_ratio > np.percentile(debt_income_ratio, 75)).astype(int))
                
                # High utilization
                utilization = X_df.iloc[:, 3] / (X_df.iloc[:, 2] + 1)
                risk_factors.append((utilization > np.percentile(utilization, 75)).astype(int))
                
                # Low payment capacity
                payment_cap = X_df.iloc[:, 1] / (X_df.iloc[:, 3] + 1)
                risk_factors.append((payment_cap < np.percentile(payment_cap, 25)).astype(int))
                
                X_df['composite_risk_score'] = sum(risk_factors)
                X_df['high_risk_flag'] = (X_df['composite_risk_score'] >= 2).astype(int)
                
        except Exception as e:
            logger.warning(f"Financial ratio engineering failed: {e}")
            # Add basic fallback features
            X_df['financial_ratio_1'] = X_df.iloc[:, 1] / (X_df.iloc[:, 0] + 1) if X_df.shape[1] >= 2 else 1
            X_df['financial_ratio_2'] = X_df.iloc[:, 2] / (X_df.iloc[:, 1] + 1) if X_df.shape[1] >= 3 else 1
        
        return X_df

class BehavioralFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create behavioral pattern features"""
    
    def __init__(self):
        self.feature_names = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Create behavioral pattern features"""
        
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        
        n_features = min(6, X_df.shape[1])
        
        try:
            # Contact responsiveness (simulated from available features)
            if n_features >= 3:
                # Use feature interactions to simulate behavioral patterns
                contact_proxy = X_df.iloc[:, :3].mean(axis=1)
                X_df['contact_responsiveness'] = (contact_proxy > np.median(contact_proxy)).astype(int)
                X_df['contact_consistency'] = 1 - (X_df.iloc[:, :3].std(axis=1) / (X_df.iloc[:, :3].mean(axis=1) + 0.001))
            
            # Promise-to-pay reliability (derived from payment patterns)
            if n_features >= 4:
                payment_pattern = X_df.iloc[:, :4]
                # Simulate promise reliability based on payment consistency
                payment_variance = payment_pattern.var(axis=1)
                X_df['promise_reliability'] = (payment_variance < np.percentile(payment_variance, 50)).astype(int)
                
                # Payment channel preference (categorical simulation)
                channel_preference = payment_pattern.idxmax(axis=1)
                X_df['prefers_channel_0'] = (channel_preference == 0).astype(int)
                X_df['prefers_channel_1'] = (channel_preference == 1).astype(int)
                X_df['channel_diversity'] = (payment_pattern > 0).sum(axis=1)
            
            # Engagement patterns
            if n_features >= 2:
                engagement_score = X_df.iloc[:, :2].sum(axis=1)
                X_df['high_engagement'] = (engagement_score > np.percentile(engagement_score, 75)).astype(int)
                X_df['low_engagement'] = (engagement_score < np.percentile(engagement_score, 25)).astype(int)
                
            # Behavioral risk indicators
            if n_features >= 5:
                behavior_data = X_df.iloc[:, :5]
                
                # Erratic behavior (high variance across features)
                X_df['erratic_behavior'] = (behavior_data.std(axis=1) > np.percentile(behavior_data.std(axis=1), 75)).astype(int)
                
                # Consistent behavior (low variance)
                X_df['consistent_behavior'] = (behavior_data.std(axis=1) < np.percentile(behavior_data.std(axis=1), 25)).astype(int)
                
                # Extreme values indicator
                X_df['has_extreme_values'] = ((behavior_data > behavior_data.quantile(0.95, axis=1).values.reshape(-1, 1)) | 
                                            (behavior_data < behavior_data.quantile(0.05, axis=1).values.reshape(-1, 1))).any(axis=1).astype(int)
                
        except Exception as e:
            logger.warning(f"Behavioral feature engineering failed: {e}")
            # Add basic fallback features
            X_df['behavioral_score'] = X_df.iloc[:, 0] if X_df.shape[1] > 0 else 0
            X_df['engagement_level'] = X_df.iloc[:, 1] if X_df.shape[1] > 1 else 0
        
        return X_df

class EnhancedFeatureEngineer:
    """Enhanced feature engineering pipeline with domain-specific features"""
    
    def __init__(self, 
                 include_time_series: bool = True,
                 include_financial_ratios: bool = True,
                 include_behavioral_features: bool = True,
                 include_polynomial: bool = False,
                 polynomial_degree: int = 2,
                 feature_selection: bool = True,
                 selection_k: int = 50,
                 selection_method: str = 'f_classif'):
        
        self.include_time_series = include_time_series
        self.include_financial_ratios = include_financial_ratios
        self.include_behavioral_features = include_behavioral_features
        self.include_polynomial = include_polynomial
        self.polynomial_degree = polynomial_degree
        self.feature_selection = feature_selection
        self.selection_k = selection_k
        self.selection_method = selection_method
        
        # Feature engineering components
        self.time_series_engineer = TimeSeriesFeatureEngineer() if include_time_series else None
        self.financial_engineer = FinancialRatioEngineer() if include_financial_ratios else None
        self.behavioral_engineer = BehavioralFeatureEngineer() if include_behavioral_features else None
        
        # Advanced components
        self.polynomial_features = None
        self.feature_selector = None
        
        # Feature tracking
        self.original_features = None
        self.engineered_features = None
        self.selected_features = None
        self.feature_importance_scores = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit enhanced feature engineering pipeline and transform data"""
        
        self.logger.info("Starting enhanced feature engineering...")
        
        # Store original features
        self.original_features = list(X.columns) if hasattr(X, 'columns') else None
        original_shape = X.shape
        
        # Start with original data
        X_enhanced = X.copy() if hasattr(X, 'copy') else pd.DataFrame(X)
        
        # Apply time-series feature engineering
        if self.time_series_engineer:
            try:
                X_time = self.time_series_engineer.fit_transform(X_enhanced)
                X_enhanced = X_time
                self.logger.info(f"Time-series features added. Shape: {X_enhanced.shape}")
            except Exception as e:
                self.logger.warning(f"Time-series feature engineering failed: {e}")
        
        # Apply financial ratio engineering
        if self.financial_engineer:
            try:
                X_financial = self.financial_engineer.fit_transform(X_enhanced)
                X_enhanced = X_financial
                self.logger.info(f"Financial ratio features added. Shape: {X_enhanced.shape}")
            except Exception as e:
                self.logger.warning(f"Financial ratio engineering failed: {e}")
        
        # Apply behavioral feature engineering
        if self.behavioral_engineer:
            try:
                X_behavioral = self.behavioral_engineer.fit_transform(X_enhanced)
                X_enhanced = X_behavioral
                self.logger.info(f"Behavioral features added. Shape: {X_enhanced.shape}")
            except Exception as e:
                self.logger.warning(f"Behavioral feature engineering failed: {e}")
        
        # Convert to DataFrame if not already
        if not isinstance(X_enhanced, pd.DataFrame):
            X_enhanced = pd.DataFrame(X_enhanced)
        
        # Polynomial features (only for small feature sets)
        if self.include_polynomial and X_enhanced.shape[1] < 30:
            try:
                # Use only first 8 features for polynomial to avoid explosion
                X_poly_input = X_enhanced.iloc[:, :min(8, X_enhanced.shape[1])]
                
                self.polynomial_features = PolynomialFeatures(
                    degree=self.polynomial_degree,
                    include_bias=False,
                    interaction_only=True
                )
                X_poly = self.polynomial_features.fit_transform(X_poly_input)
                
                # Combine with existing features
                X_poly_df = pd.DataFrame(X_poly, columns=[f'poly_{i}' for i in range(X_poly.shape[1])])
                X_enhanced = pd.concat([X_enhanced.reset_index(drop=True), X_poly_df], axis=1)
                
                self.logger.info(f"Polynomial features added. Shape: {X_enhanced.shape}")
            except Exception as e:
                self.logger.warning(f"Polynomial features failed: {e}")
        
        # Feature selection using mutual information or f_classif
        if self.feature_selection and y is not None and X_enhanced.shape[1] > self.selection_k:
            try:
                # Choose scoring function
                if self.selection_method == 'mutual_info':
                    score_func = mutual_info_classif
                else:
                    score_func = f_classif
                
                self.feature_selector = SelectKBest(
                    score_func=score_func,
                    k=min(self.selection_k, X_enhanced.shape[1])
                )
                
                X_selected = self.feature_selector.fit_transform(X_enhanced, y)
                
                # Store feature importance scores
                if hasattr(self.feature_selector, 'scores_'):
                    selected_indices = self.feature_selector.get_support(indices=True)
                    self.feature_importance_scores = {
                        f'feature_{i}': score for i, score in 
                        zip(selected_indices, self.feature_selector.scores_[selected_indices])
                    }
                
                self.selected_features = [f"selected_feature_{i}" for i in range(X_selected.shape[1])]
                X_enhanced = pd.DataFrame(X_selected, columns=self.selected_features)
                
                self.logger.info(f"Feature selection completed. Selected {X_selected.shape[1]} features")
                
            except Exception as e:
                self.logger.warning(f"Feature selection failed: {e}")
        
        self.engineered_features = list(X_enhanced.columns)
        
        self.logger.info(f"Enhanced feature engineering complete.")
        self.logger.info(f"Original shape: {original_shape} -> Final shape: {X_enhanced.shape}")
        
        return X_enhanced.values
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted enhanced feature engineering pipeline"""
        
        # Start with original data
        X_enhanced = X.copy() if hasattr(X, 'copy') else pd.DataFrame(X)
        
        # Apply time-series feature engineering
        if self.time_series_engineer:
            try:
                X_enhanced = self.time_series_engineer.transform(X_enhanced)
            except Exception as e:
                self.logger.warning(f"Time-series transform failed: {e}")
        
        # Apply financial ratio engineering
        if self.financial_engineer:
            try:
                X_enhanced = self.financial_engineer.transform(X_enhanced)
            except Exception as e:
                self.logger.warning(f"Financial ratio transform failed: {e}")
        
        # Apply behavioral feature engineering
        if self.behavioral_engineer:
            try:
                X_enhanced = self.behavioral_engineer.transform(X_enhanced)
            except Exception as e:
                self.logger.warning(f"Behavioral transform failed: {e}")
        
        # Convert to DataFrame if not already
        if not isinstance(X_enhanced, pd.DataFrame):
            X_enhanced = pd.DataFrame(X_enhanced)
        
        # Apply polynomial features if fitted
        if self.polynomial_features is not None:
            try:
                X_poly_input = X_enhanced.iloc[:, :min(8, X_enhanced.shape[1])]
                X_poly = self.polynomial_features.transform(X_poly_input)
                X_poly_df = pd.DataFrame(X_poly, columns=[f'poly_{i}' for i in range(X_poly.shape[1])])
                X_enhanced = pd.concat([X_enhanced.reset_index(drop=True), X_poly_df], axis=1)
            except Exception as e:
                self.logger.warning(f"Polynomial transform failed: {e}")
        
        # Apply feature selection if fitted
        if self.feature_selector is not None:
            try:
                X_selected = self.feature_selector.transform(X_enhanced)
                X_enhanced = pd.DataFrame(X_selected, columns=self.selected_features)
            except Exception as e:
                self.logger.warning(f"Feature selection transform failed: {e}")
        
        return X_enhanced.values
    
    def get_feature_importance_analysis(self) -> Dict[str, Any]:
        """Get comprehensive feature importance analysis"""
        
        analysis = {
            'original_feature_count': len(self.original_features) if self.original_features else 0,
            'final_feature_count': len(self.engineered_features) if self.engineered_features else 0,
            'feature_engineering_methods': [],
            'feature_importance_scores': self.feature_importance_scores
        }
        
        if self.include_time_series:
            analysis['feature_engineering_methods'].append('Time Series Features')
        
        if self.include_financial_ratios:
            analysis['feature_engineering_methods'].append('Financial Ratios')
        
        if self.include_behavioral_features:
            analysis['feature_engineering_methods'].append('Behavioral Features')
        
        if self.include_polynomial:
            analysis['feature_engineering_methods'].append('Polynomial Features')
        
        if self.feature_selection:
            analysis['feature_engineering_methods'].append(f'Feature Selection ({self.selection_method})')
        
        return analysis
    
    def save_feature_engineer(self, filepath: str):
        """Save enhanced feature engineering pipeline"""
        
        engineer_data = {
            'time_series_engineer': self.time_series_engineer,
            'financial_engineer': self.financial_engineer,
            'behavioral_engineer': self.behavioral_engineer,
            'polynomial_features': self.polynomial_features,
            'feature_selector': self.feature_selector,
            'original_features': self.original_features,
            'engineered_features': self.engineered_features,
            'selected_features': self.selected_features,
            'feature_importance_scores': self.feature_importance_scores,
            'config': {
                'include_time_series': self.include_time_series,
                'include_financial_ratios': self.include_financial_ratios,
                'include_behavioral_features': self.include_behavioral_features,
                'include_polynomial': self.include_polynomial,
                'polynomial_degree': self.polynomial_degree,
                'feature_selection': self.feature_selection,
                'selection_k': self.selection_k,
                'selection_method': self.selection_method
            }
        }
        
        joblib.dump(engineer_data, filepath)
        self.logger.info(f"Enhanced feature engineer saved to {filepath}")
    
    def load_feature_engineer(self, filepath: str):
        """Load enhanced feature engineering pipeline"""
        
        engineer_data = joblib.load(filepath)
        
        self.time_series_engineer = engineer_data['time_series_engineer']
        self.financial_engineer = engineer_data['financial_engineer']
        self.behavioral_engineer = engineer_data['behavioral_engineer']
        self.polynomial_features = engineer_data['polynomial_features']
        self.feature_selector = engineer_data['feature_selector']
        self.original_features = engineer_data['original_features']
        self.engineered_features = engineer_data['engineered_features']
        self.selected_features = engineer_data['selected_features']
        self.feature_importance_scores = engineer_data['feature_importance_scores']
        
        config = engineer_data['config']
        self.include_time_series = config['include_time_series']
        self.include_financial_ratios = config['include_financial_ratios']
        self.include_behavioral_features = config['include_behavioral_features']
        self.include_polynomial = config['include_polynomial']
        self.polynomial_degree = config['polynomial_degree']
        self.feature_selection = config['feature_selection']
        self.selection_k = config['selection_k']
        self.selection_method = config['selection_method']
        
        self.logger.info(f"Enhanced feature engineer loaded from {filepath}")

if __name__ == "__main__":
    # Test the enhanced feature engineer
    import pandas as pd
    import numpy as np
    
    # Create sample preprocessed data
    np.random.seed(42)
    data = np.random.randn(1000, 15)  # 1000 samples, 15 features
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(15)])
    y = np.random.randint(0, 3, 1000)  # Sample target
    
    # Test enhanced feature engineer
    engineer = EnhancedFeatureEngineer(
        include_time_series=True,
        include_financial_ratios=True,
        include_behavioral_features=True,
        include_polynomial=False,  # Disable for test
        feature_selection=True,
        selection_k=30,
        selection_method='f_classif'
    )
    
    X_enhanced = engineer.fit_transform(df, y)
    
    print(f"Original shape: {df.shape}")
    print(f"Enhanced shape: {X_enhanced.shape}")
    
    # Get feature importance analysis
    analysis = engineer.get_feature_importance_analysis()
    print(f"Feature engineering methods used: {analysis['feature_engineering_methods']}")
    print(f"Original features: {analysis['original_feature_count']}")
    print(f"Final features: {analysis['final_feature_count']}")
    
    print("Enhanced feature engineering test completed!")