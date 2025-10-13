import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """Financial domain-specific feature engineering"""
    
    def __init__(self):
        self.feature_names = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Create financial domain features"""
        
        if isinstance(X, np.ndarray):
            # Convert to DataFrame for easier manipulation
            # Assume standard column order from preprocessor
            columns = ['Age', 'Income', 'Loan_Amount', 'Outstanding_Balance', 
                      'Days_Past_Due', 'Number_of_Calls', 'Response_Rate', 'Credit_Score']
            X_df = pd.DataFrame(X[:, :len(columns)], columns=columns[:X.shape[1]])
        else:
            X_df = X.copy()
        
        # Financial ratios
        X_df['Debt_to_Income_Ratio'] = X_df['Outstanding_Balance'] / (X_df['Income'] + 1)
        X_df['Loan_Utilization_Ratio'] = X_df['Outstanding_Balance'] / (X_df['Loan_Amount'] + 1)
        X_df['Payment_Capacity'] = (X_df['Income'] - X_df['Outstanding_Balance']) / (X_df['Income'] + 1)
        
        # Risk indicators
        X_df['High_Risk_DPD'] = (X_df['Days_Past_Due'] > 90).astype(int)
        X_df['Critical_Risk_DPD'] = (X_df['Days_Past_Due'] > 180).astype(int)
        X_df['Low_Credit_Score'] = (X_df['Credit_Score'] < 600).astype(int)
        
        # Behavioral features
        X_df['Call_Response_Efficiency'] = X_df['Response_Rate'] / (X_df['Number_of_Calls'] + 1)
        X_df['Communication_Engagement'] = X_df['Response_Rate'] * X_df['Number_of_Calls'] / 100
        
        # Age-based features
        X_df['Young_Borrower'] = (X_df['Age'] < 30).astype(int)
        X_df['Senior_Borrower'] = (X_df['Age'] > 55).astype(int)
        X_df['Prime_Age_Borrower'] = ((X_df['Age'] >= 30) & (X_df['Age'] <= 55)).astype(int)
        
        # Income-based features
        income_percentiles = np.percentile(X_df['Income'], [25, 50, 75])
        X_df['Low_Income'] = (X_df['Income'] <= income_percentiles[0]).astype(int)
        X_df['Medium_Income'] = ((X_df['Income'] > income_percentiles[0]) & 
                                (X_df['Income'] <= income_percentiles[2])).astype(int)
        X_df['High_Income'] = (X_df['Income'] > income_percentiles[2]).astype(int)
        
        # Interaction features
        X_df['Age_Income_Interaction'] = X_df['Age'] * X_df['Income'] / 1000000
        X_df['Credit_Income_Interaction'] = X_df['Credit_Score'] * X_df['Income'] / 100000
        X_df['DPD_Debt_Interaction'] = X_df['Days_Past_Due'] * X_df['Outstanding_Balance'] / 100000
        
        # Urgency and priority scores
        X_df['Collection_Urgency'] = (
            0.4 * (X_df['Days_Past_Due'] / 365) +
            0.3 * (X_df['Outstanding_Balance'] / X_df['Income']) +
            0.2 * (1 - X_df['Credit_Score'] / 850) +
            0.1 * (1 - X_df['Response_Rate'] / 100)
        )
        
        X_df['Recovery_Potential'] = (
            0.3 * (X_df['Credit_Score'] / 850) +
            0.25 * (X_df['Income'] / X_df['Outstanding_Balance']) +
            0.25 * (X_df['Response_Rate'] / 100) +
            0.2 * (1 - X_df['Days_Past_Due'] / 365)
        )
        
        return X_df.values

class TimeBasedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Time-based feature engineering"""
    
    def __init__(self):
        self.reference_date = datetime.now()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Create time-based features"""
        
        X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
        
        # Assuming Days_Past_Due is available
        if 'Days_Past_Due' in X_df.columns:
            # Time-based risk categories
            X_df['DPD_Category'] = pd.cut(
                X_df['Days_Past_Due'],
                bins=[0, 30, 60, 90, 180, 365, float('inf')],
                labels=['Current', 'Early', 'Moderate', 'Serious', 'Critical', 'Severe']
            ).astype(str)
            
            # Seasonal effects (mock - in real scenario, use actual dates)
            X_df['Days_Since_Last_Contact'] = X_df['Days_Past_Due'] + np.random.randint(-10, 10, len(X_df))
            X_df['Contact_Recency'] = np.exp(-X_df['Days_Since_Last_Contact'] / 30)  # Exponential decay
            
            # Time pressure indicators
            X_df['Time_Pressure_Score'] = np.minimum(X_df['Days_Past_Due'] / 180, 1.0)
            
        return X_df.values

class BehavioralFeatureEngineer(BaseEstimator, TransformerMixin):
    """Behavioral pattern feature engineering"""
    
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Create behavioral features"""
        
        X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
        
        # Communication behavior
        if 'Number_of_Calls' in X_df.columns and 'Response_Rate' in X_df.columns:
            X_df['Communication_Score'] = (
                0.6 * (X_df['Response_Rate'] / 100) +
                0.4 * np.minimum(X_df['Number_of_Calls'] / 20, 1.0)
            )
            
            # Response patterns
            X_df['High_Responder'] = (X_df['Response_Rate'] > 70).astype(int)
            X_df['Low_Responder'] = (X_df['Response_Rate'] < 30).astype(int)
            X_df['Frequent_Contact'] = (X_df['Number_of_Calls'] > 10).astype(int)
            
        # Payment behavior indicators
        if 'Payment_Made_Last_30_Days' in X_df.columns:
            X_df['Recent_Payment_Behavior'] = X_df['Payment_Made_Last_30_Days']
            
        return X_df.values

class AdvancedFeatureEngineer:
    """Advanced feature engineering pipeline"""
    
    def __init__(self, 
                 include_polynomial: bool = False,
                 polynomial_degree: int = 2,
                 include_pca: bool = False,
                 pca_components: int = 10,
                 feature_selection: bool = True,
                 selection_k: int = 50):
        
        self.include_polynomial = include_polynomial
        self.polynomial_degree = polynomial_degree
        self.include_pca = include_pca
        self.pca_components = pca_components
        self.feature_selection = feature_selection
        self.selection_k = selection_k
        
        # Feature engineering components
        self.financial_engineer = FinancialFeatureEngineer()
        self.time_engineer = TimeBasedFeatureEngineer()
        self.behavioral_engineer = BehavioralFeatureEngineer()
        
        # Advanced components
        self.polynomial_features = None
        self.pca = None
        self.feature_selector = None
        
        # Feature tracking
        self.original_features = None
        self.engineered_features = None
        self.selected_features = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit feature engineering pipeline and transform data"""
        
        self.logger.info("Starting advanced feature engineering...")
        
        # Store original features
        self.original_features = list(X.columns)
        
        # Apply domain-specific feature engineering
        X_financial = self.financial_engineer.fit_transform(X)
        X_time = self.time_engineer.fit_transform(X_financial)
        X_behavioral = self.behavioral_engineer.fit_transform(X_time)
        
        # Convert back to DataFrame for easier handling
        X_engineered = pd.DataFrame(X_behavioral)
        
        self.logger.info(f"Created {X_engineered.shape[1]} features after domain engineering")
        
        # Polynomial features
        if self.include_polynomial:
            self.polynomial_features = PolynomialFeatures(
                degree=self.polynomial_degree,
                include_bias=False,
                interaction_only=True
            )
            X_poly = self.polynomial_features.fit_transform(X_engineered)
            X_engineered = pd.DataFrame(X_poly)
            self.logger.info(f"Added polynomial features. Total: {X_engineered.shape[1]}")
        
        # Feature selection
        if self.feature_selection and y is not None:
            self.feature_selector = SelectKBest(
                score_func=mutual_info_classif,
                k=min(self.selection_k, X_engineered.shape[1])
            )
            X_selected = self.feature_selector.fit_transform(X_engineered, y)
            
            # Get selected feature indices
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_features = [f"feature_{i}" for i in selected_indices]
            
            self.logger.info(f"Selected {X_selected.shape[1]} best features")
            X_engineered = pd.DataFrame(X_selected, columns=self.selected_features)
        
        # PCA (applied after feature selection)
        if self.include_pca:
            self.pca = PCA(n_components=min(self.pca_components, X_engineered.shape[1]))
            X_pca = self.pca.fit_transform(X_engineered)
            
            pca_features = [f"pca_component_{i}" for i in range(X_pca.shape[1])]
            X_engineered = pd.DataFrame(X_pca, columns=pca_features)
            
            self.logger.info(f"Applied PCA. Components: {X_engineered.shape[1]}")
            self.logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_[:5]}")
        
        self.engineered_features = list(X_engineered.columns)
        
        self.logger.info(f"Feature engineering complete. Final shape: {X_engineered.shape}")
        
        return X_engineered.values
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted feature engineering pipeline"""
        
        # Apply domain-specific feature engineering
        X_financial = self.financial_engineer.transform(X)
        X_time = self.time_engineer.transform(X_financial)
        X_behavioral = self.behavioral_engineer.transform(X_time)
        
        # Convert to DataFrame
        X_engineered = pd.DataFrame(X_behavioral)
        
        # Apply polynomial features if fitted
        if self.polynomial_features is not None:
            X_poly = self.polynomial_features.transform(X_engineered)
            X_engineered = pd.DataFrame(X_poly)
        
        # Apply feature selection if fitted
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_engineered)
            X_engineered = pd.DataFrame(X_selected, columns=self.selected_features)
        
        # Apply PCA if fitted
        if self.pca is not None:
            X_pca = self.pca.transform(X_engineered)
            pca_features = [f"pca_component_{i}" for i in range(X_pca.shape[1])]
            X_engineered = pd.DataFrame(X_pca, columns=pca_features)
        
        return X_engineered.values
    
    def get_feature_importance_analysis(self) -> Dict[str, Any]:
        """Get feature importance analysis"""
        
        analysis = {
            'original_feature_count': len(self.original_features) if self.original_features else 0,
            'engineered_feature_count': len(self.engineered_features) if self.engineered_features else 0,
            'feature_engineering_methods': []
        }
        
        if self.include_polynomial:
            analysis['feature_engineering_methods'].append('Polynomial Features')
        
        if self.include_pca:
            analysis['feature_engineering_methods'].append('PCA')
            if self.pca:
                analysis['pca_explained_variance'] = self.pca.explained_variance_ratio_.tolist()
        
        if self.feature_selection:
            analysis['feature_engineering_methods'].append('Feature Selection')
            if self.feature_selector:
                analysis['selected_feature_scores'] = self.feature_selector.scores_.tolist()
        
        return analysis
    
    def save_feature_engineer(self, filepath: str):
        """Save feature engineering pipeline"""
        
        engineer_data = {
            'financial_engineer': self.financial_engineer,
            'time_engineer': self.time_engineer,
            'behavioral_engineer': self.behavioral_engineer,
            'polynomial_features': self.polynomial_features,
            'pca': self.pca,
            'feature_selector': self.feature_selector,
            'original_features': self.original_features,
            'engineered_features': self.engineered_features,
            'selected_features': self.selected_features,
            'config': {
                'include_polynomial': self.include_polynomial,
                'polynomial_degree': self.polynomial_degree,
                'include_pca': self.include_pca,
                'pca_components': self.pca_components,
                'feature_selection': self.feature_selection,
                'selection_k': self.selection_k
            }
        }
        
        joblib.dump(engineer_data, filepath)
        self.logger.info(f"Feature engineer saved to {filepath}")
    
    def load_feature_engineer(self, filepath: str):
        """Load feature engineering pipeline"""
        
        engineer_data = joblib.load(filepath)
        
        self.financial_engineer = engineer_data['financial_engineer']
        self.time_engineer = engineer_data['time_engineer']
        self.behavioral_engineer = engineer_data['behavioral_engineer']
        self.polynomial_features = engineer_data['polynomial_features']
        self.pca = engineer_data['pca']
        self.feature_selector = engineer_data['feature_selector']
        self.original_features = engineer_data['original_features']
        self.engineered_features = engineer_data['engineered_features']
        self.selected_features = engineer_data['selected_features']
        
        config = engineer_data['config']
        self.include_polynomial = config['include_polynomial']
        self.polynomial_degree = config['polynomial_degree']
        self.include_pca = config['include_pca']
        self.pca_components = config['pca_components']
        self.feature_selection = config['feature_selection']
        self.selection_k = config['selection_k']
        
        self.logger.info(f"Feature engineer loaded from {filepath}")
