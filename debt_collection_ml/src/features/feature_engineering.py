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
import joblib
warnings.filterwarnings('ignore')

class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """Financial domain-specific feature engineering for preprocessed data"""
    
    def __init__(self):
        self.feature_names = []
        self.original_columns = None
        
    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.original_columns = list(X.columns)
        return self
    
    def transform(self, X):
        """Create financial domain features from preprocessed data"""
        
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        
        # Since data is already preprocessed with OneHot encoding, 
        # we work with the available numeric features (first few columns)
        
        try:
            # Create simple derived features using available data
            n_features = min(10, X_df.shape[1])  # Use first 10 features
            
            # Basic ratios and interactions
            if n_features >= 4:
                X_df[f'ratio_1_2'] = X_df.iloc[:, 1] / (X_df.iloc[:, 0] + 0.001)
                X_df[f'ratio_3_4'] = X_df.iloc[:, 3] / (X_df.iloc[:, 2] + 0.001)
                X_df[f'product_1_2'] = X_df.iloc[:, 0] * X_df.iloc[:, 1]
                X_df[f'product_3_4'] = X_df.iloc[:, 2] * X_df.iloc[:, 3]
            
            # Statistical features
            if n_features >= 5:
                X_df['feature_sum'] = X_df.iloc[:, :n_features].sum(axis=1)
                X_df['feature_mean'] = X_df.iloc[:, :n_features].mean(axis=1)
                X_df['feature_std'] = X_df.iloc[:, :n_features].std(axis=1).fillna(0)
                X_df['feature_max'] = X_df.iloc[:, :n_features].max(axis=1)
                X_df['feature_min'] = X_df.iloc[:, :n_features].min(axis=1)
            
            # Risk indicators using quantiles
            if n_features >= 3:
                for i in range(min(3, n_features)):
                    col_name = f'high_risk_{i}'
                    threshold = X_df.iloc[:, i].quantile(0.75)
                    X_df[col_name] = (X_df.iloc[:, i] > threshold).astype(int)
                    
                    col_name = f'low_risk_{i}'
                    threshold = X_df.iloc[:, i].quantile(0.25)
                    X_df[col_name] = (X_df.iloc[:, i] < threshold).astype(int)
            
            # Polynomial features for first few columns
            if n_features >= 2:
                X_df['squared_1'] = X_df.iloc[:, 0] ** 2
                X_df['squared_2'] = X_df.iloc[:, 1] ** 2
                if n_features >= 3:
                    X_df['cubic_1'] = X_df.iloc[:, 0] ** 3
            
        except Exception as e:
            # Fallback: just add some basic features
            X_df['feature_sum'] = X_df.sum(axis=1)
            X_df['feature_mean'] = X_df.mean(axis=1)
            X_df['constant_feature'] = 1
        
        return X_df.values

class AdvancedFeatureEngineer:
    """Advanced feature engineering pipeline for preprocessed data"""
    
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
        self.original_features = list(X.columns) if hasattr(X, 'columns') else None
        
        # Apply domain-specific feature engineering
        X_financial = self.financial_engineer.fit_transform(X)
        
        # Convert back to DataFrame for easier handling
        X_engineered = pd.DataFrame(X_financial)
        
        self.logger.info(f"Created {X_engineered.shape[1]} features after domain engineering")
        
        # Polynomial features (only if reasonable number of features)
        if self.include_polynomial and X_engineered.shape[1] < 50:
            try:
                # Use only first 10 features for polynomial to avoid explosion
                X_poly_input = X_engineered.iloc[:, :min(10, X_engineered.shape[1])]
                
                self.polynomial_features = PolynomialFeatures(
                    degree=self.polynomial_degree,
                    include_bias=False,
                    interaction_only=True
                )
                X_poly = self.polynomial_features.fit_transform(X_poly_input)
                
                # Combine original features with polynomial
                X_poly_df = pd.DataFrame(X_poly)
                X_engineered = pd.concat([X_engineered, X_poly_df], axis=1)
                
                self.logger.info(f"Added polynomial features. Total: {X_engineered.shape[1]}")
            except Exception as e:
                self.logger.warning(f"Polynomial features failed: {e}")
        
        # Feature selection
        if self.feature_selection and y is not None and X_engineered.shape[1] > self.selection_k:
            try:
                self.feature_selector = SelectKBest(
                    score_func=f_classif,  # Use f_classif instead of mutual_info_classif for speed
                    k=min(self.selection_k, X_engineered.shape[1])
                )
                X_selected = self.feature_selector.fit_transform(X_engineered, y)
                
                # Get selected feature indices
                selected_indices = self.feature_selector.get_support(indices=True)
                self.selected_features = [f"feature_{i}" for i in selected_indices]
                
                self.logger.info(f"Selected {X_selected.shape[1]} best features")
                X_engineered = pd.DataFrame(X_selected, columns=self.selected_features)
            except Exception as e:
                self.logger.warning(f"Feature selection failed: {e}")
        
        # PCA (applied after feature selection)
        if self.include_pca and X_engineered.shape[1] > self.pca_components:
            try:
                self.pca = PCA(n_components=min(self.pca_components, X_engineered.shape[1]))
                X_pca = self.pca.fit_transform(X_engineered)
                
                pca_features = [f"pca_component_{i}" for i in range(X_pca.shape[1])]
                X_engineered = pd.DataFrame(X_pca, columns=pca_features)
                
                self.logger.info(f"Applied PCA. Components: {X_engineered.shape[1]}")
                if hasattr(self.pca, 'explained_variance_ratio_'):
                    self.logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_[:5]}")
            except Exception as e:
                self.logger.warning(f"PCA failed: {e}")
        
        self.engineered_features = list(X_engineered.columns)
        
        self.logger.info(f"Feature engineering complete. Final shape: {X_engineered.shape}")
        
        return X_engineered.values
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted feature engineering pipeline"""
        
        # Apply domain-specific feature engineering
        X_financial = self.financial_engineer.transform(X)
        
        # Convert to DataFrame
        X_engineered = pd.DataFrame(X_financial)
        
        # Apply polynomial features if fitted
        if self.polynomial_features is not None:
            try:
                X_poly_input = X_engineered.iloc[:, :min(10, X_engineered.shape[1])]
                X_poly = self.polynomial_features.transform(X_poly_input)
                X_poly_df = pd.DataFrame(X_poly)
                X_engineered = pd.concat([X_engineered, X_poly_df], axis=1)
            except Exception as e:
                self.logger.warning(f"Polynomial transform failed: {e}")
        
        # Apply feature selection if fitted
        if self.feature_selector is not None:
            try:
                X_selected = self.feature_selector.transform(X_engineered)
                X_engineered = pd.DataFrame(X_selected, columns=self.selected_features)
            except Exception as e:
                self.logger.warning(f"Feature selection transform failed: {e}")
        
        # Apply PCA if fitted
        if self.pca is not None:
            try:
                X_pca = self.pca.transform(X_engineered)
                pca_features = [f"pca_component_{i}" for i in range(X_pca.shape[1])]
                X_engineered = pd.DataFrame(X_pca, columns=pca_features)
            except Exception as e:
                self.logger.warning(f"PCA transform failed: {e}")
        
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
            if self.pca and hasattr(self.pca, 'explained_variance_ratio_'):
                analysis['pca_explained_variance'] = self.pca.explained_variance_ratio_.tolist()
        
        if self.feature_selection:
            analysis['feature_engineering_methods'].append('Feature Selection')
            if self.feature_selector and hasattr(self.feature_selector, 'scores_'):
                analysis['selected_feature_scores'] = self.feature_selector.scores_.tolist()
        
        return analysis
    
    def save_feature_engineer(self, filepath: str):
        """Save feature engineering pipeline"""
        
        engineer_data = {
            'financial_engineer': self.financial_engineer,
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

def main():
    """Main function with command line argument support"""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Engineer features for debt collection data')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory with preprocessed data')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for engineered features')
    parser.add_argument('--polynomial', action='store_true',
                       help='Include polynomial features')
    parser.add_argument('--selection-k', type=int, default=50,
                       help='Number of features to select')
    
    args = parser.parse_args()
    
    # Setup logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading preprocessed data from {args.input}")
    
    # Load preprocessed data
    input_path = Path(args.input)
    X_train = np.load(input_path / 'X_train.npy')
    X_test = np.load(input_path / 'X_test.npy')
    y_train = np.load(input_path / 'y_train.npy')
    y_test = np.load(input_path / 'y_test.npy')
    
    logger.info(f"Loaded training data with shape: {X_train.shape}")
    logger.info(f"Loaded test data with shape: {X_test.shape}")
    
    # Convert to DataFrame for feature engineering
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    
    # Initialize feature engineer
    engineer = AdvancedFeatureEngineer(
        include_polynomial=args.polynomial,
        feature_selection=True,
        selection_k=args.selection_k
    )
    
    # Fit and transform training data
    X_train_engineered = engineer.fit_transform(X_train_df, y_train)
    
    # Transform test data
    X_test_engineered = engineer.transform(X_test_df)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save engineered features
    np.save(output_path / 'X_train_features.npy', X_train_engineered)
    np.save(output_path / 'X_test_features.npy', X_test_engineered)
    
    # Save feature engineer
    engineer.save_feature_engineer(output_path / 'feature_engineer.joblib')
    
    # Save feature info
    import json
    feature_info = {
        'original_features': X_train.shape[1],
        'engineered_features': X_train_engineered.shape[1],
        'selected_features': len(engineer.selected_features) if engineer.selected_features else 0,
        'feature_names': engineer.feature_names if hasattr(engineer, 'feature_names') else [],
        'polynomial_features': args.polynomial,
        'selection_k': args.selection_k
    }
    
    with open(output_path / 'feature_engineering_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    logger.info(f"✅ Feature engineering complete!")
    logger.info(f"📁 Saved to: {output_path}")
    logger.info(f"📊 Original features: {X_train.shape[1]}")
    logger.info(f"📊 Engineered features: {X_train_engineered.shape[1]}")
    logger.info(f"🎯 Selected features: {len(engineer.selected_features) if engineer.selected_features else 0}")

if __name__ == "__main__":
    main()