import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class OutlierHandler(BaseEstimator, TransformerMixin):
    """Custom transformer for handling outliers"""
    
    def __init__(self, method='iqr', threshold=1.5):
        self.method = method
        self.threshold = threshold
        self.bounds_ = {}
        
    def fit(self, X, y=None):
        """Fit outlier detection parameters"""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        for col in X_df.select_dtypes(include=[np.number]).columns:
            if self.method == 'iqr':
                Q1 = X_df[col].quantile(0.25)
                Q3 = X_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.threshold * IQR
                upper_bound = Q3 + self.threshold * IQR
                self.bounds_[col] = (lower_bound, upper_bound)
            
            elif self.method == 'zscore':
                mean = X_df[col].mean()
                std = X_df[col].std()
                lower_bound = mean - self.threshold * std
                upper_bound = mean + self.threshold * std
                self.bounds_[col] = (lower_bound, upper_bound)
        
        return self
    
    def transform(self, X):
        """Transform data by capping outliers"""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col, (lower_bound, upper_bound) in self.bounds_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return X_df.values if not isinstance(X, pd.DataFrame) else X_df

class DataQualityChecker:
    """Data quality assessment and reporting"""
    
    def __init__(self):
        self.quality_report = {}
    
    def assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data': {},
            'duplicates': {},
            'data_types': {},
            'outliers': {},
            'cardinality': {},
            'quality_score': 0
        }
        
        # Missing data analysis
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        for col in df.columns:
            report['missing_data'][col] = {
                'count': int(missing_counts[col]),
                'percentage': float(missing_percentages[col])
            }
        
        # Duplicate analysis
        duplicate_rows = df.duplicated().sum()
        report['duplicates'] = {
            'total_duplicates': int(duplicate_rows),
            'percentage': float((duplicate_rows / len(df)) * 100)
        }
        
        # Data types
        for col in df.columns:
            report['data_types'][col] = str(df[col].dtype)
        
        # Cardinality analysis
        for col in df.columns:
            unique_count = df[col].nunique()
            report['cardinality'][col] = {
                'unique_values': int(unique_count),
                'cardinality_ratio': float(unique_count / len(df))
            }
        
        # Outlier detection for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            report['outliers'][col] = {
                'count': int(outliers),
                'percentage': float((outliers / len(df)) * 100)
            }
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(report)
        report['quality_score'] = quality_score
        
        self.quality_report = report
        return report
    
    def _calculate_quality_score(self, report: Dict) -> float:
        """Calculate overall data quality score (0-100)"""
        
        score = 100
        
        # Penalize for missing data
        avg_missing = np.mean([info['percentage'] for info in report['missing_data'].values()])
        score -= min(avg_missing * 2, 30)  # Max penalty: 30 points
        
        # Penalize for duplicates
        duplicate_penalty = min(report['duplicates']['percentage'] * 3, 20)  # Max penalty: 20 points
        score -= duplicate_penalty
        
        # Penalize for excessive outliers
        if report['outliers']:
            avg_outliers = np.mean([info['percentage'] for info in report['outliers'].values()])
            outlier_penalty = min(avg_outliers * 0.5, 15)  # Max penalty: 15 points
            score -= outlier_penalty
        
        return max(score, 0)
    
    def generate_quality_report(self) -> str:
        """Generate human-readable quality report"""
        
        if not self.quality_report:
            return "No quality assessment performed yet."
        
        report = self.quality_report
        
        report_text = f"""
DATA QUALITY REPORT
==================
Overall Quality Score: {report['quality_score']:.1f}/100

Dataset Overview:
- Total Rows: {report['total_rows']:,}
- Total Columns: {report['total_columns']}
- Duplicate Rows: {report['duplicates']['total_duplicates']} ({report['duplicates']['percentage']:.1f}%)

Missing Data Summary:
"""
        
        # Missing data details
        high_missing = [(col, info['percentage']) for col, info in report['missing_data'].items() 
                       if info['percentage'] > 5]
        
        if high_missing:
            report_text += "Columns with >5% missing data:\n"
            for col, pct in sorted(high_missing, key=lambda x: x[1], reverse=True):
                report_text += f"  - {col}: {pct:.1f}%\n"
        else:
            report_text += "No columns with significant missing data (>5%)\n"
        
        # Outlier summary
        if report['outliers']:
            high_outliers = [(col, info['percentage']) for col, info in report['outliers'].items() 
                           if info['percentage'] > 5]
            
            if high_outliers:
                report_text += "\nColumns with >5% outliers:\n"
                for col, pct in sorted(high_outliers, key=lambda x: x[1], reverse=True):
                    report_text += f"  - {col}: {pct:.1f}%\n"
        
        return report_text

class AdvancedDataPreprocessor:
    """Advanced data preprocessing pipeline for debt collection ML"""
    
    def __init__(self, 
                 imputation_strategy: str = 'knn',
                 scaling_method: str = 'standard',
                 encoding_method: str = 'onehot',
                 handle_outliers: bool = True,
                 outlier_method: str = 'iqr'):
        
        self.imputation_strategy = imputation_strategy
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        
        # Pipeline components
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        self.numeric_features = None
        self.categorical_features = None
        
        # Quality checker
        self.quality_checker = DataQualityChecker()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze and report data quality"""
        
        self.logger.info("Analyzing data quality...")
        quality_report = self.quality_checker.assess_data_quality(df)
        
        print(self.quality_checker.generate_quality_report())
        
        return quality_report
    
    def detect_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Automatically detect numeric and categorical features"""
        
        # Exclude target and ID columns
        exclude_cols = ['Customer_ID', 'Outcome']
        available_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Numeric features
        numeric_features = []
        for col in available_cols:
            if df[col].dtype in ['int64', 'float64']:
                # Check if it's actually categorical (low cardinality)
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.05:  # More than 5% unique values
                    numeric_features.append(col)
        
        # Categorical features
        categorical_features = []
        for col in available_cols:
            if col not in numeric_features:
                categorical_features.append(col)
        
        self.logger.info(f"Detected {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
        
        return numeric_features, categorical_features
    
    def create_preprocessing_pipeline(self, 
                                    numeric_features: List[str], 
                                    categorical_features: List[str]) -> ColumnTransformer:
        """Create comprehensive preprocessing pipeline"""
        
        # Numeric preprocessing pipeline
        numeric_steps = []
        
        # Imputation
        if self.imputation_strategy == 'mean':
            numeric_steps.append(('imputer', SimpleImputer(strategy='mean')))
        elif self.imputation_strategy == 'median':
            numeric_steps.append(('imputer', SimpleImputer(strategy='median')))
        elif self.imputation_strategy == 'knn':
            numeric_steps.append(('imputer', KNNImputer(n_neighbors=5)))
        elif self.imputation_strategy == 'iterative':
            numeric_steps.append(('imputer', IterativeImputer(random_state=42)))
        
        # Outlier handling
        if self.handle_outliers:
            numeric_steps.append(('outlier_handler', OutlierHandler(method=self.outlier_method)))
        
        # Scaling
        if self.scaling_method == 'standard':
            numeric_steps.append(('scaler', StandardScaler()))
        elif self.scaling_method == 'robust':
            numeric_steps.append(('scaler', RobustScaler()))
        elif self.scaling_method == 'minmax':
            numeric_steps.append(('scaler', MinMaxScaler()))
        
        numeric_transformer = Pipeline(steps=numeric_steps)
        
        # Categorical preprocessing pipeline
        categorical_steps = []
        
        # Imputation
        categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        
        # Encoding
        if self.encoding_method == 'onehot':
            categorical_steps.append(('encoder', OneHotEncoder(
                drop='first', 
                sparse_output=False, 
                handle_unknown='ignore'
            )))
        elif self.encoding_method == 'ordinal':
            categorical_steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', 
                                                              unknown_value=-1)))
        
        categorical_transformer = Pipeline(steps=categorical_steps)
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def fit_transform(self, df: pd.DataFrame, target_column: str = 'Outcome') -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessor and transform data"""
        
        self.logger.info("Starting preprocessing pipeline...")
        
        # Analyze data quality
        self.analyze_data_quality(df)
        
        # Separate features and target
        if target_column in df.columns:
            X = df.drop([target_column], axis=1)
            y = df[target_column]
        else:
            X = df.copy()
            y = None
        
        # Remove Customer_ID if present
        if 'Customer_ID' in X.columns:
            X = X.drop(['Customer_ID'], axis=1)
        
        # Detect feature types
        self.numeric_features, self.categorical_features = self.detect_feature_types(X)
        
        # Create preprocessing pipeline
        self.preprocessor = self.create_preprocessing_pipeline(
            self.numeric_features, 
            self.categorical_features
        )
        
        # Fit and transform features
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names after preprocessing
        self.feature_names = self._get_feature_names()
        
        # Encode target variable if present
        if y is not None:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            self.logger.info(f"Target classes: {list(self.label_encoder.classes_)}")
        else:
            y_encoded = None
        
        self.logger.info(f"Preprocessing complete. Shape: {X_processed.shape}")
        self.logger.info(f"Feature names: {len(self.feature_names)} features")
        
        return X_processed, y_encoded
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor"""
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        # Remove Customer_ID and target if present
        X = df.copy()
        columns_to_drop = ['Customer_ID', 'Outcome']
        for col in columns_to_drop:
            if col in X.columns:
                X = X.drop([col], axis=1)
        
        # Transform
        X_processed = self.preprocessor.transform(X)
        
        return X_processed
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing"""
        
        feature_names = []
        
        # Numeric feature names
        feature_names.extend(self.numeric_features)
        
        # Categorical feature names
        if self.categorical_features:
            if self.encoding_method == 'onehot':
                cat_encoder = self.preprocessor.named_transformers_['cat']['encoder']
                cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_features)
                feature_names.extend(cat_feature_names)
            else:
                feature_names.extend(self.categorical_features)
        
        return feature_names
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about preprocessing steps"""
        
        return {
            'imputation_strategy': self.imputation_strategy,
            'scaling_method': self.scaling_method,
            'encoding_method': self.encoding_method,
            'handle_outliers': self.handle_outliers,
            'outlier_method': self.outlier_method,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'total_features_after_preprocessing': len(self.feature_names) if self.feature_names else 0,
            'target_classes': list(self.label_encoder.classes_) if self.label_encoder else None
        }
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor to file"""
        
        preprocessor_data = {
            'preprocessor': self.preprocessor,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'preprocessing_info': self.get_preprocessing_info()
        }
        
        joblib.dump(preprocessor_data, filepath)
        self.logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor from file"""
        
        preprocessor_data = joblib.load(filepath)
        
        self.preprocessor = preprocessor_data['preprocessor']
        self.label_encoder = preprocessor_data['label_encoder']
        self.feature_names = preprocessor_data['feature_names']
        self.numeric_features = preprocessor_data['numeric_features']
        self.categorical_features = preprocessor_data['categorical_features']
        
        self.logger.info(f"Preprocessor loaded from {filepath}")
