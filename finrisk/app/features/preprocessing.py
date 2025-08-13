"""
Feature engineering and preprocessing module for FinRisk application.
Handles data cleaning, feature creation, and transformation pipelines.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import logging

# Configure logging
logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation and quality checks."""
    
    @staticmethod
    def validate_customer_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate customer profile data quality.
        
        Args:
            df: Customer profiles DataFrame
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        # Required columns
        required_cols = ['customer_id', 'customer_age', 'annual_income', 'credit_score']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Data quality checks
        if 'customer_age' in df.columns:
            invalid_ages = df[(df['customer_age'] < 18) | (df['customer_age'] > 120)]
            if len(invalid_ages) > 0:
                issues.append(f"Invalid ages found: {len(invalid_ages)} records")
        
        if 'credit_score' in df.columns:
            invalid_scores = df[(df['credit_score'] < 300) | (df['credit_score'] > 850)]
            if len(invalid_scores) > 0:
                issues.append(f"Invalid credit scores: {len(invalid_scores)} records")
        
        if 'annual_income' in df.columns:
            negative_income = df[df['annual_income'] < 0]
            if len(negative_income) > 0:
                issues.append(f"Negative income values: {len(negative_income)} records")
        
        # Missing value analysis
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 10]
        
        return {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'duplicate_records': df.duplicated().sum(),
            'missing_data_pct': missing_pct.to_dict(),
            'high_missing_columns': high_missing.to_dict(),
            'validation_issues': issues,
            'is_valid': len(issues) == 0
        }
    
    @staticmethod
    def validate_transaction_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate transaction data quality.
        
        Args:
            df: Transaction data DataFrame
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        # Required columns
        required_cols = ['transaction_id', 'customer_id', 'amount', 'transaction_date']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Amount validation
        if 'amount' in df.columns:
            negative_amounts = df[df['amount'] <= 0]
            if len(negative_amounts) > 0:
                issues.append(f"Non-positive amounts: {len(negative_amounts)} records")
            
            extreme_amounts = df[df['amount'] > 100000]
            if len(extreme_amounts) > 0:
                issues.append(f"Extreme amounts (>£100K): {len(extreme_amounts)} records")
        
        # Date validation
        if 'transaction_date' in df.columns:
            try:
                df['transaction_date'] = pd.to_datetime(df['transaction_date'])
                future_dates = df[df['transaction_date'] > datetime.now()]
                if len(future_dates) > 0:
                    issues.append(f"Future transaction dates: {len(future_dates)} records")
            except Exception as e:
                issues.append(f"Invalid transaction dates: {str(e)}")
        
        return {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'duplicate_records': df.duplicated().sum(),
            'validation_issues': issues,
            'is_valid': len(issues) == 0
        }


class FinancialFeatureEngineer:
    """Financial feature engineering for credit risk and fraud detection."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
    
    def create_credit_features(self, customer_df: pd.DataFrame, 
                             bureau_df: pd.DataFrame,
                             application_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive credit risk features.
        
        Args:
            customer_df: Customer profiles data
            bureau_df: Credit bureau data
            application_df: Credit applications data
            
        Returns:
            DataFrame with engineered credit features
        """
        logger.info("Creating credit risk features...")
        
        # Merge datasets
        features_df = customer_df.merge(bureau_df, on='customer_id', how='left', suffixes=('', '_bureau'))
        
        # Basic ratios
        features_df['debt_to_income_ratio'] = (
            features_df['total_credit_limit'] * features_df['credit_utilization'] / 
            features_df['annual_income'].replace(0, 1)
        )
        
        features_df['credit_limit_to_income'] = (
            features_df['total_credit_limit'] / features_df['annual_income'].replace(0, 1)
        )
        
        # Age-based features
        features_df['age_group'] = pd.cut(
            features_df['customer_age'], 
            bins=[0, 25, 35, 50, 65, 100], 
            labels=['young', 'adult', 'middle_age', 'senior', 'elderly']
        )
        
        # Income stability
        features_df['income_stability'] = np.where(
            features_df['employment_status'].isin(['Full-time', 'Self-employed']), 1.0,
            np.where(features_df['employment_status'] == 'Part-time', 0.7, 0.3)
        )
        
        # Credit mix score
        features_df['credit_mix_score'] = np.minimum(
            features_df['number_of_accounts'] / 5.0, 1.0
        )
        
        # Payment behavior score
        features_df['payment_behavior_score'] = (
            features_df['payment_history'] * 0.6 +
            (1 - features_df['credit_utilization']) * 0.4
        )
        
        # Risk indicators
        features_df['high_utilization'] = (features_df['credit_utilization'] > 0.8).astype(int)
        features_df['short_history'] = (features_df['credit_history_length'] < 3).astype(int)
        features_df['young_customer'] = (features_df['customer_age'] < 25).astype(int)
        features_df['low_income'] = (features_df['annual_income'] < 25000).astype(int)
        
        # Relationship value per year
        features_df['relationship_value_per_year'] = (
            features_df['relationship_value'] / 
            np.maximum(features_df['account_tenure'], 1)
        )
        
        # Credit score bands
        features_df['credit_score_band'] = pd.cut(
            features_df['credit_score'],
            bins=[0, 580, 670, 740, 800, 850],
            labels=['poor', 'fair', 'good', 'very_good', 'excellent']
        )
        
        logger.info(f"Created {len(features_df.columns)} credit features for {len(features_df)} customers")
        return features_df
    
    def create_fraud_features(self, transaction_df: pd.DataFrame,
                            customer_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive fraud detection features.
        
        Args:
            transaction_df: Transaction data
            customer_df: Customer profiles data
            
        Returns:
            DataFrame with engineered fraud features
        """
        logger.info("Creating fraud detection features...")
        
        # Convert transaction_date to datetime
        transaction_df['transaction_date'] = pd.to_datetime(transaction_df['transaction_date'])
        
        # Time-based features
        transaction_df['hour'] = transaction_df['transaction_date'].dt.hour
        transaction_df['day_of_week'] = transaction_df['transaction_date'].dt.dayofweek
        transaction_df['is_weekend'] = (transaction_df['day_of_week'] >= 5).astype(int)
        transaction_df['is_night'] = ((transaction_df['hour'] < 6) | (transaction_df['hour'] > 22)).astype(int)
        
        # Amount-based features
        transaction_df['amount_log'] = np.log1p(transaction_df['amount'])
        transaction_df['is_round_amount'] = (transaction_df['amount'] % 10 == 0).astype(int)
        
        # Customer-level aggregations
        customer_stats = transaction_df.groupby('customer_id').agg({
            'amount': ['mean', 'std', 'median', 'count'],
            'merchant_category': 'nunique',
            'location': 'nunique',
            'device_info': 'nunique'
        }).round(2)
        
        # Flatten column names
        customer_stats.columns = ['_'.join(col).strip() for col in customer_stats.columns]
        customer_stats = customer_stats.reset_index()
        
        # Merge customer stats back to transactions
        transaction_df = transaction_df.merge(customer_stats, on='customer_id', how='left')
        
        # Deviation from normal behavior
        transaction_df['amount_zscore'] = (
            (transaction_df['amount'] - transaction_df['amount_mean']) / 
            (transaction_df['amount_std'] + 1e-6)
        )
        
        # Velocity features (transactions per time window)
        transaction_df = transaction_df.sort_values(['customer_id', 'transaction_date'])
        
        # Time since last transaction
        transaction_df['time_diff'] = transaction_df.groupby('customer_id')['transaction_date'].diff()
        transaction_df['hours_since_last'] = transaction_df['time_diff'].dt.total_seconds() / 3600
        
        # Rolling features (last 24 hours)
        transaction_df['rolling_24h_count'] = (
            transaction_df.groupby('customer_id')
            .rolling('24h', on='transaction_date')['amount']
            .count()
            .reset_index(0, drop=True)
        )
        
        transaction_df['rolling_24h_sum'] = (
            transaction_df.groupby('customer_id')
            .rolling('24h', on='transaction_date')['amount']
            .sum()
            .reset_index(0, drop=True)
        )
        
        # Location and merchant patterns
        location_freq = transaction_df.groupby(['customer_id', 'location']).size().reset_index(name='location_freq')
        merchant_freq = transaction_df.groupby(['customer_id', 'merchant_category']).size().reset_index(name='merchant_freq')
        
        transaction_df = transaction_df.merge(
            location_freq, on=['customer_id', 'location'], how='left'
        ).merge(
            merchant_freq, on=['customer_id', 'merchant_category'], how='left'
        )
        
        # Anomaly indicators
        transaction_df['new_location'] = (transaction_df['location_freq'] == 1).astype(int)
        transaction_df['rare_merchant'] = (transaction_df['merchant_freq'] <= 2).astype(int)
        
        # High-risk indicators
        high_risk_merchants = ['Online Shopping', 'ATM', 'Transfer']
        transaction_df['high_risk_merchant'] = (
            transaction_df['merchant_category'].isin(high_risk_merchants)
        ).astype(int)
        
        logger.info(f"Created fraud features for {len(transaction_df)} transactions")
        return transaction_df
    
    def create_behavioral_features(self, customer_df: pd.DataFrame,
                                 transaction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral and relationship features.
        
        Args:
            customer_df: Customer profiles data
            transaction_df: Transaction data
            
        Returns:
            DataFrame with behavioral features
        """
        logger.info("Creating behavioral features...")
        
        # Transaction patterns by customer
        transaction_patterns = transaction_df.groupby('customer_id').agg({
            'amount': ['count', 'mean', 'std', 'min', 'max'],
            'merchant_category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
            'transaction_date': ['min', 'max']
        })
        
        # Flatten columns
        transaction_patterns.columns = ['_'.join(col).strip() for col in transaction_patterns.columns]
        transaction_patterns = transaction_patterns.reset_index()
        
        # Calculate transaction span
        transaction_patterns['transaction_span_days'] = (
            pd.to_datetime(transaction_patterns['transaction_date_max']) - 
            pd.to_datetime(transaction_patterns['transaction_date_min'])
        ).dt.days
        
        # Transaction frequency
        transaction_patterns['transactions_per_day'] = (
            transaction_patterns['amount_count'] / 
            np.maximum(transaction_patterns['transaction_span_days'], 1)
        )
        
        # Merge with customer data
        behavioral_df = customer_df.merge(transaction_patterns, on='customer_id', how='left')
        
        # Customer lifecycle features
        behavioral_df['products_per_tenure'] = (
            behavioral_df['product_holdings'] / 
            np.maximum(behavioral_df['account_tenure'], 1)
        )
        
        # Engagement score
        behavioral_df['engagement_score'] = (
            behavioral_df['transactions_per_day'].fillna(0) * 0.3 +
            behavioral_df['products_per_tenure'] * 0.3 +
            (behavioral_df['relationship_value'] / behavioral_df['annual_income'].replace(0, 1)) * 0.4
        )
        
        # Risk behavior indicators
        behavioral_df['high_transaction_variance'] = (
            behavioral_df['amount_std'] > behavioral_df['amount_mean']
        ).astype(int)
        
        logger.info(f"Created behavioral features for {len(behavioral_df)} customers")
        return behavioral_df


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature preprocessing."""
    
    def __init__(self, feature_type='credit'):
        self.feature_type = feature_type
        self.numeric_features = []
        self.categorical_features = []
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer on training data.
        
        Args:
            X: Feature matrix
            y: Target variable (unused)
            
        Returns:
            self
        """
        # Identify feature types
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Fit scalers for numeric features
        for feature in self.numeric_features:
            scaler = StandardScaler()
            scaler.fit(X[[feature]])
            self.scalers[feature] = scaler
            
            # Store feature statistics
            self.feature_stats[feature] = {
                'mean': X[feature].mean(),
                'std': X[feature].std(),
                'min': X[feature].min(),
                'max': X[feature].max(),
                'missing_pct': X[feature].isnull().mean()
            }
        
        # Fit encoders for categorical features
        for feature in self.categorical_features:
            encoder = LabelEncoder()
            # Handle categorical columns properly
            if X[feature].dtype.name == 'category':
                # Convert to string first to handle missing values
                feature_values = X[feature].astype(str).fillna('unknown')
            else:
                feature_values = X[feature].fillna('unknown')
            encoder.fit(feature_values)
            self.encoders[feature] = encoder
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted transformers.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Transformed feature matrix
        """
        X_transformed = X.copy()
        
        # Transform numeric features
        for feature in self.numeric_features:
            if feature in X_transformed.columns:
                # Handle missing values
                X_transformed[feature] = X_transformed[feature].fillna(
                    self.feature_stats[feature]['mean']
                )
                
                # Scale features
                X_transformed[feature] = self.scalers[feature].transform(
                    X_transformed[[feature]]
                ).flatten()
        
        # Transform categorical features
        for feature in self.categorical_features:
            if feature in X_transformed.columns:
                # Handle categorical columns properly
                if X_transformed[feature].dtype.name == 'category':
                    # Convert to string first to handle missing values
                    feature_values = X_transformed[feature].astype(str).fillna('unknown')
                else:
                    feature_values = X_transformed[feature].fillna('unknown')
                
                # Handle unseen categories
                known_categories = set(self.encoders[feature].classes_)
                feature_values = feature_values.apply(
                    lambda x: x if x in known_categories else 'unknown'
                )
                
                # Encode categories
                X_transformed[feature] = self.encoders[feature].transform(feature_values)
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after transformation."""
        return self.numeric_features + self.categorical_features
    
    def get_feature_importance_data(self) -> Dict[str, Any]:
        """Get feature statistics for importance analysis."""
        return {
            'feature_stats': self.feature_stats,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'total_features': len(self.numeric_features) + len(self.categorical_features)
        }


class OutlierDetector:
    """Outlier detection and handling."""
    
    @staticmethod
    def detect_outliers_iqr(df: pd.DataFrame, feature: str, 
                           multiplier: float = 1.5) -> pd.Series:
        """
        Detect outliers using IQR method.
        
        Args:
            df: DataFrame
            feature: Feature column name
            multiplier: IQR multiplier
            
        Returns:
            Boolean series indicating outliers
        """
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return (df[feature] < lower_bound) | (df[feature] > upper_bound)
    
    @staticmethod
    def detect_outliers_zscore(df: pd.DataFrame, feature: str, 
                              threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers using Z-score method.
        
        Args:
            df: DataFrame
            feature: Feature column name
            threshold: Z-score threshold
            
        Returns:
            Boolean series indicating outliers
        """
        z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
        return z_scores > threshold
    
    @staticmethod
    def handle_outliers(df: pd.DataFrame, feature: str, 
                       method: str = 'cap') -> pd.DataFrame:
        """
        Handle outliers in a feature.
        
        Args:
            df: DataFrame
            feature: Feature column name
            method: Method to handle outliers ('cap', 'remove', 'transform')
            
        Returns:
            DataFrame with outliers handled
        """
        df_handled = df.copy()
        
        if method == 'cap':
            # Cap outliers at 99th percentile
            upper_cap = df[feature].quantile(0.99)
            lower_cap = df[feature].quantile(0.01)
            df_handled[feature] = df_handled[feature].clip(lower_cap, upper_cap)
            
        elif method == 'remove':
            # Remove outlier rows
            outliers = OutlierDetector.detect_outliers_iqr(df, feature)
            df_handled = df_handled[~outliers]
            
        elif method == 'transform':
            # Log transform for positive values
            if df[feature].min() > 0:
                df_handled[feature] = np.log1p(df[feature])
        
        return df_handled


def create_feature_pipeline(feature_type: str = 'credit') -> FeatureTransformer:
    """
    Create a complete feature engineering pipeline.
    
    Args:
        feature_type: Type of features ('credit' or 'fraud')
        
    Returns:
        Configured FeatureTransformer
    """
    return FeatureTransformer(feature_type=feature_type)


def validate_feature_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate feature quality and provide recommendations.
    
    Args:
        df: Feature DataFrame
        
    Returns:
        Dictionary with quality metrics and recommendations
    """
    quality_report = {
        'total_features': len(df.columns),
        'total_samples': len(df),
        'missing_data_analysis': {},
        'correlation_analysis': {},
        'recommendations': []
    }
    
    # Missing data analysis
    missing_pct = (df.isnull().sum() / len(df)) * 100
    quality_report['missing_data_analysis'] = {
        'features_with_missing': (missing_pct > 0).sum(),
        'high_missing_features': missing_pct[missing_pct > 20].to_dict(),
        'avg_missing_pct': missing_pct.mean()
    }
    
    # Correlation analysis for numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.9:
                    high_corr_pairs.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        quality_report['correlation_analysis'] = {
            'high_correlation_pairs': high_corr_pairs,
            'multicollinearity_risk': len(high_corr_pairs) > 0
        }
    
    # Generate recommendations
    if quality_report['missing_data_analysis']['avg_missing_pct'] > 15:
        quality_report['recommendations'].append("High missing data percentage - consider imputation strategies")
    
    if quality_report['correlation_analysis'].get('multicollinearity_risk', False):
        quality_report['recommendations'].append("High correlation detected - consider feature selection")
    
    if len(df.columns) > 100:
        quality_report['recommendations'].append("High feature count - consider dimensionality reduction")
    
    return quality_report
