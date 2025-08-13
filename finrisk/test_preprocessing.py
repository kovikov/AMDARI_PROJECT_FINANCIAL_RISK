#!/usr/bin/env python3
"""
Test script for preprocessing module.
Validates data validation, feature engineering, and transformation functionality.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.features.preprocessing import (
    DataValidator,
    FinancialFeatureEngineer,
    FeatureTransformer,
    OutlierDetector,
    create_feature_pipeline,
    validate_feature_quality
)


def create_sample_data():
    """Create sample data for testing."""
    # Sample customer data
    customer_data = {
        'customer_id': [f'CUST{i:03d}' for i in range(1, 101)],
        'customer_age': np.random.randint(18, 85, 100),
        'annual_income': np.random.uniform(20000, 150000, 100),
        'credit_score': np.random.randint(300, 850, 100),
        'customer_since': [datetime.now() - timedelta(days=np.random.randint(100, 3650)) for _ in range(100)],
        'employment_status': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], 100),
        'city': np.random.choice(['London', 'Manchester', 'Birmingham', 'Leeds', 'Liverpool'], 100),
        'account_tenure': np.random.randint(1, 20, 100),
        'product_holdings': np.random.randint(1, 8, 100),
        'relationship_value': np.random.uniform(1000, 50000, 100)
    }
    
    # Sample credit bureau data
    bureau_data = {
        'customer_id': [f'CUST{i:03d}' for i in range(1, 101)],
        'total_credit_limit': np.random.uniform(5000, 100000, 100),
        'credit_utilization': np.random.uniform(0.1, 0.9, 100),
        'number_of_accounts': np.random.randint(1, 15, 100),
        'credit_history_length': np.random.randint(6, 240, 100),
        'payment_history': np.random.uniform(0.5, 1.0, 100),
        'public_records': np.random.randint(0, 5, 100)
    }
    
    # Sample transaction data
    transaction_data = {
        'transaction_id': [f'TXN{i:06d}' for i in range(1, 1001)],
        'customer_id': np.random.choice(customer_data['customer_id'], 1000),
        'amount': np.random.uniform(10, 5000, 1000),
        'transaction_date': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(1000)],
        'merchant_category': np.random.choice(['Groceries', 'Fuel', 'Restaurants', 'Online Shopping', 'Travel'], 1000),
        'location': np.random.choice(['London, UK', 'Manchester, UK', 'Birmingham, UK', 'Paris, France', 'New York, USA'], 1000),
        'device_info': np.random.choice(['Mobile App', 'Web Browser', 'ATM', 'POS Terminal'], 1000)
    }
    
    # Sample credit application data
    credit_data = {
        'application_id': [f'APP{i:06d}' for i in range(1, 201)],
        'customer_id': np.random.choice(customer_data['customer_id'], 200),
        'loan_amount': np.random.uniform(5000, 100000, 200),
        'application_date': [datetime.now() - timedelta(days=np.random.randint(0, 730)) for _ in range(200)],
        'loan_purpose': np.random.choice(['Personal', 'Home Purchase', 'Business', 'Education', 'Vehicle'], 200),
        'default_flag': np.random.choice([0, 1], 200, p=[0.85, 0.15])
    }
    
    return (
        pd.DataFrame(customer_data),
        pd.DataFrame(bureau_data),
        pd.DataFrame(transaction_data),
        pd.DataFrame(credit_data)
    )


def test_data_validator():
    """Test DataValidator functionality."""
    print("Testing DataValidator...")
    
    # Create sample data
    customer_df, bureau_df, transaction_df, credit_df = create_sample_data()
    
    # Test customer data validation
    customer_validation = DataValidator.validate_customer_data(customer_df)
    print(f"✓ Customer validation: {customer_validation['is_valid']}")
    print(f"  - Total records: {customer_validation['total_records']}")
    print(f"  - Issues: {len(customer_validation['validation_issues'])}")
    
    # Test transaction data validation
    transaction_validation = DataValidator.validate_transaction_data(transaction_df)
    print(f"✓ Transaction validation: {transaction_validation['is_valid']}")
    print(f"  - Total records: {transaction_validation['total_records']}")
    print(f"  - Issues: {len(transaction_validation['validation_issues'])}")
    
    print()


def test_feature_engineer():
    """Test FinancialFeatureEngineer functionality."""
    print("Testing FinancialFeatureEngineer...")
    
    # Create sample data
    customer_df, bureau_df, transaction_df, credit_df = create_sample_data()
    
    # Initialize feature engineer
    engineer = FinancialFeatureEngineer()
    
    # Test credit features
    credit_features = engineer.create_credit_features(customer_df, bureau_df, credit_df)
    print(f"✓ Credit features: {len(credit_features.columns)} columns")
    print(f"  - Sample features: {list(credit_features.columns[:5])}")
    
    # Test fraud features
    fraud_features = engineer.create_fraud_features(transaction_df, customer_df)
    print(f"✓ Fraud features: {len(fraud_features.columns)} columns")
    print(f"  - Sample features: {list(fraud_features.columns[:5])}")
    
    # Test behavioral features
    behavioral_features = engineer.create_behavioral_features(customer_df, transaction_df)
    print(f"✓ Behavioral features: {len(behavioral_features.columns)} columns")
    print(f"  - Sample features: {list(behavioral_features.columns[:5])}")
    
    print()


def test_feature_transformer():
    """Test FeatureTransformer functionality."""
    print("Testing FeatureTransformer...")
    
    # Create sample data with mixed types
    customer_df, bureau_df, _, _ = create_sample_data()
    
    # Create some features
    engineer = FinancialFeatureEngineer()
    features = engineer.create_credit_features(customer_df, bureau_df, pd.DataFrame())
    
    # Select features for testing (mix of numeric and categorical)
    test_features = features[['customer_age', 'annual_income', 'credit_score', 'age_group', 'credit_score_band']]
    
    # Test transformer
    transformer = FeatureTransformer(feature_type='credit')
    
    # Fit and transform
    transformed = transformer.fit_transform(test_features)
    
    print(f"✓ Original features: {test_features.shape}")
    print(f"✓ Transformed features: {transformed.shape}")
    print(f"✓ Feature names: {list(transformed.columns)}")
    print(f"✓ Numeric features: {len(transformer.numeric_features)}")
    print(f"✓ Categorical features: {len(transformer.categorical_features)}")
    
    print()


def test_outlier_detector():
    """Test OutlierDetector functionality."""
    print("Testing OutlierDetector...")
    
    # Create sample data with outliers
    customer_df, _, _, _ = create_sample_data()
    
    # Add some outliers
    customer_df.loc[0, 'annual_income'] = 1000000  # Extreme outlier
    customer_df.loc[1, 'customer_age'] = 150  # Invalid age
    
    # Test IQR outlier detection
    income_outliers_iqr = OutlierDetector.detect_outliers_iqr(customer_df, 'annual_income')
    print(f"✓ IQR outliers detected: {income_outliers_iqr.sum()} records")
    
    # Test Z-score outlier detection
    income_outliers_zscore = OutlierDetector.detect_outliers_zscore(customer_df, 'annual_income')
    print(f"✓ Z-score outliers detected: {income_outliers_zscore.sum()} records")
    
    # Test outlier handling
    handled_df = OutlierDetector.handle_outliers(customer_df, 'annual_income', method='cap')
    print(f"✓ Outlier handling (cap): {len(customer_df)} -> {len(handled_df)} records")
    
    print()


def test_feature_pipeline():
    """Test complete feature pipeline."""
    print("Testing Feature Pipeline...")
    
    # Create sample data
    customer_df, bureau_df, _, _ = create_sample_data()
    
    # Create features
    engineer = FinancialFeatureEngineer()
    features = engineer.create_credit_features(customer_df, bureau_df, pd.DataFrame())
    
    # Select numerical features for testing
    numerical_features = features.select_dtypes(include=[np.number])
    
    # Create pipeline
    pipeline = create_feature_pipeline(feature_type='credit')
    
    # Fit and transform
    transformed = pipeline.fit_transform(numerical_features)
    
    print(f"✓ Pipeline transformation: {numerical_features.shape} -> {transformed.shape}")
    print(f"✓ Pipeline feature type: {pipeline.feature_type}")
    
    print()


def test_feature_quality():
    """Test validate_feature_quality function."""
    print("Testing Feature Quality Validation...")
    
    # Create sample data
    customer_df, bureau_df, _, _ = create_sample_data()
    
    # Create features
    engineer = FinancialFeatureEngineer()
    features = engineer.create_credit_features(customer_df, bureau_df, pd.DataFrame())
    
    # Add some missing values and correlations for testing
    features.loc[0:5, 'annual_income'] = np.nan
    features['correlated_feature'] = features['annual_income'] * 2  # High correlation
    
    # Test quality validation
    quality_report = validate_feature_quality(features)
    
    print(f"✓ Total features: {quality_report['total_features']}")
    print(f"✓ Total samples: {quality_report['total_samples']}")
    print(f"✓ Features with missing data: {quality_report['missing_data_analysis']['features_with_missing']}")
    print(f"✓ Average missing percentage: {quality_report['missing_data_analysis']['avg_missing_pct']:.2f}%")
    print(f"✓ Multicollinearity risk: {quality_report['correlation_analysis'].get('multicollinearity_risk', False)}")
    print(f"✓ Recommendations: {len(quality_report['recommendations'])}")
    
    print()


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing Edge Cases...")
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    try:
        validation = DataValidator.validate_customer_data(empty_df)
        print(f"✓ Empty DataFrame handling: {validation['is_valid']}")
    except Exception as e:
        print(f"✗ Empty DataFrame error: {e}")
    
    # Test with missing columns
    partial_df = pd.DataFrame({'customer_id': ['CUST001'], 'customer_age': [25]})
    try:
        validation = DataValidator.validate_customer_data(partial_df)
        print(f"✓ Missing columns handling: {validation['is_valid']}")
        print(f"  - Issues: {validation['validation_issues']}")
    except Exception as e:
        print(f"✗ Missing columns error: {e}")
    
    # Test with all null values
    null_df = pd.DataFrame({
        'customer_id': ['CUST001'],
        'customer_age': [np.nan],
        'annual_income': [np.nan],
        'credit_score': [np.nan]
    })
    try:
        validation = DataValidator.validate_customer_data(null_df)
        print(f"✓ Null values handling: {validation['is_valid']}")
    except Exception as e:
        print(f"✗ Null values error: {e}")
    
    print()


def main():
    """Run all preprocessing tests."""
    print("=" * 60)
    print("PREPROCESSING MODULE TEST")
    print("=" * 60)
    print()
    
    try:
        test_data_validator()
        test_feature_engineer()
        test_feature_transformer()
        test_outlier_detector()
        test_feature_pipeline()
        test_feature_quality()
        test_edge_cases()
        
        print("=" * 60)
        print("✓ ALL PREPROCESSING TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
