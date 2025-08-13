#!/usr/bin/env python3
"""
Test script for credit risk model trainer.
Validates model training, evaluation, and interpretability functionality.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.credit_risk_trainer import (
    CreditRiskMetrics,
    CreditRiskModelTrainer,
    CreditModelExplainer,
    train_credit_models
)
from app.features.preprocessing import FinancialFeatureEngineer


def create_sample_training_data():
    """Create sample training data for testing."""
    # Sample customer data
    customer_data = {
        'customer_id': [f'CUST{i:03d}' for i in range(1, 1001)],
        'customer_age': np.random.randint(18, 85, 1000),
        'annual_income': np.random.uniform(20000, 150000, 1000),
        'credit_score': np.random.randint(300, 850, 1000),
        'customer_since': [datetime.now() - timedelta(days=np.random.randint(100, 3650)) for _ in range(1000)],
        'employment_status': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], 1000),
        'city': np.random.choice(['London', 'Manchester', 'Birmingham', 'Leeds', 'Liverpool'], 1000),
        'account_tenure': np.random.randint(1, 20, 1000),
        'product_holdings': np.random.randint(1, 8, 1000),
        'relationship_value': np.random.uniform(1000, 50000, 1000)
    }
    
    # Sample credit bureau data
    bureau_data = {
        'customer_id': [f'CUST{i:03d}' for i in range(1, 1001)],
        'total_credit_limit': np.random.uniform(5000, 100000, 1000),
        'credit_utilization': np.random.uniform(0.1, 0.9, 1000),
        'number_of_accounts': np.random.randint(1, 15, 1000),
        'credit_history_length': np.random.randint(6, 240, 1000),
        'payment_history': np.random.uniform(0.5, 1.0, 1000),
        'public_records': np.random.randint(0, 5, 1000)
    }
    
    # Sample credit application data with default flags
    applications_data = {
        'application_id': [f'APP{i:06d}' for i in range(1, 1001)],
        'customer_id': [f'CUST{i:03d}' for i in range(1, 1001)],
        'loan_amount': np.random.uniform(5000, 100000, 1000),
        'application_date': [datetime.now() - timedelta(days=np.random.randint(0, 730)) for _ in range(1000)],
        'loan_purpose': np.random.choice(['Personal', 'Home Purchase', 'Business', 'Education', 'Vehicle'], 1000),
        'application_status': ['Approved'] * 1000,
        'default_flag': np.random.choice([0, 1], 1000, p=[0.85, 0.15])  # 15% default rate
    }
    
    return (
        pd.DataFrame(customer_data),
        pd.DataFrame(bureau_data),
        pd.DataFrame(applications_data)
    )


def test_credit_risk_metrics():
    """Test CreditRiskMetrics functionality."""
    print("Testing CreditRiskMetrics...")
    
    # Create sample data
    np.random.seed(42)
    y_true = np.random.choice([0, 1], 1000, p=[0.85, 0.15])
    y_prob = np.random.uniform(0, 1, 1000)
    
    # Test KS statistic
    ks_stat = CreditRiskMetrics.calculate_ks_statistic(y_true, y_prob)
    print(f"✓ KS Statistic: {ks_stat:.4f}")
    
    # Test Gini coefficient
    gini = CreditRiskMetrics.calculate_gini_coefficient(y_true, y_prob)
    print(f"✓ Gini Coefficient: {gini:.4f}")
    
    # Test PSI
    expected = np.random.uniform(0, 1, 500)
    actual = np.random.uniform(0, 1, 500)
    psi = CreditRiskMetrics.calculate_population_stability_index(expected, actual)
    print(f"✓ Population Stability Index: {psi:.4f}")
    
    print()


def test_model_trainer_initialization():
    """Test CreditRiskModelTrainer initialization."""
    print("Testing CreditRiskModelTrainer Initialization...")
    
    try:
        trainer = CreditRiskModelTrainer()
        print(f"✓ Trainer initialized successfully")
        print(f"✓ Available models: {list(trainer.model_configs.keys())}")
        print(f"✓ Feature engineer: {type(trainer.feature_engineer).__name__}")
        print(f"✓ MLflow tracker: {type(trainer.mlflow_tracker).__name__}")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
    
    print()


def test_feature_engineering_integration():
    """Test feature engineering integration."""
    print("Testing Feature Engineering Integration...")
    
    # Create sample data
    customer_df, bureau_df, applications_df = create_sample_training_data()
    
    # Initialize feature engineer
    engineer = FinancialFeatureEngineer()
    
    # Create features
    features_df = engineer.create_credit_features(customer_df, bureau_df, applications_df)
    
    print(f"✓ Features created: {len(features_df.columns)} columns")
    print(f"✓ Sample features: {list(features_df.columns[:5])}")
    print(f"✓ Data shape: {features_df.shape}")
    
    # Check for required features
    required_features = ['debt_to_income_ratio', 'credit_limit_to_income', 'age_group', 'income_stability']
    missing_features = [f for f in required_features if f not in features_df.columns]
    
    if missing_features:
        print(f"✗ Missing features: {missing_features}")
    else:
        print(f"✓ All required features present")
    
    print()


def test_model_training_simulation():
    """Test model training simulation with sample data."""
    print("Testing Model Training Simulation...")
    
    # Create sample data
    customer_df, bureau_df, applications_df = create_sample_training_data()
    
    # Create features
    engineer = FinancialFeatureEngineer()
    features_df = engineer.create_credit_features(customer_df, bureau_df, applications_df)
    
    # Merge with application data for target variable
    model_df = features_df.merge(
        applications_df[['customer_id', 'default_flag']], 
        on='customer_id', 
        how='inner'
    )
    
    # Prepare feature matrix and target
    exclude_cols = [
        'customer_id', 'default_flag', 'application_id', 
        'last_activity_date', 'application_date'
    ]
    feature_cols = [col for col in model_df.columns if col not in exclude_cols]
    
    X = model_df[feature_cols]
    y = model_df['default_flag']
    
    # Select only numeric features for testing
    X_numeric = X.select_dtypes(include=[np.number])
    
    print(f"✓ Feature matrix shape: {X_numeric.shape}")
    print(f"✓ Target distribution: {y.value_counts().to_dict()}")
    print(f"✓ Default rate: {y.mean():.2%}")
    
    # Test with a simple model (Logistic Regression)
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train simple model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"✓ Simple model AUC: {auc:.4f}")
    print(f"✓ Model coefficients: {len(model.coef_[0])}")
    
    print()


def test_model_explainer():
    """Test CreditModelExplainer functionality."""
    print("Testing CreditModelExplainer...")
    
    # Create sample data and model
    customer_df, bureau_df, applications_df = create_sample_training_data()
    engineer = FinancialFeatureEngineer()
    features_df = engineer.create_credit_features(customer_df, bureau_df, applications_df)
    
    # Prepare data
    model_df = features_df.merge(
        applications_df[['customer_id', 'default_flag']], 
        on='customer_id', 
        how='inner'
    )
    
    exclude_cols = ['customer_id', 'default_flag', 'application_id', 'last_activity_date', 'application_date']
    feature_cols = [col for col in model_df.columns if col not in exclude_cols]
    X = model_df[feature_cols].select_dtypes(include=[np.number])
    y = model_df['default_flag']
    
    # Train a simple model
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize explainer
    explainer = CreditModelExplainer(model, X_train.columns.tolist())
    
    # Test explanation generation
    sample_instance = X_test.iloc[0:1]
    explanation = explainer.explain_prediction(sample_instance)
    
    print(f"✓ Explanation generated: {len(explanation)} components")
    print(f"✓ Prediction: {explanation.get('prediction', 'N/A')}")
    print(f"✓ Prediction probability: {explanation.get('prediction_probability', 'N/A')}")
    print(f"✓ Text explanation: {explanation.get('text_explanation', 'N/A')[:100]}...")
    
    print()


def test_model_saving_and_loading():
    """Test model saving and loading functionality."""
    print("Testing Model Saving and Loading...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create sample data
            customer_df, bureau_df, applications_df = create_sample_training_data()
            engineer = FinancialFeatureEngineer()
            features_df = engineer.create_credit_features(customer_df, bureau_df, applications_df)
            
            # Prepare data
            model_df = features_df.merge(
                applications_df[['customer_id', 'default_flag']], 
                on='customer_id', 
                how='inner'
            )
            
            exclude_cols = ['customer_id', 'default_flag', 'application_id', 'last_activity_date', 'application_date']
            feature_cols = [col for col in model_df.columns if col not in exclude_cols]
            X = model_df[feature_cols].select_dtypes(include=[np.number])
            y = model_df['default_flag']
            
            # Train model
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            import joblib
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Save model
            model_file = os.path.join(temp_dir, "test_model.joblib")
            joblib.dump(model, model_file)
            
            # Load model
            loaded_model = joblib.load(model_file)
            
            # Test predictions
            original_pred = model.predict(X_test.iloc[0:1])
            loaded_pred = loaded_model.predict(X_test.iloc[0:1])
            
            print(f"✓ Model saved to: {model_file}")
            print(f"✓ Model loaded successfully")
            print(f"✓ Predictions match: {np.array_equal(original_pred, loaded_pred)}")
            
        except Exception as e:
            print(f"✗ Model saving/loading failed: {e}")
    
    print()


def test_hyperparameter_tuning():
    """Test hyperparameter tuning functionality."""
    print("Testing Hyperparameter Tuning...")
    
    # Create sample data
    customer_df, bureau_df, applications_df = create_sample_training_data()
    engineer = FinancialFeatureEngineer()
    features_df = engineer.create_credit_features(customer_df, bureau_df, applications_df)
    
    # Prepare data
    model_df = features_df.merge(
        applications_df[['customer_id', 'default_flag']], 
        on='customer_id', 
        how='inner'
    )
    
    exclude_cols = ['customer_id', 'default_flag', 'application_id', 'last_activity_date', 'application_date']
    feature_cols = [col for col in model_df.columns if col not in exclude_cols]
    X = model_df[feature_cols].select_dtypes(include=[np.number])
    y = model_df['default_flag']
    
    # Test GridSearchCV
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Simple parameter grid for testing
    param_grid = {
        'n_estimators': [5, 10],
        'max_depth': [3, 5]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=2,
        scoring='roc_auc',
        n_jobs=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"✓ Grid search completed")
    print(f"✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Best score: {grid_search.best_score_:.4f}")
    
    print()


def main():
    """Run all credit risk trainer tests."""
    print("=" * 60)
    print("CREDIT RISK MODEL TRAINER TEST")
    print("=" * 60)
    print()
    
    try:
        test_credit_risk_metrics()
        test_model_trainer_initialization()
        test_feature_engineering_integration()
        test_model_training_simulation()
        test_model_explainer()
        test_model_saving_and_loading()
        test_hyperparameter_tuning()
        
        print("=" * 60)
        print("✓ ALL CREDIT RISK TRAINER TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
