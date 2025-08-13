"""
Tests for the credit risk model training module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from app.models.credit_risk_trainer import (
    CreditRiskModelTrainer,
    CreditModelExplainer,
    CreditRiskMetrics,
    train_credit_models
)
from app.features.preprocessing import FinancialFeatureEngineer


class TestCreditRiskMetrics:
    """Test credit risk specific metrics."""
    
    def test_ks_statistic(self):
        """Test KS statistic calculation."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])
        
        ks = CreditRiskMetrics.calculate_ks_statistic(y_true, y_prob)
        
        assert isinstance(ks, float)
        assert 0 <= ks <= 1
        assert ks > 0  # Should be positive for good separation
    
    def test_gini_coefficient(self):
        """Test Gini coefficient calculation."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])
        
        gini = CreditRiskMetrics.calculate_gini_coefficient(y_true, y_prob)
        
        assert isinstance(gini, float)
        assert -1 <= gini <= 1
        assert gini > 0  # Should be positive for good separation
    
    def test_population_stability_index(self):
        """Test PSI calculation."""
        expected = np.random.random(1000)
        actual = expected + np.random.normal(0, 0.1, 1000)
        
        psi = CreditRiskMetrics.calculate_population_stability_index(expected, actual)
        
        assert isinstance(psi, float)
        assert psi >= 0  # PSI should be non-negative


class TestFinancialFeatureEngineer:
    """Test feature engineering functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.engineer = FinancialFeatureEngineer()
        
        # Create sample customer data
        self.customer_df = pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003'],
            'customer_age': [25, 35, 45],
            'annual_income': [50000, 75000, 100000],
            'employment_status': ['Full-time', 'Part-time', 'Self-employed'],
            'customer_since': ['2020-01-01', '2018-06-01', '2015-03-01'],
            'employment_start_date': ['2020-02-01', '2018-07-01', '2015-04-01'],
            'city': ['New York', 'Los Angeles', 'Chicago'],
            'credit_limit': [10000, 15000, 20000],
            'current_balance': [2000, 5000, 8000]
        })
        
        # Create sample bureau data
        self.bureau_df = pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003'],
            'credit_score': [650, 720, 780],
            'payment_history': ['000000', '000000', '000000'],
            'total_accounts': [5, 8, 12],
            'credit_age_months': [24, 48, 72],
            'inquiries_6m': [2, 1, 0]
        })
        
        # Create sample application data
        self.applications_df = pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003'],
            'loan_amount': [5000, 10000, 15000],
            'application_date': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'loan_purpose': ['Personal', 'Business', 'Home Improvement'],
            'default_flag': [0, 1, 0]
        })
    
    def test_create_customer_features(self):
        """Test customer feature creation."""
        features = self.engineer._create_customer_features(self.customer_df)
        
        assert 'age_group' in features.columns
        assert 'income_category' in features.columns
        assert 'employment_duration_years' in features.columns
        assert 'city_category' in features.columns
        assert 'credit_utilization' in features.columns
    
    def test_create_bureau_features(self):
        """Test bureau feature creation."""
        features = self.engineer._create_bureau_features(self.bureau_df)
        
        assert 'credit_score_category' in features.columns
        assert 'account_diversity' in features.columns
        assert 'inquiry_intensity' in features.columns
    
    def test_create_application_features(self):
        """Test application feature creation."""
        features = self.engineer._create_application_features(self.applications_df)
        
        assert 'total_applications' in features.columns
        assert 'avg_loan_amount' in features.columns
        assert 'default_rate' in features.columns
    
    def test_create_credit_features_integration(self):
        """Test complete feature engineering pipeline."""
        features_df = self.engineer.create_credit_features(
            self.customer_df, self.bureau_df, self.applications_df
        )
        
        assert not features_df.empty
        assert len(features_df) == 3  # Should have 3 customers
        assert 'customer_id' in features_df.columns


class TestCreditRiskModelTrainer:
    """Test credit risk model trainer."""
    
    def setup_method(self):
        """Setup test data."""
        self.trainer = CreditRiskModelTrainer()
        
        # Create sample training data
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        self.X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y = pd.Series(np.random.binomial(1, 0.3, n_samples))
    
    def test_model_configs(self):
        """Test model configurations."""
        assert 'logistic_regression' in self.trainer.model_configs
        assert 'random_forest' in self.trainer.model_configs
        assert 'xgboost' in self.trainer.model_configs
        
        # Check that each model has required keys
        for model_name, config in self.trainer.model_configs.items():
            assert 'model' in config
            assert 'params' in config
            assert 'search_params' in config
    
    @patch('app.models.credit_risk_trainer.read_sql_query')
    def test_load_and_prepare_data(self, mock_read_sql):
        """Test data loading and preparation."""
        # Mock database responses
        mock_read_sql.side_effect = [
            pd.DataFrame({'customer_id': ['C001'], 'customer_age': [30]}),
            pd.DataFrame({'customer_id': ['C001'], 'credit_score': [700]}),
            pd.DataFrame({'customer_id': ['C001'], 'default_flag': [0]})
        ]
        
        X, y = self.trainer.load_and_prepare_data()
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
    
    def test_train_model(self):
        """Test individual model training."""
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        
        # Test training without hyperparameter tuning
        result = self.trainer.train_model(
            'logistic_regression', X_train, y_train, X_val, y_val, 
            hyperparameter_tuning=False
        )
        
        assert 'model' in result
        assert 'metrics' in result
        assert 'training_time' in result
        assert 'feature_names' in result
        
        # Check metrics
        metrics = result['metrics']
        assert 'auc' in metrics
        assert 'gini' in metrics
        assert 'ks_statistic' in metrics
        assert 0 <= metrics['auc'] <= 1
        assert -1 <= metrics['gini'] <= 1
        assert 0 <= metrics['ks_statistic'] <= 1
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        from sklearn.linear_model import LogisticRegression
        
        # Train a simple model
        model = LogisticRegression(random_state=42)
        model.fit(self.X, self.y)
        
        # Evaluate
        metrics = self.trainer._evaluate_model(model, self.X, self.y, 'test_model')
        
        assert 'auc' in metrics
        assert 'gini' in metrics
        assert 'ks_statistic' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'accuracy' in metrics


class TestCreditModelExplainer:
    """Test model explainability functionality."""
    
    def setup_method(self):
        """Setup test data."""
        from sklearn.linear_model import LogisticRegression
        
        # Create a simple model
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.binomial(1, 0.3, 100))
        
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X, y)
        self.feature_names = X.columns.tolist()
        self.X_sample = X.head(10)
        
        self.explainer = CreditModelExplainer(self.model, self.feature_names)
    
    def test_initialization(self):
        """Test explainer initialization."""
        assert self.explainer.model is not None
        assert self.explainer.feature_names == self.feature_names
        assert self.explainer.shap_explainer is None
        assert self.explainer.lime_explainer is None
    
    def test_explain_prediction(self):
        """Test prediction explanation."""
        X_instance = self.X_sample.head(1)
        
        explanation = self.explainer.explain_prediction(X_instance)
        
        assert 'prediction' in explanation
        assert 'prediction_probability' in explanation
        assert 'text_explanation' in explanation
        assert isinstance(explanation['prediction'], int)
        assert explanation['prediction'] in [0, 1]
    
    def test_generate_text_explanation(self):
        """Test text explanation generation."""
        explanations = {
            'prediction_probability': {'default': 0.75, 'no_default': 0.25},
            'shap_values': {'feature_0': 0.1, 'feature_1': -0.05}
        }
        
        text = self.explainer._generate_text_explanation(explanations)
        
        assert isinstance(text, str)
        assert 'Default probability' in text
        assert 'HIGH RISK' in text or 'MEDIUM RISK' in text or 'LOW RISK' in text


class TestIntegration:
    """Integration tests."""
    
    @patch('app.models.credit_risk_trainer.read_sql_query')
    def test_train_credit_models_function(self, mock_read_sql):
        """Test the main training function."""
        # Mock database responses
        mock_read_sql.side_effect = [
            pd.DataFrame({'customer_id': ['C001'], 'customer_age': [30]}),
            pd.DataFrame({'customer_id': ['C001'], 'credit_score': [700]}),
            pd.DataFrame({'customer_id': ['C001'], 'default_flag': [0]})
        ]
        
        # Test with minimal data (this will likely fail due to insufficient data,
        # but we can test the function structure)
        try:
            results = train_credit_models(hyperparameter_tuning=False)
            assert 'models' in results
            assert 'comparison' in results
        except Exception as e:
            # Expected due to insufficient data
            assert "insufficient" in str(e).lower() or "empty" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__])
