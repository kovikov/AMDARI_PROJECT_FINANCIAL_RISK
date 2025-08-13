#!/usr/bin/env python3
"""
Test script for MLflow integration with FinRisk application.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.monitoring.mlflow_tracker import (
    FinRiskMLflowTracker, 
    FinRiskModelRegistry,
    log_credit_risk_model,
    log_fraud_detection_model,
    register_model_to_production
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_mlflow_tracker_initialization():
    """Test MLflow tracker initialization."""
    print("Testing MLflow tracker initialization...")
    
    try:
        tracker = FinRiskMLflowTracker()
        print(f"✓ MLflow tracker initialized successfully")
        print(f"  - Tracking URI: {tracker.tracking_uri}")
        print(f"  - Experiment: {tracker.experiment_name}")
        print(f"  - Experiment ID: {tracker.experiment_id}")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize MLflow tracker: {e}")
        return False


def test_model_registry_initialization():
    """Test model registry initialization."""
    print("\nTesting model registry initialization...")
    
    try:
        registry = FinRiskModelRegistry()
        print(f"✓ Model registry initialized successfully")
        print(f"  - Registry URI: {registry.registry_uri}")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize model registry: {e}")
        return False


def test_credit_risk_experiment_logging():
    """Test credit risk experiment logging."""
    print("\nTesting credit risk experiment logging...")
    
    try:
        # Generate synthetic data
        X, y = make_classification(
            n_samples=1000, n_features=10, n_informative=8,
            n_redundant=2, random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y = pd.Series(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'auc': roc_auc_score(y_test, y_prob),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        # Get feature importance
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        
        # Log experiment
        run_id = log_credit_risk_model(
            model=model,
            model_name="random_forest_test",
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            hyperparameters=model.get_params(),
            metrics=metrics,
            feature_importance=feature_importance,
            model_type="sklearn"
        )
        
        print(f"✓ Credit risk experiment logged successfully")
        print(f"  - Run ID: {run_id}")
        print(f"  - Metrics: {metrics}")
        return run_id
        
    except Exception as e:
        print(f"✗ Failed to log credit risk experiment: {e}")
        return None


def test_fraud_detection_experiment_logging():
    """Test fraud detection experiment logging."""
    print("\nTesting fraud detection experiment logging...")
    
    try:
        # Generate synthetic data with more anomalies
        X, y = make_classification(
            n_samples=1000, n_features=10, n_informative=8,
            n_redundant=2, n_clusters_per_class=1,
            weights=[0.9, 0.1], random_state=42  # 10% fraud rate
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y = pd.Series(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'auc': roc_auc_score(y_test, y_prob),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        # Log experiment
        run_id = log_fraud_detection_model(
            model=model,
            model_name="random_forest_fraud_test",
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            hyperparameters=model.get_params(),
            metrics=metrics,
            anomaly_scores=y_prob,
            model_type="sklearn"
        )
        
        print(f"✓ Fraud detection experiment logged successfully")
        print(f"  - Run ID: {run_id}")
        print(f"  - Metrics: {metrics}")
        return run_id
        
    except Exception as e:
        print(f"✗ Failed to log fraud detection experiment: {e}")
        return None


def test_model_registration(run_id: str):
    """Test model registration."""
    print("\nTesting model registration...")
    
    if not run_id:
        print("✗ No run ID provided for model registration")
        return False
    
    try:
        # Register model to production
        version = register_model_to_production(
            run_id=run_id,
            model_name="test_credit_risk_model",
            description="Test credit risk model for MLflow integration"
        )
        
        print(f"✓ Model registered successfully")
        print(f"  - Model name: test_credit_risk_model")
        print(f"  - Version: {version}")
        print(f"  - Stage: Production")
        return True
        
    except Exception as e:
        print(f"✗ Failed to register model: {e}")
        return False


def test_experiment_summary():
    """Test experiment summary retrieval."""
    print("\nTesting experiment summary...")
    
    try:
        tracker = FinRiskMLflowTracker()
        summary = tracker.get_experiment_summary()
        
        print(f"✓ Experiment summary retrieved successfully")
        print(f"  - Total runs: {summary['total_runs']}")
        print(f"  - Experiment name: {summary['experiment_name']}")
        print(f"  - Model types: {summary['model_types']}")
        print(f"  - Algorithms: {summary['algorithms']}")
        
        if summary['latest_runs']:
            print(f"  - Latest runs: {len(summary['latest_runs'])}")
            for run in summary['latest_runs'][:3]:  # Show first 3
                print(f"    * {run['run_name']} ({run['model_type']})")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to get experiment summary: {e}")
        return False


def test_model_loading():
    """Test model loading from MLflow."""
    print("\nTesting model loading...")
    
    try:
        tracker = FinRiskMLflowTracker()
        
        # Get best run by AUC
        best_run_id = tracker.get_best_run("auc", greater_is_better=True, model_type="credit_risk")
        
        if best_run_id:
            # Load model
            model = tracker.load_model(best_run_id)
            print(f"✓ Model loaded successfully")
            print(f"  - Run ID: {best_run_id}")
            print(f"  - Model type: {type(model)}")
            return True
        else:
            print("✗ No runs found for model loading")
            return False
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False


def main():
    """Run all MLflow integration tests."""
    print("=" * 60)
    print("MLflow Integration Test Suite")
    print("=" * 60)
    
    # Set environment variables for testing
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    os.environ.setdefault("MLFLOW_REGISTRY_URI", "sqlite:///mlflow.db")
    
    test_results = []
    
    # Test 1: Tracker initialization
    test_results.append(test_mlflow_tracker_initialization())
    
    # Test 2: Registry initialization
    test_results.append(test_model_registry_initialization())
    
    # Test 3: Credit risk experiment logging
    credit_run_id = test_credit_risk_experiment_logging()
    test_results.append(credit_run_id is not None)
    
    # Test 4: Fraud detection experiment logging
    fraud_run_id = test_fraud_detection_experiment_logging()
    test_results.append(fraud_run_id is not None)
    
    # Test 5: Model registration
    if credit_run_id:
        test_results.append(test_model_registration(credit_run_id))
    else:
        test_results.append(False)
    
    # Test 6: Experiment summary
    test_results.append(test_experiment_summary())
    
    # Test 7: Model loading
    test_results.append(test_model_loading())
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("✓ All tests passed! MLflow integration is working correctly.")
    else:
        print("✗ Some tests failed. Please check the MLflow setup.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
