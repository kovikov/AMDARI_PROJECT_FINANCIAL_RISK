#!/usr/bin/env python3
"""
Test script for fraud detection schemas.
Validates all fraud detection schemas and demonstrates their usage.
"""

import sys
import os
from datetime import datetime, date
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.schemas.fraud_detection import (
    TransactionData,
    FraudDetectionRequest,
    FraudDetectionResponse,
    FraudFeatures,
    FraudAlert,
    FraudModelMetrics,
    FraudPrediction,
    FraudInvestigation,
    FraudPortfolioMetrics,
    BatchFraudDetectionRequest,
    BatchFraudDetectionResponse,
    FraudRuleEngine,
    FraudRuleViolation
)
from app.schemas.base import MerchantCategory, TransactionType, ModelType


def test_transaction_data():
    """Test TransactionData schema."""
    print("Testing TransactionData...")
    
    transaction = TransactionData(
        transaction_id="TXN001",
        customer_id="CUST001",
        transaction_date=datetime.now(),
        amount=150.50,
        merchant_category=MerchantCategory.ONLINE_SHOPPING,
        transaction_type=TransactionType.PURCHASE,
        location="London, UK",
        device_info="iPhone 14",
        fraud_flag=0,
        investigation_status="None"
    )
    print(f"✓ Valid transaction: {transaction}")
    print()


def test_fraud_detection_request():
    """Test FraudDetectionRequest schema."""
    print("Testing FraudDetectionRequest...")
    
    request = FraudDetectionRequest(
        transaction_id="TXN001",
        customer_id="CUST001",
        amount=150.50,
        merchant_category=MerchantCategory.ONLINE_SHOPPING,
        transaction_type=TransactionType.PURCHASE,
        location="London, UK",
        device_info="iPhone 14",
        include_explanation=True
    )
    print(f"✓ Valid request: {request}")
    
    # Test validation
    try:
        invalid_request = FraudDetectionRequest(
            transaction_id="TXN002",
            customer_id="CUST002",
            amount=150000.0,  # Exceeds max
            merchant_category=MerchantCategory.TRAVEL,
            transaction_type=TransactionType.PURCHASE,
            location="Paris, France",
            device_info="Desktop",
            include_explanation=False
        )
    except ValueError as e:
        print(f"✓ Validation caught error: {e}")
    
    print()


def test_fraud_detection_response():
    """Test FraudDetectionResponse schema."""
    print("Testing FraudDetectionResponse...")
    
    response = FraudDetectionResponse(
        transaction_id="TXN001",
        customer_id="CUST001",
        prediction_id="PRED001",
        fraud_probability=0.15,
        fraud_score=250,
        risk_level="LOW",
        recommended_action="ALLOW",
        confidence_score=0.85,
        model_version="v1.2.0",
        processing_time_ms=45.2
    )
    print(f"✓ Valid response: {response}")
    
    # Test validation
    try:
        invalid_response = FraudDetectionResponse(
            transaction_id="TXN002",
            customer_id="CUST002",
            prediction_id="PRED002",
            fraud_probability=0.85,
            fraud_score=750,
            risk_level="INVALID",  # Invalid level
            recommended_action="BLOCK",
            confidence_score=0.90,
            model_version="v1.2.0",
            processing_time_ms=50.0
        )
    except ValueError as e:
        print(f"✓ Validation caught error: {e}")
    
    print()


def test_fraud_features():
    """Test FraudFeatures schema."""
    print("Testing FraudFeatures...")
    
    features = FraudFeatures(
        transaction_id="TXN001",
        customer_id="CUST001",
        amount=150.50,
        merchant_category=MerchantCategory.ONLINE_SHOPPING,
        transaction_type=TransactionType.PURCHASE,
        location="London, UK",
        device_info="iPhone 14",
        transaction_hour=14,
        transaction_day_of_week=2,
        avg_transaction_amount=125.0,
        transaction_frequency=2.5,
        location_frequency=0.8,
        merchant_frequency=0.6,
        device_frequency=0.9,
        transactions_last_hour=1,
        transactions_last_day=5,
        amount_last_hour=150.50,
        amount_last_day=750.0,
        amount_zscore=1.2,
        time_since_last_transaction=2.5,
        distance_from_home=5.0,
        new_merchant_flag=False,
        new_location_flag=False,
        new_device_flag=False,
        high_risk_merchant=False,
        unusual_time=False,
        round_amount=False,
        weekend_transaction=False
    )
    print(f"✓ Valid features: {features}")
    print()


def test_fraud_alert():
    """Test FraudAlert schema."""
    print("Testing FraudAlert...")
    
    alert = FraudAlert(
        alert_id="ALERT001",
        transaction_id="TXN001",
        customer_id="CUST001",
        alert_type="High Amount",
        fraud_probability=0.75,
        risk_level="HIGH",
        alert_message="Unusual transaction amount detected",
        investigation_required=True,
        assigned_to="INV001",
        investigation_status="In Progress"
    )
    print(f"✓ Valid alert: {alert}")
    print()


def test_fraud_model_metrics():
    """Test FraudModelMetrics schema."""
    print("Testing FraudModelMetrics...")
    
    metrics = FraudModelMetrics(
        model_name="XGBoost_Fraud_Detection",
        model_version="v1.2.0",
        evaluation_date=datetime.now(),
        dataset_size=50000,
        precision=0.85,
        recall=0.78,
        f1_score=0.81,
        auc_score=0.92,
        false_positive_rate=0.12,
        false_negative_rate=0.22,
        detection_rate=0.78,
        average_processing_time_ms=45.2,
        throughput_per_second=1000.0
    )
    print(f"✓ Valid metrics: {metrics}")
    
    # Test validation
    try:
        invalid_metrics = FraudModelMetrics(
            model_name="Test_Model",
            model_version="v1.0.0",
            evaluation_date=datetime.now(),
            dataset_size=10000,
            precision=1.5,  # Invalid score
            recall=0.80,
            f1_score=0.82,
            auc_score=0.90,
            false_positive_rate=0.10,
            false_negative_rate=0.20,
            detection_rate=0.80,
            average_processing_time_ms=50.0,
            throughput_per_second=800.0
        )
    except ValueError as e:
        print(f"✓ Validation caught error: {e}")
    
    print()


def test_fraud_prediction():
    """Test FraudPrediction schema."""
    print("Testing FraudPrediction...")
    
    prediction = FraudPrediction(
        prediction_id="PRED001",
        customer_id="CUST001",
        model_name="XGBoost_Fraud_Detection",
        model_version="v1.2.0",
        prediction_type=ModelType.FRAUD_DETECTION,
        prediction_date=datetime.now(),
        model_features={
            "amount": 150.50,
            "merchant_category": "Online Shopping",
            "location": "London, UK"
        },
        business_decision="ALLOW",
        fraud_probability=0.15,
        fraud_score=250,
        risk_level="LOW",
        anomaly_score=0.25,
        fraud_indicators=["Normal transaction pattern"],
        risk_factors={"amount": 0.1, "location": 0.05},
        isolation_forest_score=0.2,
        one_class_svm_score=0.3,
        ensemble_score=0.25,
        recommended_action="ALLOW",
        block_transaction=False,
        require_verification=False
    )
    print(f"✓ Valid prediction: {prediction}")
    print()


def test_fraud_investigation():
    """Test FraudInvestigation schema."""
    print("Testing FraudInvestigation...")
    
    investigation = FraudInvestigation(
        investigation_id="INV001",
        transaction_id="TXN001",
        customer_id="CUST001",
        alert_id="ALERT001",
        investigator_id="INVEST001",
        investigation_type="Transaction Review",
        priority="Medium",
        status="In Progress",
        investigation_notes=["Customer confirmed transaction", "No fraud detected"],
        evidence=[{"type": "Customer Call", "details": "Customer confirmed purchase"}]
    )
    print(f"✓ Valid investigation: {investigation}")
    print()


def test_fraud_portfolio_metrics():
    """Test FraudPortfolioMetrics schema."""
    print("Testing FraudPortfolioMetrics...")
    
    metrics = FraudPortfolioMetrics(
        portfolio_date=date.today(),
        total_transactions=100000,
        flagged_transactions=500,
        confirmed_frauds=50,
        false_positives=450,
        total_transaction_volume=5000000.0,
        fraud_loss_amount=25000.0,
        prevented_fraud_amount=100000.0,
        investigation_costs=5000.0,
        fraud_detection_rate=0.90,
        false_positive_rate=0.10,
        precision=0.85,
        recall=0.78,
        average_investigation_time=4.5,
        pending_investigations=25
    )
    print(f"✓ Valid portfolio metrics: {metrics}")
    print()


def test_batch_fraud_detection():
    """Test batch fraud detection schemas."""
    print("Testing BatchFraudDetectionRequest and Response...")
    
    # Create individual requests
    requests = [
        FraudDetectionRequest(
            transaction_id="TXN001",
            customer_id="CUST001",
            amount=150.50,
            merchant_category=MerchantCategory.ONLINE_SHOPPING,
            transaction_type=TransactionType.PURCHASE,
            location="London, UK",
            device_info="iPhone 14"
        ),
        FraudDetectionRequest(
            transaction_id="TXN002",
            customer_id="CUST002",
            amount=500.0,
            merchant_category=MerchantCategory.TRAVEL,
            transaction_type=TransactionType.PURCHASE,
            location="Paris, France",
            device_info="Desktop"
        )
    ]
    
    batch_request = BatchFraudDetectionRequest(
        batch_id="BATCH001",
        transactions=requests,
        include_explanations=True,
        model_version="v1.2.0"
    )
    print(f"✓ Valid batch request: {batch_request}")
    
    # Create individual responses
    responses = [
        FraudDetectionResponse(
            transaction_id="TXN001",
            customer_id="CUST001",
            prediction_id="PRED001",
            fraud_probability=0.15,
            fraud_score=250,
            risk_level="LOW",
            recommended_action="ALLOW",
            confidence_score=0.85,
            model_version="v1.2.0",
            processing_time_ms=45.2
        ),
        FraudDetectionResponse(
            transaction_id="TXN002",
            customer_id="CUST002",
            prediction_id="PRED002",
            fraud_probability=0.45,
            fraud_score=450,
            risk_level="MEDIUM",
            recommended_action="REVIEW",
            confidence_score=0.75,
            model_version="v1.2.0",
            processing_time_ms=52.1
        )
    ]
    
    batch_response = BatchFraudDetectionResponse(
        batch_id="BATCH001",
        total_requests=2,
        successful_detections=2,
        failed_detections=0,
        processing_time_seconds=2.5,
        average_processing_time_ms=48.65,
        results=responses
    )
    print(f"✓ Valid batch response: {batch_response}")
    
    # Test validation
    try:
        invalid_batch_request = BatchFraudDetectionRequest(
            batch_id="BATCH002",
            transactions=[
                FraudDetectionRequest(
                    transaction_id="TXN001",  # Duplicate ID
                    customer_id="CUST001",
                    amount=100.0,
                    merchant_category=MerchantCategory.GROCERIES,
                    transaction_type=TransactionType.PURCHASE,
                    location="London, UK",
                    device_info="Mobile"
                ),
                FraudDetectionRequest(
                    transaction_id="TXN001",  # Duplicate ID
                    customer_id="CUST002",
                    amount=200.0,
                    merchant_category=MerchantCategory.RESTAURANTS,
                    transaction_type=TransactionType.PURCHASE,
                    location="Manchester, UK",
                    device_info="Desktop"
                )
            ]
        )
    except ValueError as e:
        print(f"✓ Validation caught error: {e}")
    
    print()


def test_fraud_rule_engine():
    """Test FraudRuleEngine schema."""
    print("Testing FraudRuleEngine...")
    
    rule = FraudRuleEngine(
        rule_id="RULE001",
        rule_name="High Amount Threshold",
        rule_description="Flag transactions above £1000",
        rule_type="threshold",
        threshold_value=1000.0,
        time_window_minutes=None,
        conditions={"amount": "> 1000", "merchant_category": "not in ['Groceries', 'Fuel']"},
        severity="HIGH",
        created_by="ADMIN001"
    )
    print(f"✓ Valid rule: {rule}")
    print()


def test_fraud_rule_violation():
    """Test FraudRuleViolation schema."""
    print("Testing FraudRuleViolation...")
    
    violation = FraudRuleViolation(
        violation_id="VIOL001",
        rule_id="RULE001",
        transaction_id="TXN001",
        customer_id="CUST001",
        violation_type="Amount Threshold",
        violation_value=1500.0,
        threshold_value=1000.0,
        severity="HIGH",
        action_taken="BLOCK",
        alert_generated=True,
        transaction_blocked=True
    )
    print(f"✓ Valid violation: {violation}")
    print()


def main():
    """Run all schema tests."""
    print("=" * 60)
    print("FRAUD DETECTION SCHEMAS TEST")
    print("=" * 60)
    print()
    
    try:
        test_transaction_data()
        test_fraud_detection_request()
        test_fraud_detection_response()
        test_fraud_features()
        test_fraud_alert()
        test_fraud_model_metrics()
        test_fraud_prediction()
        test_fraud_investigation()
        test_fraud_portfolio_metrics()
        test_batch_fraud_detection()
        test_fraud_rule_engine()
        test_fraud_rule_violation()
        
        print("=" * 60)
        print("✓ ALL FRAUD DETECTION SCHEMA TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
