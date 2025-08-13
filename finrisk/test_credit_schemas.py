#!/usr/bin/env python3
"""
Test script for credit risk schemas.
Validates all credit risk schemas and demonstrates their usage.
"""

import sys
import os
from datetime import datetime, date
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.schemas.credit_risk import (
    CreditApplicationRequest,
    CreditApplicationResponse,
    CreditScoringRequest,
    CreditScoringResponse,
    CreditFeatures,
    CreditModelMetrics,
    CreditPrediction,
    CreditDecision,
    CreditPortfolioMetrics,
    CreditStressTestScenario,
    CreditStressTestResult,
    BatchCreditScoringRequest,
    BatchCreditScoringResponse
)
from app.schemas.base import EmploymentStatus, LoanPurpose, ApplicationStatus, ModelType


def test_credit_application_request():
    """Test CreditApplicationRequest schema."""
    print("Testing CreditApplicationRequest...")
    
    # Valid request
    request = CreditApplicationRequest(
        customer_id="CUST001",
        loan_amount=50000,
        loan_purpose=LoanPurpose.PERSONAL,
        employment_status=EmploymentStatus.FULL_TIME,
        annual_income=75000.0
    )
    print(f"✓ Valid request: {request}")
    
    # Test validation
    try:
        invalid_request = CreditApplicationRequest(
            customer_id="CUST002",
            loan_amount=2000000,  # Exceeds max
            loan_purpose=LoanPurpose.HOME_PURCHASE,
            employment_status=EmploymentStatus.SELF_EMPLOYED,
            annual_income=100000.0
        )
    except ValueError as e:
        print(f"✓ Validation caught error: {e}")
    
    print()


def test_credit_application_response():
    """Test CreditApplicationResponse schema."""
    print("Testing CreditApplicationResponse...")
    
    response = CreditApplicationResponse(
        application_id="APP001",
        customer_id="CUST001",
        application_date=date.today(),
        loan_amount=50000,
        loan_purpose=LoanPurpose.PERSONAL,
        employment_status=EmploymentStatus.FULL_TIME,
        annual_income=75000.0,
        debt_to_income_ratio=0.35,
        credit_score=720,
        application_status=ApplicationStatus.APPROVED,
        default_flag=0
    )
    print(f"✓ Valid response: {response}")
    print()


def test_credit_scoring_request():
    """Test CreditScoringRequest schema."""
    print("Testing CreditScoringRequest...")
    
    request = CreditScoringRequest(
        customer_id="CUST001",
        loan_amount=50000,
        loan_purpose=LoanPurpose.PERSONAL,
        include_explanation=True,
        model_version="v1.2.0"
    )
    print(f"✓ Valid scoring request: {request}")
    print()


def test_credit_scoring_response():
    """Test CreditScoringResponse schema."""
    print("Testing CreditScoringResponse...")
    
    response = CreditScoringResponse(
        customer_id="CUST001",
        prediction_id="PRED001",
        credit_score=720,
        risk_score=250.5,
        probability_of_default=0.15,
        risk_grade="B",
        recommended_decision="APPROVE",
        confidence_interval={"lower": 0.12, "upper": 0.18},
        model_version="v1.2.0"
    )
    print(f"✓ Valid scoring response: {response}")
    
    # Test validation
    try:
        invalid_response = CreditScoringResponse(
            customer_id="CUST002",
            prediction_id="PRED002",
            credit_score=750,
            risk_score=300.0,
            probability_of_default=0.20,
            risk_grade="X",  # Invalid grade
            recommended_decision="APPROVE",
            confidence_interval={"lower": 0.15, "upper": 0.25},
            model_version="v1.2.0"
        )
    except ValueError as e:
        print(f"✓ Validation caught error: {e}")
    
    print()


def test_credit_features():
    """Test CreditFeatures schema."""
    print("Testing CreditFeatures...")
    
    features = CreditFeatures(
        customer_id="CUST001",
        age=35,
        annual_income=75000.0,
        employment_status=EmploymentStatus.FULL_TIME,
        account_tenure=5,
        credit_score=720,
        credit_history_length=120,
        number_of_accounts=8,
        total_credit_limit=50000,
        credit_utilization=0.45,
        payment_history=0.85,
        public_records=0,
        loan_amount=50000,
        debt_to_income_ratio=0.35,
        loan_to_income_ratio=0.67,
        behavioral_score=0.78,
        product_holdings=3,
        relationship_value=25000.0,
        income_stability=0.82,
        credit_mix_score=0.75,
        recent_inquiry_count=2
    )
    print(f"✓ Valid features: {features}")
    print()


def test_credit_model_metrics():
    """Test CreditModelMetrics schema."""
    print("Testing CreditModelMetrics...")
    
    metrics = CreditModelMetrics(
        model_name="XGBoost_Credit_Risk",
        model_version="v1.2.0",
        evaluation_date=datetime.now(),
        dataset_size=10000,
        auc_score=0.85,
        gini_coefficient=0.70,
        ks_statistic=0.45,
        precision=0.78,
        recall=0.82,
        f1_score=0.80,
        approval_rate=0.65,
        default_rate=0.12,
        expected_loss=0.08
    )
    print(f"✓ Valid metrics: {metrics}")
    
    # Test validation
    try:
        invalid_metrics = CreditModelMetrics(
            model_name="Test_Model",
            model_version="v1.0.0",
            evaluation_date=datetime.now(),
            dataset_size=5000,
            auc_score=1.5,  # Invalid score
            gini_coefficient=0.60,
            ks_statistic=0.40,
            precision=0.75,
            recall=0.80,
            f1_score=0.77,
            approval_rate=0.70,
            default_rate=0.10,
            expected_loss=0.06
        )
    except ValueError as e:
        print(f"✓ Validation caught error: {e}")
    
    print()


def test_credit_prediction():
    """Test CreditPrediction schema."""
    print("Testing CreditPrediction...")
    
    prediction = CreditPrediction(
        prediction_id="PRED001",
        customer_id="CUST001",
        model_name="XGBoost_Credit_Risk",
        model_version="v1.2.0",
        prediction_type=ModelType.CREDIT_RISK,
        prediction_date=datetime.now(),
        model_features={
            "credit_score": 720,
            "debt_to_income_ratio": 0.35,
            "annual_income": 75000.0
        },
        business_decision="APPROVE",
        risk_score=250.5,
        probability_of_default=0.15,
        credit_grade="B",
        recommended_action="APPROVE",
        confidence_score=0.85,
        primary_risk_factors=["High debt-to-income ratio", "Recent credit inquiries"],
        protective_factors=["Good payment history", "Stable employment"],
        estimated_loss_rate=0.08,
        suggested_interest_rate=5.5,
        maximum_loan_amount=75000
    )
    print(f"✓ Valid prediction: {prediction}")
    print()


def test_credit_decision():
    """Test CreditDecision schema."""
    print("Testing CreditDecision...")
    
    # Create a prediction first
    prediction = CreditPrediction(
        prediction_id="PRED001",
        customer_id="CUST001",
        model_name="XGBoost_Credit_Risk",
        model_version="v1.2.0",
        prediction_type=ModelType.CREDIT_RISK,
        prediction_date=datetime.now(),
        model_features={
            "credit_score": 720,
            "debt_to_income_ratio": 0.35,
            "annual_income": 75000.0
        },
        business_decision="APPROVE",
        risk_score=250.5,
        probability_of_default=0.15,
        credit_grade="B",
        recommended_action="APPROVE",
        confidence_score=0.85,
        primary_risk_factors=["High debt-to-income ratio"],
        protective_factors=["Good payment history"],
        estimated_loss_rate=0.08,
        suggested_interest_rate=5.5,
        maximum_loan_amount=75000
    )
    
    decision = CreditDecision(
        decision_id="DEC001",
        customer_id="CUST001",
        application_id="APP001",
        model_prediction=prediction,
        final_decision="APPROVED",
        decision_reason="Customer meets all criteria with good credit history",
        override_flag=False,
        decision_maker="AUTO_SYSTEM"
    )
    print(f"✓ Valid decision: {decision}")
    print()


def test_credit_portfolio_metrics():
    """Test CreditPortfolioMetrics schema."""
    print("Testing CreditPortfolioMetrics...")
    
    metrics = CreditPortfolioMetrics(
        portfolio_date=date.today(),
        total_applications=1000,
        approved_applications=650,
        declined_applications=350,
        total_outstanding_balance=25000000.0,
        average_loan_amount=38461.54,
        total_customers=650,
        portfolio_default_rate=0.12,
        average_credit_score=715.5,
        risk_segment_distribution={"Low": 200, "Medium": 300, "High": 150},
        approval_rate=0.65,
        loss_rate=0.08,
        net_interest_margin=0.045
    )
    print(f"✓ Valid portfolio metrics: {metrics}")
    print()


def test_credit_stress_test_scenario():
    """Test CreditStressTestScenario schema."""
    print("Testing CreditStressTestScenario...")
    
    scenario = CreditStressTestScenario(
        scenario_name="Economic Recession",
        scenario_description="Severe economic downturn with high unemployment",
        unemployment_rate_change=5.0,
        gdp_growth_change=-3.0,
        interest_rate_change=2.0,
        house_price_change=-15.0,
        expected_default_rate_change=2.5,
        expected_loss_change=1.8,
        capital_impact=0.15
    )
    print(f"✓ Valid stress test scenario: {scenario}")
    print()


def test_credit_stress_test_result():
    """Test CreditStressTestResult schema."""
    print("Testing CreditStressTestResult...")
    
    scenario = CreditStressTestScenario(
        scenario_name="Economic Recession",
        scenario_description="Severe economic downturn",
        unemployment_rate_change=5.0,
        gdp_growth_change=-3.0,
        interest_rate_change=2.0,
        house_price_change=-15.0,
        expected_default_rate_change=2.5,
        expected_loss_change=1.8,
        capital_impact=0.15
    )
    
    result = CreditStressTestResult(
        test_id="STRESS001",
        scenario=scenario,
        test_date=datetime.now(),
        baseline_default_rate=0.12,
        stressed_default_rate=0.15,
        default_rate_change=0.03,
        baseline_loss=0.08,
        stressed_loss=0.10,
        loss_change=0.02,
        current_capital_ratio=0.12,
        stressed_capital_ratio=0.10,
        capital_shortfall=0.0,
        grade_migrations={
            "A": {"A": 80, "B": 15, "C": 5},
            "B": {"B": 70, "C": 20, "D": 10},
            "C": {"C": 60, "D": 30, "E": 10}
        }
    )
    print(f"✓ Valid stress test result: {result}")
    print()


def test_batch_credit_scoring():
    """Test batch credit scoring schemas."""
    print("Testing BatchCreditScoringRequest and Response...")
    
    request = BatchCreditScoringRequest(
        batch_id="BATCH001",
        customer_ids=["CUST001", "CUST002", "CUST003"],
        include_explanations=True,
        model_version="v1.2.0"
    )
    print(f"✓ Valid batch request: {request}")
    
    # Create individual responses
    responses = [
        CreditScoringResponse(
            customer_id="CUST001",
            prediction_id="PRED001",
            credit_score=720,
            risk_score=250.5,
            probability_of_default=0.15,
            risk_grade="B",
            recommended_decision="APPROVE",
            confidence_interval={"lower": 0.12, "upper": 0.18},
            model_version="v1.2.0"
        ),
        CreditScoringResponse(
            customer_id="CUST002",
            prediction_id="PRED002",
            credit_score=680,
            risk_score=350.0,
            probability_of_default=0.25,
            risk_grade="C",
            recommended_decision="REVIEW",
            confidence_interval={"lower": 0.20, "upper": 0.30},
            model_version="v1.2.0"
        )
    ]
    
    batch_response = BatchCreditScoringResponse(
        batch_id="BATCH001",
        total_requests=3,
        successful_scores=2,
        failed_scores=1,
        processing_time_seconds=5.2,
        results=responses,
        errors=[{"customer_id": "CUST003", "error": "Insufficient data"}]
    )
    print(f"✓ Valid batch response: {batch_response}")
    
    # Test validation
    try:
        invalid_request = BatchCreditScoringRequest(
            batch_id="BATCH002",
            customer_ids=["CUST001", "CUST001"],  # Duplicate IDs
            include_explanations=False
        )
    except ValueError as e:
        print(f"✓ Validation caught error: {e}")
    
    print()


def main():
    """Run all schema tests."""
    print("=" * 60)
    print("CREDIT RISK SCHEMAS TEST")
    print("=" * 60)
    print()
    
    try:
        test_credit_application_request()
        test_credit_application_response()
        test_credit_scoring_request()
        test_credit_scoring_response()
        test_credit_features()
        test_credit_model_metrics()
        test_credit_prediction()
        test_credit_decision()
        test_credit_portfolio_metrics()
        test_credit_stress_test_scenario()
        test_credit_stress_test_result()
        test_batch_credit_scoring()
        
        print("=" * 60)
        print("✓ ALL SCHEMA TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
