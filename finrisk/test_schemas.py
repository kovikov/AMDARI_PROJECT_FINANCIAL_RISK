#!/usr/bin/env python3
"""
Test script for FinRisk Pydantic schemas.
"""

import sys
from pathlib import Path
from datetime import datetime, date, timezone

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.schemas import (
    CustomerCreate, CustomerResponse, RiskSegment, EmploymentStatus,
    CreditApplicationCreate, ApplicationStatus, LoanPurpose,
    CreditRiskPrediction, FraudDetectionPrediction, ModelType,
    HealthCheckResponse, ErrorResponse, PaginationParams, PaginatedResponse
)


def test_customer_schemas():
    """Test customer-related schemas."""
    print("🔍 Testing customer schemas...")
    
    # Test CustomerCreate
    customer_data = {
        "customer_id": "CUST001",
        "customer_age": 35,
        "annual_income": 75000.0,
        "employment_status": EmploymentStatus.FULL_TIME,
        "account_tenure": 5,
        "product_holdings": 3,
        "relationship_value": 150000.0,
        "risk_segment": RiskSegment.PRIME,
        "behavioral_score": 850.0,
        "credit_score": 720,
        "city": "London",
        "last_activity_date": date.today()
    }
    
    customer = CustomerCreate(**customer_data)
    print(f"✅ CustomerCreate: {customer.customer_id} - {customer.risk_segment}")
    
    # Test CustomerResponse
    customer_response_data = {
        **customer_data,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc)
    }
    
    customer_response = CustomerResponse(**customer_response_data)
    print(f"✅ CustomerResponse: {customer_response.customer_id} - {customer_response.employment_status}")
    
    return True


def test_application_schemas():
    """Test credit application schemas."""
    print("\n🔍 Testing credit application schemas...")
    
    # Test CreditApplicationCreate
    application_data = {
        "customer_id": "CUST001",
        "loan_amount": 50000.0,
        "loan_purpose": LoanPurpose.HOME_PURCHASE,
        "loan_term": 240,
        "interest_rate": 3.5,
        "monthly_payment": 2500.0,
        "debt_to_income_ratio": 0.35,
        "application_source": "online"
    }
    
    application = CreditApplicationCreate(**application_data)
    print(f"✅ CreditApplicationCreate: {application.loan_purpose} - £{application.loan_amount:,.0f}")
    
    return True


def test_prediction_schemas():
    """Test model prediction schemas."""
    print("\n🔍 Testing model prediction schemas...")
    
    # Test CreditRiskPrediction
    credit_prediction_data = {
        "customer_id": "CUST001",
        "model_version": "v1.2.0",
        "risk_score": 0.25,
        "default_probability": 0.15,
        "credit_limit_recommendation": 75000.0,
        "interest_rate_recommendation": 3.2,
        "risk_segment": "Prime",
        "decision": "approve",
        "confidence_score": 0.92,
        "model_features": {"age": 35, "income": 75000, "credit_score": 720},
        "feature_importance": [
            {"feature_name": "credit_score", "importance": 0.45, "rank": 1},
            {"feature_name": "income", "importance": 0.30, "rank": 2}
        ]
    }
    
    credit_prediction = CreditRiskPrediction(**credit_prediction_data)
    print(f"✅ CreditRiskPrediction: {credit_prediction.decision} - {credit_prediction.risk_score:.2%}")
    
    # Test FraudDetectionPrediction
    fraud_prediction_data = {
        "customer_id": "CUST001",
        "transaction_id": "TXN123",
        "model_version": "v1.1.0",
        "fraud_probability": 0.05,
        "risk_level": "low",
        "fraud_score": 0.12,
        "decision": "allow",
        "confidence_score": 0.88,
        "model_features": {"amount": 150.0, "location": "London", "time": "14:30"},
        "feature_importance": [
            {"feature_name": "amount", "importance": 0.60, "rank": 1},
            {"feature_name": "location", "importance": 0.25, "rank": 2}
        ]
    }
    
    fraud_prediction = FraudDetectionPrediction(**fraud_prediction_data)
    print(f"✅ FraudDetectionPrediction: {fraud_prediction.decision} - {fraud_prediction.fraud_probability:.2%}")
    
    return True


def test_common_schemas():
    """Test common schemas."""
    print("\n🔍 Testing common schemas...")
    
    # Test HealthCheckResponse
    health_data = {
        "status": "healthy",
        "checks": {
            "database": "connected",
            "redis": "connected",
            "models": "loaded"
        }
    }
    
    health_response = HealthCheckResponse(**health_data)
    print(f"✅ HealthCheckResponse: {health_response.status} - {health_response.version}")
    
    # Test ErrorResponse
    error_data = {
        "error": "Validation failed",
        "detail": "Invalid customer ID format",
        "request_id": "req_123456"
    }
    
    error_response = ErrorResponse(**error_data)
    print(f"✅ ErrorResponse: {error_response.error} - {error_response.request_id}")
    
    # Test PaginationParams
    pagination = PaginationParams(page=2, size=50)
    print(f"✅ PaginationParams: page {pagination.page}, size {pagination.size}, offset {pagination.offset}")
    
    # Test PaginatedResponse
    paginated_data = {
        "items": [{"id": 1, "name": "Test"}],
        "total": 100,
        "page": 1,
        "size": 10
    }
    
    paginated_response = PaginatedResponse(**paginated_data)
    print(f"✅ PaginatedResponse: {paginated_response.total} items, {paginated_response.pages} pages")
    
    return True


def test_enum_values():
    """Test enum values."""
    print("\n🔍 Testing enum values...")
    
    print(f"✅ RiskSegment: {RiskSegment.PRIME} = {RiskSegment.PRIME.value}")
    print(f"✅ EmploymentStatus: {EmploymentStatus.FULL_TIME} = {EmploymentStatus.FULL_TIME.value}")
    print(f"✅ ApplicationStatus: {ApplicationStatus.APPROVED} = {ApplicationStatus.APPROVED.value}")
    print(f"✅ LoanPurpose: {LoanPurpose.HOME_PURCHASE} = {LoanPurpose.HOME_PURCHASE.value}")
    print(f"✅ ModelType: {ModelType.CREDIT_RISK} = {ModelType.CREDIT_RISK.value}")
    
    return True


def main():
    """Run all schema tests."""
    print("🚀 FinRisk Schema Tests")
    print("=" * 50)
    
    tests = [
        ("Customer Schemas", test_customer_schemas),
        ("Application Schemas", test_application_schemas),
        ("Prediction Schemas", test_prediction_schemas),
        ("Common Schemas", test_common_schemas),
        ("Enum Values", test_enum_values)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All schema tests passed! Schemas are working correctly.")
        return 0
    else:
        print("⚠️  Some schema tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
