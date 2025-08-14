"""
Test script for FinRisk FastAPI endpoints.
Run this after starting the API server to test all endpoints.
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_root_endpoint():
    """Test the root endpoint."""
    print("🧪 Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    
    if response.status_code == 200:
        print("✅ Root endpoint working")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ Root endpoint failed: {response.status_code}")
    print("-" * 50)

def test_health_endpoint():
    """Test the health check endpoint."""
    print("🧪 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        print("✅ Health endpoint working")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ Health endpoint failed: {response.status_code}")
    print("-" * 50)

def test_credit_scoring():
    """Test credit scoring endpoint."""
    print("🧪 Testing credit scoring endpoint...")
    
    # Sample credit application
    credit_request = {
        "customer_id": "CUST_001234",
        "age": 35,
        "annual_income": 75000.0,
        "employment_status": "Full-time",
        "loan_amount": 25000,
        "credit_score": 720,
        "debt_to_income_ratio": 0.35
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/credit/score",
        json=credit_request
    )
    
    if response.status_code == 200:
        print("✅ Credit scoring working")
        result = response.json()
        print(f"Customer: {result['customer_id']}")
        print(f"Risk Score: {result['risk_score']}")
        print(f"Default Probability: {result['probability_of_default']:.2%}")
        print(f"Risk Grade: {result['risk_grade']}")
        print(f"Decision: {result['recommended_decision']}")
    else:
        print(f"❌ Credit scoring failed: {response.status_code}")
        print(response.text)
    print("-" * 50)

def test_fraud_detection():
    """Test fraud detection endpoint."""
    print("🧪 Testing fraud detection endpoint...")
    
    # Sample transaction
    fraud_request = {
        "transaction_id": "TXN_789012",
        "customer_id": "CUST_001234",
        "amount": 1500.0,
        "merchant_category": "Online Shopping",
        "location": "London",
        "hour": 23,
        "is_weekend": True
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/fraud/detect",
        json=fraud_request
    )
    
    if response.status_code == 200:
        print("✅ Fraud detection working")
        result = response.json()
        print(f"Transaction: {result['transaction_id']}")
        print(f"Fraud Probability: {result['fraud_probability']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Action: {result['recommended_action']}")
        print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
    else:
        print(f"❌ Fraud detection failed: {response.status_code}")
        print(response.text)
    print("-" * 50)

def test_model_status():
    """Test model status endpoint."""
    print("🧪 Testing model status endpoint...")
    response = requests.get(f"{BASE_URL}/api/v1/models/status")
    
    if response.status_code == 200:
        print("✅ Model status working")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ Model status failed: {response.status_code}")
    print("-" * 50)

def test_batch_credit_scoring():
    """Test batch credit scoring endpoint."""
    print("🧪 Testing batch credit scoring...")
    
    # Multiple credit applications
    batch_requests = [
        {
            "customer_id": "CUST_001",
            "age": 25,
            "annual_income": 45000.0,
            "employment_status": "Full-time",
            "loan_amount": 15000,
            "credit_score": 650,
            "debt_to_income_ratio": 0.4
        },
        {
            "customer_id": "CUST_002",
            "age": 45,
            "annual_income": 85000.0,
            "employment_status": "Full-time",
            "loan_amount": 30000,
            "credit_score": 780,
            "debt_to_income_ratio": 0.25
        }
    ]
    
    response = requests.post(
        f"{BASE_URL}/api/v1/credit/batch",
        json=batch_requests
    )
    
    if response.status_code == 200:
        print("✅ Batch credit scoring working")
        result = response.json()
        print(f"Total requests: {result['total_requests']}")
        print(f"Successful: {result['successful_results']}")
        print(f"Failed: {result['failed_requests']}")
    else:
        print(f"❌ Batch credit scoring failed: {response.status_code}")
    print("-" * 50)

def run_all_tests():
    """Run all API tests."""
    print("🚀 Starting FinRisk API Tests")
    print("=" * 50)
    
    try:
        test_root_endpoint()
        test_health_endpoint()
        test_credit_scoring()
        test_fraud_detection()
        test_model_status()
        test_batch_credit_scoring()
        
        print("🎉 All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server.")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    run_all_tests()
