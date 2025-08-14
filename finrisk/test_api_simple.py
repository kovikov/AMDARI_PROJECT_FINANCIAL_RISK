#!/usr/bin/env python3
"""
Simple test script for FinRisk FastAPI endpoints.
This script tests the main API endpoints to ensure they're working correctly.
"""

import requests
import json
from datetime import datetime

# API base URL - try different ports
BASE_URLS = ["http://127.0.0.1:8000", "http://127.0.0.1:8001", "http://127.0.0.1:8002", "http://127.0.0.1:8003"]

def find_working_server():
    """Find which port the server is running on."""
    for url in BASE_URLS:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Found server running on {url}")
                return url
        except:
            continue
    return None

def test_health_endpoint(base_url):
    """Test the health endpoint."""
    print("🧪 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health endpoint working")
            print(f"   Status: {data['status']}")
            print(f"   Version: {data['version']}")
            print(f"   Models: {data.get('models', 'N/A')}")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_root_endpoint(base_url):
    """Test the root endpoint."""
    print("🧪 Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Root endpoint working")
            print(f"   Service: {data['service']}")
            print(f"   Version: {data['version']}")
            print(f"   Message: {data.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_credit_scoring(base_url):
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
    
    try:
        response = requests.post(
            f"{base_url}/api/v1/credit/score",
            json=credit_request,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Credit scoring working")
            print(f"   Customer: {result['customer_id']}")
            print(f"   Risk Score: {result['risk_score']}")
            print(f"   Default Probability: {result['probability_of_default']:.2%}")
            print(f"   Risk Grade: {result['risk_grade']}")
            print(f"   Decision: {result['recommended_decision']}")
            return True
        else:
            print(f"❌ Credit scoring failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Credit scoring error: {e}")
        return False

def test_fraud_detection(base_url):
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
    
    try:
        response = requests.post(
            f"{base_url}/api/v1/fraud/detect",
            json=fraud_request,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Fraud detection working")
            print(f"   Transaction: {result['transaction_id']}")
            print(f"   Fraud Probability: {result['fraud_probability']:.2%}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Action: {result['recommended_action']}")
            print(f"   Processing Time: {result['processing_time_ms']:.2f}ms")
            return True
        else:
            print(f"❌ Fraud detection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Fraud detection error: {e}")
        return False

def test_model_status(base_url):
    """Test model status endpoint."""
    print("🧪 Testing model status endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v1/models/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Model status working")
            print(f"   Credit Risk Model: {data['models']['credit_risk']['status']}")
            print(f"   Fraud Detection Model: {data['models']['fraud_detection']['status']}")
            return True
        else:
            print(f"❌ Model status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model status error: {e}")
        return False

def main():
    """Run all API tests."""
    print("🚀 Starting FinRisk API Tests")
    print("=" * 50)
    
    # Find working server
    base_url = find_working_server()
    if not base_url:
        print("❌ Cannot find running server on any port")
        print("💡 Make sure the server is running with: uvicorn main:app --host 127.0.0.1 --port 8001")
        return
    
    print(f"\n🔗 Testing server at: {base_url}")
    print("-" * 30)
    
    # Run tests
    tests = [
        test_health_endpoint,
        test_root_endpoint,
        test_credit_scoring,
        test_fraud_detection,
        test_model_status
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test(base_url):
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
        print()
    
    # Summary
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! FinRisk API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the server logs for details.")
    
    print(f"\n🔗 API Documentation: {base_url}/docs")
    print(f"🔗 ReDoc Documentation: {base_url}/redoc")

if __name__ == "__main__":
    main()

