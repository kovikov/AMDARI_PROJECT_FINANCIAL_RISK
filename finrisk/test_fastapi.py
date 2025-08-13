#!/usr/bin/env python3
"""
Test script for FinRisk FastAPI endpoints.
This script tests the main API endpoints to ensure they're working correctly.
"""

import requests
import json
import time
from datetime import datetime

# API Configuration
BASE_URL = "http://localhost:8000"
API_ENDPOINTS = {
    "health": "/health",
    "status": "/status",
    "metrics": "/metrics",
    "credit_score": "/api/v1/credit/score",
    "credit_assess": "/api/v1/credit/assess",
    "fraud_detect": "/api/v1/fraud/detect",
    "portfolio_summary": "/api/v1/portfolio/summary"
}

def test_health_endpoint():
    """Test the health endpoint."""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_status_endpoint():
    """Test the status endpoint."""
    print("📊 Testing status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status check passed: {data['status']}")
            print(f"   Database: {data['components']['database']['status']}")
            print(f"   Cache: {data['components']['cache']['status']}")
            return True
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return False

def test_metrics_endpoint():
    """Test the metrics endpoint."""
    print("📈 Testing metrics endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Metrics endpoint working")
            print(f"   Available metrics: {list(data['metrics'].keys())}")
            return True
        else:
            print(f"❌ Metrics check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Metrics check error: {e}")
        return False

def test_credit_scoring():
    """Test credit scoring endpoint."""
    print("💳 Testing credit scoring endpoint...")
    try:
        # Test data
        test_data = {
            "customer_id": "TEST_CUST_001",
            "features": {
                "annual_income": 75000,
                "credit_score": 720,
                "employment_years": 5,
                "existing_debt": 15000,
                "loan_amount": 50000
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/credit/score",
            json=test_data,
            timeout=10
        )
        
        if response.status_code in [200, 401]:  # 401 is expected without auth
            print(f"✅ Credit scoring endpoint responding (status: {response.status_code})")
            if response.status_code == 200:
                data = response.json()
                print(f"   Credit score: {data.get('credit_score', 'N/A')}")
                print(f"   Risk level: {data.get('risk_level', 'N/A')}")
            else:
                print("   Authentication required (expected)")
            return True
        else:
            print(f"❌ Credit scoring failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Credit scoring error: {e}")
        return False

def test_fraud_detection():
    """Test fraud detection endpoint."""
    print("🕵️ Testing fraud detection endpoint...")
    try:
        # Test data
        test_data = {
            "transaction_id": "TEST_TXN_001",
            "customer_id": "TEST_CUST_001",
            "amount": 1500.00,
            "merchant_category": "electronics",
            "location": "New York, NY",
            "timestamp": datetime.utcnow().isoformat(),
            "device_id": "TEST_DEV_001"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/fraud/detect",
            json=test_data,
            timeout=10
        )
        
        if response.status_code in [200, 401]:  # 401 is expected without auth
            print(f"✅ Fraud detection endpoint responding (status: {response.status_code})")
            if response.status_code == 200:
                data = response.json()
                print(f"   Fraud score: {data.get('fraud_score', 'N/A')}")
                print(f"   Risk level: {data.get('risk_level', 'N/A')}")
            else:
                print("   Authentication required (expected)")
            return True
        else:
            print(f"❌ Fraud detection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Fraud detection error: {e}")
        return False

def test_portfolio_analysis():
    """Test portfolio analysis endpoint."""
    print("📊 Testing portfolio analysis endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/portfolio/summary", timeout=10)
        if response.status_code in [200, 401]:  # 401 is expected without auth
            print(f"✅ Portfolio analysis endpoint responding (status: {response.status_code})")
            if response.status_code == 200:
                data = response.json()
                print(f"   Total customers: {data.get('total_customers', 'N/A')}")
                print(f"   Portfolio health: {data.get('portfolio_health', 'N/A')}")
            else:
                print("   Authentication required (expected)")
            return True
        else:
            print(f"❌ Portfolio analysis failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Portfolio analysis error: {e}")
        return False

def test_api_documentation():
    """Test API documentation endpoints."""
    print("📚 Testing API documentation...")
    try:
        # Test OpenAPI JSON
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=10)
        if response.status_code == 200:
            print("✅ OpenAPI JSON available")
            return True
        else:
            print(f"❌ OpenAPI JSON failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API documentation error: {e}")
        return False

def main():
    """Run all API tests."""
    print("🚀 Starting FinRisk FastAPI Tests")
    print("=" * 50)
    
    # Check if server is running
    print("🔍 Checking if FastAPI server is running...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ FastAPI server is running!")
        else:
            print(f"⚠️  Server responded with status: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to FastAPI server: {e}")
        print("💡 Make sure to start the server first:")
        print("   python run_api.py")
        return
    
    print("\n🧪 Running API tests...")
    print("-" * 30)
    
    # Run tests
    tests = [
        test_health_endpoint,
        test_status_endpoint,
        test_metrics_endpoint,
        test_credit_scoring,
        test_fraud_detection,
        test_portfolio_analysis,
        test_api_documentation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
        print()
    
    # Summary
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! FastAPI is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the server logs for details.")
    
    print("\n🔗 Useful URLs:")
    print(f"   - API Documentation: {BASE_URL}/docs")
    print(f"   - ReDoc Documentation: {BASE_URL}/redoc")
    print(f"   - OpenAPI JSON: {BASE_URL}/openapi.json")
    print(f"   - Health Check: {BASE_URL}/health")

if __name__ == "__main__":
    main()
