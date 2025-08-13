#!/usr/bin/env python3
"""
FinRisk Setup Test
Tests all components to ensure they are working correctly.
"""

import sys
import os
from pathlib import Path
from datetime import date

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    try:
        import pandas as pd
        print(f"✅ pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ numpy {np.__version__}")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False
    
    try:
        import fastapi
        print("✅ fastapi")
    except ImportError as e:
        print(f"❌ fastapi: {e}")
        return False
    
    try:
        import streamlit
        print("✅ streamlit")
    except ImportError as e:
        print(f"❌ streamlit: {e}")
        return False
    
    try:
        import xgboost
        print("✅ xgboost")
    except ImportError as e:
        print(f"❌ xgboost: {e}")
        return False
    
    try:
        import lightgbm
        print("✅ lightgbm")
    except ImportError as e:
        print(f"❌ lightgbm: {e}")
        return False
    
    try:
        import shap
        print("✅ shap")
    except ImportError as e:
        print(f"❌ shap: {e}")
        return False
    
    try:
        import sqlalchemy
        print("✅ sqlalchemy")
    except ImportError as e:
        print(f"❌ sqlalchemy: {e}")
        return False
    
    try:
        import mlflow
        print("✅ mlflow")
    except ImportError as e:
        print(f"❌ mlflow: {e}")
        return False
    
    try:
        import evidently
        print("✅ evidently")
    except ImportError as e:
        print(f"❌ evidently: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration loading"""
    print("\n🔍 Testing configuration...")
    
    try:
        from app.config import settings
        print("✅ Configuration loaded successfully")
        print(f"   Database: {settings.database.host}:{settings.database.port}")
        print(f"   API Port: {settings.api.port}")
        print(f"   Environment: {settings.environment}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_file_structure():
    """Test that required files and directories exist"""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        "requirements.txt",
        "README.md",
        "app/config.py",
        "sql/001_init_schema.sql",
        "sql/010_indexes.sql",
        "sql/020_sample_views.sql",
        "scripts/setup_database.py"
    ]
    
    required_dirs = [
        "app",
        "app/api",
        "app/api/routes",
        "app/features",
        "app/models",
        "app/monitoring",
        "app/portfolio",
        "app/dashboards",
        "app/schemas",
        "app/infra",
        "data",
        "data/seed",
        "data/exports",
        "data/models",
        "sql",
        "scripts",
        "tests"
    ]
    
    all_good = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            all_good = False
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ (missing)")
            all_good = False
    
    return all_good

def test_database_connection():
    """Test database connection (optional)"""
    print("\n🔍 Testing database connection...")
    
    try:
        import psycopg2
        from app.config import settings
        
        # Try to connect to database
        conn = psycopg2.connect(
            host=settings.database.host,
            port=settings.database.port,
            user=settings.database.user,
            password=settings.database.password,
            database="postgres"  # Try connecting to default database first
        )
        conn.close()
        print("✅ Database connection successful (postgres)")
        return True
    except Exception as e:
        print(f"⚠️  Database connection failed: {e}")
        print("   This is expected if PostgreSQL is not running")
        return True  # Don't fail the test for this


def test_cache_infrastructure():
    """Test cache infrastructure (optional)"""
    print("\n🔍 Testing cache infrastructure...")
    
    try:
        from app.infra.cache import get_cache, get_cache_stats
        from app.config import settings
        
        # Test cache manager
        cache = get_cache()
        print("✅ Cache manager initialized")
        
        # Test configuration
        print(f"✅ Cache configuration loaded (Redis: {settings.redis.host}:{settings.redis.port})")
        
        # Test cache stats
        stats = get_cache_stats()
        print(f"✅ Cache stats retrieved (Status: {stats['status']})")
        
        return True
    except Exception as e:
        print(f"⚠️  Cache infrastructure test failed: {e}")
        print("   This is expected if Redis is not running")
        return True  # Don't fail the test for this


def test_schemas():
    """Test Pydantic schemas"""
    print("\n🔍 Testing Pydantic schemas...")
    
    try:
        from app.schemas import (
            CustomerCreate, RiskSegment, EmploymentStatus,
            CreditApplicationCreate, ApplicationStatus, LoanPurpose,
            CreditRiskPrediction, ModelType, HealthCheckResponse
        )
        
        # Test basic schema creation
        customer = CustomerCreate(
            customer_id="TEST001",
            customer_age=30,
            annual_income=60000.0,
            employment_status=EmploymentStatus.FULL_TIME,
            account_tenure=3,
            product_holdings=2,
            relationship_value=100000.0,
            risk_segment=RiskSegment.PRIME,
            behavioral_score=800.0,
            credit_score=700,
            city="London",
            last_activity_date=date.today()
        )
        print("✅ Customer schema validation")
        
        # Test enum values
        print(f"✅ Enum validation: {RiskSegment.PRIME.value}")
        
        # Test health check response
        health = HealthCheckResponse(status="healthy")
        print("✅ Health check schema validation")
        
        return True
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 FinRisk Setup Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Configuration", test_configuration),
        ("File Structure", test_file_structure),
        ("Database Connection", test_database_connection),
        ("Cache Infrastructure", test_cache_infrastructure),
        ("Pydantic Schemas", test_schemas)
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
        print("🎉 All tests passed! FinRisk is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
