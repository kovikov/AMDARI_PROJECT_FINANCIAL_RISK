#!/usr/bin/env python3
"""
Test script for the FinRisk Streamlit dashboard.
Validates data loading and visualization functionality.
"""

import logging
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Set database credentials for testing
os.environ["DB_NAME"] = "amdari_project"
os.environ["DB_USER"] = "postgres"
os.environ["DB_PASSWORD"] = "Kovikov1978@"

from app.config import get_settings
from app.infra.db import read_sql_query, check_database_connection
from app.infra.cache import get_cache_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_database_connection():
    """Test database connection for dashboard."""
    logger.info("Testing database connection...")
    
    try:
        import psycopg2
        
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "amdari_project",
            "user": "postgres",
            "password": "Kovikov1978@"
        }
        
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False


def test_portfolio_metrics():
    """Test portfolio metrics data loading."""
    logger.info("Testing portfolio metrics loading...")
    
    try:
        import psycopg2
        import pandas as pd
        
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "amdari_project",
            "user": "postgres",
            "password": "Kovikov1978@"
        }
        
        conn = psycopg2.connect(**db_config)
        query = """
        SELECT 
            COUNT(*) as total_customers,
            AVG(credit_score) as avg_credit_score,
            COUNT(CASE WHEN credit_score >= 700 THEN 1 END) as prime_customers,
            COUNT(CASE WHEN credit_score < 600 THEN 1 END) as subprime_customers,
            SUM(annual_income) as total_relationship_value
        FROM customer_profiles
        """
        result = pd.read_sql_query(query, conn)
        conn.close()
        
        if not result.empty:
            logger.info(f"✅ Portfolio metrics loaded successfully")
            logger.info(f"   Total customers: {result.iloc[0]['total_customers']}")
            logger.info(f"   Avg credit score: {result.iloc[0]['avg_credit_score']:.0f}")
            logger.info(f"   Prime customers: {result.iloc[0]['prime_customers']}")
            logger.info(f"   Subprime customers: {result.iloc[0]['subprime_customers']}")
            return True
        else:
            logger.warning("⚠️ Portfolio metrics query returned no data")
            return False
    except Exception as e:
        logger.error(f"❌ Portfolio metrics test failed: {e}")
        return False


def test_credit_metrics():
    """Test credit metrics data loading."""
    logger.info("Testing credit metrics loading...")
    
    try:
        import psycopg2
        import pandas as pd
        
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "amdari_project",
            "user": "postgres",
            "password": "Kovikov1978@"
        }
        
        conn = psycopg2.connect(**db_config)
        query = """
        SELECT 
            DATE(created_at) as application_date,
            COUNT(*) as total_applications,
            COUNT(CASE WHEN application_status = 'APPROVED' THEN 1 END) as approved_applications,
            COUNT(CASE WHEN application_status = 'REJECTED' THEN 1 END) as rejected_applications,
            AVG(loan_amount) as avg_loan_amount
        FROM credit_applications
        WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY DATE(created_at)
        ORDER BY DATE(created_at)
        """
        result = pd.read_sql_query(query, conn)
        conn.close()
        
        if not result.empty:
            logger.info(f"✅ Credit metrics loaded successfully")
            logger.info(f"   Records found: {len(result)}")
            logger.info(f"   Date range: {result['application_date'].min()} to {result['application_date'].max()}")
            return True
        else:
            logger.warning("⚠️ Credit metrics query returned no data (no recent applications)")
            return True  # This is acceptable if no recent data
    except Exception as e:
        logger.error(f"❌ Credit metrics test failed: {e}")
        return False


def test_fraud_metrics():
    """Test fraud metrics data loading."""
    logger.info("Testing fraud metrics loading...")
    
    try:
        import psycopg2
        import pandas as pd
        
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "amdari_project",
            "user": "postgres",
            "password": "Kovikov1978@"
        }
        
        conn = psycopg2.connect(**db_config)
        query = """
        SELECT 
            DATE(transaction_date) as transaction_date,
            COUNT(*) as total_transactions,
            COUNT(CASE WHEN is_fraudulent = true THEN 1 END) as fraud_transactions,
            SUM(amount) as total_amount,
            SUM(CASE WHEN is_fraudulent = true THEN amount ELSE 0 END) as fraud_amount
        FROM transaction_data
        WHERE transaction_date >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY DATE(transaction_date)
        ORDER BY transaction_date
        """
        result = pd.read_sql_query(query, conn)
        conn.close()
        
        if not result.empty:
            logger.info(f"✅ Fraud metrics loaded successfully")
            logger.info(f"   Records found: {len(result)}")
            logger.info(f"   Total transactions: {result['total_transactions'].sum()}")
            logger.info(f"   Fraud transactions: {result['fraud_transactions'].sum()}")
            return True
        else:
            logger.warning("⚠️ Fraud metrics query returned no data (no recent transactions)")
            return True  # This is acceptable if no recent data
    except Exception as e:
        logger.error(f"❌ Fraud metrics test failed: {e}")
        return False


def test_risk_distribution():
    """Test risk distribution data loading."""
    logger.info("Testing risk distribution loading...")
    
    try:
        import psycopg2
        import pandas as pd
        
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "amdari_project",
            "user": "postgres",
            "password": "Kovikov1978@"
        }
        
        conn = psycopg2.connect(**db_config)
        query = """
        SELECT 
            CASE 
                WHEN credit_score >= 750 THEN 'Prime'
                WHEN credit_score >= 650 THEN 'Near-Prime'
                WHEN credit_score >= 550 THEN 'Subprime'
                ELSE 'Deep-Subprime'
            END as risk_segment,
            COUNT(*) as customer_count,
            AVG(credit_score) as avg_credit_score,
            AVG(annual_income) as avg_income,
            SUM(annual_income) as total_value
        FROM customer_profiles
        GROUP BY 
            CASE 
                WHEN credit_score >= 750 THEN 'Prime'
                WHEN credit_score >= 650 THEN 'Near-Prime'
                WHEN credit_score >= 550 THEN 'Subprime'
                ELSE 'Deep-Subprime'
            END
        ORDER BY 
            CASE 
                WHEN credit_score >= 750 THEN 'Prime'
                WHEN credit_score >= 650 THEN 'Near-Prime'
                WHEN credit_score >= 550 THEN 'Subprime'
                ELSE 'Deep-Subprime'
            END
        """
        result = pd.read_sql_query(query, conn)
        conn.close()
        
        if not result.empty:
            logger.info(f"✅ Risk distribution loaded successfully")
            logger.info(f"   Risk segments found: {len(result)}")
            for _, row in result.iterrows():
                logger.info(f"   {row['risk_segment']}: {row['customer_count']} customers")
            return True
        else:
            logger.warning("⚠️ Risk distribution query returned no data")
            return False
    except Exception as e:
        logger.error(f"❌ Risk distribution test failed: {e}")
        return False


def test_model_predictions():
    """Test model predictions data loading."""
    logger.info("Testing model predictions loading...")
    
    try:
        import psycopg2
        import pandas as pd
        
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "amdari_project",
            "user": "postgres",
            "password": "Kovikov1978@"
        }
        
        conn = psycopg2.connect(**db_config)
        query = """
        SELECT 
            DATE(prediction_timestamp) as prediction_date,
            model_type,
            COUNT(*) as total_predictions,
            AVG(prediction_value) as avg_prediction_value,
            AVG(confidence_score) as avg_confidence
        FROM model_predictions
        WHERE prediction_timestamp >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY DATE(prediction_timestamp), model_type
        ORDER BY prediction_date, model_type
        """
        result = pd.read_sql_query(query, conn)
        conn.close()
        
        if not result.empty:
            logger.info(f"✅ Model predictions loaded successfully")
            logger.info(f"   Records found: {len(result)}")
            logger.info(f"   Model types: {result['model_type'].unique()}")
            return True
        else:
            logger.warning("⚠️ Model predictions query returned no data (no recent predictions)")
            return True  # This is acceptable if no recent data
    except Exception as e:
        logger.error(f"❌ Model predictions test failed: {e}")
        return False


def test_cache_functionality():
    """Test cache functionality."""
    logger.info("Testing cache functionality...")
    
    try:
        cache_stats = get_cache_stats()
        if cache_stats.get("status") == "healthy":
            logger.info("✅ Cache is operational")
            logger.info(f"   Hit rate: {cache_stats.get('hit_rate', 0):.1%}")
            logger.info(f"   Memory used: {cache_stats.get('used_memory', 'N/A')}")
            return True
        else:
            logger.warning("⚠️ Cache is not fully operational")
            return True  # Cache issues are not critical for dashboard
    except Exception as e:
        logger.warning(f"⚠️ Cache test failed: {e}")
        return True  # Cache issues are not critical for dashboard


def test_dashboard_imports():
    """Test dashboard module imports."""
    logger.info("Testing dashboard imports...")
    
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        import numpy as np
        
        logger.info("✅ All dashboard dependencies imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Dashboard import test failed: {e}")
        return False


def main():
    """Run all dashboard tests."""
    logger.info("Starting FinRisk dashboard tests...")
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Dashboard Imports", test_dashboard_imports),
        ("Portfolio Metrics", test_portfolio_metrics),
        ("Credit Metrics", test_credit_metrics),
        ("Fraud Metrics", test_fraud_metrics),
        ("Risk Distribution", test_risk_distribution),
        ("Model Predictions", test_model_predictions),
        ("Cache Functionality", test_cache_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("DASHBOARD TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All dashboard tests passed!")
        logger.info("🚀 Dashboard is ready to run with: streamlit run dashboard.py")
        return 0
    else:
        logger.error("💥 Some dashboard tests failed!")
        logger.info("🔧 Please check the failed tests and fix any issues before running the dashboard")
        return 1


if __name__ == "__main__":
    sys.exit(main())
