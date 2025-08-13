#!/usr/bin/env python3
"""
Test script for the enhanced data loader.
"""

import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_data_loader import EnhancedDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_enhanced_data_loader():
    """Test the enhanced data loader functionality."""
    logger.info("Testing enhanced data loader...")
    
    try:
        # Create data loader instance
        loader = EnhancedDataLoader()
        logger.info("Enhanced data loader created successfully")
        
        # Test CSV data loading
        logger.info("Testing CSV data loading...")
        loaded_data = loader.load_csv_data()
        
        # Print summary of loaded data
        for table_name, df in loaded_data.items():
            logger.info(f"Table {table_name}: {len(df)} records loaded")
            if not df.empty:
                logger.info(f"  Columns: {list(df.columns)}")
                logger.info(f"  Data types: {df.dtypes.to_dict()}")
                logger.info(f"  Sample data:")
                logger.info(f"    {df.head(2).to_dict('records')}")
        
        # Test data cleaning
        logger.info("Testing data cleaning...")
        for table_name, df in loaded_data.items():
            if not df.empty:
                logger.info(f"Cleaned {table_name}: {len(df)} records")
                # Check for any remaining NaN values in required columns
                if table_name == 'customer_profiles':
                    nan_count = df['customer_id'].isna().sum()
                    logger.info(f"  NaN customer_ids: {nan_count}")
                elif table_name == 'credit_applications':
                    nan_count = df[['application_id', 'customer_id']].isna().sum().sum()
                    logger.info(f"  NaN in required columns: {nan_count}")
        
        logger.info("Enhanced data loader test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


def test_data_validation():
    """Test data validation functionality."""
    logger.info("Testing data validation...")
    
    try:
        loader = EnhancedDataLoader()
        
        # Test customer profiles validation
        logger.info("Testing customer profiles validation...")
        customer_data = {
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
            'customer_age': [25, 150, 30],  # Invalid age
            'annual_income': [50000, 75000, -1000],  # Invalid income
            'credit_score': [750, 900, 650],  # Invalid credit score
            'zip_code': [12345, 67890, 11111]
        }
        
        import pandas as pd
        df = pd.DataFrame(customer_data)
        cleaned_df = loader._clean_customer_profiles(df)
        logger.info(f"Original records: {len(df)}")
        logger.info(f"Cleaned records: {len(cleaned_df)}")
        
        # Test credit applications validation
        logger.info("Testing credit applications validation...")
        app_data = {
            'application_id': ['APP_001', 'APP_002', 'APP_003'],
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
            'loan_amount': [10000, -5000, 20000],  # Invalid loan amount
            'default_flag': [0, 1, 2]  # Invalid default flag
        }
        
        df = pd.DataFrame(app_data)
        cleaned_df = loader._clean_credit_applications(df)
        logger.info(f"Original applications: {len(df)}")
        logger.info(f"Cleaned applications: {len(cleaned_df)}")
        
        logger.info("Data validation test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Data validation test failed: {e}")
        return False


def test_database_operations():
    """Test database operations."""
    logger.info("Testing database operations...")
    
    try:
        loader = EnhancedDataLoader()
        
        # Test database initialization
        logger.info("Testing database initialization...")
        loader.initialize_database()
        logger.info("Database initialization completed")
        
        # Test data integrity verification
        logger.info("Testing data integrity verification...")
        loader.verify_data_integrity()
        logger.info("Data integrity verification completed")
        
        logger.info("Database operations test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Database operations test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting enhanced data loader tests...")
    
    tests = [
        ("Enhanced Data Loader", test_enhanced_data_loader),
        ("Data Validation", test_data_validation),
        ("Database Operations", test_database_operations)
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
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
