#!/usr/bin/env python3
"""
FinRisk Database Setup Script
Initializes the PostgreSQL database with all schemas, tables, and indexes.
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_database_connection(db_name=None):
    """Create database connection"""
    try:
        # Get connection parameters from environment variables
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_user = os.getenv('DB_USER', 'finrisk_user')
        db_password = os.getenv('DB_PASSWORD', 'finrisk_pass')
        
        if db_name:
            db_name = db_name
        else:
            db_name = os.getenv('DB_NAME', 'finrisk_db')
        
        # Connect to database
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return None

def create_database():
    """Create the database if it doesn't exist"""
    try:
        # Connect to default postgres database
        conn = get_database_connection('postgres')
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Check if database exists
        db_name = os.getenv('DB_NAME', 'finrisk_db')
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cursor.fetchone()
        
        if not exists:
            logger.info(f"Creating database: {db_name}")
            cursor.execute(f"CREATE DATABASE {db_name}")
            logger.info(f"Database {db_name} created successfully")
        else:
            logger.info(f"Database {db_name} already exists")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        return False

def execute_sql_file(conn, file_path):
    """Execute SQL file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            sql_content = file.read()
        
        cursor = conn.cursor()
        cursor.execute(sql_content)
        cursor.close()
        
        logger.info(f"Successfully executed: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to execute {file_path}: {e}")
        return False

def setup_database():
    """Main function to set up the database"""
    logger.info("Starting FinRisk database setup...")
    
    # Create database if it doesn't exist
    if not create_database():
        logger.error("Failed to create database")
        return False
    
    # Connect to the finrisk database
    conn = get_database_connection()
    if not conn:
        logger.error("Failed to connect to finrisk database")
        return False
    
    try:
        # Get the SQL directory path
        current_dir = Path(__file__).parent
        sql_dir = current_dir.parent / 'sql'
        
        # Execute SQL files in order
        sql_files = [
            '001_init_schema.sql',
            '010_indexes.sql', 
            '020_sample_views.sql'
        ]
        
        for sql_file in sql_files:
            file_path = sql_dir / sql_file
            if file_path.exists():
                logger.info(f"Executing {sql_file}...")
                if not execute_sql_file(conn, file_path):
                    logger.error(f"Failed to execute {sql_file}")
                    return False
            else:
                logger.warning(f"SQL file not found: {file_path}")
        
        logger.info("Database setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False
    finally:
        conn.close()

def verify_setup():
    """Verify that the database setup was successful"""
    conn = get_database_connection()
    if not conn:
        logger.error("Failed to connect to database for verification")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Check if main tables exist
        tables_to_check = [
            'finrisk.customer_profiles',
            'finrisk.credit_bureau_data', 
            'finrisk.credit_applications',
            'finrisk.transaction_data',
            'finrisk.model_predictions',
            'audit.decision_log',
            'monitoring.drift_detection'
        ]
        
        for table in tables_to_check:
            cursor.execute(f"SELECT 1 FROM {table} LIMIT 1")
            logger.info(f"✓ Table {table} exists and is accessible")
        
        # Check if views exist
        views_to_check = [
            'finrisk.customer_risk_summary',
            'finrisk.credit_application_performance',
            'finrisk.fraud_detection_analysis'
        ]
        
        for view in views_to_check:
            cursor.execute(f"SELECT 1 FROM {view} LIMIT 1")
            logger.info(f"✓ View {view} exists and is accessible")
        
        cursor.close()
        logger.info("Database verification completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Setup database
    if setup_database():
        # Verify setup
        if verify_setup():
            logger.info("🎉 FinRisk database is ready!")
            sys.exit(0)
        else:
            logger.error("❌ Database verification failed")
            sys.exit(1)
    else:
        logger.error("❌ Database setup failed")
        sys.exit(1)

