#!/usr/bin/env python3
"""
Test database connection with provided credentials.
"""

import os
import sys
from pathlib import Path

# Set environment variables BEFORE importing any app modules
os.environ["DB_NAME"] = "AMDARI_PROJECT"
os.environ["DB_USER"] = "postgres"
os.environ["DB_PASSWORD"] = "Kovikov1978@"

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

import psycopg2
from app.config import get_settings

def test_database_connection():
    """Test database connection with current settings."""
    print("Testing database connection...")
    
    # Test with provided credentials directly
    test_credentials = {
        "host": "localhost",
        "port": 5432,
        "database": "postgres",  # Connect to default database first
        "user": "postgres",
        "password": "Kovikov1978@"
    }
    
    print(f"Testing with credentials:")
    print(f"Host: {test_credentials['host']}")
    print(f"Port: {test_credentials['port']}")
    print(f"Database: {test_credentials['database']}")
    print(f"User: {test_credentials['user']}")
    print(f"Password: {'*' * len(test_credentials['password'])}")
    
    try:
        # Test direct connection with psycopg2
        print("\nTesting direct psycopg2 connection...")
        conn = psycopg2.connect(**test_credentials)
        
        # Test a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"PostgreSQL version: {version[0]}")
        
        # Test if the database exists and list tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        print(f"Existing tables: {[table[0] for table in tables]}")
        
        # Check if AMDARI_PROJECT database exists
        cursor.execute("""
            SELECT datname FROM pg_database WHERE datname = 'AMDARI_PROJECT';
        """)
        db_exists = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if db_exists:
            print("Database 'AMDARI_PROJECT' exists!")
        else:
            print("Database 'AMDARI_PROJECT' does not exist. Creating it...")
            # Create a new connection with autocommit for database creation
            conn = psycopg2.connect(**test_credentials)
            conn.autocommit = True
            cursor = conn.cursor()
            cursor.execute("CREATE DATABASE AMDARI_PROJECT;")
            cursor.close()
            conn.close()
            print("Database 'AMDARI_PROJECT' created successfully!")
        
        print("Database connection successful!")
        return True
        
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

if __name__ == "__main__":
    test_database_connection()
