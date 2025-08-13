#!/usr/bin/env python3
"""
Initialize database schema with correct credentials.
"""

import logging
import psycopg2
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database():
    """Initialize the database schema."""
    # Database credentials for default database
    default_db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "postgres",
        "user": "postgres",
        "password": "Kovikov1978@"
    }
    
    # Database credentials for target database
    target_db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "AMDARI_PROJECT",
        "user": "postgres",
        "password": "Kovikov1978@"
    }
    
    # Read SQL schema file
    sql_file = Path("./sql/001_init_schema.sql")
    if not sql_file.exists():
        logger.error(f"Schema file not found: {sql_file}")
        return False
    
    try:
        # First, connect to default database and create AMDARI_PROJECT if it doesn't exist
        logger.info("Connecting to default database...")
        conn = psycopg2.connect(**default_db_config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # List all databases to see what exists
        cursor.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
        databases = cursor.fetchall()
        logger.info(f"Available databases: {[db[0] for db in databases]}")
        
        # Check if AMDARI_PROJECT database exists (case insensitive)
        cursor.execute("SELECT datname FROM pg_database WHERE LOWER(datname) = LOWER('AMDARI_PROJECT');")
        db_exists = cursor.fetchone()
        
        if not db_exists:
            logger.info("Creating AMDARI_PROJECT database...")
            try:
                cursor.execute("CREATE DATABASE AMDARI_PROJECT;")
                logger.info("Database AMDARI_PROJECT created successfully!")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info("Database AMDARI_PROJECT already exists!")
                else:
                    raise e
        else:
            logger.info(f"Database {db_exists[0]} already exists!")
            # Use the actual database name (case-sensitive)
            target_db_config["database"] = db_exists[0]
        
        cursor.close()
        conn.close()
        
        # Now connect to AMDARI_PROJECT database and initialize schema
        logger.info("Connecting to AMDARI_PROJECT database...")
        conn = psycopg2.connect(**target_db_config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Read SQL content
        with open(sql_file, 'r') as f:
            sql_content = f.read()
        
        # Split SQL into statements more carefully
        statements = []
        current_statement = ""
        in_dollar_quote = False
        dollar_tag = ""
        
        for line in sql_content.split('\n'):
            line = line.strip()
            if not line or line.startswith('--'):
                continue
                
            current_statement += line + " "
            
            # Handle dollar-quoted strings
            if '$$' in line:
                if not in_dollar_quote:
                    # Find the dollar tag
                    start = line.find('$$')
                    if start != -1:
                        end = line.find('$$', start + 2)
                        if end != -1:
                            # Single line dollar quote
                            continue
                        else:
                            # Multi-line dollar quote
                            in_dollar_quote = True
                            dollar_tag = line[start:start+2]
                else:
                    # Check if this line ends the dollar quote
                    if dollar_tag in line:
                        in_dollar_quote = False
                        dollar_tag = ""
            
            # Only split on semicolon if not in a dollar quote
            if not in_dollar_quote and line.endswith(';'):
                statements.append(current_statement.strip())
                current_statement = ""
        
        # Add any remaining statement
        if current_statement.strip():
            statements.append(current_statement.strip())
        
        logger.info(f"Executing {len(statements)} SQL statements...")
        for i, statement in enumerate(statements, 1):
            if statement.strip():
                logger.info(f"Executing statement {i}/{len(statements)}")
                try:
                    cursor.execute(statement)
                except Exception as e:
                    logger.error(f"Error in statement {i}: {e}")
                    logger.error(f"Statement: {statement[:100]}...")
                    raise
        
        cursor.close()
        conn.close()
        
        logger.info("Database schema initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

if __name__ == "__main__":
    init_database()
