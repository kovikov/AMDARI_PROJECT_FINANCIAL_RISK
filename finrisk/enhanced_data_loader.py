#!/usr/bin/env python3
"""
Enhanced data loading script for FinRisk application.
Loads seed CSV data into PostgreSQL database with proper schema validation and data cleaning.
"""

import pandas as pd
import numpy as np
import logging
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.config import get_settings
from app.infra.db import (
    get_db_session, execute_sql_file, insert_dataframe, 
    truncate_table, get_row_count, create_indexes
)
from app.features.preprocessing import DataValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedDataLoader:
    """Enhanced data loader for FinRisk seed data."""
    
    def __init__(self):
        # Set database credentials directly
        os.environ["DB_NAME"] = "amdari_project"
        os.environ["DB_USER"] = "postgres"
        os.environ["DB_PASSWORD"] = "Kovikov1978@"
        
        self.settings = get_settings()
        self.data_path = Path("./data/seed")
        self.sql_path = Path("./sql")
        self.loaded_data = {}
        
        # Data file mappings
        self.data_files = {
            'customer_profiles': 'customer_profiles.csv',
            'credit_bureau_data': 'credit_bureau_data.csv',
            'credit_applications': 'credit_applications.csv',
            'transaction_data': 'transaction_data.csv',
            'model_predictions': 'model_predictions.csv'
        }
        
        # Ensure directories exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.sql_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Enhanced data loader initialized with data path: {self.data_path}")
    
    def initialize_database(self) -> None:
        """Initialize database schema and indexes."""
        logger.info("Initializing database schema...")
        
        # Execute schema creation script using direct psycopg2
        schema_file = self.sql_path / "001_init_schema.sql"
        if not schema_file.exists():
            logger.error(f"Schema file not found: {schema_file}")
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
        
        # Use direct psycopg2 connection
        import psycopg2
        
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "amdari_project",
            "user": "postgres",
            "password": "Kovikov1978@"
        }
        
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        try:
            # Read SQL content
            with open(schema_file, 'r') as f:
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
                        # Handle "already exists" errors gracefully
                        error_msg = str(e).lower()
                        if any(keyword in error_msg for keyword in ['already exists', 'duplicate', 'exists']):
                            logger.warning(f"Statement {i} skipped (already exists): {e}")
                        else:
                            logger.error(f"Error in statement {i}: {e}")
                            logger.error(f"Statement: {statement[:100]}...")
                            raise
            
            logger.info("Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def load_csv_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files into pandas DataFrames.
        
        Returns:
            Dictionary of DataFrames
        """
        logger.info("Loading CSV data files...")
        
        for table_name, filename in self.data_files.items():
            file_path = self.data_path / filename
            
            if not file_path.exists():
                logger.warning(f"Data file not found: {file_path}")
                continue
            
            try:
                # Load CSV with proper data types
                df = pd.read_csv(file_path)
                
                # Clean and validate data
                df = self._clean_dataframe(df, table_name)
                
                self.loaded_data[table_name] = df
                logger.info(f"Loaded {len(df)} records from {filename}")
                
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
                raise
        
        return self.loaded_data
    
    def _clean_dataframe(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """
        Clean and validate DataFrame data.
        
        Args:
            df: DataFrame to clean
            table_name: Name of the table
            
        Returns:
            Cleaned DataFrame
        """
        # Remove any unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Handle specific table cleaning
        if table_name == 'customer_profiles':
            df = self._clean_customer_profiles(df)
        elif table_name == 'credit_applications':
            df = self._clean_credit_applications(df)
        elif table_name == 'transaction_data':
            df = self._clean_transaction_data(df)
        elif table_name == 'model_predictions':
            df = self._clean_model_predictions(df)
        elif table_name == 'credit_bureau_data':
            df = self._clean_credit_bureau_data(df)
        
        return df
    
    def _clean_customer_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean customer profiles data."""
        # Convert zip_code to string if it's numeric
        if 'zip_code' in df.columns:
            df['zip_code'] = df['zip_code'].astype(str)
        
        # Ensure proper data types
        if 'customer_age' in df.columns:
            df['customer_age'] = pd.to_numeric(df['customer_age'], errors='coerce')
        if 'annual_income' in df.columns:
            df['annual_income'] = pd.to_numeric(df['annual_income'], errors='coerce')
        if 'credit_score' in df.columns:
            df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce')
        if 'behavioral_score' in df.columns:
            df['behavioral_score'] = pd.to_numeric(df['behavioral_score'], errors='coerce')
        
        # Handle date columns
        if 'date_of_birth' in df.columns:
            df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
        if 'last_activity_date' in df.columns:
            df['last_activity_date'] = pd.to_datetime(df['last_activity_date'], errors='coerce')
        
        # Remove invalid records
        required_columns = ['customer_id']
        df = df.dropna(subset=required_columns)
        
        # Validate ranges if columns exist
        if 'customer_age' in df.columns:
            df = df[(df['customer_age'] >= 18) & (df['customer_age'] <= 120)]
        if 'annual_income' in df.columns:
            df = df[df['annual_income'] >= 0]
        if 'credit_score' in df.columns:
            df = df[(df['credit_score'] >= 300) & (df['credit_score'] <= 850)]
        
        return df
    
    def _clean_credit_applications(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean credit applications data."""
        # Handle NaN collateral_value
        if 'collateral_value' in df.columns:
            # Replace empty strings with NaN first
            df['collateral_value'] = df['collateral_value'].replace('', np.nan)
            df['collateral_value'] = df['collateral_value'].replace([np.inf, -np.inf], np.nan)
            # Convert to numeric, coercing errors to NaN
            df['collateral_value'] = pd.to_numeric(df['collateral_value'], errors='coerce')
            # Replace NaN with None for database insertion
            df['collateral_value'] = df['collateral_value'].where(pd.notna(df['collateral_value']), None)
        
        # Ensure proper data types
        if 'loan_amount' in df.columns:
            df['loan_amount'] = pd.to_numeric(df['loan_amount'], errors='coerce')
        if 'annual_income' in df.columns:
            df['annual_income'] = pd.to_numeric(df['annual_income'], errors='coerce')
        if 'debt_to_income_ratio' in df.columns:
            df['debt_to_income_ratio'] = pd.to_numeric(df['debt_to_income_ratio'], errors='coerce')
        if 'credit_score' in df.columns:
            df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce')
        if 'default_flag' in df.columns:
            df['default_flag'] = pd.to_numeric(df['default_flag'], errors='coerce')
        
        # Handle date columns
        if 'application_date' in df.columns:
            df['application_date'] = pd.to_datetime(df['application_date'], errors='coerce')
        
        # Remove invalid records
        required_columns = ['application_id', 'customer_id']
        df = df.dropna(subset=required_columns)
        
        # Validate ranges
        if 'loan_amount' in df.columns:
            df = df[df['loan_amount'] > 0]
        if 'annual_income' in df.columns:
            df = df[df['annual_income'] >= 0]
        if 'default_flag' in df.columns:
            df = df[df['default_flag'].isin([0, 1])]
        
        return df
    
    def _clean_transaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean transaction data."""
        # Ensure proper data types
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        if 'fraud_flag' in df.columns:
            df['fraud_flag'] = pd.to_numeric(df['fraud_flag'], errors='coerce')
        
        # Handle date columns
        if 'transaction_date' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        
        # Remove invalid records
        required_columns = ['transaction_id', 'customer_id']
        df = df.dropna(subset=required_columns)
        
        # Validate ranges
        if 'amount' in df.columns:
            df = df[df['amount'] > 0]
        if 'fraud_flag' in df.columns:
            df = df[df['fraud_flag'].isin([0, 1])]
        
        return df
    
    def _clean_credit_bureau_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean credit bureau data."""
        # Handle date columns
        if 'report_date' in df.columns:
            df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
        
        # Ensure numeric columns
        numeric_columns = [
            'credit_score', 'credit_utilization', 'length_of_credit_history',
            'number_of_accounts', 'derogatory_marks', 'inquiries_last_6_months',
            'public_records'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove invalid records
        required_columns = ['record_id', 'customer_id']
        df = df.dropna(subset=required_columns)
        
        return df
    
    def _clean_model_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean model predictions data."""
        # Handle NaN application_id
        if 'application_id' in df.columns:
            # Replace empty strings with NaN first
            df['application_id'] = df['application_id'].replace('', np.nan)
            df['application_id'] = df['application_id'].replace([np.inf, -np.inf], np.nan)
            # Replace NaN with None for database insertion
            df['application_id'] = df['application_id'].where(pd.notna(df['application_id']), None)
        
        # Handle features_used column (convert string to dict, then to JSON string)
        if 'features_used' in df.columns:
            import json
            df['features_used'] = df['features_used'].apply(
                lambda x: json.dumps(eval(x)) if isinstance(x, str) and x.startswith('{') else None
            )
        
        # Handle date columns
        if 'prediction_date' in df.columns:
            df['prediction_date'] = pd.to_datetime(df['prediction_date'], errors='coerce')
        if 'prediction_timestamp' in df.columns:
            df['prediction_timestamp'] = pd.to_datetime(df['prediction_timestamp'], errors='coerce')
        
        # Ensure numeric columns
        if 'risk_score' in df.columns:
            df['risk_score'] = pd.to_numeric(df['risk_score'], errors='coerce')
        if 'fraud_probability' in df.columns:
            df['fraud_probability'] = pd.to_numeric(df['fraud_probability'], errors='coerce')
        if 'prediction_value' in df.columns:
            df['prediction_value'] = pd.to_numeric(df['prediction_value'], errors='coerce')
        if 'confidence_score' in df.columns:
            df['confidence_score'] = pd.to_numeric(df['confidence_score'], errors='coerce')
        
        # Remove invalid records
        required_columns = ['prediction_id', 'customer_id']
        df = df.dropna(subset=required_columns)
        
        return df
    
    def load_data_to_database(self, force_reload: bool = False) -> None:
        """Load cleaned data into database tables."""
        logger.info("Loading data into database...")
        
        # Get existing customer IDs for foreign key validation
        existing_customers = set()
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
            cursor.execute("SELECT customer_id FROM customer_profiles")
            existing_customers = {row[0] for row in cursor.fetchall()}
            cursor.close()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not fetch existing customers: {e}")
        
        for table_name, df in self.loaded_data.items():
            if df.empty:
                logger.warning(f"No data to load for {table_name}")
                continue
            
            # Filter for existing customers if this table has customer_id
            if 'customer_id' in df.columns and existing_customers:
                original_count = len(df)
                df_filtered = df[df['customer_id'].isin(existing_customers)]
                filtered_count = len(df_filtered)
                
                if filtered_count < original_count:
                    logger.warning(f"Filtered out {original_count - filtered_count} records for non-existent customers in {table_name}")
                
                if df_filtered.empty:
                    logger.warning(f"No valid records to insert for {table_name}")
                    continue
                
                df = df_filtered
            
            # Additional filtering for model_predictions to check application_id references
            if table_name == 'model_predictions' and 'application_id' in df.columns:
                # Get existing application IDs
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
                    cursor.execute("SELECT application_id FROM credit_applications")
                    existing_applications = {row[0] for row in cursor.fetchall()}
                    cursor.close()
                    conn.close()
                    
                    # Filter out predictions with invalid application IDs
                    original_count = len(df)
                    df_filtered = df[df['application_id'].isin(existing_applications) | df['application_id'].isna()]
                    filtered_count = len(df_filtered)
                    
                    if filtered_count < original_count:
                        logger.warning(f"Filtered out {original_count - filtered_count} records for non-existent application IDs in {table_name}")
                    
                    if df_filtered.empty:
                        logger.warning(f"No valid records to insert for {table_name}")
                        continue
                    
                    df = df_filtered
                    
                except Exception as e:
                    logger.warning(f"Could not fetch existing application IDs: {e}")
            
            try:
                # Insert data with conflict handling
                self._insert_with_conflict_handling(table_name, df)
                logger.info(f"Successfully loaded {len(df)} records into {table_name}")
                
            except Exception as e:
                logger.error(f"Failed to load data into {table_name}: {e}")
                raise
    
    def _insert_with_conflict_handling(self, table_name: str, df: pd.DataFrame) -> None:
        """Insert DataFrame with appropriate conflict handling."""
        if df.empty:
            return
        
        # Import psycopg2 for direct database operations
        import psycopg2
        
        # Database credentials
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "amdari_project",
            "user": "postgres",
            "password": "Kovikov1978@"
        }
        
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        try:
            # Get column names
            columns = list(df.columns)
            
            # Create placeholders for SQL
            placeholders = ', '.join(['%s'] * len(columns))
            column_names = ', '.join(columns)
            
            # Prepare INSERT statement with ON CONFLICT handling
            if table_name == "customer_profiles":
                # For customer_profiles, use email as conflict key
                conflict_columns = ["email"]
                update_columns = [col for col in columns if col not in conflict_columns]
                update_set = ', '.join([f"{col} = EXCLUDED.{col}" for col in update_columns])
                insert_query = f"""
                    INSERT INTO {table_name} ({column_names}) 
                    VALUES ({placeholders})
                    ON CONFLICT (email) DO UPDATE SET {update_set}
                """
            elif table_name == "credit_applications":
                # For credit_applications, use application_id as conflict key
                conflict_columns = ["application_id"]
                update_columns = [col for col in columns if col not in conflict_columns]
                update_set = ', '.join([f"{col} = EXCLUDED.{col}" for col in update_columns])
                insert_query = f"""
                    INSERT INTO {table_name} ({column_names}) 
                    VALUES ({placeholders})
                    ON CONFLICT (application_id) DO UPDATE SET {update_set}
                """
            elif table_name == "model_predictions":
                # For model_predictions, use prediction_id as conflict key
                conflict_columns = ["prediction_id"]
                update_columns = [col for col in columns if col not in conflict_columns]
                update_set = ', '.join([f"{col} = EXCLUDED.{col}" for col in update_columns])
                insert_query = f"""
                    INSERT INTO {table_name} ({column_names}) 
                    VALUES ({placeholders})
                    ON CONFLICT (prediction_id) DO UPDATE SET {update_set}
                """
            else:
                # For other tables, use simple INSERT with ON CONFLICT DO NOTHING
                insert_query = f"""
                    INSERT INTO {table_name} ({column_names}) 
                    VALUES ({placeholders})
                    ON CONFLICT DO NOTHING
                """
            
            # Convert DataFrame to list of tuples, handling None values
            data = []
            for _, row in df.iterrows():
                row_data = []
                for value in row.values:
                    if pd.isna(value):
                        row_data.append(None)
                    elif isinstance(value, (np.integer, np.floating)):
                        row_data.append(value.item())
                    else:
                        row_data.append(value)
                data.append(tuple(row_data))
            
            # Execute batch insert
            cursor.executemany(insert_query, data)
            conn.commit()
            
            logger.info(f"Inserted/updated {len(data)} rows into {table_name}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting data into {table_name}: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def verify_data_integrity(self) -> None:
        """Verify data integrity and relationships."""
        logger.info("Verifying data integrity...")
        
        # Use direct psycopg2 connection instead of SQLAlchemy
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
        
        try:
            # Check table row counts
            tables = list(self.data_files.keys())
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"Table {table}: {count} rows")
            
            # Check foreign key relationships
            logger.info("Checking foreign key relationships...")
            
            # Check credit applications reference valid customers
            cursor.execute("""
                SELECT COUNT(*) as invalid_apps
                FROM credit_applications ca
                LEFT JOIN customer_profiles cp ON ca.customer_id = cp.customer_id
                WHERE cp.customer_id IS NULL
            """)
            invalid_apps = cursor.fetchone()[0]
            logger.info(f"Credit applications with invalid customer references: {invalid_apps}")
            
            # Check transactions reference valid customers
            cursor.execute("""
                SELECT COUNT(*) as invalid_trans
                FROM transaction_data td
                LEFT JOIN customer_profiles cp ON td.customer_id = cp.customer_id
                WHERE cp.customer_id IS NULL
            """)
            invalid_trans = cursor.fetchone()[0]
            logger.info(f"Transactions with invalid customer references: {invalid_trans}")
            
            # Check model predictions reference valid customers
            cursor.execute("""
                SELECT COUNT(*) as invalid_preds
                FROM model_predictions mp
                LEFT JOIN customer_profiles cp ON mp.customer_id = cp.customer_id
                WHERE cp.customer_id IS NULL
            """)
            invalid_preds = cursor.fetchone()[0]
            logger.info(f"Model predictions with invalid customer references: {invalid_preds}")
            
        except Exception as e:
            logger.error(f"Error verifying data integrity: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def run(self, force_reload: bool = False, verify_only: bool = False) -> None:
        """Run the complete data loading process."""
        logger.info("Starting enhanced data loading process...")
        
        try:
            if not verify_only:
                # Initialize database
                self.initialize_database()
                
                # Load and clean CSV data
                self.load_csv_data()
                
                # Load data into database
                self.load_data_to_database(force_reload)
            
            # Verify data integrity
            self.verify_data_integrity()
            
            logger.info("Enhanced data loading completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in enhanced data loading: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced data loader for FinRisk database")
    parser.add_argument("--force-reload", action="store_true", help="Truncate tables before loading data")
    parser.add_argument("--verify-only", action="store_true", help="Only verify data integrity without loading")
    
    args = parser.parse_args()
    
    try:
        loader = EnhancedDataLoader()
        loader.run(force_reload=args.force_reload, verify_only=args.verify_only)
    except Exception as e:
        logger.error(f"Error during enhanced data loading: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
