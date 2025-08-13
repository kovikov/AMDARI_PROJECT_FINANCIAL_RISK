#!/usr/bin/env python3
"""
Simple data loading script for FinRisk application.
Loads seed CSV data into PostgreSQL database with correct credentials.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.schemas.customers import CustomerProfile
from app.schemas.applications import CreditApplication
from app.schemas.predictions import ModelPrediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleDataLoader:
    """Simple data loader for FinRisk application."""
    
    def __init__(self):
        """Initialize the data loader."""
        # Database credentials
        self.db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "amdari_project",
            "user": "postgres",
            "password": "Kovikov1978@"
        }
        
        self.data_dir = Path("./data/seed")
        self.sql_dir = Path("./sql")
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sql_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data loader initialized with data directory: {self.data_dir}")
        logger.info(f"SQL directory: {self.sql_dir}")
    
    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.db_config)
    
    def load_data(self, force_reload: bool = False, verify_only: bool = False) -> None:
        """Load all seed data into the database."""
        logger.info("Starting data loading process...")
        
        try:
            if verify_only:
                logger.info("Verification mode - checking data integrity...")
                self._verify_data_integrity()
                return
            
            # Load customer profiles
            logger.info("Loading customer profiles...")
            self._load_customer_profiles(force_reload)
            
            # Load credit applications
            logger.info("Loading credit applications...")
            self._load_credit_applications(force_reload)
            
            # Load transaction data
            logger.info("Loading transaction data...")
            self._load_transaction_data(force_reload)
            
            # Load credit bureau data
            logger.info("Loading credit bureau data...")
            self._load_credit_bureau_data(force_reload)
            
            # Load model predictions
            logger.info("Loading model predictions...")
            self._load_model_predictions(force_reload)
            
            # Verify data integrity
            logger.info("Verifying data integrity...")
            self._verify_data_integrity()
            
            logger.info("Data loading completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in data loading: {e}")
            raise
    
    def _load_customer_profiles(self, force_reload: bool = False) -> None:
        """Load customer profiles data."""
        csv_file = self.data_dir / "customer_profiles.csv"
        
        if not csv_file.exists():
            logger.warning(f"Customer profiles CSV not found: {csv_file}")
            return
        
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} customer profiles from CSV")
        
        # Validate data using Pydantic schema
        validated_data = []
        for _, row in df.iterrows():
            try:
                # Convert date strings to datetime objects
                if 'date_of_birth' in row and pd.notna(row['date_of_birth']):
                    row['date_of_birth'] = pd.to_datetime(row['date_of_birth']).date()
                
                # Convert zip_code to string if it's numeric
                if 'zip_code' in row and pd.notna(row['zip_code']):
                    row['zip_code'] = str(int(row['zip_code']))
                
                customer = CustomerProfile(**row.to_dict())
                validated_data.append(customer.model_dump())
            except Exception as e:
                logger.warning(f"Invalid customer data: {row.get('customer_id', 'unknown')} - {e}")
        
        if validated_data:
            self._insert_dataframe("customer_profiles", pd.DataFrame(validated_data))
            logger.info(f"Inserted {len(validated_data)} valid customer profiles")
    
    def _load_credit_applications(self, force_reload: bool = False) -> None:
        """Load credit applications data."""
        csv_file = self.data_dir / "credit_applications.csv"
        
        if not csv_file.exists():
            logger.warning(f"Credit applications CSV not found: {csv_file}")
            return
        
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} credit applications from CSV")
        
        # Validate data using Pydantic schema
        validated_data = []
        for _, row in df.iterrows():
            try:
                # Handle NaN collateral_value
                if 'collateral_value' in row and pd.isna(row['collateral_value']):
                    row['collateral_value'] = None
                
                credit_app = CreditApplication(**row.to_dict())
                validated_data.append(credit_app.model_dump())
            except Exception as e:
                logger.warning(f"Invalid credit application data: {row.get('application_id', 'unknown')} - {e}")
        
        if validated_data:
            # Filter out applications for non-existent customers
            valid_applications = []
            existing_customers = set()
            
            # Get existing customer IDs
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT customer_id FROM customer_profiles")
            existing_customers = {row[0] for row in cursor.fetchall()}
            cursor.close()
            conn.close()
            
            for app in validated_data:
                if app['customer_id'] in existing_customers:
                    valid_applications.append(app)
                else:
                    logger.warning(f"Skipping credit application {app.get('application_id', 'unknown')} - customer {app['customer_id']} not found")
            
            if valid_applications:
                self._insert_dataframe("credit_applications", pd.DataFrame(valid_applications))
                logger.info(f"Inserted {len(valid_applications)} valid credit applications")
            else:
                logger.warning("No valid credit applications to insert")
    
    def _load_transaction_data(self, force_reload: bool = False) -> None:
        """Load transaction data."""
        csv_file = self.data_dir / "transaction_data.csv"
        
        if not csv_file.exists():
            logger.warning(f"Transaction data CSV not found: {csv_file}")
            return
        
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} transactions from CSV")
        
        # Convert timestamp columns
        if 'transaction_date' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # Filter out transactions for non-existent customers
        existing_customers = set()
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT customer_id FROM customer_profiles")
        existing_customers = {row[0] for row in cursor.fetchall()}
        cursor.close()
        conn.close()
        
        # Filter DataFrame to only include existing customers
        df_filtered = df[df['customer_id'].isin(existing_customers)]
        
        if len(df_filtered) < len(df):
            logger.warning(f"Filtered out {len(df) - len(df_filtered)} transactions for non-existent customers")
        
        if not df_filtered.empty:
            self._insert_dataframe("transaction_data", df_filtered)
            logger.info(f"Inserted {len(df_filtered)} transactions")
        else:
            logger.warning("No valid transactions to insert")
    
    def _load_credit_bureau_data(self, force_reload: bool = False) -> None:
        """Load credit bureau data."""
        csv_file = self.data_dir / "credit_bureau_data.csv"
        
        if not csv_file.exists():
            logger.warning(f"Credit bureau data CSV not found: {csv_file}")
            return
        
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} credit bureau records from CSV")
        
        # Convert timestamp columns
        if 'report_date' in df.columns:
            df['report_date'] = pd.to_datetime(df['report_date'])
        
        # Filter out records for non-existent customers
        existing_customers = set()
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT customer_id FROM customer_profiles")
        existing_customers = {row[0] for row in cursor.fetchall()}
        cursor.close()
        conn.close()
        
        # Filter DataFrame to only include existing customers
        df_filtered = df[df['customer_id'].isin(existing_customers)]
        
        if len(df_filtered) < len(df):
            logger.warning(f"Filtered out {len(df) - len(df_filtered)} credit bureau records for non-existent customers")
        
        if not df_filtered.empty:
            self._insert_dataframe("credit_bureau_data", df_filtered)
            logger.info(f"Inserted {len(df_filtered)} credit bureau records")
        else:
            logger.warning("No valid credit bureau records to insert")
    
    def _load_model_predictions(self, force_reload: bool = False) -> None:
        """Load model predictions data."""
        csv_file = self.data_dir / "model_predictions.csv"
        
        if not csv_file.exists():
            logger.warning(f"Model predictions CSV not found: {csv_file}")
            return
        
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} model predictions from CSV")
        
        # Validate data using Pydantic schema
        validated_data = []
        for _, row in df.iterrows():
            try:
                # Convert timestamp columns
                if 'prediction_timestamp' in row and pd.notna(row['prediction_timestamp']):
                    row['prediction_timestamp'] = pd.to_datetime(row['prediction_timestamp'])
                
                prediction = ModelPrediction(**row.to_dict())
                validated_data.append(prediction.model_dump())
            except Exception as e:
                logger.warning(f"Invalid model prediction data: {row.get('prediction_id', 'unknown')} - {e}")
        
        if validated_data:
            # Filter out predictions for non-existent customers
            valid_predictions = []
            existing_customers = set()
            
            # Get existing customer IDs
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT customer_id FROM customer_profiles")
            existing_customers = {row[0] for row in cursor.fetchall()}
            cursor.close()
            conn.close()
            
            for pred in validated_data:
                if pred['customer_id'] in existing_customers:
                    valid_predictions.append(pred)
                else:
                    logger.warning(f"Skipping model prediction {pred.get('prediction_id', 'unknown')} - customer {pred['customer_id']} not found")
            
            if valid_predictions:
                self._insert_dataframe("model_predictions", pd.DataFrame(valid_predictions))
                logger.info(f"Inserted {len(valid_predictions)} valid model predictions")
            else:
                logger.warning("No valid model predictions to insert")
    
    def _insert_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Insert DataFrame into database table."""
        if df.empty:
            return
        
        conn = self.get_connection()
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
            
            # Convert DataFrame to list of tuples
            data = [tuple(row) for row in df.values]
            
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
    
    def _verify_data_integrity(self) -> None:
        """Verify data integrity and relationships."""
        logger.info("Verifying data integrity...")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Check table row counts
            tables = [
                "customer_profiles",
                "credit_applications", 
                "transaction_data",
                "credit_bureau_data",
                "model_predictions"
            ]
            
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load seed data into FinRisk database")
    parser.add_argument("--force-reload", action="store_true", help="Truncate tables before loading data")
    parser.add_argument("--verify-only", action="store_true", help="Only verify data integrity without loading")
    
    args = parser.parse_args()
    
    try:
        loader = SimpleDataLoader()
        loader.load_data(force_reload=args.force_reload, verify_only=args.verify_only)
    except Exception as e:
        logger.error(f"Error during data loading: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
