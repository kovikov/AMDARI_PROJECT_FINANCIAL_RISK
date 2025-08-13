#!/usr/bin/env python3
"""
Data loading script for FinRisk application.
Loads seed CSV data into PostgreSQL database with proper schema validation.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import text

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.config import get_settings, reload_settings
from app.infra.db import get_db_session, execute_sql_file, insert_dataframe, truncate_table, get_row_count, reset_engine
from app.schemas.customers import CustomerProfile
from app.schemas.applications import CreditApplication
from app.schemas.predictions import ModelPrediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for FinRisk application."""
    
    def __init__(self):
        """Initialize the data loader."""
        # Set database credentials directly
        os.environ["DB_NAME"] = "amdari_project"
        os.environ["DB_USER"] = "postgres"
        os.environ["DB_PASSWORD"] = "Kovikov1978@"
        
        # Reset the database engine to use new credentials
        reset_engine()
        
        # Reload settings with new environment variables
        self.settings = reload_settings()
        self.data_dir = Path(self.settings.data.seed_path)
        self.sql_dir = Path("./sql")
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sql_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data loader initialized with data directory: {self.data_dir}")
        logger.info(f"SQL directory: {self.sql_dir}")
        logger.info(f"Database URL: {self.settings.database.url}")
    
    def load_data(self, force_reload: bool = False, verify_only: bool = False) -> None:
        """Load all seed data into the database."""
        logger.info("Starting data loading process...")
        
        try:
            # Initialize database schema
            logger.info("Initializing database schema...")
            self._init_schema()
            
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
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        schema_file = self.sql_dir / "001_init_schema.sql"
        if not schema_file.exists():
            logger.error(f"Schema file not found: {schema_file}")
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
        
        execute_sql_file(str(schema_file))
        logger.info("Database schema initialized successfully")
    
    def _load_customer_profiles(self, force_reload: bool = False) -> None:
        """Load customer profiles data."""
        csv_file = self.data_dir / "customer_profiles.csv"
        
        if not csv_file.exists():
            logger.warning(f"Customer profiles CSV not found: {csv_file}")
            return
        
        if force_reload:
            truncate_table("customer_profiles")
        
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} customer profiles from CSV")
        
        # Validate data using Pydantic schema
        validated_data = []
        for _, row in df.iterrows():
            try:
                # Convert date strings to datetime objects
                if 'date_of_birth' in row and pd.notna(row['date_of_birth']):
                    row['date_of_birth'] = pd.to_datetime(row['date_of_birth']).date()
                
                customer = CustomerProfile(**row.to_dict())
                validated_data.append(customer.dict())
            except Exception as e:
                logger.warning(f"Invalid customer data: {row.get('customer_id', 'unknown')} - {e}")
        
        if validated_data:
            insert_dataframe("customer_profiles", pd.DataFrame(validated_data))
            logger.info(f"Inserted {len(validated_data)} valid customer profiles")
    
    def _load_credit_applications(self, force_reload: bool = False) -> None:
        """Load credit applications data."""
        csv_file = self.data_dir / "credit_applications.csv"
        
        if not csv_file.exists():
            logger.warning(f"Credit applications CSV not found: {csv_file}")
            return
        
        if force_reload:
            truncate_table("credit_applications")
        
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} credit applications from CSV")
        
        # Validate data using Pydantic schema
        validated_data = []
        for _, row in df.iterrows():
            try:
                credit_app = CreditApplication(**row.to_dict())
                validated_data.append(credit_app.dict())
            except Exception as e:
                logger.warning(f"Invalid credit application data: {row.get('application_id', 'unknown')} - {e}")
        
        if validated_data:
            insert_dataframe("credit_applications", pd.DataFrame(validated_data))
            logger.info(f"Inserted {len(validated_data)} valid credit applications")
    
    def _load_transaction_data(self, force_reload: bool = False) -> None:
        """Load transaction data."""
        csv_file = self.data_dir / "transaction_data.csv"
        
        if not csv_file.exists():
            logger.warning(f"Transaction data CSV not found: {csv_file}")
            return
        
        if force_reload:
            truncate_table("transaction_data")
        
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} transactions from CSV")
        
        # Convert timestamp columns
        if 'transaction_timestamp' in df.columns:
            df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])
        
        insert_dataframe("transaction_data", df)
        logger.info(f"Inserted {len(df)} transactions")
    
    def _load_credit_bureau_data(self, force_reload: bool = False) -> None:
        """Load credit bureau data."""
        csv_file = self.data_dir / "credit_bureau_data.csv"
        
        if not csv_file.exists():
            logger.warning(f"Credit bureau data CSV not found: {csv_file}")
            return
        
        if force_reload:
            truncate_table("credit_bureau_data")
        
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} credit bureau records from CSV")
        
        # Convert timestamp columns
        if 'report_date' in df.columns:
            df['report_date'] = pd.to_datetime(df['report_date'])
        
        insert_dataframe("credit_bureau_data", df)
        logger.info(f"Inserted {len(df)} credit bureau records")
    
    def _load_model_predictions(self, force_reload: bool = False) -> None:
        """Load model predictions data."""
        csv_file = self.data_dir / "model_predictions.csv"
        
        if not csv_file.exists():
            logger.warning(f"Model predictions CSV not found: {csv_file}")
            return
        
        if force_reload:
            truncate_table("model_predictions")
        
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
                validated_data.append(prediction.dict())
            except Exception as e:
                logger.warning(f"Invalid model prediction data: {row.get('prediction_id', 'unknown')} - {e}")
        
        if validated_data:
            insert_dataframe("model_predictions", pd.DataFrame(validated_data))
            logger.info(f"Inserted {len(validated_data)} valid model predictions")
    
    def _verify_data_integrity(self) -> None:
        """Verify data integrity and relationships."""
        logger.info("Verifying data integrity...")
        
        with get_db_session() as session:
            # Check table row counts
            tables = [
                "customer_profiles",
                "credit_applications", 
                "transaction_data",
                "credit_bureau_data",
                "model_predictions"
            ]
            
            for table in tables:
                count = get_row_count(table)
                logger.info(f"Table {table}: {count} rows")
            
            # Check foreign key relationships
            logger.info("Checking foreign key relationships...")
            
            # Check credit applications reference valid customers
            result = session.execute(text("""
                SELECT COUNT(*) as invalid_apps
                FROM credit_applications ca
                LEFT JOIN customer_profiles cp ON ca.customer_id = cp.customer_id
                WHERE cp.customer_id IS NULL
            """))
            invalid_apps = result.fetchone()[0]
            logger.info(f"Credit applications with invalid customer references: {invalid_apps}")
            
            # Check transactions reference valid customers
            result = session.execute(text("""
                SELECT COUNT(*) as invalid_trans
                FROM transaction_data td
                LEFT JOIN customer_profiles cp ON td.customer_id = cp.customer_id
                WHERE cp.customer_id IS NULL
            """))
            invalid_trans = result.fetchone()[0]
            logger.info(f"Transactions with invalid customer references: {invalid_trans}")
            
            # Check model predictions reference valid customers
            result = session.execute(text("""
                SELECT COUNT(*) as invalid_preds
                FROM model_predictions mp
                LEFT JOIN customer_profiles cp ON mp.customer_id = cp.customer_id
                WHERE cp.customer_id IS NULL
            """))
            invalid_preds = result.fetchone()[0]
            logger.info(f"Model predictions with invalid customer references: {invalid_preds}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load seed data into FinRisk database")
    parser.add_argument("--force-reload", action="store_true", help="Truncate tables before loading data")
    parser.add_argument("--verify-only", action="store_true", help="Only verify data integrity without loading")
    
    args = parser.parse_args()
    
    try:
        loader = DataLoader()
        loader.load_data(force_reload=args.force_reload, verify_only=args.verify_only)
    except Exception as e:
        logger.error(f"Error during data loading: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
