"""
Data loading script for FinRisk application.
Loads seed CSV data into PostgreSQL database with proper schema validation.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.config import get_settings
from app.infra.db import get_db_session, execute_sql_file
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
        self.settings = get_settings()
        self.data_dir = Path(self.settings.data.seed_path)
        self.sql_dir = Path("./sql")
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sql_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data loader initialized with data directory: {self.data_dir}")
        logger.info(f"SQL directory: {self.sql_dir}")
    
    def load_all_data(self, force_reload: bool = False) -> Dict[str, int]:
        """
        Load all seed data into the database.
        
        Args:
            force_reload: If True, truncate tables before loading
            
        Returns:
            Dictionary with table names and row counts
        """
        logger.info("Starting data loading process...")
        
        try:
            # Initialize database schema
            self._init_schema()
            
            # Load data in order of dependencies
            results = {}
            
            # Load customer profiles first (no dependencies)
            results['customer_profiles'] = self._load_customer_profiles(force_reload)
            
            # Load credit applications (depends on customers)
            results['credit_applications'] = self._load_credit_applications(force_reload)
            
            # Load transaction data (depends on customers)
            results['transaction_data'] = self._load_transaction_data(force_reload)
            
            # Load credit bureau data (depends on customers)
            results['credit_bureau_data'] = self._load_credit_bureau_data(force_reload)
            
            # Load model predictions (depends on applications)
            results['model_predictions'] = self._load_model_predictions(force_reload)
            
            logger.info("Data loading completed successfully!")
            logger.info(f"Loaded records: {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during data loading: {e}")
            raise
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        logger.info("Initializing database schema...")
        
        schema_file = self.sql_dir / "001_init_schema.sql"
        if not schema_file.exists():
            logger.warning(f"Schema file not found: {schema_file}")
            logger.info("Creating basic schema...")
            self._create_basic_schema()
        else:
            execute_sql_file(schema_file)
            logger.info("Schema initialized from SQL file")
    
    def _create_basic_schema(self) -> None:
        """Create basic schema if SQL file doesn't exist."""
        with get_db_session() as session:
            # Create customer_profiles table
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS customer_profiles (
                    customer_id VARCHAR(50) PRIMARY KEY,
                    first_name VARCHAR(100) NOT NULL,
                    last_name VARCHAR(100) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    phone VARCHAR(20),
                    date_of_birth DATE NOT NULL,
                    address TEXT,
                    city VARCHAR(100),
                    state VARCHAR(50),
                    zip_code VARCHAR(20),
                    country VARCHAR(100) DEFAULT 'USA',
                    employment_status VARCHAR(50),
                    annual_income DECIMAL(15,2),
                    credit_score INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create credit_applications table
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS credit_applications (
                    application_id VARCHAR(50) PRIMARY KEY,
                    customer_id VARCHAR(50) REFERENCES customer_profiles(customer_id),
                    loan_amount DECIMAL(15,2) NOT NULL,
                    loan_purpose VARCHAR(100) NOT NULL,
                    employment_status VARCHAR(50),
                    annual_income DECIMAL(15,2),
                    credit_score INTEGER,
                    existing_debt DECIMAL(15,2) DEFAULT 0,
                    collateral_value DECIMAL(15,2),
                    application_status VARCHAR(50) DEFAULT 'PENDING',
                    risk_score DECIMAL(5,4),
                    approval_probability DECIMAL(5,4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create transaction_data table
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS transaction_data (
                    transaction_id VARCHAR(50) PRIMARY KEY,
                    customer_id VARCHAR(50) REFERENCES customer_profiles(customer_id),
                    transaction_date TIMESTAMP NOT NULL,
                    amount DECIMAL(15,2) NOT NULL,
                    transaction_type VARCHAR(50),
                    merchant_category VARCHAR(100),
                    merchant_name VARCHAR(255),
                    location VARCHAR(255),
                    is_fraudulent BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create credit_bureau_data table
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS credit_bureau_data (
                    record_id VARCHAR(50) PRIMARY KEY,
                    customer_id VARCHAR(50) REFERENCES customer_profiles(customer_id),
                    credit_score INTEGER,
                    payment_history TEXT,
                    credit_utilization DECIMAL(5,4),
                    length_of_credit_history INTEGER,
                    number_of_accounts INTEGER,
                    derogatory_marks INTEGER DEFAULT 0,
                    inquiries_last_6_months INTEGER DEFAULT 0,
                    public_records INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create model_predictions table
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS model_predictions (
                    prediction_id VARCHAR(50) PRIMARY KEY,
                    customer_id VARCHAR(50) REFERENCES customer_profiles(customer_id),
                    application_id VARCHAR(50) REFERENCES credit_applications(application_id),
                    model_type VARCHAR(50) NOT NULL,
                    prediction_type VARCHAR(50) NOT NULL,
                    prediction_value DECIMAL(10,6) NOT NULL,
                    confidence_score DECIMAL(5,4),
                    features_used JSONB,
                    model_version VARCHAR(50),
                    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            session.commit()
            logger.info("Basic schema created successfully")
    
    def _load_customer_profiles(self, force_reload: bool = False) -> int:
        """Load customer profiles data."""
        csv_file = self.data_dir / "customer_profiles.csv"
        if not csv_file.exists():
            logger.warning(f"Customer profiles CSV not found: {csv_file}")
            return 0
        
        logger.info(f"Loading customer profiles from {csv_file}")
        
        # Read CSV data
        df = pd.read_csv(csv_file)
        logger.info(f"Read {len(df)} customer profiles from CSV")
        
        # Validate data
        df = self._validate_customer_data(df)
        
        # Load into database
        with get_db_session() as session:
            if force_reload:
                session.execute(text("TRUNCATE TABLE customer_profiles CASCADE"))
                logger.info("Truncated customer_profiles table")
            
            # Insert data
            for _, row in df.iterrows():
                customer = CustomerProfile(
                    customer_id=str(row['customer_id']),
                    first_name=str(row['first_name']),
                    last_name=str(row['last_name']),
                    email=str(row['email']),
                    phone=str(row.get('phone', '')),
                    date_of_birth=pd.to_datetime(row['date_of_birth']).date(),
                    address=str(row.get('address', '')),
                    city=str(row.get('city', '')),
                    state=str(row.get('state', '')),
                    zip_code=str(row.get('zip_code', '')),
                    country=str(row.get('country', 'USA')),
                    employment_status=str(row.get('employment_status', '')),
                    annual_income=float(row.get('annual_income', 0)),
                    credit_score=int(row.get('credit_score', 0)) if pd.notna(row.get('credit_score')) else None
                )
                
                session.merge(customer)
            
            session.commit()
            logger.info(f"Loaded {len(df)} customer profiles into database")
            return len(df)
    
    def _load_credit_applications(self, force_reload: bool = False) -> int:
        """Load credit applications data."""
        csv_file = self.data_dir / "credit_applications.csv"
        if not csv_file.exists():
            logger.warning(f"Credit applications CSV not found: {csv_file}")
            return 0
        
        logger.info(f"Loading credit applications from {csv_file}")
        
        # Read CSV data
        df = pd.read_csv(csv_file)
        logger.info(f"Read {len(df)} credit applications from CSV")
        
        # Validate data
        df = self._validate_application_data(df)
        
        # Load into database
        with get_db_session() as session:
            if force_reload:
                session.execute(text("TRUNCATE TABLE credit_applications CASCADE"))
                logger.info("Truncated credit_applications table")
            
            # Insert data
            for _, row in df.iterrows():
                application = CreditApplication(
                    application_id=str(row['application_id']),
                    customer_id=str(row['customer_id']),
                    loan_amount=float(row['loan_amount']),
                    loan_purpose=str(row['loan_purpose']),
                    employment_status=str(row.get('employment_status', '')),
                    annual_income=float(row.get('annual_income', 0)),
                    credit_score=int(row.get('credit_score', 0)) if pd.notna(row.get('credit_score')) else None,
                    existing_debt=float(row.get('existing_debt', 0)),
                    collateral_value=float(row.get('collateral_value', 0)) if pd.notna(row.get('collateral_value')) else None,
                    application_status=str(row.get('application_status', 'PENDING')),
                    risk_score=float(row.get('risk_score', 0)) if pd.notna(row.get('risk_score')) else None,
                    approval_probability=float(row.get('approval_probability', 0)) if pd.notna(row.get('approval_probability')) else None
                )
                
                session.merge(application)
            
            session.commit()
            logger.info(f"Loaded {len(df)} credit applications into database")
            return len(df)
    
    def _load_transaction_data(self, force_reload: bool = False) -> int:
        """Load transaction data."""
        csv_file = self.data_dir / "transaction_data.csv"
        if not csv_file.exists():
            logger.warning(f"Transaction data CSV not found: {csv_file}")
            return 0
        
        logger.info(f"Loading transaction data from {csv_file}")
        
        # Read CSV data
        df = pd.read_csv(csv_file)
        logger.info(f"Read {len(df)} transactions from CSV")
        
        # Validate data
        df = self._validate_transaction_data(df)
        
        # Load into database
        with get_db_session() as session:
            if force_reload:
                session.execute(text("TRUNCATE TABLE transaction_data CASCADE"))
                logger.info("Truncated transaction_data table")
            
            # Insert data in batches
            batch_size = 1000
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                for _, row in batch.iterrows():
                    session.execute(text("""
                        INSERT INTO transaction_data (
                            transaction_id, customer_id, transaction_date, amount,
                            transaction_type, merchant_category, merchant_name,
                            location, is_fraudulent
                        ) VALUES (
                            :transaction_id, :customer_id, :transaction_date, :amount,
                            :transaction_type, :merchant_category, :merchant_name,
                            :location, :is_fraudulent
                        )
                    """), {
                        'transaction_id': str(row['transaction_id']),
                        'customer_id': str(row['customer_id']),
                        'transaction_date': pd.to_datetime(row['transaction_date']),
                        'amount': float(row['amount']),
                        'transaction_type': str(row.get('transaction_type', '')),
                        'merchant_category': str(row.get('merchant_category', '')),
                        'merchant_name': str(row.get('merchant_name', '')),
                        'location': str(row.get('location', '')),
                        'is_fraudulent': bool(row.get('is_fraudulent', False))
                    })
                
                session.commit()
                logger.info(f"Loaded batch {i//batch_size + 1} ({len(batch)} transactions)")
            
            logger.info(f"Loaded {len(df)} transactions into database")
            return len(df)
    
    def _load_credit_bureau_data(self, force_reload: bool = False) -> int:
        """Load credit bureau data."""
        csv_file = self.data_dir / "credit_bureau_data.csv"
        if not csv_file.exists():
            logger.warning(f"Credit bureau data CSV not found: {csv_file}")
            return 0
        
        logger.info(f"Loading credit bureau data from {csv_file}")
        
        # Read CSV data
        df = pd.read_csv(csv_file)
        logger.info(f"Read {len(df)} credit bureau records from CSV")
        
        # Validate data
        df = self._validate_credit_bureau_data(df)
        
        # Load into database
        with get_db_session() as session:
            if force_reload:
                session.execute(text("TRUNCATE TABLE credit_bureau_data CASCADE"))
                logger.info("Truncated credit_bureau_data table")
            
            # Insert data
            for _, row in df.iterrows():
                session.execute(text("""
                    INSERT INTO credit_bureau_data (
                        record_id, customer_id, credit_score, payment_history,
                        credit_utilization, length_of_credit_history,
                        number_of_accounts, derogatory_marks,
                        inquiries_last_6_months, public_records
                    ) VALUES (
                        :record_id, :customer_id, :credit_score, :payment_history,
                        :credit_utilization, :length_of_credit_history,
                        :number_of_accounts, :derogatory_marks,
                        :inquiries_last_6_months, :public_records
                    )
                """), {
                    'record_id': str(row['record_id']),
                    'customer_id': str(row['customer_id']),
                    'credit_score': int(row.get('credit_score', 0)) if pd.notna(row.get('credit_score')) else None,
                    'payment_history': str(row.get('payment_history', '')),
                    'credit_utilization': float(row.get('credit_utilization', 0)) if pd.notna(row.get('credit_utilization')) else None,
                    'length_of_credit_history': int(row.get('length_of_credit_history', 0)) if pd.notna(row.get('length_of_credit_history')) else None,
                    'number_of_accounts': int(row.get('number_of_accounts', 0)) if pd.notna(row.get('number_of_accounts')) else None,
                    'derogatory_marks': int(row.get('derogatory_marks', 0)),
                    'inquiries_last_6_months': int(row.get('inquiries_last_6_months', 0)),
                    'public_records': int(row.get('public_records', 0))
                })
            
            session.commit()
            logger.info(f"Loaded {len(df)} credit bureau records into database")
            return len(df)
    
    def _load_model_predictions(self, force_reload: bool = False) -> int:
        """Load model predictions data."""
        csv_file = self.data_dir / "model_predictions.csv"
        if not csv_file.exists():
            logger.warning(f"Model predictions CSV not found: {csv_file}")
            return 0
        
        logger.info(f"Loading model predictions from {csv_file}")
        
        # Read CSV data
        df = pd.read_csv(csv_file)
        logger.info(f"Read {len(df)} model predictions from CSV")
        
        # Validate data
        df = self._validate_prediction_data(df)
        
        # Load into database
        with get_db_session() as session:
            if force_reload:
                session.execute(text("TRUNCATE TABLE model_predictions CASCADE"))
                logger.info("Truncated model_predictions table")
            
            # Insert data
            for _, row in df.iterrows():
                prediction = ModelPrediction(
                    prediction_id=str(row['prediction_id']),
                    customer_id=str(row['customer_id']),
                    application_id=str(row.get('application_id', '')),
                    model_type=str(row['model_type']),
                    prediction_type=str(row['prediction_type']),
                    prediction_value=float(row['prediction_value']),
                    confidence_score=float(row.get('confidence_score', 0)) if pd.notna(row.get('confidence_score')) else None,
                    features_used=row.get('features_used', {}),
                    model_version=str(row.get('model_version', '1.0')),
                    prediction_timestamp=pd.to_datetime(row.get('prediction_timestamp', pd.Timestamp.now()))
                )
                
                session.merge(prediction)
            
            session.commit()
            logger.info(f"Loaded {len(df)} model predictions into database")
            return len(df)
    
    def _validate_customer_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate customer profile data."""
        logger.info("Validating customer data...")
        
        # Check required columns
        required_cols = ['customer_id', 'first_name', 'last_name', 'email', 'date_of_birth']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['customer_id'])
        if len(df) < initial_count:
            logger.warning(f"Removed {initial_count - len(df)} duplicate customer IDs")
        
        # Validate email format
        invalid_emails = df[~df['email'].str.contains(r'^[^@]+@[^@]+\.[^@]+', na=False)]
        if len(invalid_emails) > 0:
            logger.warning(f"Found {len(invalid_emails)} invalid email addresses")
        
        # Validate credit score range
        if 'credit_score' in df.columns:
            invalid_scores = df[(df['credit_score'] < 300) | (df['credit_score'] > 850)]
            if len(invalid_scores) > 0:
                logger.warning(f"Found {len(invalid_scores)} credit scores outside valid range (300-850)")
        
        logger.info(f"Customer data validation completed. Valid records: {len(df)}")
        return df
    
    def _validate_application_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate credit application data."""
        logger.info("Validating credit application data...")
        
        # Check required columns
        required_cols = ['application_id', 'customer_id', 'loan_amount', 'loan_purpose']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['application_id'])
        if len(df) < initial_count:
            logger.warning(f"Removed {initial_count - len(df)} duplicate application IDs")
        
        # Validate loan amount
        invalid_amounts = df[df['loan_amount'] <= 0]
        if len(invalid_amounts) > 0:
            logger.warning(f"Found {len(invalid_amounts)} applications with invalid loan amounts")
        
        logger.info(f"Credit application data validation completed. Valid records: {len(df)}")
        return df
    
    def _validate_transaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate transaction data."""
        logger.info("Validating transaction data...")
        
        # Check required columns
        required_cols = ['transaction_id', 'customer_id', 'transaction_date', 'amount']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['transaction_id'])
        if len(df) < initial_count:
            logger.warning(f"Removed {initial_count - len(df)} duplicate transaction IDs")
        
        # Validate amount
        invalid_amounts = df[df['amount'] == 0]
        if len(invalid_amounts) > 0:
            logger.warning(f"Found {len(invalid_amounts)} transactions with zero amount")
        
        logger.info(f"Transaction data validation completed. Valid records: {len(df)}")
        return df
    
    def _validate_credit_bureau_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate credit bureau data."""
        logger.info("Validating credit bureau data...")
        
        # Check required columns
        required_cols = ['record_id', 'customer_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['record_id'])
        if len(df) < initial_count:
            logger.warning(f"Removed {initial_count - len(df)} duplicate record IDs")
        
        logger.info(f"Credit bureau data validation completed. Valid records: {len(df)}")
        return df
    
    def _validate_prediction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate model prediction data."""
        logger.info("Validating model prediction data...")
        
        # Check required columns
        required_cols = ['prediction_id', 'customer_id', 'model_type', 'prediction_type', 'prediction_value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['prediction_id'])
        if len(df) < initial_count:
            logger.warning(f"Removed {initial_count - len(df)} duplicate prediction IDs")
        
        logger.info(f"Model prediction data validation completed. Valid records: {len(df)}")
        return df
    
    def verify_data_integrity(self) -> Dict[str, Dict[str, int]]:
        """
        Verify data integrity after loading.
        
        Returns:
            Dictionary with table names and their row counts
        """
        logger.info("Verifying data integrity...")
        
        with get_db_session() as session:
            tables = [
                'customer_profiles',
                'credit_applications', 
                'transaction_data',
                'credit_bureau_data',
                'model_predictions'
            ]
            
            results = {}
            for table in tables:
                result = session.execute(text(f"SELECT COUNT(*) as count FROM {table}"))
                count = result.fetchone()[0]
                results[table] = {'count': count}
                logger.info(f"Table {table}: {count} records")
            
            # Check foreign key relationships
            fk_checks = [
                ("credit_applications", "customer_id", "customer_profiles"),
                ("transaction_data", "customer_id", "customer_profiles"),
                ("credit_bureau_data", "customer_id", "customer_profiles"),
                ("model_predictions", "customer_id", "customer_profiles"),
                ("model_predictions", "application_id", "credit_applications")
            ]
            
            for table, fk_col, ref_table in fk_checks:
                result = session.execute(text(f"""
                    SELECT COUNT(*) as count 
                    FROM {table} t 
                    LEFT JOIN {ref_table} r ON t.{fk_col} = r.{fk_col.split('_')[0]}_id
                    WHERE r.{fk_col.split('_')[0]}_id IS NULL
                """))
                orphan_count = result.fetchone()[0]
                results[table]['orphan_records'] = orphan_count
                
                if orphan_count > 0:
                    logger.warning(f"Found {orphan_count} orphan records in {table}.{fk_col}")
                else:
                    logger.info(f"All {table}.{fk_col} references are valid")
        
        return results


def main():
    """Main function to run the data loader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load seed data into FinRisk database")
    parser.add_argument(
        "--force-reload", 
        action="store_true", 
        help="Truncate tables before loading data"
    )
    parser.add_argument(
        "--verify-only", 
        action="store_true", 
        help="Only verify data integrity without loading"
    )
    
    args = parser.parse_args()
    
    try:
        loader = DataLoader()
        
        if args.verify_only:
            results = loader.verify_data_integrity()
            print("\nData Integrity Report:")
            for table, stats in results.items():
                print(f"  {table}: {stats['count']} records")
                if 'orphan_records' in stats:
                    print(f"    Orphan records: {stats['orphan_records']}")
        else:
            results = loader.load_all_data(force_reload=args.force_reload)
            print(f"\nData loading completed successfully!")
            print(f"Loaded records: {results}")
            
            # Verify integrity after loading
            print("\nVerifying data integrity...")
            integrity_results = loader.verify_data_integrity()
            print("Data integrity verification completed.")
            
    except Exception as e:
        logger.error(f"Error in data loading: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
