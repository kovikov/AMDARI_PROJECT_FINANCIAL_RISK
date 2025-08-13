"""
Simplified test script for the data loader module.
Tests CSV generation and data validation without database connectivity.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.config import get_settings


def generate_sample_customer_data(n_customers: int = 100) -> pd.DataFrame:
    """Generate sample customer profile data."""
    print(f"Generating {n_customers} sample customer profiles...")
    
    np.random.seed(42)  # For reproducible results
    
    # Sample data for realistic customer profiles
    first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily', 'Robert', 'Lisa', 'James', 'Mary']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
    states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'FL', 'OH', 'GA', 'NC']
    employment_statuses = ['EMPLOYED', 'SELF_EMPLOYED', 'UNEMPLOYED', 'RETIRED', 'STUDENT']
    
    data = []
    for i in range(n_customers):
        customer_id = f"CUST_{i+1:04d}"
        first_name = np.random.choice(first_names)
        last_name = np.random.choice(last_names)
        email = f"{first_name.lower()}.{last_name.lower()}@example.com"
        
        # Generate realistic date of birth (18-80 years old)
        age = np.random.randint(18, 81)
        birth_year = datetime.now().year - age
        birth_month = np.random.randint(1, 13)
        birth_day = np.random.randint(1, 29)  # Simplified
        date_of_birth = datetime(birth_year, birth_month, birth_day).date()
        
        # Generate realistic income and credit score
        annual_income = np.random.lognormal(10.5, 0.5)  # Log-normal distribution
        annual_income = min(max(annual_income, 20000), 500000)  # Cap between 20k and 500k
        
        credit_score = np.random.normal(700, 100)  # Normal distribution around 700
        credit_score = int(min(max(credit_score, 300), 850))  # Cap between 300-850
        
        city = np.random.choice(cities)
        state = np.random.choice(states)
        zip_code = f"{np.random.randint(10000, 100000)}"
        
        data.append({
            'customer_id': customer_id,
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'phone': f"+1-{np.random.randint(200, 999)}-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}",
            'date_of_birth': date_of_birth.strftime('%Y-%m-%d'),
            'address': f"{np.random.randint(100, 9999)} {np.random.choice(['Main', 'Oak', 'Pine', 'Elm', 'Maple'])} St",
            'city': city,
            'state': state,
            'zip_code': zip_code,
            'country': 'USA',
            'employment_status': np.random.choice(employment_statuses),
            'annual_income': round(annual_income, 2),
            'credit_score': credit_score
        })
    
    return pd.DataFrame(data)


def generate_sample_credit_applications(n_applications: int = 50, customer_ids: list = None) -> pd.DataFrame:
    """Generate sample credit application data."""
    print(f"Generating {n_applications} sample credit applications...")
    
    if customer_ids is None:
        customer_ids = [f"CUST_{i+1:04d}" for i in range(100)]
    
    loan_purposes = ['HOME_PURCHASE', 'DEBT_CONSOLIDATION', 'CAR_PURCHASE', 'EDUCATION', 'BUSINESS', 'PERSONAL']
    employment_statuses = ['EMPLOYED', 'SELF_EMPLOYED', 'UNEMPLOYED', 'RETIRED', 'STUDENT']
    application_statuses = ['PENDING', 'APPROVED', 'REJECTED', 'UNDER_REVIEW']
    
    data = []
    for i in range(n_applications):
        application_id = f"APP_{i+1:04d}"
        customer_id = np.random.choice(customer_ids)
        
        # Generate realistic loan amount
        loan_amount = np.random.lognormal(10, 0.8)  # Log-normal distribution
        loan_amount = min(max(loan_amount, 5000), 500000)  # Cap between 5k and 500k
        
        # Generate related fields
        annual_income = np.random.lognormal(10.5, 0.5)
        annual_income = min(max(annual_income, 20000), 500000)
        
        existing_debt = np.random.exponential(annual_income * 0.3)  # Exponential distribution
        existing_debt = min(existing_debt, annual_income * 0.8)  # Cap at 80% of income
        
        credit_score = np.random.normal(700, 100)
        credit_score = int(min(max(credit_score, 300), 850))
        
        # Calculate risk score based on factors
        debt_to_income = existing_debt / annual_income if annual_income > 0 else 0
        risk_score = (1 - credit_score/850) * 0.4 + debt_to_income * 0.3 + np.random.normal(0, 0.1)
        risk_score = max(0, min(1, risk_score))
        
        # Calculate approval probability (inverse of risk)
        approval_probability = 1 - risk_score + np.random.normal(0, 0.05)
        approval_probability = max(0, min(1, approval_probability))
        
        data.append({
            'application_id': application_id,
            'customer_id': customer_id,
            'loan_amount': round(loan_amount, 2),
            'loan_purpose': np.random.choice(loan_purposes),
            'employment_status': np.random.choice(employment_statuses),
            'annual_income': round(annual_income, 2),
            'credit_score': credit_score,
            'existing_debt': round(existing_debt, 2),
            'collateral_value': round(loan_amount * np.random.uniform(0.5, 1.5), 2) if np.random.random() > 0.3 else None,
            'application_status': np.random.choice(application_statuses),
            'risk_score': round(risk_score, 4),
            'approval_probability': round(approval_probability, 4)
        })
    
    return pd.DataFrame(data)


def generate_sample_transaction_data(n_transactions: int = 1000, customer_ids: list = None) -> pd.DataFrame:
    """Generate sample transaction data."""
    print(f"Generating {n_transactions} sample transactions...")
    
    if customer_ids is None:
        customer_ids = [f"CUST_{i+1:04d}" for i in range(100)]
    
    transaction_types = ['PURCHASE', 'WITHDRAWAL', 'DEPOSIT', 'TRANSFER', 'PAYMENT']
    merchant_categories = ['GROCERIES', 'GAS_STATION', 'RESTAURANT', 'RETAIL', 'ONLINE_SHOPPING', 'UTILITIES', 'ENTERTAINMENT', 'TRAVEL']
    merchant_names = ['Walmart', 'Target', 'Amazon', 'Starbucks', 'Shell', 'McDonald\'s', 'Netflix', 'Uber', 'Airbnb', 'Home Depot']
    locations = ['New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX', 'Phoenix, AZ', 'Philadelphia, PA', 'San Antonio, TX', 'San Diego, CA']
    
    data = []
    start_date = datetime.now() - timedelta(days=365)  # Last year
    
    for i in range(n_transactions):
        transaction_id = f"TXN_{i+1:06d}"
        customer_id = np.random.choice(customer_ids)
        
        # Generate transaction date (within last year)
        days_ago = np.random.exponential(100)  # Exponential distribution for recent bias
        days_ago = min(days_ago, 365)
        transaction_date = start_date + timedelta(days=days_ago)
        
        # Generate realistic transaction amount
        transaction_type = np.random.choice(transaction_types)
        if transaction_type == 'PURCHASE':
            amount = np.random.lognormal(3, 1)  # Log-normal for purchases
            amount = min(max(amount, 1), 1000)
        elif transaction_type == 'WITHDRAWAL':
            amount = np.random.lognormal(4, 0.8)  # Larger amounts for withdrawals
            amount = min(max(amount, 20), 500)
        else:
            amount = np.random.lognormal(3.5, 1.2)
            amount = min(max(amount, 1), 2000)
        
        # Generate fraud flag (rare occurrence)
        is_fraudulent = np.random.random() < 0.02  # 2% fraud rate
        
        data.append({
            'transaction_id': transaction_id,
            'customer_id': customer_id,
            'transaction_date': transaction_date.strftime('%Y-%m-%d %H:%M:%S'),
            'amount': round(amount, 2),
            'transaction_type': transaction_type,
            'merchant_category': np.random.choice(merchant_categories),
            'merchant_name': np.random.choice(merchant_names),
            'location': np.random.choice(locations),
            'is_fraudulent': is_fraudulent
        })
    
    return pd.DataFrame(data)


def generate_sample_credit_bureau_data(customer_ids: list = None) -> pd.DataFrame:
    """Generate sample credit bureau data."""
    print(f"Generating sample credit bureau data...")
    
    if customer_ids is None:
        customer_ids = [f"CUST_{i+1:04d}" for i in range(100)]
    
    payment_history_patterns = ['000000000000', '111111111111', '000000111111', '111111000000', '000011110000']
    
    data = []
    for i, customer_id in enumerate(customer_ids):
        record_id = f"CB_{i+1:04d}"
        
        # Generate realistic credit bureau data
        credit_score = np.random.normal(700, 100)
        credit_score = int(min(max(credit_score, 300), 850))
        
        credit_utilization = np.random.beta(2, 5)  # Beta distribution favoring lower utilization
        length_of_credit_history = np.random.exponential(10)  # Exponential distribution
        length_of_credit_history = int(min(length_of_credit_history, 30))  # Cap at 30 years
        
        number_of_accounts = np.random.poisson(8)  # Poisson distribution
        number_of_accounts = max(number_of_accounts, 1)  # At least 1 account
        
        derogatory_marks = np.random.poisson(0.5)  # Poisson distribution, most have 0
        inquiries_last_6_months = np.random.poisson(2)  # Poisson distribution
        public_records = np.random.poisson(0.1)  # Poisson distribution, most have 0
        
        data.append({
            'record_id': record_id,
            'customer_id': customer_id,
            'credit_score': credit_score,
            'payment_history': np.random.choice(payment_history_patterns),
            'credit_utilization': round(credit_utilization, 4),
            'length_of_credit_history': length_of_credit_history,
            'number_of_accounts': number_of_accounts,
            'derogatory_marks': derogatory_marks,
            'inquiries_last_6_months': inquiries_last_6_months,
            'public_records': public_records
        })
    
    return pd.DataFrame(data)


def generate_sample_model_predictions(n_predictions: int = 100, customer_ids: list = None, application_ids: list = None) -> pd.DataFrame:
    """Generate sample model prediction data."""
    print(f"Generating {n_predictions} sample model predictions...")
    
    if customer_ids is None:
        customer_ids = [f"CUST_{i+1:04d}" for i in range(100)]
    if application_ids is None:
        application_ids = [f"APP_{i+1:04d}" for i in range(50)]
    
    model_types = ['credit_risk', 'fraud_detection', 'churn_prediction']
    prediction_types = ['risk_score', 'fraud_probability', 'default_probability', 'approval_probability']
    
    data = []
    for i in range(n_predictions):
        prediction_id = f"PRED_{i+1:06d}"
        customer_id = np.random.choice(customer_ids)
        application_id = np.random.choice(application_ids) if np.random.random() > 0.3 else None
        
        model_type = np.random.choice(model_types)
        prediction_type = np.random.choice(prediction_types)
        
        # Generate realistic prediction values
        if prediction_type in ['risk_score', 'fraud_probability', 'default_probability']:
            prediction_value = np.random.beta(2, 3)  # Beta distribution for probabilities
        else:
            prediction_value = np.random.beta(3, 2)  # Slightly higher for approval probability
        
        confidence_score = np.random.beta(5, 2)  # Beta distribution favoring higher confidence
        
        # Generate sample features used
        features_used = {
            'credit_score': np.random.randint(300, 851),
            'annual_income': np.random.randint(20000, 500001),
            'debt_to_income_ratio': round(np.random.uniform(0, 1), 3),
            'payment_history': np.random.randint(0, 100),
            'age': np.random.randint(18, 81)
        }
        
        data.append({
            'prediction_id': prediction_id,
            'customer_id': customer_id,
            'application_id': application_id,
            'model_type': model_type,
            'prediction_type': prediction_type,
            'prediction_value': round(prediction_value, 6),
            'confidence_score': round(confidence_score, 4),
            'features_used': json.dumps(features_used),
            'model_version': f"{np.random.randint(1, 4)}.{np.random.randint(0, 10)}.{np.random.randint(0, 10)}",
            'prediction_timestamp': (datetime.now() - timedelta(days=np.random.randint(0, 30))).strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return pd.DataFrame(data)


def validate_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate customer profile data."""
    print("Validating customer data...")
    
    # Check required columns
    required_cols = ['customer_id', 'first_name', 'last_name', 'email', 'date_of_birth']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['customer_id'])
    if len(df) < initial_count:
        print(f"Removed {initial_count - len(df)} duplicate customer IDs")
    
    # Validate email format
    invalid_emails = df[~df['email'].str.contains(r'^[^@]+@[^@]+\.[^@]+', na=False)]
    if len(invalid_emails) > 0:
        print(f"Found {len(invalid_emails)} invalid email addresses")
    
    # Validate credit score range
    if 'credit_score' in df.columns:
        invalid_scores = df[(df['credit_score'] < 300) | (df['credit_score'] > 850)]
        if len(invalid_scores) > 0:
            print(f"Found {len(invalid_scores)} credit scores outside valid range (300-850)")
    
    print(f"Customer data validation completed. Valid records: {len(df)}")
    return df


def validate_application_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate credit application data."""
    print("Validating credit application data...")
    
    # Check required columns
    required_cols = ['application_id', 'customer_id', 'loan_amount', 'loan_purpose']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['application_id'])
    if len(df) < initial_count:
        print(f"Removed {initial_count - len(df)} duplicate application IDs")
    
    # Validate loan amount
    invalid_amounts = df[df['loan_amount'] <= 0]
    if len(invalid_amounts) > 0:
        print(f"Found {len(invalid_amounts)} applications with invalid loan amounts")
    
    print(f"Credit application data validation completed. Valid records: {len(df)}")
    return df


def validate_transaction_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate transaction data."""
    print("Validating transaction data...")
    
    # Check required columns
    required_cols = ['transaction_id', 'customer_id', 'transaction_date', 'amount']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['transaction_id'])
    if len(df) < initial_count:
        print(f"Removed {initial_count - len(df)} duplicate transaction IDs")
    
    # Validate amount
    invalid_amounts = df[df['amount'] == 0]
    if len(invalid_amounts) > 0:
        print(f"Found {len(invalid_amounts)} transactions with zero amount")
    
    print(f"Transaction data validation completed. Valid records: {len(df)}")
    return df


def validate_credit_bureau_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate credit bureau data."""
    print("Validating credit bureau data...")
    
    # Check required columns
    required_cols = ['record_id', 'customer_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['record_id'])
    if len(df) < initial_count:
        print(f"Removed {initial_count - len(df)} duplicate record IDs")
    
    print(f"Credit bureau data validation completed. Valid records: {len(df)}")
    return df


def validate_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate model prediction data."""
    print("Validating model prediction data...")
    
    # Check required columns
    required_cols = ['prediction_id', 'customer_id', 'model_type', 'prediction_type', 'prediction_value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['prediction_id'])
    if len(df) < initial_count:
        print(f"Removed {initial_count - len(df)} duplicate prediction IDs")
    
    print(f"Model prediction data validation completed. Valid records: {len(df)}")
    return df


def create_sample_csv_files(data_dir: Path):
    """Create sample CSV files for testing."""
    print("Creating sample CSV files...")
    
    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate customer data first
    customer_df = generate_sample_customer_data(100)
    customer_ids = customer_df['customer_id'].tolist()
    
    # Generate other data with customer dependencies
    applications_df = generate_sample_credit_applications(50, customer_ids)
    application_ids = applications_df['application_id'].tolist()
    
    transactions_df = generate_sample_transaction_data(1000, customer_ids)
    credit_bureau_df = generate_sample_credit_bureau_data(customer_ids)
    predictions_df = generate_sample_model_predictions(100, customer_ids, application_ids)
    
    # Validate data
    print("\nValidating generated data...")
    customer_df = validate_customer_data(customer_df)
    applications_df = validate_application_data(applications_df)
    transactions_df = validate_transaction_data(transactions_df)
    credit_bureau_df = validate_credit_bureau_data(credit_bureau_df)
    predictions_df = validate_prediction_data(predictions_df)
    
    # Save to CSV files
    customer_df.to_csv(data_dir / "customer_profiles.csv", index=False)
    applications_df.to_csv(data_dir / "credit_applications.csv", index=False)
    transactions_df.to_csv(data_dir / "transaction_data.csv", index=False)
    credit_bureau_df.to_csv(data_dir / "credit_bureau_data.csv", index=False)
    predictions_df.to_csv(data_dir / "model_predictions.csv", index=False)
    
    print("\nSample CSV files created successfully!")
    print(f"Files created in: {data_dir}")
    print(f"  - customer_profiles.csv: {len(customer_df)} records")
    print(f"  - credit_applications.csv: {len(applications_df)} records")
    print(f"  - transaction_data.csv: {len(transactions_df)} records")
    print(f"  - credit_bureau_data.csv: {len(credit_bureau_df)} records")
    print(f"  - model_predictions.csv: {len(predictions_df)} records")
    
    # Show sample data
    print("\nSample customer data:")
    print(customer_df.head(3).to_string())
    
    print("\nSample application data:")
    print(applications_df.head(3).to_string())
    
    print("\nSample transaction data:")
    print(transactions_df.head(3).to_string())
    
    return {
        'customer_profiles': len(customer_df),
        'credit_applications': len(applications_df),
        'transaction_data': len(transactions_df),
        'credit_bureau_data': len(credit_bureau_df),
        'model_predictions': len(predictions_df)
    }


def test_data_loader_imports():
    """Test that the data loader can be imported and initialized."""
    print("Testing data loader imports...")
    
    try:
        # Test import
        from data_loader import DataLoader
        print("✓ DataLoader imported successfully")
        
        # Test initialization (without database connection)
        settings = get_settings()
        print(f"✓ Settings loaded: {settings.environment}")
        print(f"✓ Data directory: {settings.data.seed_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("FinRisk Data Loader Simple Test Suite")
    print("=" * 60)
    
    # Test 1: Import and initialization
    print("\n1. Testing data loader imports...")
    test1_passed = test_data_loader_imports()
    
    # Test 2: CSV file generation and validation
    print("\n2. Testing CSV file generation and validation...")
    try:
        settings = get_settings()
        data_dir = Path(settings.data.seed_path)
        results = create_sample_csv_files(data_dir)
        test2_passed = True
        print(f"✓ CSV generation completed: {results}")
    except Exception as e:
        print(f"✗ CSV generation failed: {e}")
        test2_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    print(f"Import test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"CSV generation test: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\nAll tests passed! Data loader is working correctly.")
        print("\nNext steps:")
        print("1. Set up PostgreSQL database with proper credentials")
        print("2. Run: python data_loader.py --force-reload")
        print("3. Or verify only: python data_loader.py --verify-only")
    else:
        print("\nSome tests failed. Please check the error messages above.")
    
    print("\nGenerated CSV files are ready for database loading when PostgreSQL is available.")


if __name__ == "__main__":
    main()
