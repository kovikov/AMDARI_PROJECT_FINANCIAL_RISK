"""
FinRisk Synthetic Data Generator
Generates realistic financial datasets for credit risk and fraud detection
Mimics real UK banking patterns with proper correlations and business logic
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker('en_GB')  # UK locale
Faker.seed(42)

class FinRiskDataGenerator:
    def __init__(self, n_customers=25000):
        self.n_customers = n_customers
        self.start_date = datetime(2022, 1, 1)
        self.end_date = datetime(2024, 12, 31)
        
        # UK-specific data
        self.uk_cities = ['London', 'Manchester', 'Birmingham', 'Leeds', 'Glasgow', 
                         'Sheffield', 'Liverpool', 'Newcastle', 'Bristol', 'Cardiff']
        
        self.employment_types = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed', 
                               'Retired', 'Student', 'Contract']
        
        self.loan_purposes = ['Home Purchase', 'Debt Consolidation', 'Home Improvement', 
                            'Car Purchase', 'Business', 'Education', 'Personal']
        
        self.merchant_categories = ['Groceries', 'Fuel', 'Restaurants', 'Online Shopping', 
                                  'Entertainment', 'Healthcare', 'Travel', 'Utilities', 
                                  'ATM', 'Transfer']
        
        print(f"Initializing FinRisk Data Generator for {n_customers:,} customers")
        
    def generate_customer_profiles(self):
        """Generate realistic customer profiles with demographics"""
        print("Generating customer profiles...")
        
        customers = []
        for i in range(self.n_customers):
            # Demographics with realistic distributions
            age = max(18, int(np.random.gamma(2, 20)))  # Skewed towards younger adults
            
            # Income correlated with age (career progression)
            base_income = 25000 + (age - 18) * 1200  # Career progression
            income_noise = np.random.normal(0, 15000)
            annual_income = max(16000, base_income + income_noise)  # UK minimum wage floor
            
            # Employment status based on age
            if age < 22:
                employment_status = np.random.choice(['Student', 'Part-time', 'Full-time'], 
                                                   p=[0.4, 0.35, 0.25])
            elif age > 65:
                employment_status = np.random.choice(['Retired', 'Part-time'], p=[0.8, 0.2])
            else:
                employment_status = np.random.choice(
                    ['Full-time', 'Part-time', 'Self-employed', 'Unemployed', 'Contract'], 
                    p=[0.65, 0.15, 0.1, 0.05, 0.05]
                )
            
            # Adjust income based on employment
            if employment_status == 'Part-time':
                annual_income *= 0.6
            elif employment_status == 'Unemployed':
                annual_income = np.random.uniform(0, 12000)  # Benefits
            elif employment_status == 'Student':
                annual_income = np.random.uniform(0, 15000)
            elif employment_status == 'Self-employed':
                annual_income *= np.random.uniform(0.7, 1.8)  # Variable income
            
            # Account tenure (relationship with bank)
            account_tenure = max(0, int(np.random.gamma(2, 2)))  # Most customers are newer
            
            # Credit score based on age, income, and tenure
            base_score = 650
            age_bonus = min(50, (age - 18) * 1.5)  # Older = better credit
            income_bonus = min(100, (annual_income - 20000) / 1000)  # Higher income = better score
            tenure_bonus = min(80, account_tenure * 15)  # Longer relationship = better score
            
            credit_score = int(base_score + age_bonus + income_bonus + tenure_bonus + 
                             np.random.normal(0, 80))
            credit_score = np.clip(credit_score, 300, 850)  # UK credit score range
            
            # Risk segment based on credit score
            if credit_score >= 750:
                risk_segment = 'Prime'
            elif credit_score >= 650:
                risk_segment = 'Near-Prime'
            elif credit_score >= 550:
                risk_segment = 'Subprime'
            else:
                risk_segment = 'Deep-Subprime'
            
            # Behavioral score (separate from credit score)
            behavioral_score = np.random.beta(2, 5) * 1000  # Skewed towards lower scores
            
            # Product holdings based on risk segment
            if risk_segment == 'Prime':
                product_holdings = np.random.randint(2, 6)
            elif risk_segment == 'Near-Prime':
                product_holdings = np.random.randint(1, 4)
            else:
                product_holdings = np.random.randint(1, 3)
            
            # Relationship value
            relationship_value = annual_income * 0.1 * product_holdings * (1 + account_tenure/10)
            
            customers.append({
                'customer_id': f'CUST_{i+1:06d}',
                'customer_age': age,
                'annual_income': round(annual_income, 2),
                'employment_status': employment_status,
                'account_tenure': account_tenure,
                'product_holdings': product_holdings,
                'relationship_value': round(relationship_value, 2),
                'risk_segment': risk_segment,
                'behavioral_score': round(behavioral_score, 2),
                'credit_score': credit_score,
                'city': np.random.choice(self.uk_cities),
                'last_activity_date': fake.date_between(start_date='-30d', end_date='today')
            })
            
            if (i + 1) % 5000 == 0:
                print(f"Generated {i+1:,} customer profiles")
        
        return pd.DataFrame(customers)
    
    def generate_credit_bureau_data(self, customers_df):
        """Generate credit bureau data linked to customers"""
        print("Generating credit bureau data...")
        
        bureau_data = []
        for _, customer in customers_df.iterrows():
            # Credit history length correlated with age
            max_history = min(customer['customer_age'] - 18, 30)
            credit_history_length = max(0, int(np.random.gamma(1.5, max_history/3)))
            
            # Number of accounts based on age and credit score
            base_accounts = max(1, (customer['credit_score'] - 300) // 100)
            number_of_accounts = max(0, int(np.random.poisson(base_accounts)))
            
            # Total credit limit based on income and credit score
            income_factor = customer['annual_income'] / 30000
            score_factor = customer['credit_score'] / 700
            total_credit_limit = max(500, int(np.random.gamma(2, 5000) * income_factor * score_factor))
            
            # Credit utilization (lower for better credit scores)
            if customer['credit_score'] >= 750:
                utilization = np.random.beta(1, 4)  # Low utilization for good credit
            elif customer['credit_score'] >= 650:
                utilization = np.random.beta(2, 3)  # Moderate utilization
            else:
                utilization = np.random.beta(3, 2)  # High utilization for poor credit
            
            credit_utilization = round(utilization, 3)
            
            # Payment history (better for higher credit scores)
            if customer['credit_score'] >= 750:
                payment_history = np.random.uniform(0.95, 1.0)
            elif customer['credit_score'] >= 650:
                payment_history = np.random.uniform(0.85, 0.95)
            else:
                payment_history = np.random.uniform(0.6, 0.85)
            
            # Public records (bankruptcies, judgments) - rare and correlated with low scores
            public_records = 0
            if customer['credit_score'] < 600:
                public_records = np.random.poisson(0.3)  # Low probability
            
            bureau_data.append({
                'customer_id': customer['customer_id'],
                'credit_score': customer['credit_score'],  # Same as customer profile
                'credit_history_length': credit_history_length,
                'number_of_accounts': number_of_accounts,
                'total_credit_limit': total_credit_limit,
                'credit_utilization': credit_utilization,
                'payment_history': round(payment_history, 3),
                'public_records': public_records
            })
        
        return pd.DataFrame(bureau_data)
    
    def generate_credit_applications(self, customers_df, n_applications=100000):
        """Generate credit applications with realistic approval/default patterns"""
        print(f"Generating {n_applications:,} credit applications...")
        
        applications = []
        
        for i in range(n_applications):
            # Select random customer (some customers have multiple applications)
            customer = customers_df.sample(1).iloc[0]
            
            # Application date
            app_date = fake.date_between(start_date=self.start_date, end_date=self.end_date)
            
            # Loan amount based on income and credit score
            max_loan = customer['annual_income'] * 5  # 5x annual income max
            if customer['credit_score'] >= 750:
                loan_multiplier = np.random.uniform(0.1, 0.8)
            elif customer['credit_score'] >= 650:
                loan_multiplier = np.random.uniform(0.1, 0.6)
            else:
                loan_multiplier = np.random.uniform(0.1, 0.4)
            
            loan_amount = max(1000, int(max_loan * loan_multiplier))
            
            # Debt-to-income ratio
            existing_debt = customer['annual_income'] * np.random.uniform(0.1, 0.6)
            debt_to_income = (existing_debt + loan_amount * 0.2) / customer['annual_income']
            
            # Loan purpose
            loan_purpose = np.random.choice(self.loan_purposes)
            
            # Application approval logic (realistic banking criteria)
            approval_score = 0
            
            # Credit score weight (40%)
            if customer['credit_score'] >= 750:
                approval_score += 40
            elif customer['credit_score'] >= 650:
                approval_score += 25
            elif customer['credit_score'] >= 550:
                approval_score += 10
            
            # DTI weight (30%)
            if debt_to_income < 0.3:
                approval_score += 30
            elif debt_to_income < 0.5:
                approval_score += 15
            elif debt_to_income < 0.7:
                approval_score += 5
            
            # Employment weight (20%)
            if customer['employment_status'] == 'Full-time':
                approval_score += 20
            elif customer['employment_status'] in ['Part-time', 'Self-employed']:
                approval_score += 10
            elif customer['employment_status'] == 'Contract':
                approval_score += 15
            
            # Relationship weight (10%)
            if customer['account_tenure'] > 5:
                approval_score += 10
            elif customer['account_tenure'] > 2:
                approval_score += 5
            
            # Add some randomness
            approval_score += np.random.normal(0, 15)
            
            # Decision
            if approval_score >= 70:
                application_status = 'Approved'
            elif approval_score >= 50:
                application_status = np.random.choice(['Approved', 'Declined'], p=[0.3, 0.7])
            else:
                application_status = 'Declined'
            
            # Default probability for approved loans
            default_flag = 0
            if application_status == 'Approved':
                # Default probability based on credit score and DTI
                base_default_prob = 0.05  # 5% base default rate
                
                if customer['credit_score'] < 600:
                    default_prob = base_default_prob * 3
                elif customer['credit_score'] < 650:
                    default_prob = base_default_prob * 2
                elif customer['credit_score'] < 750:
                    default_prob = base_default_prob * 1.2
                else:
                    default_prob = base_default_prob * 0.5
                
                # Adjust for DTI
                if debt_to_income > 0.5:
                    default_prob *= 2
                elif debt_to_income > 0.4:
                    default_prob *= 1.5
                
                # Adjust for employment
                if customer['employment_status'] in ['Unemployed', 'Student']:
                    default_prob *= 2.5
                elif customer['employment_status'] in ['Part-time', 'Self-employed']:
                    default_prob *= 1.3
                
                # Random default assignment
                default_flag = 1 if np.random.random() < default_prob else 0
            
            applications.append({
                'application_id': f'APP_{i+1:07d}',
                'customer_id': customer['customer_id'],
                'application_date': app_date,
                'loan_amount': loan_amount,
                'loan_purpose': loan_purpose,
                'employment_status': customer['employment_status'],
                'annual_income': customer['annual_income'],
                'debt_to_income_ratio': round(debt_to_income, 3),
                'credit_score': customer['credit_score'],
                'application_status': application_status,
                'default_flag': default_flag
            })
            
            if (i + 1) % 10000 == 0:
                print(f"Generated {i+1:,} applications")
        
        return pd.DataFrame(applications)
    
    def generate_transactions(self, customers_df, n_transactions=100000):
        """Generate transaction data with fraud patterns"""
        print(f"Generating {n_transactions:,} transactions...")
        
        transactions = []
        
        # Pre-calculate customer spending patterns
        customer_patterns = {}
        for _, customer in customers_df.iterrows():
            # Monthly spending based on income
            monthly_spending = customer['annual_income'] / 12 * np.random.uniform(0.6, 1.2)
            avg_transaction = monthly_spending / np.random.uniform(20, 100)  # 20-100 transactions/month
            
            # Preferred merchants based on demographics
            if customer['customer_age'] < 30:
                merchant_prefs = ['Online Shopping', 'Restaurants', 'Entertainment', 'Groceries']
            elif customer['customer_age'] > 60:
                merchant_prefs = ['Groceries', 'Healthcare', 'Utilities', 'Fuel']
            else:
                merchant_prefs = ['Groceries', 'Fuel', 'Restaurants', 'Online Shopping']
            
            customer_patterns[customer['customer_id']] = {
                'avg_transaction': avg_transaction,
                'monthly_spending': monthly_spending,
                'merchant_prefs': merchant_prefs
            }
        
        for i in range(n_transactions):
            # Select customer
            customer = customers_df.sample(1).iloc[0]
            pattern = customer_patterns[customer['customer_id']]
            
            # Transaction date
            trans_date = fake.date_time_between(start_date=self.start_date, end_date=self.end_date)
            
            # Transaction amount (log-normal distribution)
            base_amount = pattern['avg_transaction']
            amount = max(1, np.random.lognormal(np.log(base_amount), 1))
            
            # Merchant category
            if np.random.random() < 0.7:  # 70% chance of preferred category
                merchant_category = np.random.choice(pattern['merchant_prefs'])
            else:
                merchant_category = np.random.choice(self.merchant_categories)
            
            # Transaction type
            if merchant_category == 'ATM':
                transaction_type = 'ATM Withdrawal'
            elif merchant_category == 'Transfer':
                transaction_type = 'Transfer'
            else:
                transaction_type = 'Purchase'
            
            # Location (mostly home city, sometimes travel)
            if np.random.random() < 0.85:  # 85% in home city
                location = customer['city']
            else:
                location = np.random.choice(self.uk_cities)
            
            # Device info
            device_types = ['Mobile', 'Desktop', 'Chip', 'Contactless', 'ATM']
            device_info = np.random.choice(device_types)
            
            # Fraud detection logic
            fraud_flag = 0
            fraud_indicators = 0
            
            # Amount-based fraud indicators
            if amount > pattern['avg_transaction'] * 10:  # Unusually large transaction
                fraud_indicators += 3
            elif amount > pattern['avg_transaction'] * 5:
                fraud_indicators += 1
            
            # Time-based indicators (late night transactions)
            if trans_date.hour < 6 or trans_date.hour > 22:
                if merchant_category not in ['Fuel', 'ATM']:
                    fraud_indicators += 1
            
            # Location-based indicators
            if location != customer['city']:
                fraud_indicators += 2
                # Multiple locations same day (velocity)
                if np.random.random() < 0.1:  # 10% chance
                    fraud_indicators += 3
            
            # Merchant category unusual for customer
            if merchant_category not in pattern['merchant_prefs'] and merchant_category in ['Online Shopping']:
                fraud_indicators += 1
            
            # Device inconsistency
            if device_info in ['Desktop'] and merchant_category in ['Fuel', 'Groceries']:
                fraud_indicators += 2
            
            # Final fraud determination (very low base rate)
            base_fraud_rate = 0.001  # 0.1% base fraud rate
            fraud_multiplier = 1 + fraud_indicators * 0.5
            
            if np.random.random() < base_fraud_rate * fraud_multiplier:
                fraud_flag = 1
                # Fraudulent transactions often have higher amounts
                amount *= np.random.uniform(2, 10)
            
            # Investigation status for fraud cases
            if fraud_flag == 1:
                investigation_status = np.random.choice(['Pending', 'Confirmed', 'False Positive'], 
                                                      p=[0.3, 0.6, 0.1])
            else:
                investigation_status = 'Not Investigated'
            
            transactions.append({
                'transaction_id': f'TXN_{i+1:08d}',
                'customer_id': customer['customer_id'],
                'transaction_date': trans_date,
                'amount': round(amount, 2),
                'merchant_category': merchant_category,
                'transaction_type': transaction_type,
                'location': location,
                'device_info': device_info,
                'fraud_flag': fraud_flag,
                'investigation_status': investigation_status
            })
            
            if (i + 1) % 100000 == 0:
                print(f"Generated {i+1:,} transactions")
        
        return pd.DataFrame(transactions)
    
    def generate_model_predictions(self, applications_df, transactions_df, n_predictions=50000):
        """Generate historical model predictions for monitoring"""
        print(f"Generating {n_predictions:,} model predictions...")
        
        predictions = []
        model_versions = ['v1.0', 'v1.1', 'v1.2', 'v2.0']
        
        # Sample from applications and transactions
        sample_apps = applications_df.sample(n_predictions//2) if len(applications_df) >= n_predictions//2 else applications_df
        sample_txns = transactions_df.sample(n_predictions//2) if len(transactions_df) >= n_predictions//2 else transactions_df.sample(n_predictions//2)
        
        # Credit predictions
        for _, app in sample_apps.iterrows():
            model_version = np.random.choice(model_versions)
            
            # Risk score based on actual default
            if app['default_flag'] == 1:
                risk_score = np.random.beta(3, 2) * 1000  # Higher scores for defaults
            else:
                risk_score = np.random.beta(2, 3) * 1000  # Lower scores for non-defaults
            
            # Feature importance (mock)
            features = {
                'credit_score': app['credit_score'],
                'debt_to_income': app['debt_to_income_ratio'],
                'income': app['annual_income'],
                'employment': app['employment_status']
            }
            
            # Business decision
            if app['application_status'] == 'Approved':
                business_decision = 'Approve' if risk_score < 600 else 'Decline'
            else:
                business_decision = 'Decline'
            
            predictions.append({
                'prediction_id': f'PRED_{len(predictions)+1:07d}',
                'model_version': model_version,
                'customer_id': app['customer_id'],
                'prediction_date': app['application_date'],
                'prediction_type': 'Credit Risk',
                'risk_score': round(risk_score, 2),
                'fraud_probability': None,
                'model_features': str(features),
                'prediction_explanation': f"Risk driven by credit_score={app['credit_score']}, DTI={app['debt_to_income_ratio']:.2f}",
                'business_decision': business_decision,
                'actual_outcome': 'Default' if app['default_flag'] == 1 else 'No Default'
            })
        
        # Fraud predictions
        for _, txn in sample_txns.iterrows():
            model_version = np.random.choice(model_versions)
            
            # Fraud probability based on actual fraud
            if txn['fraud_flag'] == 1:
                fraud_prob = np.random.beta(3, 1)  # High probability for fraud
            else:
                fraud_prob = np.random.beta(1, 9)  # Low probability for legitimate
            
            risk_score = None  # Not applicable for fraud detection
            
            features = {
                'amount': txn['amount'],
                'merchant_category': txn['merchant_category'],
                'location': txn['location'],
                'device': txn['device_info']
            }
            
            business_decision = 'Block' if fraud_prob > 0.7 else 'Allow'
            
            predictions.append({
                'prediction_id': f'PRED_{len(predictions)+1:07d}',
                'model_version': model_version,
                'customer_id': txn['customer_id'],
                'prediction_date': txn['transaction_date'],
                'prediction_type': 'Fraud Detection',
                'risk_score': risk_score,
                'fraud_probability': round(fraud_prob, 4),
                'model_features': str(features),
                'prediction_explanation': f"Fraud risk from amount=${txn['amount']:.2f}, location={txn['location']}",
                'business_decision': business_decision,
                'actual_outcome': 'Fraud' if txn['fraud_flag'] == 1 else 'Legitimate'
            })
        
        return pd.DataFrame(predictions)
    
    def generate_all_datasets(self):
        """Generate all datasets and save to CSV files"""
        print("=" * 60)
        print("FINRISK SYNTHETIC DATA GENERATION")
        print("=" * 60)
        
        # Generate datasets
        customers_df = self.generate_customer_profiles()
        bureau_df = self.generate_credit_bureau_data(customers_df)
        applications_df = self.generate_credit_applications(customers_df)
        transactions_df = self.generate_transactions(customers_df)
        predictions_df = self.generate_model_predictions(applications_df, transactions_df)
        
        # Save datasets
        print("\nSaving datasets to CSV files...")
        customers_df.to_csv('customer_profiles.csv', index=False)
        bureau_df.to_csv('credit_bureau_data.csv', index=False)
        applications_df.to_csv('credit_applications.csv', index=False)
        transactions_df.to_csv('transaction_data.csv', index=False)
        predictions_df.to_csv('model_predictions.csv', index=False)
        
        # Generate summary statistics
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        
        print(f"Customer Profiles: {len(customers_df):,} records")
        print(f"  - Age range: {customers_df['customer_age'].min()}-{customers_df['customer_age'].max()}")
        print(f"  - Income range: £{customers_df['annual_income'].min():,.0f}-£{customers_df['annual_income'].max():,.0f}")
        print(f"  - Credit score range: {customers_df['credit_score'].min()}-{customers_df['credit_score'].max()}")
        print(f"  - Risk segments: {customers_df['risk_segment'].value_counts().to_dict()}")
        
        print(f"\nCredit Applications: {len(applications_df):,} records")
        print(f"  - Approval rate: {(applications_df['application_status'] == 'Approved').mean():.1%}")
        approved_apps = applications_df[applications_df['application_status'] == 'Approved']
        if len(approved_apps) > 0:
            print(f"  - Default rate: {approved_apps['default_flag'].mean():.1%}")
        print(f"  - Loan amount range: £{applications_df['loan_amount'].min():,.0f}-£{applications_df['loan_amount'].max():,.0f}")
        
        print(f"\nTransactions: {len(transactions_df):,} records")
        print(f"  - Fraud rate: {transactions_df['fraud_flag'].mean():.3%}")
        print(f"  - Amount range: £{transactions_df['amount'].min():.2f}-£{transactions_df['amount'].max():,.2f}")
        print(f"  - Top merchant categories: {transactions_df['merchant_category'].value_counts().head(3).to_dict()}")
        
        print(f"\nCredit Bureau Data: {len(bureau_df):,} records")
        print(f"  - Credit utilization range: {bureau_df['credit_utilization'].min():.1%}-{bureau_df['credit_utilization'].max():.1%}")
        print(f"  - Average payment history: {bureau_df['payment_history'].mean():.1%}")
        
        print(f"\nModel Predictions: {len(predictions_df):,} records")
        print(f"  - Credit predictions: {(predictions_df['prediction_type'] == 'Credit Risk').sum():,}")
        print(f"  - Fraud predictions: {(predictions_df['prediction_type'] == 'Fraud Detection').sum():,}")
        
        print("\n" + "=" * 60)
        print("DATA QUALITY VALIDATION")
        print("=" * 60)
        
        # Validate data quality
        datasets = {
            'customers': customers_df,
            'bureau': bureau_df, 
            'applications': applications_df,
            'transactions': transactions_df,
            'predictions': predictions_df
        }
        
        for name, df in datasets.items():
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            duplicates = df.duplicated().sum()
            print(f"{name.capitalize()}: {missing_pct:.2f}% missing values, {duplicates} duplicates")
        
        # Validate business logic
        print("\n" + "=" * 60)
        print("BUSINESS LOGIC VALIDATION")
        print("=" * 60)
        
        # Credit score vs default rate correlation
        score_default_corr = applications_df[applications_df['application_status'] == 'Approved'].groupby(
            pd.cut(applications_df[applications_df['application_status'] == 'Approved']['credit_score'], 
                   bins=[0, 600, 650, 700, 750, 850])
        )['default_flag'].mean()
        print("Default rates by credit score ranges:")
        for score_range, default_rate in score_default_corr.items():
            print(f"  {score_range}: {default_rate:.1%}")
        
        # Transaction amount vs fraud correlation
        fraud_by_amount = transactions_df.groupby(
            pd.cut(transactions_df['amount'], bins=[0, 50, 200, 1000, 10000, float('inf')])
        )['fraud_flag'].mean()
        print("\nFraud rates by transaction amount:")
        for amount_range, fraud_rate in fraud_by_amount.items():
            print(f"  {amount_range}: {fraud_rate:.3%}")
        
        print("\n" + "=" * 60)
        print("DATA GENERATION COMPLETE!")
        print("Generated files:")
        print("  - customer_profiles.csv")
        print("  - credit_bureau_data.csv") 
        print("  - credit_applications.csv")
        print("  - transaction_data.csv")
        print("  - model_predictions.csv")
        print("=" * 60)
        
        return {
            'customers': customers_df,
            'bureau': bureau_df,
            'applications': applications_df, 
            'transactions': transactions_df,
            'predictions': predictions_df
        }

# Usage example
if __name__ == "__main__":
    # Generate datasets
    generator = FinRiskDataGenerator(n_customers=25000)
    datasets = generator.generate_all_datasets()
    
    print("\nDataset generation completed successfully!")
    print("You can now use these datasets for the FinRisk project.")