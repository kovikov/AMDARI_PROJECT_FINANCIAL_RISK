-- FinRisk Database Schema Initialization
-- This file creates all necessary tables for the FinRisk application

-- Enable UUID extension for PostgreSQL
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Customer Profiles Table
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
    credit_score INTEGER CHECK (credit_score >= 300 AND credit_score <= 850),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Credit Applications Table
CREATE TABLE IF NOT EXISTS credit_applications (
    application_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL REFERENCES customer_profiles(customer_id) ON DELETE CASCADE,
    loan_amount DECIMAL(15,2) NOT NULL CHECK (loan_amount > 0),
    loan_purpose VARCHAR(100) NOT NULL,
    employment_status VARCHAR(50),
    annual_income DECIMAL(15,2),
    credit_score INTEGER CHECK (credit_score >= 300 AND credit_score <= 850),
    existing_debt DECIMAL(15,2) DEFAULT 0 CHECK (existing_debt >= 0),
    collateral_value DECIMAL(15,2) CHECK (collateral_value >= 0),
    application_status VARCHAR(50) DEFAULT 'PENDING' CHECK (application_status IN ('PENDING', 'APPROVED', 'REJECTED', 'UNDER_REVIEW')),
    risk_score DECIMAL(5,4) CHECK (risk_score >= 0 AND risk_score <= 1),
    approval_probability DECIMAL(5,4) CHECK (approval_probability >= 0 AND approval_probability <= 1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transaction Data Table
CREATE TABLE IF NOT EXISTS transaction_data (
    transaction_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL REFERENCES customer_profiles(customer_id) ON DELETE CASCADE,
    transaction_date TIMESTAMP NOT NULL,
    amount DECIMAL(15,2) NOT NULL CHECK (amount != 0),
    transaction_type VARCHAR(50),
    merchant_category VARCHAR(100),
    merchant_name VARCHAR(255),
    location VARCHAR(255),
    is_fraudulent BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Credit Bureau Data Table
CREATE TABLE IF NOT EXISTS credit_bureau_data (
    record_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL REFERENCES customer_profiles(customer_id) ON DELETE CASCADE,
    credit_score INTEGER CHECK (credit_score >= 300 AND credit_score <= 850),
    payment_history TEXT,
    credit_utilization DECIMAL(5,4) CHECK (credit_utilization >= 0 AND credit_utilization <= 1),
    length_of_credit_history INTEGER CHECK (length_of_credit_history >= 0),
    number_of_accounts INTEGER CHECK (number_of_accounts >= 0),
    derogatory_marks INTEGER DEFAULT 0 CHECK (derogatory_marks >= 0),
    inquiries_last_6_months INTEGER DEFAULT 0 CHECK (inquiries_last_6_months >= 0),
    public_records INTEGER DEFAULT 0 CHECK (public_records >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model Predictions Table
CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL REFERENCES customer_profiles(customer_id) ON DELETE CASCADE,
    application_id VARCHAR(50) REFERENCES credit_applications(application_id) ON DELETE CASCADE,
    model_type VARCHAR(50) NOT NULL,
    prediction_type VARCHAR(50) NOT NULL,
    prediction_value DECIMAL(10,6) NOT NULL,
    confidence_score DECIMAL(5,4) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    features_used JSONB,
    model_version VARCHAR(50),
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fraud Detection Results Table
CREATE TABLE IF NOT EXISTS fraud_detection_results (
    detection_id VARCHAR(50) PRIMARY KEY,
    transaction_id VARCHAR(50) REFERENCES transaction_data(transaction_id) ON DELETE CASCADE,
    customer_id VARCHAR(50) NOT NULL REFERENCES customer_profiles(customer_id) ON DELETE CASCADE,
    model_type VARCHAR(50) NOT NULL,
    anomaly_score DECIMAL(10,6) NOT NULL,
    risk_level VARCHAR(20) CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    is_fraudulent BOOLEAN DEFAULT FALSE,
    confidence_score DECIMAL(5,4) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    features_used JSONB,
    detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model Training History Table
CREATE TABLE IF NOT EXISTS model_training_history (
    training_id VARCHAR(50) PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    training_date TIMESTAMP NOT NULL,
    dataset_size INTEGER NOT NULL,
    training_metrics JSONB,
    hyperparameters JSONB,
    model_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit Log Table
CREATE TABLE IF NOT EXISTS audit_log (
    log_id SERIAL PRIMARY KEY,
    user_id VARCHAR(50),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(50),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_customer_profiles_email ON customer_profiles(email);
CREATE INDEX IF NOT EXISTS idx_customer_profiles_credit_score ON customer_profiles(credit_score);
CREATE INDEX IF NOT EXISTS idx_credit_applications_customer_id ON credit_applications(customer_id);
CREATE INDEX IF NOT EXISTS idx_credit_applications_status ON credit_applications(application_status);
CREATE INDEX IF NOT EXISTS idx_credit_applications_risk_score ON credit_applications(risk_score);
CREATE INDEX IF NOT EXISTS idx_transaction_data_customer_id ON transaction_data(customer_id);
CREATE INDEX IF NOT EXISTS idx_transaction_data_date ON transaction_data(transaction_date);
CREATE INDEX IF NOT EXISTS idx_transaction_data_fraudulent ON transaction_data(is_fraudulent);
CREATE INDEX IF NOT EXISTS idx_credit_bureau_data_customer_id ON credit_bureau_data(customer_id);
CREATE INDEX IF NOT EXISTS idx_model_predictions_customer_id ON model_predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_model_predictions_model_type ON model_predictions(model_type);
CREATE INDEX IF NOT EXISTS idx_fraud_detection_results_customer_id ON fraud_detection_results(customer_id);
CREATE INDEX IF NOT EXISTS idx_fraud_detection_results_risk_level ON fraud_detection_results(risk_level);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);

-- Create composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_transaction_data_customer_date ON transaction_data(customer_id, transaction_date);
CREATE INDEX IF NOT EXISTS idx_credit_applications_customer_status ON credit_applications(customer_id, application_status);
CREATE INDEX IF NOT EXISTS idx_model_predictions_customer_type ON model_predictions(customer_id, model_type);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_customer_profiles_updated_at 
    BEFORE UPDATE ON customer_profiles 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_credit_applications_updated_at 
    BEFORE UPDATE ON credit_applications 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE OR REPLACE VIEW customer_summary AS
SELECT 
    cp.customer_id,
    cp.first_name,
    cp.last_name,
    cp.email,
    cp.credit_score,
    cp.annual_income,
    cp.employment_status,
    COUNT(DISTINCT ca.application_id) as total_applications,
    COUNT(DISTINCT CASE WHEN ca.application_status = 'APPROVED' THEN ca.application_id END) as approved_applications,
    COUNT(DISTINCT td.transaction_id) as total_transactions,
    COUNT(DISTINCT CASE WHEN td.is_fraudulent = TRUE THEN td.transaction_id END) as fraudulent_transactions,
    AVG(ca.risk_score) as avg_risk_score,
    MAX(ca.created_at) as last_application_date
FROM customer_profiles cp
LEFT JOIN credit_applications ca ON cp.customer_id = ca.customer_id
LEFT JOIN transaction_data td ON cp.customer_id = td.customer_id
GROUP BY cp.customer_id, cp.first_name, cp.last_name, cp.email, cp.credit_score, cp.annual_income, cp.employment_status;

CREATE OR REPLACE VIEW fraud_risk_summary AS
SELECT 
    cp.customer_id,
    cp.first_name,
    cp.last_name,
    COUNT(td.transaction_id) as total_transactions,
    COUNT(CASE WHEN td.is_fraudulent = TRUE THEN td.transaction_id END) as fraudulent_transactions,
    ROUND(COUNT(CASE WHEN td.is_fraudulent = TRUE THEN td.transaction_id END)::DECIMAL / COUNT(td.transaction_id) * 100, 2) as fraud_rate,
    AVG(fdr.anomaly_score) as avg_anomaly_score,
    MAX(fdr.risk_level) as highest_risk_level
FROM customer_profiles cp
LEFT JOIN transaction_data td ON cp.customer_id = td.customer_id
LEFT JOIN fraud_detection_results fdr ON cp.customer_id = fdr.customer_id
GROUP BY cp.customer_id, cp.first_name, cp.last_name
HAVING COUNT(td.transaction_id) > 0;

-- Insert initial configuration data
INSERT INTO model_training_history (training_id, model_type, model_version, training_date, dataset_size, training_metrics, hyperparameters, model_path) 
VALUES 
    ('initial_credit_risk_v1', 'credit_risk', '1.0.0', CURRENT_TIMESTAMP, 0, '{"auc": 0.0, "gini": 0.0}', '{"n_estimators": 100}', '/models/credit_risk_v1.pkl'),
    ('initial_fraud_detection_v1', 'fraud_detection', '1.0.0', CURRENT_TIMESTAMP, 0, '{"precision": 0.0, "recall": 0.0}', '{"contamination": 0.1}', '/models/fraud_detection_v1.pkl')
ON CONFLICT DO NOTHING;

-- Grant necessary permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO finrisk_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO finrisk_user;

