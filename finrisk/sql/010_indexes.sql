-- FinRisk Database Indexes
-- Additional indexes for performance optimization

-- Set search path
SET search_path TO finrisk, public;

-- Customer Profiles Indexes
CREATE INDEX IF NOT EXISTS idx_customer_profiles_age ON customer_profiles(customer_age);
CREATE INDEX IF NOT EXISTS idx_customer_profiles_income ON customer_profiles(annual_income);
CREATE INDEX IF NOT EXISTS idx_customer_profiles_employment ON customer_profiles(employment_status);
CREATE INDEX IF NOT EXISTS idx_customer_profiles_city ON customer_profiles(city);
CREATE INDEX IF NOT EXISTS idx_customer_profiles_activity ON customer_profiles(last_activity_date);
CREATE INDEX IF NOT EXISTS idx_customer_profiles_created ON customer_profiles(created_at);

-- Credit Bureau Data Indexes
CREATE INDEX IF NOT EXISTS idx_credit_bureau_score ON credit_bureau_data(credit_score);
CREATE INDEX IF NOT EXISTS idx_credit_bureau_history ON credit_bureau_data(credit_history_length);
CREATE INDEX IF NOT EXISTS idx_credit_bureau_accounts ON credit_bureau_data(number_of_accounts);
CREATE INDEX IF NOT EXISTS idx_credit_bureau_utilization ON credit_bureau_data(credit_utilization);
CREATE INDEX IF NOT EXISTS idx_credit_bureau_payment ON credit_bureau_data(payment_history);

-- Credit Applications Indexes
CREATE INDEX IF NOT EXISTS idx_credit_apps_customer ON credit_applications(customer_id);
CREATE INDEX IF NOT EXISTS idx_credit_apps_amount ON credit_applications(loan_amount);
CREATE INDEX IF NOT EXISTS idx_credit_apps_purpose ON credit_applications(loan_purpose);
CREATE INDEX IF NOT EXISTS idx_credit_apps_dti ON credit_applications(debt_to_income_ratio);
CREATE INDEX IF NOT EXISTS idx_credit_apps_default ON credit_applications(default_flag);
CREATE INDEX IF NOT EXISTS idx_credit_apps_created ON credit_applications(created_at);

-- Transaction Data Indexes
CREATE INDEX IF NOT EXISTS idx_transactions_customer ON transaction_data(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_amount ON transaction_data(amount);
CREATE INDEX IF NOT EXISTS idx_transactions_merchant ON transaction_data(merchant_category);
CREATE INDEX IF NOT EXISTS idx_transactions_type ON transaction_data(transaction_type);
CREATE INDEX IF NOT EXISTS idx_transactions_location ON transaction_data(location);
CREATE INDEX IF NOT EXISTS idx_transactions_device ON transaction_data(device_info);
CREATE INDEX IF NOT EXISTS idx_transactions_investigation ON transaction_data(investigation_status);

-- Model Predictions Indexes
CREATE INDEX IF NOT EXISTS idx_predictions_customer ON model_predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON model_predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_risk_score ON model_predictions(risk_score);
CREATE INDEX IF NOT EXISTS idx_predictions_fraud_prob ON model_predictions(fraud_probability);
CREATE INDEX IF NOT EXISTS idx_predictions_decision ON model_predictions(business_decision);
CREATE INDEX IF NOT EXISTS idx_predictions_outcome ON model_predictions(actual_outcome);

-- Composite Indexes for Common Queries
CREATE INDEX IF NOT EXISTS idx_customer_risk_credit ON customer_profiles(risk_segment, credit_score);
CREATE INDEX IF NOT EXISTS idx_apps_status_date ON credit_applications(application_status, application_date);
CREATE INDEX IF NOT EXISTS idx_transactions_fraud_date ON transaction_data(fraud_flag, transaction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_type_date ON model_predictions(prediction_type, prediction_date);

-- Audit Schema Indexes
SET search_path TO audit, public;

CREATE INDEX IF NOT EXISTS idx_decision_log_customer ON decision_log(customer_id);
CREATE INDEX IF NOT EXISTS idx_decision_log_type ON decision_log(decision_type);
CREATE INDEX IF NOT EXISTS idx_decision_log_date ON decision_log(created_at);
CREATE INDEX IF NOT EXISTS idx_decision_log_model ON decision_log(model_version);

CREATE INDEX IF NOT EXISTS idx_performance_log_model ON model_performance_log(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_performance_log_date ON model_performance_log(evaluation_date);

CREATE INDEX IF NOT EXISTS idx_quality_log_table ON data_quality_log(table_name);
CREATE INDEX IF NOT EXISTS idx_quality_log_date ON data_quality_log(check_date);

-- Monitoring Schema Indexes
SET search_path TO monitoring, public;

CREATE INDEX IF NOT EXISTS idx_drift_feature ON drift_detection(feature_name);
CREATE INDEX IF NOT EXISTS idx_drift_model ON drift_detection(model_name);
CREATE INDEX IF NOT EXISTS idx_drift_date ON drift_detection(detection_date);
CREATE INDEX IF NOT EXISTS idx_drift_score ON drift_detection(drift_score);
CREATE INDEX IF NOT EXISTS idx_drift_is_drifted ON drift_detection(is_drifted);

CREATE INDEX IF NOT EXISTS idx_kpi_name ON kpi_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_kpi_date ON kpi_metrics(metric_date);
CREATE INDEX IF NOT EXISTS idx_kpi_period ON kpi_metrics(metric_period);

CREATE INDEX IF NOT EXISTS idx_alert_type ON alert_history(alert_type);
CREATE INDEX IF NOT EXISTS idx_alert_severity ON alert_history(alert_severity);
CREATE INDEX IF NOT EXISTS idx_alert_resolved ON alert_history(is_resolved);
CREATE INDEX IF NOT EXISTS idx_alert_date ON alert_history(created_at);

-- Partial Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_fraud_transactions ON transaction_data(transaction_date, amount) 
    WHERE fraud_flag = 1;

CREATE INDEX IF NOT EXISTS idx_default_applications ON credit_applications(application_date, loan_amount) 
    WHERE default_flag = 1;

CREATE INDEX IF NOT EXISTS idx_high_risk_customers ON customer_profiles(credit_score, annual_income) 
    WHERE risk_segment = 'HIGH';

CREATE INDEX IF NOT EXISTS idx_critical_alerts ON alert_history(created_at, alert_type) 
    WHERE alert_severity = 'CRITICAL';

-- Reset search path
SET search_path TO public;

