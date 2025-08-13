-- FinRisk Sample Views
-- Common analytical views for reporting and analysis

-- Set search path
SET search_path TO finrisk, public;

-- Customer Risk Summary View
CREATE OR REPLACE VIEW customer_risk_summary AS
SELECT 
    cp.customer_id,
    cp.customer_age,
    cp.annual_income,
    cp.employment_status,
    cp.risk_segment,
    cp.credit_score,
    cp.behavioral_score,
    cbd.credit_history_length,
    cbd.credit_utilization,
    cbd.payment_history,
    cbd.public_records,
    COUNT(ca.application_id) as total_applications,
    COUNT(CASE WHEN ca.application_status = 'Approved' THEN 1 END) as approved_applications,
    COUNT(CASE WHEN ca.default_flag = 1 THEN 1 END) as defaults,
    COUNT(t.transaction_id) as total_transactions,
    COUNT(CASE WHEN t.fraud_flag = 1 THEN 1 END) as fraud_transactions,
    AVG(t.amount) as avg_transaction_amount,
    MAX(t.transaction_date) as last_transaction_date
FROM customer_profiles cp
LEFT JOIN credit_bureau_data cbd ON cp.customer_id = cbd.customer_id
LEFT JOIN credit_applications ca ON cp.customer_id = ca.customer_id
LEFT JOIN transaction_data t ON cp.customer_id = t.customer_id
GROUP BY 
    cp.customer_id, cp.customer_age, cp.annual_income, cp.employment_status,
    cp.risk_segment, cp.credit_score, cp.behavioral_score,
    cbd.credit_history_length, cbd.credit_utilization, cbd.payment_history, cbd.public_records;

-- Credit Application Performance View
CREATE OR REPLACE VIEW credit_application_performance AS
SELECT 
    ca.application_id,
    ca.customer_id,
    ca.application_date,
    ca.loan_amount,
    ca.loan_purpose,
    ca.employment_status,
    ca.annual_income,
    ca.debt_to_income_ratio,
    ca.credit_score,
    ca.application_status,
    ca.default_flag,
    cp.risk_segment,
    cp.behavioral_score,
    cbd.credit_utilization,
    cbd.payment_history,
    cbd.public_records,
    mp.risk_score,
    mp.business_decision,
    mp.actual_outcome,
    CASE 
        WHEN mp.actual_outcome = 'Default' AND ca.default_flag = 1 THEN 'Correct Prediction'
        WHEN mp.actual_outcome = 'No Default' AND ca.default_flag = 0 THEN 'Correct Prediction'
        ELSE 'Incorrect Prediction'
    END as prediction_accuracy
FROM credit_applications ca
JOIN customer_profiles cp ON ca.customer_id = cp.customer_id
JOIN credit_bureau_data cbd ON ca.customer_id = cbd.customer_id
LEFT JOIN model_predictions mp ON ca.customer_id = mp.customer_id 
    AND mp.prediction_type = 'Credit Risk'
    AND mp.prediction_date >= ca.application_date;

-- Fraud Detection Analysis View
CREATE OR REPLACE VIEW fraud_detection_analysis AS
SELECT 
    t.transaction_id,
    t.customer_id,
    t.transaction_date,
    t.amount,
    t.merchant_category,
    t.transaction_type,
    t.location,
    t.device_info,
    t.fraud_flag,
    t.investigation_status,
    cp.risk_segment,
    cp.behavioral_score,
    mp.fraud_probability,
    mp.business_decision,
    CASE 
        WHEN t.fraud_flag = 1 AND mp.fraud_probability > 0.5 THEN 'True Positive'
        WHEN t.fraud_flag = 0 AND mp.fraud_probability <= 0.5 THEN 'True Negative'
        WHEN t.fraud_flag = 1 AND mp.fraud_probability <= 0.5 THEN 'False Negative'
        WHEN t.fraud_flag = 0 AND mp.fraud_probability > 0.5 THEN 'False Positive'
        ELSE 'No Prediction'
    END as detection_result
FROM transaction_data t
JOIN customer_profiles cp ON t.customer_id = cp.customer_id
LEFT JOIN model_predictions mp ON t.customer_id = mp.customer_id 
    AND mp.prediction_type = 'Fraud Detection'
    AND mp.prediction_date >= t.transaction_date;

-- Model Performance Summary View
CREATE OR REPLACE VIEW model_performance_summary AS
SELECT 
    mp.model_version,
    mp.prediction_type,
    COUNT(*) as total_predictions,
    COUNT(CASE WHEN mp.actual_outcome IS NOT NULL THEN 1 END) as predictions_with_outcome,
    AVG(mp.risk_score) as avg_risk_score,
    AVG(mp.fraud_probability) as avg_fraud_probability,
    COUNT(CASE WHEN mp.business_decision = 'Approve' THEN 1 END) as approved_decisions,
    COUNT(CASE WHEN mp.business_decision = 'Decline' THEN 1 END) as declined_decisions,
    COUNT(CASE WHEN mp.business_decision = 'Review' THEN 1 END) as review_decisions,
    MIN(mp.prediction_date) as first_prediction,
    MAX(mp.prediction_date) as last_prediction
FROM model_predictions mp
GROUP BY mp.model_version, mp.prediction_type;

-- Risk Portfolio Summary View
CREATE OR REPLACE VIEW risk_portfolio_summary AS
SELECT 
    cp.risk_segment,
    COUNT(DISTINCT cp.customer_id) as customer_count,
    AVG(cp.annual_income) as avg_income,
    AVG(cp.credit_score) as avg_credit_score,
    AVG(cp.behavioral_score) as avg_behavioral_score,
    SUM(ca.loan_amount) as total_loan_amount,
    COUNT(ca.application_id) as total_applications,
    COUNT(CASE WHEN ca.application_status = 'Approved' THEN 1 END) as approved_applications,
    COUNT(CASE WHEN ca.default_flag = 1 THEN 1 END) as defaults,
    ROUND(
        COUNT(CASE WHEN ca.default_flag = 1 THEN 1 END) * 100.0 / 
        NULLIF(COUNT(CASE WHEN ca.application_status = 'Approved' THEN 1 END), 0), 2
    ) as default_rate_percent,
    AVG(t.amount) as avg_transaction_amount,
    COUNT(CASE WHEN t.fraud_flag = 1 THEN 1 END) as fraud_transactions
FROM customer_profiles cp
LEFT JOIN credit_applications ca ON cp.customer_id = ca.customer_id
LEFT JOIN transaction_data t ON cp.customer_id = t.customer_id
GROUP BY cp.risk_segment;

-- Monthly Performance Trends View
CREATE OR REPLACE VIEW monthly_performance_trends AS
SELECT 
    DATE_TRUNC('month', ca.application_date) as month,
    COUNT(ca.application_id) as total_applications,
    COUNT(CASE WHEN ca.application_status = 'Approved' THEN 1 END) as approved_applications,
    COUNT(CASE WHEN ca.application_status = 'Declined' THEN 1 END) as declined_applications,
    COUNT(CASE WHEN ca.default_flag = 1 THEN 1 END) as defaults,
    SUM(ca.loan_amount) as total_loan_amount,
    AVG(ca.loan_amount) as avg_loan_amount,
    ROUND(
        COUNT(CASE WHEN ca.application_status = 'Approved' THEN 1 END) * 100.0 / 
        NULLIF(COUNT(ca.application_id), 0), 2
    ) as approval_rate_percent,
    ROUND(
        COUNT(CASE WHEN ca.default_flag = 1 THEN 1 END) * 100.0 / 
        NULLIF(COUNT(CASE WHEN ca.application_status = 'Approved' THEN 1 END), 0), 2
    ) as default_rate_percent
FROM credit_applications ca
GROUP BY DATE_TRUNC('month', ca.application_date)
ORDER BY month;

-- Customer Activity Summary View
CREATE OR REPLACE VIEW customer_activity_summary AS
SELECT 
    cp.customer_id,
    cp.customer_age,
    cp.risk_segment,
    cp.credit_score,
    COUNT(t.transaction_id) as transaction_count,
    SUM(t.amount) as total_spent,
    AVG(t.amount) as avg_transaction_amount,
    COUNT(DISTINCT t.merchant_category) as unique_merchants,
    COUNT(DISTINCT t.location) as unique_locations,
    COUNT(CASE WHEN t.fraud_flag = 1 THEN 1 END) as fraud_count,
    MAX(t.transaction_date) as last_transaction_date,
    EXTRACT(DAYS FROM (CURRENT_DATE - MAX(t.transaction_date))) as days_since_last_transaction
FROM customer_profiles cp
LEFT JOIN transaction_data t ON cp.customer_id = t.customer_id
GROUP BY cp.customer_id, cp.customer_age, cp.risk_segment, cp.credit_score;

-- Reset search path
SET search_path TO public;

