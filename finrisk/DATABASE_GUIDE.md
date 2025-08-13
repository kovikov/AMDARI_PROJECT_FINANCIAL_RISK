# FinRisk Database Guide

This guide covers all database operations for the FinRisk financial risk management system.

## 🏗️ Database Architecture

The FinRisk system uses PostgreSQL with three main schemas:

- **`finrisk`** - Core business tables (customer profiles, applications, transactions, predictions)
- **`audit`** - Audit logging and compliance tracking
- **`monitoring`** - Model monitoring and KPI tracking

## 📋 Database Tables

### Core Tables (finrisk schema)

| Table | Description | Key Fields |
|-------|-------------|------------|
| `customer_profiles` | Customer demographic and risk data | customer_id, credit_score, risk_segment |
| `credit_bureau_data` | Credit bureau information | customer_id, credit_history_length |
| `credit_applications` | Loan applications and decisions | application_id, customer_id, loan_amount |
| `transaction_data` | Transaction and fraud data | transaction_id, customer_id, amount |
| `model_predictions` | ML model outputs | prediction_id, customer_id, prediction_type |

### Audit Tables (audit schema)

| Table | Description |
|-------|-------------|
| `decision_log` | All model decisions and reasoning |
| `model_performance_log` | Model performance metrics |
| `data_quality_log` | Data quality checks and issues |

### Monitoring Tables (monitoring schema)

| Table | Description |
|-------|-------------|
| `drift_detection` | Feature drift detection results |
| `kpi_metrics` | Key performance indicators |
| `alert_history` | System alerts and notifications |

## 🛠️ Command Line Interface

The FinRisk system provides a comprehensive CLI for database operations:

### Basic Commands

```bash
# Show all available commands
python scripts/db_cli.py --help

# Check database configuration
python scripts/db_cli.py config

# Check database connection
python scripts/db_cli.py check

# Set up database schema (creates all tables, indexes, views)
python scripts/db_cli.py setup
```

### Table Operations

```bash
# Show all tables in finrisk schema
python scripts/db_cli.py tables

# Show tables in audit schema
python scripts/db_cli.py tables --schema audit

# Show tables in monitoring schema
python scripts/db_cli.py tables --schema monitoring

# Get row count for a table
python scripts/db_cli.py count customer_profiles

# Get row count for table in specific schema
python scripts/db_cli.py count decision_log --schema audit
```

### Query Operations

```bash
# Execute a simple query
python scripts/db_cli.py query "SELECT COUNT(*) FROM finrisk.customer_profiles"

# Execute query and save results to CSV
python scripts/db_cli.py query "SELECT * FROM finrisk.customer_profiles LIMIT 10" --output customers.csv

# Execute query and save results to Excel
python scripts/db_cli.py query "SELECT * FROM finrisk.credit_applications WHERE application_status = 'Approved'" --output approved_apps.xlsx

# Complex query example
python scripts/db_cli.py query "
SELECT 
    cp.risk_segment,
    COUNT(*) as customer_count,
    AVG(cp.credit_score) as avg_credit_score,
    AVG(cp.annual_income) as avg_income
FROM finrisk.customer_profiles cp
GROUP BY cp.risk_segment
ORDER BY avg_credit_score DESC
"
```

### Maintenance Operations

```bash
# Create backup of a table
python scripts/db_cli.py backup customer_profiles

# Create backup with custom suffix
python scripts/db_cli.py backup customer_profiles --suffix "_20241201"

# Restore table from backup
python scripts/db_cli.py restore customer_profiles customer_profiles_backup

# Run VACUUM on a table (reclaim storage and update statistics)
python scripts/db_cli.py vacuum customer_profiles

# Show database size information
python scripts/db_cli.py size

# Truncate a table (remove all rows)
python scripts/db_cli.py truncate customer_profiles

# Create database indexes for performance
python scripts/db_cli.py indexes

# Perform database health check
python scripts/db_cli.py health
```

## 🔧 Python API Usage

### Basic Database Operations

```python
from app.infra.db import (
    get_db_session, read_sql_query, insert_dataframe,
    get_table_count, check_database_connection
)

# Check connection
if check_database_connection():
    print("Database is accessible")

# Execute query and get DataFrame
df = read_sql_query("SELECT * FROM finrisk.customer_profiles LIMIT 10")

# Get table count
count = get_table_count("customer_profiles")
print(f"Customer profiles: {count} rows")

# Insert DataFrame into table
import pandas as pd
new_data = pd.DataFrame({
    'customer_id': ['CUST001', 'CUST002'],
    'customer_age': [30, 35],
    'annual_income': [75000, 85000],
    # ... other fields
})
insert_dataframe(new_data, "customer_profiles")
```

### Session Management

```python
from app.infra.db import get_db_session
from sqlalchemy import text

# Use context manager for automatic cleanup
with get_db_session() as session:
    # Execute raw SQL
    result = session.execute(text("SELECT COUNT(*) FROM finrisk.customer_profiles"))
    count = result.fetchone()[0]
    
    # Execute parameterized query
    result = session.execute(
        text("SELECT * FROM finrisk.customer_profiles WHERE risk_segment = :segment"),
        {"segment": "HIGH"}
    )
    high_risk_customers = result.fetchall()
```

### Advanced Operations

```python
from app.infra.db import (
    backup_table, restore_table_from_backup,
    vacuum_table, get_database_size, truncate_table,
    create_indexes, health_check
)

# Backup table
backup_name = backup_table("customer_profiles")

# Restore from backup
restore_table_from_backup("customer_profiles", backup_name)

# Run maintenance
vacuum_table("customer_profiles")

# Get size information
size_info = get_database_size()
print(f"Total database size: {size_info['total_size_pretty']}")

# Truncate table (remove all rows)
truncate_table("customer_profiles")

# Create performance indexes
create_indexes()

# Perform health check
health_info = health_check()
print(f"Database status: {health_info['status']}")
print(f"Response time: {health_info['response_time_seconds']} seconds")
```

## 📊 Sample Queries

### Customer Analysis

```sql
-- Customer risk distribution
SELECT 
    risk_segment,
    COUNT(*) as customer_count,
    ROUND(AVG(credit_score), 2) as avg_credit_score,
    ROUND(AVG(annual_income), 2) as avg_income
FROM finrisk.customer_profiles
GROUP BY risk_segment
ORDER BY avg_credit_score DESC;

-- High-risk customers with low credit scores
SELECT 
    customer_id,
    customer_age,
    annual_income,
    credit_score,
    risk_segment
FROM finrisk.customer_profiles
WHERE risk_segment = 'HIGH' AND credit_score < 600
ORDER BY credit_score ASC;
```

### Application Performance

```sql
-- Application approval rates by month
SELECT 
    DATE_TRUNC('month', application_date) as month,
    COUNT(*) as total_applications,
    COUNT(CASE WHEN application_status = 'Approved' THEN 1 END) as approved,
    ROUND(
        COUNT(CASE WHEN application_status = 'Approved' THEN 1 END) * 100.0 / COUNT(*), 2
    ) as approval_rate
FROM finrisk.credit_applications
GROUP BY DATE_TRUNC('month', application_date)
ORDER BY month;

-- Default rates by loan amount range
SELECT 
    CASE 
        WHEN loan_amount < 10000 THEN 'Small (<10k)'
        WHEN loan_amount < 50000 THEN 'Medium (10k-50k)'
        ELSE 'Large (>50k)'
    END as loan_size,
    COUNT(*) as total_loans,
    COUNT(CASE WHEN default_flag = 1 THEN 1 END) as defaults,
    ROUND(
        COUNT(CASE WHEN default_flag = 1 THEN 1 END) * 100.0 / COUNT(*), 2
    ) as default_rate
FROM finrisk.credit_applications
WHERE application_status = 'Approved'
GROUP BY 
    CASE 
        WHEN loan_amount < 10000 THEN 'Small (<10k)'
        WHEN loan_amount < 50000 THEN 'Medium (10k-50k)'
        ELSE 'Large (>50k)'
    END;
```

### Fraud Detection

```sql
-- Fraud detection performance
SELECT 
    t.fraud_flag,
    COUNT(*) as transaction_count,
    ROUND(AVG(t.amount), 2) as avg_amount,
    COUNT(CASE WHEN mp.fraud_probability > 0.5 THEN 1 END) as flagged_as_fraud
FROM finrisk.transaction_data t
LEFT JOIN finrisk.model_predictions mp ON t.customer_id = mp.customer_id 
    AND mp.prediction_type = 'Fraud Detection'
GROUP BY t.fraud_flag;

-- High-value suspicious transactions
SELECT 
    t.transaction_id,
    t.customer_id,
    t.amount,
    t.merchant_category,
    t.location,
    mp.fraud_probability,
    CASE 
        WHEN mp.fraud_probability > 0.8 THEN 'High Risk'
        WHEN mp.fraud_probability > 0.5 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END as risk_level
FROM finrisk.transaction_data t
LEFT JOIN finrisk.model_predictions mp ON t.customer_id = mp.customer_id 
    AND mp.prediction_type = 'Fraud Detection'
WHERE t.amount > 1000 AND mp.fraud_probability > 0.5
ORDER BY t.amount DESC, mp.fraud_probability DESC;
```

### Model Performance

```sql
-- Model prediction accuracy
SELECT 
    mp.prediction_type,
    mp.model_version,
    COUNT(*) as total_predictions,
    COUNT(CASE WHEN mp.actual_outcome IS NOT NULL THEN 1 END) as with_outcome,
    ROUND(
        COUNT(CASE WHEN mp.actual_outcome IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 2
    ) as outcome_coverage
FROM finrisk.model_predictions mp
GROUP BY mp.prediction_type, mp.model_version
ORDER BY mp.prediction_type, mp.model_version;

-- Recent model decisions
SELECT 
    mp.prediction_date,
    mp.prediction_type,
    mp.business_decision,
    COUNT(*) as decision_count
FROM finrisk.model_predictions mp
WHERE mp.prediction_date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY mp.prediction_date, mp.prediction_type, mp.business_decision
ORDER BY mp.prediction_date DESC, mp.prediction_type;
```

## 🔍 Monitoring and Maintenance

### Regular Maintenance Tasks

```bash
# Daily: Check database connection and health
python scripts/db_cli.py health
python scripts/db_cli.py check

# Weekly: Run VACUUM on large tables
python scripts/db_cli.py vacuum customer_profiles
python scripts/db_cli.py vacuum transaction_data
python scripts/db_cli.py vacuum credit_applications

# Monthly: Check database size and performance
python scripts/db_cli.py size
python scripts/db_cli.py health

# Before major updates: Create backups
python scripts/db_cli.py backup customer_profiles --suffix "_$(date +%Y%m%d)"

# After schema changes: Recreate indexes
python scripts/db_cli.py indexes
```

### Health Monitoring

The health check provides comprehensive database status:

```bash
python scripts/db_cli.py health
```

**Health Check Results Include:**
- **Status**: `healthy` or `unhealthy`
- **Connection**: Database connectivity status
- **Response Time**: Query performance in seconds
- **Customer Count**: Current data volume
- **Timestamp**: When the check was performed
- **Error Details**: If any issues are detected

**Example Output:**
```
🏥 Performing database health check...

📊 Health Check Results:
  Status: healthy
  Connection: ✅ Connected
  Response Time: 0.023 seconds
  Customer Count: 15420
  Timestamp: 2024-12-01T10:30:15.123456
```

### Performance Monitoring

```sql
-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname IN ('finrisk', 'audit', 'monitoring')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes
WHERE schemaname IN ('finrisk', 'audit', 'monitoring')
ORDER BY idx_scan DESC;
```

## 🚨 Troubleshooting

### Common Issues

1. **Connection Failed**
   ```bash
   # Check if PostgreSQL is running
   python scripts/db_cli.py check
   
   # Verify configuration
   python scripts/db_cli.py config
   ```

2. **Permission Denied**
   - Ensure the database user has proper permissions
   - Check if the database exists: `python scripts/db_cli.py setup`

3. **Table Not Found**
   - Run setup: `python scripts/db_cli.py setup`
   - Check tables: `python scripts/db_cli.py tables`

4. **Performance Issues**
   - Run VACUUM: `python scripts/db_cli.py vacuum <table_name>`
   - Check size: `python scripts/db_cli.py size`

### Getting Help

```bash
# Show all available commands
python scripts/db_cli.py --help

# Show help for specific command
python scripts/db_cli.py query --help
python scripts/db_cli.py backup --help
```

## 📈 Best Practices

1. **Always backup before major changes**
2. **Use parameterized queries to prevent SQL injection**
3. **Run VACUUM regularly on frequently updated tables**
4. **Monitor database size and performance**
5. **Use appropriate indexes for your query patterns**
6. **Test queries on small datasets first**
7. **Keep database credentials secure**

## 🔗 Related Files

- `app/infra/db.py` - Database infrastructure module
- `scripts/db_cli.py` - Command-line interface
- `sql/001_init_schema.sql` - Database schema
- `sql/010_indexes.sql` - Performance indexes
- `sql/020_sample_views.sql` - Analytical views
- `app/config.py` - Database configuration
