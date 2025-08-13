# FinRisk Setup Instructions

This guide will help you set up the FinRisk Credit Risk Assessment & Fraud Detection Engine on your local machine.

## Prerequisites

- Python 3.10 or higher
- PostgreSQL 12 or higher
- Redis 6 or higher
- Git
- VS Code (recommended)

## Step 1: Clone and Setup Project

```bash
# Clone the repository (or create the project structure)
mkdir finrisk
cd finrisk

# Create virtual environment
python -m venv finrisk-env

# Activate virtual environment
# On Windows:
finrisk-env\Scripts\activate
# On macOS/Linux:
source finrisk-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Install and Configure PostgreSQL

### Install PostgreSQL

**Windows:**
1. Download PostgreSQL installer from https://www.postgresql.org/download/windows/
2. Run installer and follow setup wizard
3. Remember the password for the `postgres` user

**macOS:**
```bash
# Using Homebrew
brew install postgresql
brew services start postgresql
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### Create Database and User

```bash
# Connect to PostgreSQL as superuser
sudo -u postgres psql

# Or on Windows:
psql -U postgres
```

```sql
-- Create database and user
CREATE DATABASE finrisk_db;
CREATE USER finrisk_user WITH PASSWORD 'finrisk_pass';
GRANT ALL PRIVILEGES ON DATABASE finrisk_db TO finrisk_user;

-- Exit PostgreSQL
\q
```

## Step 3: Install and Configure Redis

### Install Redis

**Windows:**
1. Download Redis from https://github.com/microsoftarchive/redis/releases
2. Extract and run `redis-server.exe`

**macOS:**
```bash
# Using Homebrew
brew install redis
brew services start redis
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

### Test Redis Connection

```bash
redis-cli ping
# Should return: PONG
```

## Step 4: Configure Environment

Copy the example environment file and update with your settings:

```bash
cp .env.example .env
```

Edit `.env` file with your database and Redis configurations:

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=finrisk_db
DB_USER=finrisk_user
DB_PASSWORD=finrisk_pass

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Application Configuration
APP_NAME=FinRisk
APP_VERSION=1.0.0
DEBUG=True
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security Configuration
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30
ALGORITHM=HS256

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=finrisk-experiments
```

## Step 5: Initialize Database Schema

Run the database initialization script to create all necessary tables, indexes, and views:

```bash
# Initialize database schema
python enhanced_data_loader.py --init-only

# Or run the complete data loading process
python enhanced_data_loader.py
```

## Step 6: Install Additional Dependencies

Install dashboard-specific dependencies:

```bash
pip install -r requirements_dashboard.txt
```

## Step 7: Test the Setup

### Test Database Connection

```bash
python test_enhanced_data_loader.py
```

### Test Dashboard

```bash
python test_dashboard.py
```

### Test API Server

```bash
python test_api_server.py
```

## Step 8: Start the Application

### Start the FastAPI Server

```bash
# Start the API server
uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload
```

### Start the Streamlit Dashboard

```bash
# Start the dashboard (in a new terminal)
streamlit run dashboard.py --server.port 8501
```

### Start MLflow Tracking Server (Optional)

```bash
# Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000
```

## Step 9: Access the Application

- **API Documentation**: http://localhost:8000/docs
- **Streamlit Dashboard**: http://localhost:8501
- **MLflow UI**: http://localhost:5000 (if started)

## Step 10: Verify Installation

### Check API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Credit risk assessment
curl -X POST "http://localhost:8000/api/v1/credit/assess" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 1,
    "income": 75000,
    "credit_score": 720,
    "debt_to_income": 0.35,
    "payment_history": "good"
  }'
```

### Check Dashboard Pages

1. **Portfolio Overview**: View overall portfolio metrics
2. **Credit Risk**: Monitor credit risk assessments
3. **Fraud Detection**: Track fraud detection metrics
4. **System Health**: Check system status and database connectivity

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Verify PostgreSQL is running
   - Check credentials in `.env` file
   - Ensure database and user exist

2. **Redis Connection Failed**
   - Verify Redis server is running
   - Check Redis configuration in `.env`

3. **Port Already in Use**
   - Change ports in `.env` file
   - Kill existing processes using the ports

4. **Module Import Errors**
   - Ensure virtual environment is activated
   - Install all dependencies: `pip install -r requirements.txt`

### Logs and Debugging

```bash
# Check application logs
tail -f logs/finrisk.log

# Test database connection directly
python -c "
import psycopg2
conn = psycopg2.connect(
    host='localhost',
    database='finrisk_db',
    user='finrisk_user',
    password='finrisk_pass'
)
print('Database connection successful')
conn.close()
"
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run specific test files
pytest test_enhanced_data_loader.py
pytest test_dashboard.py
pytest test_api_server.py
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

### Database Migrations

```bash
# Apply schema changes
python enhanced_data_loader.py --init-only

# Reset database (WARNING: This will delete all data)
python enhanced_data_loader.py --reset
```

## Production Deployment

### Environment Variables

For production, ensure these environment variables are set:

```bash
DEBUG=False
LOG_LEVEL=WARNING
SECRET_KEY=your-production-secret-key
DB_PASSWORD=your-production-db-password
REDIS_PASSWORD=your-production-redis-password
```

### Security Considerations

1. Use strong passwords for database and Redis
2. Enable SSL/TLS for database connections
3. Configure firewall rules
4. Use environment variables for sensitive data
5. Regular security updates

### Monitoring

1. Set up application monitoring (e.g., Prometheus, Grafana)
2. Configure log aggregation
3. Set up alerting for critical issues
4. Monitor database performance
5. Track API response times

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review application logs
3. Test individual components
4. Verify environment configuration
5. Check system requirements

## Next Steps

After successful setup:

1. **Train Models**: Use the credit risk and fraud detection trainers
2. **Customize Dashboard**: Modify dashboard components for your needs
3. **Add Data Sources**: Integrate with your data sources
4. **Scale Infrastructure**: Consider containerization with Docker
5. **Add Authentication**: Implement user authentication and authorization
6. **Backup Strategy**: Set up regular database backups
7. **Performance Tuning**: Optimize database queries and API performance
