# FinRisk API Documentation

## Overview

The FinRisk API is a comprehensive REST API for credit risk assessment and fraud detection. Built with FastAPI, it provides real-time scoring, risk assessment, and portfolio analysis capabilities.

## 🚀 Quick Start

### Start the API Server

```bash
# Development mode (with hot reload)
python run_api.py

# Or directly with uvicorn
uvicorn app.api.server:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🔐 Authentication

The API uses JWT-based authentication. Include the token in the Authorization header:

```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" http://localhost:8000/api/v1/credit/score
```

## 📊 API Endpoints

### Health & Status

#### GET `/health`
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

#### GET `/status`
Get comprehensive system status including database and cache health.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "connected": true
    },
    "cache": {
      "status": "healthy",
      "hit_rate": 0.85
    },
    "api": {
      "status": "healthy",
      "version": "1.0.0"
    }
  }
}
```

### Credit Risk Assessment

#### POST `/api/v1/credit/score`
Calculate credit score for a customer.

**Request:**
```json
{
  "customer_id": "CUST_001",
  "features": {
    "annual_income": 75000,
    "credit_score": 720,
    "employment_years": 5,
    "existing_debt": 15000,
    "loan_amount": 50000
  }
}
```

**Response:**
```json
{
  "customer_id": "CUST_001",
  "credit_score": 0.85,
  "risk_level": "LOW",
  "confidence": 0.92,
  "factors": [
    {"factor": "credit_score", "impact": "positive", "weight": 0.3},
    {"factor": "income_debt_ratio", "impact": "positive", "weight": 0.25}
  ],
  "recommendation": "Approve with standard terms",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### POST `/api/v1/credit/assess`
Assess credit risk for a loan application.

**Request:**
```json
{
  "customer_id": "CUST_001",
  "loan_amount": 50000,
  "loan_purpose": "home_improvement",
  "employment_status": "full_time",
  "annual_income": 75000,
  "credit_score": 720,
  "existing_debt": 15000,
  "collateral_value": 25000
}
```

**Response:**
```json
{
  "application_id": "APP_20240115_103000_CUST_001",
  "customer_id": "CUST_001",
  "risk_score": 0.25,
  "risk_level": "LOW",
  "approval_probability": 0.88,
  "recommended_amount": 50000,
  "conditions": ["Income verification required"],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### POST `/api/v1/credit/apply`
Submit a credit application for processing.

**Request:** Same as `/assess` endpoint

**Response:**
```json
{
  "application_id": "APP_20240115_103000_CUST_001",
  "status": "APPROVED",
  "message": "Application submitted successfully",
  "estimated_processing_time": "Immediate",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### GET `/api/v1/credit/applications/{customer_id}`
Get credit applications for a customer.

**Response:**
```json
{
  "customer_id": "CUST_001",
  "applications": [
    {
      "application_id": "APP_000001",
      "customer_id": "CUST_001",
      "loan_amount": 50000,
      "status": "APPROVED",
      "submitted_date": "2024-01-15T10:30:00Z",
      "risk_score": 0.25
    }
  ],
  "total_count": 1,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### GET `/api/v1/credit/status/{application_id}`
Get status of a credit application.

**Response:**
```json
{
  "application_id": "APP_000001",
  "status": "APPROVED",
  "risk_score": 0.25,
  "approval_probability": 0.88,
  "recommended_amount": 50000,
  "conditions": ["Income verification required"],
  "last_updated": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T10:00:00Z"
}
```

### Fraud Detection

#### POST `/api/v1/fraud/detect`
Detect fraud in a transaction.

**Request:**
```json
{
  "transaction_id": "TXN_001",
  "customer_id": "CUST_001",
  "amount": 1500.00,
  "merchant_category": "electronics",
  "location": "New York, NY",
  "timestamp": "2024-01-15T10:30:00Z",
  "device_id": "DEV_001"
}
```

**Response:**
```json
{
  "transaction_id": "TXN_001",
  "fraud_score": 0.15,
  "risk_level": "LOW",
  "anomaly_detected": false,
  "confidence": 0.92,
  "recommendation": "Approve transaction",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### POST `/api/v1/fraud/analyze`
Analyze fraud patterns for a customer.

**Request:**
```json
{
  "customer_id": "CUST_001",
  "time_period": "30d"
}
```

**Response:**
```json
{
  "customer_id": "CUST_001",
  "fraud_risk_score": 0.12,
  "risk_level": "LOW",
  "suspicious_patterns": [],
  "recommendations": ["Continue monitoring"],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Portfolio Analysis

#### GET `/api/v1/portfolio/summary`
Get portfolio summary and risk metrics.

**Response:**
```json
{
  "total_customers": 15000,
  "total_exposure": 250000000,
  "average_risk_score": 0.35,
  "risk_distribution": {
    "low": 0.45,
    "medium": 0.35,
    "high": 0.20
  },
  "portfolio_health": "GOOD",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### GET `/api/v1/portfolio/risk-analysis`
Get detailed portfolio risk analysis.

**Response:**
```json
{
  "portfolio_metrics": {
    "var_95": 15000000,
    "expected_loss": 2500000,
    "concentration_risk": 0.15
  },
  "risk_factors": [
    {"factor": "economic_conditions", "impact": "medium"},
    {"factor": "industry_concentration", "impact": "low"}
  ],
  "recommendations": ["Diversify high-risk segments"],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=true
API_WORKERS=4

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=finrisk_db
DB_USER=finrisk_user
DB_PASSWORD=finrisk_pass

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# MLflow Configuration
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_REGISTRY_URI=sqlite:///mlflow.db

# Email Configuration
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=rfondufe@gmail.com
EMAIL_PASSWORD=your_app_password
NOTIFICATION_EMAIL=rfondufe@gmail.com
```

### Docker Configuration

```bash
# Start with Docker Compose
docker-compose up api

# Or build and run manually
docker build -t finrisk-api .
docker run -p 8000:8000 finrisk-api
```

## 📈 Monitoring & Metrics

### GET `/metrics`
Get application metrics for monitoring systems.

**Response:**
```json
{
  "metrics": {
    "requests_total": "counter",
    "request_duration_seconds": "histogram",
    "active_connections": "gauge",
    "database_connections": "gauge",
    "cache_hit_rate": "gauge"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🧪 Testing

### Run API Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api_server.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

### Test API Endpoints

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test credit scoring (with authentication)
curl -X POST http://localhost:8000/api/v1/credit/score \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "TEST_001", "features": {"annual_income": 50000}}'
```

## 🚨 Error Handling

The API returns consistent error responses:

```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456",
  "path": "/api/v1/credit/score"
}
```

### Common HTTP Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error
- `503` - Service Unavailable

## 🔒 Security

### Rate Limiting
- Default: 100 requests per minute per user
- Configurable per endpoint

### Input Validation
- All inputs validated using Pydantic models
- SQL injection protection
- XSS protection

### Audit Logging
- All API requests logged
- Decision logging for ML models
- User action tracking

## 📚 SDK Examples

### Python Client

```python
import requests

class FinRiskClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def calculate_credit_score(self, customer_id: str, features: dict):
        response = requests.post(
            f"{self.base_url}/api/v1/credit/score",
            headers=self.headers,
            json={"customer_id": customer_id, "features": features}
        )
        return response.json()

# Usage
client = FinRiskClient("http://localhost:8000", "your_token")
score = client.calculate_credit_score("CUST_001", {"annual_income": 75000})
```

### JavaScript Client

```javascript
class FinRiskClient {
    constructor(baseUrl, token) {
        this.baseUrl = baseUrl;
        this.headers = { 'Authorization': `Bearer ${token}` };
    }
    
    async calculateCreditScore(customerId, features) {
        const response = await fetch(`${this.baseUrl}/api/v1/credit/score`, {
            method: 'POST',
            headers: { ...this.headers, 'Content-Type': 'application/json' },
            body: JSON.stringify({ customer_id: customerId, features })
        });
        return response.json();
    }
}

// Usage
const client = new FinRiskClient('http://localhost:8000', 'your_token');
const score = await client.calculateCreditScore('CUST_001', { annual_income: 75000 });
```

## 🆘 Support

For API support:
1. Check the `/docs` endpoint for interactive documentation
2. Review error logs in the application
3. Contact the development team
4. Create an issue in the repository

---

*This documentation is maintained by the FinRisk development team. Last updated: January 2024*
