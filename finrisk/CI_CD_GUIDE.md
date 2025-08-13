# FinRisk CI/CD Pipeline Guide

This guide explains the Continuous Integration and Continuous Deployment (CI/CD) pipeline setup for the FinRisk Financial Risk Assessment & Fraud Detection Engine.

## Overview

The CI/CD pipeline consists of three main workflows:

1. **CI (Continuous Integration)** - Automated testing, linting, and code quality checks
2. **CD (Continuous Deployment)** - Automated deployment to staging and production
3. **ML Training** - Automated machine learning model training and deployment

## Workflow Structure

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

**Jobs:**

#### Test Job
- **Matrix Strategy:** Tests against Python 3.9, 3.10, and 3.11
- **Services:** PostgreSQL 13 and Redis 6 containers
- **Steps:**
  - Install dependencies
  - Run linting (flake8, black, isort)
  - Run type checking (mypy)
  - Run tests with coverage
  - Upload coverage to Codecov

#### Security Job
- **Tools:** Bandit (security linting) and Safety (dependency vulnerability scanning)
- **Output:** Security reports as artifacts

#### Build Job
- **Dependencies:** Requires successful test completion
- **Steps:**
  - Build Docker image
  - Upload Docker image as artifact

### 2. CD Workflow (`.github/workflows/cd.yml`)

**Triggers:**
- Push to `main` branch
- Completion of CI workflow

**Jobs:**

#### Deploy Staging
- **Condition:** Push to `develop` branch
- **Environment:** `staging`
- **Steps:**
  - Deploy to staging environment
  - Run smoke tests

#### Deploy Production
- **Condition:** Push to `main` branch
- **Environment:** `production`
- **Steps:**
  - Deploy to production environment
  - Run health checks
  - Send deployment notifications

#### Model Deployment
- **Dependencies:** Production deployment
- **Steps:**
  - Deploy latest ML models using MLflow

#### Monitoring
- **Dependencies:** Production and model deployment
- **Steps:**
  - Start monitoring systems
  - Verify monitoring setup

### 3. ML Training Workflow (`.github/workflows/ml-training.yml`)

**Triggers:**
- **Scheduled:** Every Sunday at 2 AM UTC
- **Manual:** Workflow dispatch with parameters

**Jobs:**

#### Data Validation
- **Purpose:** Check for new data in the last 7 days
- **Output:** Boolean flag indicating if new data exists

#### Train Credit Risk Models
- **Condition:** New data available or manual trigger
- **Steps:**
  - Train all credit risk models
  - Evaluate performance metrics
  - Upload model artifacts

#### Train Fraud Detection Models
- **Condition:** New data available or manual trigger
- **Steps:**
  - Train all fraud detection models
  - Evaluate performance metrics
  - Upload model artifacts

#### Model Registry
- **Dependencies:** Both training jobs
- **Steps:**
  - Register best models in MLflow
  - Generate training reports
  - Upload reports as artifacts

## Environment Setup

### GitHub Environments

Create the following environments in your GitHub repository:

1. **staging**
   - Protection rules: Require reviewers
   - Environment variables:
     ```
     STAGING_DB_HOST=staging-db.example.com
     STAGING_DB_PORT=5432
     STAGING_DB_NAME=finrisk_staging
     STAGING_DB_USER=finrisk_user
     STAGING_DB_PASSWORD=<secure-password>
     ```

2. **production**
   - Protection rules: Require reviewers, wait timer
   - Environment variables:
     ```
     PROD_DB_HOST=prod-db.example.com
     PROD_DB_PORT=5432
     PROD_DB_NAME=finrisk_prod
     PROD_DB_USER=finrisk_user
     PROD_DB_PASSWORD=<secure-password>
     ```

### Secrets Management

Add the following secrets to your GitHub repository:

```
# Database Secrets
DB_PASSWORD=<production-db-password>
STAGING_DB_PASSWORD=<staging-db-password>

# API Keys
SLACK_WEBHOOK_URL=<slack-notification-webhook>
EMAIL_SMTP_PASSWORD=<email-notification-password>

# Cloud Provider Secrets
AWS_ACCESS_KEY_ID=<aws-access-key>
AWS_SECRET_ACCESS_KEY=<aws-secret-key>
DOCKER_REGISTRY_PASSWORD=<docker-registry-password>
```

## Local Development with Docker

### Quick Start

```bash
# Clone the repository
git clone https://github.com/kovikov/AMDARI_PROJECT_FINANCIAL_RISK.git
cd AMDARI_PROJECT_FINANCIAL_RISK/finrisk

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api
```

### Service URLs

- **API Server:** http://localhost:8000
- **Streamlit Dashboard:** http://localhost:8501
- **MLflow Dashboard:** http://localhost:8502
- **MLflow Tracking:** http://localhost:5000
- **PostgreSQL:** localhost:5432
- **Redis:** localhost:6379

### Development Commands

```bash
# Run tests locally
docker-compose exec api pytest tests/ -v

# Run linting
docker-compose exec api flake8 app/ --max-line-length=100
docker-compose exec api black app/
docker-compose exec api isort app/

# Train models
docker-compose exec api python scripts/train_credit_models.py
docker-compose exec api python -c "from app.models.fraud_detection_trainer import FraudDetectionTrainer; FraudDetectionTrainer().train_all_models()"

# Load data
docker-compose exec api python enhanced_data_loader.py
```

## Deployment Strategies

### Staging Deployment

1. **Branch Strategy:** Use `develop` branch for staging
2. **Automated Deployment:** Every push to `develop` triggers staging deployment
3. **Testing:** Smoke tests run after deployment
4. **Manual Promotion:** Staging must be approved before production

### Production Deployment

1. **Branch Strategy:** Use `main` branch for production
2. **Automated Deployment:** Every push to `main` triggers production deployment
3. **Health Checks:** Comprehensive health checks after deployment
4. **Rollback:** Automatic rollback on health check failure

### Model Deployment

1. **Scheduled Training:** Weekly automated model training
2. **Performance Monitoring:** Models are evaluated before deployment
3. **A/B Testing:** New models can be deployed alongside existing ones
4. **Rollback:** Easy rollback to previous model versions

## Monitoring and Alerting

### Application Monitoring

- **Health Checks:** `/health` endpoint for application status
- **Metrics:** Prometheus metrics collection
- **Logging:** Structured logging with correlation IDs
- **Tracing:** Distributed tracing for request flows

### Model Monitoring

- **Performance Metrics:** ROC AUC, Gini, Precision, Recall
- **Data Drift:** Population Stability Index (PSI) monitoring
- **Prediction Drift:** Model prediction distribution monitoring
- **Business Metrics:** Financial impact of model decisions

### Alerting

- **Deployment Alerts:** Slack notifications for deployment status
- **Model Performance Alerts:** Alerts when model performance degrades
- **Infrastructure Alerts:** System resource and availability alerts
- **Security Alerts:** Vulnerability and security incident alerts

## Best Practices

### Code Quality

1. **Linting:** All code must pass flake8, black, and isort checks
2. **Type Checking:** Use mypy for static type checking
3. **Test Coverage:** Maintain >80% test coverage
4. **Documentation:** Keep documentation up to date

### Security

1. **Dependency Scanning:** Regular security scans with Safety
2. **Code Scanning:** Security analysis with Bandit
3. **Secret Management:** Use GitHub Secrets for sensitive data
4. **Access Control:** Implement proper authentication and authorization

### Performance

1. **Caching:** Use Redis for caching frequently accessed data
2. **Database Optimization:** Proper indexing and query optimization
3. **Resource Management:** Monitor and optimize resource usage
4. **Scalability:** Design for horizontal scaling

### Monitoring

1. **Observability:** Comprehensive logging, metrics, and tracing
2. **Alerting:** Proactive alerting for issues
3. **Dashboards:** Real-time monitoring dashboards
4. **Incident Response:** Clear procedures for incident handling

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Check dependency conflicts
   - Verify Python version compatibility
   - Review linting errors

2. **Test Failures**
   - Check database connectivity
   - Verify test data availability
   - Review test environment setup

3. **Deployment Failures**
   - Check environment variables
   - Verify service dependencies
   - Review deployment logs

4. **Model Training Failures**
   - Check data quality
   - Verify MLflow connectivity
   - Review model configuration

### Debug Commands

```bash
# Check service status
docker-compose ps

# View service logs
docker-compose logs -f <service-name>

# Access service shell
docker-compose exec <service-name> bash

# Check database connectivity
docker-compose exec api python -c "from app.infra.db import get_db_session; print('DB OK')"

# Check Redis connectivity
docker-compose exec api python -c "from app.infra.cache import get_redis_client; print('Redis OK')"

# Check MLflow connectivity
docker-compose exec api python -c "import mlflow; print('MLflow OK')"
```

## Future Enhancements

1. **Multi-Cloud Deployment:** Support for AWS, Azure, and GCP
2. **Kubernetes Integration:** Native Kubernetes deployment
3. **Advanced Monitoring:** Integration with APM tools
4. **Feature Flags:** A/B testing and feature toggles
5. **Blue-Green Deployment:** Zero-downtime deployments
6. **Chaos Engineering:** Resilience testing
7. **Cost Optimization:** Resource usage optimization
8. **Compliance:** SOC2, GDPR, and other compliance frameworks

## Support

For issues with the CI/CD pipeline:

1. Check the GitHub Actions logs
2. Review the troubleshooting section
3. Create an issue in the repository
4. Contact the development team

---

*This guide is maintained by the FinRisk development team. Last updated: August 2024*
