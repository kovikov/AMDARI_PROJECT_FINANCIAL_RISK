# 🏦 FinRisk - Financial Risk Management Platform

A comprehensive, enterprise-grade financial risk management system that provides advanced credit scoring, fraud detection, portfolio risk analysis, and real-time monitoring capabilities.

## 🌟 Key Features

### 🔐 Credit Risk Assessment
- **Advanced ML Models**: XGBoost, LightGBM, and ensemble methods for credit scoring
- **Real-time Scoring**: Instant credit risk assessment with explainable AI
- **Risk Grading**: A-F risk classification system
- **Batch Processing**: High-throughput batch credit scoring

### 🚨 Fraud Detection
- **Real-time Detection**: Instant fraud probability scoring for transactions
- **Multi-factor Analysis**: Amount, time, location, and merchant-based detection
- **Risk Levels**: LOW, MEDIUM, HIGH risk classification
- **Automated Actions**: Block/Allow recommendations

### 📊 Portfolio Management
- **Risk Metrics**: VaR, Expected Shortfall, and stress testing
- **Portfolio Analytics**: Comprehensive risk analysis and reporting
- **Stress Testing**: Scenario-based portfolio stress testing
- **Performance Tracking**: Historical performance and trend analysis

### 📈 Model Monitoring & MLOps
- **MLflow Integration**: Complete model lifecycle management
- **Drift Detection**: Automated model performance monitoring
- **A/B Testing**: Model comparison and validation
- **Performance Tracking**: Real-time model metrics and alerts

### 🎯 Interactive Dashboards
- **Streamlit Dashboards**: User-friendly risk visualization
- **Real-time Metrics**: Live performance and risk indicators
- **Customizable Views**: Role-based dashboard access
- **Export Capabilities**: PDF and Excel report generation

## 🏗️ System Architecture

```
AMDARI_FINANCE_PROJECT/
├── finrisk/                    # Main application directory
│   ├── app/                   # Core application modules
│   │   ├── api/              # FastAPI REST endpoints
│   │   ├── models/           # ML model training & inference
│   │   ├── monitoring/       # Model monitoring & drift detection
│   │   ├── portfolio/        # Portfolio risk management
│   │   └── dashboards/       # Streamlit dashboard applications
│   ├── data/                 # Data storage and management
│   ├── sql/                  # Database schemas and migrations
│   ├── scripts/              # Utility and automation scripts
│   ├── tests/                # Comprehensive test suite
│   ├── main.py              # FastAPI application entry point
│   ├── dashboard.py         # Streamlit dashboard application
│   └── mlflow_dashboard.py  # MLflow tracking dashboard
├── .github/                  # GitHub Actions CI/CD workflows
├── .venv/                   # Python virtual environment
├── transaction_data.csv     # Sample transaction dataset
├── credit_applications.csv  # Sample credit applications
├── customer_profiles.csv    # Customer profile data
└── credit_bureau_data.csv   # Credit bureau information
```

## 🛠️ Technology Stack

### Backend & API
- **FastAPI**: High-performance async API framework
- **SQLAlchemy**: Database ORM and migrations
- **PostgreSQL**: Primary database
- **Redis**: Caching and session management

### Machine Learning
- **scikit-learn**: Traditional ML algorithms
- **XGBoost**: Gradient boosting for credit scoring
- **LightGBM**: Fast gradient boosting for fraud detection
- **Pandas/NumPy**: Data manipulation and numerical computing
- **MLflow**: Model lifecycle management

### Monitoring & Observability
- **MLflow**: Model tracking and registry
- **Evidently**: Model monitoring and drift detection
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and alerting

### Frontend & Dashboards
- **Streamlit**: Interactive web applications
- **Plotly**: Interactive charts and visualizations
- **Dash**: Advanced dashboard components

### DevOps & Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-service orchestration
- **GitHub Actions**: CI/CD pipelines
- **Kubernetes**: Production deployment (optional)

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.9+
- PostgreSQL 13+
- Redis 6+
- Docker & Docker Compose (optional)

### 1. Clone and Setup
```bash
git clone https://github.com/kovikov/AMDARI_PROJECT_FINANCIAL_RISK.git
cd AMDARI_PROJECT_FINANCIAL_RISK
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r finrisk/requirements.txt
```

### 3. Database Setup
```bash
# Initialize database and load sample data
cd finrisk
python init_database.py
python enhanced_data_loader.py
```

### 4. Start Services

#### Option A: Individual Services
```bash
# Start FastAPI server
python main.py

# Start Streamlit dashboard (in new terminal)
streamlit run dashboard.py

# Start MLflow tracking server (in new terminal)
python mlflow_dashboard.py
```

#### Option B: Docker Compose
```bash
docker-compose up -d
```

### 5. Access Applications
- **API Documentation**: http://localhost:8001/docs
- **Streamlit Dashboard**: http://localhost:8501
- **MLflow Dashboard**: http://localhost:5000

## 📚 API Endpoints

### Credit Risk Assessment
- `POST /api/v1/credit/score` - Score individual credit application
- `POST /api/v1/credit/batch` - Batch credit scoring
- `GET /api/v1/models/status` - Model status and versions

### Fraud Detection
- `POST /api/v1/fraud/detect` - Real-time fraud detection
- `GET /api/v1/metrics` - API performance metrics

### Health & Monitoring
- `GET /health` - Health check endpoint
- `GET /` - API information and endpoints

## 🧪 Testing

### Run All Tests
```bash
cd finrisk
pytest tests/ -v
```

### Specific Test Categories
```bash
# API tests
pytest tests/test_api.py

# Model tests
pytest tests/test_credit_risk_trainer.py
pytest tests/test_fraud_detection_trainer.py

# Integration tests
pytest tests/test_data_loader.py
```

## 📊 Sample Data

The project includes comprehensive sample datasets:
- **Transaction Data**: 10MB of transaction records for fraud detection
- **Credit Applications**: 9MB of credit application data
- **Customer Profiles**: 2.1MB of customer demographic data
- **Credit Bureau Data**: 989KB of credit history information

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the `finrisk/` directory:
```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/finrisk
REDIS_URL=redis://localhost:6379

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=finrisk

# API Configuration
API_HOST=127.0.0.1
API_PORT=8001
DEBUG=True
```

### Model Configuration
- Model hyperparameters in `finrisk/app/models/config.py`
- Feature engineering settings in `finrisk/app/features/config.py`
- Monitoring thresholds in `finrisk/app/monitoring/config.py`

## 📈 Model Training

### Train All Models
```bash
cd finrisk
python scripts/train_all.py
```

### Individual Model Training
```bash
# Credit risk model
python -m app.models.credit_risk_trainer

# Fraud detection model
python -m app.models.fraud_detection_trainer
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build individual services
docker build -t finrisk-api .
docker run -p 8001:8001 finrisk-api
```

### Production Deployment
- Use Kubernetes for production scaling
- Configure proper logging and monitoring
- Set up SSL/TLS certificates
- Implement proper authentication and authorization

## 📖 Documentation

- **[API Documentation](finrisk/API_DOCUMENTATION.md)** - Complete API reference
- **[Setup Instructions](finrisk/SETUP_INSTRUCTIONS.md)** - Detailed setup guide
- **[Database Guide](finrisk/DATABASE_GUIDE.md)** - Database schema and management
- **[MLflow Guide](finrisk/README_MLFLOW.md)** - Model tracking and monitoring
- **[Dashboard Guide](finrisk/README_DASHBOARD.md)** - Dashboard usage and customization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive tests for new features
- Update documentation for API changes
- Use conventional commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/kovikov/AMDARI_PROJECT_FINANCIAL_RISK/issues)
- **Documentation**: Check the `finrisk/` directory for detailed guides
- **API Docs**: Available at `/docs` when the server is running

## 🙏 Acknowledgments

- Built with FastAPI, Streamlit, and MLflow
- Sample data generated for demonstration purposes
- Inspired by modern fintech risk management practices

---

**FinRisk** - Empowering financial institutions with intelligent risk management solutions. 🚀
