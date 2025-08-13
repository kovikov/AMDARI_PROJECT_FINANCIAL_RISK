# FinRisk - Financial Risk Management System

A comprehensive financial risk management platform that provides credit scoring, fraud detection, and portfolio risk analysis capabilities.

## 🚀 Features

- **Credit Risk Assessment**: Advanced credit scoring models using ML algorithms
- **Fraud Detection**: Real-time fraud detection with explainable AI
- **Portfolio Management**: Risk metrics and stress testing capabilities
- **Model Monitoring**: Drift detection and performance tracking
- **API Interface**: RESTful API for integration with external systems
- **Dashboard**: Interactive dashboards for risk visualization

## 🏗️ Architecture

```
finrisk/
├── app/                    # Main application code
│   ├── api/               # FastAPI endpoints
│   ├── features/          # Feature engineering
│   ├── models/            # ML model training and inference
│   ├── monitoring/        # Model monitoring and drift detection
│   ├── portfolio/         # Portfolio risk management
│   └── dashboards/        # Streamlit dashboards
├── data/                  # Data storage and management
├── sql/                   # Database schema and migrations
├── scripts/               # Utility scripts
└── tests/                 # Test suite
```

## 🛠️ Technology Stack

- **Backend**: FastAPI, SQLAlchemy, PostgreSQL
- **ML**: scikit-learn, XGBoost, LightGBM
- **Monitoring**: MLflow, Evidently, Prometheus
- **Dashboard**: Streamlit, Plotly, Dash
- **Data Processing**: Pandas, NumPy, PyArrow

## 📋 Prerequisites

- Python 3.9+
- PostgreSQL 13+
- Redis 6+
- Docker (optional)

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd finrisk
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Initialize database**
   ```bash
   python scripts/load_data.py
   ```

5. **Start the API server**
   ```bash
   uvicorn app.api.server:app --reload
   ```

6. **Launch dashboard**
   ```bash
   streamlit run app/dashboards/app_streamlit.py
   ```

## 📊 API Documentation

Once the server is running, visit:
- API docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📈 Model Training

Train all models:
```bash
python scripts/train_all.py
```

## 🔧 Configuration

Key configuration options in `.env`:
- Database connection strings
- Redis configuration
- MLflow tracking URI
- API keys and secrets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions, please open an issue in the GitHub repository. 