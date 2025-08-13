# FinRisk MLflow Integration

This document describes the MLflow integration for the FinRisk Credit Risk Assessment & Fraud Detection Engine.

## Overview

MLflow is integrated into the FinRisk application to provide comprehensive experiment tracking, model management, and model registry capabilities. The integration supports both credit risk and fraud detection models with specialized tracking for each domain.

## Features

### 🧪 Experiment Tracking
- **Credit Risk Models**: Track XGBoost, Random Forest, Logistic Regression experiments
- **Fraud Detection Models**: Track Isolation Forest, One-Class SVM, Elliptic Envelope experiments
- **Hyperparameter Tuning**: Log grid search results and best parameters
- **Model Comparison**: Compare multiple models side-by-side
- **Data Quality Reports**: Track data quality metrics and distributions

### 📦 Model Registry
- **Model Registration**: Register models from experiments
- **Version Management**: Track model versions and stages
- **Production Deployment**: Promote models to production
- **Model Loading**: Load production models for inference

### 📊 Metrics Tracking
- **Credit Risk Metrics**: AUC, Gini coefficient, KS statistic, PSI, Precision, Recall, F1-score
- **Fraud Detection Metrics**: AUC, Precision at K, Lift at Percentile, Detection Rate at FPR
- **Data Quality Metrics**: Missing values, data distributions, feature statistics
- **Training Metrics**: Training time, data sizes, feature counts

## Architecture

### Core Components

1. **FinRiskMLflowTracker** (`app/monitoring/mlflow_tracker.py`)
   - Main tracking class for experiments
   - Handles experiment creation and management
   - Logs models, metrics, parameters, and artifacts

2. **FinRiskModelRegistry** (`app/monitoring/mlflow_tracker.py`)
   - Model registry management
   - Handles model registration and versioning
   - Manages model stages (Development, Staging, Production)

3. **Integration Points**
   - Credit Risk Trainer (`app/models/credit_risk_trainer.py`)
   - Fraud Detection Trainer (`app/models/fraud_detection_trainer.py`)
   - Streamlit Dashboard (`mlflow_dashboard.py`)

### Data Flow

```
Training Script → MLflow Tracker → Experiment Logging → Model Registry → Production Deployment
     ↓
Dashboard ← MLflow UI ← Experiment Data ← Model Artifacts
```

## Setup and Configuration

### Environment Variables

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=sqlite:///mlflow.db          # Local SQLite (default)
MLFLOW_REGISTRY_URI=sqlite:///mlflow.db          # Local SQLite (default)
MLFLOW_EXPERIMENT_NAME=finrisk-experiments       # Experiment name
```

### For Production

```bash
# Remote MLflow Server
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_REGISTRY_URI=http://mlflow-server:5000
MLFLOW_EXPERIMENT_NAME=finrisk-production
```

### Dependencies

The MLflow integration requires the following packages (already included in `requirements.txt`):

```bash
mlflow==2.6.0
```

## Usage

### Basic Experiment Tracking

```python
from app.monitoring.mlflow_tracker import log_credit_risk_model

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Log experiment
run_id = log_credit_risk_model(
    model=model,
    model_name="random_forest",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    hyperparameters=model.get_params(),
    metrics={
        'auc': 0.85,
        'precision': 0.82,
        'recall': 0.78,
        'f1_score': 0.80
    },
    feature_importance=feature_importance_dict,
    model_type="sklearn"
)
```

### Fraud Detection Tracking

```python
from app.monitoring.mlflow_tracker import log_fraud_detection_model

# Log fraud detection experiment
run_id = log_fraud_detection_model(
    model=model,
    model_name="isolation_forest",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    hyperparameters=model.get_params(),
    metrics={
        'auc': 0.92,
        'precision': 0.88,
        'recall': 0.85
    },
    anomaly_scores=anomaly_scores,
    model_type="sklearn"
)
```

### Model Registration

```python
from app.monitoring.mlflow_tracker import register_model_to_production

# Register model to production
version = register_model_to_production(
    run_id="your_run_id",
    model_name="credit_risk_model",
    description="Production credit risk model v2.0"
)
```

### Loading Production Models

```python
from app.monitoring.mlflow_tracker import model_registry

# Load production model
model = model_registry.load_production_model("credit_risk_model")
predictions = model.predict(X_new)
```

## Dashboard

### MLflow Dashboard

The FinRisk MLflow Dashboard (`mlflow_dashboard.py`) provides a comprehensive web interface for:

- **Overview**: Experiment summary and key metrics
- **Experiments**: Browse and filter experiments
- **Model Registry**: Manage registered models
- **Model Comparison**: Compare model performance
- **Data Quality**: View data quality reports
- **Settings**: Configure MLflow connection

### Starting the Dashboard

```bash
streamlit run mlflow_dashboard.py --server.port 8502
```

Access the dashboard at: http://localhost:8502

### MLflow UI

For the standard MLflow UI:

```bash
mlflow ui --port 5000
```

Access the UI at: http://localhost:5000

## Experiment Types

### Credit Risk Experiments

**Tags**: `model_type: credit_risk`

**Metrics Tracked**:
- AUC (Area Under Curve)
- Gini Coefficient
- KS Statistic (Kolmogorov-Smirnov)
- Population Stability Index (PSI)
- Precision, Recall, F1-score
- Accuracy

**Artifacts Logged**:
- Trained model
- Feature importance
- Data distributions
- Feature statistics

### Fraud Detection Experiments

**Tags**: `model_type: fraud_detection`

**Metrics Tracked**:
- AUC
- Precision at K
- Lift at Percentile
- Detection Rate at FPR
- Precision, Recall, F1-score

**Artifacts Logged**:
- Trained model
- Anomaly score distributions
- Data distributions
- Feature statistics

### Hyperparameter Tuning

**Tags**: `experiment_type: hyperparameter_tuning`

**Artifacts Logged**:
- Parameter grid
- Cross-validation results
- Best parameters
- Performance summary

### Data Quality Reports

**Tags**: `experiment_type: data_quality`

**Metrics Tracked**:
- Missing value percentages
- Data type distributions
- Statistical summaries
- Quality scores

## Model Registry Workflow

### 1. Development Stage

```python
# Train and log experiment
run_id = log_credit_risk_model(...)

# Register model (Development stage)
version = model_registry.register_model(
    run_id=run_id,
    model_name="credit_risk_model",
    description="Initial development model"
)
```

### 2. Staging Stage

```python
# Promote to staging
model_registry.promote_model(
    model_name="credit_risk_model",
    version=version,
    stage="Staging"
)
```

### 3. Production Stage

```python
# Promote to production
model_registry.promote_model(
    model_name="credit_risk_model",
    version=version,
    stage="Production"
)
```

## Testing

### Run Integration Tests

```bash
python test_mlflow_integration.py
```

This test suite validates:
- MLflow tracker initialization
- Model registry initialization
- Credit risk experiment logging
- Fraud detection experiment logging
- Model registration
- Experiment summary retrieval
- Model loading

### Expected Output

```
============================================================
MLflow Integration Test Suite
============================================================
✓ MLflow tracker initialized successfully
✓ Model registry initialized successfully
✓ Credit risk experiment logged successfully
✓ Fraud detection experiment logged successfully
✓ Model registered successfully
✓ Experiment summary retrieved successfully
✓ Model loaded successfully
============================================================
Tests passed: 7/7
Success rate: 100.0%
✓ All tests passed! MLflow integration is working correctly.
```

## Best Practices

### 1. Experiment Naming

Use descriptive experiment names:
```python
run_name = f"credit_risk_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

### 2. Tagging

Use consistent tags for filtering:
```python
tags = {
    "model_type": "credit_risk",
    "algorithm": "random_forest",
    "dataset_version": "v2.1",
    "feature_set": "comprehensive"
}
```

### 3. Metrics Logging

Log all relevant metrics:
```python
metrics = {
    'auc': roc_auc_score(y_test, y_prob),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'gini': 2 * roc_auc_score(y_test, y_prob) - 1
}
```

### 4. Model Artifacts

Log important artifacts:
```python
# Feature importance
mlflow.log_dict(feature_importance, "feature_importance.json")

# Data distributions
mlflow.log_dict(data_stats, "data_statistics.json")

# Model configuration
mlflow.log_dict(model_config, "model_config.json")
```

### 5. Version Control

Always version your models:
```python
# Use semantic versioning
model_name = "credit_risk_model_v2_1_0"
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```
   Error: Failed to establish a new connection
   Solution: Check MLFLOW_TRACKING_URI and ensure MLflow server is running
   ```

2. **Experiment Not Found**
   ```
   Error: Experiment 'finrisk-experiments' not found
   Solution: The experiment will be created automatically on first use
   ```

3. **Model Registration Failed**
   ```
   Error: Model already exists
   Solution: Use different model name or version
   ```

4. **Artifact Not Found**
   ```
   Error: No artifacts found at path 'model'
   Solution: Ensure model was logged correctly during training
   ```

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Database Issues

For SQLite issues:
```bash
# Check database file
ls -la mlflow.db

# Reset database (WARNING: This will delete all data)
rm mlflow.db
```

## Integration with Existing Components

### Credit Risk Trainer

The credit risk trainer automatically logs experiments:

```python
# In app/models/credit_risk_trainer.py
run_id = log_credit_risk_model(
    model=model,
    model_name=model_name,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    hyperparameters=model.get_params(),
    metrics=metrics,
    feature_importance=feature_importance,
    model_type="sklearn"
)
```

### Fraud Detection Trainer

The fraud detection trainer automatically logs experiments:

```python
# In app/models/fraud_detection_trainer.py
run_id = log_fraud_detection_model(
    model=model,
    model_name=model_name,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    hyperparameters=model.get_params(),
    metrics=metrics,
    anomaly_scores=y_scores,
    model_type="sklearn"
)
```

### API Integration

Load production models in API endpoints:

```python
# In API routes
from app.monitoring.mlflow_tracker import model_registry

@router.post("/predict")
async def predict(data: PredictionRequest):
    model = model_registry.load_production_model("credit_risk_model")
    prediction = model.predict(data.features)
    return {"prediction": prediction}
```

## Future Enhancements

1. **A/B Testing**: Support for model A/B testing
2. **Model Monitoring**: Drift detection and performance monitoring
3. **Automated Retraining**: Trigger retraining based on performance degradation
4. **Multi-tenancy**: Support for multiple organizations/teams
5. **Advanced Analytics**: Custom dashboards and reporting
6. **Integration with External Tools**: Prometheus, Grafana, etc.

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review MLflow documentation: https://mlflow.org/docs/
3. Check application logs for detailed error messages
4. Verify environment configuration
5. Test with the integration test suite

## References

- [MLflow Documentation](https://mlflow.org/docs/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [FinRisk Project Documentation](./README.md)
