# Credit Risk Modeling Module

This document provides a comprehensive guide to the credit risk modeling capabilities of the FinRisk application.

## Overview

The credit risk modeling module provides a complete pipeline for training, evaluating, and deploying credit risk models. It includes:

- **Multiple Algorithm Support**: XGBoost, Random Forest, and Logistic Regression
- **Comprehensive Feature Engineering**: Customer, credit bureau, and application features
- **Advanced Evaluation Metrics**: AUC, Gini coefficient, KS statistic, PSI
- **Model Interpretability**: SHAP and LIME explanations
- **MLflow Integration**: Experiment tracking and model versioning
- **Production-Ready**: Model serialization and deployment support

## Architecture

```
finrisk/
├── app/
│   ├── features/
│   │   └── preprocessing.py          # Feature engineering and transformation
│   ├── models/
│   │   └── credit_risk_trainer.py    # Main training module
│   ├── monitoring/
│   │   └── mlflow_utils.py           # MLflow integration
│   └── schemas/
│       └── base.py                   # Pydantic schemas
├── scripts/
│   └── train_credit_models.py        # Training script
└── data/
    └── models/                       # Model storage
```

## Key Components

### 1. Feature Engineering (`app/features/preprocessing.py`)

The `FinancialFeatureEngineer` class creates comprehensive credit risk features:

#### Customer Features
- Age groups and risk indicators
- Income categories and employment duration
- Location-based features
- Credit utilization ratios

#### Credit Bureau Features
- Credit score categories
- Payment history analysis
- Account diversity metrics
- Inquiry intensity

#### Application Features
- Application frequency and history
- Loan amount trends
- Default rate calculations
- Purpose diversity

#### Derived Features
- Risk indicators (high-risk age, low income, etc.)
- Credit risk flags
- Interaction features

### 2. Model Training (`app/models/credit_risk_trainer.py`)

The `CreditRiskModelTrainer` class provides:

#### Supported Algorithms
- **Logistic Regression**: Linear model with regularization
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting with advanced features

#### Training Features
- Hyperparameter tuning with GridSearchCV
- Probability calibration
- Cross-validation
- Model comparison and selection

#### Evaluation Metrics
- **AUC**: Area Under ROC Curve
- **Gini Coefficient**: 2 × AUC - 1
- **KS Statistic**: Kolmogorov-Smirnov statistic
- **PSI**: Population Stability Index
- **Standard Metrics**: Precision, Recall, F1, Accuracy

### 3. Model Interpretability (`CreditModelExplainer`)

Provides explanations for model predictions:

#### SHAP (SHapley Additive exPlanations)
- Feature importance for individual predictions
- Global feature importance
- Interaction effects

#### LIME (Local Interpretable Model-agnostic Explanations)
- Local explanations for specific instances
- Feature contribution analysis

#### Text Explanations
- Human-readable risk assessments
- Top contributing factors
- Risk level classifications

### 4. MLflow Integration (`app/monitoring/mlflow_utils.py`)

Comprehensive experiment tracking:

#### Experiment Management
- Create and manage experiments
- Track model versions
- Compare model performance

#### Artifact Logging
- Model files and metadata
- Feature importance plots
- Evaluation metrics and plots
- Training parameters

#### Visualization
- ROC curves
- Precision-recall curves
- Confusion matrices
- Feature importance charts

## Usage

### Basic Training

```python
from app.models.credit_risk_trainer import train_credit_models

# Train all models with hyperparameter tuning
results = train_credit_models(hyperparameter_tuning=True)

# Access results
print(results['comparison'])  # Model performance comparison
print(results['models'])      # Trained models
```

### Advanced Training

```python
from app.models.credit_risk_trainer import CreditRiskModelTrainer

# Initialize trainer
trainer = CreditRiskModelTrainer()

# Train specific model
model_result = trainer.train_model(
    model_name='xgboost',
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    hyperparameter_tuning=True
)

# Access model and metrics
model = model_result['model']
metrics = model_result['metrics']
```

### Model Explanations

```python
from app.models.credit_risk_trainer import CreditModelExplainer

# Initialize explainer
explainer = CreditModelExplainer(model, feature_names)

# Explain single prediction
explanation = explainer.explain_prediction(X_instance)
print(explanation['text_explanation'])

# Global feature importance
global_explanation = explainer.generate_global_explanation(X_sample)
```

### Command Line Training

```bash
# Basic training
python scripts/train_credit_models.py

# Skip hyperparameter tuning
python scripts/train_credit_models.py --no-tuning

# Custom test size
python scripts/train_credit_models.py --test-size 0.3

# Include explanations
python scripts/train_credit_models.py --explain
```

## Configuration

### Environment Variables

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=finrisk_db
DB_USER=finrisk_user
DB_PASSWORD=finrisk_pass

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlruns

# Model Storage
MODEL_STORE_PATH=./data/models
```

### Model Configuration

Models can be configured in the `CreditRiskModelTrainer` class:

```python
self.model_configs = {
    'logistic_regression': {
        'model': LogisticRegression,
        'params': {
            'random_state': 42,
            'max_iter': 1000,
            'class_weight': 'balanced'
        },
        'search_params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    }
    # ... other models
}
```

## Data Requirements

### Database Tables

The module expects the following tables:

#### `finrisk.customer_profiles`
- `customer_id` (Primary Key)
- `customer_age`
- `annual_income`
- `employment_status`
- `credit_score`
- `city`
- `last_activity_date`

#### `finrisk.credit_bureau_data`
- `customer_id` (Foreign Key)
- `credit_score`
- `credit_history_length`
- `number_of_accounts`
- `total_credit_limit`
- `credit_utilization`
- `payment_history`

#### `finrisk.credit_applications`
- `customer_id` (Foreign Key)
- `application_id`
- `loan_amount`
- `loan_purpose`
- `application_date`
- `application_status`
- `default_flag`

## Model Evaluation

### Performance Metrics

1. **AUC (Area Under Curve)**
   - Range: 0.5 to 1.0
   - Higher is better
   - Industry standard: > 0.7

2. **Gini Coefficient**
   - Range: 0 to 1
   - Higher is better
   - Industry standard: > 0.4

3. **KS Statistic**
   - Range: 0 to 1
   - Higher is better
   - Industry standard: > 0.3

4. **Population Stability Index (PSI)**
   - < 0.1: Stable
   - 0.1-0.25: Some drift
   - > 0.25: Significant drift

### Model Selection Criteria

1. **Primary**: AUC score
2. **Secondary**: Gini coefficient
3. **Tertiary**: KS statistic
4. **Operational**: Training time and interpretability

## Production Deployment

### Model Serialization

Models are automatically saved with metadata:

```python
# Model file
credit_xgboost_20231201_143022.joblib

# Metadata file
credit_xgboost_20231201_143022_metadata.json
```

### Model Loading

```python
import joblib

# Load model
model = joblib.load('credit_xgboost_20231201_143022.joblib')

# Load transformer
transformer = joblib.load('credit_feature_transformer_20231201_143022.joblib')

# Make predictions
X_transformed = transformer.transform(X_new)
predictions = model.predict_proba(X_transformed)
```

### API Integration

```python
from fastapi import FastAPI
from app.models.credit_risk_trainer import CreditModelExplainer

app = FastAPI()

@app.post("/predict")
async def predict_credit_risk(customer_data: dict):
    # Transform features
    X_transformed = transformer.transform(customer_data)
    
    # Make prediction
    prediction = model.predict_proba(X_transformed)
    
    # Generate explanation
    explanation = explainer.explain_prediction(X_transformed)
    
    return {
        "prediction": float(prediction[0][1]),
        "explanation": explanation
    }
```

## Monitoring and Maintenance

### Model Monitoring

1. **Performance Drift**: Monitor PSI and KS statistics
2. **Data Quality**: Check for missing values and outliers
3. **Feature Drift**: Monitor feature distributions
4. **Business Metrics**: Track approval rates and default rates

### Retraining Schedule

- **Monthly**: Retrain with new data
- **Quarterly**: Full hyperparameter tuning
- **Annually**: Feature engineering review

### Model Versioning

- Use MLflow for experiment tracking
- Tag production models
- Maintain model lineage
- Document model changes

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce batch size
   - Use feature selection
   - Optimize data types

2. **Training Time**
   - Use smaller parameter grids
   - Reduce cross-validation folds
   - Use parallel processing

3. **Poor Performance**
   - Check data quality
   - Review feature engineering
   - Try different algorithms

### Debugging

```python
# Enable debug logging
import logging
logging.getLogger('app.models.credit_risk_trainer').setLevel(logging.DEBUG)

# Check data quality
print(X.isnull().sum())
print(y.value_counts())

# Validate features
print(X.describe())
```

## Best Practices

1. **Data Quality**
   - Validate input data
   - Handle missing values appropriately
   - Check for data leakage

2. **Feature Engineering**
   - Create domain-specific features
   - Avoid over-engineering
   - Monitor feature importance

3. **Model Selection**
   - Use cross-validation
   - Compare multiple algorithms
   - Consider business constraints

4. **Interpretability**
   - Provide explanations for decisions
   - Monitor feature drift
   - Document model behavior

5. **Production**
   - Version control models
   - Monitor performance
   - Plan for retraining

## Contributing

When contributing to the credit risk modeling module:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Validate with real data
5. Consider performance implications

## License

This module is part of the FinRisk application and follows the same licensing terms.
