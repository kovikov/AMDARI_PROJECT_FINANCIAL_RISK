# FinRisk Schema Guide

This guide documents all Pydantic schemas used in the FinRisk financial risk management system.

## Overview

The FinRisk application uses Pydantic schemas for:
- Data validation and serialization
- API request/response models
- Database model definitions
- Configuration management
- Type safety throughout the application

## Schema Structure

### Base Schemas (`app/schemas/base.py`)

#### Enums
- **RiskSegment**: `PRIME`, `NEAR_PRIME`, `SUB_PRIME`, `DEEP_SUB_PRIME`
- **EmploymentStatus**: `FULL_TIME`, `PART_TIME`, `SELF_EMPLOYED`, `UNEMPLOYED`, `RETIRED`
- **ApplicationStatus**: `PENDING`, `APPROVED`, `REJECTED`, `UNDER_REVIEW`, `CANCELLED`
- **LoanPurpose**: `PERSONAL`, `HOME_IMPROVEMENT`, `DEBT_CONSOLIDATION`, `BUSINESS`, `EDUCATION`
- **MerchantCategory**: `RETAIL`, `FOOD_AND_BEVERAGE`, `TRAVEL`, `HEALTHCARE`, `ENTERTAINMENT`
- **TransactionType**: `PURCHASE`, `WITHDRAWAL`, `TRANSFER`, `PAYMENT`, `REFUND`
- **ModelType**: `CREDIT_RISK`, `FRAUD_DETECTION`, `CHURN_PREDICTION`, `CUSTOMER_LIFETIME_VALUE`
- **AlertSeverity**: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`

#### Base Classes
- **BaseSchema**: Base Pydantic model with configuration
- **TimestampMixin**: Adds `created_at` and `updated_at` fields
- **HealthCheckResponse**: API health check response
- **ErrorResponse**: Standard error response format
- **PaginationParams**: Pagination parameters
- **PaginatedResponse**: Paginated response wrapper

### Customer Schemas (`app/schemas/customers.py`)

#### Core Customer Models
- **CustomerCreate**: Create new customer
- **CustomerUpdate**: Update existing customer
- **CustomerResponse**: Customer data response
- **CustomerSummary**: Customer summary for lists
- **CustomerRiskProfile**: Customer risk assessment
- **CustomerSearchParams**: Customer search parameters
- **CustomerAnalytics**: Customer analytics data
- **CustomerPortfolio**: Customer portfolio summary

### Application Schemas (`app/schemas/applications.py`)

#### Credit Application Models
- **CreditApplicationCreate**: Create new credit application
- **CreditApplicationUpdate**: Update application
- **CreditApplicationResponse**: Application response
- **CreditApplicationSummary**: Application summary
- **CreditDecision**: Credit decision details
- **CreditApplicationSearchParams**: Search parameters
- **CreditApplicationAnalytics**: Application analytics
- **CreditApplicationPortfolio**: Application portfolio

### Prediction Schemas (`app/schemas/predictions.py`)

#### Model Prediction Models
- **CreditRiskPrediction**: Credit risk prediction
- **FraudDetectionPrediction**: Fraud detection prediction
- **ModelPredictionCreate**: Create prediction record
- **ModelPredictionResponse**: Prediction response
- **ModelPredictionSummary**: Prediction summary
- **ModelPerformanceMetrics**: Model performance data
- **ModelDriftAlert**: Model drift alert
- **PredictionSearchParams**: Prediction search
- **ModelExplanationRequest**: Model explanation request
- **ModelExplanationResponse**: Model explanation response

## Usage Examples

### Creating a Customer

```python
from app.schemas.customers import CustomerCreate
from app.schemas.base import RiskSegment, EmploymentStatus
from datetime import date

customer = CustomerCreate(
    customer_id="CUST001",
    customer_age=35,
    annual_income=75000.0,
    employment_status=EmploymentStatus.FULL_TIME,
    account_tenure=5,
    product_holdings=3,
    relationship_value=150000.0,
    risk_segment=RiskSegment.PRIME,
    behavioral_score=850.0,
    credit_score=720,
    city="New York",
    last_activity_date=date.today()
)
```

### Creating a Credit Application

```python
from app.schemas.applications import CreditApplicationCreate
from app.schemas.base import LoanPurpose

application = CreditApplicationCreate(
    application_id="APP001",
    customer_id="CUST001",
    loan_amount=25000.0,
    loan_purpose=LoanPurpose.DEBT_CONSOLIDATION,
    loan_term_months=36,
    interest_rate=8.5,
    monthly_payment=789.50,
    debt_to_income_ratio=0.35,
    collateral_value=30000.0,
    application_score=750.0
)
```

### Creating a Prediction

```python
from app.schemas.predictions import CreditRiskPrediction
from app.schemas.base import ModelType

prediction = CreditRiskPrediction(
    prediction_id="PRED001",
    customer_id="CUST001",
    application_id="APP001",
    model_type=ModelType.CREDIT_RISK,
    model_version="v1.0.0",
    prediction_score=0.85,
    risk_probability=0.15,
    confidence_score=0.92,
    prediction_timestamp=datetime.now(timezone.utc),
    features_used=["age", "income", "credit_score", "dti_ratio"],
    model_explanation={"feature_importance": {"credit_score": 0.4, "income": 0.3}}
)
```

## Validation Rules

### Customer Validation
- `customer_age`: 18-100
- `annual_income`: > 0
- `credit_score`: 300-850
- `behavioral_score`: 0-1000

### Application Validation
- `loan_amount`: > 0
- `loan_term_months`: 12-360
- `interest_rate`: 0-50
- `debt_to_income_ratio`: 0-1

### Prediction Validation
- `prediction_score`: 0-1
- `risk_probability`: 0-1
- `confidence_score`: 0-1

## API Integration

### FastAPI Usage

```python
from fastapi import FastAPI
from app.schemas.customers import CustomerCreate, CustomerResponse

app = FastAPI()

@app.post("/customers/", response_model=CustomerResponse)
async def create_customer(customer: CustomerCreate):
    # Process customer creation
    return CustomerResponse(**customer.dict(), id=1)
```

### Database Integration

```python
from sqlalchemy.orm import Session
from app.schemas.customers import CustomerCreate

def create_customer_in_db(db: Session, customer: CustomerCreate):
    # Convert Pydantic model to dict
    customer_data = customer.dict()
    
    # Create database record
    db_customer = Customer(**customer_data)
    db.add(db_customer)
    db.commit()
    db.refresh(db_customer)
    
    return db_customer
```

## Testing Schemas

Run the schema tests:

```bash
# Test all schemas
python test_schemas.py

# Test specific schema
python -c "
from app.schemas.customers import CustomerCreate
from app.schemas.base import RiskSegment, EmploymentStatus
from datetime import date

customer = CustomerCreate(
    customer_id='TEST001',
    customer_age=30,
    annual_income=60000.0,
    employment_status=EmploymentStatus.FULL_TIME,
    account_tenure=3,
    product_holdings=2,
    relationship_value=100000.0,
    risk_segment=RiskSegment.PRIME,
    behavioral_score=800.0,
    credit_score=700,
    city='London',
    last_activity_date=date.today()
)
print('Schema validation successful!')
"
```

## Best Practices

1. **Always validate input data** using Pydantic schemas
2. **Use type hints** for better code documentation
3. **Leverage enums** for fixed choice fields
4. **Implement custom validators** for complex business rules
5. **Use mixins** for common field patterns
6. **Document schemas** with field descriptions
7. **Test schema validation** thoroughly

## Error Handling

Pydantic provides detailed validation errors:

```python
from pydantic import ValidationError
from app.schemas.customers import CustomerCreate

try:
    customer = CustomerCreate(
        customer_id="CUST001",
        customer_age=150,  # Invalid age
        # ... other fields
    )
except ValidationError as e:
    print(f"Validation error: {e}")
    # Handle validation errors
```

## Migration and Versioning

When updating schemas:

1. **Maintain backward compatibility** when possible
2. **Use optional fields** for new additions
3. **Version your schemas** for breaking changes
4. **Update tests** to reflect changes
5. **Document changes** in release notes

## Performance Considerations

- **Use `Config.validate_assignment = False`** for read-only models
- **Implement `__slots__`** for high-performance models
- **Use `parse_obj_as`** for bulk validation
- **Cache validated models** when appropriate

## Security

- **Sanitize input data** before validation
- **Use `SecretStr`** for sensitive fields
- **Validate file uploads** with appropriate schemas
- **Implement rate limiting** for API endpoints

## Troubleshooting

### Common Issues

1. **Validation errors**: Check field types and constraints
2. **Import errors**: Ensure all dependencies are installed
3. **Circular imports**: Use forward references or lazy imports
4. **Performance issues**: Optimize validation rules

### Debug Tips

```python
# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Print model structure
print(CustomerCreate.schema())

# Validate with detailed errors
try:
    customer = CustomerCreate(**data)
except ValidationError as e:
    print(e.json())
```

## Related Documentation

- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Database Guide](./DATABASE_GUIDE.md)
- [Cache Guide](./CACHE_GUIDE.md)
- [API Documentation](./API_GUIDE.md)


