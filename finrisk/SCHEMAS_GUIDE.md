# FinRisk Schemas Guide

This guide covers all Pydantic schemas used in the FinRisk financial risk management system.

## 🏗️ Schema Architecture

The FinRisk system uses a modular schema architecture with:

- **Base Schemas**: Common enums, base classes, and shared schemas
- **Domain Schemas**: Specific schemas for each business domain
- **Validation**: Comprehensive field validation and business rules
- **Serialization**: Automatic JSON serialization with proper date handling

## 📋 Schema Categories

### 1. Base Schemas (`app/schemas/base.py`)

#### **Enums**
```python
from app.schemas import RiskSegment, EmploymentStatus, ApplicationStatus

# Risk segments
RiskSegment.PRIME           # "Prime"
RiskSegment.NEAR_PRIME      # "Near-Prime"
RiskSegment.SUBPRIME        # "Subprime"
RiskSegment.DEEP_SUBPRIME   # "Deep-Subprime"

# Employment status
EmploymentStatus.FULL_TIME      # "Full-time"
EmploymentStatus.PART_TIME      # "Part-time"
EmploymentStatus.SELF_EMPLOYED  # "Self-employed"
EmploymentStatus.UNEMPLOYED     # "Unemployed"
EmploymentStatus.RETIRED        # "Retired"
EmploymentStatus.STUDENT        # "Student"
EmploymentStatus.CONTRACT       # "Contract"

# Application status
ApplicationStatus.APPROVED      # "Approved"
ApplicationStatus.DECLINED      # "Declined"
ApplicationStatus.PENDING       # "Pending"

# Loan purposes
LoanPurpose.HOME_PURCHASE       # "Home Purchase"
LoanPurpose.DEBT_CONSOLIDATION  # "Debt Consolidation"
LoanPurpose.HOME_IMPROVEMENT    # "Home Improvement"
LoanPurpose.CAR_PURCHASE        # "Car Purchase"
LoanPurpose.BUSINESS            # "Business"
LoanPurpose.EDUCATION           # "Education"
LoanPurpose.PERSONAL            # "Personal"

# Model types
ModelType.CREDIT_RISK           # "Credit Risk"
ModelType.FRAUD_DETECTION       # "Fraud Detection"

# Alert severity
AlertSeverity.LOW               # "LOW"
AlertSeverity.MEDIUM            # "MEDIUM"
AlertSeverity.HIGH              # "HIGH"
AlertSeverity.CRITICAL          # "CRITICAL"
```

#### **Base Classes**
```python
from app.schemas import BaseSchema, TimestampMixin

# BaseSchema - Common configuration for all schemas
class MySchema(BaseSchema):
    name: str = Field(..., description="Name field")
    
# TimestampMixin - Adds created_at and updated_at fields
class MyTimestampedSchema(BaseSchema, TimestampMixin):
    name: str = Field(..., description="Name field")
    # Automatically includes created_at and updated_at
```

#### **Common Schemas**
```python
from app.schemas import (
    HealthCheckResponse, ErrorResponse, 
    PaginationParams, PaginatedResponse,
    FilterParams
)

# Health check response
health_response = HealthCheckResponse(
    status="healthy",
    checks={"database": "connected", "redis": "connected"}
)

# Error response
error_response = ErrorResponse(
    error="Validation failed",
    detail="Invalid customer ID format",
    request_id="req_123456"
)

# Pagination parameters
pagination = PaginationParams(page=2, size=50)
print(f"Offset: {pagination.offset}")  # 50

# Paginated response
paginated = PaginatedResponse(
    items=[{"id": 1, "name": "Test"}],
    total=100,
    page=1,
    size=10
)
print(f"Pages: {paginated.pages}")  # 10 (calculated automatically)

# Filter parameters
filters = FilterParams(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
    customer_ids=["CUST001", "CUST002"],
    risk_segments=[RiskSegment.PRIME, RiskSegment.NEAR_PRIME]
)
```

### 2. Customer Schemas (`app/schemas/customers.py`)

#### **Customer Creation and Updates**
```python
from app.schemas import CustomerCreate, CustomerUpdate

# Create new customer
customer_create = CustomerCreate(
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
    city="London",
    last_activity_date=date.today()
)

# Update customer (partial)
customer_update = CustomerUpdate(
    annual_income=80000.0,
    credit_score=750
)
```

#### **Customer Responses**
```python
from app.schemas import CustomerResponse, CustomerSummary, CustomerRiskProfile

# Full customer response
customer_response = CustomerResponse(
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
    city="London",
    last_activity_date=date.today(),
    created_at=datetime.now(timezone.utc),
    updated_at=datetime.now(timezone.utc)
)

# Customer summary (for lists)
customer_summary = CustomerSummary(
    customer_id="CUST001",
    customer_age=35,
    annual_income=75000.0,
    employment_status=EmploymentStatus.FULL_TIME,
    risk_segment=RiskSegment.PRIME,
    credit_score=720,
    city="London",
    last_activity_date=date.today()
)

# Customer risk profile
risk_profile = CustomerRiskProfile(
    customer_id="CUST001",
    risk_segment=RiskSegment.PRIME,
    credit_score=720,
    behavioral_score=850.0,
    risk_factors=["high_income", "stable_employment"],
    risk_score=0.15,
    last_assessment_date=datetime.now(timezone.utc)
)
```

#### **Customer Search and Analytics**
```python
from app.schemas import CustomerSearchParams, CustomerAnalytics, CustomerPortfolio

# Search parameters
search_params = CustomerSearchParams(
    risk_segment=RiskSegment.PRIME,
    min_age=25,
    max_age=50,
    min_income=50000,
    max_income=100000,
    city="London"
)

# Customer analytics
analytics = CustomerAnalytics(
    customer_id="CUST001",
    total_transactions=150,
    total_spend=25000.0,
    avg_transaction_value=166.67,
    transaction_frequency=12.5,
    risk_trend="decreasing",
    segment_movement="Prime to Super-Prime",
    last_analysis_date=datetime.now(timezone.utc)
)

# Customer portfolio
portfolio = CustomerPortfolio(
    customer_id="CUST001",
    total_balance=50000.0,
    total_credit_limit=100000.0,
    credit_utilization=0.5,
    number_of_accounts=3,
    account_types=["checking", "savings", "credit_card"],
    portfolio_value=150000.0,
    last_portfolio_date=datetime.now(timezone.utc)
)
```

### 3. Credit Application Schemas (`app/schemas/applications.py`)

#### **Application Creation and Updates**
```python
from app.schemas import CreditApplicationCreate, CreditApplicationUpdate

# Create new application
application_create = CreditApplicationCreate(
    customer_id="CUST001",
    loan_amount=50000.0,
    loan_purpose=LoanPurpose.HOME_PURCHASE,
    loan_term=240,
    interest_rate=3.5,
    monthly_payment=2500.0,
    debt_to_income_ratio=0.35,
    application_source="online"
)

# Update application
application_update = CreditApplicationUpdate(
    loan_amount=55000.0,
    application_status=ApplicationStatus.PENDING
)
```

#### **Application Responses**
```python
from app.schemas import CreditApplicationResponse, CreditApplicationSummary, CreditDecision

# Full application response
application_response = CreditApplicationResponse(
    application_id="APP001",
    customer_id="CUST001",
    loan_amount=50000.0,
    loan_purpose=LoanPurpose.HOME_PURCHASE,
    loan_term=240,
    interest_rate=3.5,
    monthly_payment=2500.0,
    debt_to_income_ratio=0.35,
    application_status=ApplicationStatus.APPROVED,
    application_date=date.today(),
    decision_date=datetime.now(timezone.utc),
    application_source="online",
    created_at=datetime.now(timezone.utc),
    updated_at=datetime.now(timezone.utc)
)

# Application summary
application_summary = CreditApplicationSummary(
    application_id="APP001",
    customer_id="CUST001",
    loan_amount=50000.0,
    loan_purpose=LoanPurpose.HOME_PURCHASE,
    application_status=ApplicationStatus.APPROVED,
    application_date=date.today(),
    decision_date=datetime.now(timezone.utc)
)

# Credit decision
credit_decision = CreditDecision(
    application_id="APP001",
    decision=ApplicationStatus.APPROVED,
    decision_reason="Strong credit profile and stable income",
    risk_score=0.15,
    credit_score=720,
    debt_to_income_ratio=0.35,
    loan_to_value_ratio=0.8,
    decision_factors=["high_credit_score", "stable_employment", "low_dti"],
    decision_date=datetime.now(timezone.utc),
    decision_officer="john.doe@bank.com"
)
```

### 4. Model Prediction Schemas (`app/schemas/predictions.py`)

#### **Credit Risk Predictions**
```python
from app.schemas import CreditRiskPrediction, FeatureImportance

# Feature importance
feature_importance = [
    FeatureImportance(feature_name="credit_score", importance=0.45, rank=1),
    FeatureImportance(feature_name="income", importance=0.30, rank=2),
    FeatureImportance(feature_name="debt_to_income", importance=0.25, rank=3)
]

# Credit risk prediction
credit_prediction = CreditRiskPrediction(
    customer_id="CUST001",
    model_version="v1.2.0",
    risk_score=0.25,
    default_probability=0.15,
    credit_limit_recommendation=75000.0,
    interest_rate_recommendation=3.2,
    risk_segment="Prime",
    decision="approve",
    confidence_score=0.92,
    model_features={
        "age": 35,
        "income": 75000,
        "credit_score": 720,
        "employment_years": 8
    },
    feature_importance=feature_importance,
    explanation="Customer has strong credit profile with stable income"
)
```

#### **Fraud Detection Predictions**
```python
from app.schemas import FraudDetectionPrediction

# Fraud detection prediction
fraud_prediction = FraudDetectionPrediction(
    customer_id="CUST001",
    transaction_id="TXN123",
    model_version="v1.1.0",
    fraud_probability=0.05,
    risk_level="low",
    fraud_score=0.12,
    decision="allow",
    confidence_score=0.88,
    model_features={
        "amount": 150.0,
        "location": "London",
        "time": "14:30",
        "merchant_category": "restaurants"
    },
    feature_importance=[
        FeatureImportance(feature_name="amount", importance=0.60, rank=1),
        FeatureImportance(feature_name="location", importance=0.25, rank=2)
    ],
    explanation="Transaction appears normal based on customer's spending patterns",
    alert_triggered=False
)
```

#### **Model Performance and Drift**
```python
from app.schemas import ModelPerformanceMetrics, ModelDriftAlert

# Model performance metrics
performance = ModelPerformanceMetrics(
    model_name="credit_risk_v1",
    model_version="v1.2.0",
    prediction_type=ModelType.CREDIT_RISK,
    accuracy=0.92,
    precision=0.89,
    recall=0.94,
    f1_score=0.91,
    auc_score=0.95,
    confusion_matrix={
        "true_positive": 850,
        "true_negative": 920,
        "false_positive": 80,
        "false_negative": 50
    },
    total_predictions=1900,
    evaluation_date=datetime.now(timezone.utc)
)

# Model drift alert
drift_alert = ModelDriftAlert(
    model_name="credit_risk_v1",
    model_version="v1.2.0",
    drift_metric="PSI",
    drift_score=0.35,
    threshold_value=0.25,
    is_drifted=True,
    affected_features=["income", "credit_score"],
    alert_severity="HIGH",
    alert_message="Significant data drift detected in income and credit_score features",
    detection_date=datetime.now(timezone.utc)
)
```

## 🔧 Schema Usage Patterns

### 1. API Request/Response Pattern
```python
from fastapi import APIRouter
from app.schemas import CustomerCreate, CustomerResponse, PaginatedResponse

router = APIRouter()

@router.post("/customers", response_model=CustomerResponse)
async def create_customer(customer: CustomerCreate):
    # Process customer creation
    return CustomerResponse(**customer_data)

@router.get("/customers", response_model=PaginatedResponse[CustomerResponse])
async def list_customers(page: int = 1, size: int = 100):
    # Process customer listing
    return PaginatedResponse(
        items=customers,
        total=total_count,
        page=page,
        size=size
    )
```

### 2. Validation Pattern
```python
from app.schemas import CustomerSearchParams

# Automatic validation
try:
    search_params = CustomerSearchParams(
        min_age=25,
        max_age=20  # This will raise a validation error
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

### 3. Serialization Pattern
```python
from app.schemas import CustomerResponse

# Automatic JSON serialization
customer = CustomerResponse(...)
customer_json = customer.model_dump_json()

# With custom serialization
customer_dict = customer.model_dump(
    exclude={"created_at", "updated_at"},
    by_alias=True
)
```

### 4. Database Integration Pattern
```python
from app.schemas import CustomerCreate
from app.infra.db import get_db_session

async def create_customer_in_db(customer_data: dict):
    # Validate input data
    customer = CustomerCreate(**customer_data)
    
    # Convert to database model
    db_customer = CustomerModel(
        customer_id=customer.customer_id,
        customer_age=customer.customer_age,
        # ... other fields
    )
    
    # Save to database
    with get_db_session() as session:
        session.add(db_customer)
        session.commit()
    
    return customer
```

## 📊 Schema Validation Rules

### Field Validation
```python
# Age validation (18-120)
customer_age: int = Field(..., ge=18, le=120)

# Income validation (non-negative)
annual_income: float = Field(..., ge=0)

# Credit score validation (300-850)
credit_score: int = Field(..., ge=300, le=850)

# Ratio validation (0-1)
debt_to_income_ratio: float = Field(..., ge=0, le=1)

# Optional fields
collateral_value: Optional[float] = Field(None, ge=0)
```

### Custom Validators
```python
from pydantic import validator

class FilterParams(BaseModel):
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    
    @validator('end_date')
    def end_date_after_start_date(cls, v, values):
        start_date = values.get('start_date')
        if start_date and v and v < start_date:
            raise ValueError('end_date must be after start_date')
        return v
```

### Enum Validation
```python
# Automatic enum validation
employment_status: EmploymentStatus = Field(...)

# This will raise validation error for invalid values
# employment_status: "Invalid"  # ❌ ValidationError
```

## 🚨 Error Handling

### Validation Errors
```python
from pydantic import ValidationError
from app.schemas import CustomerCreate

try:
    customer = CustomerCreate(
        customer_id="CUST001",
        customer_age=15,  # Invalid age
        # ... other fields
    )
except ValidationError as e:
    print(f"Validation errors: {e.errors()}")
    # [
    #   {
    #     'type': 'value_error.number.not_ge',
    #     'loc': ('customer_age',),
    #     'msg': 'ensure this value is greater than or equal to 18',
    #     'input': 15
    #   }
    # ]
```

### Business Logic Errors
```python
from app.schemas import ErrorResponse

def handle_business_error(error_message: str, detail: str = None):
    return ErrorResponse(
        error=error_message,
        detail=detail,
        request_id=generate_request_id()
    )
```

## 🔍 Testing Schemas

### Unit Testing
```python
import pytest
from app.schemas import CustomerCreate, RiskSegment

def test_customer_create_valid():
    customer = CustomerCreate(
        customer_id="CUST001",
        customer_age=35,
        annual_income=75000.0,
        employment_status=EmploymentStatus.FULL_TIME,
        # ... other required fields
    )
    assert customer.customer_id == "CUST001"
    assert customer.risk_segment == RiskSegment.PRIME

def test_customer_create_invalid_age():
    with pytest.raises(ValidationError):
        CustomerCreate(
            customer_id="CUST001",
            customer_age=15,  # Invalid age
            # ... other fields
        )
```

### Integration Testing
```python
def test_customer_api_integration():
    customer_data = {
        "customer_id": "CUST001",
        "customer_age": 35,
        # ... other fields
    }
    
    response = client.post("/customers", json=customer_data)
    assert response.status_code == 200
    
    customer = CustomerResponse(**response.json())
    assert customer.customer_id == "CUST001"
```

## 📈 Best Practices

### 1. Schema Design
- Use descriptive field names
- Provide comprehensive field descriptions
- Use appropriate field types and validators
- Keep schemas focused and single-purpose

### 2. Validation
- Validate at the schema level, not in business logic
- Use custom validators for complex business rules
- Provide clear error messages
- Handle validation errors gracefully

### 3. Performance
- Use `exclude` and `include` for partial serialization
- Cache schema instances when possible
- Use `model_dump()` instead of `dict()` for better performance

### 4. Documentation
- Keep field descriptions up to date
- Document complex validation rules
- Provide usage examples
- Maintain schema versioning

## 🔗 Related Files

- `app/schemas/base.py` - Base schemas and enums
- `app/schemas/customers.py` - Customer-related schemas
- `app/schemas/applications.py` - Credit application schemas
- `app/schemas/predictions.py` - Model prediction schemas
- `app/schemas/__init__.py` - Schema package exports
- `test_schemas.py` - Schema testing script

## 📚 Additional Resources

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Pydantic V2 Migration Guide](https://docs.pydantic.dev/latest/migration/)
- [FastAPI with Pydantic](https://fastapi.tiangolo.com/tutorial/body/)
- [Data Validation with Pydantic](https://docs.pydantic.dev/latest/concepts/validators/)
