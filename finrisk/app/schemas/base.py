"""
Common Pydantic schemas and base classes for FinRisk application.
"""

from datetime import datetime, date
from typing import Optional, Dict, Any, List
from enum import Enum

from pydantic import BaseModel, Field, validator


class RiskSegment(str, Enum):
    """Customer risk segment classifications."""
    PRIME = "Prime"
    NEAR_PRIME = "Near-Prime"
    SUBPRIME = "Subprime"
    DEEP_SUBPRIME = "Deep-Subprime"


class EmploymentStatus(str, Enum):
    """Employment status classifications."""
    FULL_TIME = "Full-time"
    PART_TIME = "Part-time"
    SELF_EMPLOYED = "Self-employed"
    UNEMPLOYED = "Unemployed"
    RETIRED = "Retired"
    STUDENT = "Student"
    CONTRACT = "Contract"


class ApplicationStatus(str, Enum):
    """Credit application status."""
    APPROVED = "Approved"
    DECLINED = "Declined"
    PENDING = "Pending"


class LoanPurpose(str, Enum):
    """Loan purpose classifications."""
    HOME_PURCHASE = "Home Purchase"
    DEBT_CONSOLIDATION = "Debt Consolidation"
    HOME_IMPROVEMENT = "Home Improvement"
    CAR_PURCHASE = "Car Purchase"
    BUSINESS = "Business"
    EDUCATION = "Education"
    PERSONAL = "Personal"


class MerchantCategory(str, Enum):
    """Transaction merchant categories."""
    GROCERIES = "Groceries"
    FUEL = "Fuel"
    RESTAURANTS = "Restaurants"
    ONLINE_SHOPPING = "Online Shopping"
    ENTERTAINMENT = "Entertainment"
    HEALTHCARE = "Healthcare"
    TRAVEL = "Travel"
    UTILITIES = "Utilities"
    ATM = "ATM"
    TRANSFER = "Transfer"


class TransactionType(str, Enum):
    """Transaction type classifications."""
    PURCHASE = "Purchase"
    ATM_WITHDRAWAL = "ATM Withdrawal"
    TRANSFER = "Transfer"


class ModelType(str, Enum):
    """ML model type classifications."""
    CREDIT_RISK = "Credit Risk"
    FRAUD_DETECTION = "Fraud Detection"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        allow_population_by_field_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)


class CustomerProfile(BaseSchema):
    """Customer profile schema."""
    customer_id: str = Field(..., description="Unique customer identifier")
    customer_age: int = Field(..., ge=18, le=120, description="Customer age")
    annual_income: float = Field(..., ge=0, description="Annual income in GBP")
    employment_status: EmploymentStatus = Field(..., description="Employment status")
    account_tenure: int = Field(..., ge=0, description="Account tenure in years")
    product_holdings: int = Field(..., ge=0, description="Number of products held")
    relationship_value: float = Field(..., description="Customer relationship value")
    risk_segment: RiskSegment = Field(..., description="Risk segment classification")
    behavioral_score: float = Field(..., ge=0, le=1000, description="Behavioral score")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    city: str = Field(..., description="Customer city")
    last_activity_date: date = Field(..., description="Last activity date")


class CreditBureauData(BaseSchema):
    """Credit bureau data schema."""
    customer_id: str = Field(..., description="Customer identifier")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    credit_history_length: int = Field(..., ge=0, description="Credit history length in years")
    number_of_accounts: int = Field(..., ge=0, description="Number of credit accounts")
    total_credit_limit: int = Field(..., ge=0, description="Total credit limit")
    credit_utilization: float = Field(..., ge=0, le=1, description="Credit utilization ratio")
    payment_history: float = Field(..., ge=0, le=1, description="Payment history score")
    public_records: int = Field(..., ge=0, description="Number of public records")


class HealthCheckResponse(BaseSchema):
    """Health check response schema."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0", description="API version")
    checks: Dict[str, Any] = Field(default_factory=dict, description="Individual service checks")


class ErrorResponse(BaseSchema):
    """Error response schema."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request identifier")


class PaginationParams(BaseModel):
    """Pagination parameters schema."""
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=100, ge=1, le=1000, description="Page size")
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.size


class PaginatedResponse(BaseSchema):
    """Paginated response schema."""
    items: List[Any] = Field(..., description="Response items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page")
    size: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total number of pages")
    
    @validator('pages', always=True)
    def calculate_pages(cls, v, values):
        """Calculate total pages based on total items and page size."""
        total = values.get('total', 0)
        size = values.get('size', 1)
        return (total + size - 1) // size if total > 0 else 0


class ModelPredictionBase(BaseSchema):
    """Base model prediction schema."""
    customer_id: str = Field(..., description="Customer identifier")
    model_version: str = Field(..., description="Model version")
    prediction_type: ModelType = Field(..., description="Type of prediction")
    prediction_date: datetime = Field(default_factory=datetime.utcnow)
    model_features: Dict[str, Any] = Field(..., description="Input features used")
    prediction_explanation: Optional[str] = Field(None, description="Prediction explanation")
    business_decision: str = Field(..., description="Business decision made")


class FeatureImportance(BaseSchema):
    """Feature importance schema."""
    feature_name: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Feature importance value")
    rank: int = Field(..., description="Feature importance rank")


class ModelExplanation(BaseSchema):
    """Model explanation schema."""
    prediction_id: str = Field(..., description="Prediction identifier")
    shap_values: Dict[str, float] = Field(..., description="SHAP feature values")
    lime_explanation: Optional[Dict[str, Any]] = Field(None, description="LIME explanation")
    feature_importance: List[FeatureImportance] = Field(..., description="Feature importance ranking")
    explanation_text: str = Field(..., description="Human-readable explanation")


class AlertBase(BaseSchema):
    """Base alert schema."""
    alert_type: str = Field(..., description="Type of alert")
    alert_severity: AlertSeverity = Field(..., description="Alert severity level")
    alert_message: str = Field(..., description="Alert message")
    alert_data: Optional[Dict[str, Any]] = Field(None, description="Additional alert data")
    is_resolved: bool = Field(default=False, description="Whether alert is resolved")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")


class KPIMetric(BaseSchema):
    """KPI metric schema."""
    metric_name: str = Field(..., description="Metric name")
    metric_value: float = Field(..., description="Metric value")
    metric_date: date = Field(..., description="Metric date")
    metric_period: str = Field(default="daily", description="Metric period")


class DriftDetectionResult(BaseSchema):
    """Data drift detection result schema."""
    feature_name: str = Field(..., description="Feature name")
    model_name: str = Field(..., description="Model name")
    drift_metric: str = Field(..., description="Drift detection metric used")
    drift_score: float = Field(..., description="Drift score")
    threshold_value: float = Field(..., description="Drift threshold")
    is_drifted: bool = Field(..., description="Whether drift is detected")
    detection_date: datetime = Field(..., description="Detection timestamp")


class FilterParams(BaseModel):
    """Common filter parameters."""
    start_date: Optional[date] = Field(None, description="Start date filter")
    end_date: Optional[date] = Field(None, description="End date filter")
    customer_ids: Optional[List[str]] = Field(None, description="Customer ID filters")
    risk_segments: Optional[List[RiskSegment]] = Field(None, description="Risk segment filters")
    
    @validator('end_date')
    def end_date_after_start_date(cls, v, values):
        """Validate that end_date is after start_date."""
        start_date = values.get('start_date')
        if start_date and v and v < start_date:
            raise ValueError('end_date must be after start_date')
        return v
