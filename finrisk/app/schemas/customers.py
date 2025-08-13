"""
Customer-related schemas for FinRisk application.
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import Field, validator

from .base import BaseSchema, RiskSegment, EmploymentStatus


class CustomerProfile(BaseSchema):
    """Schema for customer profile data (matches database schema)."""
    customer_id: str = Field(..., description="Unique customer identifier")
    first_name: str = Field(..., description="Customer first name")
    last_name: str = Field(..., description="Customer last name")
    email: str = Field(..., description="Customer email address")
    phone: Optional[str] = Field(None, description="Customer phone number")
    date_of_birth: date = Field(..., description="Customer date of birth")
    address: Optional[str] = Field(None, description="Customer address")
    city: Optional[str] = Field(None, description="Customer city")
    state: Optional[str] = Field(None, description="Customer state")
    zip_code: Optional[str] = Field(None, description="Customer zip code")
    country: str = Field(default="USA", description="Customer country")
    employment_status: Optional[str] = Field(None, description="Employment status")
    annual_income: Optional[float] = Field(None, ge=0, description="Annual income")
    credit_score: Optional[int] = Field(None, ge=300, le=850, description="Credit score")
    created_at: Optional[datetime] = Field(None, description="Record creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Record update timestamp")


class CustomerCreate(BaseSchema):
    """Schema for creating a new customer."""
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


class CustomerUpdate(BaseSchema):
    """Schema for updating customer information."""
    customer_age: Optional[int] = Field(None, ge=18, le=120, description="Customer age")
    annual_income: Optional[float] = Field(None, ge=0, description="Annual income in GBP")
    employment_status: Optional[EmploymentStatus] = Field(None, description="Employment status")
    account_tenure: Optional[int] = Field(None, ge=0, description="Account tenure in years")
    product_holdings: Optional[int] = Field(None, ge=0, description="Number of products held")
    relationship_value: Optional[float] = Field(None, description="Customer relationship value")
    risk_segment: Optional[RiskSegment] = Field(None, description="Risk segment classification")
    behavioral_score: Optional[float] = Field(None, ge=0, le=1000, description="Behavioral score")
    credit_score: Optional[int] = Field(None, ge=300, le=850, description="Credit score")
    city: Optional[str] = Field(None, description="Customer city")
    last_activity_date: Optional[date] = Field(None, description="Last activity date")


class CustomerResponse(BaseSchema):
    """Schema for customer response data."""
    customer_id: str = Field(..., description="Unique customer identifier")
    customer_age: int = Field(..., description="Customer age")
    annual_income: float = Field(..., description="Annual income in GBP")
    employment_status: EmploymentStatus = Field(..., description="Employment status")
    account_tenure: int = Field(..., description="Account tenure in years")
    product_holdings: int = Field(..., description="Number of products held")
    relationship_value: float = Field(..., description="Customer relationship value")
    risk_segment: RiskSegment = Field(..., description="Risk segment classification")
    behavioral_score: float = Field(..., description="Behavioral score")
    credit_score: int = Field(..., description="Credit score")
    city: str = Field(..., description="Customer city")
    last_activity_date: date = Field(..., description="Last activity date")
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Record update timestamp")


class CustomerSummary(BaseSchema):
    """Schema for customer summary data."""
    customer_id: str = Field(..., description="Unique customer identifier")
    customer_age: int = Field(..., description="Customer age")
    annual_income: float = Field(..., description="Annual income in GBP")
    employment_status: EmploymentStatus = Field(..., description="Employment status")
    risk_segment: RiskSegment = Field(..., description="Risk segment classification")
    credit_score: int = Field(..., description="Credit score")
    city: str = Field(..., description="Customer city")
    last_activity_date: date = Field(..., description="Last activity date")


class CustomerRiskProfile(BaseSchema):
    """Schema for customer risk profile."""
    customer_id: str = Field(..., description="Customer identifier")
    risk_segment: RiskSegment = Field(..., description="Risk segment classification")
    credit_score: int = Field(..., description="Credit score")
    behavioral_score: float = Field(..., description="Behavioral score")
    risk_factors: List[str] = Field(default_factory=list, description="List of risk factors")
    risk_score: float = Field(..., ge=0, le=1, description="Overall risk score")
    last_assessment_date: datetime = Field(..., description="Last risk assessment date")


class CustomerSearchParams(BaseSchema):
    """Schema for customer search parameters."""
    customer_id: Optional[str] = Field(None, description="Customer ID filter")
    risk_segment: Optional[RiskSegment] = Field(None, description="Risk segment filter")
    employment_status: Optional[EmploymentStatus] = Field(None, description="Employment status filter")
    city: Optional[str] = Field(None, description="City filter")
    min_age: Optional[int] = Field(None, ge=18, description="Minimum age filter")
    max_age: Optional[int] = Field(None, le=120, description="Maximum age filter")
    min_income: Optional[float] = Field(None, ge=0, description="Minimum income filter")
    max_income: Optional[float] = Field(None, description="Maximum income filter")
    min_credit_score: Optional[int] = Field(None, ge=300, description="Minimum credit score filter")
    max_credit_score: Optional[int] = Field(None, le=850, description="Maximum credit score filter")
    
    @validator('max_age')
    def max_age_greater_than_min_age(cls, v, values):
        """Validate that max_age is greater than min_age."""
        min_age = values.get('min_age')
        if min_age and v and v < min_age:
            raise ValueError('max_age must be greater than min_age')
        return v
    
    @validator('max_income')
    def max_income_greater_than_min_income(cls, v, values):
        """Validate that max_income is greater than min_income."""
        min_income = values.get('min_income')
        if min_income and v and v < min_income:
            raise ValueError('max_income must be greater than min_income')
        return v
    
    @validator('max_credit_score')
    def max_credit_score_greater_than_min_credit_score(cls, v, values):
        """Validate that max_credit_score is greater than min_credit_score."""
        min_credit_score = values.get('min_credit_score')
        if min_credit_score and v and v < min_credit_score:
            raise ValueError('max_credit_score must be greater than min_credit_score')
        return v


class CustomerAnalytics(BaseSchema):
    """Schema for customer analytics data."""
    customer_id: str = Field(..., description="Customer identifier")
    total_transactions: int = Field(..., ge=0, description="Total number of transactions")
    total_spend: float = Field(..., ge=0, description="Total spend amount")
    avg_transaction_value: float = Field(..., ge=0, description="Average transaction value")
    transaction_frequency: float = Field(..., ge=0, description="Transactions per month")
    risk_trend: str = Field(..., description="Risk trend (increasing/decreasing/stable)")
    segment_movement: Optional[str] = Field(None, description="Risk segment movement")
    last_analysis_date: datetime = Field(..., description="Last analysis date")


class CustomerPortfolio(BaseSchema):
    """Schema for customer portfolio data."""
    customer_id: str = Field(..., description="Customer identifier")
    total_balance: float = Field(..., description="Total account balance")
    total_credit_limit: float = Field(..., description="Total credit limit")
    credit_utilization: float = Field(..., ge=0, le=1, description="Credit utilization ratio")
    number_of_accounts: int = Field(..., ge=0, description="Number of accounts")
    account_types: List[str] = Field(default_factory=list, description="Types of accounts held")
    portfolio_value: float = Field(..., description="Total portfolio value")
    last_portfolio_date: datetime = Field(..., description="Last portfolio update date")

