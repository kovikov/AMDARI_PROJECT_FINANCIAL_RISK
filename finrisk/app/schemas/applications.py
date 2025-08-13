"""
Credit application schemas for FinRisk application.
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import Field, validator

from .base import BaseSchema, ApplicationStatus, LoanPurpose, RiskSegment


class CreditApplication(BaseSchema):
    """Schema for credit application data (matches database schema)."""
    application_id: str = Field(..., description="Unique application identifier")
    customer_id: str = Field(..., description="Customer identifier")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_purpose: str = Field(..., description="Purpose of the loan")
    employment_status: Optional[str] = Field(None, description="Employment status")
    annual_income: Optional[float] = Field(None, ge=0, description="Annual income")
    credit_score: Optional[int] = Field(None, ge=300, le=850, description="Credit score")
    existing_debt: float = Field(default=0, ge=0, description="Existing debt amount")
    collateral_value: Optional[float] = Field(None, ge=0, description="Collateral value")
    application_status: str = Field(default="PENDING", description="Application status")
    risk_score: Optional[float] = Field(None, ge=0, le=1, description="Risk score")
    approval_probability: Optional[float] = Field(None, ge=0, le=1, description="Approval probability")
    created_at: Optional[datetime] = Field(None, description="Record creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Record update timestamp")


class CreditApplicationCreate(BaseSchema):
    """Schema for creating a new credit application."""
    customer_id: str = Field(..., description="Customer identifier")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_purpose: LoanPurpose = Field(..., description="Purpose of the loan")
    loan_term: int = Field(..., ge=1, le=360, description="Loan term in months")
    interest_rate: float = Field(..., ge=0, le=100, description="Interest rate percentage")
    monthly_payment: float = Field(..., gt=0, description="Monthly payment amount")
    debt_to_income_ratio: float = Field(..., ge=0, le=1, description="Debt to income ratio")
    collateral_value: Optional[float] = Field(None, ge=0, description="Collateral value if applicable")
    application_date: date = Field(default_factory=date.today, description="Application date")
    application_source: str = Field(..., description="Source of application (online, branch, etc.)")


class CreditApplicationUpdate(BaseSchema):
    """Schema for updating credit application information."""
    loan_amount: Optional[float] = Field(None, gt=0, description="Requested loan amount")
    loan_purpose: Optional[LoanPurpose] = Field(None, description="Purpose of the loan")
    loan_term: Optional[int] = Field(None, ge=1, le=360, description="Loan term in months")
    interest_rate: Optional[float] = Field(None, ge=0, le=100, description="Interest rate percentage")
    monthly_payment: Optional[float] = Field(None, gt=0, description="Monthly payment amount")
    debt_to_income_ratio: Optional[float] = Field(None, ge=0, le=1, description="Debt to income ratio")
    collateral_value: Optional[float] = Field(None, ge=0, description="Collateral value if applicable")
    application_status: Optional[ApplicationStatus] = Field(None, description="Application status")
    application_source: Optional[str] = Field(None, description="Source of application")


class CreditApplicationResponse(BaseSchema):
    """Schema for credit application response data."""
    application_id: str = Field(..., description="Unique application identifier")
    customer_id: str = Field(..., description="Customer identifier")
    loan_amount: float = Field(..., description="Requested loan amount")
    loan_purpose: LoanPurpose = Field(..., description="Purpose of the loan")
    loan_term: int = Field(..., description="Loan term in months")
    interest_rate: float = Field(..., description="Interest rate percentage")
    monthly_payment: float = Field(..., description="Monthly payment amount")
    debt_to_income_ratio: float = Field(..., description="Debt to income ratio")
    collateral_value: Optional[float] = Field(None, description="Collateral value if applicable")
    application_status: ApplicationStatus = Field(..., description="Application status")
    application_date: date = Field(..., description="Application date")
    decision_date: Optional[datetime] = Field(None, description="Decision date")
    application_source: str = Field(..., description="Source of application")
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Record update timestamp")


class CreditApplicationSummary(BaseSchema):
    """Schema for credit application summary data."""
    application_id: str = Field(..., description="Unique application identifier")
    customer_id: str = Field(..., description="Customer identifier")
    loan_amount: float = Field(..., description="Requested loan amount")
    loan_purpose: LoanPurpose = Field(..., description="Purpose of the loan")
    application_status: ApplicationStatus = Field(..., description="Application status")
    application_date: date = Field(..., description="Application date")
    decision_date: Optional[datetime] = Field(None, description="Decision date")


class CreditDecision(BaseSchema):
    """Schema for credit decision data."""
    application_id: str = Field(..., description="Application identifier")
    decision: ApplicationStatus = Field(..., description="Credit decision")
    decision_reason: str = Field(..., description="Reason for decision")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    debt_to_income_ratio: float = Field(..., ge=0, le=1, description="Debt to income ratio")
    loan_to_value_ratio: Optional[float] = Field(None, ge=0, le=1, description="Loan to value ratio")
    decision_factors: List[str] = Field(default_factory=list, description="Factors influencing decision")
    decision_date: datetime = Field(..., description="Decision timestamp")
    decision_officer: Optional[str] = Field(None, description="Decision officer")


class CreditApplicationSearchParams(BaseSchema):
    """Schema for credit application search parameters."""
    application_id: Optional[str] = Field(None, description="Application ID filter")
    customer_id: Optional[str] = Field(None, description="Customer ID filter")
    application_status: Optional[ApplicationStatus] = Field(None, description="Application status filter")
    loan_purpose: Optional[LoanPurpose] = Field(None, description="Loan purpose filter")
    application_source: Optional[str] = Field(None, description="Application source filter")
    min_loan_amount: Optional[float] = Field(None, gt=0, description="Minimum loan amount filter")
    max_loan_amount: Optional[float] = Field(None, gt=0, description="Maximum loan amount filter")
    min_application_date: Optional[date] = Field(None, description="Minimum application date filter")
    max_application_date: Optional[date] = Field(None, description="Maximum application date filter")
    
    @validator('max_loan_amount')
    def max_loan_amount_greater_than_min_loan_amount(cls, v, values):
        """Validate that max_loan_amount is greater than min_loan_amount."""
        min_loan_amount = values.get('min_loan_amount')
        if min_loan_amount and v and v < min_loan_amount:
            raise ValueError('max_loan_amount must be greater than min_loan_amount')
        return v
    
    @validator('max_application_date')
    def max_application_date_after_min_application_date(cls, v, values):
        """Validate that max_application_date is after min_application_date."""
        min_application_date = values.get('min_application_date')
        if min_application_date and v and v < min_application_date:
            raise ValueError('max_application_date must be after min_application_date')
        return v


class CreditApplicationAnalytics(BaseSchema):
    """Schema for credit application analytics data."""
    application_id: str = Field(..., description="Application identifier")
    processing_time_hours: float = Field(..., ge=0, description="Application processing time in hours")
    risk_assessment_score: float = Field(..., ge=0, le=1, description="Risk assessment score")
    fraud_score: Optional[float] = Field(None, ge=0, le=1, description="Fraud detection score")
    compliance_score: float = Field(..., ge=0, le=1, description="Compliance score")
    customer_satisfaction_score: Optional[float] = Field(None, ge=0, le=5, description="Customer satisfaction score")
    last_analytics_date: datetime = Field(..., description="Last analytics update date")


class CreditApplicationPortfolio(BaseSchema):
    """Schema for credit application portfolio data."""
    total_applications: int = Field(..., ge=0, description="Total number of applications")
    approved_applications: int = Field(..., ge=0, description="Number of approved applications")
    declined_applications: int = Field(..., ge=0, description="Number of declined applications")
    pending_applications: int = Field(..., ge=0, description="Number of pending applications")
    total_loan_amount: float = Field(..., ge=0, description="Total loan amount requested")
    approved_loan_amount: float = Field(..., ge=0, description="Total approved loan amount")
    average_loan_amount: float = Field(..., ge=0, description="Average loan amount")
    approval_rate: float = Field(..., ge=0, le=1, description="Application approval rate")
    portfolio_date: datetime = Field(..., description="Portfolio calculation date")

