"""
Credit risk assessment Pydantic schemas for FinRisk application.
"""

from datetime import datetime, date
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field, validator

from .base import (
    BaseSchema, ModelPredictionBase, EmploymentStatus,
    ApplicationStatus, LoanPurpose, ModelExplanation
)


class CreditApplicationRequest(BaseSchema):
    """Credit application request schema."""
    customer_id: str = Field(..., description="Customer identifier")
    loan_amount: int = Field(..., gt=0, description="Requested loan amount")
    loan_purpose: LoanPurpose = Field(..., description="Purpose of the loan")
    employment_status: EmploymentStatus = Field(..., description="Employment status")
    annual_income: float = Field(..., gt=0, description="Annual income")
    
    @validator('loan_amount')
    def validate_loan_amount(cls, v):
        """Validate loan amount is reasonable."""
        if v > 1000000:  # £1M max
            raise ValueError('Loan amount cannot exceed £1,000,000')
        if v < 1000:  # £1K min
            raise ValueError('Loan amount must be at least £1,000')
        return v


class CreditApplicationResponse(BaseSchema):
    """Credit application response schema."""
    application_id: str = Field(..., description="Application identifier")
    customer_id: str = Field(..., description="Customer identifier")
    application_date: date = Field(..., description="Application date")
    loan_amount: int = Field(..., description="Loan amount")
    loan_purpose: LoanPurpose = Field(..., description="Loan purpose")
    employment_status: EmploymentStatus = Field(..., description="Employment status")
    annual_income: float = Field(..., description="Annual income")
    debt_to_income_ratio: float = Field(..., description="Debt-to-income ratio")
    credit_score: int = Field(..., description="Credit score")
    application_status: ApplicationStatus = Field(..., description="Application status")
    default_flag: int = Field(..., description="Default flag (0 or 1)")


class CreditScoringRequest(BaseSchema):
    """Credit scoring request schema."""
    customer_id: str = Field(..., description="Customer identifier")
    loan_amount: int = Field(..., gt=0, description="Requested loan amount")
    loan_purpose: LoanPurpose = Field(..., description="Purpose of the loan")
    include_explanation: bool = Field(default=True, description="Include model explanation")
    model_version: Optional[str] = Field(None, description="Specific model version to use")


class CreditScoringResponse(BaseSchema):
    """Credit scoring response schema."""
    customer_id: str = Field(..., description="Customer identifier")
    prediction_id: str = Field(..., description="Prediction identifier")
    credit_score: int = Field(..., description="Predicted credit score")
    risk_score: float = Field(..., description="Risk score (0-1000)")
    probability_of_default: float = Field(..., description="Probability of default (0-1)")
    risk_grade: str = Field(..., description="Risk grade (A-F)")
    recommended_decision: str = Field(..., description="Recommended business decision")
    confidence_interval: Dict[str, float] = Field(..., description="Prediction confidence interval")
    model_version: str = Field(..., description="Model version used")
    prediction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('risk_grade')
    def validate_risk_grade(cls, v):
        """Validate risk grade is in valid range."""
        valid_grades = ['A', 'B', 'C', 'D', 'E', 'F']
        if v not in valid_grades:
            raise ValueError(f'Risk grade must be one of: {valid_grades}')
        return v


class CreditFeatures(BaseSchema):
    """Credit risk features schema."""
    customer_id: str = Field(..., description="Customer identifier")
    
    # Demographic features
    age: int = Field(..., description="Customer age")
    annual_income: float = Field(..., description="Annual income")
    employment_status: EmploymentStatus = Field(..., description="Employment status")
    account_tenure: int = Field(..., description="Account tenure in years")
    
    # Credit bureau features
    credit_score: int = Field(..., description="Credit score")
    credit_history_length: int = Field(..., description="Credit history length")
    number_of_accounts: int = Field(..., description="Number of credit accounts")
    total_credit_limit: int = Field(..., description="Total credit limit")
    credit_utilization: float = Field(..., description="Credit utilization ratio")
    payment_history: float = Field(..., description="Payment history score")
    public_records: int = Field(..., description="Number of public records")
    
    # Application features
    loan_amount: int = Field(..., description="Requested loan amount")
    debt_to_income_ratio: float = Field(..., description="Debt-to-income ratio")
    loan_to_income_ratio: float = Field(..., description="Loan-to-income ratio")
    
    # Behavioral features
    behavioral_score: float = Field(..., description="Behavioral score")
    product_holdings: int = Field(..., description="Number of products held")
    relationship_value: float = Field(..., description="Customer relationship value")
    
    # Derived features
    income_stability: float = Field(..., description="Income stability score")
    credit_mix_score: float = Field(..., description="Credit mix diversity score")
    recent_inquiry_count: int = Field(default=0, description="Recent credit inquiries")


class CreditModelMetrics(BaseSchema):
    """Credit model performance metrics schema."""
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    evaluation_date: datetime = Field(..., description="Evaluation date")
    dataset_size: int = Field(..., description="Evaluation dataset size")
    
    # Performance metrics
    auc_score: float = Field(..., description="Area Under Curve score")
    gini_coefficient: float = Field(..., description="Gini coefficient")
    ks_statistic: float = Field(..., description="Kolmogorov-Smirnov statistic")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1_score: float = Field(..., description="F1 score")
    
    # Business metrics
    approval_rate: float = Field(..., description="Model approval rate")
    default_rate: float = Field(..., description="Portfolio default rate")
    expected_loss: float = Field(..., description="Expected loss estimate")
    
    @validator('auc_score', 'precision', 'recall', 'f1_score')
    def validate_score_range(cls, v):
        """Validate scores are in valid range [0, 1]."""
        if not 0 <= v <= 1:
            raise ValueError('Score must be between 0 and 1')
        return v


class CreditPrediction(ModelPredictionBase):
    """Credit prediction result schema."""
    risk_score: float = Field(..., description="Risk score (0-1000)")
    probability_of_default: float = Field(..., ge=0, le=1, description="Default probability")
    credit_grade: str = Field(..., description="Credit grade (A-F)")
    recommended_action: str = Field(..., description="Recommended action")
    confidence_score: float = Field(..., description="Prediction confidence")
    
    # Risk factors
    primary_risk_factors: List[str] = Field(..., description="Primary risk factors")
    protective_factors: List[str] = Field(..., description="Protective factors")
    
    # Financial metrics
    estimated_loss_rate: float = Field(..., description="Estimated loss rate")
    suggested_interest_rate: float = Field(..., description="Suggested interest rate")
    maximum_loan_amount: int = Field(..., description="Maximum recommended loan amount")


class CreditDecision(BaseSchema):
    """Credit decision audit schema."""
    decision_id: str = Field(..., description="Decision identifier")
    customer_id: str = Field(..., description="Customer identifier")
    application_id: str = Field(..., description="Application identifier")
    model_prediction: CreditPrediction = Field(..., description="Model prediction")
    final_decision: str = Field(..., description="Final business decision")
    decision_reason: str = Field(..., description="Reason for decision")
    override_flag: bool = Field(default=False, description="Manual override applied")
    override_reason: Optional[str] = Field(None, description="Reason for override")
    decision_maker: str = Field(..., description="Decision maker identifier")
    decision_timestamp: datetime = Field(default_factory=datetime.utcnow)


class CreditPortfolioMetrics(BaseSchema):
    """Credit portfolio metrics schema."""
    portfolio_date: date = Field(..., description="Portfolio snapshot date")
    total_applications: int = Field(..., description="Total applications")
    approved_applications: int = Field(..., description="Approved applications")
    declined_applications: int = Field(..., description="Declined applications")
    
    # Portfolio composition
    total_outstanding_balance: float = Field(..., description="Total outstanding balance")
    average_loan_amount: float = Field(..., description="Average loan amount")
    total_customers: int = Field(..., description="Total unique customers")
    
    # Risk metrics
    portfolio_default_rate: float = Field(..., description="Portfolio default rate")
    average_credit_score: float = Field(..., description="Average credit score")
    risk_segment_distribution: Dict[str, int] = Field(..., description="Risk segment counts")
    
    # Performance metrics
    approval_rate: float = Field(..., description="Overall approval rate")
    loss_rate: float = Field(..., description="Portfolio loss rate")
    net_interest_margin: float = Field(..., description="Net interest margin")


class CreditStressTestScenario(BaseSchema):
    """Credit stress test scenario schema."""
    scenario_name: str = Field(..., description="Scenario name")
    scenario_description: str = Field(..., description="Scenario description")
    
    # Economic factors
    unemployment_rate_change: float = Field(..., description="Unemployment rate change (%)")
    gdp_growth_change: float = Field(..., description="GDP growth change (%)")
    interest_rate_change: float = Field(..., description="Interest rate change (%)")
    house_price_change: float = Field(..., description="House price change (%)")
    
    # Expected impacts
    expected_default_rate_change: float = Field(..., description="Expected default rate change (%)")
    expected_loss_change: float = Field(..., description="Expected loss change (%)")
    capital_impact: float = Field(..., description="Capital requirement impact")


class CreditStressTestResult(BaseSchema):
    """Credit stress test result schema."""
    test_id: str = Field(..., description="Stress test identifier")
    scenario: CreditStressTestScenario = Field(..., description="Test scenario")
    test_date: datetime = Field(..., description="Test execution date")
    
    # Results
    baseline_default_rate: float = Field(..., description="Baseline default rate")
    stressed_default_rate: float = Field(..., description="Stressed default rate")
    default_rate_change: float = Field(..., description="Default rate change")
    
    baseline_loss: float = Field(..., description="Baseline expected loss")
    stressed_loss: float = Field(..., description="Stressed expected loss")
    loss_change: float = Field(..., description="Loss change")
    
    # Capital impact
    current_capital_ratio: float = Field(..., description="Current capital ratio")
    stressed_capital_ratio: float = Field(..., description="Stressed capital ratio")
    capital_shortfall: float = Field(..., description="Capital shortfall if any")
    
    # Risk grade migrations
    grade_migrations: Dict[str, Dict[str, int]] = Field(..., description="Risk grade migration matrix")


class BatchCreditScoringRequest(BaseSchema):
    """Batch credit scoring request schema."""
    batch_id: str = Field(..., description="Batch identifier")
    customer_ids: List[str] = Field(..., min_items=1, max_items=1000, description="Customer IDs to score")
    include_explanations: bool = Field(default=False, description="Include explanations")
    model_version: Optional[str] = Field(None, description="Model version to use")
    
    @validator('customer_ids')
    def validate_customer_ids(cls, v):
        """Validate customer IDs are unique."""
        if len(v) != len(set(v)):
            raise ValueError('Customer IDs must be unique')
        return v


class BatchCreditScoringResponse(BaseSchema):
    """Batch credit scoring response schema."""
    batch_id: str = Field(..., description="Batch identifier")
    total_requests: int = Field(..., description="Total scoring requests")
    successful_scores: int = Field(..., description="Successful scores")
    failed_scores: int = Field(..., description="Failed scores")
    processing_time_seconds: float = Field(..., description="Total processing time")
    results: List[CreditScoringResponse] = Field(..., description="Individual scoring results")
    errors: List[Dict[str, str]] = Field(default_factory=list, description="Error details for failed scores")


class CreditApplication(BaseSchema):
    """Credit application schema for API compatibility."""
    customer_id: str = Field(..., description="Customer identifier")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_purpose: str = Field(..., description="Purpose of the loan")
    employment_status: str = Field(..., description="Employment status")
    annual_income: float = Field(..., gt=0, description="Annual income")
    credit_score: Optional[int] = Field(None, ge=300, le=850, description="Credit score")
    existing_debt: float = Field(0, ge=0, description="Existing debt amount")
    collateral_value: Optional[float] = Field(None, ge=0, description="Collateral value")


class CreditScoreRequest(BaseSchema):
    """Credit score request schema for API compatibility."""
    customer_id: str = Field(..., description="Customer identifier")
    features: Dict[str, Any] = Field(..., description="Customer features")


class CreditScoreResponse(BaseSchema):
    """Credit score response schema for API compatibility."""
    customer_id: str = Field(..., description="Customer identifier")
    credit_score: float = Field(..., description="Credit score")
    risk_level: str = Field(..., description="Risk level")
    confidence: float = Field(..., description="Confidence score")
    factors: List[Dict[str, Any]] = Field(..., description="Contributing factors")
    recommendation: str = Field(..., description="Recommendation")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RiskAssessment(BaseSchema):
    """Risk assessment schema for API compatibility."""
    application_id: str = Field(..., description="Application identifier")
    customer_id: str = Field(..., description="Customer identifier")
    risk_score: float = Field(..., description="Risk score")
    risk_level: str = Field(..., description="Risk level")
    approval_probability: float = Field(..., description="Approval probability")
    recommended_amount: Optional[float] = Field(None, description="Recommended loan amount")
    conditions: List[str] = Field(..., description="Approval conditions")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
