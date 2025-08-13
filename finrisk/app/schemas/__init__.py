"""
FinRisk Pydantic Schemas Package

This package contains all Pydantic schemas for the FinRisk application.
"""

# Base schemas and enums
from .base import (
    BaseSchema,
    TimestampMixin,
    RiskSegment,
    EmploymentStatus,
    ApplicationStatus,
    LoanPurpose,
    MerchantCategory,
    TransactionType,
    ModelType,
    AlertSeverity,
    CustomerProfile,
    CreditBureauData,
    HealthCheckResponse,
    ErrorResponse,
    PaginationParams,
    PaginatedResponse,
    ModelPredictionBase,
    FeatureImportance,
    ModelExplanation,
    AlertBase,
    KPIMetric,
    DriftDetectionResult,
    FilterParams
)

# Customer schemas
from .customers import (
    CustomerCreate,
    CustomerUpdate,
    CustomerResponse,
    CustomerSummary,
    CustomerRiskProfile,
    CustomerSearchParams,
    CustomerAnalytics,
    CustomerPortfolio
)

# Credit application schemas
from .applications import (
    CreditApplicationCreate,
    CreditApplicationUpdate,
    CreditApplicationResponse,
    CreditApplicationSummary,
    CreditDecision,
    CreditApplicationSearchParams,
    CreditApplicationAnalytics,
    CreditApplicationPortfolio
)

# Model prediction schemas
from .predictions import (
    CreditRiskPrediction,
    FraudDetectionPrediction,
    ModelPredictionCreate,
    ModelPredictionResponse,
    ModelPredictionSummary,
    ModelPerformanceMetrics,
    ModelDriftAlert,
    PredictionSearchParams,
    ModelExplanationRequest,
    ModelExplanationResponse
)

# Credit risk assessment schemas
from .credit_risk import (
    CreditApplicationRequest,
    CreditApplicationResponse,
    CreditScoringRequest,
    CreditScoringResponse,
    CreditFeatures,
    CreditModelMetrics,
    CreditPrediction,
    CreditDecision,
    CreditPortfolioMetrics,
    CreditStressTestScenario,
    CreditStressTestResult,
    BatchCreditScoringRequest,
    BatchCreditScoringResponse
)

# Fraud detection schemas
from .fraud_detection import (
    TransactionData,
    FraudDetectionRequest,
    FraudDetectionResponse,
    FraudFeatures,
    FraudAlert,
    FraudModelMetrics,
    FraudPrediction,
    FraudInvestigation,
    FraudPortfolioMetrics,
    BatchFraudDetectionRequest,
    BatchFraudDetectionResponse,
    FraudRuleEngine,
    FraudRuleViolation
)

__all__ = [
    # Base schemas
    "BaseSchema",
    "TimestampMixin",
    "RiskSegment",
    "EmploymentStatus",
    "ApplicationStatus",
    "LoanPurpose",
    "MerchantCategory",
    "TransactionType",
    "ModelType",
    "AlertSeverity",
    "CustomerProfile",
    "CreditBureauData",
    "HealthCheckResponse",
    "ErrorResponse",
    "PaginationParams",
    "PaginatedResponse",
    "ModelPredictionBase",
    "FeatureImportance",
    "ModelExplanation",
    "AlertBase",
    "KPIMetric",
    "DriftDetectionResult",
    "FilterParams",
    
    # Customer schemas
    "CustomerCreate",
    "CustomerUpdate",
    "CustomerResponse",
    "CustomerSummary",
    "CustomerRiskProfile",
    "CustomerSearchParams",
    "CustomerAnalytics",
    "CustomerPortfolio",
    
    # Credit application schemas
    "CreditApplicationCreate",
    "CreditApplicationUpdate",
    "CreditApplicationResponse",
    "CreditApplicationSummary",
    "CreditDecision",
    "CreditApplicationSearchParams",
    "CreditApplicationAnalytics",
    "CreditApplicationPortfolio",
    
    # Model prediction schemas
    "CreditRiskPrediction",
    "FraudDetectionPrediction",
    "ModelPredictionCreate",
    "ModelPredictionResponse",
    "ModelPredictionSummary",
    "ModelPerformanceMetrics",
    "ModelDriftAlert",
    "PredictionSearchParams",
    "ModelExplanationRequest",
    "ModelExplanationResponse",
    
    # Credit risk assessment schemas
    "CreditApplicationRequest",
    "CreditApplicationResponse",
    "CreditScoringRequest",
    "CreditScoringResponse",
    "CreditFeatures",
    "CreditModelMetrics",
    "CreditPrediction",
    "CreditDecision",
    "CreditPortfolioMetrics",
    "CreditStressTestScenario",
    "CreditStressTestResult",
    "BatchCreditScoringRequest",
    "BatchCreditScoringResponse",
    
    # Fraud detection schemas
    "TransactionData",
    "FraudDetectionRequest",
    "FraudDetectionResponse",
    "FraudFeatures",
    "FraudAlert",
    "FraudModelMetrics",
    "FraudPrediction",
    "FraudInvestigation",
    "FraudPortfolioMetrics",
    "BatchFraudDetectionRequest",
    "BatchFraudDetectionResponse",
    "FraudRuleEngine",
    "FraudRuleViolation"
]

