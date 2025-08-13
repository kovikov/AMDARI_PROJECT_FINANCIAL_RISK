"""
Fraud detection Pydantic schemas for FinRisk application.
"""

from datetime import datetime, date
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field, validator

from .base import (
    BaseSchema, ModelPredictionBase, MerchantCategory,
    TransactionType, ModelExplanation
)


class TransactionData(BaseSchema):
    """Transaction data schema."""
    transaction_id: str = Field(..., description="Transaction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    transaction_date: datetime = Field(..., description="Transaction timestamp")
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_category: MerchantCategory = Field(..., description="Merchant category")
    transaction_type: TransactionType = Field(..., description="Transaction type")
    location: str = Field(..., description="Transaction location")
    device_info: str = Field(..., description="Device used for transaction")
    fraud_flag: int = Field(..., description="Fraud flag (0 or 1)")
    investigation_status: str = Field(..., description="Investigation status")


class FraudDetectionRequest(BaseSchema):
    """Fraud detection request schema."""
    transaction_id: str = Field(..., description="Transaction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_category: MerchantCategory = Field(..., description="Merchant category")
    transaction_type: TransactionType = Field(..., description="Transaction type")
    location: str = Field(..., description="Transaction location")
    device_info: str = Field(..., description="Device information")
    transaction_time: datetime = Field(default_factory=datetime.utcnow, description="Transaction timestamp")
    include_explanation: bool = Field(default=True, description="Include fraud explanation")
    
    @validator('amount')
    def validate_amount(cls, v):
        """Validate transaction amount is reasonable."""
        if v > 100000:  # £100K max single transaction
            raise ValueError('Transaction amount exceeds maximum limit')
        return v


class FraudDetectionResponse(BaseSchema):
    """Fraud detection response schema."""
    transaction_id: str = Field(..., description="Transaction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    prediction_id: str = Field(..., description="Prediction identifier")
    fraud_probability: float = Field(..., ge=0, le=1, description="Fraud probability")
    fraud_score: int = Field(..., ge=0, le=1000, description="Fraud score (0-1000)")
    risk_level: str = Field(..., description="Risk level (LOW/MEDIUM/HIGH/CRITICAL)")
    recommended_action: str = Field(..., description="Recommended action")
    confidence_score: float = Field(..., description="Model confidence")
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    prediction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('risk_level')
    def validate_risk_level(cls, v):
        """Validate risk level is in valid range."""
        valid_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        if v not in valid_levels:
            raise ValueError(f'Risk level must be one of: {valid_levels}')
        return v


class FraudFeatures(BaseSchema):
    """Fraud detection features schema."""
    transaction_id: str = Field(..., description="Transaction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    
    # Transaction features
    amount: float = Field(..., description="Transaction amount")
    merchant_category: MerchantCategory = Field(..., description="Merchant category")
    transaction_type: TransactionType = Field(..., description="Transaction type")
    location: str = Field(..., description="Transaction location")
    device_info: str = Field(..., description="Device information")
    transaction_hour: int = Field(..., ge=0, le=23, description="Hour of transaction")
    transaction_day_of_week: int = Field(..., ge=0, le=6, description="Day of week")
    
    # Customer behavior features
    avg_transaction_amount: float = Field(..., description="Customer average transaction amount")
    transaction_frequency: float = Field(..., description="Recent transaction frequency")
    location_frequency: float = Field(..., description="Location usage frequency")
    merchant_frequency: float = Field(..., description="Merchant usage frequency")
    device_frequency: float = Field(..., description="Device usage frequency")
    
    # Velocity features
    transactions_last_hour: int = Field(..., description="Transactions in last hour")
    transactions_last_day: int = Field(..., description="Transactions in last day")
    amount_last_hour: float = Field(..., description="Total amount in last hour")
    amount_last_day: float = Field(..., description="Total amount in last day")
    
    # Anomaly features
    amount_zscore: float = Field(..., description="Amount Z-score vs customer history")
    time_since_last_transaction: float = Field(..., description="Hours since last transaction")
    distance_from_home: float = Field(..., description="Distance from home location (km)")
    new_merchant_flag: bool = Field(..., description="New merchant for customer")
    new_location_flag: bool = Field(..., description="New location for customer")
    new_device_flag: bool = Field(..., description="New device for customer")
    
    # Risk indicators
    high_risk_merchant: bool = Field(..., description="High-risk merchant category")
    unusual_time: bool = Field(..., description="Unusual transaction time")
    round_amount: bool = Field(..., description="Round amount indicator")
    weekend_transaction: bool = Field(..., description="Weekend transaction")


class FraudAlert(BaseSchema):
    """Fraud alert schema."""
    alert_id: str = Field(..., description="Alert identifier")
    transaction_id: str = Field(..., description="Transaction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    alert_type: str = Field(..., description="Alert type")
    fraud_probability: float = Field(..., description="Fraud probability")
    risk_level: str = Field(..., description="Risk level")
    alert_message: str = Field(..., description="Alert message")
    investigation_required: bool = Field(..., description="Investigation required flag")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Investigation details
    assigned_to: Optional[str] = Field(None, description="Assigned investigator")
    investigation_status: str = Field(default="Pending", description="Investigation status")
    resolution: Optional[str] = Field(None, description="Alert resolution")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    false_positive: Optional[bool] = Field(None, description="False positive flag")


class FraudModelMetrics(BaseSchema):
    """Fraud model performance metrics schema."""
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    evaluation_date: datetime = Field(..., description="Evaluation date")
    dataset_size: int = Field(..., description="Evaluation dataset size")
    
    # Classification metrics
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1_score: float = Field(..., description="F1 score")
    auc_score: float = Field(..., description="Area Under Curve score")
    
    # Business metrics
    false_positive_rate: float = Field(..., description="False positive rate")
    false_negative_rate: float = Field(..., description="False negative rate")
    detection_rate: float = Field(..., description="Fraud detection rate")
    
    # Operational metrics
    average_processing_time_ms: float = Field(..., description="Average processing time")
    throughput_per_second: float = Field(..., description="Transactions per second")
    
    @validator('precision', 'recall', 'f1_score', 'auc_score')
    def validate_score_range(cls, v):
        """Validate scores are in valid range [0, 1]."""
        if not 0 <= v <= 1:
            raise ValueError('Score must be between 0 and 1')
        return v


class FraudPrediction(ModelPredictionBase):
    """Fraud prediction result schema."""
    fraud_probability: float = Field(..., ge=0, le=1, description="Fraud probability")
    fraud_score: int = Field(..., ge=0, le=1000, description="Fraud score")
    risk_level: str = Field(..., description="Risk level")
    anomaly_score: float = Field(..., description="Anomaly detection score")
    
    # Risk factors
    fraud_indicators: List[str] = Field(..., description="Fraud risk indicators")
    risk_factors: Dict[str, float] = Field(..., description="Risk factor scores")
    
    # Model outputs
    isolation_forest_score: float = Field(..., description="Isolation Forest anomaly score")
    one_class_svm_score: float = Field(..., description="One-Class SVM score")
    ensemble_score: float = Field(..., description="Ensemble model score")
    
    # Business context
    recommended_action: str = Field(..., description="Recommended action")
    block_transaction: bool = Field(..., description="Block transaction flag")
    require_verification: bool = Field(..., description="Require additional verification")


class FraudInvestigation(BaseSchema):
    """Fraud investigation schema."""
    investigation_id: str = Field(..., description="Investigation identifier")
    transaction_id: str = Field(..., description="Transaction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    alert_id: str = Field(..., description="Related alert identifier")
    
    # Investigation details
    investigator_id: str = Field(..., description="Investigator identifier")
    investigation_type: str = Field(..., description="Investigation type")
    priority: str = Field(..., description="Investigation priority")
    status: str = Field(..., description="Investigation status")
    
    # Findings
    is_fraud: Optional[bool] = Field(None, description="Fraud determination")
    fraud_type: Optional[str] = Field(None, description="Type of fraud if confirmed")
    loss_amount: Optional[float] = Field(None, description="Loss amount if fraud")
    recovery_amount: Optional[float] = Field(None, description="Recovered amount")
    
    # Timestamps
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = Field(None, description="Investigation close time")
    
    # Notes and evidence
    investigation_notes: List[str] = Field(default_factory=list, description="Investigation notes")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Evidence collected")


class FraudPortfolioMetrics(BaseSchema):
    """Fraud portfolio metrics schema."""
    portfolio_date: date = Field(..., description="Portfolio snapshot date")
    total_transactions: int = Field(..., description="Total transactions")
    flagged_transactions: int = Field(..., description="Flagged transactions")
    confirmed_frauds: int = Field(..., description="Confirmed fraud cases")
    false_positives: int = Field(..., description="False positive cases")
    
    # Financial metrics
    total_transaction_volume: float = Field(..., description="Total transaction volume")
    fraud_loss_amount: float = Field(..., description="Total fraud losses")
    prevented_fraud_amount: float = Field(..., description="Prevented fraud amount")
    investigation_costs: float = Field(..., description="Investigation costs")
    
    # Performance metrics
    fraud_detection_rate: float = Field(..., description="Fraud detection rate")
    false_positive_rate: float = Field(..., description="False positive rate")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    
    # Operational metrics
    average_investigation_time: float = Field(..., description="Average investigation time (hours)")
    pending_investigations: int = Field(..., description="Pending investigations count")


class BatchFraudDetectionRequest(BaseSchema):
    """Batch fraud detection request schema."""
    batch_id: str = Field(..., description="Batch identifier")
    transactions: List[FraudDetectionRequest] = Field(..., min_items=1, max_items=1000, description="Transactions to score")
    include_explanations: bool = Field(default=False, description="Include explanations")
    model_version: Optional[str] = Field(None, description="Model version to use")
    
    @validator('transactions')
    def validate_unique_transactions(cls, v):
        """Validate transaction IDs are unique."""
        transaction_ids = [t.transaction_id for t in v]
        if len(transaction_ids) != len(set(transaction_ids)):
            raise ValueError('Transaction IDs must be unique')
        return v


class BatchFraudDetectionResponse(BaseSchema):
    """Batch fraud detection response schema."""
    batch_id: str = Field(..., description="Batch identifier")
    total_requests: int = Field(..., description="Total detection requests")
    successful_detections: int = Field(..., description="Successful detections")
    failed_detections: int = Field(..., description="Failed detections")
    processing_time_seconds: float = Field(..., description="Total processing time")
    average_processing_time_ms: float = Field(..., description="Average processing time per transaction")
    results: List[FraudDetectionResponse] = Field(..., description="Individual detection results")
    errors: List[Dict[str, str]] = Field(default_factory=list, description="Error details for failed detections")


class FraudRuleEngine(BaseSchema):
    """Fraud rule engine configuration schema."""
    rule_id: str = Field(..., description="Rule identifier")
    rule_name: str = Field(..., description="Rule name")
    rule_description: str = Field(..., description="Rule description")
    rule_type: str = Field(..., description="Rule type (threshold, velocity, etc.)")
    
    # Rule parameters
    threshold_value: Optional[float] = Field(None, description="Threshold value")
    time_window_minutes: Optional[int] = Field(None, description="Time window in minutes")
    conditions: Dict[str, Any] = Field(..., description="Rule conditions")
    
    # Rule metadata
    is_active: bool = Field(default=True, description="Rule active status")
    severity: str = Field(..., description="Rule severity level")
    created_by: str = Field(..., description="Rule creator")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_modified: datetime = Field(default_factory=datetime.utcnow)


class FraudRuleViolation(BaseSchema):
    """Fraud rule violation schema."""
    violation_id: str = Field(..., description="Violation identifier")
    rule_id: str = Field(..., description="Violated rule identifier")
    transaction_id: str = Field(..., description="Transaction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    
    # Violation details
    violation_type: str = Field(..., description="Type of violation")
    violation_value: float = Field(..., description="Violation value")
    threshold_value: float = Field(..., description="Rule threshold")
    severity: str = Field(..., description="Violation severity")
    
    # Timestamps
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Actions taken
    action_taken: str = Field(..., description="Action taken")
    alert_generated: bool = Field(..., description="Alert generated flag")
    transaction_blocked: bool = Field(..., description="Transaction blocked flag")
