"""
Model prediction schemas for FinRisk application.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import Field, validator

from .base import BaseSchema, ModelType, ModelPredictionBase, FeatureImportance, ModelExplanation


class ModelPrediction(BaseSchema):
    """Schema for model prediction data (matches database schema)."""
    prediction_id: str = Field(..., description="Unique prediction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    application_id: Optional[str] = Field(None, description="Application identifier")
    model_type: str = Field(..., description="Type of model")
    prediction_type: str = Field(..., description="Type of prediction")
    prediction_value: float = Field(..., description="Prediction value")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Model confidence score")
    features_used: Optional[Dict[str, Any]] = Field(None, description="Features used for prediction")
    model_version: str = Field(default="1.0", description="Model version")
    prediction_timestamp: Optional[datetime] = Field(None, description="Prediction timestamp")
    created_at: Optional[datetime] = Field(None, description="Record creation timestamp")


class CreditRiskPrediction(BaseSchema):
    """Schema for credit risk prediction."""
    customer_id: str = Field(..., description="Customer identifier")
    model_version: str = Field(..., description="Model version")
    prediction_date: datetime = Field(default_factory=datetime.utcnow)
    risk_score: float = Field(..., ge=0, le=1, description="Credit risk score")
    default_probability: float = Field(..., ge=0, le=1, description="Default probability")
    credit_limit_recommendation: float = Field(..., ge=0, description="Recommended credit limit")
    interest_rate_recommendation: float = Field(..., ge=0, le=100, description="Recommended interest rate")
    risk_segment: str = Field(..., description="Risk segment classification")
    decision: str = Field(..., description="Credit decision (approve/decline)")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence score")
    model_features: Dict[str, Any] = Field(..., description="Input features used")
    feature_importance: List[FeatureImportance] = Field(default_factory=list, description="Feature importance ranking")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")


class FraudDetectionPrediction(BaseSchema):
    """Schema for fraud detection prediction."""
    customer_id: str = Field(..., description="Customer identifier")
    transaction_id: str = Field(..., description="Transaction identifier")
    model_version: str = Field(..., description="Model version")
    prediction_date: datetime = Field(default_factory=datetime.utcnow)
    fraud_probability: float = Field(..., ge=0, le=1, description="Fraud probability")
    risk_level: str = Field(..., description="Risk level (low/medium/high)")
    fraud_score: float = Field(..., ge=0, le=1, description="Fraud detection score")
    decision: str = Field(..., description="Fraud decision (allow/flag/block)")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence score")
    model_features: Dict[str, Any] = Field(..., description="Input features used")
    feature_importance: List[FeatureImportance] = Field(default_factory=list, description="Feature importance ranking")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")
    alert_triggered: bool = Field(default=False, description="Whether alert was triggered")


class ModelPredictionCreate(BaseSchema):
    """Schema for creating a new model prediction."""
    customer_id: str = Field(..., description="Customer identifier")
    model_version: str = Field(..., description="Model version")
    prediction_type: ModelType = Field(..., description="Type of prediction")
    prediction_date: datetime = Field(default_factory=datetime.utcnow)
    model_features: Dict[str, Any] = Field(..., description="Input features used")
    prediction_score: float = Field(..., ge=0, le=1, description="Prediction score")
    business_decision: str = Field(..., description="Business decision made")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence score")
    prediction_explanation: Optional[str] = Field(None, description="Prediction explanation")
    feature_importance: Optional[List[FeatureImportance]] = Field(None, description="Feature importance ranking")
    actual_outcome: Optional[Any] = Field(None, description="Actual outcome (for training data)")


class ModelPredictionResponse(BaseSchema):
    """Schema for model prediction response data."""
    prediction_id: str = Field(..., description="Unique prediction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    model_version: str = Field(..., description="Model version")
    prediction_type: ModelType = Field(..., description="Type of prediction")
    prediction_date: datetime = Field(..., description="Prediction date")
    model_features: Dict[str, Any] = Field(..., description="Input features used")
    prediction_score: float = Field(..., description="Prediction score")
    business_decision: str = Field(..., description="Business decision made")
    confidence_score: float = Field(..., description="Model confidence score")
    prediction_explanation: Optional[str] = Field(None, description="Prediction explanation")
    feature_importance: Optional[List[FeatureImportance]] = Field(None, description="Feature importance ranking")
    actual_outcome: Optional[Any] = Field(None, description="Actual outcome")
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Record update timestamp")


class ModelPredictionSummary(BaseSchema):
    """Schema for model prediction summary data."""
    prediction_id: str = Field(..., description="Unique prediction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    model_version: str = Field(..., description="Model version")
    prediction_type: ModelType = Field(..., description="Type of prediction")
    prediction_date: datetime = Field(..., description="Prediction date")
    prediction_score: float = Field(..., description="Prediction score")
    business_decision: str = Field(..., description="Business decision made")
    confidence_score: float = Field(..., description="Model confidence score")


class ModelPerformanceMetrics(BaseSchema):
    """Schema for model performance metrics."""
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    prediction_type: ModelType = Field(..., description="Type of prediction")
    accuracy: float = Field(..., ge=0, le=1, description="Model accuracy")
    precision: float = Field(..., ge=0, le=1, description="Model precision")
    recall: float = Field(..., ge=0, le=1, description="Model recall")
    f1_score: float = Field(..., ge=0, le=1, description="F1 score")
    auc_score: float = Field(..., ge=0, le=1, description="AUC score")
    confusion_matrix: Dict[str, int] = Field(..., description="Confusion matrix")
    total_predictions: int = Field(..., ge=0, description="Total number of predictions")
    evaluation_date: datetime = Field(..., description="Evaluation date")


class ModelDriftAlert(BaseSchema):
    """Schema for model drift alert."""
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    drift_metric: str = Field(..., description="Drift detection metric")
    drift_score: float = Field(..., description="Drift score")
    threshold_value: float = Field(..., description="Drift threshold")
    is_drifted: bool = Field(..., description="Whether drift is detected")
    affected_features: List[str] = Field(default_factory=list, description="Features affected by drift")
    alert_severity: str = Field(..., description="Alert severity level")
    alert_message: str = Field(..., description="Alert message")
    detection_date: datetime = Field(..., description="Detection timestamp")


class PredictionSearchParams(BaseSchema):
    """Schema for prediction search parameters."""
    customer_id: Optional[str] = Field(None, description="Customer ID filter")
    model_version: Optional[str] = Field(None, description="Model version filter")
    prediction_type: Optional[ModelType] = Field(None, description="Prediction type filter")
    business_decision: Optional[str] = Field(None, description="Business decision filter")
    min_prediction_score: Optional[float] = Field(None, ge=0, le=1, description="Minimum prediction score filter")
    max_prediction_score: Optional[float] = Field(None, ge=0, le=1, description="Maximum prediction score filter")
    min_confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Minimum confidence score filter")
    max_confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Maximum confidence score filter")
    min_prediction_date: Optional[datetime] = Field(None, description="Minimum prediction date filter")
    max_prediction_date: Optional[datetime] = Field(None, description="Maximum prediction date filter")
    
    @validator('max_prediction_score')
    def max_prediction_score_greater_than_min_prediction_score(cls, v, values):
        """Validate that max_prediction_score is greater than min_prediction_score."""
        min_prediction_score = values.get('min_prediction_score')
        if min_prediction_score and v and v < min_prediction_score:
            raise ValueError('max_prediction_score must be greater than min_prediction_score')
        return v
    
    @validator('max_confidence_score')
    def max_confidence_score_greater_than_min_confidence_score(cls, v, values):
        """Validate that max_confidence_score is greater than min_confidence_score."""
        min_confidence_score = values.get('min_confidence_score')
        if min_confidence_score and v and v < min_confidence_score:
            raise ValueError('max_confidence_score must be greater than min_confidence_score')
        return v
    
    @validator('max_prediction_date')
    def max_prediction_date_after_min_prediction_date(cls, v, values):
        """Validate that max_prediction_date is after min_prediction_date."""
        min_prediction_date = values.get('min_prediction_date')
        if min_prediction_date and v and v < min_prediction_date:
            raise ValueError('max_prediction_date must be after min_prediction_date')
        return v


class ModelExplanationRequest(BaseSchema):
    """Schema for model explanation request."""
    customer_id: str = Field(..., description="Customer identifier")
    model_version: str = Field(..., description="Model version")
    prediction_type: ModelType = Field(..., description="Type of prediction")
    explanation_method: str = Field(default="shap", description="Explanation method (shap/lime)")
    include_feature_importance: bool = Field(default=True, description="Include feature importance")
    include_text_explanation: bool = Field(default=True, description="Include text explanation")


class ModelExplanationResponse(BaseSchema):
    """Schema for model explanation response."""
    prediction_id: str = Field(..., description="Prediction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    model_version: str = Field(..., description="Model version")
    prediction_type: ModelType = Field(..., description="Type of prediction")
    explanation_method: str = Field(..., description="Explanation method used")
    shap_values: Dict[str, float] = Field(..., description="SHAP feature values")
    lime_explanation: Optional[Dict[str, Any]] = Field(None, description="LIME explanation")
    feature_importance: List[FeatureImportance] = Field(..., description="Feature importance ranking")
    explanation_text: str = Field(..., description="Human-readable explanation")
    explanation_date: datetime = Field(..., description="Explanation generation date")

