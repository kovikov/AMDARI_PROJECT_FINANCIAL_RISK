"""
Audit and monitoring module for FinRisk application.
Provides decision logging and model prediction tracking.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# FinRisk modules
from app.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)
settings = get_settings()


async def log_decision(
    decision_type: str,
    model_name: Optional[str],
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    confidence: Optional[float] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> None:
    """
    Log decision-making events for audit and compliance.
    
    Args:
        decision_type: Type of decision (e.g., 'credit_score', 'fraud_detection')
        model_name: Name of the model used
        input_data: Input data for the decision
        output_data: Output data from the decision
        confidence: Confidence score of the decision
        user_id: User identifier who made the request
        request_id: Request identifier for tracing
    """
    try:
        # Create audit log entry
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "decision_type": decision_type,
            "model_name": model_name,
            "user_id": user_id,
            "request_id": request_id,
            "confidence": confidence,
            "input_data": input_data,
            "output_data": output_data,
            "environment": settings.environment
        }
        
        # Log to application logs
        logger.info(f"Decision Log: {json.dumps(audit_entry, default=str)}")
        
        # In a production environment, you would also:
        # - Store in database for audit trail
        # - Send to monitoring system (e.g., ELK stack)
        # - Store in compliance system
        # - Trigger alerts for high-risk decisions
        
        # Save to audit file (for development)
        audit_file = Path(settings.paths.logs) / "audit.log"
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(audit_file, "a") as f:
            f.write(json.dumps(audit_entry, default=str) + "\n")
            
    except Exception as e:
        logger.error(f"Error logging decision: {e}")


async def log_model_prediction(
    model_name: str,
    model_version: str,
    input_features: Dict[str, Any],
    prediction: Any,
    prediction_probability: Optional[float] = None,
    feature_importance: Optional[Dict[str, float]] = None,
    request_id: Optional[str] = None
) -> None:
    """
    Log model predictions for monitoring and debugging.
    
    Args:
        model_name: Name of the model
        model_version: Version of the model
        input_features: Input features used for prediction
        prediction: Model prediction
        prediction_probability: Prediction probability/confidence
        feature_importance: Feature importance scores
        request_id: Request identifier for tracing
    """
    try:
        # Create prediction log entry
        prediction_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "model_version": model_version,
            "request_id": request_id,
            "input_features": input_features,
            "prediction": prediction,
            "prediction_probability": prediction_probability,
            "feature_importance": feature_importance,
            "environment": settings.environment
        }
        
        # Log to application logs
        logger.info(f"Model Prediction: {json.dumps(prediction_entry, default=str)}")
        
        # In a production environment, you would also:
        # - Store in model monitoring system
        # - Track model drift
        # - Monitor prediction accuracy
        # - Store in feature store for retraining
        
        # Save to prediction log file (for development)
        prediction_file = Path(settings.paths.logs) / "predictions.log"
        prediction_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(prediction_file, "a") as f:
            f.write(json.dumps(prediction_entry, default=str) + "\n")
            
    except Exception as e:
        logger.error(f"Error logging model prediction: {e}")


async def log_data_quality_issue(
    data_source: str,
    issue_type: str,
    issue_description: str,
    affected_records: Optional[int] = None,
    severity: str = "MEDIUM",
    request_id: Optional[str] = None
) -> None:
    """
    Log data quality issues for monitoring and alerting.
    
    Args:
        data_source: Source of the data
        issue_type: Type of data quality issue
        issue_description: Description of the issue
        affected_records: Number of affected records
        severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
        request_id: Request identifier for tracing
    """
    try:
        # Create data quality log entry
        quality_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "data_source": data_source,
            "issue_type": issue_type,
            "issue_description": issue_description,
            "affected_records": affected_records,
            "severity": severity,
            "request_id": request_id,
            "environment": settings.environment
        }
        
        # Log to application logs
        log_level = logging.ERROR if severity in ["HIGH", "CRITICAL"] else logging.WARNING
        logger.log(log_level, f"Data Quality Issue: {json.dumps(quality_entry, default=str)}")
        
        # In a production environment, you would also:
        # - Send alerts for high severity issues
        # - Store in data quality monitoring system
        # - Trigger data quality workflows
        # - Update data quality metrics
        
        # Save to data quality log file (for development)
        quality_file = Path(settings.paths.logs) / "data_quality.log"
        quality_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(quality_file, "a") as f:
            f.write(json.dumps(quality_entry, default=str) + "\n")
            
    except Exception as e:
        logger.error(f"Error logging data quality issue: {e}")


async def log_performance_metric(
    metric_name: str,
    metric_value: float,
    metric_unit: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    request_id: Optional[str] = None
) -> None:
    """
    Log performance metrics for monitoring.
    
    Args:
        metric_name: Name of the metric
        metric_value: Value of the metric
        metric_unit: Unit of the metric
        tags: Additional tags for the metric
        request_id: Request identifier for tracing
    """
    try:
        # Create performance metric entry
        metric_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "metric_name": metric_name,
            "metric_value": metric_value,
            "metric_unit": metric_unit,
            "tags": tags or {},
            "request_id": request_id,
            "environment": settings.environment
        }
        
        # Log to application logs
        logger.info(f"Performance Metric: {json.dumps(metric_entry, default=str)}")
        
        # In a production environment, you would also:
        # - Send to metrics system (e.g., Prometheus, DataDog)
        # - Store in time-series database
        # - Create dashboards and alerts
        # - Track performance trends
        
        # Save to metrics log file (for development)
        metrics_file = Path(settings.paths.logs) / "metrics.log"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metric_entry, default=str) + "\n")
            
    except Exception as e:
        logger.error(f"Error logging performance metric: {e}")


async def log_security_event(
    event_type: str,
    event_description: str,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    severity: str = "MEDIUM",
    request_id: Optional[str] = None
) -> None:
    """
    Log security events for monitoring and alerting.
    
    Args:
        event_type: Type of security event
        event_description: Description of the event
        user_id: User identifier involved
        ip_address: IP address involved
        severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
        request_id: Request identifier for tracing
    """
    try:
        # Create security event entry
        security_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "event_description": event_description,
            "user_id": user_id,
            "ip_address": ip_address,
            "severity": severity,
            "request_id": request_id,
            "environment": settings.environment
        }
        
        # Log to application logs
        log_level = logging.ERROR if severity in ["HIGH", "CRITICAL"] else logging.WARNING
        logger.log(log_level, f"Security Event: {json.dumps(security_entry, default=str)}")
        
        # In a production environment, you would also:
        # - Send to security monitoring system (e.g., SIEM)
        # - Trigger security alerts
        # - Store in security audit log
        # - Update security metrics
        
        # Save to security log file (for development)
        security_file = Path(settings.paths.logs) / "security.log"
        security_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(security_file, "a") as f:
            f.write(json.dumps(security_entry, default=str) + "\n")
            
    except Exception as e:
        logger.error(f"Error logging security event: {e}")


def get_audit_summary(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    decision_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get summary of audit logs for reporting.
    
    Args:
        start_date: Start date for filtering
        end_date: End date for filtering
        decision_type: Filter by decision type
        
    Returns:
        Audit summary statistics
    """
    try:
        # In a real application, you would query the database
        # For now, return mock summary
        summary = {
            "total_decisions": 1250,
            "decision_types": {
                "credit_score": 450,
                "fraud_detection": 600,
                "portfolio_risk": 200
            },
            "average_confidence": 0.85,
            "high_risk_decisions": 45,
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting audit summary: {e}")
        return {"error": str(e)}
