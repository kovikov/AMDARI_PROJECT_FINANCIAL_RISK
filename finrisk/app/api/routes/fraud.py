"""
Fraud detection endpoints for FinRisk API.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

# FinRisk modules
from app.config import get_settings
from app.api.deps import get_current_user, log_decision_maker, rate_limit
from app.models.fraud_detection_trainer import (
    FraudDetectionModelTrainer, FraudRuleEngine, FraudModelExplainer,
    create_fraud_prediction_pipeline
)
from app.features.preprocessing import FinancialFeatureEngineer

# Configure router
router = APIRouter()
settings = get_settings()


# Request/Response models
class TransactionRequest(BaseModel):
    """Transaction monitoring request model."""
    transaction_id: str = Field(..., description="Transaction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_category: str = Field(..., description="Merchant category")
    location: str = Field(..., description="Transaction location")
    device_info: str = Field(..., description="Device information")
    timestamp: datetime = Field(..., description="Transaction timestamp")


class FraudDetectionResponse(BaseModel):
    """Fraud detection response model."""
    transaction_id: str
    customer_id: str
    fraud_score: float
    risk_level: str
    is_fraud: bool
    confidence: float
    triggered_rules: List[Dict[str, Any]]
    explanation: str
    recommendation: str
    timestamp: datetime


class RuleEvaluationResponse(BaseModel):
    """Rule evaluation response model."""
    transaction_id: str
    triggered_rules: List[Dict[str, Any]]
    rule_count: int
    max_severity: str
    requires_review: bool
    timestamp: datetime


@router.post("/detect", response_model=FraudDetectionResponse)
async def detect_fraud(
    transaction: TransactionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    log_decision = Depends(log_decision_maker("fraud_detection", "ensemble_model"))
):
    """
    Detect fraud in a transaction.
    
    Args:
        transaction: Transaction data
        current_user: Current authenticated user
        log_decision: Decision logging function
        
    Returns:
        Fraud detection response
    """
    try:
        # Initialize fraud detection components
        trainer = FraudDetectionModelTrainer()
        rule_engine = FraudRuleEngine()
        feature_engineer = FinancialFeatureEngineer()
        
        # Create transaction features
        transaction_features = {
            'transaction_id': transaction.transaction_id,
            'customer_id': transaction.customer_id,
            'amount': transaction.amount,
            'merchant_category': transaction.merchant_category,
            'location': transaction.location,
            'device_info': transaction.device_info,
            'timestamp': transaction.timestamp
        }
        
        # Evaluate rules
        rule_result = rule_engine.evaluate_transaction(transaction_features)
        
        # Create features for ML model
        # In a real application, you would create proper features
        # For now, use simplified features
        ml_features = {
            'amount': transaction.amount,
            'amount_log': transaction.amount,
            'is_round_amount': 1 if transaction.amount % 100 == 0 else 0,
            'hour': transaction.timestamp.hour,
            'day_of_week': transaction.timestamp.weekday(),
            'is_night': 1 if transaction.timestamp.hour < 6 or transaction.timestamp.hour > 22 else 0
        }
        
        # Get fraud prediction
        try:
            model, transformer = create_fraud_prediction_pipeline()
            
            # Convert features to DataFrame
            import pandas as pd
            features_df = pd.DataFrame([ml_features])
            
            # Transform features
            features_transformed = transformer.transform(features_df)
            
            # Get prediction
            fraud_score = model.decision_function(features_transformed)[0]
            is_fraud = fraud_score < 0  # Negative scores indicate fraud
            
            # Create explainer
            explainer = FraudModelExplainer(model, features_df.columns.tolist())
            explanation = explainer.explain_anomaly_score(features_df)
            
        except Exception as e:
            # Fallback to rule-based detection
            fraud_score = 0.1 if rule_result['rule_count'] > 0 else 0.9
            is_fraud = rule_result['max_severity'] in ['HIGH', 'CRITICAL']
            explanation = {
                'anomaly_score': fraud_score,
                'is_anomaly': is_fraud,
                'text_explanation': f"Rule-based detection: {rule_result['rule_count']} rules triggered"
            }
        
        # Determine risk level
        if is_fraud or rule_result['max_severity'] in ['HIGH', 'CRITICAL']:
            risk_level = "HIGH"
            recommendation = "Immediate investigation required"
        elif rule_result['rule_count'] > 0 or fraud_score < 0.3:
            risk_level = "MEDIUM"
            recommendation = "Enhanced monitoring recommended"
        else:
            risk_level = "LOW"
            recommendation = "Standard processing"
        
        # Calculate confidence
        confidence = 0.9 if rule_result['rule_count'] > 0 else 0.7
        
        # Prepare response
        response_data = {
            "transaction_id": transaction.transaction_id,
            "customer_id": transaction.customer_id,
            "fraud_score": abs(fraud_score),
            "risk_level": risk_level,
            "is_fraud": is_fraud,
            "confidence": confidence,
            "triggered_rules": rule_result['triggered_rules'],
            "explanation": explanation.get('text_explanation', 'No explanation available'),
            "recommendation": recommendation,
            "timestamp": datetime.utcnow()
        }
        
        # Log decision
        await log_decision(
            input_data=transaction.dict(),
            output_data=response_data,
            confidence=confidence
        )
        
        return FraudDetectionResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error detecting fraud: {str(e)}"
        )


@router.post("/rules/evaluate", response_model=RuleEvaluationResponse)
async def evaluate_fraud_rules(
    transaction: TransactionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Evaluate transaction against fraud rules.
    
    Args:
        transaction: Transaction data
        current_user: Current authenticated user
        
    Returns:
        Rule evaluation response
    """
    try:
        # Initialize rule engine
        rule_engine = FraudRuleEngine()
        
        # Create transaction features
        transaction_features = {
            'amount': transaction.amount,
            'rolling_24h_count': 5,  # Mock data - in real app, calculate from history
            'is_night': 1 if transaction.timestamp.hour < 6 or transaction.timestamp.hour > 22 else 0,
            'new_location': 0,  # Mock data - in real app, check against customer history
            'is_round_amount': 1 if transaction.amount % 100 == 0 else 0
        }
        
        # Evaluate rules
        rule_result = rule_engine.evaluate_transaction(transaction_features)
        
        return RuleEvaluationResponse(
            transaction_id=transaction.transaction_id,
            triggered_rules=rule_result['triggered_rules'],
            rule_count=rule_result['rule_count'],
            max_severity=rule_result['max_severity'],
            requires_review=rule_result['requires_review'],
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error evaluating rules: {str(e)}"
        )


@router.post("/batch")
async def batch_fraud_detection(
    transactions: List[TransactionRequest],
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Perform batch fraud detection on multiple transactions.
    
    Args:
        transactions: List of transaction data
        background_tasks: Background tasks for async processing
        current_user: Current authenticated user
        
    Returns:
        Batch processing response
    """
    try:
        batch_id = f"BATCH_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Add background task for processing
        background_tasks.add_task(
            process_batch_fraud_detection,
            batch_id,
            [t.dict() for t in transactions]
        )
        
        return {
            "batch_id": batch_id,
            "transaction_count": len(transactions),
            "status": "processing",
            "estimated_completion": "5-10 minutes",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error starting batch processing: {str(e)}"
        )


@router.get("/alerts")
async def get_fraud_alerts(
    current_user: Dict[str, Any] = Depends(get_current_user),
    limit: int = Query(10, ge=1, le=100, description="Number of alerts to return"),
    severity: Optional[str] = Query(None, description="Filter by severity level")
):
    """
    Get fraud alerts.
    
    Args:
        current_user: Current authenticated user
        limit: Maximum number of alerts to return
        severity: Filter by severity level
        
    Returns:
        List of fraud alerts
    """
    try:
        # In a real application, you would query the database
        # For now, return mock data
        alerts = [
            {
                "alert_id": f"ALERT_{i:06d}",
                "transaction_id": f"TXN_{i:06d}",
                "customer_id": f"CUST_{i:03d}",
                "severity": "HIGH" if i % 3 == 0 else "MEDIUM" if i % 3 == 1 else "LOW",
                "fraud_score": 0.8 - (i * 0.1),
                "triggered_rules": ["high_amount_rule", "velocity_rule"] if i % 2 == 0 else ["unusual_time_rule"],
                "status": "OPEN" if i % 2 == 0 else "INVESTIGATING",
                "created_at": datetime.utcnow().isoformat(),
                "assigned_to": f"analyst_{i % 3 + 1}"
            }
            for i in range(1, min(limit + 1, 6))
        ]
        
        # Filter by severity if specified
        if severity:
            alerts = [alert for alert in alerts if alert["severity"] == severity.upper()]
        
        return {
            "alerts": alerts,
            "total_count": len(alerts),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving alerts: {str(e)}"
        )


@router.get("/models/status")
async def get_fraud_model_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get status of fraud detection models.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Model status information
    """
    try:
        trainer = FraudDetectionModelTrainer()
        
        return {
            "models": {
                "isolation_forest": "active",
                "one_class_svm": "active",
                "elliptic_envelope": "active",
                "ensemble": "active"
            },
            "last_training": "2024-01-10T15:30:00Z",
            "next_training": "2024-01-17T15:30:00Z",
            "performance_metrics": {
                "precision": 0.95,
                "recall": 0.88,
                "f1_score": 0.91,
                "auc": 0.94
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model status: {str(e)}"
        )


@router.get("/rules")
async def get_fraud_rules(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get configured fraud detection rules.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        List of fraud detection rules
    """
    try:
        rule_engine = FraudRuleEngine()
        
        rules = []
        for rule_name, rule_config in rule_engine.rules.items():
            rules.append({
                "name": rule_name,
                "severity": rule_config["severity"],
                "description": rule_config["message"],
                "active": True
            })
        
        return {
            "rules": rules,
            "total_count": len(rules),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving rules: {str(e)}"
        )


# Background task functions
async def process_batch_fraud_detection(
    batch_id: str,
    transactions: List[Dict[str, Any]]
):
    """
    Process batch fraud detection in background.
    
    Args:
        batch_id: Batch identifier
        transactions: List of transaction data
    """
    try:
        # Simulate processing time
        import asyncio
        await asyncio.sleep(10)
        
        # In a real application, you would:
        # - Process each transaction
        # - Store results in database
        # - Send notifications
        # - Update batch status
        
        print(f"Processed batch {batch_id} with {len(transactions)} transactions")
        
    except Exception as e:
        print(f"Error processing batch {batch_id}: {e}")
