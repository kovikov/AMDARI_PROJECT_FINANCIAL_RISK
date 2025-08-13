"""
Credit risk assessment endpoints for FinRisk API.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

# FinRisk modules
from app.config import get_settings
from app.api.deps import get_current_user, log_decision_maker, rate_limit
from app.models.credit_risk_trainer import CreditRiskModelTrainer
from app.schemas.credit_risk import (
    CreditApplication, CreditScoreRequest, CreditScoreResponse,
    RiskAssessment, ApplicationStatus
)

# Configure router
router = APIRouter()
settings = get_settings()


# Request/Response models
class CreditApplicationRequest(BaseModel):
    """Credit application request model."""
    customer_id: str = Field(..., description="Customer identifier")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_purpose: str = Field(..., description="Purpose of the loan")
    employment_status: str = Field(..., description="Employment status")
    annual_income: float = Field(..., gt=0, description="Annual income")
    credit_score: Optional[int] = Field(None, ge=300, le=850, description="Credit score")
    existing_debt: float = Field(0, ge=0, description="Existing debt amount")
    collateral_value: Optional[float] = Field(None, ge=0, description="Collateral value")


class CreditScoreResponse(BaseModel):
    """Credit score response model."""
    customer_id: str
    credit_score: float
    risk_level: str
    confidence: float
    factors: List[Dict[str, Any]]
    recommendation: str
    timestamp: datetime


class RiskAssessmentResponse(BaseModel):
    """Risk assessment response model."""
    application_id: str
    customer_id: str
    risk_score: float
    risk_level: str
    approval_probability: float
    recommended_amount: Optional[float]
    conditions: List[str]
    timestamp: datetime


@router.post("/score", response_model=CreditScoreResponse)
async def calculate_credit_score(
    request: CreditScoreRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    log_decision = Depends(log_decision_maker("credit_score", "xgboost_ensemble"))
):
    """
    Calculate credit score for a customer.
    
    Args:
        request: Credit score request data
        current_user: Current authenticated user
        log_decision: Decision logging function
        
    Returns:
        Credit score response with risk assessment
    """
    try:
        # Initialize credit risk trainer
        trainer = CreditRiskModelTrainer()
        
        # Calculate credit score
        score_result = trainer.calculate_credit_score(
            customer_id=request.customer_id,
            features=request.features
        )
        
        # Log decision
        await log_decision(
            input_data=request.dict(),
            output_data=score_result,
            confidence=score_result.get("confidence", 0.8)
        )
        
        return CreditScoreResponse(
            customer_id=request.customer_id,
            credit_score=score_result["credit_score"],
            risk_level=score_result["risk_level"],
            confidence=score_result.get("confidence", 0.8),
            factors=score_result.get("factors", []),
            recommendation=score_result.get("recommendation", "Standard processing"),
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating credit score: {str(e)}"
        )


@router.post("/assess", response_model=RiskAssessmentResponse)
async def assess_credit_risk(
    application: CreditApplicationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    log_decision = Depends(log_decision_maker("risk_assessment", "ensemble_model"))
):
    """
    Assess credit risk for a loan application.
    
    Args:
        application: Credit application data
        current_user: Current authenticated user
        log_decision: Decision logging function
        
    Returns:
        Risk assessment response
    """
    try:
        # Initialize credit risk trainer
        trainer = CreditRiskModelTrainer()
        
        # Convert to application schema
        app_data = CreditApplication(
            customer_id=application.customer_id,
            loan_amount=application.loan_amount,
            loan_purpose=application.loan_purpose,
            employment_status=application.employment_status,
            annual_income=application.annual_income,
            credit_score=application.credit_score,
            existing_debt=application.existing_debt,
            collateral_value=application.collateral_value
        )
        
        # Assess risk
        risk_result = trainer.assess_credit_risk(app_data)
        
        # Log decision
        await log_decision(
            input_data=application.dict(),
            output_data=risk_result,
            confidence=risk_result.get("confidence", 0.8)
        )
        
        return RiskAssessmentResponse(
            application_id=risk_result["application_id"],
            customer_id=application.customer_id,
            risk_score=risk_result["risk_score"],
            risk_level=risk_result["risk_level"],
            approval_probability=risk_result["approval_probability"],
            recommended_amount=risk_result.get("recommended_amount"),
            conditions=risk_result.get("conditions", []),
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error assessing credit risk: {str(e)}"
        )


@router.post("/apply")
async def submit_credit_application(
    application: CreditApplicationRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    log_decision = Depends(log_decision_maker("application_submission", "manual_review"))
):
    """
    Submit a credit application for processing.
    
    Args:
        application: Credit application data
        background_tasks: Background tasks for async processing
        current_user: Current authenticated user
        log_decision: Decision logging function
        
    Returns:
        Application submission response
    """
    try:
        # Generate application ID
        application_id = f"APP_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{application.customer_id}"
        
        # Initial assessment
        trainer = CreditRiskModelTrainer()
        app_data = CreditApplication(
            customer_id=application.customer_id,
            loan_amount=application.loan_amount,
            loan_purpose=application.loan_purpose,
            employment_status=application.employment_status,
            annual_income=application.annual_income,
            credit_score=application.credit_score,
            existing_debt=application.existing_debt,
            collateral_value=application.collateral_value
        )
        
        # Quick risk assessment
        risk_result = trainer.assess_credit_risk(app_data)
        
        # Determine initial status
        if risk_result["risk_level"] == "LOW" and risk_result["approval_probability"] > 0.8:
            status = ApplicationStatus.APPROVED
        elif risk_result["risk_level"] == "HIGH" and risk_result["approval_probability"] < 0.3:
            status = ApplicationStatus.REJECTED
        else:
            status = ApplicationStatus.PENDING_REVIEW
        
        # Add background task for detailed processing
        background_tasks.add_task(
            process_credit_application,
            application_id,
            application.dict(),
            risk_result
        )
        
        # Log decision
        await log_decision(
            input_data=application.dict(),
            output_data={
                "application_id": application_id,
                "status": status,
                "risk_level": risk_result["risk_level"],
                "approval_probability": risk_result["approval_probability"]
            },
            confidence=risk_result.get("confidence", 0.8)
        )
        
        return {
            "application_id": application_id,
            "status": status,
            "message": "Application submitted successfully",
            "estimated_processing_time": "24-48 hours" if status == ApplicationStatus.PENDING_REVIEW else "Immediate",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting application: {str(e)}"
        )


@router.get("/applications/{customer_id}")
async def get_customer_applications(
    customer_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    limit: int = Query(10, ge=1, le=100, description="Number of applications to return")
):
    """
    Get credit applications for a customer.
    
    Args:
        customer_id: Customer identifier
        current_user: Current authenticated user
        limit: Maximum number of applications to return
        
    Returns:
        List of customer applications
    """
    try:
        # In a real application, you would query the database
        # For now, return mock data
        applications = [
            {
                "application_id": f"APP_{i:06d}",
                "customer_id": customer_id,
                "loan_amount": 50000 + (i * 1000),
                "status": "APPROVED" if i % 3 == 0 else "PENDING_REVIEW" if i % 3 == 1 else "REJECTED",
                "submitted_date": datetime.utcnow().isoformat(),
                "risk_score": 0.7 - (i * 0.1)
            }
            for i in range(1, min(limit + 1, 6))
        ]
        
        return {
            "customer_id": customer_id,
            "applications": applications,
            "total_count": len(applications),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving applications: {str(e)}"
        )


@router.get("/status/{application_id}")
async def get_application_status(
    application_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get status of a credit application.
    
    Args:
        application_id: Application identifier
        current_user: Current authenticated user
        
    Returns:
        Application status information
    """
    try:
        # In a real application, you would query the database
        # For now, return mock data
        status_data = {
            "application_id": application_id,
            "status": "APPROVED",
            "risk_score": 0.65,
            "approval_probability": 0.85,
            "recommended_amount": 45000,
            "conditions": ["Income verification required", "Collateral documentation needed"],
            "last_updated": datetime.utcnow().isoformat(),
            "estimated_completion": "2024-01-15T10:00:00Z"
        }
        
        return status_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving application status: {str(e)}"
        )


@router.get("/models/status")
async def get_model_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get status of credit risk models.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Model status information
    """
    try:
        trainer = CreditRiskModelTrainer()
        
        return {
            "models": trainer.get_model_status(),
            "last_training": "2024-01-10T15:30:00Z",
            "next_training": "2024-01-17T15:30:00Z",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model status: {str(e)}"
        )


# Background task function
async def process_credit_application(
    application_id: str,
    application_data: Dict[str, Any],
    risk_result: Dict[str, Any]
):
    """
    Process credit application in background.
    
    Args:
        application_id: Application identifier
        application_data: Application data
        risk_result: Risk assessment result
    """
    try:
        # Simulate processing time
        import asyncio
        await asyncio.sleep(5)
        
        # In a real application, you would:
        # - Store application in database
        # - Send notifications
        # - Update status
        # - Trigger additional workflows
        
        print(f"Processed application {application_id}")
        
    except Exception as e:
        print(f"Error processing application {application_id}: {e}")
