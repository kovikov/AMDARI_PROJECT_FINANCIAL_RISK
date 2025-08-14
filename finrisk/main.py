# main.py - FinRisk FastAPI Application

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinRisk API",
    description="Credit Risk Assessment & Fraud Detection Engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for requests/responses
class CreditScoringRequest(BaseModel):
    customer_id: str
    age: int
    annual_income: float
    employment_status: str
    loan_amount: int
    credit_score: int
    debt_to_income_ratio: float

class CreditScoringResponse(BaseModel):
    customer_id: str
    prediction_id: str
    risk_score: float
    probability_of_default: float
    risk_grade: str
    recommended_decision: str
    model_version: str
    prediction_timestamp: str

class FraudDetectionRequest(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float
    merchant_category: str
    location: str
    hour: int
    is_weekend: bool

class FraudDetectionResponse(BaseModel):
    transaction_id: str
    customer_id: str
    fraud_probability: float
    risk_level: str
    recommended_action: str
    model_version: str
    processing_time_ms: float

# Simple in-memory models (replace with actual trained models)
class SimpleCreditModel:
    def __init__(self):
        self.version = "1.0"
        logger.info("Credit model initialized")
    
    def predict_proba(self, features: Dict[str, Any]) -> float:
        # Simple rule-based model for demo
        score = 0.5  # Base probability
        
        # Age factor
        if features['age'] < 25:
            score += 0.1
        elif features['age'] > 60:
            score -= 0.1
        
        # Credit score factor
        if features['credit_score'] < 600:
            score += 0.3
        elif features['credit_score'] > 750:
            score -= 0.2
        
        # DTI factor
        if features['debt_to_income_ratio'] > 0.4:
            score += 0.2
        
        # Income factor
        if features['annual_income'] < 30000:
            score += 0.1
        
        return max(0.0, min(1.0, score))

class SimpleFraudModel:
    def __init__(self):
        self.version = "1.0"
        logger.info("Fraud model initialized")
    
    def predict_proba(self, features: Dict[str, Any]) -> float:
        # Simple rule-based model for demo
        score = 0.01  # Base fraud probability
        
        # Amount factor
        if features['amount'] > 5000:
            score += 0.3
        elif features['amount'] > 1000:
            score += 0.1
        
        # Time factor
        if features['hour'] < 6 or features['hour'] > 22:
            score += 0.2
        
        # Weekend factor
        if features['is_weekend']:
            score += 0.1
        
        # High-risk merchants
        if features['merchant_category'] in ['Online Shopping', 'ATM']:
            score += 0.2
        
        return max(0.0, min(1.0, score))

# Initialize models
credit_model = SimpleCreditModel()
fraud_model = SimpleFraudModel()

# Helper functions
def generate_prediction_id() -> str:
    return f"PRED_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"

def calculate_risk_grade(probability: float) -> str:
    if probability >= 0.8:
        return "F"
    elif probability >= 0.6:
        return "E"
    elif probability >= 0.4:
        return "D"
    elif probability >= 0.2:
        return "C"
    elif probability >= 0.1:
        return "B"
    else:
        return "A"

def get_recommended_decision(probability: float, model_type: str) -> str:
    if model_type == "credit":
        return "Decline" if probability > 0.5 else "Approve"
    else:  # fraud
        return "Block" if probability > 0.5 else "Allow"

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "FinRisk API",
        "version": "1.0.0",
        "status": "running",
        "message": "Welcome to FinRisk Credit Risk & Fraud Detection API",
        "endpoints": {
            "credit_scoring": "/api/v1/credit/score",
            "fraud_detection": "/api/v1/fraud/detect",
            "batch_credit": "/api/v1/credit/batch",
            "models_status": "/api/v1/models/status",
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "FinRisk API",
        "version": "1.0.0",
        "models": {
            "credit_model": "active",
            "fraud_model": "active"
        }
    }

@app.post("/api/v1/credit/score", response_model=CreditScoringResponse)
async def score_credit_application(request: CreditScoringRequest):
    """Score a credit application for default risk."""
    try:
        start_time = datetime.now()
        
        # Prepare features
        features = {
            'age': request.age,
            'annual_income': request.annual_income,
            'credit_score': request.credit_score,
            'debt_to_income_ratio': request.debt_to_income_ratio,
            'loan_amount': request.loan_amount
        }
        
        # Get prediction
        probability = credit_model.predict_proba(features)
        risk_score = probability * 1000  # Convert to 0-1000 scale
        risk_grade = calculate_risk_grade(probability)
        recommended_decision = get_recommended_decision(probability, "credit")
        
        # Create response
        response = CreditScoringResponse(
            customer_id=request.customer_id,
            prediction_id=generate_prediction_id(),
            risk_score=round(risk_score, 2),
            probability_of_default=round(probability, 4),
            risk_grade=risk_grade,
            recommended_decision=recommended_decision,
            model_version=credit_model.version,
            prediction_timestamp=datetime.utcnow().isoformat()
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Credit scoring completed in {processing_time:.2f}ms for customer {request.customer_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Credit scoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Credit scoring failed: {str(e)}")

@app.post("/api/v1/fraud/detect", response_model=FraudDetectionResponse)
async def detect_fraud(request: FraudDetectionRequest):
    """Detect fraud in a transaction."""
    try:
        start_time = datetime.now()
        
        # Prepare features
        features = {
            'amount': request.amount,
            'merchant_category': request.merchant_category,
            'location': request.location,
            'hour': request.hour,
            'is_weekend': request.is_weekend
        }
        
        # Get prediction
        fraud_probability = fraud_model.predict_proba(features)
        
        # Determine risk level
        if fraud_probability >= 0.7:
            risk_level = "HIGH"
        elif fraud_probability >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        recommended_action = get_recommended_decision(fraud_probability, "fraud")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Create response
        response = FraudDetectionResponse(
            transaction_id=request.transaction_id,
            customer_id=request.customer_id,
            fraud_probability=round(fraud_probability, 4),
            risk_level=risk_level,
            recommended_action=recommended_action,
            model_version=fraud_model.version,
            processing_time_ms=round(processing_time, 2)
        )
        
        logger.info(f"Fraud detection completed in {processing_time:.2f}ms for transaction {request.transaction_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Fraud detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Fraud detection failed: {str(e)}")

@app.get("/api/v1/models/status")
async def get_model_status():
    """Get status of all models."""
    return {
        "models": {
            "credit_risk": {
                "version": credit_model.version,
                "status": "active",
                "type": "rule_based",
                "description": "Simple rule-based credit risk model"
            },
            "fraud_detection": {
                "version": fraud_model.version,
                "status": "active", 
                "type": "rule_based",
                "description": "Simple rule-based fraud detection model"
            }
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/metrics")
async def get_metrics():
    """Get basic API metrics."""
    return {
        "metrics": {
            "total_requests": "Available in production",
            "avg_response_time": "Available in production",
            "error_rate": "Available in production",
            "uptime": "Available in production"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# Batch processing endpoints
@app.post("/api/v1/credit/batch")
async def batch_credit_scoring(requests: List[CreditScoringRequest]):
    """Process multiple credit applications."""
    results = []
    start_time = datetime.now()
    
    for req in requests:
        try:
            result = await score_credit_application(req)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process credit request for {req.customer_id}: {e}")
            continue
    
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return {
        "batch_id": generate_prediction_id(),
        "total_requests": len(requests),
        "successful_results": len(results),
        "failed_requests": len(requests) - len(results),
        "total_processing_time_ms": round(processing_time, 2),
        "avg_processing_time_ms": round(processing_time / len(requests), 2) if requests else 0,
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 FinRisk API is starting up...")
    logger.info("✅ Models initialized and ready")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FinRisk API server...")
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)
