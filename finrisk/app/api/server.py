"""
FastAPI server for FinRisk application.
Provides REST API endpoints for credit scoring, fraud detection, and portfolio analysis.
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# FinRisk modules
from app.config import get_settings
from app.infra.db import check_database_connection
from app.infra.cache import get_cache_stats
from app.api.routes import credit, fraud, portfolio, health
from app.api.deps import get_current_user, log_api_request
from app.monitoring.audit import log_decision

# Configure FastAPI app
settings = get_settings()

app = FastAPI(
    title="FinRisk API",
    description="Credit Risk Assessment & Fraud Detection Engine",
    version="1.0.0",
    docs_url="/docs" if settings.api.debug else None,
    redoc_url="/redoc" if settings.api.debug else None
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.api.debug else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.api.debug else ["yourdomain.com", "api.yourdomain.com"]
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all API requests for monitoring and audit."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log request details
    await log_api_request(
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        processing_time=process_time,
        user_agent=request.headers.get("user-agent"),
        client_ip=request.client.host
    )
    
    # Add response headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper error response format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, 'request_id', None),
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with proper error response format."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.api.debug else "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, 'request_id', None),
            "path": request.url.path
        }
    )


# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(credit.router, prefix="/api/v1/credit", tags=["Credit Risk"])
app.include_router(fraud.router, prefix="/api/v1/fraud", tags=["Fraud Detection"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["Portfolio Analysis"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "FinRisk API",
        "version": "1.0.0",
        "description": "Credit Risk Assessment & Fraud Detection Engine",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "health": "/health",
            "credit": "/api/v1/credit",
            "fraud": "/api/v1/fraud",
            "portfolio": "/api/v1/portfolio",
            "docs": "/docs" if settings.api.debug else "disabled"
        }
    }


# System status endpoint
@app.get("/status")
async def system_status():
    """Get comprehensive system status."""
    try:
        # Check database connection
        db_status = check_database_connection()
        
        # Check cache status
        cache_stats = get_cache_stats()
        
        # Overall system health
        overall_status = "healthy" if db_status and cache_stats.get("status") == "healthy" else "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "database": {
                    "status": "healthy" if db_status else "unhealthy",
                    "connected": db_status
                },
                "cache": cache_stats,
                "api": {
                    "status": "healthy",
                    "version": "1.0.0"
                }
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


# Metrics endpoint for monitoring
@app.get("/metrics")
async def metrics():
    """Get application metrics for monitoring systems."""
    # This would typically integrate with Prometheus or similar
    return {
        "metrics": {
            "requests_total": "counter",
            "request_duration_seconds": "histogram",
            "active_connections": "gauge",
            "database_connections": "gauge",
            "cache_hit_rate": "gauge"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("🚀 Starting FinRisk API...")
    
    # Verify database connection
    if not check_database_connection():
        print("❌ Database connection failed!")
        raise RuntimeError("Database connection failed")
    
    print("✅ Database connection successful")
    
    # Initialize cache
    cache_stats = get_cache_stats()
    if cache_stats.get("status") == "healthy":
        print("✅ Cache connection successful")
    else:
        print("⚠️  Cache connection failed - running without cache")
    
    print("🎯 FinRisk API is ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("🛑 Shutting down FinRisk API...")
    print("✅ Shutdown complete")


# Development server function
def run_development_server():
    """Run development server with hot reload."""
    uvicorn.run(
        "app.api.server:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=True,
        log_level="info"
    )


# Production server function
def run_production_server():
    """Run production server."""
    uvicorn.run(
        "app.api.server:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        log_level="warning",
        access_log=True
    )


if __name__ == "__main__":
    if settings.environment == "development":
        run_development_server()
    else:
        run_production_server()
