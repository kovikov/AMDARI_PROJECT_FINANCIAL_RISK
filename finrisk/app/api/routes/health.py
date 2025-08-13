"""
Health check endpoints for FinRisk API.
"""

from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

# FinRisk modules
from app.config import get_settings
from app.infra.db import check_database_connection, get_db_session
from app.infra.cache import get_cache_stats, get_cache_client
from app.api.deps import get_current_user

# Configure router
router = APIRouter()
settings = get_settings()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "FinRisk API",
        "version": "1.0.0"
    }


@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with component status.
    
    Returns:
        Detailed health status for all components
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "FinRisk API",
        "version": "1.0.0",
        "components": {}
    }
    
    # Check database
    try:
        db_healthy = check_database_connection()
        health_status["components"]["database"] = {
            "status": "healthy" if db_healthy else "unhealthy",
            "connected": db_healthy,
            "last_check": datetime.utcnow().isoformat()
        }
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.utcnow().isoformat()
        }
    
    # Check cache
    try:
        cache_stats = get_cache_stats()
        health_status["components"]["cache"] = cache_stats
    except Exception as e:
        health_status["components"]["cache"] = {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.utcnow().isoformat()
        }
    
    # Check API
    health_status["components"]["api"] = {
        "status": "healthy",
        "version": "1.0.0",
        "last_check": datetime.utcnow().isoformat()
    }
    
    # Determine overall status
    all_healthy = all(
        component.get("status") == "healthy" 
        for component in health_status["components"].values()
    )
    
    health_status["status"] = "healthy" if all_healthy else "unhealthy"
    
    return health_status


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check for Kubernetes/container orchestration.
    
    Returns:
        Readiness status
    """
    try:
        # Check critical dependencies
        db_ready = check_database_connection()
        cache_stats = get_cache_stats()
        cache_ready = cache_stats.get("status") == "healthy"
        
        if db_ready and cache_ready:
            return {
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "timestamp": datetime.utcnow().isoformat(),
                    "database": db_ready,
                    "cache": cache_ready
                }
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check for Kubernetes/container orchestration.
    
    Returns:
        Liveness status
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/database")
async def database_health_check() -> Dict[str, Any]:
    """
    Database-specific health check.
    
    Returns:
        Database health status
    """
    try:
        db_healthy = check_database_connection()
        
        if db_healthy:
            return {
                "status": "healthy",
                "database": "connected",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "database": "disconnected",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "database": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/cache")
async def cache_health_check() -> Dict[str, Any]:
    """
    Cache-specific health check.
    
    Returns:
        Cache health status
    """
    try:
        cache_stats = get_cache_stats()
        
        if cache_stats.get("status") == "healthy":
            return {
                "status": "healthy",
                "cache": cache_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "cache": cache_stats,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "cache": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/protected")
async def protected_health_check(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Protected health check endpoint requiring authentication.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Protected health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "user": current_user.get("username"),
        "message": "Protected endpoint accessible"
    }
