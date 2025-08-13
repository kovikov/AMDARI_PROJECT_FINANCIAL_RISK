"""
API dependencies for FinRisk application.
Provides authentication, request logging, and other common dependencies.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# FinRisk modules
from app.config import get_settings
from app.infra.db import get_db_session
from app.infra.cache import get_cache_client

# Configure logging
logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)
settings = get_settings()


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """
    Validate JWT token and return current user.
    
    Args:
        credentials: HTTP Bearer token credentials
        
    Returns:
        User information dictionary
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # In a real application, you would validate the JWT token here
        # For now, we'll use a simple token validation
        token = credentials.credentials
        
        # Simple token validation (replace with proper JWT validation)
        if token == "demo_token" or settings.api.debug:
            return {
                "user_id": "demo_user",
                "username": "demo_user",
                "role": "user",
                "permissions": ["read", "write"]
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Get current user if authenticated, otherwise return None.
    
    Args:
        credentials: HTTP Bearer token credentials
        
    Returns:
        User information dictionary or None
    """
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


async def log_api_request(
    request_id: str,
    method: str,
    url: str,
    status_code: int,
    processing_time: float,
    user_agent: Optional[str] = None,
    client_ip: Optional[str] = None,
    user_id: Optional[str] = None
) -> None:
    """
    Log API request details for monitoring and audit.
    
    Args:
        request_id: Unique request identifier
        method: HTTP method
        url: Request URL
        status_code: Response status code
        processing_time: Request processing time in seconds
        user_agent: User agent string
        client_ip: Client IP address
        user_id: User identifier
    """
    log_entry = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "method": method,
        "url": url,
        "status_code": status_code,
        "processing_time": processing_time,
        "user_agent": user_agent,
        "client_ip": client_ip,
        "user_id": user_id
    }
    
    # Log to application logs
    logger.info(f"API Request: {log_entry}")
    
    # In a production environment, you might also:
    # - Store in database for audit trail
    # - Send to monitoring system
    # - Store in cache for rate limiting


async def get_db():
    """
    Database session dependency.
    
    Yields:
        Database session
    """
    async with get_db_session() as session:
        yield session


async def get_cache():
    """
    Cache client dependency.
    
    Returns:
        Cache client instance
    """
    return get_cache_client()


def require_permission(permission: str):
    """
    Decorator to require specific permission.
    
    Args:
        permission: Required permission
        
    Returns:
        Dependency function
    """
    async def permission_dependency(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        user_permissions = current_user.get("permissions", [])
        
        if permission not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        
        return current_user
    
    return permission_dependency


def rate_limit(max_requests: int = 100, window_seconds: int = 60):
    """
    Rate limiting decorator.
    
    Args:
        max_requests: Maximum requests per window
        window_seconds: Time window in seconds
        
    Returns:
        Dependency function
    """
    async def rate_limit_dependency(
        request: Request,
        cache = Depends(get_cache)
    ):
        # Get client identifier (IP or user ID)
        client_id = request.client.host
        
        # Check rate limit
        key = f"rate_limit:{client_id}"
        current_requests = await cache.get(key)
        
        if current_requests and int(current_requests) >= max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Increment request count
        await cache.incr(key)
        await cache.expire(key, window_seconds)
        
        return True
    
    return rate_limit_dependency


async def validate_request_id(request: Request) -> str:
    """
    Validate and return request ID from headers or generate new one.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Request ID string
    """
    request_id = request.headers.get("X-Request-ID")
    
    if not request_id:
        import uuid
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
    
    return request_id


def log_decision_maker(
    decision_type: str,
    model_name: Optional[str] = None
):
    """
    Decorator to log decision-making events.
    
    Args:
        decision_type: Type of decision (e.g., 'credit_score', 'fraud_detection')
        model_name: Name of the model used
        
    Returns:
        Dependency function
    """
    async def decision_logger(
        request: Request,
        current_user: Optional[Dict[str, Any]] = Depends(get_optional_user)
    ):
        async def log_decision_event(
            input_data: Dict[str, Any],
            output_data: Dict[str, Any],
            confidence: Optional[float] = None
        ):
            from app.monitoring.audit import log_decision
            
            await log_decision(
                decision_type=decision_type,
                model_name=model_name,
                input_data=input_data,
                output_data=output_data,
                confidence=confidence,
                user_id=current_user.get("user_id") if current_user else None,
                request_id=getattr(request.state, 'request_id', None)
            )
        
        return log_decision_event
    
    return decision_logger
