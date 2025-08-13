#!/usr/bin/env python3
"""
FastAPI startup script for FinRisk application.
Provides easy startup with proper configuration and logging.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.config import get_settings
from app.api.server import app


def main():
    """Main function to run the FastAPI server."""
    settings = get_settings()
    
    print("🚀 Starting FinRisk FastAPI Server...")
    print(f"📊 Environment: {settings.environment}")
    print(f"🌐 Host: {settings.api.host}")
    print(f"🔌 Port: {settings.api.port}")
    print(f"🔧 Debug: {settings.api.debug}")
    print(f"👥 Workers: {settings.api.workers}")
    
    # Set environment variables
    os.environ.setdefault("ENVIRONMENT", settings.environment)
    os.environ.setdefault("API_HOST", settings.api.host)
    os.environ.setdefault("API_PORT", str(settings.api.port))
    
    # Configure uvicorn settings
    uvicorn_config = {
        "app": "app.api.server:app",
        "host": settings.api.host,
        "port": settings.api.port,
        "reload": settings.api.debug,
        "log_level": "info" if settings.api.debug else "warning",
        "access_log": True,
        "workers": settings.api.workers if not settings.api.debug else 1,
    }
    
    if settings.api.debug:
        print("🔍 Development mode enabled - hot reload active")
        print("📚 API Documentation available at:")
        print(f"   - Swagger UI: http://{settings.api.host}:{settings.api.port}/docs")
        print(f"   - ReDoc: http://{settings.api.host}:{settings.api.port}/redoc")
    else:
        print("🏭 Production mode enabled")
        print("📚 API Documentation disabled in production")
    
    print("\n🎯 Available endpoints:")
    print("   - Health check: /health")
    print("   - System status: /status")
    print("   - Credit risk: /api/v1/credit")
    print("   - Fraud detection: /api/v1/fraud")
    print("   - Portfolio analysis: /api/v1/portfolio")
    print("   - Metrics: /metrics")
    
    print(f"\n🚀 Server starting on http://{settings.api.host}:{settings.api.port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
