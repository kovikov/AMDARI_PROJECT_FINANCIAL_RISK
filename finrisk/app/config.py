"""
FinRisk Configuration
Handles all application configuration and environment variables.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    name: str = Field(default="finrisk_db", env="DB_NAME")
    user: str = Field(default="finrisk_user", env="DB_USER")
    password: str = Field(default="finrisk_pass", env="DB_PASSWORD")
    
    @property
    def url(self) -> str:
        """Get database URL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis configuration settings"""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    @property
    def url(self) -> str:
        """Get Redis URL"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"
    
    @property
    def connection_url(self) -> str:
        """Get Redis connection URL (alias for url)."""
        return self.url


class CacheSettings(BaseSettings):
    """Cache configuration settings."""
    
    prediction_ttl: int = Field(default=3600, env="CACHE_PREDICTION_TTL")      # 1 hour
    feature_ttl: int = Field(default=1800, env="CACHE_FEATURE_TTL")           # 30 minutes
    model_ttl: int = Field(default=86400, env="CACHE_MODEL_TTL")              # 24 hours
    dataframe_ttl: int = Field(default=7200, env="CACHE_DATAFRAME_TTL")       # 2 hours
    default_ttl: int = Field(default=1800, env="CACHE_DEFAULT_TTL")           # 30 minutes


class MLflowSettings(BaseSettings):
    """MLflow configuration settings"""
    tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    backend_store_uri: str = Field(default="sqlite:///mlflow.db", env="MLFLOW_BACKEND_STORE_URI")
    default_artifact_root: str = Field(default="./mlruns", env="MLFLOW_DEFAULT_ARTIFACT_ROOT")


class APISettings(BaseSettings):
    """API configuration settings"""
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=4, env="API_WORKERS")
    secret_key: str = Field(default="your-secret-key-change-this-in-production", env="API_SECRET_KEY")
    debug: bool = Field(default=True, env="DEBUG")


class ModelSettings(BaseSettings):
    """Model configuration settings"""
    cache_ttl: int = Field(default=3600, env="MODEL_CACHE_TTL")
    feature_cache_ttl: int = Field(default=1800, env="FEATURE_CACHE_TTL")
    prediction_cache_ttl: int = Field(default=300, env="PREDICTION_CACHE_TTL")


class MonitoringSettings(BaseSettings):
    """Monitoring configuration settings"""
    drift_threshold_psi: float = Field(default=0.2, env="DRIFT_THRESHOLD_PSI")
    drift_threshold_kl: float = Field(default=0.1, env="DRIFT_THRESHOLD_KL")
    fraud_threshold: float = Field(default=0.5, env="FRAUD_THRESHOLD")
    credit_threshold: int = Field(default=600, env="CREDIT_THRESHOLD")


class LoggingSettings(BaseSettings):
    """Logging configuration settings"""
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="json", env="LOG_FORMAT")


class DataSettings(BaseSettings):
    """Data path configuration settings"""
    seed_path: str = Field(default="./data/seed", env="DATA_SEED_PATH")
    export_path: str = Field(default="./data/exports", env="DATA_EXPORT_PATH")
    model_store_path: str = Field(default="./data/models", env="MODEL_STORE_PATH")


class StreamlitSettings(BaseSettings):
    """Streamlit configuration settings"""
    port: int = Field(default=8501, env="STREAMLIT_PORT")
    host: str = Field(default="localhost", env="STREAMLIT_HOST")


class Settings(BaseSettings):
    """Main application settings"""
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Sub-settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    cache: CacheSettings = CacheSettings()
    mlflow: MLflowSettings = MLflowSettings()
    api: APISettings = APISettings()
    model: ModelSettings = ModelSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    logging: LoggingSettings = LoggingSettings()
    data: DataSettings = DataSettings()
    streamlit: StreamlitSettings = StreamlitSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def reload_settings() -> Settings:
    """Reload application settings from environment variables."""
    global settings
    settings = Settings()
    return settings


# Convenience functions
def get_database_url() -> str:
    """Get database URL"""
    return settings.database.url


def get_redis_url() -> str:
    """Get Redis URL"""
    return settings.redis.url


def get_mlflow_tracking_uri() -> str:
    """Get MLflow tracking URI"""
    return settings.mlflow.tracking_uri


def is_development() -> bool:
    """Check if running in development mode"""
    return settings.environment.lower() == "development"


def is_production() -> bool:
    """Check if running in production mode"""
    return settings.environment.lower() == "production"
