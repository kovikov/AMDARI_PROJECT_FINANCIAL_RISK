"""
Redis cache infrastructure for FinRisk application.
Handles caching of model predictions, features, and frequently accessed data.
"""

import json
import logging
import pickle
from typing import Any, Optional, Union, List
from datetime import timedelta

import redis
import pandas as pd
from redis.exceptions import RedisError, ConnectionError

from app.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Global Redis client
_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> redis.Redis:
    """
    Get or create Redis client with connection pooling.
    
    Returns:
        Redis client instance
    """
    global _redis_client
    
    if _redis_client is None:
        settings = get_settings()
        
        try:
            _redis_client = redis.from_url(
                settings.redis.connection_url,
                decode_responses=False,  # Keep binary for pickle
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=20
            )
            
            # Test connection
            _redis_client.ping()
            logger.info("Redis client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            # Create a mock client for development
            _redis_client = MockRedisClient()
    
    return _redis_client


def get_cache_client() -> redis.Redis:
    """
    Get cache client (alias for get_redis_client for API compatibility).
    
    Returns:
        Redis client instance
    """
    return get_redis_client()


class MockRedisClient:
    """Mock Redis client for development when Redis is not available."""
    
    def __init__(self):
        self._data = {}
        logger.warning("Using mock Redis client - caching disabled")
    
    def get(self, key: str) -> None:
        return None
    
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        return True
    
    def delete(self, *keys) -> int:
        return len(keys)
    
    def exists(self, key: str) -> bool:
        return False
    
    def ping(self) -> bool:
        return True
    
    def flushdb(self) -> bool:
        return True
    
    def incr(self, key: str) -> int:
        """Mock increment method."""
        return 1
    
    def expire(self, key: str, seconds: int) -> bool:
        """Mock expire method."""
        return True
    
    def keys(self, pattern: str) -> List[bytes]:
        """Mock keys method."""
        return []
    
    def info(self) -> dict:
        """Mock info method for development."""
        return {
            "connected_clients": 0,
            "used_memory_human": "0B",
            "total_commands_processed": 0,
            "keyspace_hits": 0,
            "keyspace_misses": 0
        }


class CacheManager:
    """High-level cache manager for FinRisk application."""
    
    def __init__(self):
        self.client = get_redis_client()
        self.settings = get_settings()
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage in Redis."""
        if isinstance(value, (dict, list)):
            return json.dumps(value).encode('utf-8')
        elif isinstance(value, pd.DataFrame):
            return pickle.dumps(value)
        elif isinstance(value, (str, int, float, bool)):
            return json.dumps(value).encode('utf-8')
        else:
            return pickle.dumps(value)
    
    def _deserialize_value(self, value: bytes, value_type: str = 'auto') -> Any:
        """Deserialize value from Redis storage."""
        if value is None:
            return None
        
        try:
            if value_type == 'dataframe':
                return pickle.loads(value)
            elif value_type == 'json':
                return json.loads(value.decode('utf-8'))
            elif value_type == 'pickle':
                return pickle.loads(value)
            else:  # auto-detect
                try:
                    return json.loads(value.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return pickle.loads(value)
        except Exception as e:
            logger.error(f"Failed to deserialize cached value: {e}")
            return None
    
    def get(self, key: str, value_type: str = 'auto') -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            value_type: Expected value type ('auto', 'json', 'pickle', 'dataframe')
            
        Returns:
            Cached value or None if not found
        """
        try:
            value = self.client.get(key)
            return self._deserialize_value(value, value_type)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            serialized_value = self._serialize_value(value)
            return self.client.set(key, serialized_value, ex=ttl)
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, *keys: str) -> int:
        """
        Delete keys from cache.
        
        Args:
            keys: Cache keys to delete
            
        Returns:
            Number of keys deleted
        """
        try:
            return self.client.delete(*keys)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return 0
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.
        
        Args:
            pattern: Redis pattern (e.g., 'model:*', 'features:*')
            
        Returns:
            Number of keys deleted
        """
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear pattern error for {pattern}: {e}")
            return 0


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager()
    
    return _cache_manager


# Specialized cache functions for different data types
def cache_model_prediction(customer_id: str, model_type: str, 
                          prediction: dict, ttl: Optional[int] = None) -> bool:
    """
    Cache model prediction result.
    
    Args:
        customer_id: Customer ID
        model_type: Type of model ('credit', 'fraud')
        prediction: Prediction result dictionary
        ttl: Time to live in seconds
        
    Returns:
        True if cached successfully
    """
    cache = get_cache()
    settings = get_settings()
    
    key = f"prediction:{model_type}:{customer_id}"
    ttl = ttl or settings.cache.prediction_ttl
    
    return cache.set(key, prediction, ttl)


def get_cached_prediction(customer_id: str, model_type: str) -> Optional[dict]:
    """
    Get cached model prediction.
    
    Args:
        customer_id: Customer ID
        model_type: Type of model ('credit', 'fraud')
        
    Returns:
        Cached prediction or None
    """
    cache = get_cache()
    key = f"prediction:{model_type}:{customer_id}"
    
    return cache.get(key, 'json')


def cache_features(customer_id: str, features: dict, ttl: Optional[int] = None) -> bool:
    """
    Cache customer features.
    
    Args:
        customer_id: Customer ID
        features: Feature dictionary
        ttl: Time to live in seconds
        
    Returns:
        True if cached successfully
    """
    cache = get_cache()
    settings = get_settings()
    
    key = f"features:{customer_id}"
    ttl = ttl or settings.cache.feature_ttl
    
    return cache.set(key, features, ttl)


def get_cached_features(customer_id: str) -> Optional[dict]:
    """
    Get cached customer features.
    
    Args:
        customer_id: Customer ID
        
    Returns:
        Cached features or None
    """
    cache = get_cache()
    key = f"features:{customer_id}"
    
    return cache.get(key, 'json')


def cache_model_artifacts(model_name: str, model_version: str, 
                         artifacts: dict, ttl: Optional[int] = None) -> bool:
    """
    Cache model artifacts (encoders, scalers, etc.).
    
    Args:
        model_name: Model name
        model_version: Model version
        artifacts: Model artifacts dictionary
        ttl: Time to live in seconds
        
    Returns:
        True if cached successfully
    """
    cache = get_cache()
    settings = get_settings()
    
    key = f"model:{model_name}:{model_version}"
    ttl = ttl or settings.cache.model_ttl
    
    return cache.set(key, artifacts, ttl)


def get_cached_model_artifacts(model_name: str, model_version: str) -> Optional[dict]:
    """
    Get cached model artifacts.
    
    Args:
        model_name: Model name
        model_version: Model version
        
    Returns:
        Cached artifacts or None
    """
    cache = get_cache()
    key = f"model:{model_name}:{model_version}"
    
    return cache.get(key, 'pickle')


def cache_dataframe(key: str, df: pd.DataFrame, ttl: Optional[int] = None) -> bool:
    """
    Cache pandas DataFrame.
    
    Args:
        key: Cache key
        df: DataFrame to cache
        ttl: Time to live in seconds
        
    Returns:
        True if cached successfully
    """
    cache = get_cache()
    return cache.set(key, df, ttl)


def get_cached_dataframe(key: str) -> Optional[pd.DataFrame]:
    """
    Get cached pandas DataFrame.
    
    Args:
        key: Cache key
        
    Returns:
        Cached DataFrame or None
    """
    cache = get_cache()
    return cache.get(key, 'dataframe')


def clear_model_cache(model_name: Optional[str] = None) -> int:
    """
    Clear model-related cache entries.
    
    Args:
        model_name: Specific model name or None for all models
        
    Returns:
        Number of keys deleted
    """
    cache = get_cache()
    
    if model_name:
        pattern = f"model:{model_name}:*"
    else:
        pattern = "model:*"
    
    return cache.clear_pattern(pattern)


def clear_prediction_cache(customer_id: Optional[str] = None) -> int:
    """
    Clear prediction cache entries.
    
    Args:
        customer_id: Specific customer ID or None for all predictions
        
    Returns:
        Number of keys deleted
    """
    cache = get_cache()
    
    if customer_id:
        pattern = f"prediction:*:{customer_id}"
    else:
        pattern = "prediction:*"
    
    return cache.clear_pattern(pattern)


def get_cache_stats() -> dict:
    """
    Get cache statistics and health information.
    
    Returns:
        Dictionary with cache statistics
    """
    try:
        client = get_redis_client()
        info = client.info()
        
        return {
            "status": "healthy",
            "connected_clients": info.get("connected_clients", 0),
            "used_memory": info.get("used_memory_human", "0B"),
            "total_commands_processed": info.get("total_commands_processed", 0),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "hit_rate": info.get("keyspace_hits", 0) / max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0))
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
