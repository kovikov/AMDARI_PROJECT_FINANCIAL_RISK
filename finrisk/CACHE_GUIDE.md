# FinRisk Cache Guide

This guide covers all Redis cache operations for the FinRisk financial risk management system.

## 🏗️ Cache Architecture

The FinRisk system uses Redis for high-performance caching of:

- **Model Predictions**: Credit risk and fraud detection results
- **Customer Features**: Pre-computed feature vectors
- **Model Artifacts**: Encoders, scalers, and other ML components
- **DataFrames**: Frequently accessed data tables
- **Session Data**: User sessions and temporary data

## 📋 Cache Configuration

### Redis Settings

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_password  # Optional
```

### Cache TTL Settings

```bash
# Cache Time-to-Live (in seconds)
CACHE_PREDICTION_TTL=3600    # 1 hour
CACHE_FEATURE_TTL=1800       # 30 minutes
CACHE_MODEL_TTL=86400        # 24 hours
CACHE_DATAFRAME_TTL=7200     # 2 hours
CACHE_DEFAULT_TTL=1800       # 30 minutes
```

## 🛠️ Command Line Interface

The FinRisk system provides a comprehensive CLI for cache operations:

### Basic Commands

```bash
# Show all available commands
python scripts/cache_cli.py --help

# Check Redis connection
python scripts/cache_cli.py check

# Show cache configuration
python scripts/cache_cli.py config

# Show cache statistics
python scripts/cache_cli.py stats
```

### Model Prediction Caching

```bash
# Cache a credit risk prediction
python scripts/cache_cli.py cache-pred CUST001 credit '{"risk_score": 0.75, "decision": "approve", "confidence": 0.92}'

# Cache a fraud detection prediction
python scripts/cache_cli.py cache-pred CUST001 fraud '{"fraud_probability": 0.15, "risk_level": "low", "confidence": 0.88}'

# Get cached prediction
python scripts/cache_cli.py get-pred CUST001 credit
python scripts/cache_cli.py get-pred CUST001 fraud
```

### Feature Caching

```bash
# Cache customer features
python scripts/cache_cli.py cache-feat CUST001 '{"age": 35, "income": 75000, "credit_score": 720, "employment_years": 8}'

# Get cached features
python scripts/cache_cli.py get-feat CUST001
```

### DataFrame Caching

```bash
# Cache DataFrame from CSV file
python scripts/cache_cli.py cache-df customer_data data/customers.csv

# Cache DataFrame with custom TTL
python scripts/cache_cli.py cache-df transaction_data data/transactions.csv --ttl 3600

# Get cached DataFrame
python scripts/cache_cli.py get-df customer_data

# Get cached DataFrame and save to file
python scripts/cache_cli.py get-df customer_data --output customers_export.csv
```

### Cache Management

```bash
# Clear all model cache
python scripts/cache_cli.py clear-model

# Clear specific model cache
python scripts/cache_cli.py clear-model --model credit_risk

# Clear all prediction cache
python scripts/cache_cli.py clear-pred

# Clear specific customer prediction cache
python scripts/cache_cli.py clear-pred --customer CUST001
```

## 🔧 Python API Usage

### Basic Cache Operations

```python
from app.infra.cache import get_cache, get_redis_client

# Get cache manager
cache = get_cache()

# Basic operations
cache.set("my_key", "my_value", ttl=3600)
value = cache.get("my_key")
exists = cache.exists("my_key")
deleted = cache.delete("my_key")
```

### Model Prediction Caching

```python
from app.infra.cache import cache_model_prediction, get_cached_prediction

# Cache prediction
prediction = {
    "risk_score": 0.75,
    "decision": "approve",
    "confidence": 0.92,
    "model_version": "v1.2.0"
}

success = cache_model_prediction("CUST001", "credit", prediction)

# Get cached prediction
cached_pred = get_cached_prediction("CUST001", "credit")
if cached_pred:
    print(f"Risk score: {cached_pred['risk_score']}")
```

### Feature Caching

```python
from app.infra.cache import cache_features, get_cached_features

# Cache features
features = {
    "age": 35,
    "income": 75000,
    "credit_score": 720,
    "employment_years": 8,
    "debt_to_income": 0.25
}

success = cache_features("CUST001", features)

# Get cached features
cached_features = get_cached_features("CUST001")
if cached_features:
    print(f"Customer age: {cached_features['age']}")
```

### Model Artifacts Caching

```python
from app.infra.cache import cache_model_artifacts, get_cached_model_artifacts

# Cache model artifacts
artifacts = {
    "scaler": scaler_object,
    "encoder": encoder_object,
    "feature_names": ["age", "income", "credit_score"],
    "model_metadata": {"version": "v1.2.0", "trained_date": "2024-01-15"}
}

success = cache_model_artifacts("credit_risk", "v1.2.0", artifacts)

# Get cached artifacts
cached_artifacts = get_cached_model_artifacts("credit_risk", "v1.2.0")
if cached_artifacts:
    scaler = cached_artifacts["scaler"]
    feature_names = cached_artifacts["feature_names"]
```

### DataFrame Caching

```python
from app.infra.cache import cache_dataframe, get_cached_dataframe
import pandas as pd

# Cache DataFrame
df = pd.read_csv("data/customers.csv")
success = cache_dataframe("customer_data", df, ttl=7200)

# Get cached DataFrame
cached_df = get_cached_dataframe("customer_data")
if cached_df is not None:
    print(f"DataFrame shape: {cached_df.shape}")
    print(f"Columns: {list(cached_df.columns)}")
```

### Cache Statistics

```python
from app.infra.cache import get_cache_stats

# Get cache statistics
stats = get_cache_stats()

if stats["status"] == "healthy":
    print(f"Connected clients: {stats['connected_clients']}")
    print(f"Used memory: {stats['used_memory']}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")
    print(f"Total commands: {stats['total_commands_processed']:,}")
else:
    print(f"Cache unhealthy: {stats.get('error', 'Unknown error')}")
```

## 📊 Cache Patterns

### Prediction Caching Pattern

```python
def get_credit_prediction(customer_id: str) -> dict:
    """Get credit prediction with caching."""
    
    # Try to get from cache first
    cached_pred = get_cached_prediction(customer_id, "credit")
    if cached_pred:
        return cached_pred
    
    # If not in cache, compute prediction
    prediction = compute_credit_prediction(customer_id)
    
    # Cache the result
    cache_model_prediction(customer_id, "credit", prediction)
    
    return prediction
```

### Feature Caching Pattern

```python
def get_customer_features(customer_id: str) -> dict:
    """Get customer features with caching."""
    
    # Try to get from cache first
    cached_features = get_cached_features(customer_id)
    if cached_features:
        return cached_features
    
    # If not in cache, compute features
    features = compute_customer_features(customer_id)
    
    # Cache the result
    cache_features(customer_id, features)
    
    return features
```

### Model Artifacts Pattern

```python
def get_model_artifacts(model_name: str, version: str) -> dict:
    """Get model artifacts with caching."""
    
    # Try to get from cache first
    cached_artifacts = get_cached_model_artifacts(model_name, version)
    if cached_artifacts:
        return cached_artifacts
    
    # If not in cache, load from storage
    artifacts = load_model_artifacts(model_name, version)
    
    # Cache the result
    cache_model_artifacts(model_name, version, artifacts)
    
    return artifacts
```

## 🔍 Monitoring and Maintenance

### Regular Monitoring Tasks

```bash
# Daily: Check cache health
python scripts/cache_cli.py stats
python scripts/cache_cli.py check

# Weekly: Monitor cache performance
python scripts/cache_cli.py stats | grep "Hit Rate"

# Monthly: Review cache configuration
python scripts/cache_cli.py config
```

### Cache Performance Metrics

**Key Metrics to Monitor:**
- **Hit Rate**: Percentage of cache hits vs misses
- **Memory Usage**: Redis memory consumption
- **Connected Clients**: Number of active connections
- **Command Processing**: Total commands processed

**Example Monitoring Script:**
```python
import time
from app.infra.cache import get_cache_stats

def monitor_cache():
    while True:
        stats = get_cache_stats()
        
        if stats["status"] == "healthy":
            hit_rate = stats["hit_rate"]
            memory = stats["used_memory"]
            
            print(f"Hit Rate: {hit_rate:.2%}")
            print(f"Memory: {memory}")
            
            # Alert if hit rate is low
            if hit_rate < 0.8:
                print("⚠️  Low cache hit rate detected!")
                
            # Alert if memory usage is high
            if "MB" in memory and float(memory.replace("MB", "")) > 1000:
                print("⚠️  High memory usage detected!")
        
        time.sleep(300)  # Check every 5 minutes
```

### Cache Invalidation Strategies

```python
# Clear specific model cache when model is updated
def update_model(model_name: str, version: str):
    # Update model logic here
    ...
    
    # Clear old model cache
    clear_model_cache(model_name)

# Clear customer cache when data is updated
def update_customer_data(customer_id: str):
    # Update customer data logic here
    ...
    
    # Clear customer-related cache
    clear_prediction_cache(customer_id)
    # Note: Features cache will be automatically refreshed on next access
```

## 🚨 Troubleshooting

### Common Issues

1. **Connection Failed**
   ```bash
   # Check if Redis is running
   python scripts/cache_cli.py check
   
   # Verify configuration
   python scripts/cache_cli.py config
   ```

2. **Low Hit Rate**
   - Review TTL settings
   - Check cache invalidation patterns
   - Monitor cache key patterns

3. **High Memory Usage**
   - Review cached data sizes
   - Adjust TTL settings
   - Implement cache eviction policies

4. **Serialization Errors**
   - Check data types being cached
   - Ensure objects are serializable
   - Use appropriate serialization methods

### Development Mode

When Redis is not available, the system automatically uses a mock client:

```python
# Mock client behavior
cache.set("key", "value")  # Always returns True
cache.get("key")           # Always returns None
cache.exists("key")        # Always returns False
```

This allows development to continue without Redis, but caching is effectively disabled.

## 📈 Best Practices

1. **Use Appropriate TTLs**
   - Short TTL for frequently changing data
   - Long TTL for stable data
   - No TTL for model artifacts

2. **Cache Key Naming**
   - Use consistent naming patterns
   - Include version information
   - Use descriptive prefixes

3. **Error Handling**
   - Always handle cache failures gracefully
   - Implement fallback mechanisms
   - Log cache errors for monitoring

4. **Memory Management**
   - Monitor cache size
   - Implement eviction policies
   - Use compression for large objects

5. **Security**
   - Use Redis authentication
   - Restrict network access
   - Encrypt sensitive cached data

## 🔗 Related Files

- `app/infra/cache.py` - Cache infrastructure module
- `scripts/cache_cli.py` - Command-line interface
- `app/config.py` - Cache configuration
- `.env` - Environment variables

## 📚 Additional Resources

- [Redis Documentation](https://redis.io/documentation)
- [Redis Python Client](https://redis-py.readthedocs.io/)
- [Cache Patterns](https://redis.io/topics/patterns)
- [Redis Performance](https://redis.io/topics/optimization)

