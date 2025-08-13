#!/usr/bin/env python3
"""
FinRisk Cache Command Line Interface
Provides command-line tools for Redis cache operations.
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.infra.cache import (
    get_cache, get_redis_client, get_cache_stats,
    cache_model_prediction, get_cached_prediction,
    cache_features, get_cached_features,
    cache_model_artifacts, get_cached_model_artifacts,
    cache_dataframe, get_cached_dataframe,
    clear_model_cache, clear_prediction_cache
)
from app.config import get_settings


def check_connection():
    """Check Redis connection"""
    print("🔍 Checking Redis connection...")
    
    try:
        client = get_redis_client()
        if client.ping():
            print("✅ Redis connection successful")
            return True
        else:
            print("❌ Redis connection failed")
            return False
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False


def show_stats():
    """Show cache statistics"""
    print("📊 Cache Statistics:")
    
    try:
        stats = get_cache_stats()
        
        if stats["status"] == "healthy":
            print(f"  Status: ✅ {stats['status']}")
            print(f"  Connected Clients: {stats['connected_clients']}")
            print(f"  Used Memory: {stats['used_memory']}")
            print(f"  Total Commands: {stats['total_commands_processed']:,}")
            print(f"  Cache Hits: {stats['keyspace_hits']:,}")
            print(f"  Cache Misses: {stats['keyspace_misses']:,}")
            print(f"  Hit Rate: {stats['hit_rate']:.2%}")
        else:
            print(f"  Status: ❌ {stats['status']}")
            print(f"  Error: {stats.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Failed to get cache stats: {e}")


def show_config():
    """Show cache configuration"""
    print("⚙️  Cache configuration:")
    
    try:
        settings = get_settings()
        print(f"  Redis Host: {settings.redis.host}")
        print(f"  Redis Port: {settings.redis.port}")
        print(f"  Redis DB: {settings.redis.db}")
        print(f"  Redis URL: {settings.redis.url}")
        print(f"  Prediction TTL: {settings.cache.prediction_ttl} seconds")
        print(f"  Feature TTL: {settings.cache.feature_ttl} seconds")
        print(f"  Model TTL: {settings.cache.model_ttl} seconds")
        print(f"  DataFrame TTL: {settings.cache.dataframe_ttl} seconds")
        
    except Exception as e:
        print(f"❌ Failed to get configuration: {e}")


def cache_prediction_cmd(customer_id, model_type, prediction_data):
    """Cache a model prediction"""
    print(f"💾 Caching prediction for customer {customer_id}, model {model_type}...")
    
    try:
        # Parse prediction data as JSON
        import json
        prediction = json.loads(prediction_data)
        
        success = cache_model_prediction(customer_id, model_type, prediction)
        if success:
            print(f"✅ Prediction cached successfully")
        else:
            print(f"❌ Failed to cache prediction")
            
    except Exception as e:
        print(f"❌ Failed to cache prediction: {e}")


def get_prediction_cmd(customer_id, model_type):
    """Get cached prediction"""
    print(f"🔍 Getting cached prediction for customer {customer_id}, model {model_type}...")
    
    try:
        prediction = get_cached_prediction(customer_id, model_type)
        
        if prediction:
            print(f"✅ Found cached prediction:")
            import json
            print(json.dumps(prediction, indent=2))
        else:
            print(f"❌ No cached prediction found")
            
    except Exception as e:
        print(f"❌ Failed to get prediction: {e}")


def cache_features_cmd(customer_id, features_data):
    """Cache customer features"""
    print(f"💾 Caching features for customer {customer_id}...")
    
    try:
        # Parse features data as JSON
        import json
        features = json.loads(features_data)
        
        success = cache_features(customer_id, features)
        if success:
            print(f"✅ Features cached successfully")
        else:
            print(f"❌ Failed to cache features")
            
    except Exception as e:
        print(f"❌ Failed to cache features: {e}")


def get_features_cmd(customer_id):
    """Get cached features"""
    print(f"🔍 Getting cached features for customer {customer_id}...")
    
    try:
        features = get_cached_features(customer_id)
        
        if features:
            print(f"✅ Found cached features:")
            import json
            print(json.dumps(features, indent=2))
        else:
            print(f"❌ No cached features found")
            
    except Exception as e:
        print(f"❌ Failed to get features: {e}")


def clear_model_cache_cmd(model_name=None):
    """Clear model cache"""
    if model_name:
        print(f"🗑️  Clearing cache for model {model_name}...")
    else:
        print(f"🗑️  Clearing all model cache...")
    
    try:
        deleted = clear_model_cache(model_name)
        print(f"✅ Deleted {deleted} cache entries")
        
    except Exception as e:
        print(f"❌ Failed to clear model cache: {e}")


def clear_prediction_cache_cmd(customer_id=None):
    """Clear prediction cache"""
    if customer_id:
        print(f"🗑️  Clearing prediction cache for customer {customer_id}...")
    else:
        print(f"🗑️  Clearing all prediction cache...")
    
    try:
        deleted = clear_prediction_cache(customer_id)
        print(f"✅ Deleted {deleted} cache entries")
        
    except Exception as e:
        print(f"❌ Failed to clear prediction cache: {e}")


def cache_dataframe_cmd(key, file_path, ttl=None):
    """Cache DataFrame from file"""
    print(f"💾 Caching DataFrame with key '{key}' from {file_path}...")
    
    try:
        # Load DataFrame from file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            print(f"❌ Unsupported file format: {file_path}")
            return
        
        success = cache_dataframe(key, df, ttl)
        if success:
            print(f"✅ DataFrame cached successfully ({len(df)} rows)")
        else:
            print(f"❌ Failed to cache DataFrame")
            
    except Exception as e:
        print(f"❌ Failed to cache DataFrame: {e}")


def get_dataframe_cmd(key, output_file=None):
    """Get cached DataFrame"""
    print(f"🔍 Getting cached DataFrame with key '{key}'...")
    
    try:
        df = get_cached_dataframe(key)
        
        if df is not None:
            print(f"✅ Found cached DataFrame ({len(df)} rows)")
            
            if output_file:
                # Save to file
                if output_file.endswith('.csv'):
                    df.to_csv(output_file, index=False)
                elif output_file.endswith('.xlsx'):
                    df.to_excel(output_file, index=False)
                elif output_file.endswith('.parquet'):
                    df.to_parquet(output_file, index=False)
                else:
                    df.to_csv(output_file, index=False)
                print(f"💾 DataFrame saved to {output_file}")
            else:
                # Display summary
                print(f"\n📊 DataFrame Summary:")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        else:
            print(f"❌ No cached DataFrame found")
            
    except Exception as e:
        print(f"❌ Failed to get DataFrame: {e}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="FinRisk Cache CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Check connection command
    check_parser = subparsers.add_parser("check", help="Check Redis connection")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show cache statistics")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show cache configuration")
    
    # Cache prediction command
    cache_pred_parser = subparsers.add_parser("cache-pred", help="Cache model prediction")
    cache_pred_parser.add_argument("customer_id", help="Customer ID")
    cache_pred_parser.add_argument("model_type", help="Model type (credit, fraud)")
    cache_pred_parser.add_argument("prediction", help="Prediction data (JSON)")
    
    # Get prediction command
    get_pred_parser = subparsers.add_parser("get-pred", help="Get cached prediction")
    get_pred_parser.add_argument("customer_id", help="Customer ID")
    get_pred_parser.add_argument("model_type", help="Model type (credit, fraud)")
    
    # Cache features command
    cache_feat_parser = subparsers.add_parser("cache-feat", help="Cache customer features")
    cache_feat_parser.add_argument("customer_id", help="Customer ID")
    cache_feat_parser.add_argument("features", help="Features data (JSON)")
    
    # Get features command
    get_feat_parser = subparsers.add_parser("get-feat", help="Get cached features")
    get_feat_parser.add_argument("customer_id", help="Customer ID")
    
    # Clear model cache command
    clear_model_parser = subparsers.add_parser("clear-model", help="Clear model cache")
    clear_model_parser.add_argument("--model", help="Specific model name")
    
    # Clear prediction cache command
    clear_pred_parser = subparsers.add_parser("clear-pred", help="Clear prediction cache")
    clear_pred_parser.add_argument("--customer", help="Specific customer ID")
    
    # Cache DataFrame command
    cache_df_parser = subparsers.add_parser("cache-df", help="Cache DataFrame")
    cache_df_parser.add_argument("key", help="Cache key")
    cache_df_parser.add_argument("file", help="Input file (CSV, Excel, Parquet)")
    cache_df_parser.add_argument("--ttl", type=int, help="Time to live in seconds")
    
    # Get DataFrame command
    get_df_parser = subparsers.add_parser("get-df", help="Get cached DataFrame")
    get_df_parser.add_argument("key", help="Cache key")
    get_df_parser.add_argument("--output", help="Output file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == "check":
        success = check_connection()
        sys.exit(0 if success else 1)
        
    elif args.command == "stats":
        show_stats()
        
    elif args.command == "config":
        show_config()
        
    elif args.command == "cache-pred":
        cache_prediction_cmd(args.customer_id, args.model_type, args.prediction)
        
    elif args.command == "get-pred":
        get_prediction_cmd(args.customer_id, args.model_type)
        
    elif args.command == "cache-feat":
        cache_features_cmd(args.customer_id, args.features)
        
    elif args.command == "get-feat":
        get_features_cmd(args.customer_id)
        
    elif args.command == "clear-model":
        clear_model_cache_cmd(args.model)
        
    elif args.command == "clear-pred":
        clear_prediction_cache_cmd(args.customer)
        
    elif args.command == "cache-df":
        cache_dataframe_cmd(args.key, args.file, args.ttl)
        
    elif args.command == "get-df":
        get_dataframe_cmd(args.key, args.output)


if __name__ == "__main__":
    main()

