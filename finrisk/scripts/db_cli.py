#!/usr/bin/env python3
"""
FinRisk Database Command Line Interface
Provides command-line tools for database operations.
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.infra.db import (
    check_database_connection, execute_sql_file, get_table_info,
    read_sql_query, insert_dataframe, get_table_count, backup_table,
    restore_table_from_backup, vacuum_table, get_database_size,
    truncate_table, create_indexes, health_check
)
from app.config import get_settings


def setup_database():
    """Set up the database schema"""
    print("🔧 Setting up FinRisk database...")
    
    try:
        # Execute SQL files in order
        sql_dir = project_root / "sql"
        sql_files = [
            "001_init_schema.sql",
            "010_indexes.sql",
            "020_sample_views.sql"
        ]
        
        for sql_file in sql_files:
            file_path = sql_dir / sql_file
            if file_path.exists():
                print(f"📄 Executing {sql_file}...")
                execute_sql_file(str(file_path))
                print(f"✅ {sql_file} executed successfully")
            else:
                print(f"❌ SQL file not found: {file_path}")
                return False
        
        print("🎉 Database setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False


def check_connection():
    """Check database connection"""
    print("🔍 Checking database connection...")
    
    if check_database_connection():
        print("✅ Database connection successful")
        return True
    else:
        print("❌ Database connection failed")
        return False


def show_tables(schema="finrisk"):
    """Show tables in the specified schema"""
    print(f"📋 Tables in schema '{schema}':")
    
    try:
        df = get_table_info(schema)
        if not df.empty:
            # Group by table name and show columns
            for table_name in df['table_name'].unique():
                table_info = df[df['table_name'] == table_name]
                print(f"\n📊 Table: {table_name}")
                print("-" * 50)
                for _, row in table_info.iterrows():
                    nullable = "NULL" if row['is_nullable'] == 'YES' else "NOT NULL"
                    default = f" DEFAULT {row['column_default']}" if row['column_default'] else ""
                    print(f"  {row['column_name']:<20} {row['data_type']:<15} {nullable}{default}")
        else:
            print(f"No tables found in schema '{schema}'")
            
    except Exception as e:
        print(f"❌ Failed to get table info: {e}")


def execute_query(query, output_file=None):
    """Execute a SQL query and display results"""
    print(f"🔍 Executing query: {query[:50]}...")
    
    try:
        df = read_sql_query(query)
        
        if output_file:
            # Save to file
            if output_file.endswith('.csv'):
                df.to_csv(output_file, index=False)
            elif output_file.endswith('.xlsx'):
                df.to_excel(output_file, index=False)
            else:
                df.to_csv(output_file, index=False)
            print(f"💾 Results saved to {output_file}")
        else:
            # Display results
            print(f"\n📊 Query Results ({len(df)} rows):")
            print("=" * 80)
            print(df.to_string(index=False))
            
    except Exception as e:
        print(f"❌ Query execution failed: {e}")


def show_table_count(table_name, schema="finrisk"):
    """Show row count for a table"""
    print(f"📊 Row count for {schema}.{table_name}:")
    
    try:
        count = get_table_count(table_name, schema)
        print(f"  {count:,} rows")
    except Exception as e:
        print(f"❌ Failed to get count: {e}")


def backup_table_cmd(table_name, schema="finrisk", suffix="_backup"):
    """Backup a table"""
    print(f"💾 Creating backup of {schema}.{table_name}...")
    
    try:
        backup_name = backup_table(table_name, schema, suffix)
        print(f"✅ Backup created: {schema}.{backup_name}")
    except Exception as e:
        print(f"❌ Backup failed: {e}")


def restore_table_cmd(table_name, backup_table_name, schema="finrisk"):
    """Restore a table from backup"""
    print(f"🔄 Restoring {schema}.{table_name} from backup...")
    
    try:
        restore_table_from_backup(table_name, backup_table_name, schema)
        print(f"✅ Table restored successfully")
    except Exception as e:
        print(f"❌ Restore failed: {e}")


def vacuum_table_cmd(table_name, schema="finrisk"):
    """Run VACUUM on a table"""
    print(f"🧹 Running VACUUM on {schema}.{table_name}...")
    
    try:
        vacuum_table(table_name, schema)
        print(f"✅ VACUUM completed successfully")
    except Exception as e:
        print(f"❌ VACUUM failed: {e}")


def show_database_size():
    """Show database size information"""
    print("📏 Database size information:")
    
    try:
        size_info = get_database_size()
        
        print(f"\n📊 Total size: {size_info['total_size_pretty']}")
        print("\n📋 Table sizes:")
        print("-" * 80)
        print(f"{'Schema':<12} {'Table':<25} {'Size':<15}")
        print("-" * 80)
        
        for table in size_info['tables']:
            print(f"{table['schemaname']:<12} {table['tablename']:<25} {table['size']:<15}")
            
    except Exception as e:
        print(f"❌ Failed to get database size: {e}")


def show_config():
    """Show database configuration"""
    print("⚙️  Database configuration:")
    
    try:
        settings = get_settings()
        print(f"  Host: {settings.database.host}")
        print(f"  Port: {settings.database.port}")
        print(f"  Database: {settings.database.name}")
        print(f"  User: {settings.database.user}")
        print(f"  URL: {settings.database.url}")
        
    except Exception as e:
        print(f"❌ Failed to get configuration: {e}")


def truncate_table_cmd(table_name, schema="finrisk"):
    """Truncate a table"""
    print(f"🗑️  Truncating table {schema}.{table_name}...")
    
    try:
        truncate_table(table_name, schema)
        print(f"✅ Table {schema}.{table_name} truncated successfully")
    except Exception as e:
        print(f"❌ Truncate failed: {e}")


def create_indexes_cmd():
    """Create database indexes"""
    print("🔧 Creating database indexes...")
    
    try:
        create_indexes()
        print("✅ Database indexes created successfully")
    except Exception as e:
        print(f"❌ Index creation failed: {e}")


def health_check_cmd():
    """Perform database health check"""
    print("🏥 Performing database health check...")
    
    try:
        health_info = health_check()
        
        print(f"\n📊 Health Check Results:")
        print(f"  Status: {health_info['status']}")
        print(f"  Connection: {'✅ Connected' if health_info['connection'] else '❌ Failed'}")
        print(f"  Response Time: {health_info.get('response_time_seconds', 'N/A')} seconds")
        print(f"  Customer Count: {health_info.get('customer_count', 'N/A')}")
        print(f"  Timestamp: {health_info['timestamp']}")
        
        if 'error' in health_info:
            print(f"  Error: {health_info['error']}")
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="FinRisk Database CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up database schema")
    
    # Check connection command
    check_parser = subparsers.add_parser("check", help="Check database connection")
    
    # Show tables command
    tables_parser = subparsers.add_parser("tables", help="Show tables in schema")
    tables_parser.add_argument("--schema", default="finrisk", help="Schema name")
    
    # Execute query command
    query_parser = subparsers.add_parser("query", help="Execute SQL query")
    query_parser.add_argument("sql", help="SQL query to execute")
    query_parser.add_argument("--output", help="Output file (CSV or Excel)")
    
    # Show table count command
    count_parser = subparsers.add_parser("count", help="Show table row count")
    count_parser.add_argument("table", help="Table name")
    count_parser.add_argument("--schema", default="finrisk", help="Schema name")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Backup a table")
    backup_parser.add_argument("table", help="Table name")
    backup_parser.add_argument("--schema", default="finrisk", help="Schema name")
    backup_parser.add_argument("--suffix", default="_backup", help="Backup suffix")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore table from backup")
    restore_parser.add_argument("table", help="Table name")
    restore_parser.add_argument("backup", help="Backup table name")
    restore_parser.add_argument("--schema", default="finrisk", help="Schema name")
    
    # VACUUM command
    vacuum_parser = subparsers.add_parser("vacuum", help="Run VACUUM on table")
    vacuum_parser.add_argument("table", help="Table name")
    vacuum_parser.add_argument("--schema", default="finrisk", help="Schema name")
    
    # Size command
    size_parser = subparsers.add_parser("size", help="Show database size")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show database configuration")
    
    # Truncate command
    truncate_parser = subparsers.add_parser("truncate", help="Truncate a table")
    truncate_parser.add_argument("table", help="Table name")
    truncate_parser.add_argument("--schema", default="finrisk", help="Schema name")
    
    # Create indexes command
    indexes_parser = subparsers.add_parser("indexes", help="Create database indexes")
    
    # Health check command
    health_parser = subparsers.add_parser("health", help="Perform database health check")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == "setup":
        success = setup_database()
        sys.exit(0 if success else 1)
        
    elif args.command == "check":
        success = check_connection()
        sys.exit(0 if success else 1)
        
    elif args.command == "tables":
        show_tables(args.schema)
        
    elif args.command == "query":
        execute_query(args.sql, args.output)
        
    elif args.command == "count":
        show_table_count(args.table, args.schema)
        
    elif args.command == "backup":
        backup_table_cmd(args.table, args.schema, args.suffix)
        
    elif args.command == "restore":
        restore_table_cmd(args.table, args.backup, args.schema)
        
    elif args.command == "vacuum":
        vacuum_table_cmd(args.table, args.schema)
        
    elif args.command == "size":
        show_database_size()
        
    elif args.command == "config":
        show_config()
        
    elif args.command == "truncate":
        truncate_table_cmd(args.table, args.schema)
        
    elif args.command == "indexes":
        create_indexes_cmd()
        
    elif args.command == "health":
        health_check_cmd()


if __name__ == "__main__":
    main()
