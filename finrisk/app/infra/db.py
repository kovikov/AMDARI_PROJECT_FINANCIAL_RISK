"""
Database infrastructure module for FinRisk application.
Handles PostgreSQL connections, session management, and database operations.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional

import pandas as pd
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from app.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Global variables
_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def get_engine() -> Engine:
    """
    Get or create the database engine with connection pooling.
    
    Returns:
        SQLAlchemy Engine instance
    """
    global _engine
    
    if _engine is None:
        settings = get_settings()
        
        _engine = create_engine(
            settings.database.url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
            echo=settings.api.debug,  # Log SQL queries in debug mode
        )
        
        logger.info("Database engine initialized")
    
    return _engine


def reset_engine() -> None:
    """
    Reset the database engine to force recreation with new settings.
    """
    global _engine, _session_factory
    
    if _engine is not None:
        _engine.dispose()
        _engine = None
    
    _session_factory = None
    logger.info("Database engine reset")


def get_session_factory() -> sessionmaker:
    """
    Get or create the session factory.
    
    Returns:
        SQLAlchemy sessionmaker instance
    """
    global _session_factory
    
    if _session_factory is None:
        engine = get_engine()
        _session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        logger.info("Database session factory initialized")
    
    return _session_factory


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions with automatic cleanup.
    
    Yields:
        SQLAlchemy Session instance
        
    Example:
        with get_db_session() as session:
            result = session.execute(text("SELECT 1"))
    """
    session_factory = get_session_factory()
    session = session_factory()
    
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def execute_sql_file(file_path: str) -> None:
    """
    Execute SQL commands from a file.
    
    Args:
        file_path: Path to the SQL file
        
    Raises:
        SQLAlchemyError: If SQL execution fails
    """
    try:
        with open(file_path, 'r') as file:
            sql_content = file.read()
        
        engine = get_engine()
        with engine.connect() as connection:
            # Split by semicolon and execute each statement
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
            
            for statement in statements:
                connection.execute(text(statement))
                connection.commit()
        
        logger.info(f"Successfully executed SQL file: {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to execute SQL file {file_path}: {e}")
        raise SQLAlchemyError(f"SQL file execution failed: {e}")


def check_database_connection() -> bool:
    """
    Check if database connection is working.
    
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        engine = get_engine()
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            return result.fetchone()[0] == 1
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


def get_table_info(schema: str = "finrisk") -> pd.DataFrame:
    """
    Get information about tables in the specified schema.
    
    Args:
        schema: Database schema name
        
    Returns:
        DataFrame with table information
    """
    query = f"""
    SELECT 
        table_name,
        column_name,
        data_type,
        is_nullable,
        column_default
    FROM information_schema.columns 
    WHERE table_schema = '{schema}'
    ORDER BY table_name, ordinal_position;
    """
    
    try:
        return read_sql_query(query)
    except Exception as e:
        logger.error(f"Failed to get table info for schema {schema}: {e}")
        return pd.DataFrame()


def read_sql_query(query: str, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Execute a SQL query and return results as DataFrame.
    
    Args:
        query: SQL query string
        params: Query parameters
        
    Returns:
        DataFrame with query results
    """
    try:
        engine = get_engine()
        return pd.read_sql_query(query, engine, params=params)
    except Exception as e:
        logger.error(f"Failed to execute query: {e}")
        raise SQLAlchemyError(f"Query execution failed: {e}")


def insert_dataframe(df: pd.DataFrame, table_name: str, 
                    schema: str = "finrisk", if_exists: str = "append") -> None:
    """
    Insert DataFrame into database table.
    
    Args:
        df: DataFrame to insert
        table_name: Target table name
        schema: Database schema
        if_exists: How to behave if table exists ('fail', 'replace', 'append')
    """
    try:
        engine = get_engine()
        df.to_sql(
            table_name, 
            engine, 
            schema=schema, 
            if_exists=if_exists, 
            index=False,
            method='multi',  # Use multi-row INSERT for better performance
            chunksize=1000
        )
        logger.info(f"Successfully inserted {len(df)} rows into {schema}.{table_name}")
        
    except Exception as e:
        logger.error(f"Failed to insert DataFrame into {schema}.{table_name}: {e}")
        raise SQLAlchemyError(f"DataFrame insertion failed: {e}")


def bulk_insert_data(data: list, table_name: str, schema: str = "finrisk") -> None:
    """
    Bulk insert data using SQLAlchemy core for better performance.
    
    Args:
        data: List of dictionaries with row data
        table_name: Target table name
        schema: Database schema
    """
    try:
        engine = get_engine()
        metadata = MetaData()
        metadata.reflect(bind=engine, schema=schema)
        
        table = metadata.tables[f"{schema}.{table_name}"]
        
        with engine.connect() as connection:
            connection.execute(table.insert(), data)
            connection.commit()
        
        logger.info(f"Successfully bulk inserted {len(data)} rows into {schema}.{table_name}")
        
    except Exception as e:
        logger.error(f"Failed to bulk insert into {schema}.{table_name}: {e}")
        raise SQLAlchemyError(f"Bulk insert failed: {e}")


def get_table_count(table_name: str, schema: str = "finrisk") -> int:
    """
    Get row count for a table.
    
    Args:
        table_name: Table name
        schema: Database schema
        
    Returns:
        Number of rows in the table
    """
    try:
        query = f"SELECT COUNT(*) FROM {schema}.{table_name}"
        result = read_sql_query(query)
        return result.iloc[0, 0]
    except Exception as e:
        logger.error(f"Failed to get count for {schema}.{table_name}: {e}")
        return 0


def get_row_count(table_name: str, schema: str = "finrisk") -> int:
    """
    Get row count for a table (alias for get_table_count).
    
    Args:
        table_name: Table name
        schema: Database schema
        
    Returns:
        Number of rows in the table
    """
    return get_table_count(table_name, schema)


def truncate_table(table_name: str, schema: str = "finrisk") -> None:
    """
    Truncate a table (remove all rows).
    
    Args:
        table_name: Table name to truncate
        schema: Database schema
    """
    try:
        engine = get_engine()
        with engine.connect() as connection:
            connection.execute(text(f"TRUNCATE TABLE {schema}.{table_name} CASCADE"))
            connection.commit()
        
        logger.info(f"Successfully truncated table {schema}.{table_name}")
        
    except Exception as e:
        logger.error(f"Failed to truncate table {schema}.{table_name}: {e}")
        raise SQLAlchemyError(f"Table truncation failed: {e}")


def create_indexes() -> None:
    """Create database indexes for performance optimization."""
    indexes_sql = """
    -- Performance indexes for customer queries
    CREATE INDEX IF NOT EXISTS idx_customer_age_income ON finrisk.customer_profiles(customer_age, annual_income);
    CREATE INDEX IF NOT EXISTS idx_customer_city ON finrisk.customer_profiles(city);
    CREATE INDEX IF NOT EXISTS idx_customer_employment ON finrisk.customer_profiles(employment_status);
    
    -- Credit application indexes
    CREATE INDEX IF NOT EXISTS idx_credit_app_customer_date ON finrisk.credit_applications(customer_id, application_date);
    CREATE INDEX IF NOT EXISTS idx_credit_app_amount ON finrisk.credit_applications(loan_amount);
    CREATE INDEX IF NOT EXISTS idx_credit_app_purpose ON finrisk.credit_applications(loan_purpose);
    
    -- Transaction indexes for fraud detection
    CREATE INDEX IF NOT EXISTS idx_transaction_customer_date ON finrisk.transaction_data(customer_id, transaction_date);
    CREATE INDEX IF NOT EXISTS idx_transaction_amount ON finrisk.transaction_data(amount);
    CREATE INDEX IF NOT EXISTS idx_transaction_merchant ON finrisk.transaction_data(merchant_category);
    CREATE INDEX IF NOT EXISTS idx_transaction_location ON finrisk.transaction_data(location);
    
    -- Model prediction indexes
    CREATE INDEX IF NOT EXISTS idx_prediction_customer_type ON finrisk.model_predictions(customer_id, prediction_type);
    CREATE INDEX IF NOT EXISTS idx_prediction_version ON finrisk.model_predictions(model_version);
    
    -- Audit and monitoring indexes
    CREATE INDEX IF NOT EXISTS idx_decision_log_customer ON audit.decision_log(customer_id);
    CREATE INDEX IF NOT EXISTS idx_decision_log_date ON audit.decision_log(created_at);
    CREATE INDEX IF NOT EXISTS idx_drift_detection_date ON monitoring.drift_detection(detection_date);
    CREATE INDEX IF NOT EXISTS idx_kpi_metrics_date ON monitoring.kpi_metrics(metric_date);
    """
    
    try:
        engine = get_engine()
        with engine.connect() as connection:
            connection.execute(text(indexes_sql))
            connection.commit()
        
        logger.info("Successfully created database indexes")
        
    except Exception as e:
        logger.error(f"Failed to create indexes: {e}")
        raise SQLAlchemyError(f"Index creation failed: {e}")


def health_check() -> dict:
    """
    Perform database health check.
    
    Returns:
        Dictionary with health check results
    """
    try:
        start_time = pd.Timestamp.now()
        
        # Test basic connection
        is_connected = check_database_connection()
        
        # Test query performance
        test_query = "SELECT COUNT(*) FROM finrisk.customer_profiles"
        result = read_sql_query(test_query)
        
        end_time = pd.Timestamp.now()
        response_time = (end_time - start_time).total_seconds()
        
        return {
            "status": "healthy" if is_connected else "unhealthy",
            "connection": is_connected,
            "response_time_seconds": response_time,
            "customer_count": int(result.iloc[0, 0]) if is_connected else 0,
            "timestamp": end_time.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "connection": False,
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }


def backup_table(table_name: str, schema: str = "finrisk", backup_suffix: str = "_backup") -> str:
    """
    Create a backup of a table.
    
    Args:
        table_name: Table name to backup
        schema: Database schema
        backup_suffix: Suffix for backup table name
        
    Returns:
        Backup table name
    """
    backup_table_name = f"{table_name}{backup_suffix}"
    
    try:
        engine = get_engine()
        with engine.connect() as connection:
            # Create backup table
            connection.execute(text(f"CREATE TABLE {schema}.{backup_table_name} AS SELECT * FROM {schema}.{table_name}"))
            connection.commit()
        
        logger.info(f"Successfully created backup table {schema}.{backup_table_name}")
        return backup_table_name
        
    except Exception as e:
        logger.error(f"Failed to backup table {schema}.{table_name}: {e}")
        raise SQLAlchemyError(f"Table backup failed: {e}")


def restore_table_from_backup(table_name: str, backup_table_name: str, schema: str = "finrisk") -> None:
    """
    Restore a table from its backup.
    
    Args:
        table_name: Table name to restore
        backup_table_name: Backup table name
        schema: Database schema
    """
    try:
        engine = get_engine()
        with engine.connect() as connection:
            # Drop original table and rename backup
            connection.execute(text(f"DROP TABLE IF EXISTS {schema}.{table_name}"))
            connection.execute(text(f"ALTER TABLE {schema}.{backup_table_name} RENAME TO {table_name}"))
            connection.commit()
        
        logger.info(f"Successfully restored table {schema}.{table_name} from backup")
        
    except Exception as e:
        logger.error(f"Failed to restore table {schema}.{table_name}: {e}")
        raise SQLAlchemyError(f"Table restore failed: {e}")


def vacuum_table(table_name: str, schema: str = "finrisk") -> None:
    """
    Run VACUUM on a table to reclaim storage and update statistics.
    
    Args:
        table_name: Table name
        schema: Database schema
    """
    try:
        engine = get_engine()
        with engine.connect() as connection:
            connection.execute(text(f"VACUUM ANALYZE {schema}.{table_name}"))
            connection.commit()
        
        logger.info(f"Successfully ran VACUUM on {schema}.{table_name}")
        
    except Exception as e:
        logger.error(f"Failed to VACUUM table {schema}.{table_name}: {e}")
        raise SQLAlchemyError(f"VACUUM failed: {e}")


def get_database_size() -> dict:
    """
    Get database size information.
    
    Returns:
        Dictionary with database size information
    """
    query = """
    SELECT 
        schemaname,
        tablename,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
    FROM pg_tables 
    WHERE schemaname IN ('finrisk', 'audit', 'monitoring')
    ORDER BY size_bytes DESC;
    """
    
    try:
        df = read_sql_query(query)
        return {
            'tables': df.to_dict('records'),
            'total_size_bytes': df['size_bytes'].sum(),
            'total_size_pretty': f"{df['size_bytes'].sum() / (1024**3):.2f} GB"
        }
    except Exception as e:
        logger.error(f"Failed to get database size: {e}")
        return {'tables': [], 'total_size_bytes': 0, 'total_size_pretty': '0 B'}
