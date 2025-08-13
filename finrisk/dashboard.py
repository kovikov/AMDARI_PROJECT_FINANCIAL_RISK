#!/usr/bin/env python3
"""
Streamlit dashboard for FinRisk application.
Real-time monitoring dashboard for credit risk, fraud detection, and portfolio analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta, date
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set database credentials for dashboard
os.environ["DB_NAME"] = "amdari_project"
os.environ["DB_USER"] = "postgres"
os.environ["DB_PASSWORD"] = "Kovikov1978@"

from app.config import get_settings
from app.infra.db import read_sql_query, check_database_connection
from app.infra.cache import get_cache_stats

# Configure Streamlit page
st.set_page_config(
    page_title="FinRisk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffe6e6;
        border-left: 4px solid #ff4444;
    }
    .alert-medium {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .alert-low {
        background-color: #e6ffe6;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)


# Initialize settings
@st.cache_resource
def get_app_settings():
    return get_settings()


settings = get_app_settings()


# Data loading functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_portfolio_metrics():
    """Load portfolio metrics from database."""
    try:
        import psycopg2
        
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "amdari_project",
            "user": "postgres",
            "password": "Kovikov1978@"
        }
        
        conn = psycopg2.connect(**db_config)
        query = """
        SELECT 
            COUNT(*) as total_customers,
            AVG(credit_score) as avg_credit_score,
            COUNT(CASE WHEN credit_score >= 700 THEN 1 END) as prime_customers,
            COUNT(CASE WHEN credit_score < 600 THEN 1 END) as subprime_customers,
            SUM(annual_income) as total_relationship_value
        FROM customer_profiles
        """
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result
    except Exception as e:
        st.error(f"Failed to load portfolio metrics: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_credit_metrics():
    """Load credit risk metrics from database."""
    try:
        import psycopg2
        
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "amdari_project",
            "user": "postgres",
            "password": "Kovikov1978@"
        }
        
        conn = psycopg2.connect(**db_config)
        query = """
        SELECT 
            DATE(created_at) as application_date,
            COUNT(*) as total_applications,
            COUNT(CASE WHEN application_status = 'APPROVED' THEN 1 END) as approved_applications,
            COUNT(CASE WHEN application_status = 'REJECTED' THEN 1 END) as rejected_applications,
            AVG(loan_amount) as avg_loan_amount
        FROM credit_applications
        WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY DATE(created_at)
        ORDER BY DATE(created_at)
        """
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result
    except Exception as e:
        st.error(f"Failed to load credit metrics: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_fraud_metrics():
    """Load fraud detection metrics from database."""
    try:
        import psycopg2
        
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "amdari_project",
            "user": "postgres",
            "password": "Kovikov1978@"
        }
        
        conn = psycopg2.connect(**db_config)
        query = """
        SELECT 
            DATE(transaction_date) as transaction_date,
            COUNT(*) as total_transactions,
            COUNT(CASE WHEN is_fraudulent = true THEN 1 END) as fraud_transactions,
            SUM(amount) as total_amount,
            SUM(CASE WHEN is_fraudulent = true THEN amount ELSE 0 END) as fraud_amount
        FROM transaction_data
        WHERE transaction_date >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY DATE(transaction_date)
        ORDER BY transaction_date
        """
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result
    except Exception as e:
        st.error(f"Failed to load fraud metrics: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_risk_distribution():
    """Load risk segment distribution data."""
    try:
        import psycopg2
        
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "amdari_project",
            "user": "postgres",
            "password": "Kovikov1978@"
        }
        
        conn = psycopg2.connect(**db_config)
        query = """
        SELECT 
            CASE 
                WHEN credit_score >= 750 THEN 'Prime'
                WHEN credit_score >= 650 THEN 'Near-Prime'
                WHEN credit_score >= 550 THEN 'Subprime'
                ELSE 'Deep-Subprime'
            END as risk_segment,
            COUNT(*) as customer_count,
            AVG(credit_score) as avg_credit_score,
            AVG(annual_income) as avg_income,
            SUM(annual_income) as total_value
        FROM customer_profiles
        GROUP BY 
            CASE 
                WHEN credit_score >= 750 THEN 'Prime'
                WHEN credit_score >= 650 THEN 'Near-Prime'
                WHEN credit_score >= 550 THEN 'Subprime'
                ELSE 'Deep-Subprime'
            END
        ORDER BY 
            CASE 
                WHEN credit_score >= 750 THEN 'Prime'
                WHEN credit_score >= 650 THEN 'Near-Prime'
                WHEN credit_score >= 550 THEN 'Subprime'
                ELSE 'Deep-Subprime'
            END
        """
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result
    except Exception as e:
        st.error(f"Failed to load risk distribution: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_model_predictions():
    """Load model predictions data."""
    try:
        import psycopg2
        
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "amdari_project",
            "user": "postgres",
            "password": "Kovikov1978@"
        }
        
        conn = psycopg2.connect(**db_config)
        query = """
        SELECT 
            DATE(prediction_timestamp) as prediction_date,
            model_type,
            COUNT(*) as total_predictions,
            AVG(prediction_value) as avg_prediction_value,
            AVG(confidence_score) as avg_confidence
        FROM model_predictions
        WHERE prediction_timestamp >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY DATE(prediction_timestamp), model_type
        ORDER BY prediction_date, model_type
        """
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result
    except Exception as e:
        st.error(f"Failed to load model predictions: {e}")
        return pd.DataFrame()


def check_system_health():
    """Check system health status."""
    health_status = {
        "database": False,
        "cache": False,
        "overall": False
    }
    
    try:
        # Check database with direct connection
        import psycopg2
        
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "amdari_project",
            "user": "postgres",
            "password": "Kovikov1978@"
        }
        
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        health_status["database"] = True
        
        # Check cache
        cache_stats = get_cache_stats()
        health_status["cache"] = cache_stats.get("status") == "healthy"
        
        # Overall health
        health_status["overall"] = health_status["database"] and health_status["cache"]
        
    except Exception as e:
        st.error(f"Health check failed: {e}")
    
    return health_status


def main():
    """Main dashboard function."""
    
    # Sidebar
    st.sidebar.title("🏦 FinRisk Dashboard")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["📊 Overview", "💳 Credit Risk", "🚨 Fraud Detection", "📈 Portfolio Analysis", "🤖 Model Performance", "⚙️ System Health"]
    )
    
    # Date range selector
    st.sidebar.markdown("### Date Range")
    date_range = st.sidebar.date_input(
        "Select date range:",
        value=(date.today() - timedelta(days=30), date.today()),
        max_value=date.today()
    )
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        st.rerun()
    
    # Main content area
    if page == "📊 Overview":
        show_overview_page()
    elif page == "💳 Credit Risk":
        show_credit_risk_page()
    elif page == "🚨 Fraud Detection":
        show_fraud_detection_page()
    elif page == "📈 Portfolio Analysis":
        show_portfolio_analysis_page()
    elif page == "🤖 Model Performance":
        show_model_performance_page()
    elif page == "⚙️ System Health":
        show_system_health_page()


def show_overview_page():
    """Show overview dashboard page."""
    st.title("📊 FinRisk Overview Dashboard")
    st.markdown("Real-time monitoring of credit risk and fraud detection systems")
    
    # Load data
    portfolio_metrics = load_portfolio_metrics()
    credit_metrics = load_credit_metrics()
    fraud_metrics = load_fraud_metrics()
    
    if portfolio_metrics.empty:
        st.warning("No data available. Please check database connection.")
        return
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = portfolio_metrics.iloc[0]['total_customers']
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        avg_credit_score = portfolio_metrics.iloc[0]['avg_credit_score']
        st.metric("Avg Credit Score", f"{avg_credit_score:.0f}")
    
    with col3:
        if not credit_metrics.empty:
            approval_rate = (credit_metrics['approved_applications'].sum() / 
                           credit_metrics['total_applications'].sum()) * 100
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
        else:
            st.metric("Approval Rate", "N/A")
    
    with col4:
        if not fraud_metrics.empty:
            fraud_rate = (fraud_metrics['fraud_transactions'].sum() / 
                         fraud_metrics['total_transactions'].sum()) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.3f}%")
        else:
            st.metric("Fraud Rate", "N/A")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Credit Applications Trend")
        if not credit_metrics.empty:
            fig = px.line(credit_metrics, x='application_date', y='total_applications',
                         title="Daily Credit Applications")
            fig.add_scatter(x=credit_metrics['application_date'], 
                          y=credit_metrics['approved_applications'],
                          mode='lines', name='Approved')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No credit data available")
    
    with col2:
        st.subheader("Transaction Volume Trend")
        if not fraud_metrics.empty:
            fig = px.bar(fraud_metrics, x='transaction_date', y='total_transactions',
                        title="Daily Transaction Volume")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No transaction data available")
    
    # Risk distribution
    st.subheader("Risk Segment Distribution")
    risk_dist = load_risk_distribution()
    if not risk_dist.empty:
        fig = px.pie(risk_dist, values='customer_count', names='risk_segment',
                    title="Customer Distribution by Risk Segment")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No risk distribution data available")


def show_credit_risk_page():
    """Show credit risk analysis page."""
    st.title("💳 Credit Risk Analysis")
    
    # Load credit data
    credit_metrics = load_credit_metrics()
    risk_dist = load_risk_distribution()
    
    if credit_metrics.empty:
        st.warning("No credit data available.")
        return
    
    # Credit metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_apps = credit_metrics['total_applications'].sum()
        st.metric("Total Applications", f"{total_apps:,}")
    
    with col2:
        approved_apps = credit_metrics['approved_applications'].sum()
        approval_rate = (approved_apps / total_apps) * 100 if total_apps > 0 else 0
        st.metric("Approval Rate", f"{approval_rate:.1f}%")
    
    with col3:
        rejected_apps = credit_metrics['rejected_applications'].sum()
        rejection_rate = (rejected_apps / total_apps) * 100 if total_apps > 0 else 0
        st.metric("Rejection Rate", f"{rejection_rate:.1f}%")
    
    # Credit trend analysis
    st.subheader("Credit Application Trends")
    
    # Calculate approval rates over time
    credit_metrics['approval_rate'] = (credit_metrics['approved_applications'] / 
                                     credit_metrics['total_applications']) * 100
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=credit_metrics['application_date'], 
                  y=credit_metrics['total_applications'],
                  name="Total Applications"),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=credit_metrics['application_date'], 
                  y=credit_metrics['approval_rate'],
                  name="Approval Rate %"),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Applications", secondary_y=False)
    fig.update_yaxes(title_text="Approval Rate %", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk segment analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Segment Distribution")
        if not risk_dist.empty:
            fig = px.bar(risk_dist, x='risk_segment', y='customer_count',
                        title="Customers by Risk Segment")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Average Credit Score by Segment")
        if not risk_dist.empty:
            fig = px.bar(risk_dist, x='risk_segment', y='avg_credit_score',
                        title="Credit Score by Risk Segment")
            st.plotly_chart(fig, use_container_width=True)


def show_fraud_detection_page():
    """Show fraud detection analysis page."""
    st.title("🚨 Fraud Detection Analysis")
    
    # Load fraud data
    fraud_metrics = load_fraud_metrics()
    
    if fraud_metrics.empty:
        st.warning("No fraud data available.")
        return
    
    # Calculate metrics
    fraud_metrics['fraud_rate'] = (fraud_metrics['fraud_transactions'] / 
                                 fraud_metrics['total_transactions']) * 100
    fraud_metrics['fraud_amount_pct'] = (fraud_metrics['fraud_amount'] / 
                                       fraud_metrics['total_amount']) * 100
    
    # Fraud metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_transactions = fraud_metrics['total_transactions'].sum()
        st.metric("Total Transactions", f"{total_transactions:,}")
    
    with col2:
        total_fraud = fraud_metrics['fraud_transactions'].sum()
        fraud_rate = (total_fraud / total_transactions) * 100 if total_transactions > 0 else 0
        st.metric("Fraud Rate", f"{fraud_rate:.3f}%")
    
    with col3:
        total_amount = fraud_metrics['total_amount'].sum()
        st.metric("Transaction Volume", f"£{total_amount:,.0f}")
    
    with col4:
        fraud_amount = fraud_metrics['fraud_amount'].sum()
        st.metric("Fraud Losses", f"£{fraud_amount:,.0f}")
    
    # Fraud trends
    st.subheader("Fraud Detection Trends")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=fraud_metrics['transaction_date'], 
                  y=fraud_metrics['total_transactions'],
                  name="Total Transactions"),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=fraud_metrics['transaction_date'], 
                  y=fraud_metrics['fraud_rate'],
                  name="Fraud Rate %"),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Transactions", secondary_y=False)
    fig.update_yaxes(title_text="Fraud Rate %", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Fraud amount analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Daily Fraud Losses")
        fig = px.bar(fraud_metrics, x='transaction_date', y='fraud_amount',
                    title="Daily Fraud Losses (£)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Fraud Rate Trend")
        fig = px.line(fraud_metrics, x='transaction_date', y='fraud_rate',
                     title="Daily Fraud Rate (%)")
        st.plotly_chart(fig, use_container_width=True)


def show_portfolio_analysis_page():
    """Show portfolio analysis page."""
    st.title("📈 Portfolio Analysis")
    
    # Load data
    portfolio_metrics = load_portfolio_metrics()
    risk_dist = load_risk_distribution()
    
    if portfolio_metrics.empty:
        st.warning("No portfolio data available.")
        return
    
    # Portfolio overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_customers = portfolio_metrics.iloc[0]['total_customers']
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        total_value = portfolio_metrics.iloc[0]['total_relationship_value']
        st.metric("Portfolio Value", f"£{total_value:,.0f}")
    
    with col3:
        avg_credit_score = portfolio_metrics.iloc[0]['avg_credit_score']
        st.metric("Avg Credit Score", f"{avg_credit_score:.0f}")
    
    # Risk analysis
    if not risk_dist.empty:
        st.subheader("Portfolio Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution pie chart
            fig = px.pie(risk_dist, values='customer_count', names='risk_segment',
                        title="Portfolio Distribution by Risk")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Value by risk segment
            fig = px.bar(risk_dist, x='risk_segment', y='total_value',
                        title="Portfolio Value by Risk Segment")
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk segment details table
        st.subheader("Risk Segment Details")
        risk_display = risk_dist.copy()
        risk_display['avg_credit_score'] = risk_display['avg_credit_score'].round(0)
        risk_display['avg_income'] = risk_display['avg_income'].round(0)
        risk_display['total_value'] = risk_display['total_value'].round(0)
        
        st.dataframe(risk_display, use_container_width=True)


def show_model_performance_page():
    """Show model performance analysis page."""
    st.title("🤖 Model Performance Analysis")
    
    # Load model predictions data
    model_predictions = load_model_predictions()
    
    if model_predictions.empty:
        st.warning("No model performance data available.")
        return
    
    # Model performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_predictions = model_predictions['total_predictions'].sum()
        st.metric("Total Predictions", f"{total_predictions:,}")
    
    with col2:
        avg_confidence = model_predictions['avg_confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    with col3:
        unique_models = model_predictions['model_type'].nunique()
        st.metric("Active Models", f"{unique_models}")
    
    # Model performance trends
    st.subheader("Model Performance Trends")
    
    # Filter for specific model types if available
    model_types = model_predictions['model_type'].unique()
    
    if len(model_types) > 0:
        selected_model = st.selectbox("Select Model Type:", model_types)
        
        model_data = model_predictions[model_predictions['model_type'] == selected_model]
        
        if not model_data.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=model_data['prediction_date'], 
                          y=model_data['total_predictions'],
                          name="Predictions"),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Scatter(x=model_data['prediction_date'], 
                          y=model_data['avg_confidence'],
                          name="Confidence"),
                secondary_y=True,
            )
            
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Predictions", secondary_y=False)
            fig.update_yaxes(title_text="Confidence", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison
    st.subheader("Model Comparison")
    
    if len(model_types) > 1:
        model_summary = model_predictions.groupby('model_type').agg({
            'total_predictions': 'sum',
            'avg_prediction_value': 'mean',
            'avg_confidence': 'mean'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(model_summary, x='model_type', y='total_predictions',
                        title="Predictions by Model Type")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(model_summary, x='model_type', y='avg_confidence',
                        title="Average Confidence by Model Type")
            st.plotly_chart(fig, use_container_width=True)
        
        # Model summary table
        st.subheader("Model Summary")
        st.dataframe(model_summary, use_container_width=True)


def show_system_health_page():
    """Show system health monitoring page."""
    st.title("⚙️ System Health")
    
    # System health check
    health_status = check_system_health()
    
    # Overall status
    if health_status["overall"]:
        st.success("🟢 All systems operational")
    else:
        st.error("🔴 System issues detected")
    
    # Component status
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Database Status")
        if health_status["database"]:
            st.success("✅ Database connected")
        else:
            st.error("❌ Database connection failed")
    
    with col2:
        st.subheader("Cache Status")
        if health_status["cache"]:
            st.success("✅ Cache operational")
        else:
            st.warning("⚠️ Cache issues detected")
    
    # Cache statistics
    st.subheader("Cache Statistics")
    try:
        cache_stats = get_cache_stats()
        if cache_stats.get("status") == "healthy":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Hit Rate", f"{cache_stats.get('hit_rate', 0):.1%}")
            
            with col2:
                st.metric("Memory Used", cache_stats.get('used_memory', 'N/A'))
            
            with col3:
                st.metric("Connected Clients", cache_stats.get('connected_clients', 0))
        else:
            st.info("Cache statistics not available")
    except Exception as e:
        st.error(f"Failed to load cache statistics: {e}")
    
    # Database connection test
    st.subheader("Database Connection Test")
    try:
        # Test a simple query with direct connection
        import psycopg2
        
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "amdari_project",
            "user": "postgres",
            "password": "Kovikov1978@"
        }
        
        conn = psycopg2.connect(**db_config)
        test_query = "SELECT COUNT(*) as table_count FROM information_schema.tables WHERE table_schema = 'public'"
        result = pd.read_sql_query(test_query, conn)
        conn.close()
        
        if not result.empty:
            table_count = result.iloc[0]['table_count']
            st.success(f"✅ Database connection successful. Found {table_count} tables.")
        else:
            st.warning("⚠️ Database connection test returned no results.")
    except Exception as e:
        st.error(f"❌ Database connection test failed: {e}")
    
    # System metrics placeholder
    st.subheader("System Metrics")
    st.info("System metrics integration coming soon...")


if __name__ == "__main__":
    main()
