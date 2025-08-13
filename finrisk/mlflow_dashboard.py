#!/usr/bin/env python3
"""
MLflow Dashboard for FinRisk Application.
Streamlit dashboard for MLflow experiment tracking and model management.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
from datetime import datetime, timedelta
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.monitoring.mlflow_tracker import (
    FinRiskMLflowTracker, 
    FinRiskModelRegistry,
    mlflow_tracker,
    model_registry
)

# Configure page
st.set_page_config(
    page_title="FinRisk MLflow Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Set environment variables
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
os.environ.setdefault("MLFLOW_REGISTRY_URI", "sqlite:///mlflow.db")


def load_mlflow_data():
    """Load MLflow experiment data."""
    try:
        tracker = FinRiskMLflowTracker()
        summary = tracker.get_experiment_summary()
        
        # Get detailed runs
        runs = mlflow.search_runs(experiment_ids=[tracker.experiment_id])
        
        return tracker, summary, runs
    except Exception as e:
        st.error(f"Failed to load MLflow data: {e}")
        return None, None, None


def load_model_registry_data():
    """Load model registry data."""
    try:
        registry = FinRiskModelRegistry()
        models = registry.list_models()
        return registry, models
    except Exception as e:
        st.error(f"Failed to load model registry data: {e}")
        return None, None


def main():
    """Main dashboard function."""
    st.markdown('<h1 class="main-header">📊 FinRisk MLflow Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Experiments", "Model Registry", "Model Comparison", "Data Quality", "Settings"]
    )
    
    # Load data
    tracker, summary, runs = load_mlflow_data()
    registry, models = load_model_registry_data()
    
    if page == "Overview":
        show_overview_page(tracker, summary, runs, registry, models)
    elif page == "Experiments":
        show_experiments_page(tracker, runs)
    elif page == "Model Registry":
        show_model_registry_page(registry, models)
    elif page == "Model Comparison":
        show_model_comparison_page(tracker, runs)
    elif page == "Data Quality":
        show_data_quality_page(tracker, runs)
    elif page == "Settings":
        show_settings_page()


def show_overview_page(tracker, summary, runs, registry, models):
    """Show overview dashboard."""
    st.header("📈 Experiment Overview")
    
    if not summary:
        st.warning("No MLflow data available")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Runs", summary.get('total_runs', 0))
    
    with col2:
        credit_runs = summary.get('model_types', {}).get('credit_risk', 0)
        st.metric("Credit Risk Models", credit_runs)
    
    with col3:
        fraud_runs = summary.get('model_types', {}).get('fraud_detection', 0)
        st.metric("Fraud Detection Models", fraud_runs)
    
    with col4:
        registered_models = len(models) if models else 0
        st.metric("Registered Models", registered_models)
    
    # Recent activity
    st.subheader("🕒 Recent Activity")
    
    if summary.get('latest_runs'):
        recent_df = pd.DataFrame(summary['latest_runs'])
        recent_df['start_time'] = pd.to_datetime(recent_df['start_time'], unit='ms')
        
        # Display recent runs
        for _, run in recent_df.head(5).iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    st.write(f"**{run['run_name']}**")
                with col2:
                    st.write(f"Type: {run['model_type']}")
                with col3:
                    st.write(f"Algorithm: {run['algorithm']}")
                with col4:
                    st.write(f"Status: {run['status']}")
                st.divider()
    
    # Model type distribution
    if summary.get('model_types'):
        st.subheader("📊 Model Type Distribution")
        fig = px.pie(
            values=list(summary['model_types'].values()),
            names=list(summary['model_types'].keys()),
            title="Distribution of Model Types"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance trends
    if runs is not None and not runs.empty:
        st.subheader("📈 Performance Trends")
        
        # Filter runs with AUC metric
        auc_runs = runs[runs['metrics.auc'].notna()].copy()
        if not auc_runs.empty:
            auc_runs['start_time'] = pd.to_datetime(auc_runs['start_time'], unit='ms')
            auc_runs = auc_runs.sort_values('start_time')
            
            fig = px.line(
                auc_runs,
                x='start_time',
                y='metrics.auc',
                color='tags.model_type',
                title="AUC Performance Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)


def show_experiments_page(tracker, runs):
    """Show experiments page."""
    st.header("🔬 Experiments")
    
    if runs is None or runs.empty:
        st.warning("No experiments found")
        return
    
    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_types = ['All'] + list(runs['tags.model_type'].dropna().unique())
        selected_type = st.selectbox("Model Type", model_types)
    
    with col2:
        algorithms = ['All'] + list(runs['tags.algorithm'].dropna().unique())
        selected_algorithm = st.selectbox("Algorithm", algorithms)
    
    with col3:
        statuses = ['All'] + list(runs['status'].unique())
        selected_status = st.selectbox("Status", statuses)
    
    # Filter runs
    filtered_runs = runs.copy()
    if selected_type != 'All':
        filtered_runs = filtered_runs[filtered_runs['tags.model_type'] == selected_type]
    if selected_algorithm != 'All':
        filtered_runs = filtered_runs[filtered_runs['tags.algorithm'] == selected_algorithm]
    if selected_status != 'All':
        filtered_runs = filtered_runs[filtered_runs['status'] == selected_status]
    
    # Display runs
    st.subheader(f"Experiments ({len(filtered_runs)} found)")
    
    for _, run in filtered_runs.iterrows():
        with st.expander(f"{run['tags.mlflow.runName']} - {run['tags.model_type']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Run ID:** {run['run_id']}")
                st.write(f"**Algorithm:** {run['tags.algorithm']}")
                st.write(f"**Status:** {run['status']}")
                st.write(f"**Start Time:** {pd.to_datetime(run['start_time'], unit='ms')}")
                
                # Display metrics
                metrics_cols = [col for col in run.index if col.startswith('metrics.')]
                if metrics_cols:
                    st.write("**Metrics:**")
                    for metric_col in metrics_cols:
                        metric_name = metric_col.replace('metrics.', '')
                        metric_value = run[metric_col]
                        if pd.notna(metric_value):
                            st.write(f"  - {metric_name}: {metric_value:.4f}")
            
            with col2:
                # Action buttons
                if st.button(f"View Details", key=f"view_{run['run_id']}"):
                    show_run_details(run)
                
                if st.button(f"Register Model", key=f"register_{run['run_id']}"):
                    register_model_from_run(run['run_id'])


def show_model_registry_page(registry, models):
    """Show model registry page."""
    st.header("📦 Model Registry")
    
    if not models:
        st.warning("No registered models found")
        return
    
    # Model list
    st.subheader("Registered Models")
    
    for model in models:
        with st.expander(f"{model['name']} - v{model['latest_version']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Description:** {model['description'] or 'No description'}")
                st.write(f"**Latest Version:** {model['latest_version']}")
                st.write(f"**Latest Stage:** {model['latest_stage']}")
                st.write(f"**Created:** {pd.to_datetime(model['creation_timestamp'], unit='ms')}")
            
            with col2:
                if st.button(f"Load Model", key=f"load_{model['name']}"):
                    load_production_model(model['name'])
                
                if st.button(f"Promote to Production", key=f"promote_{model['name']}"):
                    promote_model_to_production(model['name'], model['latest_version'])


def show_model_comparison_page(tracker, runs):
    """Show model comparison page."""
    st.header("⚖️ Model Comparison")
    
    if runs is None or runs.empty:
        st.warning("No experiments found for comparison")
        return
    
    # Select models to compare
    st.subheader("Select Models for Comparison")
    
    # Get runs with metrics
    runs_with_metrics = runs[runs['metrics.auc'].notna()].copy()
    
    if runs_with_metrics.empty:
        st.warning("No runs with AUC metrics found")
        return
    
    # Create run selection
    run_options = []
    for _, run in runs_with_metrics.iterrows():
        run_options.append({
            'run_id': run['run_id'],
            'name': f"{run['tags.mlflow.runName']} ({run['tags.algorithm']})",
            'model_type': run['tags.model_type'],
            'algorithm': run['tags.algorithm']
        })
    
    selected_runs = st.multiselect(
        "Select runs to compare",
        options=run_options,
        format_func=lambda x: x['name'],
        default=run_options[:3] if len(run_options) >= 3 else run_options
    )
    
    if selected_runs:
        # Create comparison dataframe
        comparison_data = []
        for run_info in selected_runs:
            run_data = runs[runs['run_id'] == run_info['run_id']].iloc[0]
            
            metrics = {}
            for col in run_data.index:
                if col.startswith('metrics.'):
                    metric_name = col.replace('metrics.', '')
                    metric_value = run_data[col]
                    if pd.notna(metric_value):
                        metrics[metric_name] = metric_value
            
            comparison_data.append({
                'Model': run_info['name'],
                'Type': run_info['model_type'],
                'Algorithm': run_info['algorithm'],
                **metrics
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.subheader("Performance Comparison")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Create comparison charts
        if len(comparison_data) > 1:
            st.subheader("Visual Comparison")
            
            # Metrics to plot
            metric_cols = [col for col in comparison_df.columns if col not in ['Model', 'Type', 'Algorithm']]
            
            if metric_cols:
                # Bar chart for each metric
                for metric in metric_cols[:4]:  # Limit to first 4 metrics
                    fig = px.bar(
                        comparison_df,
                        x='Model',
                        y=metric,
                        title=f"{metric.upper()} Comparison",
                        color='Type'
                    )
                    st.plotly_chart(fig, use_container_width=True)


def show_data_quality_page(tracker, runs):
    """Show data quality page."""
    st.header("🔍 Data Quality")
    
    if runs is None or runs.empty:
        st.warning("No experiments found")
        return
    
    # Filter data quality runs
    data_quality_runs = runs[runs['tags.experiment_type'] == 'data_quality'].copy()
    
    if data_quality_runs.empty:
        st.info("No data quality reports found. Run data quality analysis to see results here.")
        return
    
    st.subheader("Data Quality Reports")
    
    for _, run in data_quality_runs.iterrows():
        with st.expander(f"Data Quality Report - {run['tags.mlflow.runName']}"):
            st.write(f"**Run ID:** {run['run_id']}")
            st.write(f"**Dataset:** {run.get('tags.dataset_name', 'Unknown')}")
            st.write(f"**Start Time:** {pd.to_datetime(run['start_time'], unit='ms')}")
            
            # Display data quality metrics
            metrics_cols = [col for col in run.index if col.startswith('metrics.')]
            if metrics_cols:
                st.write("**Quality Metrics:**")
                for metric_col in metrics_cols:
                    metric_name = metric_col.replace('metrics.', '')
                    metric_value = run[metric_col]
                    if pd.notna(metric_value):
                        st.write(f"  - {metric_name}: {metric_value}")


def show_settings_page():
    """Show settings page."""
    st.header("⚙️ Settings")
    
    st.subheader("MLflow Configuration")
    
    # Display current settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Tracking URI:**")
        st.code(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        
        st.write("**Registry URI:**")
        st.code(os.getenv("MLFLOW_REGISTRY_URI", "sqlite:///mlflow.db"))
    
    with col2:
        st.write("**Experiment Name:**")
        st.code(os.getenv("MLFLOW_EXPERIMENT_NAME", "finrisk-experiments"))
    
    # Connection test
    st.subheader("Connection Test")
    
    if st.button("Test MLflow Connection"):
        try:
            tracker = FinRiskMLflowTracker()
            summary = tracker.get_experiment_summary()
            st.success(f"✅ Connection successful! Found {summary.get('total_runs', 0)} runs.")
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")


def show_run_details(run):
    """Show detailed information about a specific run."""
    st.subheader(f"Run Details: {run['tags.mlflow.runName']}")
    
    # Basic information
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Run ID:** {run['run_id']}")
        st.write(f"**Model Type:** {run['tags.model_type']}")
        st.write(f"**Algorithm:** {run['tags.algorithm']}")
        st.write(f"**Status:** {run['status']}")
    
    with col2:
        st.write(f"**Start Time:** {pd.to_datetime(run['start_time'], unit='ms')}")
        st.write(f"**End Time:** {pd.to_datetime(run['end_time'], unit='ms')}")
        st.write(f"**Duration:** {run['end_time'] - run['start_time']} ms")
    
    # Metrics
    st.subheader("Metrics")
    metrics_cols = [col for col in run.index if col.startswith('metrics.')]
    if metrics_cols:
        metrics_data = {}
        for metric_col in metrics_cols:
            metric_name = metric_col.replace('metrics.', '')
            metric_value = run[metric_col]
            if pd.notna(metric_value):
                metrics_data[metric_name] = metric_value
        
        if metrics_data:
            metrics_df = pd.DataFrame(list(metrics_data.items()), columns=['Metric', 'Value'])
            st.dataframe(metrics_df, use_container_width=True)
    
    # Parameters
    st.subheader("Parameters")
    param_cols = [col for col in run.index if col.startswith('params.')]
    if param_cols:
        params_data = {}
        for param_col in param_cols:
            param_name = param_col.replace('params.', '')
            param_value = run[param_col]
            if pd.notna(param_value):
                params_data[param_name] = param_value
        
        if params_data:
            params_df = pd.DataFrame(list(params_data.items()), columns=['Parameter', 'Value'])
            st.dataframe(params_df, use_container_width=True)


def register_model_from_run(run_id):
    """Register a model from a run."""
    try:
        model_name = st.text_input("Enter model name for registration:")
        description = st.text_area("Enter model description:")
        
        if st.button("Register Model"):
            if model_name:
                version = register_model_to_production(
                    run_id=run_id,
                    model_name=model_name,
                    description=description
                )
                st.success(f"Model registered successfully! Version: {version}")
            else:
                st.error("Please enter a model name")
    except Exception as e:
        st.error(f"Failed to register model: {e}")


def load_production_model(model_name):
    """Load a production model."""
    try:
        registry = FinRiskModelRegistry()
        model = registry.load_production_model(model_name)
        st.success(f"Model {model_name} loaded successfully!")
        st.write(f"Model type: {type(model)}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")


def promote_model_to_production(model_name, version):
    """Promote a model to production."""
    try:
        registry = FinRiskModelRegistry()
        registry.promote_model(model_name, version, "Production")
        st.success(f"Model {model_name} v{version} promoted to Production!")
    except Exception as e:
        st.error(f"Failed to promote model: {e}")


if __name__ == "__main__":
    main()
